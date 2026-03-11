import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.layers import batched_nms
from detectron2.utils.logger import setup_logger

from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs


MODEL_CLASS_NAMES = ["aircraft"]

OUTPUT_CLASS_NAME = "aircraft"


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run DiffDet4SAR inference on a directory of .tif/.tiff images."
    )
    parser.add_argument(
        "--config-file",
        default="configs/diffdet.aircraft.single_class.yaml",
        help="Path to the model config file.",
    )
    parser.add_argument(
        "--weights",
        required=True,
        help="Path to the trained model weights (.pth).",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory with .tif/.tiff images.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where visualizations and predictions.json will be written.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for keeping detections.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help='Inference device, for example "cuda" or "cpu".',
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively search for images inside the input directory.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size used for large .tif images.",
    )
    parser.add_argument(
        "--tile-overlap",
        type=int,
        default=256,
        help="Overlap between adjacent tiles.",
    )
    parser.add_argument(
        "--nms-threshold",
        type=float,
        default=0.5,
        help="IoU threshold used to merge detections from overlapping tiles.",
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=1.0,
        help="Lower percentile for uint16 to uint8 contrast stretch.",
    )
    parser.add_argument(
        "--upper-percentile",
        type=float,
        default=99.5,
        help="Upper percentile for uint16 to uint8 contrast stretch.",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=[".tif", ".tiff"],
        help="Extensions to include.",
    )
    parser.add_argument(
        "--opts",
        default=[],
        nargs=argparse.REMAINDER,
        help="Extra config overrides in KEY VALUE format.",
    )
    return parser


def setup_cfg(args):
    cfg = get_cfg()
    add_diffusiondet_config(cfg)
    add_model_ema_configs(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHTS = args.weights
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.DATASETS.TEST = ("sar_aircraft_inference",)
    cfg.freeze()
    return cfg


def list_images(input_dir, extensions, recursive):
    normalized_exts = {ext.lower() if ext.startswith(".") else f".{ext.lower()}" for ext in extensions}
    input_path = Path(input_dir)
    if recursive:
        files = [p for p in input_path.rglob("*") if p.is_file() and p.suffix.lower() in normalized_exts]
    else:
        files = [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in normalized_exts]
    return sorted(files)


def normalize_uint16_to_uint8(image, lower_percentile, upper_percentile):
    image = image.astype(np.float32)
    lo = float(np.percentile(image, lower_percentile))
    hi = float(np.percentile(image, upper_percentile))
    if hi <= lo:
        hi = float(image.max())
    if hi <= lo:
        hi = lo + 1.0
    image = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return (image * 255.0).astype(np.uint8)


def read_tif_as_bgr(path, lower_percentile, upper_percentile):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        image = tifffile.imread(str(path))

    if image.ndim == 2:
        image = normalize_uint16_to_uint8(image, lower_percentile, upper_percentile)
        image = np.repeat(image[:, :, None], 3, axis=2)
        return image

    if image.ndim == 3:
        if image.shape[0] in (1, 3) and image.shape[0] < image.shape[-1]:
            image = np.transpose(image, (1, 2, 0))

        if image.dtype != np.uint8:
            channels = []
            for channel_idx in range(image.shape[2]):
                channels.append(
                    normalize_uint16_to_uint8(image[:, :, channel_idx], lower_percentile, upper_percentile)
                )
            image = np.stack(channels, axis=2)

        num_channels = image.shape[2]
        if num_channels == 1:
            return np.repeat(image[:, :, :1], 3, axis=2)
        if num_channels == 2:
            third = ((image[:, :, 0].astype(np.uint16) + image[:, :, 1].astype(np.uint16)) // 2).astype(np.uint8)
            return np.dstack([image, third])
        return image[:, :, :3]

    raise ValueError(f"Unsupported TIFF shape for {path}: {image.shape}")


def compute_starts(length, tile_size, tile_overlap):
    if length <= tile_size:
        return [0]

    stride = tile_size - tile_overlap
    if stride <= 0:
        raise ValueError("tile-overlap must be smaller than tile-size")

    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def generate_tiles(height, width, tile_size, tile_overlap):
    y_starts = compute_starts(height, tile_size, tile_overlap)
    x_starts = compute_starts(width, tile_size, tile_overlap)
    for y0 in y_starts:
        for x0 in x_starts:
            yield x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)


def run_tiled_inference(predictor, image_bgr, tile_size, tile_overlap):
    height, width = image_bgr.shape[:2]
    boxes = []
    scores = []
    labels = []

    for x0, y0, x1, y1 in generate_tiles(height, width, tile_size, tile_overlap):
        tile = image_bgr[y0:y1, x0:x1]
        outputs = predictor(tile)
        instances = outputs["instances"].to("cpu")
        if len(instances) == 0:
            continue

        tile_boxes = instances.pred_boxes.tensor.clone()
        tile_boxes[:, [0, 2]] += x0
        tile_boxes[:, [1, 3]] += y0

        boxes.append(tile_boxes)
        scores.append(instances.scores.clone())
        labels.append(instances.pred_classes.clone())

    if not boxes:
        return {
            "boxes": torch.empty((0, 4), dtype=torch.float32),
            "scores": torch.empty((0,), dtype=torch.float32),
            "labels": torch.empty((0,), dtype=torch.int64),
        }

    return {
        "boxes": torch.cat(boxes, dim=0),
        "scores": torch.cat(scores, dim=0),
        "labels": torch.cat(labels, dim=0),
    }


def collapse_and_merge_detections(detections, nms_threshold):
    boxes = detections["boxes"]
    scores = detections["scores"]
    labels = detections["labels"]

    if len(boxes) == 0:
        return detections

    collapsed_labels = torch.zeros_like(labels)
    keep = batched_nms(boxes, scores, collapsed_labels, nms_threshold)

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": collapsed_labels[keep],
        "source_labels": labels[keep],
    }


def detections_to_records(detections):
    records = []
    boxes = detections["boxes"].numpy()
    scores = detections["scores"].numpy()
    source_labels = detections.get("source_labels", detections["labels"]).numpy()

    for box, score, source_label in zip(boxes, scores, source_labels):
        x1, y1, x2, y2 = [float(v) for v in box]
        source_label = int(source_label)
        records.append(
            {
                "score": round(float(score), 6),
                "class_id": 0,
                "class_name": OUTPUT_CLASS_NAME,
                "source_model_class_id": source_label,
                "source_model_class_name": (
                    MODEL_CLASS_NAMES[source_label]
                    if source_label < len(MODEL_CLASS_NAMES)
                    else f"class_{source_label}"
                ),
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_xywh": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            }
        )
    return records


def draw_detections(image_bgr, detections):
    vis = image_bgr.copy()
    boxes = detections["boxes"].numpy()
    scores = detections["scores"].numpy()

    for box, score in zip(boxes, scores):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{OUTPUT_CLASS_NAME} {score:.2f}"
        cv2.putText(
            vis,
            label,
            (x1, max(y1 - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return vis


def main():
    args = build_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: %s", args)

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = list_images(input_dir, args.extensions, args.recursive)
    if not image_paths:
        raise FileNotFoundError(
            f"No images with extensions {sorted(set(args.extensions))} were found in {input_dir}"
        )

    cfg = setup_cfg(args)
    predictor = DefaultPredictor(cfg)

    results = []
    total_start = time.time()

    for image_path in image_paths:
        rel_path = image_path.relative_to(input_dir)
        vis_path = output_dir / rel_path.parent / f"{image_path.stem}_pred.png"
        vis_path.parent.mkdir(parents=True, exist_ok=True)

        image = read_tif_as_bgr(image_path, args.lower_percentile, args.upper_percentile)
        start_time = time.time()
        tiled_detections = run_tiled_inference(
            predictor,
            image,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        )
        detections = collapse_and_merge_detections(tiled_detections, args.nms_threshold)
        elapsed = time.time() - start_time

        records = detections_to_records(detections)
        visualized_output = draw_detections(image, detections)
        cv2.imwrite(str(vis_path), visualized_output)

        results.append(
            {
                "image_path": str(image_path),
                "relative_image_path": str(rel_path),
                "visualization_path": str(vis_path),
                "num_detections": len(records),
                "detections": records,
            }
        )
        logger.info("%s: %d detections in %.2fs", rel_path, len(records), elapsed)

    json_path = output_dir / "predictions.json"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "config_file": str(Path(args.config_file).expanduser().resolve()),
                "weights": str(Path(args.weights).expanduser().resolve()),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "confidence_threshold": args.confidence_threshold,
                "device": args.device,
                "output_class_names": [OUTPUT_CLASS_NAME],
                "source_model_class_names": MODEL_CLASS_NAMES,
                "tile_size": args.tile_size,
                "tile_overlap": args.tile_overlap,
                "nms_threshold": args.nms_threshold,
                "lower_percentile": args.lower_percentile,
                "upper_percentile": args.upper_percentile,
                "num_images": len(results),
                "total_time_seconds": round(time.time() - total_start, 4),
                "results": results,
            },
            handle,
            indent=2,
        )

    logger.info("Finished %d images. Predictions saved to %s", len(results), json_path)


if __name__ == "__main__":
    main()
