#!/usr/bin/env python3
import argparse
import json
import time
import warnings
from pathlib import Path

import cv2
import numpy as np
import tifffile
from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.logger import setup_logger
from sahi.models.base import DetectionModel
from sahi.predict import get_sliced_prediction
from sahi.prediction import ObjectPrediction

from diffusiondet import add_diffusiondet_config
from diffusiondet.util.model_ema import add_model_ema_configs


OUTPUT_CLASS_NAME = "aircraft"


class DiffDetSahiDetectionModel(DetectionModel):
    def __init__(self, category_name: str = OUTPUT_CLASS_NAME, *args, **kwargs):
        self.required_packages = ["torch", "detectron2", "sahi"]
        self.category_name = category_name
        super().__init__(*args, **kwargs)

    def load_model(self):
        if self.model_path is None:
            raise ValueError("model_path is required")
        if self.config_path is None:
            raise ValueError("config_path is required")

        cfg = get_cfg()
        add_diffusiondet_config(cfg)
        add_model_ema_configs(cfg)
        cfg.merge_from_file(self.config_path)
        cfg.MODEL.WEIGHTS = self.model_path
        cfg.MODEL.DEVICE = self.device.type
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = self.confidence_threshold
        cfg.DATASETS.TEST = ("sar_aircraft_inference",)
        cfg.freeze()

        self.model = DefaultPredictor(cfg)
        if self.category_mapping is None:
            self.category_mapping = {"0": self.category_name}

    def set_model(self, model, **kwargs):
        self.model = model
        if self.category_mapping is None:
            self.category_mapping = {"0": self.category_name}

    def perform_inference(self, image: np.ndarray):
        if self.model is None:
            raise RuntimeError("Model is not loaded")

        # SAHI passes RGB images. DefaultPredictor usually expects BGR.
        if isinstance(image, np.ndarray) and self.model.input_format == "BGR":
            image = image[:, :, ::-1]

        self._original_predictions = self.model(image)

    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: list[list[int]] | None = [[0, 0]],
        full_shape_list: list[list[int]] | None = None,
    ):
        original_predictions = self._original_predictions

        if isinstance(shift_amount_list[0], int):
            shift_amount_list = [shift_amount_list]
        if full_shape_list is not None and isinstance(full_shape_list[0], int):
            full_shape_list = [full_shape_list]

        shift_amount = shift_amount_list[0]
        full_shape = None if full_shape_list is None else full_shape_list[0]

        boxes = original_predictions["instances"].pred_boxes.tensor
        scores = original_predictions["instances"].scores
        category_ids = original_predictions["instances"].pred_classes

        high_confidence_mask = scores >= self.confidence_threshold
        boxes = boxes[high_confidence_mask]
        scores = scores[high_confidence_mask]
        category_ids = category_ids[high_confidence_mask]

        object_prediction_list = []
        for box, score, category_id in zip(boxes, scores, category_ids):
            category_id_int = int(category_id.item())
            category_name = self.category_mapping.get(str(category_id_int), f"class_{category_id_int}")
            object_prediction_list.append(
                ObjectPrediction(
                    bbox=box.tolist(),
                    segmentation=None,
                    category_id=category_id_int,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    score=float(score.item()),
                    full_shape=full_shape,
                )
            )

        self._object_prediction_list_per_image = [object_prediction_list]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run SAHI sliced inference with DiffDet4SAR on .tif/.tiff directories."
    )
    parser.add_argument(
        "--config-file",
        default="configs/diffdet.aircraft.single_class.yaml",
        help="Path to the model config file.",
    )
    parser.add_argument("--weights", required=True, help="Path to model weights (.pth).")
    parser.add_argument("--input-dir", required=True, help="Directory with .tif/.tiff images.")
    parser.add_argument("--output-dir", required=True, help="Destination directory.")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Detection score threshold.")
    parser.add_argument("--device", default="cuda", help='Inference device, for example "cuda" or "cpu".')
    parser.add_argument("--recursive", action="store_true", help="Recursively scan input directory.")
    parser.add_argument("--slice-height", type=int, default=1024, help="SAHI slice height.")
    parser.add_argument("--slice-width", type=int, default=1024, help="SAHI slice width.")
    parser.add_argument("--overlap-height-ratio", type=float, default=0.25, help="SAHI vertical overlap ratio.")
    parser.add_argument("--overlap-width-ratio", type=float, default=0.25, help="SAHI horizontal overlap ratio.")
    parser.add_argument(
        "--postprocess-type",
        default="GREEDYNMM",
        choices=["GREEDYNMM", "NMS", "NMM", "LSNMS"],
        help="SAHI postprocess strategy.",
    )
    parser.add_argument(
        "--postprocess-match-metric",
        default="IOS",
        choices=["IOS", "IOU"],
        help="Metric used in postprocessing merge.",
    )
    parser.add_argument(
        "--postprocess-match-threshold",
        type=float,
        default=0.5,
        help="Matching threshold for postprocessing.",
    )
    parser.add_argument(
        "--perform-standard-pred",
        action="store_true",
        help="Also run full-image prediction besides slicing (disabled by default).",
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
    return parser


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
        return np.repeat(image[:, :, None], 3, axis=2)

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


def prediction_result_to_records(prediction_result):
    records = []
    for pred in prediction_result.object_prediction_list:
        x1, y1, x2, y2 = [float(v) for v in pred.bbox.to_xyxy()]
        score = float(pred.score.value)
        class_id = int(pred.category.id)
        class_name = pred.category.name
        records.append(
            {
                "score": round(score, 6),
                "class_id": class_id,
                "class_name": class_name,
                "bbox_xyxy": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
                "bbox_xywh": [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
            }
        )
    return records


def draw_detections(image_bgr, records):
    vis = image_bgr.copy()
    for rec in records:
        x1, y1, x2, y2 = [int(round(v)) for v in rec["bbox_xyxy"]]
        score = rec["score"]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"{rec['class_name']} {score:.2f}"
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

    detection_model = DiffDetSahiDetectionModel(
        model_path=str(Path(args.weights).expanduser().resolve()),
        config_path=str(Path(args.config_file).expanduser().resolve()),
        confidence_threshold=args.confidence_threshold,
        device=args.device,
        category_mapping={"0": OUTPUT_CLASS_NAME},
    )

    results = []
    total_start = time.time()

    for image_path in image_paths:
        rel_path = image_path.relative_to(input_dir)
        vis_path = output_dir / rel_path.parent / f"{image_path.stem}_pred.png"
        vis_path.parent.mkdir(parents=True, exist_ok=True)

        image_bgr = read_tif_as_bgr(image_path, args.lower_percentile, args.upper_percentile)
        image_rgb = image_bgr[:, :, ::-1]

        start_time = time.time()
        prediction_result = get_sliced_prediction(
            image=image_rgb,
            detection_model=detection_model,
            slice_height=args.slice_height,
            slice_width=args.slice_width,
            overlap_height_ratio=args.overlap_height_ratio,
            overlap_width_ratio=args.overlap_width_ratio,
            perform_standard_pred=args.perform_standard_pred,
            postprocess_type=args.postprocess_type,
            postprocess_match_metric=args.postprocess_match_metric,
            postprocess_match_threshold=args.postprocess_match_threshold,
            postprocess_class_agnostic=True,
            verbose=0,
        )
        elapsed = time.time() - start_time

        records = prediction_result_to_records(prediction_result)
        visualized_output = draw_detections(image_bgr, records)
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
                "engine": "sahi",
                "config_file": str(Path(args.config_file).expanduser().resolve()),
                "weights": str(Path(args.weights).expanduser().resolve()),
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "confidence_threshold": args.confidence_threshold,
                "device": args.device,
                "output_class_names": [OUTPUT_CLASS_NAME],
                "slice_height": args.slice_height,
                "slice_width": args.slice_width,
                "overlap_height_ratio": args.overlap_height_ratio,
                "overlap_width_ratio": args.overlap_width_ratio,
                "postprocess_type": args.postprocess_type,
                "postprocess_match_metric": args.postprocess_match_metric,
                "postprocess_match_threshold": args.postprocess_match_threshold,
                "perform_standard_pred": args.perform_standard_pred,
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
