import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Convert a VOC dataset split into COCO JSON with a single class named 'aircraft'."
    )
    parser.add_argument("--dataset-root", required=True, help="VOC dataset root.")
    parser.add_argument("--split", required=True, choices=["train", "val", "test", "trainval"])
    parser.add_argument("--output-json", required=True, help="Destination COCO json path.")
    parser.add_argument(
        "--image-extension",
        default=".jpg",
        help="Image extension used by the dataset, default: .jpg",
    )
    return parser


def get_image_ids(split_file):
    return [line.strip() for line in split_file.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_annotation(xml_path):
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = float(bbox.find("xmin").text)
        ymin = float(bbox.find("ymin").text)
        xmax = float(bbox.find("xmax").text)
        ymax = float(bbox.find("ymax").text)
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append((xmin, ymin, xmax, ymax))
    return width, height, boxes


def convert(dataset_root, split, output_json, image_extension):
    dataset_root = Path(dataset_root).expanduser().resolve()
    output_json = Path(output_json).expanduser().resolve()

    split_file = dataset_root / "ImageSets" / "Main" / f"{split}.txt"
    annotations_dir = dataset_root / "Annotations"
    images_dir = dataset_root / "JPEGImages"

    if not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_ids = get_image_ids(split_file)

    coco = {
        "info": {"description": f"Single-class aircraft dataset ({split})"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "aircraft", "supercategory": "aircraft"}],
    }

    ann_id = 1
    for image_id_str in image_ids:
        xml_path = annotations_dir / f"{image_id_str}.xml"
        image_path = images_dir / f"{image_id_str}{image_extension}"

        if not xml_path.exists():
            raise FileNotFoundError(f"Missing annotation: {xml_path}")
        if not image_path.exists():
            raise FileNotFoundError(f"Missing image: {image_path}")

        width, height, boxes = parse_annotation(xml_path)
        image_id = int(image_id_str)
        coco["images"].append(
            {
                "id": image_id,
                "file_name": image_path.name,
                "width": width,
                "height": height,
            }
        )

        for xmin, ymin, xmax, ymax in boxes:
            box_w = xmax - xmin
            box_h = ymax - ymin
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": 1,
                    "bbox": [xmin, ymin, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(coco), encoding="utf-8")
    print(f"Saved {split} split to {output_json}")
    print(f"Images: {len(coco['images'])}")
    print(f"Annotations: {len(coco['annotations'])}")


if __name__ == "__main__":
    args = build_parser().parse_args()
    convert(args.dataset_root, args.split, args.output_json, args.image_extension)
