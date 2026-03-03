#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOC_ROOT="${1:-$REPO_ROOT/datasets/aircraft_voc}"
COCO_DIR="$REPO_ROOT/datasets/coco"

if [ ! -d "$VOC_ROOT" ]; then
  echo "Dataset root not found: $VOC_ROOT"
  exit 1
fi

bash "$REPO_ROOT/scripts/create_runtime_dirs.sh"

rm -f "$COCO_DIR/train2017" "$COCO_DIR/val2017"
ln -s "../aircraft_voc/JPEGImages" "$COCO_DIR/train2017"
ln -s "../aircraft_voc/JPEGImages" "$COCO_DIR/val2017"

python3 "$REPO_ROOT/voc_to_coco_single_class.py" \
  --dataset-root "$VOC_ROOT" \
  --split train \
  --output-json "$REPO_ROOT/datasets/coco/annotations/instances_train2017.json"

python3 "$REPO_ROOT/voc_to_coco_single_class.py" \
  --dataset-root "$VOC_ROOT" \
  --split val \
  --output-json "$REPO_ROOT/datasets/coco/annotations/instances_val2017.json"

echo "Dataset prepared under $REPO_ROOT/datasets/coco"
