#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 /absolute/path/to/local/VOC_dataset_root"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_VOC_ROOT="$1"
TARGET_VOC_ROOT="$REPO_ROOT/datasets/aircraft_voc"

mkdir -p "$TARGET_VOC_ROOT"

rsync -a "$SOURCE_VOC_ROOT/Annotations/" "$TARGET_VOC_ROOT/Annotations/"
rsync -a "$SOURCE_VOC_ROOT/ImageSets/" "$TARGET_VOC_ROOT/ImageSets/"
rsync -a "$SOURCE_VOC_ROOT/JPEGImages/" "$TARGET_VOC_ROOT/JPEGImages/"

echo "Copied local VOC dataset into:"
echo "  $TARGET_VOC_ROOT"
