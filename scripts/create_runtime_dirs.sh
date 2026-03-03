#!/usr/bin/env bash
set -euo pipefail

mkdir -p datasets/coco/annotations
mkdir -p weights
mkdir -p output_aircraft_single_class
mkdir -p logs

echo "Created runtime directories:"
echo "  datasets/coco/annotations"
echo "  weights"
echo "  output_aircraft_single_class"
echo "  logs"
