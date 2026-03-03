#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS="${1:-1}"

cd "$REPO_ROOT"

python3 train_net.py \
  --num-gpus "$NUM_GPUS" \
  --config-file configs/diffdet.aircraft.single_class.yaml
