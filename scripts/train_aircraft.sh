#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
NUM_GPUS="${1:-1}"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"

if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="python3"
fi

cd "$REPO_ROOT"

"$PYTHON_BIN" train_net.py \
  --num-gpus "$NUM_GPUS" \
  --config-file configs/diffdet.aircraft.single_class.yaml
