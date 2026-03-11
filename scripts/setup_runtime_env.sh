#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
INSTALL_TORCH="${INSTALL_TORCH:-1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

usage() {
  cat <<'USAGE'
Uso:
  bash scripts/setup_runtime_env.sh [venv_dir]

Variaveis opcionais:
  PYTHON_BIN=<python>      (default: python3)
  INSTALL_TORCH=0|1        (default: 1)
  TORCH_INDEX_URL=<url>    (default: cu121)
USAGE
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -gt 1 ]; then
  echo "Numero de argumentos invalido." >&2
  usage
  exit 1
fi

VENV_DIR="${1:-$REPO_ROOT/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python nao encontrado: $PYTHON_BIN" >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools

if [ "$INSTALL_TORCH" = "1" ]; then
  python -m pip install torch torchvision --index-url "$TORCH_INDEX_URL"
fi

python -m pip install -r "$REPO_ROOT/requirements-server.txt"
python -m pip install -e "$REPO_ROOT"

echo "Ambiente pronto em: $VENV_DIR"
