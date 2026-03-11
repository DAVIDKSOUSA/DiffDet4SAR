#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_WEIGHTS="$REPO_ROOT/output_aircraft_single_class/model_final.pth"
DEFAULT_CONFIG="$REPO_ROOT/configs/diffdet.aircraft.single_class.yaml"
DEFAULT_BASE_CONFIG="$REPO_ROOT/configs/Base-DiffusionDet.yaml"
DEFAULT_OUT_DIR="$REPO_ROOT/deploy"
DEFAULT_BUNDLE_NAME="diffdet4sar_inference_$(date +%Y%m%d_%H%M%S)"

WEIGHTS="$DEFAULT_WEIGHTS"
CONFIG="$DEFAULT_CONFIG"
BASE_CONFIG="$DEFAULT_BASE_CONFIG"
OUT_DIR="$DEFAULT_OUT_DIR"
BUNDLE_NAME="$DEFAULT_BUNDLE_NAME"

usage() {
  cat <<'USAGE'
Uso:
  bash scripts/build_inference_bundle.sh [opcoes]

Opcoes:
  --weights <path>        Caminho do .pth (default: output_aircraft_single_class/model_final.pth)
  --config <path>         Caminho do yaml principal (default: configs/diffdet.aircraft.single_class.yaml)
  --base-config <path>    Caminho do yaml base (default: configs/Base-DiffusionDet.yaml)
  --out-dir <path>        Pasta de saida do bundle (default: deploy/)
  --bundle-name <name>    Nome do pacote sem extensao
  -h, --help              Mostra esta ajuda
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --weights)
      WEIGHTS="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="${2:-}"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="${2:-}"
      shift 2
      ;;
    --bundle-name)
      BUNDLE_NAME="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Opcao invalida: $1" >&2
      usage
      exit 1
      ;;
  esac
done

required_files=(
  "$WEIGHTS"
  "$CONFIG"
  "$BASE_CONFIG"
  "$REPO_ROOT/infer_tif_dir.py"
  "$REPO_ROOT/requirements-server.txt"
  "$REPO_ROOT/setup.py"
  "$REPO_ROOT/scripts/setup_runtime_env.sh"
)

for file in "${required_files[@]}"; do
  if [ ! -f "$file" ]; then
    echo "Arquivo obrigatorio nao encontrado: $file" >&2
    exit 1
  fi
done

STAGING_DIR="$(mktemp -d)"
cleanup() {
  rm -rf "$STAGING_DIR"
}
trap cleanup EXIT

mkdir -p "$OUT_DIR"
BUNDLE_ROOT="$STAGING_DIR/$BUNDLE_NAME"
mkdir -p "$BUNDLE_ROOT/configs" "$BUNDLE_ROOT/scripts"

cp "$WEIGHTS" "$BUNDLE_ROOT/model_final.pth"
cp "$CONFIG" "$BUNDLE_ROOT/configs/$(basename "$CONFIG")"
cp "$BASE_CONFIG" "$BUNDLE_ROOT/configs/$(basename "$BASE_CONFIG")"
cp "$REPO_ROOT/infer_tif_dir.py" "$BUNDLE_ROOT/"
cp "$REPO_ROOT/requirements-server.txt" "$BUNDLE_ROOT/"
cp "$REPO_ROOT/setup.py" "$BUNDLE_ROOT/"
cp "$REPO_ROOT/scripts/setup_runtime_env.sh" "$BUNDLE_ROOT/scripts/"
cp -r "$REPO_ROOT/detectron2" "$BUNDLE_ROOT/"
cp -r "$REPO_ROOT/diffusiondet" "$BUNDLE_ROOT/"

# Remove cache and compiled artifacts from the bundle.
find "$BUNDLE_ROOT/detectron2" "$BUNDLE_ROOT/diffusiondet" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$BUNDLE_ROOT/detectron2" "$BUNDLE_ROOT/diffusiondet" -type f \
  \( -name "*.pyc" -o -name "*.pyo" -o -name "*.so" \) -delete

cat > "$BUNDLE_ROOT/README_BUNDLE.md" <<README
# DiffDet4SAR - Bundle de Inferencia

Este pacote contem tudo para inferencia em outra maquina:
- model_final.pth
- configs/
- infer_tif_dir.py
- detectron2/ e diffusiondet/
- setup.py
- requirements-server.txt

## 1) Preparar ambiente

\`\`\`bash
bash scripts/setup_runtime_env.sh
source .venv/bin/activate
\`\`\`

Obs: o setup recompila a extensao local do detectron2 para a maquina de destino.

Se precisar de outro wheel CUDA:

\`\`\`bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_runtime_env.sh
\`\`\`

## 2) Rodar inferencia

\`\`\`bash
python infer_tif_dir.py \
  --config-file configs/$(basename "$CONFIG") \
  --weights model_final.pth \
  --input-dir /caminho/para/imagens_tif \
  --output-dir /caminho/para/saida \
  --confidence-threshold 0.5 \
  --tile-size 1024 \
  --tile-overlap 256 \
  --nms-threshold 0.5 \
  --recursive
\`\`\`
README

ARCHIVE_PATH="$OUT_DIR/$BUNDLE_NAME.tar.gz"
tar -C "$STAGING_DIR" -czf "$ARCHIVE_PATH" "$BUNDLE_NAME"
sha256sum "$ARCHIVE_PATH" > "$ARCHIVE_PATH.sha256"

echo "Bundle criado:"
echo "  $ARCHIVE_PATH"
echo "Checksum:"
echo "  $ARCHIVE_PATH.sha256"
