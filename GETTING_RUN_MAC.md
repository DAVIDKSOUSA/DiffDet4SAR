# Pasta com tifs
mkdir -p meus_tifs 

# Pasta com png após inferência
mkdir -p AIRCRAFT_PRED 

# Ativar o bundle
cd /Users/davidsousa/GitHub/DiffDet4SAR/deploy/diffdet4sar_inference_clean
source .venv/bin/activate

# Código com configurações para rodar no terminal MAC (CPU)
python infer_tif_dir_sahi.py \
  --config-file configs/diffdet.aircraft.single_class.yaml \
  --weights model_final.pth \
  --input-dir /Users/davidsousa/GitHub/DiffDet4SAR/meus_tifs \
  --output-dir /Users/davidsousa/GitHub/DiffDet4SAR/AIRCRAFT_PRED \
  --device cpu \
  --confidence-threshold 0.5 \
  --slice-height 1024 \
  --slice-width 1024 \
  --overlap-height-ratio 0.25 \
  --overlap-width-ratio 0.25 \
  --recursive

