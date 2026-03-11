# Server Setup

## 1. Push the repository to GitHub

From the project root:

```bash
git add train_net.py \
  infer_tif_dir.py \
  voc_to_coco_single_class.py \
  setup.py \
  requirements-server.txt \
  SERVER_SETUP.md \
  DEPLOY_OTHER_MACHINE.md \
  scripts/create_runtime_dirs.sh \
  scripts/setup_runtime_env.sh \
  scripts/build_inference_bundle.sh \
  scripts/import_local_training_dataset.sh \
  scripts/prepare_aircraft_dataset.sh \
  scripts/train_aircraft.sh \
  configs/diffdet.aircraft.single_class.yaml \
  configs/Base-DiffusionDet.yaml \
  configs/diffdet.coco.res50.300boxes.yaml \
  detectron2/engine/defaults.py \
  .gitignore

git commit -m "Prepare server training and inference workflow"
git push origin HEAD
```

## 2. Clone on the server

```bash
git clone <YOUR_GITHUB_REPO_URL>
cd DiffDet4SAR
```

## 3. Create the Python environment

Run:

```bash
bash scripts/setup_runtime_env.sh
source .venv/bin/activate
```

If your CUDA version is different, set the wheel index before setup:

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/setup_runtime_env.sh
```

## 4. Runtime folders

```bash
bash scripts/create_runtime_dirs.sh
```

This creates:

```text
datasets/coco/annotations
weights
output_aircraft_single_class
logs
```

## 5. Prepare the training dataset

If the repository already contains `datasets/aircraft_voc`, just run:

```bash
bash scripts/prepare_aircraft_dataset.sh
```

If the VOC dataset is outside the repository:

```bash
bash scripts/prepare_aircraft_dataset.sh /absolute/path/to/SAR_AIRcraft_dataset_root
```

Expected VOC structure:

```text
VOC_ROOT/
  Annotations/
  ImageSets/Main/
  JPEGImages/
```

## 6. Train

```bash
bash scripts/train_aircraft.sh 1
```

Expected final checkpoint:

```text
output_aircraft_single_class/model_final.pth
```

## 7. Run inference on `.tif`

```bash
python3 infer_tif_dir.py \
  --config-file configs/diffdet.aircraft.single_class.yaml \
  --weights output_aircraft_single_class/model_final.pth \
  --input-dir /absolute/path/to/GALEAO \
  --output-dir /absolute/path/to/AIRCRAFT_PRED \
  --confidence-threshold 0.5 \
  --tile-size 1024 \
  --tile-overlap 256 \
  --nms-threshold 0.5 \
  --recursive
```

## 8. If the dataset is still on your local machine

Copy it into the repository before pushing:

```bash
bash scripts/import_local_training_dataset.sh /absolute/path/to/local/VOC_ROOT
```

Then regenerate the COCO files inside the repository:

```bash
bash scripts/prepare_aircraft_dataset.sh
```

## 9. Build inference bundle for another machine

```bash
bash scripts/build_inference_bundle.sh
```

This creates `deploy/*.tar.gz` with code + config + `model_final.pth`.
