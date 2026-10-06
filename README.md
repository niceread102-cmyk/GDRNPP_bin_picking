# Bin-picking 6DoF pose estimation

Three stages, one config file ([configs/default.yaml](configs/default.yaml)):

1. **Detector**: torchvision Mask R-CNN (1 class) separates touching parts → instance masks.
2. **Pose net**: GDRN-style (GDRNPP architecture, re-implemented in plain PyTorch). For each detected
   instance it takes a zoomed crop of RGB, depth (xyz relative to the part surface, plus a validity channel)
   and the instance mask. From that it predicts dense model-frame coordinates, surface regions and masks,
   and a Patch-PnP head turns those into rotation (allocentric 6D) and translation (SITE).
   Symmetry-aware (2-fold about z).
3. **ICP**: point-to-point ICP of the observed depth points onto the CAD surface, coarse to fine, with
   large corrections rejected.

Training targets (xyz maps, visible/full masks, regions) are generated on the fly by ray-casting
`models/obj_000001.ply` at the GT pose, so no pre-rendering step is needed.

## Setup
```powershell
powershell -ExecutionPolicy Bypass -File setup_venv.ps1   # .venv with torch 2.6 + CUDA 12.4
```
Data, weights and outputs are not in git (see `.gitignore`). Put the dataset at
`data/bin_ds_side_plate_bp_sample/shard_*`, or point `data.root` at it. The GDRNPP checkpoint downloads to
`pretrained/` on the first training run, and training outputs go to `runs/`.

## Train
```powershell
.venv\Scripts\python train_detector.py                       # -> runs/detector/best.pt
.venv\Scripts\python train_pose.py                           # -> runs/pose/best.pt
.venv\Scripts\tensorboard --logdir runs
```
`pose.use_pretrained_model: true` (the default) initialises the ConvNeXt-base backbone from the official
GDRNPP ITODD checkpoint (downloaded to `pretrained/` if missing). Set it to `false` to start from ImageNet
instead, which is worth comparing once.

Both can run at the same time on a 24 GB GPU. Resume with `--resume`. Override any config value with
`--set`, e.g. `--set pose.backbone=convnext_small --set pose.train.epochs=200`.

## Evaluate / run
```powershell
.venv\Scripts\python infer.py --pose runs/pose/best.pt --gt-masks                        # pose + ICP only
.venv\Scripts\python infer.py --detector runs/detector/best.pt --pose runs/pose/best.pt  # full pipeline
```
Writes per-frame JSON (poses in camera frame, mm) and overlay PNGs to `runs/infer`. Overlay colours:
green = ADD-S < 0.1·diameter, red = worse, yellow = no GT match.

Each detection in the JSON has a label-free `confident` flag: ICP accepted, fitness ≥ 0.9, and fit error
≤ 1.3 × the frame's median fit error (capped at 1.5 mm). See `refine.confidence_*` in the config.

## Visual validation and summary
```powershell
.venv\Scripts\python visualize.py --run runs/eval_full --mode 2d [--show confident|problems]   # PNG overlays
.venv\Scripts\python visualize.py --run runs/eval_full --view 60 --mode 3d                     # 3D window
.venv\Scripts\python visualize.py --run runs/eval_full --view 60 --mode html                   # 3D in a browser
```
Boxes: green = confident and correct, red = confident but wrong, orange = not confident,
thin white = GT where the pose is not ok, magenta = visible GT part the detector missed.
Axes: x red, y green, z blue.

[notebooks/results_summary.ipynb](notebooks/results_summary.ipynb) summarises the data, the pipeline, the
training curves, the evaluation tables, the confidence flag and the 2D/3D views. It is recomputed from `runs/`,
so re-run it after retraining. Kernel: "Python (bin6d .venv)".

## Data notes
- Split train/val by **scene** (`data.val_scenes`). Within a scene the pile only changes every 10 views.
- Picked parts stay in `scene_gt.json`, parked outside the bin. `bin_dataset.BinPoseDataset` drops them.
- `visib_fract`, `is_topmost` and `n_valid_grasps` in the dataset are unpopulated. Use visible pixel counts.
