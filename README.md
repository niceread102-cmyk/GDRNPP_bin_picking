# Bin-picking 6DoF pose estimation (side plate)

6D poses of metal side plates in a bin from one RGB-D frame (Photoneo / Isaac Sim), for robot picking.
All settings live in one config file, [configs/default.yaml](configs/default.yaml).

| Stage | What it does |
|---|---|
| 1. Detector | torchvision Mask R-CNN (1 class) separates touching parts into instance masks |
| 2. Pose net | GDRN-style network (GDRNPP architecture in plain PyTorch). Per instance: a zoomed crop of RGB, depth (xyz relative to the part surface + validity) and the mask → dense model coordinates, surface regions and masks → Patch-PnP head → rotation (allocentric 6D) and translation (SITE). Symmetry-aware (2-fold about the part's z axis) |
| 3. Depth anchor | slides the pose along the viewing ray onto the measured depth (the network reads depth only from apparent size) |
| 4. ICP | point-to-plane ICP of the observed depth points onto the camera-facing CAD surface, 8 → 4 → 2 mm; large corrections rejected |
| 5. Clean-up | de-duplication of the same part, then **depth verification**: outputs the depth does not confirm (no depth, CAD does not fit, partial / duplicate masks) are dropped |
| 6. Confidence | label-free flag per pose: ICP fitness ≥ 0.9 and fit error ≤ 1.3 × the frame's median (cap 1.5 mm) |

Training targets (xyz maps, visible / full masks, regions) are ray-cast from the CAD at the GT pose on the fly,
so there is no pre-rendering step.

## 6D poses in simulation (Isaac Sim picking)

To get the 6D poses of the parts to pick in an Isaac Sim scene, run
[notebooks/results_summary_sim_grippose.ipynb](notebooks/results_summary_sim_grippose.ipynb) top to bottom
and read **§6 "Isaac Sim에서 집을 부품의 6D pose 출력"**.

1. Set up the environment ([Setup](#setup)) and select the `.venv` kernel in VS Code / Jupyter.
2. Make sure the inputs exist (none of them are in git, see [Data and weights](#data-and-weights)):
   - simulation renders `~/FFB6D/data/side_plate_10k` (`SRC_ROOT` in the notebook's first cell);
   - the simulation-trained models `runs/detector/best.pt` and `runs/pose/best.pt`.
3. Run all cells. The first run undistorts the renders (`data_conversion/undistort_bop.py`, the renders contain lens distortion)
   and runs `infer.py` (about 5 min per configuration on a GB10); later runs reuse the results in `runs/sim10k_*`.
4. §6 shows the selected parts on the image and prints their poses:
   - **selection**: confident poses with nothing on top (≤ 5 % of the visible CAD surface covered,
     `MAX_COVERED`), the `TOP_N` = 8 highest;
   - **position** = the **top end point** (yellow circle): where the part axis leaves the end face that points
     up (hub end or flat face, column `end`), not the part centre;
   - **orientation** = roll / pitch / yaw in degrees, fixed axes x → y → z (`R = Rz(yaw)·Ry(pitch)·Rx(roll)`),
     the same definition as the `*_roll/pitch/yaw_deg` columns of the dataset's `poses/*.csv`. The part is
     2-fold symmetric about its z axis, so yaw ± 180° is the same pose;
   - **frames**:

     | Table / CSV columns | Frame | Units |
     |---|---|---|
     | `end_x_m … yaw_deg` / `isaac_*` | Isaac Sim world = bin frame + bin position from `config_resolved.yaml` (`scene.bin.pos_m` = (1.0, 0, 0) m, no rotation) | m (CSV: mm), deg |
     | `bin_*` | bin frame: origin at the bin's bottom centre, x along the long side, z up (= the dataset's `scene_camera.json` world) | mm, deg |
     | `cam_*` | camera frame (OpenCV: x right, y down, z forward) | mm, deg |

   - the poses are saved to `runs/sim10k_full/pick_poses/<scene>_<view>.csv` (selected) and
     `..._all_confident.csv`; the part centre is kept in the `*_centre_x/y/z_mm` columns.
5. Change `SCENE`, `VIEW`, `TOP_N` and `MAX_COVERED` in §6 for another frame or selection. If the bin sits
   elsewhere in your Isaac Sim stage, set `bin_pos_mm` / `bin_rot` there.

On the 50 evaluated frames the exported poses are within 0.35 mm (position) and 1.1° (axis tilt) of the
simulator ground truth. §1–§5 of the notebook evaluate the whole pipeline on the same data.

## Setup
```bash
bash setup_venv.sh                    # Linux: picks the PyTorch build from the NVIDIA driver
TORCH_CUDA=cu130 bash setup_venv.sh   # force torch 2.14.1 + CUDA 13.0 (GB10 / DGX Spark, driver >= 580)
TORCH_CUDA=cu124 bash setup_venv.sh   # force torch 2.6 + CUDA 12.4 (driver >= 550); cpu also accepted
```
```powershell
powershell -ExecutionPolicy Bypass -File setup_venv.ps1                    # Windows: torch 2.6 + CUDA 12.4
powershell -ExecutionPolicy Bypass -File setup_venv.ps1 -TorchCuda cu130   # Windows, recent drivers
```
Both scripts create `.venv`, install `requirements.txt`, register the venv as the Jupyter kernel
"GDRNPP bin 6DoF (.venv)" and define the git filter that stores notebooks **without outputs**
(`.gitattributes`, `tools/strip_notebook_outputs.py`): local notebooks keep their figures, the repository gets
only the code. Re-run a notebook to regenerate its outputs from `runs/`.

Commands below use the Linux path `.venv/bin/python`; on Windows use `.venv\Scripts\python`.

## Data and weights

Data, weights and outputs are not in git (see `.gitignore`):

| Path | Content |
|---|---|
| `data/bin_ds_side_plate_bp_sample/shard_*` | simulation training data (BOP layout; `data.root` in the config) |
| `data/pseudo_gt_70_bop` | 70 real Photoneo captures with FoundationPose pseudo-GT (fine-tuning / evaluation) |
| `data/real_capture_bop` | real captures without GT (`camera_convert_data.py --to-bop`) |
| `data/side_plate_10k_undist` | undistorted copy of `~/FFB6D/data/side_plate_10k` (made by the sim notebook) |
| `pretrained/` | official GDRNPP ITODD checkpoint, downloaded on the first pose training run |
| `runs/` | checkpoints (`detector/`, `pose/`, `detector_ft/`, `pose_ft/`), logs and `infer.py` outputs |

## Train
```bash
.venv/bin/python train/train_detector.py          # -> runs/detector/best.pt  (simulation)
.venv/bin/python train/train_pose.py              # -> runs/pose/best.pt      (simulation)
.venv/bin/tensorboard --logdir runs
```
`pose.use_pretrained_model: true` (default) initialises the ConvNeXt-base backbone from the official GDRNPP ITODD
checkpoint. Resume with `--resume`. Override any config value with `--set`, e.g.
`--set pose.backbone=convnext_small --set pose.train.epochs=200`. Both trainings fit together on a 24 GB GPU.

**Fine-tuning on real captures** (simulation + real frames in every batch, real frames split by view):
```bash
.venv/bin/python train/finetune/finetuning_detector.py     # -> runs/detector_ft/best.pt  (detector.finetune in the config)
.venv/bin/python train/finetune/finetuning_pose.py         # -> runs/pose_ft/best.pt      (pose.finetune in the config)
```
The fine-tuned detector removes most false detections and the missed part orientation on real Photoneo data;
the fine-tuned pose net does not change the final (post-ICP) accuracy, so `runs/pose/best.pt` stays the default.
See [notebooks/results_photoneo_summary.ipynb](notebooks/results_photoneo_summary.ipynb) §6 and §8.

## Inference
```bash
.venv/bin/python infer.py --set data.root=data/pseudo_gt_70_bop --scenes 000000 --view-stride 1 --out runs/my_run
.venv/bin/python infer.py --gt-masks ...                                     # skip the detector (GT masks)
.venv/bin/python infer.py --detector runs/detector/best.pt ...               # simulation-only detector
```
Defaults: `--detector runs/detector_ft/best.pt` (fine-tuned on real data), `--pose runs/pose/best.pt`.
Writes one JSON per frame to `--out` (poses in the camera frame, mm) plus overlay PNGs (`--no-viz` to skip);
detections removed by the depth verification go to `<out>/rejected/` with the reason.
Useful switches: `--set refine.reject_unverified=false` (keep unverified detections),
`--set refine.depth_anchor=false`, `--set infer.det_scale=0.55` (parts look larger in pixels than in training,
e.g. another camera resolution).

Each detection has a label-free `confident` flag (see `refine.confidence_*`); pick only confident poses.

## Real captures and other cameras
- `data_conversion/camera_convert_data.py`: capture RGB / depth / point clouds from a Photoneo over GigE Vision (needs the
  harvesters / Photoneo examples venv), and `--to-bop` to convert the captures to the BOP layout, undistorted onto
  the pinhole K. Capture at the training camera setting (PrimaryCamera, 1236×1032, the default).
- `data_conversion/undistort_bop.py`: resample a lens-distorted BOP dataset (rgb, depth, masks) onto its pinhole K. The pipeline
  back-projects depth with K only, so distorted images put edge parts in the wrong place.

## Visual validation
```bash
.venv/bin/python visualize.py --run runs/my_run --mode 2d [--show confident|problems]   # PNG overlays
.venv/bin/python visualize.py --run runs/my_run --view 60 --mode 3d                     # 3D window
.venv/bin/python visualize.py --run runs/my_run --view 60 --mode html                   # 3D in a browser
```
Boxes: green = confident and correct, red = confident but wrong, orange = not confident,
thin white = GT where the pose is not ok, magenta = visible GT part the detector missed. Axes: x red, y green, z blue.

## Notebooks
All are recomputed from `runs/` (and run `infer.py` for missing results); re-run them after retraining.

| Notebook | Content |
|---|---|
| [results_summary_sim_grippose.ipynb](notebooks/results_summary_sim_grippose.ipynb) | simulation (`side_plate_10k`), simulation-trained models: accuracy vs exact GT, and **§6: 6D poses for Isaac Sim picking** |
| [results_photoneo_summary.ipynb](notebooks/results_photoneo_summary.ipynb) | 70 real Photoneo captures vs pseudo-GT: depth anchor, depth verification, detector / pose fine-tuning, a full real bin without GT |
| [results_summary.ipynb](notebooks/results_summary.ipynb) | original simulation summary: training curves and evaluation on the training data's validation scene |
| [results_summary_sim.ipynb](notebooks/results_summary_sim.ipynb), [results_real_objdummy_summary.ipynb](notebooks/results_real_objdummy_summary.ipynb) | earlier variants |

## Data notes
- Split train / val by **scene** (`data.val_scenes`): within a scene the pile only changes every 10 views.
- Picked parts stay in `scene_gt.json`, parked outside the bin; `bin_dataset.BinPoseDataset` drops them.
- `visib_fract`, `is_topmost` and `n_valid_grasps` in the dataset are unpopulated; use visible pixel counts.
- Part model: `obj_000001.ply` in mm, origin at the bounding-box centre, z = plate axis (+z towards the hub),
  diameter 55.7 mm, 20.9 mm thick.
