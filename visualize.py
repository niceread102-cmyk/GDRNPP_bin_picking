"""Validation displays for infer.py results: detected parts' 6DoF as 3D boxes + xyz axes, vs GT labels.

    # 2D overlays for every frame of a run (PNG)
    .venv\\Scripts\\python visualize.py --run runs/eval_full --mode 2d
    # interactive 3D window for one frame (rotate with mouse); --screenshot saves an image instead
    .venv\\Scripts\\python visualize.py --run runs/eval_full --view 0 --mode 3d
    # interactive 3D as a standalone HTML page (opens in any browser)
    .venv\\Scripts\\python visualize.py --run runs/eval_full --view 0 --mode html

Colours: green = confident & correct, red = confident but WRONG, orange = not confident,
thin white = GT label where the pose is not ok, magenta = visible GT part the detector missed.
--show confident | problems filters the display (full bins are dense).
"""
from __future__ import annotations

from pathlib import Path

from data_conversion.bin_dataset import BinPoseDataset
from bin6d.config import parse_args
from bin6d.pose_data import ObjectModel
from bin6d.report import frame_index, load_frame, load_run
from bin6d.viz import gt_overlay_sets, open3d_scene, plotly_scene, render_2d, select


def extra_args(ap):
    ap.add_argument("--run", default="runs/eval_full", help="infer.py output directory")
    ap.add_argument("--scene", default=None, help="default: the run's (first) scene")
    ap.add_argument("--view", type=int, default=None, help="one view; default: all views in the run (2d only)")
    ap.add_argument("--mode", choices=["2d", "3d", "html"], default="2d")
    ap.add_argument("--out", default=None, help="default: <run>/viz")
    ap.add_argument("--show", choices=["all", "confident", "problems"], default="all",
                    help="problems = confident-but-wrong, not confident, and missed GT")
    ap.add_argument("--axes", choices=["confident", "all", "none"], default="confident")
    ap.add_argument("--labels", choices=["wrong", "all", "none"], default="wrong")
    ap.add_argument("--no-gt", action="store_true", help="hide GT boxes")
    ap.add_argument("--hidden", action="store_true", help="3d/html: also show labels of fully occluded parts")
    ap.add_argument("--screenshot", default=None, help="3d: save an off-screen render to this path")


def main():
    args, cfg = parse_args(__doc__, extra_args)
    run = load_run(args.run, cfg.refine)
    if not run:
        raise SystemExit(f"no infer.py results in {args.run}")
    scene = args.scene or sorted({s for s, _ in run})[0]
    keys = [(scene, args.view)] if args.view is not None else sorted(k for k in run if k[0] == scene)
    if args.mode != "2d" and len(keys) != 1:
        raise SystemExit("--mode 3d/html needs --view")
    frames = BinPoseDataset(cfg.data.root, scenes=[scene], bin_margin_mm=cfg.data.bin_margin_mm)
    model = ObjectModel(frames, cfg.data.obj_id, 0, 1000)
    idx = frame_index(frames)
    out_dir = Path(args.out or Path(args.run) / "viz")
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        dets = run[key]
        f = load_frame(frames, idx[key])
        objs = f["objects_all"]
        min_px = 1 if args.hidden and args.mode != "2d" else cfg.pose.min_visib_px
        gt = None if args.no_gt else gt_overlay_sets(objs, dets, min_px)
        if args.hidden and gt is not None and args.mode != "2d":
            gt += [("gt", objs["R"][k], objs["t"][k]) for k in range(len(objs["t"])) if objs["visib_px"][k] == 0]
        dets, gt = select(dets, gt, args.show)
        tag = f"{key[0]}_{key[1]:06d}" + ("" if args.show == "all" else f"_{args.show}")
        if args.mode == "2d":
            path = out_dir / f"{tag}.png"
            render_2d(f["rgb"], f["K"], dets, model, gt, axes=args.axes, labels=args.labels, title=tag).save(path)
        elif args.mode == "html":
            path = out_dir / f"{tag}_3d.html"
            plotly_scene(f["rgb"], f["xyz"], f["valid"], dets, model, gt, axes=args.axes, title=tag).write_html(path)
        else:
            path = args.screenshot
            open3d_scene(f["rgb"], f["xyz"], f["valid"], dets, model, gt, axes=args.axes, screenshot=path)
        if path:
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
