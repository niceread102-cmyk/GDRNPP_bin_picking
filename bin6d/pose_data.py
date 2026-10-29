"""Per-instance crop dataset for the GDRN-style pose network.

One item = one frame, returning `crops_per_frame` instance crops (images are decoded once per frame).
Dense targets (full mask, visible mask, model-frame xyz, surface region) are generated on the fly by
ray-casting the CAD mesh at the GT pose through each output pixel, so no pre-rendering is needed.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_conversion.bin_dataset import BinPoseDataset, backproject, read_ply_mesh
from bin6d.augment import depth_dropout_and_noise, photometric
from bin6d.geometry import (IMAGENET_MEAN, IMAGENET_STD, bbox_to_crop, canonicalize_symmetric_pose,
                            crop_pixel_grid, ego_to_allo, encode_site, farthest_point_sampling, pixel_rays,
                            random_camera_rotation, sample_crops, sample_surface, warp_to_source)


class ObjectModel:
    """CAD model + metadata needed by the network, losses and refinement (all mm, model frame)."""

    def __init__(self, frames: BinPoseDataset, obj_id: int, num_regions: int, num_points: int):
        self.path = frames.shards[0] / "models" / f"obj_{obj_id:06d}.ply"
        self.verts, self.faces = read_ply_mesh(self.path)
        info = frames.models_info[obj_id]
        self.diameter = float(info["diameter"])
        self.xyz_min = np.array([info["min_x"], info["min_y"], info["min_z"]], dtype=np.float32)
        self.xyz_size = np.array([info["size_x"], info["size_y"], info["size_z"]], dtype=np.float32)
        syms = np.stack(frames.symmetries(obj_id)).astype(np.float32)
        self.sym_R, self.sym_t = syms[:, :3, :3], syms[:, :3, 3]
        self.points = sample_surface(self.verts, self.faces, num_points, seed=1)
        self.dense_points, self.dense_normals = sample_surface(self.verts, self.faces, 20000, seed=2,
                                                               return_normals=True)   # ICP target
        self.region_centers = (farthest_point_sampling(sample_surface(self.verts, self.faces, 5000, seed=3),
                                                       num_regions) if num_regions else None)


def input_channels() -> int:
    return 3 + 3 + 1 + 1    # rgb, xyz relative to target surface, depth valid, target mask


def sample_raw_crops(rgb: torch.Tensor, xyz: torch.Tensor, valid: torch.Tensor, uv_in: torch.Tensor):
    """rgb (3,H,W) 0..1, xyz (3,H,W) camera mm, valid (1,H,W) sampled at uv_in (B,r,r,2) source pixels."""
    return (sample_crops(rgb, uv_in, "bilinear"), sample_crops(xyz, uv_in, "nearest"),
            sample_crops(valid, uv_in, "nearest"))


def assemble_input(img: torch.Tensor, pts: torch.Tensor, val: torch.Tensor, masks: torch.Tensor,
                   diameter: float) -> torch.Tensor:
    """Crops (img 0..1, pts camera mm, val, target masks) -> (B, 8, r, r) network input."""
    dev = img.device
    img = (img - IMAGENET_MEAN.to(dev)) / IMAGENET_STD.to(dev)
    # reference point = median of the target's observed surface (falls back to all valid points in crop)
    ref = torch.zeros(len(pts), 3, device=dev)
    for i in range(len(pts)):
        sel = (val[i, 0] > 0) & (masks[i, 0] > 0)
        if sel.sum() < 5:
            sel = val[i, 0] > 0
        if sel.any():
            ref[i] = pts[i][:, sel].median(dim=1).values
    rel = ((pts - ref[:, :, None, None]) / diameter).clamp(-3, 3) * val
    return torch.cat([img, rel, val, masks], dim=1)


def build_network_input(rgb: torch.Tensor, xyz: torch.Tensor, valid: torch.Tensor, masks: torch.Tensor,
                        uv_in: torch.Tensor, diameter: float) -> torch.Tensor:
    """Inference path: sample B crops of one frame and assemble the (B, 8, r, r) input (no augmentation)."""
    return assemble_input(*sample_raw_crops(rgb, xyz, valid, uv_in), masks, diameter)


def frame_planes(rgb: np.ndarray, depth: np.ndarray, K: np.ndarray):
    rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    xyz_t = torch.from_numpy(backproject(depth, K)).permute(2, 0, 1).contiguous()
    valid_t = torch.from_numpy(depth > 0).float()[None]
    return rgb_t, xyz_t, valid_t


def perturb_masks(m: torch.Tensor, max_px: int, prob: float, gen: np.random.Generator) -> torch.Tensor:
    """Randomly dilate / erode binary masks (B,1,r,r) so the net tolerates imperfect detector masks."""
    out = m.clone()
    for i in range(len(m)):
        if max_px <= 0 or gen.random() > prob:
            continue
        k = 2 * int(gen.integers(1, max_px + 1)) + 1
        if gen.random() < 0.5:
            out[i] = F.max_pool2d(m[i:i + 1], k, 1, k // 2)[0]
        else:
            out[i] = -F.max_pool2d(-m[i:i + 1], k, 1, k // 2)[0]
    return out


class PoseCropDataset(Dataset):
    def __init__(self, frames: BinPoseDataset, model: ObjectModel, cfg, train: bool, obj_id: int = 1):
        self.frames, self.model, self.cfg, self.train, self.obj_id = frames, model, cfg, train, obj_id
        self.k = cfg.crops_per_frame if train else cfg.eval.crops_per_frame
        self._ray_scene = None     # open3d scene, built lazily per worker (not picklable)
        self._gen = None

    def __len__(self) -> int:
        return len(self.frames)

    # ------------------------------------------------------------ lazy per-worker state
    def _raycaster(self):
        if self._ray_scene is None:
            import open3d as o3d
            mesh = o3d.t.geometry.TriangleMesh()
            mesh.vertex.positions = o3d.core.Tensor(self.model.verts)
            mesh.triangle.indices = o3d.core.Tensor(self.model.faces.astype(np.int32))
            self._ray_scene = o3d.t.geometry.RaycastingScene()
            self._ray_scene.add_triangles(mesh)
        return self._ray_scene

    def _rng(self, index: int) -> np.random.Generator:
        if not self.train:
            return np.random.default_rng(index)          # fixed val crops
        if self._gen is None:
            self._gen = np.random.default_rng(torch.initial_seed() % 2**32)
        return self._gen

    # ------------------------------------------------------------ targets
    def _raycast_targets(self, R: torch.Tensor, t: torch.Tensor, rays: torch.Tensor):
        """Cast rays (B,2,m,m normalized coords) at poses (R,t) -> hit mask (B,m,m), model xyz (B,m,m,3)."""
        B, _, m, _ = rays.shape
        d_cam = torch.cat([rays, torch.ones(B, 1, m, m)], dim=1).permute(0, 2, 3, 1).reshape(B, -1, 3)
        Rt = R.transpose(1, 2)
        d_obj = d_cam @ R                                     # R^T d  (row-vector form)
        o_obj = -(Rt @ t[..., None])[..., 0]                  # camera centre in model frame
        o = o_obj[:, None].expand_as(d_obj)
        import open3d as o3d
        ray_t = o3d.core.Tensor(torch.cat([o, d_obj], -1).reshape(-1, 6).numpy().astype(np.float32))
        t_hit = torch.from_numpy(self._raycaster().cast_rays(ray_t)["t_hit"].numpy()).view(B, -1)
        hit = torch.isfinite(t_hit)
        xyz = o + d_obj * torch.where(hit, t_hit, torch.zeros_like(t_hit))[..., None]
        return hit.view(B, m, m), xyz.view(B, m, m, 3)

    def __getitem__(self, index: int) -> dict:
        cfg, mdl = self.cfg, self.model
        gen = self._rng(index)
        meta = self.frames.frame_meta(index)
        objs = meta["objects_all"]
        eligible = np.flatnonzero((objs["visib_px"] >= cfg.min_visib_px) & (objs["obj_id"] == self.obj_id))
        if len(eligible) == 0:                                 # nearly empty bin: take the most visible part
            eligible = np.array([int(np.argmax(objs["visib_px"]))])
        pick = gen.choice(eligible, self.k, replace=len(eligible) < self.k)

        rgb, depth, inst = self.frames.load_images(index, meta["depth_scale"])
        K = torch.from_numpy(meta["K"])
        rgb_t, xyz_t, valid_t = frame_planes(rgb, depth, meta["K"])
        inst_t = torch.from_numpy(inst).float()[None]

        bbox = torch.from_numpy(objs["bbox_visib"][pick])
        center, size = bbox_to_crop(bbox, cfg.pad_scale, cfg.min_crop_px)
        if self.train:                                         # dynamic zoom-in
            wh = bbox[:, 2:]
            center = center + torch.from_numpy(gen.uniform(-cfg.dzi_shift, cfg.dzi_shift, (self.k, 2))).float() * wh
            size = size * torch.from_numpy(gen.uniform(1 - cfg.dzi_scale, 1 + cfg.dzi_scale, self.k)).float()
            size = size.clamp(min=cfg.min_crop_px)

        inst_id = torch.from_numpy(objs["inst_id"][pick]).float()
        Kb = K.expand(self.k, 3, 3)
        R = torch.from_numpy(objs["R"][pick])
        t = torch.from_numpy(objs["t"][pick])

        # Virtual camera rotation (exact): the crop is rendered from a camera rotated by Q, so every
        # quantity below lives in that virtual camera: pixels via homography, points and pose via Q.
        a = cfg.aug
        Q = torch.eye(3).expand(self.k, 3, 3).clone()
        if self.train:
            rot_on = torch.from_numpy(gen.random(self.k) < a.rotate_prob)
            Q_r, center_r = random_camera_rotation(center, Kb, a.roll_deg, a.tilt_deg, gen)
            Q = torch.where(rot_on[:, None, None], Q_r, Q)
            center = torch.where(rot_on[:, None], center_r, center)
        R, t = Q @ R, (Q @ t[..., None])[..., 0]
        uv_in = warp_to_source(crop_pixel_grid(center, size, cfg.input_res), Kb, Q)     # source-image pixels
        uv_out_v = crop_pixel_grid(center, size, cfg.output_res)                       # virtual-image pixels

        img, pts, val = sample_raw_crops(rgb_t, xyz_t, valid_t, uv_in)
        pts = torch.einsum("bij,bjhw->bihw", Q, pts)                                   # into virtual camera
        in_mask = (sample_crops(inst_t, uv_in, "nearest") == inst_id[:, None, None, None]).float()
        if self.train:
            in_mask = perturb_masks(in_mask, cfg.mask_noise_px, cfg.mask_noise_prob, gen)
            img = photometric(img, a, gen)
            pts, val = depth_dropout_and_noise(pts, val, a, gen)
        x = assemble_input(img, pts, val, in_mask, mdl.diameter)

        # canonical (symmetry-resolved) GT pose, in the (virtual) camera
        R_c, t_c, _ = canonicalize_symmetric_pose(R, t, torch.from_numpy(mdl.sym_R), torch.from_numpy(mdl.sym_t))

        rays = pixel_rays(uv_out_v, Kb)
        hit, xyz_model = self._raycast_targets(R_c, t_c, rays)
        uv_out_src = warp_to_source(uv_out_v, Kb, Q)
        visib = (sample_crops(inst_t, uv_out_src, "nearest")[:, 0] == inst_id[:, None, None]) & hit
        xyz_norm = ((xyz_model - torch.from_numpy(mdl.xyz_min)) / torch.from_numpy(mdl.xyz_size)).clamp(0, 1)
        sample = {
            "input": x,
            "rays": rays,
            "mask_full": hit.float()[:, None],
            "mask_visib": visib.float()[:, None],
            "xyz": xyz_norm.permute(0, 3, 1, 2).contiguous(),
            "R_allo": ego_to_allo(R_c, t_c),
            "R": R_c,
            "t": t_c,
            "site": encode_site(t_c, Kb, center, size, mdl.diameter),
            "center": center,
            "size": size,
            "K": Kb.clone(),
            "Q": Q,                                   # real camera -> (augmented) virtual camera rotation
            "visib_px": torch.from_numpy(objs["visib_px"][pick]),
        }
        if mdl.region_centers is not None:
            rc = torch.from_numpy(mdl.region_centers)
            region = torch.cdist(xyz_model.view(self.k, -1, 3), rc[None].expand(self.k, -1, -1)).argmin(-1) + 1
            sample["region"] = torch.where(visib.view(self.k, -1), region, 0).view(self.k, *visib.shape[1:])
        return sample


def cat_collate(batch: list[dict]) -> dict:
    return {k: torch.cat([b[k] for b in batch]) for k in batch[0]}
