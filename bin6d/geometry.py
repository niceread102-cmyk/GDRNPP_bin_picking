"""Rotation / camera / crop geometry shared by training and inference (torch, batched)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ------------------------------------------------------------------ rotations
def rot6d_to_matrix(x: torch.Tensor) -> torch.Tensor:
    """(B,6) -> (B,3,3) via Gram-Schmidt (Zhou et al. 2019); the 6 values are the first two columns."""
    a1, a2 = x[:, 0:3], x[:, 3:6]
    b1 = F.normalize(a1, dim=1)
    b2 = F.normalize(a2 - (b1 * a2).sum(1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=-1)


def ray_rotation(t: torch.Tensor) -> torch.Tensor:
    """(B,3) -> (B,3,3) rotation taking the camera z-axis onto the ray through t."""
    d = F.normalize(t, dim=-1)
    z = torch.zeros_like(d)
    z[:, 2] = 1.0
    v = torch.cross(z, d, dim=-1)
    c = d[:, 2]
    vx = torch.zeros(t.shape[0], 3, 3, dtype=t.dtype, device=t.device)
    vx[:, 0, 1], vx[:, 0, 2] = -v[:, 2], v[:, 1]
    vx[:, 1, 0], vx[:, 1, 2] = v[:, 2], -v[:, 0]
    vx[:, 2, 0], vx[:, 2, 1] = -v[:, 1], v[:, 0]
    eye = torch.eye(3, dtype=t.dtype, device=t.device).expand_as(vx)
    return eye + vx + vx @ vx / (1.0 + c).view(-1, 1, 1)  # objects are in front of the camera: c > 0


def ego_to_allo(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return ray_rotation(t).transpose(1, 2) @ R


def allo_to_ego(R_allo: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return ray_rotation(t) @ R_allo


def rotation_angle_deg(Ra: torch.Tensor, Rb: torch.Tensor) -> torch.Tensor:
    tr = (Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1)
    return torch.rad2deg(torch.acos(((tr - 1) / 2).clamp(-1, 1)))


def canonicalize_symmetric_pose(R: torch.Tensor, t: torch.Tensor, sym_R: torch.Tensor, sym_t: torch.Tensor):
    """Pick, per pose, the symmetry-equivalent pose whose ALLOCENTRIC rotation is closest to identity.

    Makes regression targets (xyz maps, rotations) single-valued for symmetric parts.
    R (B,3,3), t (B,3), sym_R (S,3,3), sym_t (S,3) -> R', t', index (B,)
    """
    R_allo = ego_to_allo(R, t)
    cand = R_allo[:, None] @ sym_R[None]                       # (B,S,3,3)
    k = cand.diagonal(dim1=-2, dim2=-1).sum(-1).argmax(1)     # max trace = min angle to I
    Rs, ts = sym_R[k], sym_t[k]
    return R @ Rs, (R @ ts[..., None])[..., 0] + t, k


# ------------------------------------------------------------------ crops
def crop_pixel_grid(centers: torch.Tensor, sizes: torch.Tensor, res: int) -> torch.Tensor:
    """Pixel coordinates (B,res,res,2) of a res x res grid covering square crops.

    Image convention: pixel (i, j) has its centre at u=i, v=j (same as K / BOP).
    """
    lin = (torch.arange(res, dtype=centers.dtype, device=centers.device) + 0.5) / res - 0.5
    u = centers[:, 0:1] + lin[None] * sizes[:, None]            # (B,res)
    v = centers[:, 1:2] + lin[None] * sizes[:, None]
    B = centers.shape[0]
    return torch.stack([u[:, None, :].expand(B, res, res), v[:, :, None].expand(B, res, res)], dim=-1)


def sample_crops(image: torch.Tensor, uv: torch.Tensor, mode: str) -> torch.Tensor:
    """Sample one (C,H,W) image at B pixel grids uv (B,r,r,2) -> (B,C,r,r) without copying the image B times."""
    C, H, W = image.shape
    B, r = uv.shape[:2]
    grid = torch.stack([(2 * uv[..., 0] + 1) / W - 1, (2 * uv[..., 1] + 1) / H - 1], dim=-1)
    out = F.grid_sample(image[None], grid.reshape(1, B * r, r, 2), mode=mode, padding_mode="zeros",
                        align_corners=False)                    # (1,C,B*r,r)
    return out.view(C, B, r, r).transpose(0, 1)


def sample_crops_batched(images: torch.Tensor, uv: torch.Tensor, mode: str) -> torch.Tensor:
    """One image per crop: images (B,C,H,W), uv (B,r,r,2) -> (B,C,r,r)."""
    H, W = images.shape[-2:]
    grid = torch.stack([(2 * uv[..., 0] + 1) / W - 1, (2 * uv[..., 1] + 1) / H - 1], dim=-1)
    return F.grid_sample(images, grid, mode=mode, padding_mode="zeros", align_corners=False)


def pixel_rays(uv: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """uv (B,r,r,2), K (B,3,3) -> normalized ray coords (B,2,r,r): ((u-cx)/fx, (v-cy)/fy)."""
    x = (uv[..., 0] - K[:, 0, 2, None, None]) / K[:, 0, 0, None, None]
    y = (uv[..., 1] - K[:, 1, 2, None, None]) / K[:, 1, 1, None, None]
    return torch.stack([x, y], dim=1)


def axis_angle_to_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues: unit axis (B,3), angle rad (B,) -> (B,3,3)."""
    x, y, z = axis.unbind(-1)
    zero = torch.zeros_like(x)
    k = torch.stack([zero, -z, y, z, zero, -x, -y, x, zero], -1).view(-1, 3, 3)
    s, c = angle.sin()[:, None, None], angle.cos()[:, None, None]
    return torch.eye(3, dtype=axis.dtype).expand_as(k) + s * k + (1 - c) * (k @ k)


def random_camera_rotation(center: torch.Tensor, K: torch.Tensor, roll_deg: float, tilt_deg: float,
                           gen: np.random.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random virtual camera rotation Q (B,3,3) about the viewing ray of each crop centre (roll),
    followed by a small tilt about a random perpendicular axis. Returns Q and the crop centres in the
    virtual image. A pure camera rotation is an exact homography, so labels stay exact: p_virtual = Q p."""
    B = len(center)
    d = F.normalize((torch.linalg.inv(K) @ torch.cat([center, torch.ones(B, 1)], 1)[..., None])[..., 0], dim=1)
    roll = torch.from_numpy(gen.uniform(-roll_deg, roll_deg, B)).float().deg2rad()
    Q = axis_angle_to_matrix(d, roll)
    if tilt_deg > 0:
        perp = F.normalize(torch.cross(d, torch.from_numpy(gen.normal(size=(B, 3))).float(), dim=1), dim=1)
        tilt = torch.from_numpy(gen.uniform(0, tilt_deg, B)).float().deg2rad()
        Q = axis_angle_to_matrix(perp, tilt) @ Q
    p = (K @ (Q @ d[..., None]))[..., 0]
    return Q, p[:, :2] / p[:, 2:3]


def warp_to_source(uv_virtual: torch.Tensor, K: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
    """Pixels of the rotated virtual camera (B,r,r,2) -> pixels of the real image: u = K Q^T K^-1 u'."""
    H = K @ Q.transpose(1, 2) @ torch.linalg.inv(K)                         # (B,3,3)
    uv1 = torch.cat([uv_virtual, torch.ones_like(uv_virtual[..., :1])], -1)
    p = torch.einsum("bij,bhwj->bhwi", H, uv1)
    return p[..., :2] / p[..., 2:3]


def bbox_to_crop(bbox_xywh: torch.Tensor, pad_scale: float, min_px: float) -> tuple[torch.Tensor, torch.Tensor]:
    center = bbox_xywh[:, :2] + bbox_xywh[:, 2:] / 2 - 0.5     # pixel-centre convention
    size = (bbox_xywh[:, 2:].max(1).values * pad_scale).clamp(min=min_px)
    return center, size


# ------------------------------------------------------------------ SITE translation (GDRN)
def encode_site(t: torch.Tensor, K: torch.Tensor, center: torch.Tensor, size: torch.Tensor, diameter: float):
    """t (B,3) mm -> (B,3): 2D centre offset in crop units, and scale-invariant depth t_z*s/(f*d)."""
    p = (K @ t[..., None])[..., 0]
    uv = p[:, :2] / p[:, 2:3]
    f = (K[:, 0, 0] + K[:, 1, 1]) / 2
    return torch.cat([(uv - center) / size[:, None], (t[:, 2] * size / (f * diameter))[:, None]], dim=1)


def decode_site(site: torch.Tensor, K: torch.Tensor, center: torch.Tensor, size: torch.Tensor, diameter: float):
    f = (K[:, 0, 0] + K[:, 1, 1]) / 2
    tz = site[:, 2] * f * diameter / size
    uv = center + site[:, :2] * size[:, None]
    tx = (uv[:, 0] - K[:, 0, 2]) * tz / K[:, 0, 0]
    ty = (uv[:, 1] - K[:, 1, 2]) * tz / K[:, 1, 1]
    return torch.stack([tx, ty, tz], dim=1)


# ------------------------------------------------------------------ mesh sampling
def sample_surface(verts: np.ndarray, faces: np.ndarray, n: int, seed: int = 0, return_normals: bool = False):
    """Area-weighted uniform samples on a triangle mesh -> (n,3) [, unit face normals (n,3)]."""
    rng = np.random.default_rng(seed)
    tri = verts[faces]                                              # (F,3,3)
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = np.linalg.norm(cross, axis=1)
    idx = rng.choice(len(faces), n, p=area / area.sum())
    r1, r2 = rng.random((2, n, 1))
    s = np.sqrt(r1)
    pts = ((1 - s) * tri[idx, 0] + s * (1 - r2) * tri[idx, 1] + s * r2 * tri[idx, 2]).astype(np.float32)
    if not return_normals:
        return pts
    return pts, (cross[idx] / np.maximum(area[idx, None], 1e-12)).astype(np.float32)


def farthest_point_sampling(points: np.ndarray, k: int) -> np.ndarray:
    chosen = [int(np.argmax(np.linalg.norm(points - points.mean(0), axis=1)))]
    dist = np.linalg.norm(points - points[chosen[0]], axis=1)
    for _ in range(k - 1):
        chosen.append(int(np.argmax(dist)))
        dist = np.minimum(dist, np.linalg.norm(points - points[chosen[-1]], axis=1))
    return points[chosen]
