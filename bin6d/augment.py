"""Label-preserving crop augmentations for the pose network (training only).

Geometric augmentation (virtual camera rotation) lives in geometry.random_camera_rotation because it
changes the labels too; everything here leaves the pose untouched.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

_GRAY = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)


def _u(gen: np.random.Generator, lo: float, hi: float, n: int) -> torch.Tensor:
    return torch.from_numpy(gen.uniform(lo, hi, n)).float().view(n, 1, 1, 1)


def _gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    r = max(1, int(round(2 * sigma)))
    k = torch.exp(-(torch.arange(-r, r + 1, dtype=x.dtype) ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1).repeat(x.shape[1], 1, 1, 1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.transpose(2, 3), groups=x.shape[1])


def photometric(img: torch.Tensor, a, gen: np.random.Generator) -> torch.Tensor:
    """img (B,3,r,r) in 0..1 -> augmented copy (brightness, contrast, saturation, gray, blur, noise)."""
    B = len(img)
    on = torch.from_numpy(gen.random(B) < a.color_prob).float().view(B, 1, 1, 1)
    gray = (img * _GRAY).sum(1, keepdim=True)
    sat = 1 + on * (_u(gen, 1 - a.saturation, 1 + a.saturation, B) - 1)
    img = gray + (img - gray) * sat
    to_gray = torch.from_numpy(gen.random(B) < a.grayscale_prob).view(B, 1, 1, 1)
    img = torch.where(to_gray, (img * _GRAY).sum(1, keepdim=True).expand_as(img), img)
    bright = 1 + on * (_u(gen, 1 - a.brightness, 1 + a.brightness, B) - 1)
    contrast = 1 + on * (_u(gen, 1 - a.contrast, 1 + a.contrast, B) - 1)
    mean = img.mean(dim=(1, 2, 3), keepdim=True)
    img = ((img - mean) * contrast + mean) * bright
    for i in np.flatnonzero(gen.random(B) < a.blur_prob):
        img[i:i + 1] = _gaussian_blur(img[i:i + 1], float(gen.uniform(0.3, a.blur_sigma)))
    img = img + torch.randn_like(img) * _u(gen, 0, a.noise_std, B)
    return img.clamp(0, 1)


def depth_dropout_and_noise(pts: torch.Tensor, val: torch.Tensor, a, gen: np.random.Generator):
    """pts (B,3,r,r) camera mm, val (B,1,r,r): add noise along the viewing ray and punch random holes,
    mimicking the specular dropout of the structured-light sensor on metal parts."""
    B, _, r, _ = pts.shape
    sigma = _u(gen, 0, a.depth_noise_mm, B)
    ray = F.normalize(pts, dim=1)
    pts = pts + ray * torch.randn_like(pts[:, :1]) * sigma
    yy, xx = torch.meshgrid(torch.arange(r, dtype=torch.float32), torch.arange(r, dtype=torch.float32),
                            indexing="ij")
    keep = torch.ones(B, 1, r, r)
    for i in np.flatnonzero(gen.random(B) < a.depth_hole_prob):
        for _ in range(int(gen.integers(1, a.depth_max_holes + 1))):
            cx, cy = gen.uniform(0, r, 2)
            rad = gen.uniform(*a.depth_hole_radius) * r
            keep[i, 0][(xx - cx) ** 2 + (yy - cy) ** 2 < rad ** 2] = 0
    val = val * keep
    return pts * val, val
