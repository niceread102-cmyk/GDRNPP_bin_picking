"""GDRN-style pose network (Wang et al. CVPR'21 / GDRNPP) for RGB-D crops.

backbone (timm) -> geometry head (visible mask, full mask, xyz, surface regions @ output_res)
                -> Patch-PnP head (xyz + regions + mask + pixel rays) -> allocentric rot6d + SITE translation
"""
from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from bin6d.geometry import IMAGENET_MEAN, IMAGENET_STD, allo_to_ego, decode_site, rot6d_to_matrix, rotation_angle_deg


def conv_gn(cin: int, cout: int, stride: int = 1, act=nn.GELU) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(cin, cout, 3, stride, 1, bias=False), nn.GroupNorm(32, cout), act())


class GDRNet(nn.Module):
    def __init__(self, cfg, in_channels: int):
        super().__init__()
        self.num_regions = cfg.num_regions
        self.backbone = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, features_only=True,
                                          in_chans=in_channels)
        c = self.backbone.feature_info.channels()[-1]
        reduction = self.backbone.feature_info.reduction()[-1]
        n_up = math.log2(cfg.output_res * reduction / cfg.input_res)
        assert n_up == int(n_up) and n_up >= 0, "output_res * reduction / input_res must be a power of 2"

        layers = [conv_gn(c, 256)]
        for _ in range(int(n_up)):
            layers += [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False), conv_gn(256, 256),
                       conv_gn(256, 256)]
        self.geo_head = nn.Sequential(*layers)
        n_reg = self.num_regions + 1 if self.num_regions else 0
        self.geo_out = nn.Conv2d(256, 1 + 1 + 3 + n_reg, 1)

        pnp_in = 3 + 2 + 1 + n_reg
        self.pnp = nn.Sequential(conv_gn(pnp_in, 128, 2, nn.ReLU), conv_gn(128, 128, 2, nn.ReLU),
                                 conv_gn(128, 128, 2, nn.ReLU))
        flat = 128 * (cfg.output_res // 8) ** 2
        self.pnp_fc = nn.Sequential(nn.Flatten(), nn.Linear(flat, 1024), nn.LeakyReLU(0.1),
                                    nn.Linear(1024, 256), nn.LeakyReLU(0.1))
        self.rot_out = nn.Linear(256, 6)
        self.trans_out = nn.Linear(256, 3)
        # start near identity rotation / centred object / typical scale
        nn.init.zeros_(self.rot_out.weight)
        with torch.no_grad():
            self.rot_out.bias.copy_(torch.tensor([1.0, 0, 0, 0, 1.0, 0]))
        nn.init.zeros_(self.trans_out.weight)
        with torch.no_grad():
            self.trans_out.bias.copy_(torch.tensor([0.0, 0.0, 1.0]))

    def forward(self, x: torch.Tensor, rays: torch.Tensor) -> dict:
        feat = self.backbone(x)[-1]
        g = self.geo_out(self.geo_head(feat))
        out = {
            "mask_visib": g[:, 0:1],
            "mask_full": g[:, 1:2],
            "xyz": torch.sigmoid(g[:, 2:5]),
        }
        pnp_in = [out["xyz"], rays, torch.sigmoid(out["mask_visib"])]
        if self.num_regions:
            out["region"] = g[:, 5:]
            pnp_in.append(torch.softmax(out["region"], dim=1))
        h = self.pnp_fc(self.pnp(torch.cat(pnp_in, dim=1)))
        out["rot6d"] = self.rot_out(h)
        out["site"] = self.trans_out(h)
        return out


def load_gdrnpp_backbone(net: GDRNet, pm) -> None:
    """Initialise net.backbone from an official GDRNPP checkpoint (timm ConvNeXt, `backbone.*` keys).

    The first conv is remapped so that, on the RGB channels, our input convention (RGB, ImageNet-normalized)
    produces exactly the activations their convention (pm.input_format, (x - pixel_mean) / pixel_std) did.
    Extra channels (depth, mask) get zero weights, so training starts from their RGB behaviour.
    """
    path = Path(pm.path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading GDRNPP checkpoint -> {path}")
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(pm.url, tmp)
        tmp.replace(path)
    sd = torch.load(path, map_location="cpu", weights_only=False)
    sd = sd.get("model", sd)
    bb = {k[len("backbone."):]: v for k, v in sd.items() if k.startswith("backbone.")}
    if not bb:
        raise ValueError(f"no backbone.* weights in {path}")

    stem_key = "stem_0.weight"
    own = net.backbone.state_dict()
    if stem_key not in own or own[stem_key].shape[0] != bb[stem_key].shape[0]:
        raise ValueError("backbone architecture does not match the GDRNPP checkpoint (expected convnext_base)")
    W, b = bb[stem_key].float(), bb["stem_0.bias"].float()            # (C,3,k,k) in their channel order
    order = {"RGB": [0, 1, 2], "BGR": [2, 1, 0]}[pm.input_format]    # their channel j reads our RGB channel order[j]
    their_of_rgb = [order.index(c) for c in range(3)]
    mean_ours, std_ours = IMAGENET_MEAN.flatten(), IMAGENET_STD.flatten()
    W_new = torch.zeros_like(own[stem_key])
    b_new = b.clone()
    for c in range(3):
        j = their_of_rgb[c]
        W_new[:, c] = W[:, j] * std_ours[c] / pm.pixel_std[j]
        b_new += W[:, j].sum(dim=(1, 2)) * (mean_ours[c] - pm.pixel_mean[j]) / pm.pixel_std[j]
    bb[stem_key], bb["stem_0.bias"] = W_new, b_new

    missing, unexpected = net.backbone.load_state_dict(bb, strict=False)
    print(f"loaded GDRNPP backbone from {path} ({len(bb) - len(unexpected)} tensors; "
          f"missing {len(missing)}, unexpected {len(unexpected)})")
    if missing:
        print("   missing:", missing[:10])


def decode_pose(out: dict, K: torch.Tensor, center: torch.Tensor, size: torch.Tensor, diameter: float):
    """Network outputs -> egocentric R (B,3,3), t (B,3) mm."""
    t = decode_site(out["site"].float(), K, center, size, diameter)
    R = allo_to_ego(rot6d_to_matrix(out["rot6d"].float()), t)
    return R, t


def pose_losses(out: dict, tgt: dict, model_pts: torch.Tensor, sym_R: torch.Tensor, w) -> dict:
    """GDRN loss set. model_pts (N,3) normalized by diameter; sym_R (S,3,3)."""
    mv = tgt["mask_visib"]
    losses = {
        "mask_visib": F.binary_cross_entropy_with_logits(out["mask_visib"].float(), mv),
        "mask_full": F.binary_cross_entropy_with_logits(out["mask_full"].float(), tgt["mask_full"]),
        "xyz": ((out["xyz"].float() - tgt["xyz"]).abs() * mv).sum() / (3 * mv.sum()).clamp(min=1),
    }
    if "region" in out:
        ce = F.cross_entropy(out["region"].float(), tgt["region"], reduction="none")
        losses["region"] = (ce * mv[:, 0]).sum() / mv.sum().clamp(min=1)

    # point matching on allocentric rotation, min over discrete symmetries
    R_pred = rot6d_to_matrix(out["rot6d"].float())
    pred = model_pts[None] @ R_pred.transpose(1, 2)                                  # (B,N,3)
    R_gt_syms = tgt["R_allo"][:, None] @ sym_R[None]                                 # (B,S,3,3)
    gt = model_pts[None, None] @ R_gt_syms.transpose(-1, -2)                         # (B,S,N,3)
    losses["pm"] = (pred[:, None] - gt).abs().sum(-1).mean(-1).min(1).values.mean()

    site = out["site"].float()
    losses["center"] = (site[:, :2] - tgt["site"][:, :2]).abs().sum(1).mean()
    losses["z"] = (site[:, 2] - tgt["site"][:, 2]).abs().mean()

    losses["total"] = sum(w[k] * v for k, v in losses.items() if k in w)
    return losses


@torch.no_grad()
def pose_errors(R: torch.Tensor, t: torch.Tensor, R_gt: torch.Tensor, t_gt: torch.Tensor,
                model_pts_mm: torch.Tensor, sym_R: torch.Tensor, sym_t: torch.Tensor) -> dict:
    """Symmetry-aware errors: ADD over discrete symmetries (min), rotation deg, translation mm."""
    pred = model_pts_mm[None] @ R.transpose(1, 2) + t[:, None]                        # (B,N,3)
    R_s = R_gt[:, None] @ sym_R[None]                                                 # (B,S,3,3)
    t_s = (R_gt[:, None] @ sym_t[None, :, :, None])[..., 0] + t_gt[:, None]           # (B,S,3)
    gt = model_pts_mm[None, None] @ R_s.transpose(-1, -2) + t_s[:, :, None]           # (B,S,N,3)
    add = (pred[:, None] - gt).norm(dim=-1).mean(-1)                                  # (B,S)
    add_min, k = add.min(1)
    ar = torch.arange(len(R), device=R.device)
    rot_err = torch.stack([rotation_angle_deg(R, R_s[:, s]) for s in range(R_s.shape[1])], 1).min(1).values
    return {"add_s": add_min, "rot_deg": rot_err, "trans_mm": (t - t_s[ar, k]).norm(dim=-1),
            "z_mm": (t[:, 2] - t_gt[:, 2]).abs()}

