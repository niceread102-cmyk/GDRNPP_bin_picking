"""Stage 1: instance segmentation of parts (torchvision Mask R-CNN, 1 foreground class)."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.models.detection import maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import masks_to_boxes
from torchvision.transforms import v2 as T

from data_conversion.bin_dataset import BinPoseDataset

ARCHS = {"maskrcnn_resnet50_fpn_v2": maskrcnn_resnet50_fpn_v2, "maskrcnn_resnet50_fpn": maskrcnn_resnet50_fpn}


def build_detector(dc) -> torch.nn.Module:
    model = ARCHS[dc.arch](weights="DEFAULT" if dc.pretrained else None)
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    model.roi_heads.detections_per_img = dc.max_detections
    model.roi_heads.fg_bg_sampler.batch_size_per_image = dc.roi_batch_size_per_image
    model.rpn._pre_nms_top_n = {"training": dc.rpn_pre_nms_top_n.train, "testing": dc.rpn_pre_nms_top_n.test}
    model.rpn._post_nms_top_n = {"training": dc.rpn_post_nms_top_n.train, "testing": dc.rpn_post_nms_top_n.test}
    return model


def set_native_scale(model: torch.nn.Module, h: int, w: int) -> None:
    """Make GeneralizedRCNNTransform keep images at native resolution (no resize)."""
    model.transform.min_size = (min(h, w),)
    model.transform.max_size = max(h, w)


class DetectionDataset(Dataset):
    """Train: random native-resolution square crops + flips + 90-degree rotations + photometric jitter.
    Val: full frames."""

    def __init__(self, frames: BinPoseDataset, dc, train: bool):
        self.frames, self.dc, self.train = frames, dc, train
        a = dc.aug
        self.jitter = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=a.brightness, contrast=a.contrast, saturation=a.saturation,
                                         hue=a.hue)], p=a.color_prob),
            T.RandomGrayscale(p=a.grayscale_prob),
            T.RandomApply([T.GaussianBlur(5, sigma=(0.3, a.blur_sigma))], p=a.blur_prob),
        ])
        self._gen = None

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, index: int):
        meta = self.frames.frame_meta(index)
        rgb, _, inst = self.frames.load_images(index, meta["depth_scale"])
        if self.train:
            if self._gen is None:
                self._gen = np.random.default_rng(torch.initial_seed() % 2**32)
            g, c = self._gen, self.dc.train_crop
            y0 = int(g.integers(0, rgb.shape[0] - c + 1))
            x0 = int(g.integers(0, rgb.shape[1] - c + 1))
            rgb, inst = rgb[y0:y0 + c, x0:x0 + c], inst[y0:y0 + c, x0:x0 + c]
            if g.random() < 0.5:
                rgb, inst = rgb[:, ::-1], inst[:, ::-1]
            if g.random() < 0.5:
                rgb, inst = rgb[::-1], inst[::-1]
            if self.dc.aug.rot90:                         # square crop: lossless
                k = int(g.integers(0, 4))
                rgb, inst = np.rot90(rgb, k), np.rot90(inst, k)
        img = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
        if self.train:
            img = self.jitter(img).contiguous()      # RandomGrayscale returns an expanded view
        inst_t = torch.from_numpy(np.ascontiguousarray(inst))
        ids, counts = torch.unique(inst_t, return_counts=True)
        ids = ids[(ids > 0) & (counts >= self.dc.min_visib_px)]
        masks = (inst_t[None] == ids[:, None, None]).to(torch.uint8)
        boxes = masks_to_boxes(masks) if len(ids) else torch.zeros(0, 4)
        boxes[:, 2:] += 1                                   # masks_to_boxes gives inclusive max
        target = {"boxes": boxes, "labels": torch.ones(len(ids), dtype=torch.int64), "masks": masks,
                  "inst_id": ids.long()}
        if not self.train:
            target["inst_full"] = inst_t
        return img, target


def det_collate(batch):
    return tuple(zip(*batch))
