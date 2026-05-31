import os
import math
import time
import random
import argparse
import datetime
import json
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

from scipy.optimize import linear_sum_assignment

import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import MambaPulse
from vim_pretrained_loader import load_vim_pretrained


def box_xyxy_to_cxcywh(boxes):
    x1, y1, x2, y2 = boxes.unbind(-1)
    return torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)


def box_cxcywh_to_xyxy(boxes):
    cx, cy, w, h = boxes.unbind(-1)
    return torch.stack([cx-w/2, cy-h/2, cx+w/2, cy+h/2], dim=-1)


def box_iou(boxes1, boxes2):
    area1 = (boxes1[:,2]-boxes1[:,0]) * (boxes1[:,3]-boxes1[:,1])
    area2 = (boxes2[:,2]-boxes2[:,0]) * (boxes2[:,3]-boxes2[:,1])
    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:,:,0] * wh[:,:,1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / (union + 1e-7)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    iou, union = box_iou(boxes1, boxes2)
    lt = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    enclosing = wh[:,:,0] * wh[:,:,1]
    giou = iou - (enclosing - union) / (enclosing + 1e-7)
    return giou


class HungarianMatcher(nn.Module):
    def __init__(self, cost_class=2.0, cost_bbox=2.0, cost_giou=5.0,
                 num_classes=10, focal_alpha=0.25, focal_gamma=2.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.num_classes = num_classes
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

    @torch.no_grad()
    def forward(self, outputs, targets):
        B, Nq = outputs['pred_logits'].shape[:2]
        out_prob = outputs['pred_logits'].sigmoid()
        out_bbox = outputs['pred_boxes']

        alpha = self.focal_alpha
        gamma = self.focal_gamma

        indices = []
        for b in range(B):
            tgt_ids = targets[b]['labels']
            tgt_bbox = targets[b]['boxes']

            if len(tgt_ids) == 0:
                indices.append((torch.tensor([], dtype=torch.long),
                                torch.tensor([], dtype=torch.long)))
                continue

            prob = out_prob[b]
            neg_cost = (1 - alpha) * (prob ** gamma) * (-(1 - prob + 1e-8).log())
            pos_cost = alpha * ((1 - prob) ** gamma) * (-(prob + 1e-8).log())
            cost_cls = pos_cost[:, tgt_ids] - neg_cost[:, tgt_ids]

            cost_l1 = torch.cdist(out_bbox[b], tgt_bbox, p=1)
            cost_giou = -generalized_box_iou(
                box_cxcywh_to_xyxy(out_bbox[b]),
                box_cxcywh_to_xyxy(tgt_bbox))

            C = (self.cost_class * cost_cls +
                 self.cost_bbox * cost_l1 +
                 self.cost_giou * cost_giou).cpu()

            C[C.isnan() | C.isinf()] = 1e4

            r, c_ = linear_sum_assignment(C.numpy())
            indices.append((torch.as_tensor(r, dtype=torch.long),
                            torch.as_tensor(c_, dtype=torch.long)))
        return indices


class SetCriterion(nn.Module):
    def __init__(self, num_classes, matcher, weight_dict, losses,
                 focal_alpha=0.25, focal_gamma=1.5, class_freq=None):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma

        if class_freq is not None:
            freq = torch.tensor(class_freq, dtype=torch.float32)
            inv_sqrt = 1.0 / freq.sqrt()
            alpha_per_class = inv_sqrt * focal_alpha / inv_sqrt.mean()
            alpha_per_class = alpha_per_class.clamp(min=0.05, max=0.8)
            self.register_buffer('alpha_per_class', alpha_per_class)
            names = ['pedestrian', 'people', 'bicycle', 'car', 'van', 'truck',
                     'tricycle', 'awning-tri', 'bus', 'motor']
            for n, a in zip(names, alpha_per_class.tolist()):
                print(f"    {n:12s}: alpha = {a:.3f}")
        else:
            self.alpha_per_class = None

    def loss_labels(self, outputs, targets, indices, num_boxes):
        pred_logits = outputs['pred_logits']
        B, Nq, C = pred_logits.shape

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([
            t['labels'][j] for t, (_, j) in zip(targets, indices)
        ])

        target_classes = torch.full(
            (B, Nq), self.num_classes,
            dtype=torch.int64, device=pred_logits.device
        )
        target_classes[idx] = target_classes_o

        target_onehot = torch.zeros(
            (B, Nq, self.num_classes + 1),
            dtype=pred_logits.dtype, device=pred_logits.device
        )
        target_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)
        target_onehot = target_onehot[:, :, :-1]

        gamma = self.focal_gamma
        prob = pred_logits.sigmoid()
        ce_loss = F.binary_cross_entropy_with_logits(
            pred_logits, target_onehot, reduction='none'
        )
        p_t = prob * target_onehot + (1 - prob) * (1 - target_onehot)
        focal = ce_loss * ((1 - p_t) ** gamma)

        if self.alpha_per_class is not None:
            pos_alpha = self.alpha_per_class.view(1, 1, -1)
            neg_alpha = 1.0 - self.focal_alpha
            alpha_t = pos_alpha * target_onehot + neg_alpha * (1 - target_onehot)
        else:
            alpha = self.focal_alpha
            alpha_t = alpha * target_onehot + (1 - alpha) * (1 - target_onehot)

        focal = alpha_t * focal

        loss_ce = focal.mean(1).sum() / max(num_boxes, 1)
        return {'loss_ce': loss_ce}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([
            t['boxes'][j] for t, (_, j) in zip(targets, indices)
        ], dim=0)

        if len(src_boxes) == 0:
            dev = outputs['pred_boxes'].device
            return {'loss_bbox': torch.tensor(0., device=dev),
                    'loss_giou': torch.tensor(0., device=dev)}

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox.sum() / max(num_boxes, 1)

        src_xyxy = box_cxcywh_to_xyxy(src_boxes)
        tgt_xyxy = box_cxcywh_to_xyxy(target_boxes)
        giou = torch.diag(generalized_box_iou(src_xyxy, tgt_xyxy))
        loss_giou = (1 - giou).sum() / max(num_boxes, 1)

        return {'loss_bbox': loss_bbox, 'loss_giou': loss_giou}

    def _get_src_permutation_idx(self, indices):
        batch_idx = torch.cat([torch.full_like(s, i) for i, (s, _) in enumerate(indices)])
        src_idx = torch.cat([s for (s, _) in indices])
        return batch_idx, src_idx

    def _compute_losses_single(self, outputs, targets, num_boxes):
        indices = self.matcher(outputs, targets)
        losses = {}
        for loss_type in self.losses:
            if loss_type == 'labels':
                losses.update(self.loss_labels(outputs, targets, indices, num_boxes))
            elif loss_type == 'boxes':
                losses.update(self.loss_boxes(outputs, targets, indices, num_boxes))
        return losses

    def _group_losses(self, outputs, targets, num_boxes, num_groups):
        if num_groups == 1:
            return self._compute_losses_single(outputs, targets, num_boxes)

        pred_logits = outputs['pred_logits']
        pred_boxes = outputs['pred_boxes']
        B, total_Nq, C = pred_logits.shape
        Nq = total_Nq // num_groups

        pred_logits_g = pred_logits.reshape(B, num_groups, Nq, C)
        pred_boxes_g = pred_boxes.reshape(B, num_groups, Nq, 4)

        accumulated = {}
        for g in range(num_groups):
            group_out = {
                'pred_logits': pred_logits_g[:, g],
                'pred_boxes': pred_boxes_g[:, g],
            }
            group_losses = self._compute_losses_single(group_out, targets, num_boxes)
            for k, v in group_losses.items():
                if k not in accumulated:
                    accumulated[k] = v
                else:
                    accumulated[k] = accumulated[k] + v

        for k in accumulated:
            accumulated[k] = accumulated[k] / num_groups

        return accumulated

    def forward(self, outputs, targets):
        num_groups = outputs.get('num_groups', 1)

        num_boxes = sum(len(t['labels']) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float,
            device=outputs['pred_logits'].device
        ).clamp(min=1).item()

        losses = self._group_losses(outputs, targets, num_boxes, num_groups)

        if 'aux_outputs' in outputs:
            for i, aux_out in enumerate(outputs['aux_outputs']):
                aux_losses = self._group_losses(aux_out, targets, num_boxes, num_groups)
                losses.update({f'{k}_aux{i}': v for k, v in aux_losses.items()})

        return losses


class VisDroneDataset(Dataset):
    CLASSES = ['pedestrian', 'people', 'bicycle', 'car', 'van',
               'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor']
    IGNORED = [0, 11]

    def __init__(self, root_dir, split='train', img_size=640,
                 transform=None, use_mosaic=True, mosaic_prob=0.5):
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.transform = transform
        self.use_mosaic = use_mosaic and (split == 'train')
        self.mosaic_prob = mosaic_prob

        base = os.path.join(root_dir, f'VisDrone2019-DET-{split}')
        for img_name in ['image', 'images']:
            p = os.path.join(base, img_name)
            if os.path.exists(p):
                self.img_dir = p
                break
        else:
            raise FileNotFoundError(f"image dir not found: {base}/image or {base}/images")

        for anno_name in ['annotation', 'annotations']:
            p = os.path.join(base, anno_name)
            if os.path.exists(p):
                self.anno_dir = p
                break
        else:
            raise FileNotFoundError(f"annotation dir not found: {base}/annotation or {base}/annotations")

        self.img_files = sorted([f for f in os.listdir(self.img_dir)
                                  if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        assert len(self.img_files) > 0, f"empty dir: {self.img_dir}"

        print(f"[{split}] loading annotations for {len(self.img_files)} images...")
        self.annos = []
        for img_f in self.img_files:
            anno_f = os.path.splitext(img_f)[0] + '.txt'
            self.annos.append(self._parse(os.path.join(self.anno_dir, anno_f)))
        print(f"[{split}] done")

    def _parse(self, path):
        boxes, labels = [], []
        if not os.path.exists(path):
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        with open(path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 8:
                    continue
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                cat = int(parts[5])
                if cat in self.IGNORED or w <= 0 or h <= 0:
                    continue
                boxes.append([x, y, x + w, y + h])
                labels.append(cat - 1)
        if not boxes:
            return np.zeros((0, 4), np.float32), np.zeros((0,), np.int64)
        return np.array(boxes, np.float32), np.array(labels, np.int64)

    def _load_img(self, idx):
        img = cv2.imread(os.path.join(self.img_dir, self.img_files[idx]))
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _resize(self, img, boxes):
        oh, ow = img.shape[:2]
        img = cv2.resize(img, (self.img_size, self.img_size))
        if len(boxes) > 0:
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= self.img_size / ow
            boxes[:, [1, 3]] *= self.img_size / oh
        return img, boxes

    def _mosaic(self, idx):
        ids = [idx] + [random.randint(0, len(self) - 1) for _ in range(3)]
        s2 = self.img_size * 2
        out = np.zeros((s2, s2, 3), np.uint8)
        cx = s2 // 2 + random.randint(-s2 // 8, s2 // 8)
        cy = s2 // 2 + random.randint(-s2 // 8, s2 // 8)
        all_b, all_l = [], []

        for i, id_ in enumerate(ids):
            img = self._load_img(id_)
            bx, lb = self.annos[id_]
            bx, lb = bx.copy(), lb.copy()
            h, w = img.shape[:2]
            if i == 0:
                x1a, y1a = max(cx-w, 0), max(cy-h, 0)
                x2a, y2a = cx, cy
                x1b, y1b = w-(x2a-x1a), h-(y2a-y1a)
                x2b, y2b = w, h
            elif i == 1:
                x1a, y1a = cx, max(cy-h, 0)
                x2a, y2a = min(cx+w, s2), cy
                x1b, y1b = 0, h-(y2a-y1a)
                x2b, y2b = x2a-x1a, h
            elif i == 2:
                x1a, y1a = max(cx-w, 0), cy
                x2a, y2a = cx, min(cy+h, s2)
                x1b, y1b = w-(x2a-x1a), 0
                x2b, y2b = w, y2a-y1a
            else:
                x1a, y1a = cx, cy
                x2a, y2a = min(cx+w, s2), min(cy+h, s2)
                x1b, y1b = 0, 0
                x2b, y2b = x2a-x1a, y2a-y1a

            out[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            if len(bx) > 0:
                bx[:, [0, 2]] = np.clip(bx[:, [0, 2]], x1b, x2b)
                bx[:, [1, 3]] = np.clip(bx[:, [1, 3]], y1b, y2b)
                keep = ((bx[:, 2]-bx[:, 0]) > 2) & ((bx[:, 3]-bx[:, 1]) > 2)
                bx, lb = bx[keep], lb[keep]
                if len(bx) > 0:
                    bx[:, [0, 2]] += x1a - x1b
                    bx[:, [1, 3]] += y1a - y1b
                    all_b.append(bx)
                    all_l.append(lb)

        out = cv2.resize(out, (self.img_size, self.img_size))
        sc = self.img_size / s2
        if all_b:
            all_b = np.concatenate(all_b) * sc
            all_l = np.concatenate(all_l)
        else:
            all_b = np.zeros((0, 4), np.float32)
            all_l = np.zeros((0,), np.int64)
        return out, all_b, all_l

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        if self.use_mosaic and random.random() < self.mosaic_prob:
            img, boxes, labels = self._mosaic(idx)
        else:
            img = self._load_img(idx)
            boxes, labels = self.annos[idx]
            boxes, labels = boxes.copy(), labels.copy()
            img, boxes = self._resize(img, boxes)

        if self.transform:
            t = self.transform(image=img,
                               bboxes=boxes.tolist() if len(boxes) > 0 else [],
                               class_labels=labels.tolist() if len(labels) > 0 else [])
            img = t['image']
            if t['bboxes']:
                boxes = np.array(t['bboxes'], np.float32)
                labels = np.array(t['class_labels'], np.int64)
            else:
                boxes = np.zeros((0, 4), np.float32)
                labels = np.zeros((0,), np.int64)
        else:
            img = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)

        if len(boxes) > 0:
            boxes[:, [0, 2]] /= self.img_size
            boxes[:, [1, 3]] /= self.img_size
            x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
            boxes = torch.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], dim=-1)
            valid = (boxes[:, 2] > 1e-4) & (boxes[:, 3] > 1e-4)
            boxes, labels = boxes[valid], labels[valid]

        return {'image': img, 'boxes': boxes, 'labels': labels, 'image_id': idx}

    @staticmethod
    def collate_fn(batch):
        return {
            'images': torch.stack([x['image'] for x in batch]),
            'boxes': [x['boxes'] for x in batch],
            'labels': [x['labels'] for x in batch],
            'image_ids': [x['image_id'] for x in batch],
        }


def get_train_transforms(img_size=640):
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.3),
        A.GaussNoise(p=0.15),
        A.GaussianBlur(blur_limit=(3, 5), p=0.15),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels'],
                                 min_visibility=0.2, min_area=4))


def get_val_transforms():
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


def build_dataloaders(args):
    train_ds = VisDroneDataset(
        args.data_root, 'train', args.img_size,
        transform=get_train_transforms(args.img_size),
        use_mosaic=args.use_mosaic, mosaic_prob=args.mosaic_prob,
    )
    val_ds = VisDroneDataset(
        args.data_root, 'val', args.img_size,
        transform=get_val_transforms(),
        use_mosaic=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
        collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True, drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
        collate_fn=VisDroneDataset.collate_fn,
        pin_memory=True, drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def prepare_targets(batch, device):
    return [{'labels': l.to(device), 'boxes': b.to(device)}
            for b, l in zip(batch['boxes'], batch['labels'])]


# ---------------------------------------------------------------------------
# COCO-style AP evaluation: AP@[0.50:0.95], AP50, AP75, APs/APm/APl.
# 101-point interpolation, greedy score-descending matching, per-scale areas
# (small <32^2, medium 32^2-96^2, large >96^2 px). Computed in float64 to
# stay numerically aligned with pycocotools.
# ---------------------------------------------------------------------------
_RECALL_GRID = np.linspace(0.0, 1.0, 101)


def _box_cxcywh_to_xyxy_np(b):
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w/2, cy - h/2, cx + w/2, cy + h/2], axis=-1)


def _box_iou_np(b1, b2):
    if len(b1) == 0 or len(b2) == 0:
        return np.zeros((len(b1), len(b2)), dtype=np.float64)
    a1 = (b1[:, 2]-b1[:, 0]) * (b1[:, 3]-b1[:, 1])
    a2 = (b2[:, 2]-b2[:, 0]) * (b2[:, 3]-b2[:, 1])
    lt = np.maximum(b1[:, None, :2], b2[None, :, :2])
    rb = np.minimum(b1[:, None, 2:], b2[None, :, 2:])
    wh = (rb - lt).clip(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = a1[:, None] + a2[None, :] - inter
    return inter / (union + 1e-7)


def _interp_ap_101(recall, precision):
    mpre = precision.astype(np.float64).copy()
    for i in range(len(mpre) - 2, -1, -1):
        if mpre[i] < mpre[i + 1]:
            mpre[i] = mpre[i + 1]
    idx = np.searchsorted(recall, _RECALL_GRID, side='left')
    q = np.zeros_like(_RECALL_GRID)
    valid = idx < len(mpre)
    q[valid] = mpre[idx[valid]]
    return q.mean()


@torch.no_grad()
def compute_map(model, data_loader, device, num_classes=10,
                score_thresh=0.05, img_size=640, max_dets=300, verbose=True):
    model.eval()

    iou_thr = np.linspace(0.5, 0.95, 10)
    T = len(iou_thr)
    area_ranges = [('all', 0.0, 1e18), ('small', 0.0, 32.0**2),
                   ('medium', 32.0**2, 96.0**2), ('large', 96.0**2, 1e18)]

    det_scores = {ak: [[] for _ in range(num_classes)] for ak, _, _ in area_ranges}
    det_match = {ak: [[] for _ in range(num_classes)] for ak, _, _ in area_ranges}
    det_ignore = {ak: [[] for _ in range(num_classes)] for ak, _, _ in area_ranges}
    npos = {ak: [0] * num_classes for ak, _, _ in area_ranges}

    for batch in data_loader:
        images = batch['images'].to(device)
        targets = prepare_targets(batch, device)
        outputs = model(images)
        probs = outputs['pred_logits'].sigmoid()
        pred_boxes = outputs['pred_boxes']
        B = probs.shape[0]

        for b in range(B):
            max_scores, max_cls = probs[b].max(dim=-1)
            keep = max_scores > score_thresh
            sc = max_scores[keep].detach().cpu().numpy().astype(np.float64)
            cl = max_cls[keep].detach().cpu().numpy().astype(int)
            bx = pred_boxes[b][keep].detach().cpu().numpy().astype(np.float64).reshape(-1, 4)
            gl = targets[b]['labels'].detach().cpu().numpy().astype(int)
            gb = targets[b]['boxes'].detach().cpu().numpy().astype(np.float64).reshape(-1, 4)

            if sc.size > 0:
                order = np.argsort(-sc, kind='mergesort')[:max_dets]
                sc, cl, bx = sc[order], cl[order], bx[order]
            n_pred, n_gt = len(sc), len(gl)

            gt_area = (gb[:, 2] * gb[:, 3]) * (img_size ** 2) if n_gt > 0 else np.zeros(0)
            pred_area = (bx[:, 2] * bx[:, 3]) * (img_size ** 2) if n_pred > 0 else np.zeros(0)
            iou = (_box_iou_np(_box_cxcywh_to_xyxy_np(bx), _box_cxcywh_to_xyxy_np(gb))
                   if (n_pred > 0 and n_gt > 0) else np.zeros((n_pred, n_gt)))

            for ak, lo, hi in area_ranges:
                gt_in = (gt_area >= lo) & (gt_area < hi) if n_gt > 0 else np.zeros(0, dtype=bool)
                for c in range(num_classes):
                    npos[ak][c] += int(((gl == c) & gt_in).sum())

                gt_used = np.zeros((T, n_gt), dtype=bool)
                for i in range(n_pred):
                    c = int(cl[i])
                    same = (gl == c)
                    ious_i = iou[i]
                    match_vec = np.zeros(T, dtype=bool)
                    ignore_vec = np.zeros(T, dtype=bool)
                    for t in range(T):
                        thr = iou_thr[t]
                        avail = same & ~(gt_used[t] & gt_in)
                        matched = False
                        if avail.any():
                            cand = np.where(avail, ious_i, -1.0)
                            bj = int(cand.argmax())
                            if cand[bj] >= thr:
                                gt_used[t, bj] = True
                                match_vec[t] = True
                                ignore_vec[t] = not gt_in[bj]
                                matched = True
                        if not matched:
                            in_range = True if ak == 'all' else (lo <= pred_area[i] < hi)
                            ignore_vec[t] = not in_range
                    det_scores[ak][c].append(float(sc[i]))
                    det_match[ak][c].append(match_vec)
                    det_ignore[ak][c].append(ignore_vec)

    ap_table = {ak: np.full((num_classes, T), np.nan) for ak, _, _ in area_ranges}
    for ak, _, _ in area_ranges:
        for c in range(num_classes):
            if npos[ak][c] == 0:
                continue
            if len(det_scores[ak][c]) == 0:
                ap_table[ak][c, :] = 0.0
                continue
            s = np.asarray(det_scores[ak][c])
            mt = np.stack(det_match[ak][c], 0)
            ig = np.stack(det_ignore[ak][c], 0)
            order = np.argsort(-s, kind='mergesort')
            mt, ig = mt[order], ig[order]
            for t in range(T):
                keep = ~ig[:, t]
                m = mt[keep, t]
                if m.size == 0:
                    ap_table[ak][c, t] = 0.0
                    continue
                tp = np.cumsum(m.astype(np.float64))
                fp = np.cumsum((~m).astype(np.float64))
                recall = tp / npos[ak][c]
                precision = tp / np.maximum(tp + fp, 1e-12)
                ap_table[ak][c, t] = _interp_ap_101(recall, precision)

    def mean_ap(ak, t_sel=None):
        tbl = ap_table[ak]
        vr = ~np.isnan(tbl[:, 0])
        if vr.sum() == 0:
            return 0.0
        sub = tbl[vr]
        return float(sub.mean()) if t_sel is None else float(sub[:, t_sel].mean())

    t50 = int(np.argmin(np.abs(iou_thr - 0.5)))
    t75 = int(np.argmin(np.abs(iou_thr - 0.75)))
    metrics = {
        'AP': mean_ap('all'),
        'AP50': mean_ap('all', t50),
        'AP75': mean_ap('all', t75),
        'APs': mean_ap('small'),
        'APm': mean_ap('medium'),
        'APl': mean_ap('large'),
    }

    if verbose:
        class_names = ['pedestrian', 'people', 'bicycle', 'car', 'van',
                       'truck', 'tricycle', 'awning-tri', 'bus', 'motor']
        per_class = ap_table['all']
        print(f"\n  === COCO AP @[0.50:0.95] (101-point) ===")
        for c in range(num_classes):
            if np.isnan(per_class[c, 0]):
                continue
            ap_c = float(np.nanmean(per_class[c]))
            ap50_c = float(per_class[c, t50])
            print(f"      {class_names[c]:>12s}: AP={ap_c:.4f}  AP50={ap50_c:.4f}")
        print(f"      {'-' * 44}")
        print(f"      AP   @[.50:.95] = {metrics['AP']:.4f}")
        print(f"      AP50            = {metrics['AP50']:.4f}")
        print(f"      AP75            = {metrics['AP75']:.4f}")
        print(f"      AP   (small)    = {metrics['APs']:.4f}")
        print(f"      AP   (medium)   = {metrics['APm']:.4f}")
        print(f"      AP   (large)    = {metrics['APl']:.4f}")

    return metrics


def train_one_epoch(model, criterion, loader, optimizer, scaler,
                    device, epoch, args):
    model.train()
    criterion.train()

    stats = {'loss': 0, 'ce': 0, 'bbox': 0, 'giou': 0, 'n': 0}
    t0 = time.time()
    optimizer.zero_grad()

    for i, batch in enumerate(loader):
        images = batch['images'].to(device)
        targets = prepare_targets(batch, device)
        if sum(len(t['labels']) for t in targets) == 0:
            continue

        with autocast(enabled=args.amp):
            outputs = model(images)
            ld = criterion(outputs, targets)

            w = criterion.weight_dict
            loss = sum(ld[k] * w[k] for k in ld if k in w)
            for k in ld:
                if 'aux' in k:
                    bk = k.rsplit('_aux', 1)[0]
                    if bk in w:
                        loss = loss + ld[k] * w[bk]

            loss = loss / args.grad_accum_steps

        if torch.isnan(loss) or torch.isinf(loss):
            optimizer.zero_grad()
            print(f"  NaN/Inf at batch {i+1}, skipping")
            continue

        loss = torch.clamp(loss, max=500.0)

        scaler.scale(loss).backward()

        if (i + 1) % args.grad_accum_steps == 0:
            if args.clip_grad > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        stats['loss'] += loss.item() * args.grad_accum_steps
        stats['ce'] += ld.get('loss_ce', torch.tensor(0)).item()
        stats['bbox'] += ld.get('loss_bbox', torch.tensor(0)).item()
        stats['giou'] += ld.get('loss_giou', torch.tensor(0)).item()
        stats['n'] += 1

        if (i+1) % args.print_freq == 0:
            n = stats['n']
            el = time.time() - t0
            eta = el / (i+1) * (len(loader) - i - 1)
            print(f"  [{i+1}/{len(loader)}] "
                  f"loss={stats['loss']/n:.4f} "
                  f"ce={stats['ce']/n:.4f} "
                  f"bbox={stats['bbox']/n:.4f} "
                  f"giou={stats['giou']/n:.4f} "
                  f"ETA={datetime.timedelta(seconds=int(eta))}")

    n = max(stats['n'], 1)
    return {k: stats[k]/n for k in ['loss', 'ce', 'bbox', 'giou']}


@torch.no_grad()
def evaluate(model, criterion, loader, device, args):
    model.eval()
    criterion.eval()
    stats = {'loss': 0, 'ce': 0, 'bbox': 0, 'giou': 0, 'n': 0}

    for batch in loader:
        images = batch['images'].to(device)
        targets = prepare_targets(batch, device)
        if sum(len(t['labels']) for t in targets) == 0:
            continue

        with autocast(enabled=args.amp):
            outputs = model(images)
            ld = criterion(outputs, targets)

        w = criterion.weight_dict
        loss = sum(ld[k] * w[k] for k in ld if k in w)

        if not torch.isnan(loss):
            stats['loss'] += loss.item()
            stats['ce'] += ld.get('loss_ce', torch.tensor(0)).item()
            stats['bbox'] += ld.get('loss_bbox', torch.tensor(0)).item()
            stats['giou'] += ld.get('loss_giou', torch.tensor(0)).item()
            stats['n'] += 1

    n = max(stats['n'], 1)
    r = {k: stats[k]/n for k in ['loss', 'ce', 'bbox', 'giou']}
    print(f"  [Val] loss={r['loss']:.4f} ce={r['ce']:.4f} "
          f"bbox={r['bbox']:.4f} giou={r['giou']:.4f}")
    return r


def build_lr_scheduler(optimizer, args):
    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        prog = (epoch - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1)
        return max(args.min_lr / args.lr, 0.5 * (1 + math.cos(math.pi * prog)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_ckpt(model, optimizer, scheduler, scaler, epoch, best, args, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_loss': best,
        'args': vars(args),
    }, path)


def load_ckpt(model, ckpt_path, device, model_only=True):
    ck = torch.load(ckpt_path, map_location=device)
    state = ck.get('model_state_dict', ck.get('model'))
    model_dict = model.state_dict()

    skip_keys = []
    missing_in_model = []

    for k in list(state.keys()):
        if k not in model_dict:
            missing_in_model.append(k)
            skip_keys.append(k)
        elif state[k].shape != model_dict[k].shape:
            print(f"  shape mismatch, skip: {k}: "
                  f"ckpt {tuple(state[k].shape)} vs model {tuple(model_dict[k].shape)}")
            skip_keys.append(k)

    if missing_in_model:
        print(f"\n  {len(missing_in_model)} keys in ckpt not present in model (skipped):")
        prefixes = {}
        for k in missing_in_model:
            prefix = '.'.join(k.split('.')[:3])
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        for p, count in sorted(prefixes.items())[:6]:
            print(f"      - {p}.* ({count})")

    new_in_model = [k for k in model_dict if k not in state]
    if new_in_model:
        print(f"\n  {len(new_in_model)} new keys in model (randomly initialized):")
        prefixes = {}
        for k in new_in_model:
            prefix = '.'.join(k.split('.')[:3])
            prefixes[prefix] = prefixes.get(prefix, 0) + 1
        for p, count in sorted(prefixes.items())[:6]:
            print(f"      + {p}.* ({count})")

    filtered = {k: v for k, v in state.items() if k not in skip_keys}
    model.load_state_dict(filtered, strict=False)

    epoch = ck.get('epoch', 0)
    best_loss = ck.get('best_loss', float('inf'))

    if model_only:
        print(f"\n  loaded model weights only from {ckpt_path} (epoch {epoch})")
        return 0, float('inf')
    else:
        print(f"\n  full resume from {ckpt_path} (epoch {epoch})")
        return epoch, best_loss


def main(args):
    print("=" * 70)
    print("MambaPulse Training")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")

    train_loader, val_loader = build_dataloaders(args)
    print(f"Train: {len(train_loader.dataset)} images, "
          f"Val: {len(val_loader.dataset)} images")

    model = MambaPulse(
        img_size=args.img_size, patch_size=16,
        vim_depth=24, vim_embed_dim=192, d_state=16,
        out_channels=256, bifpn_repeats=3,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        d_ffn=args.d_ffn,
        num_classes=args.num_classes,
        dropout=args.dropout,
        drop_path_rate=args.drop_path_rate,
        num_groups=args.num_groups,
        use_ca=args.use_ca,
        use_sa=args.use_sa,
        use_fpn=args.use_fpn,
        use_hires_mamba=args.use_hires_mamba,
        hires_d_model=args.hires_d_model,
        hires_num_layers=args.hires_num_layers,
        skip_p2=args.skip_p2,
    )

    if args.use_ca or args.use_sa or args.use_fpn or (not args.use_hires_mamba):
        print(f"\nAblation mode:")
        print(f"    HiRes path: {'HiRes-Mamba' if args.use_hires_mamba else 'CNN stem'}")
        print(f"    Cross:      {'CrossAttention' if args.use_ca else 'MQI'}")
        print(f"    Self:       {'SelfAttention' if args.use_sa else 'MQSI'}")
        print(f"    MultiScale: {'FPN' if args.use_fpn else 'BiFPN'}")
        print(f"    P2 level:   {'skip' if args.skip_p2 else 'enabled'}")

    if args.vim_pretrained and args.resume:
        print("\n  --resume detected, skipping vim_pretrained (backbone overwritten by ckpt)")
    elif args.vim_pretrained:
        if not os.path.exists(args.vim_pretrained):
            raise FileNotFoundError(f"pretrained weights not found: {args.vim_pretrained}")
        load_vim_pretrained(model.encoder.backbone, args.vim_pretrained)
    elif not args.resume:
        print("\nNo --vim_pretrained and no --resume, backbone trained from scratch (not recommended)\n")

    model = model.to(device)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    matcher = HungarianMatcher(
        cost_class=args.cost_class,
        cost_bbox=args.cost_bbox,
        cost_giou=args.cost_giou,
        num_classes=args.num_classes,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
    )
    weight_dict = {
        'loss_ce': args.w_ce,
        'loss_bbox': args.w_bbox,
        'loss_giou': args.w_giou,
    }
    visdrone_class_freq = [8844, 5125, 1287, 14064, 1975, 750, 1045, 532, 251, 4886]

    criterion = SetCriterion(
        num_classes=args.num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        losses=['labels', 'boxes'],
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        class_freq=visdrone_class_freq if args.class_balanced else None,
    ).to(device)

    print(f"\nFocal Loss: alpha={args.focal_alpha}, gamma={args.focal_gamma}")
    print(f"Loss weights: ce={args.w_ce}, bbox={args.w_bbox}, giou={args.w_giou}")
    print(f"Matcher cost: cls={args.cost_class}, bbox={args.cost_bbox}, giou={args.cost_giou}")

    backbone_lr = args.lr * args.backbone_lr_mult
    hires_lr = args.lr * args.hires_lr_mult
    rest_lr = args.lr

    backbone_params, hires_params, rest_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'encoder.backbone' in n:
            backbone_params.append(p)
        elif 'encoder.hires_branch' in n:
            hires_params.append(p)
        else:
            rest_params.append(p)

    param_groups = [
        {'params': backbone_params, 'lr': backbone_lr, 'name': 'backbone'},
        {'params': hires_params,    'lr': hires_lr,    'name': 'hires'},
        {'params': rest_params,     'lr': rest_lr,     'name': 'rest'},
    ]
    print(f"\nLR groups:")
    print(f"  backbone : lr={backbone_lr:.2e}, params={sum(p.numel() for p in backbone_params)/1e6:.2f}M")
    print(f"  hires    : lr={hires_lr:.2e}, params={sum(p.numel() for p in hires_params)/1e6:.2f}M")
    print(f"  rest     : lr={rest_lr:.2e}, params={sum(p.numel() for p in rest_params)/1e6:.2f}M")

    optimizer = torch.optim.AdamW(param_groups, lr=args.lr,
                                   weight_decay=args.weight_decay)
    scheduler = build_lr_scheduler(optimizer, args)
    scaler = GradScaler(enabled=args.amp)

    start_epoch = 0
    best_loss = float('inf')
    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"ckpt not found: {args.resume}")
        load_ckpt(model, args.resume, device, model_only=True)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nTraining: ep {start_epoch}->{args.epochs}, "
          f"bs={args.batch_size}x{args.grad_accum_steps}"
          f"={args.batch_size*args.grad_accum_steps}")
    if args.freeze_backbone_epochs > 0:
        print(f"Backbone frozen for first {args.freeze_backbone_epochs} epochs")

    log = []
    for epoch in range(start_epoch, args.epochs):
        if epoch < args.freeze_backbone_epochs:
            for n, p in model.named_parameters():
                if 'encoder.backbone' in n:
                    p.requires_grad = False
                if 'reference_point' in n:
                    p.requires_grad = False
            if epoch == start_epoch:
                print("  Backbone + reference_point frozen")
        else:
            for n, p in model.named_parameters():
                if 'encoder.backbone' in n:
                    p.requires_grad = True
                if 'reference_point' in n:
                    p.requires_grad = True
            if epoch == args.freeze_backbone_epochs:
                print("  Backbone + reference_point unfrozen")

        t0 = time.time()
        lr_disp = next((pg['lr'] for pg in optimizer.param_groups
                        if pg.get('name') == 'rest'), optimizer.param_groups[0]['lr'])
        print(f"\nEpoch {epoch+1}/{args.epochs}  lr={lr_disp:.2e}"
              f"{'  [backbone frozen]' if epoch < args.freeze_backbone_epochs else ''}")

        ts = train_one_epoch(model, criterion, train_loader, optimizer,
                             scaler, device, epoch, args)
        scheduler.step()

        vs = evaluate(model, criterion, val_loader, device, args)

        elapsed = time.time() - t0
        print(f"  Time: {datetime.timedelta(seconds=int(elapsed))}")

        log.append({'epoch': epoch+1, 'train': ts, 'val': vs, 'lr': lr_disp})
        with open(os.path.join(args.output_dir, 'log.json'), 'w') as f:
            json.dump(log, f, indent=2)

        if vs['loss'] < best_loss:
            best_loss = vs['loss']
            save_ckpt(model, optimizer, scheduler, scaler, epoch,
                      best_loss, args, os.path.join(args.output_dir, 'best.pth'))
            print(f"  Best: {best_loss:.4f}")

        if (epoch + 1) % args.save_freq == 0:
            save_ckpt(model, optimizer, scheduler, scaler, epoch,
                      best_loss, args,
                      os.path.join(args.output_dir, f'ep{epoch+1}.pth'))

        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            print(f"  Computing COCO AP...")
            t_map = time.time()
            metrics = compute_map(model, val_loader, device,
                                  num_classes=args.num_classes,
                                  img_size=args.img_size)
            print(f"  AP eval time: {time.time()-t_map:.1f}s")
            log[-1]['coco'] = metrics
            with open(os.path.join(args.output_dir, 'log.json'), 'w') as f:
                json.dump(log, f, indent=2)

        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.max_memory_allocated()/1024**3:.2f} GB")

    save_ckpt(model, optimizer, scheduler, scaler, args.epochs-1,
              best_loss, args, os.path.join(args.output_dir, 'last.pth'))
    print(f"\nDone! Best val loss: {best_loss:.4f}")


if __name__ == '__main__':
    p = argparse.ArgumentParser('MambaPulse')
    p.add_argument('--data_root', type=str, required=True)
    p.add_argument('--img_size', type=int, default=640)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--use_mosaic', action='store_true', default=True)
    p.add_argument('--no_mosaic', dest='use_mosaic', action='store_false')
    p.add_argument('--mosaic_prob', type=float, default=0.5)

    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--num_queries', type=int, default=400,
                   help='must be a perfect square (400=20^2 recommended)')
    p.add_argument('--num_decoder_layers', type=int, default=6)
    p.add_argument('--d_ffn', type=int, default=1024)
    p.add_argument('--dropout', type=float, default=0.0)
    p.add_argument('--drop_path_rate', type=float, default=0.1)

    p.add_argument('--use_hires_mamba', action='store_true', default=True,
                   help='enable HiRes-Mamba in place of image_stem')
    p.add_argument('--no_hires_mamba', dest='use_hires_mamba',
                   action='store_false', help='ablation: fall back to CNN stem')
    p.add_argument('--hires_d_model', type=int, default=128)
    p.add_argument('--hires_num_layers', type=int, default=4)
    p.add_argument('--hires_lr_mult', type=float, default=2.0,
                   help='HiRes-Mamba lr multiplier relative to rest lr')

    p.add_argument('--cost_class', type=float, default=2.0)
    p.add_argument('--cost_bbox', type=float, default=2.0)
    p.add_argument('--cost_giou', type=float, default=5.0)
    p.add_argument('--w_ce', type=float, default=2.0)
    p.add_argument('--w_bbox', type=float, default=2.0)
    p.add_argument('--w_giou', type=float, default=5.0)
    p.add_argument('--focal_alpha', type=float, default=0.25)
    p.add_argument('--focal_gamma', type=float, default=1.5)

    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--grad_accum_steps', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--backbone_lr_mult', type=float, default=0.1)
    p.add_argument('--freeze_backbone_epochs', type=int, default=5)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--clip_grad', type=float, default=1.0)
    p.add_argument('--warmup_epochs', type=int, default=3)
    p.add_argument('--min_lr', type=float, default=1e-6)
    p.add_argument('--amp', action='store_true', default=False)
    p.add_argument('--no_amp', dest='amp', action='store_false')
    p.add_argument('--num_groups', type=int, default=6)
    p.add_argument('--use_ca', action='store_true', default=False)
    p.add_argument('--use_sa', action='store_true', default=False)
    p.add_argument('--use_fpn', action='store_true', default=False)

    p.add_argument('--vim_pretrained', type=str, default='')

    p.add_argument('--output_dir', type=str, default='./output/mambapulse')
    p.add_argument('--print_freq', type=int, default=50)
    p.add_argument('--save_freq', type=int, default=10)
    p.add_argument('--eval_freq', type=int, default=5)
    p.add_argument('--resume', type=str, default='')
    p.add_argument('--model_only', action='store_true', default=True,
                   help='only restore model weights, not optimizer state')
    p.add_argument('--class_balanced', action='store_true', default=False)
    p.add_argument('--skip_p2', action='store_true', default=False,
                   help='ablation: drop P2, BiFPN uses 4 levels')

    main(p.parse_args())