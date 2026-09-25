#!/usr/bin/env python3
"""Train grayscale YOLOv8n (1ch, 640x480) for person occupancy."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

# CUDA-safe workers: spawn (set in main). Cap OpenCV threads to avoid fork/OpenMP deadlocks.
os.environ.setdefault("OPENCV_FOR_THREADS_NUM", "1")

from models import (
    ARCH_YAML,
    PRETRAINED_RGB,
    ROOT,
    build_model,
    exact_width_mode,
    load_compatible_weights,
    load_rgb_stem_into_gray,
    unwrap_model,
    yaml_wants_exact_width,
)


def _head_max_logit(head):
    scores = head["scores"]
    return scores.reshape(scores.shape[0], -1).amax(dim=1)


def occupancy_maxscore_bce(preds, batch):
    """Image-level BCE on max class logit vs empty/occupied. Cuts empty-frame FPs."""
    import torch
    import torch.nn.functional as F

    p = preds[0] if isinstance(preds, (list, tuple)) and isinstance(preds[0], dict) else preds
    if isinstance(p, tuple):
        p = next((x for x in p if isinstance(x, dict)), p[-1])
    heads = []
    if isinstance(p, dict) and "one2many" in p:
        heads.append(p["one2many"])
        if "one2one" in p:
            heads.append(p["one2one"])
    elif isinstance(p, dict) and "scores" in p:
        heads.append(p)
    else:
        import torch

        return torch.zeros((), device=batch["img"].device)
    occupied = torch.zeros(heads[0]["scores"].shape[0], device=heads[0]["scores"].device, dtype=heads[0]["scores"].dtype)
    idx = batch.get("batch_idx")
    if idx is not None and len(idx):
        occupied[idx.long().unique()] = 1.0
    loss = heads[0]["scores"].new_zeros(())
    for head in heads:
        loss = loss + F.binary_cross_entropy_with_logits(_head_max_logit(head), occupied)
    return loss / len(heads)

DEFAULT_CFG = ROOT / "configs" / "train.yaml"

# COCO-80 ids. Indoor = furniture/appliance/electronics/kitchen/indoor-stuff, and no vehicle/traffic/sports.
_STRONG_IN = set(range(39, 67)) | set(range(68, 80))  # bottle..keyboard, microwave..toothbrush
_STRONG_OUT = set(range(1, 14)) | set(range(29, 39))  # bicycle..bench, frisbee..racket
_TRUNC_RE = __import__("re").compile(r"_trunc\d+$")


def _scene_stem(path) -> str:
    return _TRUNC_RE.sub("", Path(path).stem)


def build_scene_weights(indoor_gain: float = 3.0) -> dict[str, float]:
    """Map COCO image stem -> loss weight. Indoor scenes are indoor_gain times outdoor."""
    labels = ROOT / "data" / "raw" / "coco" / "labels"
    weights = {}
    n_in = 0
    for split in ("train2017", "val2017"):
        for path in (labels / split).glob("*.txt"):
            cls = set()
            for line in path.read_text().splitlines():
                parts = line.split()
                if parts:
                    cls.add(int(float(parts[0])))
            indoor = bool(cls & _STRONG_IN) and not (cls & _STRONG_OUT)
            weights[path.stem] = indoor_gain if indoor else 1.0
            n_in += indoor
    print(f"Scene weights: indoor={n_in} other={len(weights) - n_in} gain={indoor_gain}")
    return weights


def _stamp_scene_weights(dataset, weights: dict[str, float], mean_w: float) -> None:
    labels = getattr(dataset, "labels", None)
    if not labels:
        return
    for lb in labels:
        raw = weights.get(_scene_stem(lb["im_file"]), 1.0) / mean_w
        lb["scene_weight"] = raw


def _patch_mosaic_scene_weight() -> None:
    from ultralytics.data.augment import Mosaic

    if getattr(Mosaic._cat_labels, "_scene_patched", False):
        return
    orig = Mosaic._cat_labels

    def _cat_labels(self, mosaic_labels):
        out = orig(self, mosaic_labels)
        if out and mosaic_labels and any("scene_weight" in lb for lb in mosaic_labels):
            vals = [float(lb.get("scene_weight", 1.0)) for lb in mosaic_labels]
            out["scene_weight"] = sum(vals) / len(vals)
        return out

    _cat_labels._scene_patched = True
    Mosaic._cat_labels = _cat_labels


def _install_scene_loss(criterion) -> None:
    """Scale box/cls/dfl loss of each image by its scene weight."""
    if getattr(criterion, "_scene_installed", False):
        return
    import torch
    import torch.nn.functional as F

    from ultralytics.utils.loss import bbox2dist, bbox_iou
    from ultralytics.utils.tal import make_anchors

    bbox_mod = criterion.bbox_loss
    orig_bbox = bbox_mod.forward

    def bbox_forward(
        pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride
    ):
        img_w = getattr(criterion, "_img_w", None)
        if img_w is None:
            return orig_bbox(
                pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask, imgsz, stride
            )
        idx = fg_mask.nonzero(as_tuple=True)
        weight = target_scores[idx].sum(-1, keepdim=True) * img_w[idx[0]].unsqueeze(-1)
        iou = bbox_iou(pred_bboxes[idx], target_bboxes[idx], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum
        if bbox_mod.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, bbox_mod.dfl_loss.reg_max - 1)
            loss_dfl = bbox_mod.dfl_loss(pred_dist[idx].view(-1, bbox_mod.dfl_loss.reg_max), target_ltrb[idx]) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes) * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred = pred_dist * stride
            pred[..., 0::2] /= imgsz[1]
            pred[..., 1::2] /= imgsz[0]
            loss_dfl = F.l1_loss(pred[idx], target_ltrb[idx], reduction="none").mean(-1, keepdim=True) * weight
            loss_dfl = loss_dfl.sum() / target_scores_sum
        return loss_iou, loss_dfl

    bbox_mod.forward = bbox_forward
    orig_get = criterion.get_assigned_targets_and_loss

    def get_assigned(preds, batch):
        bs = preds["boxes"].shape[0]
        raw = batch.get("scene_weight")
        if raw is None:
            criterion._img_w = None
        else:
            vals = raw if isinstance(raw, torch.Tensor) else torch.tensor(list(raw), device=criterion.device)
            criterion._img_w = vals.to(device=criterion.device, dtype=preds["boxes"].dtype).reshape(bs)
        try:
            aux, loss, items = orig_get(preds, batch)
        finally:
            criterion._img_w = None
        return aux, loss, items

    # Weight classification the same way. Box/DFL go through bbox_forward.
    def get_assigned_weighted(preds, batch):
        loss = torch.zeros(3, device=criterion.device)
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], criterion.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=criterion.device, dtype=dtype) * criterion.stride[0]
        targets = torch.cat((batch["batch_idx"].view(-1, 1), batch["cls"].view(-1, 1), batch["bboxes"]), 1)
        targets = criterion.preprocess(targets.to(criterion.device), batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)
        pred_bboxes = criterion.bbox_decode(anchor_points, pred_distri)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = criterion.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        target_scores_sum = target_scores.sum().clamp_(min=1)
        raw = batch.get("scene_weight")
        if raw is None:
            img_w = None
        else:
            vals = raw if isinstance(raw, torch.Tensor) else torch.tensor(list(raw), dtype=dtype)
            img_w = vals.to(device=criterion.device, dtype=dtype).reshape(batch_size)
        criterion._img_w = img_w
        bce_loss = criterion.bce(pred_scores, target_scores.to(dtype))
        if img_w is not None:
            bce_loss = bce_loss * img_w.view(batch_size, 1, 1)
        if criterion.class_weights is not None:
            bce_loss *= criterion.class_weights
        loss[1] = bce_loss.sum() / target_scores_sum
        loss[0], loss[2] = criterion.bbox_loss(
            pred_distri, pred_bboxes, anchor_points, target_bboxes / stride_tensor,
            target_scores, target_scores_sum, fg_mask, imgsz, stride_tensor,
        )
        criterion._img_w = None
        loss[0] *= criterion.hyp.box
        loss[1] *= criterion.hyp.cls
        loss[2] *= criterion.hyp.dfl
        return (
            (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor),
            loss,
            dict(zip(criterion.loss_names, loss.detach())),
        )

    criterion.get_assigned_targets_and_loss = get_assigned_weighted
    criterion._scene_installed = True
    del orig_get


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_devices(requested) -> int | list[int] | str:
    """Use every visible CUDA device when requested is all/None."""
    import torch

    n = torch.cuda.device_count()
    want_all = requested in (None, "", "all", "All")
    if want_all:
        if n <= 0:
            return "cpu"
        ids = list(range(n))
        print(f"Using all visible GPUs: {ids}  ({n} cards)")
        if n != 16:
            print(f"Note: asked for 16 cards, this machine exposes {n}.")
        return ids[0] if n == 1 else ids
    if isinstance(requested, str) and "," in requested:
        return [int(x) for x in requested.split(",")]
    if isinstance(requested, (list, tuple)):
        return [int(x) for x in requested]
    return requested


def resolve_arch(model_name) -> Path:
    if model_name is None:
        return ARCH_YAML
    path = Path(model_name)
    if not path.is_absolute():
        path = ROOT / path
    return path if path.is_file() else ARCH_YAML


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None, help="GPU ids, or 'all' for every visible card.")
    p.add_argument("--weights", type=Path, default=None, help="Finetune a 1-ch checkpoint.")
    args = p.parse_args()

    import torch
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except Exception:
        pass
    torch.backends.cudnn.benchmark = True

    cfg = load_cfg(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    cfg["device"] = resolve_devices(args.device if args.device is not None else cfg.get("device"))
    devices = cfg["device"]
    n_gpu = len(devices) if isinstance(devices, list) else (1 if devices not in ("cpu", "CPU") else 0)
    if n_gpu > 1:
        per_gpu = int(cfg.get("batch", 128))
        cfg["batch"] = per_gpu * n_gpu
        print(f"Scaled batch to {cfg['batch']}  ({per_gpu}/GPU x {n_gpu})")

    data = Path(cfg["data"])
    if not data.is_absolute():
        data = ROOT / data
        cfg["data"] = str(data)
    if not data.is_file():
        raise FileNotFoundError(f"Dataset yaml missing: {data}. Run prepare_data.py first.")

    project = cfg.get("project", "runs")
    if not Path(project).is_absolute():
        cfg["project"] = str(ROOT / project)

    arch = resolve_arch(cfg.pop("model", None))
    occ_w = float(cfg.pop("occupancy_loss", 0) or 0)
    indoor_gain = float(cfg.pop("indoor_weight", 0) or 0)
    init_from = cfg.pop("init_from", None)
    rgb_w = cfg.pop("pretrained_rgb", None)
    rgb_path = Path(rgb_w) if rgb_w else PRETRAINED_RGB
    if not rgb_path.is_absolute():
        rgb_path = ROOT / rgb_path

    # Trainer rebuilds DetectionModel inside train(); keep exact-width patch alive.
    with exact_width_mode(yaml_wants_exact_width(arch)):
        if args.weights is None:
            print(f"Train {arch}  data={cfg['data']}  imgsz={cfg.get('imgsz')}")
            model = build_model(weights=None, arch=arch)
            stem_out = int(model.model.model[0].conv.weight.shape[0])
            # Pico stem is 8-wide; RGB yolov8n is 16-wide and will not transfer.
            if stem_out == 16 and rgb_path.is_file() and not init_from:

                def _inject_pretrained(trainer):
                    load_rgb_stem_into_gray(trainer.model, rgb_path)
                    ema = getattr(trainer, "ema", None)
                    if ema is not None and getattr(ema, "ema", None) is not None:
                        ema.ema.load_state_dict(unwrap_model(trainer.model).state_dict())
                        ema.updates = 0

                model.add_callback("on_pretrain_routine_end", _inject_pretrained)
            else:
                print(f"Skip RGB stem inject (dest first conv out={stem_out})")
            if init_from:

                def _inject_compatible(trainer):
                    load_compatible_weights(trainer.model, init_from)
                    ema = getattr(trainer, "ema", None)
                    if ema is not None and getattr(ema, "ema", None) is not None:
                        ema.ema.load_state_dict(unwrap_model(trainer.model).state_dict())
                        ema.updates = 0

                model.add_callback("on_pretrain_routine_end", _inject_compatible)
                print(f"Will copy compatible weights from {init_from}")
        else:
            print(f"Finetune {args.weights}  data={cfg['data']}")
            model = build_model(args.weights)
        if occ_w > 0:

            def _attach_occupancy_loss(trainer):
                det = trainer.model

                def loss(batch, preds=None):
                    if getattr(det, "criterion", None) is None:
                        det.criterion = det.init_criterion()
                    if preds is None:
                        preds = det.forward(batch["img"])
                    loss_t, items = det.criterion(preds, batch)
                    extra = occupancy_maxscore_bce(preds, batch) * occ_w * int(batch["img"].shape[0])
                    if isinstance(items, dict):
                        items = dict(items)
                        items["occ_loss"] = extra.detach()
                    return loss_t + extra, items

                det.loss = loss
                print(f"Occupancy max-score BCE enabled  weight={occ_w}")

            model.add_callback("on_pretrain_routine_end", _attach_occupancy_loss)
        if indoor_gain > 1:

            def _attach_scene_weights(trainer):
                table = build_scene_weights(indoor_gain)
                ds = trainer.train_loader.dataset
                stems = [_scene_stem(lb["im_file"]) for lb in ds.labels]
                raw = [table.get(s, 1.0) for s in stems]
                mean_w = sum(raw) / max(1, len(raw))
                _stamp_scene_weights(ds, table, mean_w)
                _patch_mosaic_scene_weight()
                det = unwrap_model(trainer.model)
                orig_loss = det.loss

                def loss(batch, preds=None):
                    if getattr(det, "criterion", None) is None:
                        det.criterion = det.init_criterion()
                    _install_scene_loss(det.criterion)
                    return orig_loss(batch, preds)

                det.loss = loss
                n_in = sum(v >= indoor_gain - 1e-6 for v in raw)
                print(
                    f"Indoor loss weight {indoor_gain}x outdoor  "
                    f"train indoor={n_in}/{len(raw)}  mean_raw={mean_w:.3f} (normalized to 1)"
                )

            model.add_callback("on_pretrain_routine_end", _attach_scene_weights)
        model.train(**cfg)
    save_dir = Path(cfg["project"]) / cfg.get("name", "exp") / "weights"
    print(f"Best checkpoint: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
