#!/usr/bin/env python3
"""Evaluate occupancy (有人/没人) and person-box AP on a YOLO ONNX/PT model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parent
H, W = 480, 640


def _list_images(d: Path) -> list[Path]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(d.glob(ext))
    return sorted(files)


def read_gt_boxes(label_path: Path) -> np.ndarray:
    boxes = []
    if label_path.is_file():
        for line in label_path.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, xc, yc, bw, bh = map(float, parts)
            x1 = (xc - bw / 2) * W
            y1 = (yc - bh / 2) * H
            x2 = (xc + bw / 2) * W
            y2 = (yc + bh / 2) * H
            boxes.append([x1, y1, x2, y2])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def preprocess(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    return (img.astype(np.float32) / 255.0).reshape(1, 1, H, W)


def xywh2xyxy(xywh: np.ndarray) -> np.ndarray:
    x, y, w, h = xywh.T
    return np.stack([x - w / 2, y - h / 2, x + w / 2, y + h / 2], axis=1)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thres: float) -> list[int]:
    if len(boxes) == 0:
        return []
    x1, y1, x2, y2 = boxes.T
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_thres]
    return keep


def decode(output: np.ndarray, conf_thres: float, iou_thres: float) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(output)
    if pred.ndim == 3:
        pred = pred[0]
    if pred.shape[0] < pred.shape[1]:
        pred = pred.T
    xywh, scores = pred[:, :4], pred[:, 4]
    mask = scores >= conf_thres
    xywh, scores = xywh[mask], scores[mask]
    boxes = xywh2xyxy(xywh)
    keep = nms(boxes, scores, iou_thres)
    return boxes[keep], scores[keep]


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a.T
    bx1, by1, bx2, by2 = b.T
    inter_x1 = np.maximum(ax1[:, None], bx1)
    inter_y1 = np.maximum(ay1[:, None], by1)
    inter_x2 = np.minimum(ax2[:, None], bx2)
    inter_y2 = np.minimum(ay2[:, None], by2)
    inter = np.maximum(0.0, inter_x2 - inter_x1) * np.maximum(0.0, inter_y2 - inter_y1)
    area_a = np.maximum(0.0, ax2 - ax1) * np.maximum(0.0, ay2 - ay1)
    area_b = np.maximum(0.0, bx2 - bx1) * np.maximum(0.0, by2 - by1)
    return inter / (area_a[:, None] + area_b - inter + 1e-9)


def occupancy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    acc = (tp + tn) / max(1, len(y_true))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)
    return {
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n": int(len(y_true)),
        "n_occupied_gt": int(y_true.sum()),
        "n_empty_gt": int((1 - y_true).sum()),
    }


def average_precision(matched_scores: list[tuple[float, int]], n_gt: int) -> float:
    if n_gt == 0:
        return 1.0 if not matched_scores else 0.0
    if not matched_scores:
        return 0.0
    matched_scores.sort(key=lambda x: -x[0])
    tp = np.cumsum([m for _, m in matched_scores])
    fp = np.cumsum([1 - m for _, m in matched_scores])
    recalls = tp / n_gt
    precisions = tp / np.maximum(tp + fp, 1)
    ap = 0.0
    rec_points = np.linspace(0, 1, 101)
    for r in rec_points:
        p = precisions[recalls >= r].max() if np.any(recalls >= r) else 0.0
        ap += p
    return float(ap / 101)


def run_onnx(model: Path, image_dir: Path, label_dir: Path, conf: float, iou: float) -> dict:
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    images = _list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images in {image_dir}")

    y_true, y_pred = [], []
    match_scores = []
    n_gt = 0
    n_pred = 0
    for im in images:
        x = preprocess(im)
        out = sess.run(None, {inp: x})[0]
        boxes, scores = decode(out, conf, iou)
        gt = read_gt_boxes(label_dir / f"{im.stem}.txt")
        occupied_gt = len(gt) > 0
        occupied_pd = len(boxes) > 0
        y_true.append(int(occupied_gt))
        y_pred.append(int(occupied_pd))
        n_gt += len(gt)
        n_pred += len(boxes)
        if len(boxes) == 0:
            continue
        if len(gt) == 0:
            match_scores.extend((float(s), 0) for s in scores)
            continue
        ious = box_iou(boxes, gt)
        used = set()
        order = scores.argsort()[::-1]
        for i in order:
            j = int(ious[i].argmax())
            if ious[i, j] >= 0.5 and j not in used:
                match_scores.append((float(scores[i]), 1))
                used.add(j)
            else:
                match_scores.append((float(scores[i]), 0))

    occ = occupancy_metrics(np.asarray(y_true), np.asarray(y_pred))
    map50 = average_precision(match_scores, n_gt)
    return {
        "model": str(model),
        "n_images": len(images),
        "conf": conf,
        "iou": iou,
        "occupancy": occ,
        "detection": {
            "mAP50": round(map50, 4),
            "n_gt_boxes": n_gt,
            "n_pred_boxes": n_pred,
        },
        "size_mb": round(model.stat().st_size / 1024 / 1024, 3),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, nargs="+", required=True)
    p.add_argument("--images", type=Path, default=None)
    p.add_argument("--labels", type=Path, default=None)
    p.add_argument("--conf", type=float, default=0.03)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--out-json", type=Path, default=ROOT / "runs" / "quant" / "eval_report.json")
    args = p.parse_args()

    image_dir = args.images
    label_dir = args.labels
    if image_dir is None:
        for cand in (
            ROOT / "data" / "person" / "images" / "val",
            ROOT / "data" / "person_debug" / "images" / "val",
        ):
            if cand.is_dir() and _list_images(cand):
                image_dir = cand
                label_dir = cand.parent.parent / "labels" / cand.name
                break
    if image_dir is None:
        raise FileNotFoundError("Val images missing. Run: python3 prepare_data.py --debug")
    if label_dir is None:
        label_dir = image_dir.parent.parent / "labels" / image_dir.name

    reports = []
    for w in args.weights:
        print(f"Evaluating {w}", flush=True)
        r = run_onnx(w, image_dir, label_dir, args.conf, args.iou)
        reports.append(r)
        print(
            f"  size={r['size_mb']}MB  occ_acc={r['occupancy']['accuracy']}  "
            f"occ_p={r['occupancy']['precision']}  occ_r={r['occupancy']['recall']}  "
            f"occ_f1={r['occupancy']['f1']}  mAP50={r['detection']['mAP50']}"
        )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {"images": str(image_dir), "labels": str(label_dir), "reports": reports}
    if len(reports) >= 2:
        a, b = reports[0], reports[1]
        payload["delta_int8_minus_fp32"] = {
            "occupancy_accuracy": round(b["occupancy"]["accuracy"] - a["occupancy"]["accuracy"], 4),
            "occupancy_f1": round(b["occupancy"]["f1"] - a["occupancy"]["f1"], 4),
            "mAP50": round(b["detection"]["mAP50"] - a["detection"]["mAP50"], 4),
        }
    args.out_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
