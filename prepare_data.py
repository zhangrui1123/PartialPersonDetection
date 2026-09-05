#!/usr/bin/env python3
"""Build 640x480 grayscale person data (COCO 2017 + truncated crops)."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import numpy as np
import yaml

PERSON_CLS = 0
COCO128_URLS = [
    "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip",
    "https://ultralytics.com/assets/coco128.zip",
]

ROOT = Path(__file__).resolve().parent
COCO128_RAW = ROOT / "data" / "raw" / "coco128"
COCO_RAW = ROOT / "data" / "raw" / "coco"
GRAY = ROOT / "data" / "gray"
PERSON = ROOT / "data" / "person"
DEBUG = ROOT / "data" / "person_debug"
OUT_W, OUT_H = 640, 480


def _yolo_to_xyxy(xc, yc, w, h, iw, ih):
    bw, bh = w * iw, h * ih
    x1 = (xc * iw) - bw / 2
    y1 = (yc * ih) - bh / 2
    return x1, y1, x1 + bw, y1 + bh


def _xyxy_to_yolo(x1, y1, x2, y2, iw, ih):
    bw, bh = x2 - x1, y2 - y1
    xc = (x1 + x2) / 2 / iw
    yc = (y1 + y2) / 2 / ih
    return xc, yc, bw / iw, bh / ih


def _read_person_boxes(label_path: Path) -> list[tuple[float, float, float, float]]:
    boxes = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls, xc, yc, w, h = int(float(parts[0])), *map(float, parts[1:])
        if cls == PERSON_CLS and w > 0 and h > 0:
            boxes.append((xc, yc, w, h))
    return boxes


def _clip_boxes_to_crop(
    boxes: list[tuple[float, float, float, float]],
    img_w: int,
    img_h: int,
    x0: int,
    y0: int,
    cw: int,
    ch: int,
    min_keep: float = 0.15,
) -> list[tuple[float, float, float, float]]:
    kept = []
    for xc, yc, w, h in boxes:
        x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, w, h, img_w, img_h)
        orig_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        nx1 = max(x1, x0) - x0
        ny1 = max(y1, y0) - y0
        nx2 = min(x2, x0 + cw) - x0
        ny2 = min(y2, y0 + ch) - y0
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        new_area = (nx2 - nx1) * (ny2 - ny1)
        if orig_area <= 0 or new_area / orig_area < min_keep:
            continue
        kept.append(_xyxy_to_yolo(nx1, ny1, nx2, ny2, cw, ch))
    return kept


def _write_label(path: Path, boxes: list[tuple[float, float, float, float]]) -> None:
    lines = [f"0 {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h in boxes]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def _write_dataset_yaml(out_dir: Path) -> Path:
    yaml_path = out_dir / "person.yaml"
    yaml_path.write_text(
        yaml.dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["person"],
                "channels": 1,
            },
            sort_keys=False,
        )
    )
    return yaml_path


def _truncated_crop(img, boxes, rng: random.Random):
    h, w = img.shape[:2]
    if not boxes:
        return None
    xc, yc, bw, bh = rng.choice(boxes)
    x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, bw, bh, w, h)
    side = rng.choice(["left", "right", "top", "bottom"])
    keep = rng.uniform(0.35, 0.75)
    if side == "left":
        cut = int(x1 + (x2 - x1) * keep)
        x0, y0, cw, ch = max(0, cut), 0, w - max(0, cut), h
    elif side == "right":
        cut = int(x1 + (x2 - x1) * (1.0 - keep))
        x0, y0, cw, ch = 0, 0, max(8, cut), h
    elif side == "top":
        cut = int(y1 + (y2 - y1) * keep)
        x0, y0, cw, ch = 0, max(0, cut), w, h - max(0, cut)
    else:
        cut = int(y1 + (y2 - y1) * (1.0 - keep))
        x0, y0, cw, ch = 0, 0, w, max(8, cut)
    if cw < 32 or ch < 32:
        return None
    cropped = img[y0 : y0 + ch, x0 : x0 + cw]
    kept = _clip_boxes_to_crop(boxes, w, h, x0, y0, cw, ch)
    if not kept:
        return None
    return cropped, kept


def _gray_cover_crop(img, boxes, out_w: int = OUT_W, out_h: int = OUT_H):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    h, w = gray.shape[:2]
    scale = max(out_w / w, out_h / h)
    nw, nh = max(out_w, int(round(w * scale))), max(out_h, int(round(h * scale)))
    resized = cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    x0 = (nw - out_w) // 2
    y0 = (nh - out_h) // 2
    crop = resized[y0 : y0 + out_h, x0 : x0 + out_w]
    if crop.shape[0] != out_h or crop.shape[1] != out_w:
        canvas = np.zeros((out_h, out_w), dtype=np.uint8)
        ch, cw = crop.shape[:2]
        canvas[:ch, :cw] = crop
        crop = canvas
    kept = []
    for xc, yc, bw, bh in boxes:
        x1, y1, x2, y2 = _yolo_to_xyxy(xc, yc, bw, bh, w, h)
        x1, x2 = x1 * scale - x0, x2 * scale - x0
        y1, y2 = y1 * scale - y0, y2 * scale - y0
        nx1, ny1 = max(0.0, x1), max(0.0, y1)
        nx2, ny2 = min(float(out_w), x2), min(float(out_h), y2)
        if nx2 - nx1 < 2 or ny2 - ny1 < 2:
            continue
        orig_area = max(1e-6, (x2 - x1) * (y2 - y1))
        if (nx2 - nx1) * (ny2 - ny1) / orig_area < 0.15:
            continue
        kept.append(_xyxy_to_yolo(nx1, ny1, nx2, ny2, out_w, out_h))
    return crop, kept


def _save_gray_sample(dst_im: Path, dst_lb: Path, img, boxes) -> None:
    gray, kept = _gray_cover_crop(img, boxes)
    cv2.imwrite(str(dst_im), gray)
    _write_label(dst_lb, kept)


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def _download_coco128(force: bool = False) -> Path:
    images = COCO128_RAW / "images" / "train2017"
    labels = COCO128_RAW / "labels" / "train2017"
    if images.is_dir() and labels.is_dir() and not force:
        if any(images.glob("*.jpg")):
            return COCO128_RAW
    zip_path = ROOT / "data" / "raw" / "coco128.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for url in COCO128_URLS:
        try:
            print(f"Downloading COCO128 from {url}")
            urlretrieve(url, zip_path)
            last_err = None
            break
        except Exception as exc:
            last_err = exc
    if last_err is not None:
        raise last_err
    extract_to = ROOT / "data" / "raw" / "_extract"
    if extract_to.exists():
        shutil.rmtree(extract_to)
    extract_to.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    found = next(
        (cand.parent for cand in extract_to.rglob("images") if (cand / "train2017").is_dir()),
        None,
    )
    if found is None:
        raise FileNotFoundError("Could not locate coco128/images/train2017")
    if COCO128_RAW.exists():
        shutil.rmtree(COCO128_RAW)
    shutil.move(str(found), str(COCO128_RAW))
    shutil.rmtree(extract_to, ignore_errors=True)
    zip_path.unlink(missing_ok=True)
    return COCO128_RAW


def build_debug(trunc_per_image: int = 2, val_ratio: float = 0.2, seed: int = 42) -> Path:
    """Small COCO128 set for smoke tests."""
    rng = random.Random(seed)
    _download_coco128()
    src_img = COCO128_RAW / "images" / "train2017"
    src_lab = COCO128_RAW / "labels" / "train2017"
    items, negatives = [], []
    for im_path in sorted(src_img.glob("*.jpg")):
        boxes = _read_person_boxes(src_lab / f"{im_path.stem}.txt")
        (items if boxes else negatives).append((im_path, boxes) if boxes else im_path)
    rng.shuffle(items)
    rng.shuffle(negatives)
    n_val = max(1, int(len(items) * val_ratio))
    n_neg_val = max(1, int(len(negatives) * val_ratio)) if negatives else 0
    splits = {"val": items[:n_val], "train": items[n_val:]}
    neg_splits = {"val": negatives[:n_neg_val], "train": negatives[n_neg_val:]}
    if DEBUG.exists():
        shutil.rmtree(DEBUG)
    for split in ("train", "val"):
        (DEBUG / "images" / split).mkdir(parents=True)
        (DEBUG / "labels" / split).mkdir(parents=True)
    stats = {"train": 0, "val": 0, "trunc": 0, "empty": 0}
    for split, rows in splits.items():
        for im_path, boxes in rows:
            img = cv2.imread(str(im_path))
            if img is None:
                continue
            _save_gray_sample(
                DEBUG / "images" / split / im_path.name,
                DEBUG / "labels" / split / f"{im_path.stem}.txt",
                img,
                boxes,
            )
            stats[split] += 1
            n_trunc = trunc_per_image if split == "train" else max(1, trunc_per_image // 2)
            made, attempts = 0, 0
            while made < n_trunc and attempts < n_trunc * 8:
                attempts += 1
                out = _truncated_crop(img, boxes, rng)
                if out is None:
                    continue
                crop, kept = out
                _save_gray_sample(
                    DEBUG / "images" / split / f"{im_path.stem}_trunc{made}.jpg",
                    DEBUG / "labels" / split / f"{im_path.stem}_trunc{made}.txt",
                    crop,
                    kept,
                )
                made += 1
                stats["trunc"] += 1
                stats[split] += 1
        for im_path in neg_splits[split]:
            neg = cv2.imread(str(im_path))
            if neg is None:
                continue
            _save_gray_sample(
                DEBUG / "images" / split / f"empty_{im_path.name}",
                DEBUG / "labels" / split / f"empty_{im_path.stem}.txt",
                neg,
                [],
            )
            stats[split] += 1
            stats["empty"] += 1
    yaml_path = _write_dataset_yaml(DEBUG)
    print(f"Wrote {yaml_path}: train={stats['train']} val={stats['val']}")
    return yaml_path


def _convert_one(job: tuple) -> tuple[str, int]:
    im_path, lab_path, dst_im, dst_lb = map(Path, job)
    if dst_im.is_file() and dst_lb.is_file():
        return "skip", 0
    img = cv2.imread(str(im_path))
    if img is None:
        return "fail", 0
    boxes = _read_person_boxes(lab_path)
    _save_gray_sample(dst_im, dst_lb, img, boxes)
    return ("person" if boxes else "empty"), 1


def convert_coco(workers: int = 16) -> Path:
    img_root, lab_root = COCO_RAW / "images", COCO_RAW / "labels"
    for split in ("train2017", "val2017"):
        if not (img_root / split).is_dir() or not (lab_root / split).is_dir():
            raise FileNotFoundError(f"Missing COCO {split} under {COCO_RAW}")
    from concurrent.futures import ProcessPoolExecutor, as_completed

    jobs = []
    for split, out_split in (("train2017", "train"), ("val2017", "val")):
        (GRAY / "images" / out_split).mkdir(parents=True, exist_ok=True)
        (GRAY / "labels" / out_split).mkdir(parents=True, exist_ok=True)
        for im_path in (img_root / split).glob("*.jpg"):
            jobs.append(
                (
                    str(im_path),
                    str(lab_root / split / f"{im_path.stem}.txt"),
                    str(GRAY / "images" / out_split / im_path.name),
                    str(GRAY / "labels" / out_split / f"{im_path.stem}.txt"),
                )
            )
    stats = {"person": 0, "empty": 0, "skip": 0, "fail": 0}
    print(f"Converting {len(jobs)} COCO images to 640x480 gray")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_convert_one, job) for job in jobs]
        done = 0
        for fut in as_completed(futs):
            kind, n = fut.result()
            stats[kind] = stats.get(kind, 0) + n
            done += 1
            if done % 5000 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  {stats}", flush=True)
    print(f"Gray cache {GRAY}: {stats}")
    return GRAY


def _trunc_jobs_one(job: tuple) -> int:
    im_path, lab_path, dst_im_dir, dst_lb_dir, n_trunc, seed = (
        Path(job[0]),
        Path(job[1]),
        Path(job[2]),
        Path(job[3]),
        job[4],
        job[5],
    )
    boxes = _read_person_boxes(lab_path)
    if not boxes:
        return 0
    img = cv2.imread(str(im_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = cv2.imread(str(im_path))
    if img is None:
        return 0
    rng = random.Random(seed)
    made, attempts = 0, 0
    while made < n_trunc and attempts < n_trunc * 8:
        attempts += 1
        out = _truncated_crop(img, boxes, rng)
        if out is None:
            continue
        crop, kept = out
        dst_im = dst_im_dir / f"{im_path.stem}_trunc{made}.jpg"
        dst_lb = dst_lb_dir / f"{im_path.stem}_trunc{made}.txt"
        if dst_im.is_file() and dst_lb.is_file():
            made += 1
            continue
        _save_gray_sample(dst_im, dst_lb, crop, kept)
        made += 1
    return made


def add_truncated_crops(trunc_per_image: int = 1, workers: int = 16, seed: int = 42) -> Path:
    if not (GRAY / "images" / "train").is_dir():
        raise FileNotFoundError(f"Missing {GRAY}. Run with --full first.")
    import subprocess

    PERSON.mkdir(parents=True, exist_ok=True)
    for sub in ("images", "labels"):
        src, dst = GRAY / sub, PERSON / sub
        if not dst.exists():
            try:
                subprocess.check_call(["cp", "-a", "--link", str(src), str(dst)])
            except (subprocess.CalledProcessError, FileNotFoundError):
                shutil.copytree(src, dst, copy_function=_link_or_copy)
    jobs = []
    for split in ("train", "val"):
        src_im, src_lb = GRAY / "images" / split, GRAY / "labels" / split
        dst_im, dst_lb = PERSON / "images" / split, PERSON / "labels" / split
        n_trunc = trunc_per_image
        for im_path in src_im.glob("*.jpg"):
            lab_path = src_lb / f"{im_path.stem}.txt"
            if lab_path.is_file() and lab_path.stat().st_size > 0:
                jobs.append(
                    (
                        str(im_path),
                        str(lab_path),
                        str(dst_im),
                        str(dst_lb),
                        n_trunc,
                        seed + (zlib.crc32(im_path.stem.encode()) % 1_000_000),
                    )
                )
    print(f"Generating truncated crops for {len(jobs)} person images")
    n_made = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_trunc_jobs_one, job) for job in jobs]
        done = 0
        for fut in as_completed(futs):
            n_made += fut.result()
            done += 1
            if done % 5000 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}  trunc={n_made}", flush=True)
    yaml_path = _write_dataset_yaml(PERSON)
    n_train = len(list((PERSON / "images" / "train").glob("*.jpg")))
    n_val = len(list((PERSON / "images" / "val").glob("*.jpg")))
    print(f"Wrote {yaml_path}: train={n_train} val={n_val} trunc={n_made}")
    return yaml_path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--debug", action="store_true", help="Build a small COCO128 smoke set.")
    p.add_argument("--skip-convert", action="store_true", help="Reuse data/gray if already converted.")
    p.add_argument("--trunc-per-image", type=int, default=1)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.debug:
        build_debug(args.trunc_per_image, seed=args.seed)
        return
    if not args.skip_convert:
        convert_coco(workers=args.workers)
    add_truncated_crops(args.trunc_per_image, args.workers, args.seed)


if __name__ == "__main__":
    main()
