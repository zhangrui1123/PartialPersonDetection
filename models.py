#!/usr/bin/env python3
"""Occupancy models: YOLOv8-gray (default) and YOLO26-wide.

Grayscale 640x480, class person only. Occupancy = any box >= conf.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import yaml
from ultralytics import YOLO
from ultralytics.nn import tasks as _yolo_tasks

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "yolo_gray_640_480"
ARCH_YAML = ROOT / "configs" / "yolov8n-gray.yaml"
V26_ARCH_YAML = ROOT / "configs" / "yolo26n-gray.yaml"
DEFAULT_WEIGHTS = ROOT / "weights" / f"{MODEL_NAME}.pt"
DEFAULT_ONNX = ROOT / "weights" / f"{MODEL_NAME}.onnx"
DEFAULT_INT8_ONNX = ROOT / "weights" / f"{MODEL_NAME}_int8.onnx"
DEFAULT_INT8_OM = ROOT / "weights" / f"{MODEL_NAME}_int8.om"
PRETRAINED_RGB = ROOT / "weights" / "yolov8n.pt"
PRETRAINED_V26 = ROOT / "weights" / "yolo26n.pt"
V26_WEIGHTS = ROOT / "weights" / "yolo_gray_640_480_v26.pt"

CHANNELS = 1
NC = 1
IMGSZ = (480, 640)  # H, W
STRIDES = (8, 16, 32)
OUTPUT_SHAPE = (1, 5, 6300)  # xywh + person, 4800+1200+300 anchors


def yaml_wants_exact_width(path: Path) -> bool:
    with path.open() as f:
        spec = yaml.safe_load(f) or {}
    return bool(spec.get("exact_width"))


@contextmanager
def exact_width_mode(enabled: bool = True):
    """Keep YAML channel counts as-is (do not round up to multiples of 8)."""
    if not enabled:
        yield
        return
    orig = _yolo_tasks.make_divisible
    _yolo_tasks.make_divisible = lambda x, divisor=8: int(round(float(x)))
    try:
        yield
    finally:
        _yolo_tasks.make_divisible = orig


def build_model(
    weights: str | Path | None = DEFAULT_WEIGHTS,
    arch: str | Path | None = None,
) -> YOLO:
    """Load a checkpoint, or build from an architecture yaml."""
    if weights is not None:
        path = Path(weights)
        if not path.is_absolute():
            path = ROOT / path
        if path.is_file():
            return YOLO(str(path))
        raise FileNotFoundError(f"Weights not found: {path}")
    yaml_path = Path(arch) if arch is not None else ARCH_YAML
    if not yaml_path.is_absolute():
        yaml_path = ROOT / yaml_path
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Architecture yaml missing: {yaml_path}")
    with exact_width_mode(yaml_wants_exact_width(yaml_path)):
        return YOLO(str(yaml_path))


def load_rgb_stem_into_gray(dst_model, rgb_weights: str | Path = PRETRAINED_RGB) -> None:
    """Copy RGB yolov8n weights; average the first conv from 3ch to 1ch."""
    rgb = YOLO(str(rgb_weights))
    dst = dst_model.state_dict()
    src = rgb.model.state_dict()
    mapped = {}
    n_copy, n_avg = 0, 0
    for key, weight in src.items():
        if key not in dst:
            continue
        if weight.shape == dst[key].shape:
            mapped[key] = weight
            n_copy += 1
        elif (
            weight.ndim == 4
            and weight.shape[1] == 3
            and dst[key].shape[1] == 1
            and weight.shape[0] == dst[key].shape[0]
            and weight.shape[2:] == dst[key].shape[2:]
        ):
            mapped[key] = weight.mean(dim=1, keepdim=True)
            n_avg += 1
    missing = dst_model.load_state_dict(mapped, strict=False)
    first = dst_model.model[0].conv
    print(
        f"Transferred RGB->gray: copied={n_copy} averaged_in_ch={n_avg} "
        f"unmatched_dst={len(missing.missing_keys)} first_conv={tuple(first.weight.shape)}"
    )


def load_compatible_weights(dst_model, src_weights: str | Path) -> None:
    """Copy overlapping tensors from a same-family 1ch checkpoint (exact or sliced)."""
    src_path = Path(src_weights)
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    src = YOLO(str(src_path))
    dst_sd = dst_model.state_dict()
    src_sd = src.model.state_dict()
    mapped = {}
    n_copy, n_partial, n_skip = 0, 0, 0
    for key, dst_w in dst_sd.items():
        src_w = src_sd.get(key)
        if src_w is None or src_w.ndim != dst_w.ndim:
            n_skip += 1
            continue
        if tuple(src_w.shape) == tuple(dst_w.shape):
            mapped[key] = src_w
            n_copy += 1
            continue
        sl = tuple(slice(0, min(a, b)) for a, b in zip(src_w.shape, dst_w.shape))
        if any(s.stop == 0 for s in sl):
            n_skip += 1
            continue
        buf = dst_w.clone()
        buf[sl] = src_w[sl]
        mapped[key] = buf
        n_partial += 1
    missing = dst_model.load_state_dict(mapped, strict=False)
    print(
        f"Compatible init from {src_path.name}: exact={n_copy} partial={n_partial} "
        f"skip={n_skip} unmatched_dst={len(missing.missing_keys)}"
    )


if __name__ == "__main__":
    model = build_model()
    print(f"{MODEL_NAME}  weights={DEFAULT_WEIGHTS}")
    print(f"input  1x{CHANNELS}x{IMGSZ[0]}x{IMGSZ[1]}  output {OUTPUT_SHAPE}")
    print(model.model)
