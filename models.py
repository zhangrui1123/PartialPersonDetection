#!/usr/bin/env python3
"""Current occupancy model: yolo_gray_640_480.

1-channel grayscale YOLOv8n, native 640x480, Detect on P3/P4/P5 (no P2).
Architecture lives in configs/yolov8n-gray.yaml; published weights in weights/.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

MODEL_NAME = "yolo_gray_640_480"
ARCH_YAML = ROOT / "configs" / "yolov8n-gray.yaml"
PICO_ARCH_YAML = ROOT / "configs" / "yolov8n-gray-pico.yaml"
V5_ARCH_YAML = ROOT / "configs" / "yolov5n-gray.yaml"
V10_ARCH_YAML = ROOT / "configs" / "yolov10n-gray.yaml"
DEFAULT_WEIGHTS = ROOT / "weights" / f"{MODEL_NAME}.pt"
DEFAULT_ONNX = ROOT / "weights" / f"{MODEL_NAME}.onnx"
DEFAULT_INT8_ONNX = ROOT / "weights" / f"{MODEL_NAME}_int8.onnx"
DEFAULT_INT8_OM = ROOT / "weights" / f"{MODEL_NAME}_int8.om"
PRETRAINED_RGB = ROOT / "weights" / "yolov8n.pt"
PRETRAINED_V5 = ROOT / "weights" / "yolov5nu.pt"
PRETRAINED_V10 = ROOT / "weights" / "yolov10n.pt"

CHANNELS = 1
NC = 1
IMGSZ = (480, 640)  # H, W
STRIDES = (8, 16, 32)
OUTPUT_SHAPE = (1, 5, 6300)  # xywh + person, 4800+1200+300 anchors


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


if __name__ == "__main__":
    model = build_model()
    print(f"{MODEL_NAME}  weights={DEFAULT_WEIGHTS}")
    print(f"input  1x{CHANNELS}x{IMGSZ[0]}x{IMGSZ[1]}  output {OUTPUT_SHAPE}")
    print(model.model)
