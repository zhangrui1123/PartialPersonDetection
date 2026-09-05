#!/usr/bin/env python3
"""Train grayscale YOLOv8n (1ch, 640x480) for person occupancy."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_CFG = ROOT / "configs" / "train.yaml"


def load_cfg(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def load_rgb_stem_into_gray(dst_model, rgb_weights: str | Path) -> None:
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=DEFAULT_CFG)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--weights", type=Path, default=None, help="Finetune a 1-ch checkpoint.")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
    if args.device is not None:
        cfg["device"] = args.device

    data = Path(cfg["data"])
    if not data.is_absolute():
        data = ROOT / data
        cfg["data"] = str(data)
    if not data.is_file():
        raise FileNotFoundError(f"Dataset yaml missing: {data}. Run prepare_data.py first.")

    project = cfg.get("project", "runs")
    if not Path(project).is_absolute():
        cfg["project"] = str(ROOT / project)

    model_name = cfg.pop("model", "configs/yolov8n-gray.yaml")
    rgb_w = cfg.pop("pretrained_rgb", "weights/yolov8n.pt")
    rgb_path = ROOT / rgb_w if (ROOT / rgb_w).is_file() else Path(rgb_w)
    if not Path(model_name).is_absolute():
        yaml_path = ROOT / model_name
        if yaml_path.is_file():
            model_name = str(yaml_path)

    if args.weights is None:
        print(f"Train {model_name}  data={cfg['data']}  imgsz={cfg.get('imgsz')}")
        model = YOLO(model_name)

        def _inject_pretrained(trainer):
            load_rgb_stem_into_gray(trainer.model, rgb_path)
            ema = getattr(trainer, "ema", None)
            if ema is not None and getattr(ema, "ema", None) is not None:
                ema.ema.load_state_dict(trainer.model.state_dict())
                ema.updates = 0

        model.add_callback("on_pretrain_routine_end", _inject_pretrained)
    else:
        if not args.weights.is_file():
            raise FileNotFoundError(f"Finetune weights missing: {args.weights}")
        print(f"Finetune {args.weights}  data={cfg['data']}")
        model = YOLO(str(args.weights))
    model.train(**cfg)
    save_dir = Path(cfg["project"]) / cfg.get("name", "exp") / "weights"
    print(f"Best checkpoint: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
