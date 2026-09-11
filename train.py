#!/usr/bin/env python3
"""Train grayscale YOLOv8n (1ch, 640x480) for person occupancy."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from models import (
    ARCH_YAML,
    PICO_ARCH_YAML,
    PRETRAINED_RGB,
    ROOT,
    build_model,
    exact_width_mode,
    load_rgb_stem_into_gray,
    yaml_wants_exact_width,
)

DEFAULT_CFG = ROOT / "configs" / "train.yaml"


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
        return PICO_ARCH_YAML if PICO_ARCH_YAML.is_file() else ARCH_YAML
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
    rgb_w = cfg.pop("pretrained_rgb", None)
    rgb_path = Path(rgb_w) if rgb_w else PRETRAINED_RGB
    if not rgb_path.is_absolute():
        rgb_path = ROOT / rgb_path

    # Trainer rebuilds DetectionModel inside train(); keep exact-width patch alive.
    with exact_width_mode(yaml_wants_exact_width(arch) if args.weights is None else False):
        if args.weights is None:
            print(f"Train {arch}  data={cfg['data']}  imgsz={cfg.get('imgsz')}")
            model = build_model(weights=None, arch=arch)
            stem_out = int(model.model.model[0].conv.weight.shape[0])
            # Pico stem is 8-wide; RGB yolov8n is 16-wide and will not transfer.
            if stem_out == 16 and rgb_path.is_file():

                def _inject_pretrained(trainer):
                    load_rgb_stem_into_gray(trainer.model, rgb_path)
                    ema = getattr(trainer, "ema", None)
                    if ema is not None and getattr(ema, "ema", None) is not None:
                        ema.ema.load_state_dict(trainer.model.state_dict())
                        ema.updates = 0

                model.add_callback("on_pretrain_routine_end", _inject_pretrained)
            else:
                print(f"Skip RGB stem inject (dest first conv out={stem_out})")
        else:
            print(f"Finetune {args.weights}  data={cfg['data']}")
            model = build_model(args.weights)
        model.train(**cfg)
    save_dir = Path(cfg["project"]) / cfg.get("name", "exp") / "weights"
    print(f"Best checkpoint: {save_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
