#!/usr/bin/env python3
"""Convert the official dopt ONNX to a HiAI offline model with --compress_conf.

  omg --framework=5 --model weights/yolo_gray_640_480_int8.onnx \\
      --output weights/yolo_gray_640_480_int8 --input_shape images:1,1,480,640 \\
      --out_nodes output0:0 --compress_conf runs/quant/compress_param
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
H, W = 480, 640


def find_omg() -> Path:
    which = shutil.which("omg")
    if which:
        return Path(which)
    env = os.environ.get("DDK_HOME", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env) / "tools" / "tools_omg" / "omg")
    candidates.append(Path.home() / "cann-kit" / "tools" / "tools_omg" / "omg")
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError("omg not found. source ~/cann-kit/set_env.sh first.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=ROOT / "weights" / "yolo_gray_640_480_int8.onnx")
    p.add_argument("--compress-conf", type=Path, default=ROOT / "weights" / "yolo_gray_640_480_compress_param")
    p.add_argument("--output", type=Path, default=ROOT / "weights" / "yolo_gray_640_480_int8")
    p.add_argument("--target", default="om", choices=("om", "omc", "tiny"))
    args = p.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(args.model)
    if not args.compress_conf.is_file():
        raise FileNotFoundError(
            f"{args.compress_conf} missing. Run: python3 quantize_int8.py"
        )

    omg = find_omg()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(omg),
        f"--model={args.model.resolve()}",
        "--framework=5",
        f"--output={args.output.resolve()}",
        f"--input_shape=images:1,1,{H},{W}",
        "--out_nodes=output0:0",
        f"--compress_conf={args.compress_conf.resolve()}",
        f"--target={args.target}",
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)
    produced = args.output.with_suffix(f".{args.target}")
    if produced.is_file():
        print(f"wrote {produced}  ({produced.stat().st_size / 1024 / 1024:.2f} MB)")


if __name__ == "__main__":
    main()
