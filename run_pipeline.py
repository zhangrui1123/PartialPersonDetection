#!/usr/bin/env python3
"""Prepare data → train → occupancy infer."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--device", default="0")
    p.add_argument("--skip-prepare", action="store_true")
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    py = sys.executable

    if not args.skip_prepare:
        cmd = [py, str(ROOT / "prepare_data.py")]
        if args.debug:
            cmd.append("--debug")
        run(cmd)
    if not args.skip_train:
        run([py, str(ROOT / "train.py"), "--epochs", str(args.epochs), "--device", str(args.device)])

    data_val = ROOT / "data" / "person" / "images" / "val"
    if args.debug or not data_val.is_dir():
        data_val = ROOT / "data" / "person_debug" / "images" / "val"
    run([py, str(ROOT / "infer.py"), "--source", str(data_val), "--device", str(args.device)])


if __name__ == "__main__":
    main()
