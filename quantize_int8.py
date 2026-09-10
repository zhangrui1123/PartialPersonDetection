#!/usr/bin/env python3
"""Quant_INT8-8 post-training quantization for the grayscale occupancy ONNX.

Follows the HarmonyOS CANN Kit lightweight-tool no-train flow:
  strategy Quant_INT8-8 (8-bit weights + 8-bit activations),
  BINARY calibration tensors, input shape images:1,1,480,640.

If official dopt_so.py is on PATH or $DOPT_HOME, that tool is used.
Otherwise ONNX Runtime static 8a8w PTQ produces an evaluable INT8 ONNX.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnx
from onnx import version_converter
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_ONNX = ROOT / "weights" / "yolo_gray_640_480.onnx"
DEFAULT_OUT = ROOT / "weights" / "yolo_gray_640_480_int8.onnx"
H, W = 480, 640
INPUT_NAME = "images"


def _list_images(d: Path) -> list[Path]:
    files = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp"):
        files.extend(d.glob(ext))
    return sorted(files)


def preprocess_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    if img.shape[0] != H or img.shape[1] != W:
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
    x = img.astype(np.float32) / 255.0
    return x.reshape(1, 1, H, W)


def select_calib_images(image_dir: Path, max_images: int, seed: int) -> list[Path]:
    images = _list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images in {image_dir}")
    if len(images) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(images), size=max_images, replace=False)
        images = [images[i] for i in sorted(idx.tolist())]
    return images


def write_calib_bins(images: list[Path], out_dir: Path) -> list[Path]:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    bins = []
    for i, im in enumerate(images):
        x = preprocess_gray(im)
        bp = out_dir / f"{i:04d}_{im.stem}.bin"
        x.tofile(bp)
        bins.append(bp)
    print(f"Wrote {len(bins)} calib bins to {out_dir}  shape=1x1x{H}x{W} float32")
    return bins


def write_calib_images(images: list[Path], out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    for i, im in enumerate(images):
        shutil.copy2(im, out_dir / f"{i:04d}_{im.name}")
    print(f"Wrote {len(images)} calib images to {out_dir}")
    return out_dir


def write_prototxt(path: Path, calib_images: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Official dopt ONNX BINARY path expects a single tensor file and rejects
    # a directory. IMAGE + (x-0)/255 matches this model's /255 preprocessing.
    path.write_text(
        "# CANN Kit lightweight tool — ONNX no-train 8-bit (Quant_INT8-8).\n"
        "# https://developer.huawei.com/consumer/cn/doc/harmonyos-guides/cannkit-lightweight-tool-instructions\n"
        "strategy: 'Quant_INT8-8'\n"
        "device: USE_CPU\n"
        "preprocess_parameter:\n"
        "{\n"
        "    input_type: IMAGE\n"
        "    image_format: BGR\n"
        "    mean_value: 0.0\n"
        "    standard_deviation: 255.0\n"
        f"    input_file_path: '{calib_images.resolve()}'\n"
        "}\n"
    )
    print(f"Wrote {path}")


def find_dopt() -> Path | None:
    env = os.environ.get("DOPT_HOME", "").strip()
    candidates = []
    if env:
        candidates.append(Path(env) / "dopt_so.py")
        candidates.append(Path(env) / "dopt_onnx_py3" / "dopt_so.py")
        candidates.append(Path(env) / "tools_dopt" / "dopt_onnx_py3" / "dopt_so.py")
    which = shutil.which("dopt_so.py")
    if which:
        candidates.append(Path(which))
    for p in (
        ROOT / "tools_dopt" / "dopt_onnx_py3" / "dopt_so.py",
        Path.home() / "cann-kit" / "tools" / "tools_dopt" / "dopt_onnx_py3" / "dopt_so.py",
        Path("/usr/local/Ascend/tools_dopt/dopt_onnx_py3/dopt_so.py"),
    ):
        candidates.append(p)
    for p in candidates:
        if p.is_file():
            return p
    return None


def run_official_dopt(dopt: Path, model: Path, prototxt: Path, out: Path, compress: Path) -> None:
    # dopt cwd is its own install dir; relative paths would resolve there.
    cmd = [
        sys.executable,
        str(dopt),
        "--framework",
        "5",
        "--mode",
        "0",
        "--model",
        str(model.resolve()),
        "--cal_conf",
        str(prototxt.resolve()),
        "--output",
        str(out.resolve()),
        "--input_shape",
        f"{INPUT_NAME}:1,1,{H},{W}",
        "--out_nodes",
        "output0",
        "--compress_conf",
        str(compress.resolve()),
    ]
    env = os.environ.copy()
    # Huawei dopt ships old _pb2.py; protobuf>=4 rejects those descriptors.
    env.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(dopt.parent), env=env)


class BinReader(CalibrationDataReader):
    def __init__(self, bins: list[Path], input_name: str):
        self.bins = bins
        self.input_name = input_name
        self.i = 0

    def get_next(self):
        if self.i >= len(self.bins):
            return None
        x = np.fromfile(self.bins[self.i], dtype=np.float32).reshape(1, 1, H, W)
        self.i += 1
        return {self.input_name: x}


def run_ort_int8(model: Path, bins: list[Path], out: Path) -> None:
    upgraded = out.with_name(out.stem + "_opset13.onnx")
    m = onnx.load(str(model))
    if m.opset_import[0].version < 13:
        m = version_converter.convert_version(m, 13)
        onnx.save(m, str(upgraded))
        src = upgraded
        print(f"Upgraded opset to 13: {src}")
    else:
        src = model
    quantize_static(
        model_input=str(src),
        model_output=str(out),
        calibration_data_reader=BinReader(bins, INPUT_NAME),
        quant_format=QuantFormat.QDQ,
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        extra_options={
            "ActivationSymmetric": True,
            "WeightSymmetric": True,
        },
        op_types_to_quantize=["Conv", "MatMul", "Gemm"],
    )
    if upgraded.is_file():
        upgraded.unlink(missing_ok=True)
    print(f"ORT Quant_INT8-8 PTQ wrote {out}  ({out.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, default=DEFAULT_ONNX)
    p.add_argument("--output", type=Path, default=DEFAULT_OUT)
    p.add_argument("--calib-dir", type=Path, default=None)
    p.add_argument("--max-calib", type=int, default=32, help="Huawei recommends <= 50 images.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--work-dir", type=Path, default=ROOT / "runs" / "quant")
    args = p.parse_args()

    if not args.model.is_file():
        raise FileNotFoundError(args.model)

    calib_src = args.calib_dir
    if calib_src is None:
        for cand in (
            ROOT / "data" / "person" / "images" / "train",
            ROOT / "data" / "person_debug" / "images" / "train",
        ):
            if cand.is_dir() and _list_images(cand):
                calib_src = cand
                break
    if calib_src is None or not calib_src.is_dir():
        raise FileNotFoundError(
            "Calibration images missing. Run: python3 prepare_data.py --debug"
        )

    work = args.work_dir
    work.mkdir(parents=True, exist_ok=True)
    images = select_calib_images(calib_src, args.max_calib, args.seed)
    bins = write_calib_bins(images, work / "calib_bins")
    calib_images = write_calib_images(images, work / "calib_images")
    prototxt = work / "config.prototxt"
    write_prototxt(prototxt, calib_images)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    dopt = find_dopt()
    if dopt is not None:
        print(f"Using official CANN Kit tool: {dopt}")
        compress = work / "compress_param"
        run_official_dopt(dopt, args.model, prototxt, args.output, compress)
        published_compress = args.output.with_name(
            args.output.name.replace("_int8.onnx", "_compress_param")
            if args.output.name.endswith("_int8.onnx")
            else args.output.stem + "_compress_param"
        )
        if compress.is_file():
            shutil.copy2(compress, published_compress)
            print(f"Wrote {published_compress} ({published_compress.stat().st_size / 1024:.1f} KB)")
        backend = "dopt_so.py"
    else:
        print(
            "Official dopt_so.py not found (set DOPT_HOME to tools_dopt/dopt_onnx_py3). "
            "Falling back to ONNX Runtime static 8a8w PTQ for evaluation."
        )
        run_ort_int8(args.model, bins, args.output)
        backend = "onnxruntime_static_8a8w"

    print(f"backend={backend}  output={args.output}")


if __name__ == "__main__":
    main()
