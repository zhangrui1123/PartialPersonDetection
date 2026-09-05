#!/usr/bin/env python3
"""Detect persons and report occupied / empty per image or video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "weights" / "best.pt"


def occupancy_from_result(result, conf: float) -> dict:
    boxes = result.boxes
    n = 0
    max_conf = 0.0
    if boxes is not None and len(boxes):
        for b in boxes:
            c = float(b.conf[0])
            if c >= conf:
                n += 1
                max_conf = max(max_conf, c)
    occupied = n > 0
    return {
        "occupied": occupied,
        "status": "有人" if occupied else "没人",
        "n_person": n,
        "max_conf": round(max_conf, 4),
        "source": str(getattr(result, "path", "")),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    p.add_argument("--source", type=str, required=True, help="Image, dir, or video")
    p.add_argument("--conf", type=float, default=0.03)
    p.add_argument("--imgsz", nargs="+", type=int, default=[480, 640])
    p.add_argument("--device", default="0")
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--out-json", type=Path, default=None)
    args = p.parse_args()

    if not args.weights.is_file():
        raise FileNotFoundError(f"Weights not found: {args.weights}. Train first.")

    model = YOLO(str(args.weights))
    project = ROOT / "runs" / "predict"
    imgsz = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=imgsz,
        device=args.device,
        save=args.save,
        project=str(project),
        name="occupancy",
        exist_ok=True,
        verbose=False,
    )

    reports = [occupancy_from_result(r, args.conf) for r in results]
    n_occ = sum(1 for r in reports if r["occupied"])
    print(f"weights={args.weights}")
    print(f"frames/images={len(reports)} occupied={n_occ} empty={len(reports) - n_occ}")
    for r in reports[:30]:
        print(f"  {r['status']:4s}  n={r['n_person']}  conf={r['max_conf']:.3f}  {Path(r['source']).name}")
    if len(reports) > 30:
        print(f"  ... ({len(reports) - 30} more)")

    out_json = args.out_json or (project / "occupancy" / "occupancy_report.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps({"conf": args.conf, "reports": reports}, ensure_ascii=False, indent=2))
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
