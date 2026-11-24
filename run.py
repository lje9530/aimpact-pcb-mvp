"""Run the local AIMPACT PCB inspection workbench."""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--database", type=Path)
    parser.add_argument("--assets", type=Path, default=Path("data/pcb_demo"))
    parser.add_argument("--empty", action="store_true", help="Skip demonstration records for a new database")
    parser.add_argument("--detector", choices=("yolo", "paired"), default="yolo")
    parser.add_argument("--weights", type=Path, default=Path("outputs/detector/pcb_yolo26n/weights/best.pt"))
    parser.add_argument("--confidence", type=float, help="Override the confidence threshold selected on validation")
    args = parser.parse_args()

    assets_path = ROOT / args.assets
    weights = ROOT / args.weights if args.detector == "yolo" else None
    if weights is not None and not weights.is_file():
        parser.error("PCB detector weights are missing. Follow docs/PCB_DETECTOR.md to train the model.")
    if weights is None and not (assets_path / "model.joblib").is_file():
        parser.error("PCB assets are missing. Follow the preparation instructions in docs/PCB_BENCHMARK.md.")
    if not args.empty and not (assets_path / "samples.json").is_file():
        parser.error("Demo samples are missing. Supply --assets or start with --empty.")

    confidence = args.confidence
    if confidence is None and weights is not None:
        benchmark = weights.parent.parent / "detector_metrics.json"
        if benchmark.is_file():
            report = json.loads(benchmark.read_text(encoding="utf-8"))
            with weights.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != report["weights_sha256"]:
                    parser.error("The evaluation report belongs to different weights. Re-evaluate or set --confidence.")
            confidence = report["threshold"]
    database = args.database or Path("outputs/inspections-yolo.sqlite3" if weights else "outputs/inspections.sqlite3")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics"))
    os.environ.setdefault("YOLO_OFFLINE", "true")

    from pcb_inspection.web import create_app

    app = create_app(ROOT / database, assets_path, load_demo=not args.empty,
                     weights=weights, confidence=0.25 if confidence is None else confidence)
    app.run(host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
