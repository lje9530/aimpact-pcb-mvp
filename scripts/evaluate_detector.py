"""Select PCB YOLO confidence on Validation and evaluate held-out Test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from prepare_pcb_demo import iou_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_inspection.detector import IMAGE_SIZE, MODEL_VERSION, YoloDetector
from pcb_inspection.vision import CLASSES, read_image


IOU = 0.33
CONFIDENCES = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70)


def collect(detector, dataset, split):
    records = []
    for path in sorted((dataset / "images" / split).glob("*.jpg")):
        image = read_image(path.read_bytes())
        height, width = image.shape
        annotation = (dataset / "labels" / split / f"{path.stem}.txt").read_text(encoding="utf-8")
        boxes, types = [], []
        for line in annotation.splitlines():
            kind, x, y, w, h = map(float, line.split())
            boxes.append([(x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
            types.append(int(kind)+1)
        result = detector(image)
        records.append((boxes, np.asarray(types, dtype=int), result["signals"]))
    if not records:
        raise ValueError(f"No data in {split} split")
    return records


def compute(records, confidence):
    per_type = {kind: {"tp": 0, "fp": 0, "fn": 0} for kind in range(1, 7)}
    truth_count = location_matches = positive_images = positive_flags = control_images = control_flags = 0
    for boxes, truth_types, signals in records:
        preds = [signal for signal in signals if signal["score"] >= confidence]
        pred_types = np.asarray([signal["type"] for signal in preds], dtype=int)
        if boxes:
            positive_images += 1
            positive_flags += bool(preds)
        else:
            control_images += 1
            control_flags += bool(preds)
        overlaps = iou_matrix([signal["box"] for signal in preds], boxes) >= IOU
        row, col = linear_sum_assignment(overlaps, maximize=True)
        location_matches += int(overlaps[row, col].sum())
        same_type = overlaps & (pred_types[:, None] == truth_types[None, :])
        row, col = linear_sum_assignment(same_type, maximize=True)
        matched = pred_types[row[same_type[row, col]]]
        for kind, stats in per_type.items():
            tp = int((matched == kind).sum())
            stats["tp"] += tp
            stats["fp"] += int((pred_types == kind).sum()) - tp
            stats["fn"] += int((truth_types == kind).sum()) - tp
        truth_count += len(boxes)
    tp, fp, fn = (sum(stats[key] for stats in per_type.values()) for key in ("tp", "fp", "fn"))
    return {"confidence": confidence, "iou": IOU, "labeled_defects": truth_count,
            "location_tp": location_matches, "location_recall": location_matches/max(1, truth_count),
            "type_location_tp": tp, "type_location_fp": fp, "type_location_fn": fn,
            "type_location_recall": tp/max(1, tp+fn), "type_location_precision": tp/max(1, tp+fp),
            "type_location_f1": 2*tp/max(1, 2*tp+fn+fp),
            "positive_images": positive_images, "positive_images_flagged": positive_flags,
            "template_controls": control_images, "template_controls_flagged": control_flags,
            "by_type": {CLASSES[kind]: stats for kind, stats in per_type.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs" / "yolo_dataset")
    parser.add_argument("--run", type=Path, default=ROOT / "outputs" / "detector" / "pcb_yolo26n")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics"))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    weights = args.run / "weights" / "best.pt"
    detector = YoloDetector(weights, confidence=CONFIDENCES[0], device=args.device)
    validation = collect(detector, args.dataset, "val")
    grid = [compute(validation, confidence) for confidence in CONFIDENCES]
    selected = max(grid, key=lambda row: row["type_location_f1"])
    holdout = compute(collect(detector, args.dataset, "test"), selected["confidence"])
    ap = detector.model.val(data=str((args.dataset / "data.yaml").resolve()), split="test", imgsz=IMAGE_SIZE,
                            batch=12, device=args.device, workers=0, plots=False,
                            project=str(args.run.resolve()), name="test_ap", exist_ok=True)
    with weights.open("rb") as stream:
        weights_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"model": MODEL_VERSION, "weights_sha256": weights_sha256, "threshold": selected["confidence"],
              "dataset": json.loads((args.dataset / "dataset.json").read_text(encoding="utf-8")),
              "validation_grid": grid, "validation": selected, "test": holdout,
              "standard_detection": {"map50": float(ap.box.map50), "map50_95": float(ap.box.map)}}
    path = args.run / "detector_metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Selected confidence: {selected['confidence']:.2f}\n{json.dumps(holdout, ensure_ascii=False, indent=2)}")
    print(f"Standard AP: {report['standard_detection']}\n{path}")


if __name__ == "__main__":
    main()
