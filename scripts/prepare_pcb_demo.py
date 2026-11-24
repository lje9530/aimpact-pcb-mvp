"""Train a paired-image PCB baseline and produce the local research demo assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pcb_inspection.vision import CLASSES, MODEL_VERSION, candidates, read_image


def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1,4), np.asarray(b, dtype=float).reshape(-1,4)
    lo = np.maximum(a[:,None,:2], b[None,:,:2])
    hi = np.minimum(a[:,None,2:], b[None,:,2:])
    intersection = np.prod(np.maximum(hi-lo, 0), axis=-1)
    area_a = np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=-1)
    area_b = np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=-1)
    return intersection / np.maximum(area_a[:,None] + area_b[None,:] - intersection, 1e-12)


VAL_GROUPS = {"group12300", "group12000"}
TEST_GROUPS = {"group12100", "group90100"}


def source_samples(archive, root):
    splits = {"train": [], "validation": [], "test": []}
    for name in ("trainval", "test"):
        for row in archive.read(f"{root}/{name}.txt").decode().splitlines():
            if not row.strip():
                continue
            image, annotations = row.split()
            group = image.split("/")[0]
            if group in VAL_GROUPS:
                split = "validation"
            elif group in TEST_GROUPS:
                split = "test"
            elif name == "trainval":
                split = "train"
            else:
                continue
            prefix = f"{root}/{image[:-4]}"
            truth = [list(map(int, line.split())) for line in archive.read(f"{root}/{annotations}").decode().splitlines() if line.strip()]
            splits[split].append({"id": PurePosixPath(image).stem, "group": group,
                                  "test": f"{prefix}_test.jpg", "reference": f"{prefix}_temp.jpg", "truth": truth})
    return splits


def features_for(archive, samples, controls=False):
    records = []
    for sample in samples:
        inspected = read_image(archive.read(sample["test"]))
        template = read_image(archive.read(sample["reference"]))
        boxes, features = candidates(inspected, template)
        records.append({**sample, "boxes": boxes, "features": features, "control": False})
        if controls:
            # A template against itself is a control, not an independently measured normal board.
            records.append({**sample, "truth": [], "boxes": [], "features": [], "control": True})
    return records


def train_rows(records):
    predictors, labels = [], []
    for record in records:
        if not record["boxes"]:
            continue
        truth = record["truth"]
        overlaps = iou_matrix(record["boxes"], [row[:4] for row in truth])
        for feature, row in zip(record["features"], overlaps, strict=True):
            best = int(np.argmax(row))
            predictors.append(feature)
            labels.append(truth[best][4] if row[best] >= 0.19 else 0)
    return predictors, labels


def evaluate(model, rows, threshold):
    class_ids = np.asarray(model.classes_)
    defect = class_ids > 0
    image_truth, image_pred = [], []
    per_type = {key: Counter(tp=0, fp=0, fn=0) for key in range(1, 7)}
    for record in rows:
        boxes, truth = record["boxes"], record["truth"]
        probabilities = model.predict_proba(record["features"]) if boxes else np.empty((0, len(class_ids)))
        predictions = []
        for index, scores in enumerate(probabilities):
            if scores[defect].sum() >= threshold:
                position = np.argmax(scores[defect])
                predictions.append((float(scores[defect][position]), index, int(class_ids[defect][position])))
        image_truth.append(bool(truth))
        image_pred.append(bool(predictions))
        tracked = set()
        overlaps = iou_matrix(boxes, [row[:4] for row in truth])
        for _, index, kind in sorted(predictions, reverse=True):
            hits = [n for n, label in enumerate(truth) if n not in tracked and label[4] == kind and overlaps[index, n] >= .19]
            if hits:
                tracked.add(max(hits, key=lambda n: overlaps[index, n]))
                per_type[kind]["tp"] += 1
            else:
                per_type[kind]["fp"] += 1
        for n, label in enumerate(truth):
            if n not in tracked:
                per_type[label[4]]["fn"] += 1
    precision, recall, f1, _ = precision_recall_fscore_support(image_truth, image_pred, average="binary", zero_division=0)
    total = sum(per_type.values(), Counter())
    return {"threshold": threshold, "precision": precision, "recall": recall, "f1": f1,
            "confusion": confusion_matrix(image_truth, image_pred, labels=[False, True]).tolist(),
            "type_f1": 2*total['tp'] / max(1, 2*total['tp']+total['fp']+total['fn']),
            "types": {CLASSES[kind]: stats for kind, stats in per_type.items()}}


def write_demo(archive, rows, out):
    available = [row for row in rows if not row["control"]]
    groups = sorted({row["group"] for row in available})
    images = out / "images"
    images.mkdir(exist_ok=True)
    manifest = []
    for index, sample in enumerate([row for group in groups for row in [r for r in available if r["group"] == group][:6]]):
        reference = f"{sample['id']}_temp.jpg"
        inspected = f"{sample['id']}_test.jpg"
        (images / reference).write_bytes(archive.read(sample["reference"]))
        (images / inspected).write_bytes(archive.read(sample["test"]))
        is_control = index % 4 == 0
        manifest.append({"external_id": sample["id"] + ("-CTRL" if is_control else ""),
                         "supplier": "파트너 A" if sample["group"] == groups[0] else "파트너 B",
                         "product": "PCB-" + sample["group"][5:], "batch": f"DEMO-{index//3+1:02d}",
                         "inspected_at": f"2026-11-{20+index//4:02d}T09:{(index%4)*12:02d}:00",
                         "process": "AOI 최종 검사", "image_file": reference if is_control else inspected,
                         "reference_file": reference, "reported_result": "OK" if is_control else "NG",
                         "reported_type": "", "data_source": "DeepPCB 기준 영상 대조군" if is_control else "DeepPCB 검사 영상",
                         "ground_truth": [] if is_control else sample["truth"]})
    (out / "samples.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Download of the author’s DeepPCB GitHub repository ZIP")
    parser.add_argument("--out", type=Path, default=Path("data/pcb_demo"))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        root = next(name.removesuffix("/trainval.txt") for name in archive.namelist() if name.endswith("/PCBData/trainval.txt"))
        splits = source_samples(archive, root)
        print({key: len(samples) for key, samples in splits.items()}, flush=True)
        train = train_rows(features_for(archive, splits["train"]))
        val = features_for(archive, splits["validation"], controls=True)
        test = features_for(archive, splits["test"], controls=True)
        methods = {
            "logistic": make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced")),
            "forest": RandomForestClassifier(n_estimators=36, max_depth=11, min_samples_leaf=7, random_state=42, n_jobs=1),
        }
        results = {}
        for name, model in methods.items():
            model.fit(*train)
            validation = max((evaluate(model, val, threshold) for threshold in (.45, .6, .75, .9)), key=lambda row: row["type_f1"])
            start = perf_counter()
            test_metrics = evaluate(model, test, validation["threshold"])
            results[name] = {"validation": validation, "test": test_metrics,
                             "evaluation_ms_per_record": round((perf_counter()-start)*1000/len(test), 2)}
            print(name, round(validation["type_f1"], 3), round(test_metrics["type_f1"], 3), flush=True)
        chosen = max(results, key=lambda name: results[name]["validation"]["type_f1"])
        model_path = args.out / "model.joblib"
        joblib.dump({"model": methods[chosen], "version": MODEL_VERSION,
                     "threshold": results[chosen]["validation"]["threshold"]}, model_path, compress=3)
        from pcb_inspection.vision import predict
        times = []
        for sample in splits["test"]:
            result = predict(methods[chosen], read_image(archive.read(sample["test"])), read_image(archive.read(sample["reference"])),
                             noise_threshold=results[chosen]["validation"]["threshold"])
            times.append(result["ms"])
        with args.archive.open("rb") as source:
            archive_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
        benchmark = {"version": MODEL_VERSION, "archive_sha256": archive_sha256,
                     "splits": {name: {"pairs": len(rows), "groups": sorted({r['group'] for r in rows})} for name, rows in splits.items()},
                     "training_regions": dict(Counter(train[1])), "selected": chosen, "results": results,
                     "weight_bytes": model_path.stat().st_size, "cpu_ms_median": float(np.median(times)),
                     "cpu_ms_p95": float(np.percentile(times, 95)), "normal_control": "template paired with itself"}
        (args.out / "benchmark.json").write_text(json.dumps(benchmark, ensure_ascii=False, indent=2), encoding="utf-8")
        write_demo(archive, test, args.out)


if __name__ == "__main__":
    main()
