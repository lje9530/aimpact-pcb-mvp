"""Reproduce DeepPCB splits, model training and held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pcb_splits import (
    archive_manifests, describe_splits, load_manifests, manifest_sha256,
    read_samples, write_manifests, zip_root,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_inspection.vision import CLASSES, MODEL_VERSION, candidates, predict, read_image


IOU_THRESHOLD = 0.19
VALIDATION_THRESHOLDS = (0.45, 0.60, 0.75, 0.90)
FOREST_SETTINGS = {"n_estimators": 36, "max_depth": 11, "min_samples_leaf": 7, "random_state": 42, "n_jobs": 1}
LOGISTIC_SETTINGS = {"max_iter": 500, "class_weight": "balanced"}


def iou_matrix(a, b):
    a, b = np.asarray(a, dtype=float).reshape(-1,4), np.asarray(b, dtype=float).reshape(-1,4)
    lo = np.maximum(a[:,None,:2], b[None,:,:2])
    hi = np.minimum(a[:,None,2:], b[None,:,2:])
    intersection = np.prod(np.maximum(hi-lo, 0), axis=-1)
    area_a = np.prod(np.maximum(a[:,2:]-a[:,:2],0),axis=-1)
    area_b = np.prod(np.maximum(b[:,2:]-b[:,:2],0),axis=-1)
    return intersection / np.maximum(area_a[:,None] + area_b[None,:] - intersection, 1e-12)


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
        if not record["boxes"] or not record["truth"]:
            continue
        truth = record["truth"]
        overlaps = iou_matrix(record["boxes"], [row[:4] for row in truth])
        for feature, row in zip(record["features"], overlaps, strict=True):
            best = int(np.argmax(row))
            predictors.append(feature)
            labels.append(truth[best][4] if row[best] >= IOU_THRESHOLD else 0)
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
            hits = [n for n, label in enumerate(truth)
                    if n not in tracked and label[4] == kind and overlaps[index, n] >= IOU_THRESHOLD]
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


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def train_models(archive, root, splits, out, provenance):
    train = train_rows(features_for(archive, read_samples(archive, root, splits["train"])))
    validation = features_for(archive, read_samples(archive, root, splits["validation"]), controls=True)
    models = {
        "logistic": make_pipeline(StandardScaler(), LogisticRegression(**LOGISTIC_SETTINGS)),
        "forest": RandomForestClassifier(**FOREST_SETTINGS),
    }
    scores = {}
    for name, model in models.items():
        model.fit(*train)
        scores[name] = max((evaluate(model, validation, threshold) for threshold in VALIDATION_THRESHOLDS),
                           key=lambda row: row["type_f1"])
        print(f"{name}: validation type F1 = {scores[name]['type_f1']:.3f}", flush=True)
    chosen = max(scores, key=lambda name: scores[name]["type_f1"])
    model_path = out / "model.joblib"
    joblib.dump({"model": models[chosen], "version": MODEL_VERSION,
                 "threshold": scores[chosen]["threshold"]}, model_path, compress=3)
    joblib.dump({"models": models, "provenance": provenance, "validation": scores,
                 "selected": chosen}, out / "candidate_models.joblib", compress=3)
    summary = {**provenance, "training_regions": dict(Counter(train[1])), "selected": chosen,
               "validation": scores, "weight_bytes": model_path.stat().st_size,
               "parameters": {"logistic": LOGISTIC_SETTINGS, "forest": FOREST_SETTINGS,
                              "thresholds": VALIDATION_THRESHOLDS, "iou": IOU_THRESHOLD},
               "environment": {"python": sys.version.split()[0], "opencv": cv2.__version__,
                               **{name: version(name) for name in ("numpy", "scikit-learn", "joblib", "Pillow")}}}
    save_json(out / "training.json", summary)
    print(f"Selected on validation: {chosen}, threshold = {scores[chosen]['threshold']}", flush=True)


def evaluate_models(archive, root, splits, out, provenance, demo=False):
    summary = json.loads((out / "training.json").read_text(encoding="utf-8"))
    if any(summary.get(key) != value for key, value in provenance.items()):
        raise ValueError("Training provenance differs. Use the same archive and manifests for evaluation.")
    artifact = joblib.load(out / "candidate_models.joblib")
    if artifact["provenance"] != provenance:
        raise ValueError("Candidate models were trained with a different archive or manifest.")
    if artifact["validation"] != summary["validation"] or artifact["selected"] != summary["selected"]:
        raise ValueError("Training summary does not match the saved model selection and thresholds.")
    models = artifact["models"]
    test = features_for(archive, read_samples(archive, root, splits["test"]), controls=True)
    scores = {}
    for name, model in models.items():
        validation = artifact["validation"][name]
        start = perf_counter()
        metrics = evaluate(model, test, validation["threshold"])
        scores[name] = {"validation": validation, "test": metrics,
                        "evaluation_ms_per_record": round((perf_counter() - start) * 1000 / len(test), 2)}
        print(f"{name}: test recall = {metrics['recall']:.3f}; test type F1 = {metrics['type_f1']:.3f}", flush=True)

    chosen = artifact["selected"]
    times = []
    for sample in read_samples(archive, root, splits["test"]):
        inspected = read_image(archive.read(sample["test"]))
        reference = read_image(archive.read(sample["reference"]))
        result = predict(models[chosen], inspected, reference, noise_threshold=scores[chosen]["validation"]["threshold"])
        times.append(result["ms"])
    benchmark = {**{key: value for key, value in summary.items() if key != "validation"}, "results": scores,
                 "cpu_ms_median": float(np.median(times)), "cpu_ms_p95": float(np.percentile(times, 95)),
                 "normal_control": "template paired with itself"}
    save_json(out / "benchmark.json", benchmark)
    if demo:
        write_demo(archive, test, out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path, help="Local ZIP from the original DeepPCB repository")
    parser.add_argument("--stage", choices=("split", "train", "evaluate", "all"), default="all")
    parser.add_argument("--splits-dir", type=Path, help="Default: data/splits for training, <out>/splits for split generation")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "reproduction")
    parser.add_argument("--demo", action="store_true", help="Export local study images for run.py --assets <out>")
    args = parser.parse_args()
    if args.demo and args.stage not in ("evaluate", "all"):
        parser.error("--demo requires --stage evaluate or --stage all")
    cv2.setNumThreads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.archive) as archive:
        root = zip_root(archive)
        if args.stage == "split":
            target = args.splits_dir or args.out / "splits"
            splits = archive_manifests(archive, root)
            write_manifests(splits, target)
            print(f"Wrote {target}\n{describe_splits(splits)}", flush=True)
            return

        splits = load_manifests(args.splits_dir or ROOT / "data" / "splits")
        with args.archive.open("rb") as source:
            archive_hash = hashlib.file_digest(source, "sha256").hexdigest()
        provenance = {"version": MODEL_VERSION, "archive_sha256": archive_hash,
                      "manifest_sha256": manifest_sha256(splits), "splits": describe_splits(splits)}
        print(provenance["splits"], flush=True)
        if args.stage in ("train", "all"):
            train_models(archive, root, splits, args.out, provenance)
        if args.stage in ("evaluate", "all"):
            evaluate_models(archive, root, splits, args.out, provenance, args.demo)


if __name__ == "__main__":
    main()
