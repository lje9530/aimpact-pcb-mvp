"""Prepare DeepPCB and fine-tune a single-image PCB object detector."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from pcb_splits import load_manifests, manifest_sha256, read_samples, zip_root


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from pcb_inspection.detector import YOLO_NAMES


SPLIT_NAMES = {"train": "train", "validation": "val", "test": "test"}


def encode_labels(truth, width, height):
    labels = []
    for left, top, right, bottom, kind in truth:
        if not (0 <= left < right <= width and 0 <= top < bottom <= height and 1 <= kind <= len(YOLO_NAMES)):
            raise ValueError(f"Invalid DeepPCB box {left, top, right, bottom, kind} for {width}x{height}")
        x = (left + right) / (2 * width)
        y = (top + bottom) / (2 * height)
        labels.append(f"{kind-1} {x:.9f} {y:.9f} {(right-left)/width:.9f} {(bottom-top)/height:.9f}\n")
    return "".join(labels)


def prepare(archive_path, splits_path, destination):
    splits = load_manifests(splits_path)
    with archive_path.open("rb") as file:
        fingerprint = {"archive_sha256": hashlib.file_digest(file, "sha256").hexdigest(),
                       "manifest_sha256": manifest_sha256(splits), "template_stride": 4}
    if (destination / "data.yaml").is_file():
        existing = json.loads((destination / "dataset.json").read_text(encoding="utf-8"))
        if any(existing.get(key) != value for key, value in fingerprint.items()):
            raise ValueError("Prepared dataset has a different fingerprint. Choose a new --dataset path.")
        return destination / "data.yaml"
    destination.mkdir(parents=True, exist_ok=True)
    counts = {}
    with zipfile.ZipFile(archive_path) as archive:
        root = zip_root(archive)
        for split, folder in SPLIT_NAMES.items():
            images, labels = destination / "images" / folder, destination / "labels" / folder
            images.mkdir(parents=True, exist_ok=True)
            labels.mkdir(parents=True, exist_ok=True)
            controls = set()
            for index, row in enumerate(read_samples(archive, root, splits[split])):
                content = archive.read(row["test"])
                with Image.open(BytesIO(content)) as image:
                    normalized = encode_labels(row["truth"], image.width, image.height)
                (images / f"{row['id']}.jpg").write_bytes(content)
                (labels / f"{row['id']}.txt").write_text(normalized, encoding="utf-8")
                if index % fingerprint["template_stride"] == 0:
                    template = archive.read(row["reference"])
                    digest = hashlib.sha256(template).digest()
                    if digest in controls:
                        continue
                    controls.add(digest)
                    (images / f"{row['id']}_template.jpg").write_bytes(template)
                    (labels / f"{row['id']}_template.txt").write_text("", encoding="utf-8")
            counts[split] = {"inspection_images": len(splits[split]), "template_controls": len(controls)}
    (destination / "dataset.json").write_text(json.dumps({**fingerprint, "splits": counts}, indent=2) + "\n", encoding="utf-8")
    descriptor = {"path": destination.resolve().as_posix(), "train": "images/train", "val": "images/val",
                  "test": "images/test", "names": dict(enumerate(YOLO_NAMES))}
    # JSON is a YAML subset and preserves paths with spaces without shell quoting.
    dataset = destination / "data.yaml"
    dataset.write_text(json.dumps(descriptor, indent=2) + "\n", encoding="utf-8")
    print(f"YOLO dataset: {counts}", flush=True)
    return dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--weights", default="yolo26n.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs" / "yolo_dataset")
    parser.add_argument("--project", type=Path, default=ROOT / "outputs" / "detector")
    parser.add_argument("--name", default="pcb_yolo26n")
    parser.add_argument("--splits-dir", type=Path, default=ROOT / "data" / "splits")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    dataset = prepare(args.archive, args.splits_dir, args.dataset.resolve())
    if args.prepare_only:
        return

    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / "outputs" / "ultralytics"))
    os.environ.setdefault("YOLO_OFFLINE", "true")
    from ultralytics import YOLO

    model = YOLO(args.weights)
    model.train(data=str(dataset), project=str(args.project.resolve()), name=args.name, exist_ok=False,
                epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, workers=0,
                cache="ram", seed=42, deterministic=True, amp=False, plots=False,
                mosaic=0.35, close_mosaic=5, mixup=0, degrees=0, fliplr=0.5, flipud=0.5,
                translate=0.08, scale=0.25, hsv_h=0, hsv_s=0, hsv_v=0.08)
    print(f"Selected weights: {Path(model.trainer.save_dir) / 'weights' / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
