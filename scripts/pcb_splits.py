"""Build and validate group-disjoint manifests for the pinned DeepPCB dataset."""

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import PurePosixPath


NAMES = ("train", "validation", "test")
COLUMNS = ("id", "board_group", "inspection_image", "reference_image", "annotation", "upstream_list")
VAL_GROUPS = {"group12000", "group12300"}
TEST_GROUPS = {"group12100", "group90100"}


def zip_root(archive):
    candidates = [name.removesuffix("/trainval.txt") for name in archive.namelist()
                  if name.endswith("/PCBData/trainval.txt")]
    if len(candidates) != 1:
        raise ValueError("The ZIP must contain one DeepPCB PCBData directory.")
    return candidates[0]


def archive_manifests(archive, root):
    splits = {name: [] for name in NAMES}
    for name in ("trainval", "test"):
        list_name = f"{name}.txt"
        for line in archive.read(f"{root}/{list_name}").decode().splitlines():
            if not line.strip():
                continue
            image, annotations = line.split()
            group = image.split("/")[0]
            if group in VAL_GROUPS:
                split = "validation"
            elif group in TEST_GROUPS:
                split = "test"
            elif name == "trainval":
                split = "train"
            else:
                continue
            stem = image.removesuffix(".jpg")
            splits[split].append(dict(zip(COLUMNS, (
                PurePosixPath(image).stem, group, f"{stem}_test.jpg", f"{stem}_temp.jpg",
                annotations, list_name,
            ), strict=True)))
    validate_manifests(splits)
    return splits


def validate_manifests(splits):
    seen_ids = set()
    group_splits = defaultdict(set)
    for name, records in splits.items():
        if not records:
            raise ValueError(f"Empty split: {name}")
        for record in records:
            if None in record or not all(record.get(column) for column in COLUMNS):
                raise ValueError(f"Invalid record in {name}.csv: {record}")
            sample_id, group = record["id"], record["board_group"]
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate sample ID: {sample_id}")
            seen_ids.add(sample_id)
            group_splits[group].add(name)
            for key in ("inspection_image", "reference_image", "annotation"):
                path = PurePosixPath(record[key])
                if not path.parts or path.parts[0] != group or ".." in path.parts:
                    raise ValueError(f"{sample_id} has an invalid path in {key}")
    overlap = sorted(group for group, names in group_splits.items() if len(names) > 1)
    if overlap:
        raise ValueError(f"Board groups appear in multiple splits: {', '.join(overlap)}")


def write_manifests(splits, destination):
    destination.mkdir(parents=True, exist_ok=True)
    for name, records in splits.items():
        with (destination / f"{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(records)


def load_manifests(directory):
    splits = {}
    for name in NAMES:
        with (directory / f"{name}.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != list(COLUMNS):
                raise ValueError(f"Unexpected columns in {name}.csv: {reader.fieldnames}")
            splits[name] = list(reader)
    validate_manifests(splits)
    return splits


def manifest_sha256(splits):
    encoded = json.dumps(splits, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_samples(archive, root, records):
    for row in records:
        annotation = archive.read(f"{root}/{row['annotation']}").decode()
        yield {
            "id": row["id"], "group": row["board_group"],
            "test": f"{root}/{row['inspection_image']}",
            "reference": f"{root}/{row['reference_image']}",
            "truth": [list(map(int, line.split())) for line in annotation.splitlines() if line.strip()],
        }


def describe_splits(splits):
    return {name: {"pairs": len(records), "groups": sorted({row["board_group"] for row in records})}
            for name, records in splits.items()}
