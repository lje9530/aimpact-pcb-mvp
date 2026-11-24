from __future__ import annotations

from io import BytesIO
from time import perf_counter

import cv2
import numpy as np
from PIL import Image, ImageOps


CLASSES = {
    0: "정상", 1: "단선", 2: "단락", 3: "마우스바이트",
    4: "돌기", 5: "잉여 구리", 6: "핀홀", 7: "미분류",
}
MODEL_VERSION = "pair-rf-v1"
IMAGE_SIZE = 640


def read_image(content: bytes) -> np.ndarray:
    with Image.open(BytesIO(content)) as image:
        if image.width * image.height > 25_000_000:
            raise ValueError("이미지는 2,500만 픽셀 이하여야 합니다.")
        return np.array(ImageOps.exif_transpose(image).convert("L"))


def png(gray: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        raise ValueError("이미지를 저장할 수 없습니다.")
    return encoded.tobytes()


def roi_bounds(roi, width, height):
    if roi is None:
        roi = (0, 0, 1, 1)
    if len(roi) != 4 or not all(0 <= value <= 1 for value in roi):
        raise ValueError("ROI는 0~1 범위의 x, y, 너비, 높이 값 네 개가 필요합니다.")
    x, y, w, h = roi
    x1, y1, x2, y2 = round(x * width), round(y * height), round((x+w)*width), round((y+h)*height)
    if x2 > width or y2 > height or x2-x1 < 16 or y2-y1 < 16:
        raise ValueError("ROI는 이미지 안에 있어야 하며 가로·세로가 각각 16px 이상이어야 합니다.")
    return x1, y1, x2, y2


def candidates(test: np.ndarray, reference: np.ndarray, roi=None):
    """Extract paired difference regions and compact shape/context features."""
    if test.shape != reference.shape:
        raise ValueError("검사 영상과 정렬된 기준 영상의 해상도가 같아야 합니다.")
    height, width = test.shape
    x0, y0, x1, y1 = roi_bounds(roi, width, height)
    rw, rh = x1-x0, y1-y0
    scale = min(1, IMAGE_SIZE / max(rw, rh))
    size = (max(1, round(rw * scale)), max(1, round(rh * scale)))

    def binary(image):
        cropped = image[y0:y1, x0:x1]
        gray = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA) if scale < 1 else cropped
        return cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    test_bin, ref_bin = binary(test), binary(reference)
    diff = test_bin.astype(np.int8) - ref_bin.astype(np.int8)
    region_mask = cv2.morphologyEx(np.abs(diff).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask)
    features, boxes = [], []
    for index in range(1, count):
        x, y, w, h, area = (int(value) for value in stats[index])
        if area < 7:
            continue
        mask = labels[y:y+h, x:x+w] == index
        delta = diff[y:y+h, x:x+w][mask]
        density = []
        for pad in (2, 7, 14):
            region = np.s_[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
            density.extend((test_bin[region].mean(), ref_bin[region].mean()))
        features.append([
            w, h, min(w, h), max(w, h), area, area/(w*h),
            (delta > 0).mean(), (delta < 0).mean(), (delta == 0).mean(), *density,
        ])
        boxes.append([
            round(x0 + x * rw / size[0]), round(y0 + y * rh / size[1]),
            round(x0 + (x+w) * rw / size[0]), round(y0 + (y+h) * rh / size[1]),
        ])
    return boxes, features


def predict(model, test, reference, roi=None, noise_threshold=0.5):
    if reference is None:
        return {"result": "WAIT", "types": [], "score": None, "signals": [], "ms": None}
    start = perf_counter()
    boxes, features = candidates(test, reference, roi)
    signals, highest_defect_score = [], None
    if features:
        probabilities = model.predict_proba(features)
        class_ids = np.asarray(model.classes_)
        defect_columns = class_ids > 0
        for box, row in zip(boxes, probabilities, strict=True):
            defect_score = float(row[defect_columns].sum())
            highest_defect_score = max(highest_defect_score or 0, defect_score)
            if defect_score < noise_threshold:
                continue
            top = int(row[defect_columns].argmax())
            signals.append({"box": box, "type": int(class_ids[defect_columns][top]),
                            "score": round(float(row[defect_columns][top]), 4),
                            "defect_score": round(defect_score, 4)})
    signals.sort(key=lambda signal: signal["score"], reverse=True)
    elapsed = round((perf_counter() - start) * 1000, 2)
    score = signals[0]["score"] if signals else None
    if not signals and highest_defect_score is not None:
        score = round(1 - highest_defect_score, 4)
    return {"result": "NG" if signals else "OK", "types": sorted({s["type"] for s in signals}),
            "score": score, "signals": signals, "ms": elapsed}
