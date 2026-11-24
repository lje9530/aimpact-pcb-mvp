"""Single-image PCB defect localization and classification."""

from __future__ import annotations

from threading import Lock
from time import perf_counter

import cv2

from .vision import roi_bounds


MODEL_VERSION = "pcb-yolo26n-v1"
YOLO_NAMES = ("open", "short", "mousebite", "spur", "spurious_copper", "pin_hole")
IMAGE_SIZE = 640


class YoloDetector:
    def __init__(self, weights, confidence=0.25, device="cpu"):
        from ultralytics import YOLO

        if not 0 < confidence <= 1:
            raise ValueError("Confidence threshold must be in (0, 1].")
        self.model = YOLO(str(weights), task="detect")
        if tuple(self.model.names.values()) != YOLO_NAMES:
            raise ValueError("A PCB-trained detector with the six DeepPCB class names is required.")
        self.confidence = confidence
        self.device = device
        self.lock = Lock()

    def __call__(self, test, _reference=None, roi=None):
        height, width = test.shape
        x1, y1, x2, y2 = roi_bounds(roi, width, height)
        image = cv2.cvtColor(test[y1:y2, x1:x2], cv2.COLOR_GRAY2BGR)
        with self.lock:
            start = perf_counter()
            detected = self.model.predict(image, imgsz=IMAGE_SIZE, conf=self.confidence,
                                          device=self.device, verbose=False, max_det=100)[0].boxes
            boxes = detected.xyxy.cpu().tolist()
            scores = detected.conf.cpu().tolist()
            classes = detected.cls.cpu().tolist()
            elapsed = round((perf_counter() - start) * 1000, 2)
        signals = [
            {"box": [round(left+x1, 2), round(top+y1, 2), round(right+x1, 2), round(bottom+y1, 2)],
             "type": int(kind)+1, "score": score}
            for (left, top, right, bottom), score, kind in zip(boxes, scores, classes, strict=True)
        ]
        signals.sort(key=lambda item: item["score"], reverse=True)
        return {"result": "NG" if signals else "OK", "types": sorted({item["type"] for item in signals}),
                "score": signals[0]["score"] if signals else None, "signals": signals, "ms": elapsed}
