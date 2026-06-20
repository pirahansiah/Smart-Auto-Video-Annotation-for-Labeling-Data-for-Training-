"""Object detection module — YOLOv11-based detector with batch/video/webcam support."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils import iter_frames, list_images, open_video, resize_to_max

logger = logging.getLogger(__name__)

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


class Detection:
    """Single detection result."""

    __slots__ = ("bbox", "confidence", "class_id", "class_name", "mask")

    def __init__(
        self,
        bbox: np.ndarray,
        confidence: float,
        class_id: int,
        class_name: str = "",
        mask: np.ndarray | None = None,
    ):
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.confidence = confidence
        self.class_id = class_id
        self.class_name = class_name
        self.mask = mask

    def to_dict(self) -> dict:
        return {
            "bbox": self.bbox.tolist(),
            "confidence": float(self.confidence),
            "class_id": int(self.class_id),
            "class_name": self.class_name,
        }


class ObjectDetector:
    """YOLOv11-based object detector with configurable thresholds and class filtering."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "",
        classes: list[int] | None = None,
    ):
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = device
        self.classes = classes
        self.class_names = self.model.names

    def _parse_results(self, result) -> list[Detection]:
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        boxes = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clss = result.boxes.cls.cpu().numpy().astype(int)
        masks = result.masks.data.cpu().numpy() if result.masks is not None else None
        for i in range(len(boxes)):
            detections.append(Detection(
                bbox=boxes[i],
                confidence=float(confs[i]),
                class_id=int(clss[i]),
                class_name=self.class_names.get(int(clss[i]), ""),
                mask=masks[i] if masks is not None else None,
            ))
        return detections

    def detect(
        self,
        source: np.ndarray | str | Path,
    ) -> list[Detection]:
        """Run detection on a single image (array or path)."""
        kwargs = dict(
            conf=self.confidence,
            iou=self.iou_threshold,
            verbose=False,
        )
        if self.device:
            kwargs["device"] = self.device
        if self.classes is not None:
            kwargs["classes"] = self.classes
        results = self.model.predict(source=source, **kwargs)
        return self._parse_results(results[0])

    def detect_batch(
        self,
        sources: list[np.ndarray | str | Path],
        batch_size: int = 16,
    ) -> list[list[Detection]]:
        """Run detection on multiple images in batches."""
        all_detections: list[list[Detection]] = []
        for i in range(0, len(sources), batch_size):
            batch = sources[i : i + batch_size]
            kwargs = dict(
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False,
            )
            if self.device:
                kwargs["device"] = self.device
            if self.classes is not None:
                kwargs["classes"] = self.classes
            results = self.model.predict(source=batch, **kwargs)
            for r in results:
                all_detections.append(self._parse_results(r))
        return all_detections

    def detect_video(
        self,
        video_path: str | Path,
    ) -> Iterator[tuple[int, list[Detection]]]:
        """Yield (frame_index, detections) for every frame in a video."""
        for idx, frame in iter_frames(video_path):
            dets = self.detect(frame)
            yield idx, dets

    def detect_webcam(
        self,
        camera_id: int = 0,
        max_frames: int | None = None,
    ) -> Iterator[tuple[int, np.ndarray, list[Detection]]]:
        """Yield (frame_index, frame, detections) from a live webcam."""
        cap = open_video(camera_id)
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                dets = self.detect(frame)
                yield idx, frame, dets
                idx += 1
                if max_frames is not None and idx >= max_frames:
                    break
        finally:
            cap.release()

    def detect_images(
        self,
        directory: str | Path,
        extensions: set[str] | None = None,
    ) -> Iterator[tuple[Path, list[Detection]]]:
        """Yield (image_path, detections) for all images in a directory."""
        for img_path in list_images(directory):
            dets = self.detect(str(img_path))
            yield img_path, dets
