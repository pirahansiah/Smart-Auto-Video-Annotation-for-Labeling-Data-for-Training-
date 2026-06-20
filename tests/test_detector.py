"""Tests for the ObjectDetector module."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import ObjectDetector, Detection


class TestDetection:
    def test_to_dict(self):
        d = Detection(
            bbox=np.array([10, 20, 100, 200]),
            confidence=0.85,
            class_id=0,
            class_name="person",
        )
        result = d.to_dict()
        assert result["bbox"] == [10, 20, 100, 200]
        assert result["confidence"] == 0.85
        assert result["class_id"] == 0
        assert result["class_name"] == "person"


class TestObjectDetector:
    def test_init_default(self):
        det = ObjectDetector()
        assert det.confidence == 0.25
        assert det.iou_threshold == 0.45

    def test_init_custom(self):
        det = ObjectDetector(
            model_path="yolo11n.pt",
            confidence=0.5,
            iou_threshold=0.6,
            classes=[0, 1, 2],
        )
        assert det.confidence == 0.5
        assert det.classes == [0, 1, 2]

    def test_detect_returns_list(self, sample_image):
        det = ObjectDetector()
        results = det.detect(sample_image)
        assert isinstance(results, list)

    def test_detect_batch(self, sample_image):
        det = ObjectDetector()
        results = det.detect_batch([sample_image, sample_image], batch_size=2)
        assert len(results) == 2
        assert all(isinstance(r, list) for r in results)

    def test_detect_with_path(self, temp_dir, sample_image):
        import cv2
        img_path = temp_dir / "test.jpg"
        cv2.imwrite(str(img_path), sample_image)
        det = ObjectDetector()
        results = det.detect(str(img_path))
        assert isinstance(results, list)
