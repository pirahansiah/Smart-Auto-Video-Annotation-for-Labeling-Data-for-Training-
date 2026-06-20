"""Pytest fixtures for Smart Auto Video Annotation tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.detector import Detection, ObjectDetector
from src.tracker import ObjectTracker
from src.annotator import AnnotationGenerator


@pytest.fixture
def sample_image() -> np.ndarray:
    """A blank 640x480 image with a drawn rectangle."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (255, 255, 255), -1)
    return img


@pytest.fixture
def sample_detections() -> list[Detection]:
    return [
        Detection(bbox=np.array([100, 100, 300, 300]), confidence=0.9, class_id=0, class_name="person"),
        Detection(bbox=np.array([200, 200, 400, 400]), confidence=0.7, class_id=2, class_name="car"),
    ]


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_video(temp_dir: Path) -> Path:
    """Create a short synthetic video with 5 frames."""
    path = temp_dir / "test_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 10.0, (640, 480))
    for i in range(5):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        x = 50 + i * 40
        cv2.rectangle(frame, (x, 100), (x + 150, 350), (255, 255, 255), -1)
        writer.write(frame)
    writer.release()
    return path
