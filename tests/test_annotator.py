"""Tests for the AnnotationGenerator module."""

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.annotator import AnnotationGenerator
from src.detector import Detection


@pytest.fixture
def generator():
    return AnnotationGenerator(class_names=["person", "car", "dog"])


@pytest.fixture
def sample_annotations(temp_dir):
    """Create a sample image and annotation dict."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img_path = str(temp_dir / "image1.jpg")
    cv2.imwrite(img_path, img)
    return {
        img_path: [
            Detection(bbox=np.array([100, 100, 300, 300]), confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=np.array([200, 200, 400, 400]), confidence=0.7, class_id=1, class_name="car"),
        ]
    }


class TestCOCO:
    def test_generate_coco(self, generator, sample_annotations, temp_dir):
        out = temp_dir / "coco" / "annotations.json"
        result = generator.generate_coco(sample_annotations, out)
        assert result.exists()
        data = json.loads(result.read_text())
        assert "images" in data
        assert "annotations" in data
        assert "categories" in data
        assert len(data["images"]) == 1
        assert len(data["annotations"]) == 2
        assert len(data["categories"]) == 2

    def test_coco_empty(self, generator, temp_dir):
        out = temp_dir / "empty.json"
        result = generator.generate_coco({"empty.jpg": []}, out)
        assert result.exists()
        data = json.loads(result.read_text())
        assert len(data["annotations"]) == 0


class TestYOLO:
    def test_generate_yolo(self, generator, sample_annotations, temp_dir):
        out = temp_dir / "yolo"
        results = generator.generate_yolo(sample_annotations, out)
        assert len(results) == 1
        txt = results[0].read_text()
        lines = txt.strip().split("\n")
        assert len(lines) == 2
        parts = lines[0].split()
        assert len(parts) == 4  # class_id cx cy w h
        assert parts[0] == "0"  # class_id for person


class TestVOC:
    def test_generate_voc(self, generator, sample_annotations, temp_dir):
        out = temp_dir / "voc"
        results = generator.generate_voc(sample_annotations, out)
        assert len(results) == 1
        tree = ET.parse(str(results[0]))
        root = tree.getroot()
        assert root.tag == "annotation"
        objects = root.findall("object")
        assert len(objects) == 2
        assert objects[0].find("name").text == "person"


class TestVisualize:
    def test_visualize(self, generator, sample_image, sample_detections):
        vis = generator.visualize_annotations(sample_image, sample_detections)
        assert vis.shape == sample_image.shape
        assert vis.dtype == np.uint8
