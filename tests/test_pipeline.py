"""Tests for the AnnotationPipeline module."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.pipeline import AnnotationPipeline


@pytest.fixture
def pipeline():
    return AnnotationPipeline(
        model_path="yolo11n.pt",
        confidence=0.25,
        export_formats=["yolo", "coco"],
    )


class TestPipeline:
    def test_init(self, pipeline):
        assert pipeline.detector is not None
        assert pipeline.tracker is not None
        assert "yolo" in pipeline.export_formats
        assert "coco" in pipeline.export_formats

    def test_process_video(self, pipeline, sample_video, temp_dir):
        out = temp_dir / "output"
        result = pipeline.process_video(sample_video, out)
        assert "total_frames" in result
        assert "labeled_frames" in result
        assert "unique_tracks" in result
        assert result["total_frames"] == 5

    def test_process_folder(self, pipeline, temp_dir):
        # Create test images
        img_dir = temp_dir / "images"
        img_dir.mkdir()
        for i in range(3):
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"img_{i:03d}.jpg"), img)

        out = temp_dir / "output"
        result = pipeline.process_folder(img_dir, out)
        assert "results" in result
        assert len(result["results"]) == 3

    def test_generate_report(self, pipeline):
        results = [
            {
                "total_frames": 100,
                "labeled_frames": 100,
                "unique_tracks": 5,
                "frames": [
                    {
                        "detections": [
                            {"class_name": "person", "confidence": 0.9},
                            {"class_name": "car", "confidence": 0.7},
                        ]
                    }
                ],
            }
        ]
        report = pipeline.generate_report(results)
        assert report["summary"]["total_videos"] == 1
        assert report["summary"]["total_detections"] == 2
        assert "class_distribution" in report
        assert "confidence_stats" in report

    def test_generate_report_to_file(self, pipeline, temp_dir):
        results = [{"total_frames": 10, "labeled_frames": 10, "unique_tracks": 2, "frames": []}]
        out_path = temp_dir / "report.json"
        report = pipeline.generate_report(results, output_path=out_path)
        assert out_path.exists()
        data = json.loads(out_path.read_text())
        assert data["summary"]["total_videos"] == 1
