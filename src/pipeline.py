"""End-to-end annotation pipeline — video/folder processing, export, reporting."""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from src.annotator import AnnotationGenerator
from src.auto_labeler import AutoLabeler
from src.detector import ObjectDetector
from src.quality import QualityAssessor
from src.tracker import ObjectTracker
from src.utils import list_images, list_videos

logger = logging.getLogger(__name__)


class AnnotationPipeline:
    """Complete workflow from video/images to exported annotations."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        export_formats: list[str] | None = None,
        device: str = "",
    ):
        self.detector = ObjectDetector(
            model_path=model_path,
            confidence=confidence,
            iou_threshold=iou_threshold,
            device=device,
        )
        self.tracker = ObjectTracker(detector=self.detector)
        self.labeler = AutoLabeler(detector=self.detector, tracker=self.tracker)
        self.annotator = AnnotationGenerator(class_names=list(self.detector.class_names.values()))
        self.quality = QualityAssessor()
        self.export_formats = export_formats or ["yolo", "coco"]

    def process_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        keyframe_interval: int = 1,
    ) -> dict:
        """Run full pipeline on a single video."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        logger.info("Processing video: %s", video_path)
        meta = self.labeler.label_video(
            video_path=video_path,
            output_dir=out / "frames",
            keyframe_interval=keyframe_interval,
        )

        # Collect detections per image for export
        annotations: dict[str, list] = {}
        for frame_meta in meta["frames"]:
            fname = f"frame_{frame_meta['frame_idx']:06d}.jpg"
            fpath = out / "frames" / fname
            # Re-parse detections from stored dicts
            from src.detector import Detection
            dets = [
                Detection(
                    bbox=np.array(fd["bbox"]),
                    confidence=fd["confidence"],
                    class_id=fd["class_id"],
                    class_name=fd["class_name"],
                )
                for fd in frame_meta["detections"]
            ]
            annotations[str(fpath)] = dets

        # Export
        self._export(annotations, out)
        return meta

    def process_folder(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
    ) -> dict:
        """Process all images in a folder."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        images = list_images(input_dir)
        logger.info("Processing %d images from %s", len(images), input_dir)

        all_results: dict[str, list[dict]] = {}
        annotations: dict[str, list] = {}
        for img_path in images:
            dets = self.detector.detect(str(img_path))
            all_results[str(img_path)] = [d.to_dict() for d in dets]
            annotations[str(img_path)] = dets

        self._export(annotations, out)
        return {"image_dir": str(input_dir), "results": all_results}

    def _export(self, annotations: dict, out_dir: Path) -> None:
        """Export annotations in all configured formats."""
        if "coco" in self.export_formats:
            self.annotator.generate_coco(annotations, out_dir / "coco" / "annotations.json")
        if "yolo" in self.export_formats:
            self.annotator.generate_yolo(annotations, out_dir / "yolo")
        if "voc" in self.export_formats:
            self.annotator.generate_voc(annotations, out_dir / "voc")

    def export(
        self,
        annotations: dict[str, list],
        output_dir: str | Path,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Manually export annotations to specified formats."""
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fmts = formats or self.export_formats
        results: dict[str, Path] = {}
        if "coco" in fmts:
            results["coco"] = self.annotator.generate_coco(annotations, out / "annotations.json")
        if "yolo" in fmts:
            paths = self.annotator.generate_yolo(annotations, out / "yolo")
            results["yolo"] = Path(out / "yolo")
        if "voc" in fmts:
            paths = self.annotator.generate_voc(annotations, out / "voc")
            results["voc"] = Path(out / "voc")
        return results

    def generate_report(
        self,
        pipeline_results: Sequence[dict],
        output_path: str | Path | None = None,
    ) -> dict:
        """Generate a summary report from pipeline results."""
        total_frames = sum(r.get("total_frames", 0) for r in pipeline_results)
        total_labeled = sum(r.get("labeled_frames", 0) for r in pipeline_results)
        total_tracks = sum(r.get("unique_tracks", 0) for r in pipeline_results)

        all_dets = []
        for r in pipeline_results:
            for fr in r.get("frames", []):
                all_dets.extend(fr.get("detections", []))

        class_counts: dict[str, int] = defaultdict(int)
        confidences: list[float] = []
        for d in all_dets:
            class_counts[d["class_name"]] += 1
            confidences.append(d["confidence"])

        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": {
                "total_videos": len(pipeline_results),
                "total_frames": total_frames,
                "labeled_frames": total_labeled,
                "unique_tracks": total_tracks,
                "total_detections": len(all_dets),
            },
            "class_distribution": dict(class_counts),
            "confidence_stats": {
                "mean": float(np.mean(confidences)) if confidences else 0,
                "std": float(np.std(confidences)) if confidences else 0,
                "min": float(np.min(confidences)) if confidences else 0,
                "max": float(np.max(confidences)) if confidences else 0,
                "median": float(np.median(confidences)) if confidences else 0,
            },
        }

        if output_path:
            Path(output_path).write_text(json.dumps(report, indent=2))
            logger.info("Report written to %s", output_path)

        return report
