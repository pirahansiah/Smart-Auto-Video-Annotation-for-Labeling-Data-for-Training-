"""Smart auto-labeling — semi-automatic labeling with label propagation and active learning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from src.detector import Detection, ObjectDetector
from src.tracker import ObjectTracker
from src.utils import iter_frames, list_images, open_video

logger = logging.getLogger(__name__)


class AutoLabeler:
    """Semi-automatic labeling pipeline with human-in-the-loop and label propagation."""

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        tracker: ObjectTracker | None = None,
        confidence_threshold: float = 0.5,
    ):
        self.detector = detector or ObjectDetector()
        self.tracker = tracker or ObjectTracker(detector=self.detector)
        self.confidence_threshold = confidence_threshold

    def label_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        keyframe_interval: int = 1,
        confidence_threshold: float | None = None,
    ) -> dict:
        """Label all frames in a video with detection + tracking.

        Returns metadata dict with frame count, track count, and per-frame results.
        """
        conf = confidence_threshold or self.confidence_threshold
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tracker.reset()
        frame_results: list[dict] = []
        frame_idx = 0
        for idx, frame in iter_frames(video_path):
            if idx % keyframe_interval != 0:
                frame_idx += 1
                continue
            dets = self.detector.detect(frame)
            # Filter by confidence
            filtered = [d for d in dets if d.confidence >= conf]
            active = self.tracker.tracker.update(filtered)
            result = {
                "frame_idx": idx,
                "detections": [d.to_dict() for d in filtered],
                "tracks": [
                    {
                        "track_id": t.track_id,
                        "class_name": t.class_name,
                        "bbox": t.bbox.tolist(),
                        "confidence": t.confidence,
                    }
                    for t in active
                ],
            }
            frame_results.append(result)
            # Save frame image
            cv2.imwrite(str(out_dir / f"frame_{idx:06d}.jpg"), frame)
            frame_idx += 1

        meta = {
            "video": str(video_path),
            "total_frames": frame_idx,
            "labeled_frames": len(frame_results),
            "unique_tracks": len(self.tracker.tracker.tracks),
            "frames": frame_results,
        }
        logger.info("Labeled %d / %d frames from %s", len(frame_results), frame_idx, video_path)
        return meta

    def label_images(
        self,
        image_dir: str | Path,
        output_dir: str | Path | None = None,
        confidence_threshold: float | None = None,
    ) -> dict:
        """Label all images in a directory."""
        conf = confidence_threshold or self.confidence_threshold
        out_dir = Path(output_dir) if output_dir else Path(image_dir) / "labels"
        out_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, list[dict]] = {}
        for img_path in list_images(image_dir):
            dets = self.detector.detect(str(img_path))
            filtered = [d for d in dets if d.confidence >= conf]
            results[str(img_path)] = [d.to_dict() for d in filtered]
        logger.info("Labeled %d images in %s", len(results), image_dir)
        return {"image_dir": str(image_dir), "results": results}

    def interactive_label(
        self,
        source: str | np.ndarray,
        callback: Callable[[int, np.ndarray, list[Detection]], list[Detection]] | None = None,
    ) -> list[dict]:
        """Interactive labeling with optional human correction callback.

        The callback receives (frame_idx, frame, detections) and returns
        corrected detections.
        """
        self.tracker.reset()
        results: list[dict] = []
        for idx, frame in iter_frames(source):
            dets = self.detector.detect(frame)
            if callback is not None:
                dets = callback(idx, frame, dets)
            active = self.tracker.tracker.update(dets)
            results.append({
                "frame_idx": idx,
                "detections": [d.to_dict() for d in dets],
                "tracks": [
                    {"track_id": t.track_id, "class_name": t.class_name, "bbox": t.bbox.tolist()}
                    for t in active
                ],
            })
        return results

    def propagate_labels(
        self,
        keyframe_detections: dict[int, list[Detection]],
        video_path: str | Path,
        method: str = "tracking",
    ) -> dict[int, list[Detection]]:
        """Propagate labels from keyframes to all frames.

        method: "tracking" uses tracker to propagate, "interpolation" linearly
        interpolates bounding boxes between keyframes.
        """
        if method == "tracking":
            return self._propagate_via_tracking(keyframe_detections, video_path)
        elif method == "interpolation":
            return self._propagate_via_interpolation(keyframe_detections, video_path)
        else:
            raise ValueError(f"Unknown propagation method: {method}")

    def _propagate_via_tracking(
        self,
        keyframe_detections: dict[int, list[Detection]],
        video_path: str | Path,
    ) -> dict[int, list[Detection]]:
        """Use the tracker to propagate labels from keyframes."""
        self.tracker.reset()
        all_labels: dict[int, list[Detection]] = {}
        keyframe_set = set(keyframe_detections.keys())
        for idx, frame in iter_frames(video_path):
            if idx in keyframe_set:
                # Use keyframe detections to seed tracker
                dets = keyframe_detections[idx]
                self.tracker.tracker.update(dets)
                all_labels[idx] = dets
            else:
                dets = self.detector.detect(frame)
                active = self.tracker.tracker.update(dets)
                # Map active tracks back to detections
                propagated = [
                    Detection(
                        bbox=t.bbox.copy(),
                        confidence=t.confidence,
                        class_id=t.class_id,
                        class_name=t.class_name,
                    )
                    for t in active
                ]
                all_labels[idx] = propagated
        return all_labels

    def _propagate_via_interpolation(
        self,
        keyframe_detections: dict[int, list[Detection]],
        video_path: str | Path,
    ) -> dict[int, list[Detection]]:
        """Linearly interpolate bounding boxes between keyframes."""
        sorted_kf = sorted(keyframe_detections.keys())
        all_labels: dict[int, list[Detection]] = {}

        # Get total frame count
        cap = open_video(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        if len(sorted_kf) < 2:
            return keyframe_detections

        for i in range(len(sorted_kf) - 1):
            start_f = sorted_kf[i]
            end_f = sorted_kf[i + 1]
            start_dets = keyframe_detections[start_f]
            end_dets = keyframe_detections[end_f]

            for f in range(start_f, end_f + 1):
                t = (f - start_f) / max(end_f - start_f, 1)
                interp_dets = []
                for j in range(min(len(start_dets), len(end_dets))):
                    s = start_dets[j]
                    e = end_dets[j]
                    bbox = s.bbox * (1 - t) + e.bbox * t
                    interp_dets.append(Detection(
                        bbox=bbox,
                        confidence=s.confidence * (1 - t) + e.confidence * t,
                        class_id=s.class_id,
                        class_name=s.class_name,
                    ))
                all_labels[f] = interp_dets

        return all_labels

    def select_active_learning_samples(
        self,
        detections_per_frame: dict[int, list[Detection]],
        budget: int = 10,
    ) -> list[int]:
        """Select frames with lowest-confidence detections for human review."""
        scores: list[tuple[int, float]] = []
        for frame_idx, dets in detections_per_frame.items():
            if dets:
                min_conf = min(d.confidence for d in dets)
                scores.append((frame_idx, min_conf))
        scores.sort(key=lambda x: x[1])
        return [f for f, _ in scores[:budget]]
