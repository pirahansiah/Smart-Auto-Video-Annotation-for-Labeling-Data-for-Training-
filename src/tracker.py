"""Multi-object tracking — ByteTrack with trajectory smoothing."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d

from src.detector import Detection, ObjectDetector
from src.utils import COLORS, draw_boxes, iter_frames, open_video

logger = logging.getLogger(__name__)


@dataclass
class TrackState:
    track_id: int
    class_id: int
    class_name: str
    bbox: np.ndarray
    confidence: float
    age: int = 0
    hits: int = 1
    time_since_update: int = 0
    history: list[np.ndarray] = field(default_factory=list)

    def update(self, bbox: np.ndarray, confidence: float) -> None:
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        self.history.append(bbox.copy())

    def predict(self) -> None:
        self.age += 1
        self.time_since_update += 1


class SimpleTracker:
    """Lightweight IoU-based tracker inspired by ByteTrack.

    Matches high-confidence detections first, then remaining detections.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
        low_conf_threshold: float = 0.1,
    ):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.low_conf_threshold = low_conf_threshold
        self.tracks: dict[int, TrackState] = {}
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()
        self._next_id = 1

    @staticmethod
    def _compute_iou_matrix(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> np.ndarray:
        """Compute pairwise IoU between two sets of boxes [N,4] and [M,4]."""
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
        xa = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        ya = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        xb = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        yb = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-6)

    def _match(
        self, detections: list[Detection], high_mask: np.ndarray
    ) -> tuple[list[tuple[int, int]], list[int], list[int]]:
        """Match detections to existing tracks using IoU.

        Returns (matched_pairs, unmatched_detection_indices, unmatched_track_ids).
        """
        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[tid].bbox for tid in track_ids]) if track_ids else np.zeros((0, 4))
        det_boxes = np.array([d.bbox for d in detections]) if detections else np.zeros((0, 4))

        if len(track_ids) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), track_ids

        iou_mat = self._compute_iou_matrix(det_boxes, track_boxes)

        matched: list[tuple[int, int]] = []
        used_det: set[int] = set()
        used_track: set[int] = set()

        # Phase 1: high-confidence detections
        for det_idx in np.where(high_mask)[0]:
            if det_idx in used_det:
                continue
            best_iou = 0.0
            best_tid_idx = -1
            for tid_idx in range(len(track_ids)):
                if tid_idx in used_track:
                    continue
                if iou_mat[det_idx, tid_idx] > best_iou:
                    best_iou = iou_mat[det_idx, tid_idx]
                    best_tid_idx = tid_idx
            if best_tid_idx >= 0 and best_iou >= self.iou_threshold:
                matched.append((det_idx, track_ids[best_tid_idx]))
                used_det.add(det_idx)
                used_track.add(best_tid_idx)

        # Phase 2: low-confidence detections
        for det_idx in range(len(detections)):
            if det_idx in used_det:
                continue
            best_iou = 0.0
            best_tid_idx = -1
            for tid_idx in range(len(track_ids)):
                if tid_idx in used_track:
                    continue
                if iou_mat[det_idx, tid_idx] > best_iou:
                    best_iou = iou_mat[det_idx, tid_idx]
                    best_tid_idx = tid_idx
            if best_tid_idx >= 0 and best_iou >= self.iou_threshold:
                matched.append((det_idx, track_ids[best_tid_idx]))
                used_det.add(det_idx)
                used_track.add(best_tid_idx)

        unmatched_det = [i for i in range(len(detections)) if i not in used_det]
        unmatched_tracks = [track_ids[i] for i in range(len(track_ids)) if i not in used_track]
        return matched, unmatched_det, unmatched_tracks

    def update(self, detections: list[Detection]) -> list[TrackState]:
        """Update tracks with new detections; return active tracks."""
        # Predict all existing tracks
        for track in self.tracks.values():
            track.predict()

        confs = np.array([d.confidence for d in detections]) if detections else np.array([])
        high_mask = confs >= 0.5 if len(confs) > 0 else np.array([], dtype=bool)

        matched, unmatched_det, unmatched_track = self._match(detections, high_mask)

        # Update matched
        for det_idx, track_id in matched:
            self.tracks[track_id].update(detections[det_idx].bbox, detections[det_idx].confidence)

        # Create new tracks for unmatched high-confidence detections
        for det_idx in unmatched_det:
            d = detections[det_idx]
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = TrackState(
                track_id=tid,
                class_id=d.class_id,
                class_name=d.class_name,
                bbox=d.bbox.copy(),
                confidence=d.confidence,
                history=[d.bbox.copy()],
            )

        # Remove stale tracks
        stale = [tid for tid, t in self.tracks.items() if t.time_since_update > self.max_age]
        for tid in stale:
            del self.tracks[tid]

        # Return active tracks
        return [
            t for t in self.tracks.values()
            if t.hits >= self.min_hits or t.time_since_update == 0
        ]


class ObjectTracker:
    """High-level tracker that wraps SimpleTracker and adds trajectory utilities."""

    def __init__(
        self,
        detector: ObjectDetector | None = None,
        iou_threshold: float = 0.3,
        max_age: int = 30,
        min_hits: int = 3,
    ):
        self.detector = detector or ObjectDetector()
        self.tracker = SimpleTracker(
            iou_threshold=iou_threshold,
            max_age=max_age,
            min_hits=min_hits,
        )

    def reset(self) -> None:
        self.tracker.reset()

    def track(
        self,
        source: str | np.ndarray,
    ) -> list[dict]:
        """Process a video/image source and return a list of frame results.

        Each result: {"frame_idx": int, "tracks": list[TrackState]}.
        """
        self.reset()
        results: list[dict] = []
        for idx, frame in iter_frames(source):
            dets = self.detector.detect(frame)
            active = self.tracker.update(dets)
            results.append({"frame_idx": idx, "tracks": active})
        return results

    def get_trajectories(
        self,
        min_length: int = 5,
        smooth: bool = True,
        sigma: float = 2.0,
    ) -> dict[int, list[np.ndarray]]:
        """Return trajectories per track_id, optionally smoothed."""
        raw: dict[int, list[np.ndarray]] = defaultdict(list)
        for track in self.tracker.tracks.values():
            if len(track.history) >= min_length:
                raw[track.track_id] = [b.copy() for b in track.history]
        if not smooth:
            return dict(raw)
        smoothed: dict[int, list[np.ndarray]] = {}
        for tid, hist in raw.items():
            arr = np.array(hist)  # (T, 4)
            smoothed_arr = uniform_filter1d(arr, size=max(3, int(sigma * 2)), axis=0)
            smoothed[tid] = [smoothed_arr[i] for i in range(len(smoothed_arr))]
        return smoothed

    def visualize_tracks(
        self,
        video_path: str | np.ndarray,
        output_path: str | None = None,
        show: bool = False,
    ) -> list[np.ndarray]:
        """Track objects and yield annotated frames."""
        cap = open_video(video_path)
        annotated: list[np.ndarray] = []
        self.reset()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            dets = self.detector.detect(frame)
            active = self.tracker.update(dets)
            boxes = np.array([t.bbox for t in active]) if active else np.zeros((0, 4))
            ids = [t.track_id for t in active]
            labels = [t.class_name for t in active]
            scores = [t.confidence for t in active]
            vis = draw_boxes(frame, boxes, labels=labels, scores=scores, track_ids=ids)
            annotated.append(vis)
            if show:
                cv2.imshow("Tracking", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        cap.release()
        if show:
            cv2.destroyAllWindows()
        return annotated
