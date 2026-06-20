"""Tests for the ObjectTracker module."""

from __future__ import annotations

import numpy as np
import pytest

from src.detector import Detection
from src.tracker import SimpleTracker, ObjectTracker


class TestSimpleTracker:
    def test_empty_update(self):
        tracker = SimpleTracker()
        result = tracker.update([])
        assert result == []

    def test_create_track(self):
        tracker = SimpleTracker(min_hits=1)
        dets = [Detection(bbox=np.array([100, 100, 200, 200]), confidence=0.9, class_id=0, class_name="person")]
        result = tracker.update(dets)
        assert len(result) == 1
        assert result[0].track_id == 1

    def test_multiple_detections(self):
        tracker = SimpleTracker(min_hits=1)
        dets = [
            Detection(bbox=np.array([100, 100, 200, 200]), confidence=0.9, class_id=0, class_name="person"),
            Detection(bbox=np.array([300, 300, 400, 400]), confidence=0.8, class_id=2, class_name="car"),
        ]
        result = tracker.update(dets)
        assert len(result) == 2

    def test_track_persistence(self):
        tracker = SimpleTracker(min_hits=1)
        dets = [Detection(bbox=np.array([100, 100, 200, 200]), confidence=0.9, class_id=0, class_name="person")]
        tracker.update(dets)
        # Same box, should match existing track
        result = tracker.update(dets)
        assert len(result) == 1
        assert result[0].hits == 2

    def test_track_removal(self):
        tracker = SimpleTracker(max_age=2, min_hits=1)
        dets = [Detection(bbox=np.array([100, 100, 200, 200]), confidence=0.9, class_id=0, class_name="person")]
        tracker.update(dets)
        tracker.update([])  # No detections
        tracker.update([])
        result = tracker.update([])
        # Track should be removed after max_age
        assert len(result) == 0

    def test_iou_computation(self):
        box_a = np.array([0, 0, 10, 10])
        box_b = np.array([5, 5, 15, 15])
        iou = SimpleTracker._compute_iou_matrix(
            box_a.reshape(1, 4), box_b.reshape(1, 4)
        )
        # Intersection = 25, Union = 175, IoU = 25/175 ≈ 0.143
        assert abs(iou[0, 0] - 25 / 175) < 1e-5


class TestObjectTracker:
    def test_init(self):
        tracker = ObjectTracker()
        assert tracker.detector is not None
        assert tracker.tracker is not None

    def test_get_trajectories(self):
        tracker = ObjectTracker()
        # Manually add history
        from src.tracker import TrackState
        t = TrackState(
            track_id=1, class_id=0, class_name="person",
            bbox=np.array([100, 100, 200, 200]), confidence=0.9,
        )
        for i in range(10):
            t.history.append(np.array([100 + i, 100 + i, 200 + i, 200 + i]))
        tracker.tracker.tracks[1] = t
        trajectories = tracker.get_trajectories(min_length=5)
        assert 1 in trajectories
        assert len(trajectories[1]) == 10
