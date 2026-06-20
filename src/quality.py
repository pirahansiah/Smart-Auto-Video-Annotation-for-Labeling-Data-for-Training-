"""Annotation quality assessment — IoU, inter-annotator agreement, duplicate detection."""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Sequence

import numpy as np

from src.detector import Detection

logger = logging.getLogger(__name__)


class QualityAssessor:
    """Assess annotation quality via IoU, agreement, and duplicate detection."""

    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold

    # ------------------------------------------------------------------
    # IoU computation
    # ------------------------------------------------------------------

    @staticmethod
    def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
        """Compute IoU between two [x1,y1,x2,y2] boxes."""
        xa = max(box_a[0], box_b[0])
        ya = max(box_a[1], box_b[1])
        xb = min(box_a[2], box_b[2])
        yb = min(box_a[3], box_b[3])
        inter = max(0, xb - xa) * max(0, yb - ya)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    @staticmethod
    def compute_iou_matrix(
        boxes_a: np.ndarray, boxes_b: np.ndarray
    ) -> np.ndarray:
        """Pairwise IoU between [N,4] and [M,4] box arrays."""
        if len(boxes_a) == 0 or len(boxes_b) == 0:
            return np.zeros((len(boxes_a), len(boxes_b)))
        xa = np.maximum(boxes_a[:, None, 0], boxes_b[None, :, 0])
        ya = np.maximum(boxes_a[:, None, 1], boxes_b[None, :, 1])
        xb = np.minimum(boxes_a[:, None, 2], boxes_b[None, :, 2])
        yb = np.minimum(boxes_a[:, None, 3], boxes_b[None, :, 3])
        inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / np.maximum(union, 1e-6)

    def compute_iow(
        self, box_a: np.ndarray, box_b: np.ndarray
    ) -> float:
        """Intersection over Weighted area (harmonic mean of IoU with each box as reference)."""
        iou_ab = self.compute_iou(box_a, box_b)
        # IoW: harmonic mean approach
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        if area_a == 0 or area_b == 0:
            return 0.0
        inter = iou_ab * min(area_a, area_b)  # approximate
        return 2 * inter / (area_a + area_b)

    # ------------------------------------------------------------------
    # Quality assessment
    # ------------------------------------------------------------------

    def assess_quality(
        self,
        predictions: list[Detection],
        ground_truth: list[Detection],
    ) -> dict:
        """Compare predictions against ground truth.

        Returns dict with precision, recall, F1, mean IoU, per-class metrics.
        """
        if not ground_truth and not predictions:
            return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mean_iou": 1.0, "per_class": {}}
        if not ground_truth:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0, "per_class": {}}
        if not predictions:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0, "per_class": {}}

        pred_boxes = np.array([d.bbox for d in predictions])
        gt_boxes = np.array([d.bbox for d in ground_truth])
        iou_mat = self.compute_iou_matrix(pred_boxes, gt_boxes)

        # Greedy matching
        matched_gt: set[int] = set()
        matched_pred: set[int] = set()
        ious: list[float] = []
        for _ in range(min(len(predictions), len(ground_truth))):
            if iou_mat.size == 0:
                break
            idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            if iou_mat[idx] < self.iou_threshold:
                break
            ious.append(iou_mat[idx])
            matched_pred.add(idx[0])
            matched_gt.add(idx[1])
            iou_mat[idx[0], :] = 0
            iou_mat[:, idx[1]] = 0

        tp = len(matched_pred)
        fp = len(predictions) - tp
        fn = len(ground_truth) - len(matched_gt)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-6)
        mean_iou = float(np.mean(ious)) if ious else 0.0

        # Per-class
        per_class: dict[str, dict] = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
        for pidx in matched_pred:
            cn = predictions[pidx].class_name
            per_class[cn]["tp"] += 1
        for pidx in range(len(predictions)):
            if pidx not in matched_pred:
                cn = predictions[pidx].class_name
                per_class[cn]["fp"] += 1
        for gidx in range(len(ground_truth)):
            if gidx not in matched_gt:
                cn = ground_truth[gidx].class_name
                per_class[cn]["fn"] += 1

        for cn, m in per_class.items():
            p = m["tp"] / max(m["tp"] + m["fp"], 1)
            r = m["tp"] / max(m["tp"] + m["fn"], 1)
            m["precision"] = p
            m["recall"] = r
            m["f1"] = 2 * p * r / max(p + r, 1e-6)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou": mean_iou,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "per_class": dict(per_class),
        }

    # ------------------------------------------------------------------
    # Inter-annotator agreement (Cohen's kappa)
    # ------------------------------------------------------------------

    def inter_annotator_agreement(
        self,
        annotator_a: list[Detection],
        annotator_b: list[Detection],
    ) -> dict:
        """Compute agreement between two annotators using IoU matching."""
        if not annotator_a or not annotator_b:
            return {"agreement": 0.0, "matched_pairs": 0}

        boxes_a = np.array([d.bbox for d in annotator_a])
        boxes_b = np.array([d.bbox for d in annotator_b])
        iou_mat = self.compute_iou_matrix(boxes_a, boxes_b)

        matched = 0
        total = max(len(annotator_a), len(annotator_b))
        for _ in range(min(len(annotator_a), len(annotator_b))):
            if iou_mat.size == 0:
                break
            idx = np.unravel_index(iou_mat.argmax(), iou_mat.shape)
            if iou_mat[idx] < self.iou_threshold:
                break
            matched += 1
            iou_mat[idx[0], :] = 0
            iou_mat[:, idx[1]] = 0

        return {
            "agreement": matched / max(total, 1),
            "matched_pairs": matched,
            "total_a": len(annotator_a),
            "total_b": len(annotator_b),
        }

    # ------------------------------------------------------------------
    # Duplicate detection
    # ------------------------------------------------------------------

    def detect_duplicates(
        self,
        detections: list[Detection],
    ) -> list[tuple[int, int, float]]:
        """Find duplicate (highly overlapping) detections.

        Returns list of (idx_a, idx_b, iou) for pairs above threshold.
        """
        if len(detections) < 2:
            return []
        boxes = np.array([d.bbox for d in detections])
        iou_mat = self.compute_iou_matrix(boxes, boxes)
        np.fill_diagonal(iou_mat, 0)

        duplicates: list[tuple[int, int, float]] = []
        seen: set[tuple[int, int]] = set()
        for i in range(len(detections)):
            for j in range(i + 1, len(detections)):
                if iou_mat[i, j] >= self.iou_threshold:
                    pair = (min(i, j), max(i, j))
                    if pair not in seen:
                        duplicates.append((pair[0], pair[1], float(iou_mat[i, j])))
                        seen.add(pair)
        return duplicates

    # ------------------------------------------------------------------
    # Suggest corrections
    # ------------------------------------------------------------------

    def suggest_corrections(
        self,
        detections: list[Detection],
    ) -> list[dict]:
        """Analyze detections and suggest quality improvements."""
        suggestions: list[dict] = []

        # Check for duplicates
        dupes = self.detect_duplicates(detections)
        if dupes:
            suggestions.append({
                "type": "duplicate",
                "message": f"Found {len(dupes)} duplicate detection pairs",
                "pairs": dupes,
                "action": "Remove lower-confidence duplicate from each pair",
            })

        # Check for low confidence
        low_conf = [i for i, d in enumerate(detections) if d.confidence < 0.3]
        if low_conf:
            suggestions.append({
                "type": "low_confidence",
                "message": f"{len(low_conf)} detections have confidence < 0.3",
                "indices": low_conf,
                "action": "Review low-confidence detections; consider removing or relabeling",
            })

        # Check for tiny boxes
        for i, d in enumerate(detections):
            w = d.bbox[2] - d.bbox[0]
            h = d.bbox[3] - d.bbox[1]
            if w < 10 or h < 10:
                suggestions.append({
                    "type": "tiny_box",
                    "message": f"Detection {i} is very small ({w:.0f}x{h:.0f})",
                    "index": i,
                    "action": "May be noise; verify if object is meaningful",
                })

        # Check for very large boxes
        for i, d in enumerate(detections):
            w = d.bbox[2] - d.bbox[0]
            h = d.bbox[3] - d.bbox[1]
            if w > 2000 or h > 2000:
                suggestions.append({
                    "type": "oversized_box",
                    "message": f"Detection {i} is very large ({w:.0f}x{h:.0f})",
                    "index": i,
                    "action": "May be over-segmented or covering multiple objects",
                })

        return suggestions
