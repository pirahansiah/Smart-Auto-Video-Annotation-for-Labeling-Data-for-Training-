"""Shared utilities — video I/O, image preprocessing, format conversion, visualization."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------

def open_video(path: str | Path) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    return cap


def get_video_info(path: str | Path) -> dict:
    cap = open_video(path)
    info = {
        "path": str(path),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    info["duration_s"] = info["frame_count"] / max(info["fps"], 1e-6)
    return info


def iter_frames(path: str | Path):
    """Yield (frame_index, frame_bgr) for every frame in a video."""
    cap = open_video(path)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield idx, frame
        idx += 1
    cap.release()


def write_video(
    path: str | Path,
    frames: Sequence[np.ndarray],
    fps: float = 30.0,
) -> None:
    if not frames:
        raise ValueError("No frames to write")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    return img


def resize_to_max(img: np.ndarray, max_side: int = 1280) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= max_side:
        return img
    scale = max_side / max(h, w)
    return cv2.resize(img, (int(w * scale), int(h * scale)))


def letterbox(
    img: np.ndarray,
    new_shape: tuple[int, int] = (640, 640),
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize with letterbox padding, returns (image, scale, (pad_w, pad_h))."""
    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    new_unpad = (int(round(w * r)), int(round(h * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2
    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return img, r, (int(round(dw)), int(round(dh)))


# ---------------------------------------------------------------------------
# Format conversion
# ---------------------------------------------------------------------------

def xyxy_to_xywh(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 2] -= out[:, 0]
    out[:, 3] -= out[:, 1]
    return out


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = boxes.copy()
    out[:, 0] += out[:, 2]
    out[:, 1] += out[:, 3]
    return out


def xyxy_to_normalized(
    boxes: np.ndarray, img_w: int, img_h: int
) -> np.ndarray:
    out = xyxy_to_xywh(boxes).astype(np.float32)
    out[:, 0] /= img_w
    out[:, 1] /= img_h
    out[:, 2] /= img_w
    out[:, 3] /= img_h
    return out


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

COLORS = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
    (82, 0, 133), (203, 56, 255), (255, 149, 200), (255, 55, 199),
]


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    labels: Sequence[str] | None = None,
    scores: Sequence[float] | None = None,
    track_ids: Sequence[int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    vis = img.copy()
    for i, box in enumerate(boxes.astype(int)):
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(vis, (box[0], box[1]), (box[2], box[3]), color, thickness)
        parts = []
        if track_ids is not None:
            parts.append(f"ID:{track_ids[i]}")
        if labels is not None:
            parts.append(labels[i])
        if scores is not None:
            parts.append(f"{scores[i]:.2f}")
        text = " ".join(parts) if parts else ""
        if text:
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(vis, (box[0], box[1] - th - 8), (box[0] + tw + 4, box[1]), color, -1)
            cv2.putText(vis, text, (box[0] + 2, box[1] - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def list_videos(directory: str | Path) -> list[Path]:
    d = Path(directory)
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in VIDEO_EXTENSIONS)


def list_images(directory: str | Path) -> list[Path]:
    d = Path(directory)
    return sorted(p for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
