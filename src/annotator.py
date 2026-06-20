"""Annotation generation — COCO, YOLO, Pascal VOC formats."""

from __future__ import annotations

import json
import logging
import uuid
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from src.detector import Detection
from src.utils import draw_boxes, xyxy_to_normalized, xyxy_to_xywh

logger = logging.getLogger(__name__)


class AnnotationGenerator:
    """Generate annotations in COCO, YOLO, and Pascal VOC formats."""

    def __init__(self, class_names: list[str] | None = None):
        self.class_names = class_names or []

    # ------------------------------------------------------------------
    # COCO JSON
    # ------------------------------------------------------------------

    def generate_coco(
        self,
        annotations: dict[str, list[Detection]],
        output_path: str | Path,
    ) -> Path:
        """Generate COCO JSON from {image_filename: [Detection]} mapping.

        Args:
            annotations: mapping of image filenames to detection lists.
            output_path: path to write the JSON file.
        """
        coco = {
            "info": {
                "description": "Auto-generated annotations",
                "version": "1.0",
                "year": datetime.now().year,
                "date_created": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        # Categories
        seen: dict[str, int] = {}
        for dets in annotations.values():
            for d in dets:
                if d.class_name not in seen:
                    seen[d.class_name] = len(seen) + 1
                    coco["categories"].append({
                        "id": seen[d.class_name],
                        "name": d.class_name,
                        "supercategory": "object",
                    })

        ann_id = 1
        for img_id, (img_name, dets) in enumerate(annotations.items(), start=1):
            if not dets:
                img_h, img_w = 0, 0
            else:
                # Read image for dimensions if available
                img = cv2.imread(img_name)
                if img is not None:
                    img_h, img_w = img.shape[:2]
                else:
                    img_h, img_w = dets[0].bbox[3], dets[0].bbox[2]

            coco["images"].append({
                "id": img_id,
                "file_name": img_name,
                "width": img_w,
                "height": img_h,
            })

            for d in dets:
                xywh = xyxy_to_xywh(d.bbox.reshape(1, 4))[0]
                area = float(xywh[2] * xywh[3])
                seg = [[
                    float(d.bbox[0]), float(d.bbox[1]),
                    float(d.bbox[2]), float(d.bbox[1]),
                    float(d.bbox[2]), float(d.bbox[3]),
                    float(d.bbox[0]), float(d.bbox[3]),
                ]]
                coco["annotations"].append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": seen.get(d.class_name, 1),
                    "bbox": xywh.tolist(),
                    "area": area,
                    "segmentation": seg,
                    "iscrowd": 0,
                    "score": d.confidence,
                })
                ann_id += 1

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(coco, indent=2))
        logger.info("COCO annotations written to %s (%d images, %d annotations)",
                     out, len(coco["images"]), len(coco["annotations"]))
        return out

    # ------------------------------------------------------------------
    # YOLO TXT
    # ------------------------------------------------------------------

    def generate_yolo(
        self,
        annotations: dict[str, list[Detection]],
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate YOLO-format .txt files per image.

        Each line: <class_id> <x_center> <y_center> <width> <height> [confidence]
        """
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for img_path, dets in annotations.items():
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Cannot read %s for YOLO export, skipping", img_path)
                continue
            h, w = img.shape[:2]
            norm = xyxy_to_normalized(
                np.array([d.bbox for d in dets]) if dets else np.zeros((0, 4)),
                w, h,
            )
            txt_name = Path(img_path).stem + ".txt"
            txt_path = out_dir / txt_name
            lines = []
            for i, d in enumerate(dets):
                cx, cy, bw, bh = norm[i]
                lines.append(f"{d.class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            txt_path.write_text("\n".join(lines))
            written.append(txt_path)
        logger.info("YOLO annotations written: %d files to %s", len(written), out_dir)
        return written

    # ------------------------------------------------------------------
    # Pascal VOC XML
    # ------------------------------------------------------------------

    def generate_voc(
        self,
        annotations: dict[str, list[Detection]],
        output_dir: str | Path,
    ) -> list[Path]:
        """Generate Pascal VOC XML files per image."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for img_path, dets in annotations.items():
            img = cv2.imread(img_path)
            if img is None:
                logger.warning("Cannot read %s for VOC export, skipping", img_path)
                continue
            h, w, c = img.shape

            root = ET.Element("annotation")
            ET.SubElement(root, "folder").text = ""
            ET.SubElement(root, "filename").text = Path(img_path).name
            ET.SubElement(root, "path").text = str(img_path)

            size = ET.SubElement(root, "size")
            ET.SubElement(size, "width").text = str(w)
            ET.SubElement(size, "height").text = str(h)
            ET.SubElement(size, "depth").text = str(c)

            for d in dets:
                obj = ET.SubElement(root, "object")
                ET.SubElement(obj, "name").text = d.class_name
                ET.SubElement(obj, "pose").text = "Unspecified"
                ET.SubElement(obj, "truncated").text = "0"
                ET.SubElement(obj, "difficult").text = "0"
                bbox = ET.SubElement(obj, "bndbox")
                ET.SubElement(bbox, "xmin").text = str(int(d.bbox[0]))
                ET.SubElement(bbox, "ymin").text = str(int(d.bbox[1]))
                ET.SubElement(bbox, "xmax").text = str(int(d.bbox[2]))
                ET.SubElement(bbox, "ymax").text = str(int(d.bbox[3]))

            xml_path = out_dir / (Path(img_path).stem + ".xml")
            tree = ET.ElementTree(root)
            ET.indent(tree, space="  ")
            tree.write(str(xml_path), encoding="unicode", xml_declaration=True)
            written.append(xml_path)
        logger.info("VOC annotations written: %d files to %s", len(written), out_dir)
        return written

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def visualize_annotations(
        self,
        image: np.ndarray,
        detections: list[Detection],
        track_ids: list[int] | None = None,
    ) -> np.ndarray:
        """Draw annotations on an image and return the visualization."""
        boxes = np.array([d.bbox for d in detections]) if detections else np.zeros((0, 4))
        labels = [f"{d.class_name} {d.confidence:.2f}" for d in detections]
        return draw_boxes(
            image, boxes, labels=labels, track_ids=track_ids,
        )
