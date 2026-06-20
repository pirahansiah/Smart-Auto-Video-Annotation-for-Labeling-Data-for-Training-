# Smart Auto Video Annotation for Labeling Data for Training

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MOT%20/%20Object%20Tracking-red)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automated video annotation with multi-object tracking for generating labeled training data.

## Overview

This repository provides an **automatic video annotation pipeline** that combines:
- **Multi-object tracking (MOT)** for consistent identity assignment across frames
- **Object detection** for bounding box generation
- **Auto-labeling** to reduce manual annotation effort by 80–95%
- **Export** to standard formats (YOLO, COCO, Pascal VOC, Darknet)

## Pipeline

```
Input Video → Object Detection → Multi-Object Tracking → Auto Annotation → Export
     ↓              ↓                    ↓                     ↓            ↓
   MP4/AVI     YOLOv8/RT-DETR      BoT-SORT/OCSORT      Bounding Boxes   YOLO/COCO
```

## Features

- **Zero-shot tracking**: Track objects without pre-training on target classes
- **Identity preservation**: Consistent IDs across occlusions and re-entries
- **Configurable annotation formats**: YOLO, COCO JSON, Pascal VOC XML, CVAT
- **Active learning**: Flag uncertain detections for human review
- **Batch processing**: Process multiple videos in parallel

## 2025–2026: State-of-the-Art in Video Annotation

### Detection Models

| Model | mAP | Speed | Use Case |
|-------|-----|-------|----------|
| **YOLOv11** (Ultralytics) | 54.7 mAP | Real-time | General-purpose detection |
| **RT-DETR** (Baidu) | 54.8 mAP | Real-time | Transformer-based, end-to-end |
| **Grounding DINO 2** (IDEA) | 55.7 mAP | Near real-time | Open-vocabulary detection |
| **Florence-2** (Microsoft) | 54.4 mAP | Near real-time | Multi-task vision-language |
| **YOLO-World** | 52.3 mAP | Real-time | Text-prompted detection |

### Tracking Algorithms

| Tracker | MOTA | IDF1 | Notes |
|---------|------|------|-------|
| **BoT-SORT** | 80.5 | 81.3 | Camera motion compensation |
| **OCSORT** | 78.4 | 79.8 | Observation-centric, robust |
| **ByteTrack** | 77.7 | 79.5 | Multi-frame association |
| **StrongSORT** | 79.8 | 82.1 | Appearance + motion features |
| **MotionRNN** | 76.2 | 77.5 | Predictive motion modeling |

### Auto-Labeling Tools (2025–2026)

| Tool | Key Feature |
|------|-------------|
| **Roboflow** | Cloud-based auto-label, active learning, versioning |
| **CVAT** | Open-source, multi-annotator, AI-assisted |
| **Label Studio** | Multi-format, ML backend, enterprise-ready |
| **FiftyOne** | Visual curation, model evaluation, dataset quality |
| **Prodigy** | Active learning, NLP + CV, spaCy ecosystem |
| **V7 (Darwin)** | AI-assisted annotation, auto-polygon, tracking |

### Recommended Pipeline (2025+)

```bash
# 1. Install dependencies
pip install ultralytics supervision torch torchvision

# 2. Run detection + tracking
yolo track source=video.mp4 model=yolo11x.pt tracker=bytetrack.yaml

# 3. Export to training format
yolo export model=best.pt format=onnx  # For edge deployment
```

## Quick Start

```python
from ultralytics import YOLO

# Load pre-trained model
model = YOLO("yolo11x.pt")

# Track objects in video
results = model.track(
    source="input_video.mp4",
    tracker="bytetrack.yaml",
    persist=True,
    save=True,
    save_txt=True  # Export labels
)

# Results saved to runs/track/
```

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| YOLO | `.txt` | Ultralytics training |
| COCO | `.json` | General CV research |
| Pascal VOC | `.xml` | Legacy systems |
| Darknet | `.data` | Darknet training |
| CVAT | `.xml` | Annotation platform |

## Requirements

```
ultralytics>=8.3.0
supervision>=0.22.0
torch>=2.0
opencv-python>=4.8
```

## Related Projects

- [FairMOT](https://github.com/pirahansiah/FairMOT) — Fair multi-object tracking
- [eot-training-multi-object-tracking](https://github.com/pirahansiah/eot-training-multi-object-tracking) — MOT training

## References

- [BoT-SORT: Robust Associations Multi-Pedestrian Tracking](https://arxiv.org/abs/2206.14651)
- [ByteTrack: Multi-Object Tracking by Associating Every Detection Box](https://arxiv.org/abs/2110.02033)
- [YOLOv11: Latest YOLO Architecture](https://docs.ultralytics.com/models/yolo11/)
- [Grounding DINO 2: Open-Set Detection](https://arxiv.org/abs/2405.10300)

## Author

**Farshid Pirahansiah**
- Website: [pirahansiah.com](https://www.pirahansiah.com)
- GitHub: [github.com/pirahansiah](https://github.com/pirahansiah)
- LinkedIn: [linkedin.com/in/pirahansiah](https://www.linkedin.com/in/pirahansiah)

## License

MIT License — See [LICENSE](LICENSE) for details.
