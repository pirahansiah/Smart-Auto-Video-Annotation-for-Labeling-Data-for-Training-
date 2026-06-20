# ROADMAP.md — Smart Auto Video Annotation for Labeling Data for Training

## 12-Month Vision (2026 Q3 – 2027 Q2)

Evolve from a tracking-based annotation script into a production-grade auto-labeling platform with model-in-the-loop quality control, real-time processing, and seamless integration with training pipelines.

### Q3 2026 — Core Pipeline
- Refactor into modular Python package with CLI and API interface
- Add RT-DETR and YOLO-World as detection backends alongside YOLOv11
- Implement StrongSORT tracker with appearance + motion fusion
- Add dataset quality metrics (annotation coverage, class distribution, frame consistency)

### Q4 2026 — Quality & Scale
- Implement model-in-the-loop verification (ensemble detection for high-confidence filtering)
- Add multi-GPU batch processing with video queue management
- Create annotation confidence visualization overlay for quality review
- Support streaming video input (RTSP, webcam, IP cameras) for real-time annotation

### Q1 2027 — Advanced Features
- Integrate SAM 2 for pixel-level segmentation annotations (auto-polygon)
- Add 3D bounding box estimation from monocular video using depth estimation models
- Implement temporal consistency smoothing across detection frames
- Create annotation diff tool to compare human vs auto annotations

### Q2 2027 — Production Platform
- Add web-based annotation review interface with collaborative editing
- Implement active learning pipeline with human-in-the-loop confidence routing
- Create training pipeline integration (auto-split, augment, and upload to training cluster)
- Add support for video analytics use cases (action recognition, temporal localization)

## Technical Debt

| Item | Priority | Description |
|------|----------|-------------|
| No package structure | High | Not installable; requires manual script execution |
| No CLI/API | High | Hardcoded parameters in source code |
| No tests | Medium | Zero unit or integration tests |
| Single tracker | Medium | Only ByteTrack documented; need BoT-SORT/OCSORT implementation |
| No error handling | Medium | Pipeline fails silently on corrupted frames |
| No configuration | Medium | Detection model and tracker settings not configurable |
| Documentation gaps | Low | README describes features; no installation/setup guide |

## Future Features

- **Pixel-Level Auto-Segmentation**: SAM 2 integration for instance and semantic segmentation masks
- **3D Annotation**: Depth estimation + 3D bounding boxes from monocular video
- **Real-Time Dashboard**: Web UI showing live annotation progress and quality metrics
- **Video Analytics Integration**: Extend beyond object detection to action recognition and temporal localization
- **Federated Annotation**: Multi-site annotation with centralized quality control
- **Auto-Class Discovery**: Unsupervised clustering to discover object categories in unlabeled video
- **Cost Estimator**: Predict annotation cost savings based on video characteristics
