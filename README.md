# Smart Auto Video Annotation for Labeling Data for Training

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Deep Learning](https://img.shields.io/badge/Deep%20Learning-MOT%20/%20Object%20Tracking-red)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Automated video annotation with multi-object tracking for generating labeled training data.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   AnnotationPipeline                     │
├──────────┬──────────┬───────────┬──────────┬────────────┤
│ Detector │ Tracker  │ Annotator │ Quality  │ AutoLabeler│
│ (YOLO)   │ (Byte)   │ (COCO/    │ (IoU/    │ (Propagate/│
│          │ (Track)  │  YOLO/VOC)│  Match)  │  Active)   │
└─────┬────┴─────┬────┴─────┬─────┴────┬─────┴─────┬──────┘
      │          │          │          │           │
      ▼          ▼          ▼          ▼           ▼
  ultralytics  scipy    pathlib     numpy     human-in-
                                       the-loop
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
# or
pip install -e .
```

### Basic Usage

```python
from src.pipeline import AnnotationPipeline

# Run full pipeline on a video
pipeline = AnnotationPipeline(model_path="yolo11x.pt", confidence=0.25)
result = pipeline.process_video("input.mp4", output_dir="output/")
print(f"Processed {result['labeled_frames']} frames, {result['unique_tracks']} tracks")
```

### Per-Module Usage

```python
# Detection only
from src.detector import ObjectDetector
det = ObjectDetector("yolo11n.pt", confidence=0.3)
detections = det.detect("image.jpg")
for d in detections:
    print(f"{d.class_name}: {d.confidence:.2f}")

# Tracking
from src.tracker import ObjectTracker
tracker = ObjectTracker()
results = tracker.track("video.mp4")

# Annotation export
from src.annotator import AnnotationGenerator
gen = AnnotationGenerator()
gen.generate_coco(annotations_dict, "output/annotations.json")
gen.generate_yolo(annotations_dict, "output/yolo/")
gen.generate_voc(annotations_dict, "output/voc/")

# Auto-labeling with propagation
from src.auto_labeler import AutoLabeler
labeler = AutoLabeler()
labels = labeler.propagate_labels(keyframe_dets, "video.mp4", method="tracking")

# Quality assessment
from src.quality import QualityAssessor
qa = QualityAssessor()
metrics = qa.assess_quality(predictions, ground_truth)
print(f"mAP: {metrics['f1']:.3f}, mIoU: {metrics['mean_iou']:.3f}")
```

### CLI

```bash
# Process a video
python -m src.pipeline

# Docker
docker build -t savann .
docker run -v $(pwd)/data:/data savann python -m src.pipeline
```

## Module Reference

### `ObjectDetector` (`src/detector.py`)

| Method | Description |
|--------|-------------|
| `detect(source)` | Detect objects in a single image/array |
| `detect_batch(sources, batch_size)` | Batch detection across multiple images |
| `detect_video(video_path)` | Yield `(frame_idx, detections)` per frame |
| `detect_webcam(camera_id, max_frames)` | Live webcam detection stream |
| `detect_images(directory)` | Detect all images in a folder |

### `ObjectTracker` (`src/tracker.py`)

| Method | Description |
|--------|-------------|
| `track(source)` | Track objects across all frames in video |
| `get_trajectories(min_length, smooth)` | Extract smoothed trajectories per track |
| `visualize_tracks(video_path)` | Annotated frames with track IDs |

### `AnnotationGenerator` (`src/annotator.py`)

| Method | Description |
|--------|-------------|
| `generate_coco(annotations, output_path)` | COCO JSON format |
| `generate_yolo(annotations, output_dir)` | YOLO `.txt` per image |
| `generate_voc(annotations, output_dir)` | Pascal VOC `.xml` per image |
| `visualize_annotations(image, detections)` | Draw annotations on image |

### `AutoLabeler` (`src/auto_labeler.py`)

| Method | Description |
|--------|-------------|
| `label_video(video_path, output_dir)` | Label all frames with detection + tracking |
| `label_images(image_dir)` | Label all images in directory |
| `interactive_label(source, callback)` | Human-in-the-loop labeling |
| `propagate_labels(keyframes, video, method)` | Propagate labels across frames |
| `select_active_learning_samples(dets, budget)` | Select frames for human review |

### `QualityAssessor` (`src/quality.py`)

| Method | Description |
|--------|-------------|
| `assess_quality(predictions, ground_truth)` | Precision, recall, F1, mIoU |
| `compute_iou(box_a, box_b)` | Intersection over Union |
| `inter_annotator_agreement(ann_a, ann_b)` | Agreement between annotators |
| `detect_duplicates(detections)` | Find overlapping detections |
| `suggest_corrections(detections)` | Quality improvement suggestions |

### `AnnotationPipeline` (`src/pipeline.py`)

| Method | Description |
|--------|-------------|
| `process_video(video_path, output_dir)` | Full pipeline: detect → track → annotate → export |
| `process_folder(input_dir, output_dir)` | Batch process all images |
| `export(annotations, output_dir, formats)` | Export to COCO/YOLO/VOC |
| `generate_report(results)` | Summary report with stats |

## Export Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| YOLO | `.txt` | Ultralytics training |
| COCO | `.json` | General CV research |
| Pascal VOC | `.xml` | Legacy systems |

## Requirements

```
opencv-python-headless>=4.13.0
ultralytics>=8.3.0
numpy>=1.26.0
supervision>=0.22.0
scipy>=1.14.0
pytest>=8.0.0
```

## Roadmap

- [x] YOLOv11 object detection
- [x] ByteTrack multi-object tracking
- [x] COCO / YOLO / VOC annotation export
- [x] Video and image batch processing
- [x] Label propagation (tracking + interpolation)
- [x] Quality assessment (IoU, precision/recall, duplicate detection)
- [ ] Deep SORT with ReID features
- [ ] Active learning selection UI
- [ ] CVAT / Label Studio export
- [ ] Multi-GPU batch inference
- [ ] Web-based annotation dashboard
- [ ] Auto-mask generation (SAM integration)

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
