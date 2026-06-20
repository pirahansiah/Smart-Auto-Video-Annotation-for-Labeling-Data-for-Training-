"""Smart Auto Video Annotation — automated video annotation with MOT."""

from src.detector import ObjectDetector
from src.tracker import ObjectTracker
from src.annotator import AnnotationGenerator
from src.auto_labeler import AutoLabeler
from src.quality import QualityAssessor
from src.pipeline import AnnotationPipeline

__all__ = [
    "ObjectDetector",
    "ObjectTracker",
    "AnnotationGenerator",
    "AutoLabeler",
    "QualityAssessor",
    "AnnotationPipeline",
]
