"""aidetect — AI-generated text detection toolkit.

Two detectors:
- Binoculars (zero-shot, GPU, transformers): from aidetect.binoculars import Binoculars
- FeatureDetector (supervised, CPU, sklearn):  from aidetect import FeatureDetector

Binoculars is imported lazily so the CPU parts work without torch installed.
"""

from .classifier import FeatureDetector
from .ensemble import ScoreEnsemble, chunk_text, score_long_text
from .evaluate import EvalReport, evaluate, pick_threshold, pick_threshold_max_accuracy
from .features import FEATURE_NAMES, extract_features
from .readers import (
    docx_to_markdown,
    load_directory,
    load_text,
    pdf_to_markdown,
    read_docx,
    read_pdf,
    save_markdown,
    to_markdown,
)
from .scoring import (
    DetectorUnavailable,
    DocumentVerdict,
    Section,
    Verdict,
    score_document,
    score_text,
)

__all__ = [
    "FeatureDetector",
    "EvalReport",
    "evaluate",
    "pick_threshold",
    "pick_threshold_max_accuracy",
    "ScoreEnsemble",
    "chunk_text",
    "score_long_text",
    "FEATURE_NAMES",
    "extract_features",
    "load_text",
    "load_directory",
    "read_docx",
    "read_pdf",
    "to_markdown",
    "docx_to_markdown",
    "pdf_to_markdown",
    "save_markdown",
    "score_text",
    "score_document",
    "Verdict",
    "DocumentVerdict",
    "Section",
    "DetectorUnavailable",
]
__version__ = "1.7.0"
