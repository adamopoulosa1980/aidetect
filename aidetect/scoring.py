"""One scoring entry point shared by the CLI and the GUI.

Keeps the two frontends from drifting on the detail that is easiest to get
wrong: the two detectors disagree on direction. Binoculars scores AI text
*low* (below its threshold); FeatureDetector returns P(AI), so AI text scores
*high*. Both are normalised into a Verdict here, once.
"""

from __future__ import annotations

from dataclasses import dataclass, field

DETECTORS = ("binoculars", "features")


class DetectorUnavailable(RuntimeError):
    """A detector's dependencies are missing from this build/environment."""


@dataclass
class Verdict:
    detector: str
    score: float
    threshold: float
    is_ai: bool
    extra: dict[str, float] = field(default_factory=dict)
    device: str | None = None
    percentile: float | None = None

    @property
    def comparison(self) -> str:
        """Where this sits among the user's own documents.

        More defensible than the verdict: it is measured on their writing, so
        the bias that perplexity detectors show against non-native English
        cancels out instead of counting against them.
        """
        if self.percentile is None:
            return ""
        return (
            f"lower than {100 - self.percentile:.0f}% of your reference documents"
            if self.percentile < 50
            else f"higher than {self.percentile:.0f}% of your reference documents"
        )

    @property
    def label(self) -> str:
        return "AI-generated" if self.is_ai else "human"

    @property
    def headline(self) -> str:
        """The verdict in words, for people who are not reading the number."""
        return "Reads as AI-generated" if self.is_ai else "Reads as human-written"

    def __str__(self) -> str:
        head = f"score={self.score:.4f}  threshold={self.threshold:.4f}  verdict={self.label}"
        parts = [f"{k}={v:.2f}" for k, v in self.extra.items()]
        if self.device:
            parts.append(f"device={self.device}")
        if self.comparison:
            parts.append(self.comparison)
        return f"{head}\n{'  '.join(parts)}" if parts else head


# One loaded model pair, reused across calls. Loading Falcon-7B twice takes
# minutes and tens of GB, so rebuilding it per document made every analysis
# after the first pay the full startup cost again. Single-slot on purpose:
# holding two pairs at once would double an already large footprint.
_loaded: tuple[tuple[str, str], object] | None = None


def _load_binoculars(cls, observer: str, performer: str):
    global _loaded
    key = (observer, performer)
    if _loaded is not None:
        cached_key, detector = _loaded
        if cached_key == key:
            return detector
        # Drop the old pair before loading the new one, so the two never
        # occupy memory simultaneously.
        _loaded = None
        del detector
        _free_memory()

    detector = cls(observer_name=observer, performer_name=performer)
    _loaded = (key, detector)
    return detector


def _free_memory() -> None:
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def clear_model_cache() -> None:
    """Release the loaded Binoculars pair and its memory."""
    global _loaded
    _loaded = None
    _free_memory()


def score_text(
    text: str,
    detector: str = "binoculars",
    *,
    model: str | None = None,
    observer: str = "tiiuae/falcon-7b",
    performer: str = "tiiuae/falcon-7b-instruct",
    threshold: float | None = None,
    mode: str = "accuracy",
) -> Verdict:
    """Score ``text`` with the named detector. Raises DetectorUnavailable if
    the detector's dependencies are not present in this build."""
    if detector not in DETECTORS:
        raise ValueError(f"Unknown detector '{detector}'. Choose from {DETECTORS}.")

    # Whitespace is made of tokens, so it counts towards perplexity: the same
    # wording joined with newlines rather than spaces scored 3.8% differently
    # in testing, which is wider than the margin many documents sit from the
    # threshold. Normalising at the single point everything passes through
    # means a document scores the same whether it arrived as .docx, .pdf, or a
    # paste into the window.
    from .readers import canonical_text

    text = canonical_text(text)

    if detector == "features":
        if not model:
            raise DetectorUnavailable(
                "The features detector needs a trained model file "
                "(produced by FeatureDetector.save())."
            )
        from .classifier import FeatureDetector

        det = FeatureDetector.load(model)
        score = float(det.predict_proba([text])[0])
        thr = 0.5 if threshold is None else threshold
        # Higher = more AI here, the opposite of Binoculars below.
        return Verdict("features", score, thr, is_ai=score >= thr)

    try:
        from .binoculars import THRESHOLDS, Binoculars
    except ImportError as e:
        raise DetectorUnavailable(
            f"The binoculars detector needs torch and transformers ({e}). "
            "This build does not include them. Use --stylometry for writing "
            "indicators that need no model, or "
            'install the extras with: pip install -e ".[gpu]"'
        ) from e

    if mode not in THRESHOLDS:
        raise ValueError(f"Unknown mode {mode!r}. Choose from {tuple(THRESHOLDS)}.")

    det = _load_binoculars(Binoculars, observer, performer)
    if threshold is not None:
        det.threshold = threshold
    else:
        det.threshold = _threshold_for(observer, performer, mode)
    r = det.score(text)
    return Verdict(
        "binoculars",
        r.score,
        r.threshold,
        is_ai=r.is_ai,
        extra={"perplexity": r.perplexity, "cross_perplexity": r.cross_perplexity},
        device=det.device,
        percentile=_percentile_against_own(r.score, observer, performer),
    )


def _percentile_against_own(score: float, observer: str, performer: str) -> float | None:
    from .calibrate import percentile_of
    from .models import find_by_models

    pair = find_by_models(observer, performer)
    return percentile_of(score, pair.key) if pair else None


def _threshold_for(observer: str, performer: str, mode: str) -> float:
    """The threshold to judge this pair by.

    A threshold calibrated on the user's own documents wins, because it was
    measured on the documents that matter rather than on the paper's benchmark.
    Otherwise fall back to the published constant for the requested mode, which
    is only meaningful for the Falcon pair.
    """
    from .calibrate import load_threshold
    from .models import find_by_models

    pair = find_by_models(observer, performer)
    if pair is not None:
        calibrated = load_threshold(pair.key)
        if calibrated is not None:
            return calibrated
    from .binoculars import THRESHOLDS

    return THRESHOLDS[mode]


def detector_available(name: str) -> bool:
    """Whether ``name`` can actually run here, without importing anything heavy.

    Lets a frontend grey out a detector up front rather than letting the user
    pick it and hit an error — the lite executable ships without torch.
    """
    if name not in DETECTORS:
        raise ValueError(f"Unknown detector '{name}'. Choose from {DETECTORS}.")
    if name == "features":
        return True
    from importlib.util import find_spec

    try:
        return find_spec("torch") is not None and find_spec("transformers") is not None
    except (ImportError, ValueError):
        return False


def describe_device() -> str:
    """One line on what Binoculars would run on here.

    Imports torch, so it is slow enough to belong on a worker thread rather
    than in a window's constructor.
    """
    if not detector_available("binoculars"):
        return "Binoculars is not available in this build."
    import torch

    if torch.cuda.is_available():
        return f"GPU ready: {torch.cuda.get_device_name(0)}"
    return "No CUDA GPU detected - Binoculars will run on CPU and be very slow."


# --------------------------------------------------------------------------
# Whole-document scoring
#
# Binoculars truncates at 512 tokens, roughly 380 words, so a single call
# judges only the first page of a long document and silently ignores the rest.
# Scoring in overlapping chunks covers the whole text and shows which passages
# carry the risk, which is what you act on when revising.
# --------------------------------------------------------------------------


@dataclass
class Section:
    index: int
    text: str
    score: float
    is_ai: bool
    style: object | None = None  # features.StyleNote, kept loose to avoid a cycle

    @property
    def notes(self) -> list[str]:
        return list(self.style.observations) if self.style is not None else []

    @property
    def preview(self) -> str:
        flat = " ".join(self.text.split())
        return flat[:70] + ("..." if len(flat) > 70 else "")


@dataclass
class DocumentVerdict:
    detector: str
    threshold: float
    sections: list[Section]
    device: str | None = None

    @property
    def flagged(self) -> list[Section]:
        """Sections a detector would call AI-generated."""
        return [s for s in self.sections if s.is_ai]

    @property
    def mean_score(self) -> float:
        return sum(s.score for s in self.sections) / len(self.sections)

    @property
    def headline(self) -> str:
        """Plain-language summary. A count of flagged passages is what you act
        on; the mean score across a whole document rarely tells you anything."""
        if self.detector == "stylometry":
            noted = sum(1 for s in self.sections if s.notes)
            if noted == 0:
                return "No stylometric warning signs - sentence variety looks natural"
            return f"{noted} of {len(self.sections)} sections read as formulaic"
        n, total = len(self.flagged), len(self.sections)
        if n == 0:
            return f"Reads as human-written - none of {total} sections would be flagged"
        if n == total:
            return f"Reads as AI-generated - all {total} sections would be flagged"
        return f"{n} of {total} sections would be flagged"

    @property
    def riskiest(self) -> Section:
        """The section most likely to be flagged.

        Binoculars scores AI text low and FeatureDetector scores it high, so
        which end counts as risky depends on the detector.
        """
        pick = min if self.detector == "binoculars" else max
        return pick(self.sections, key=lambda s: s.score)

    def __str__(self) -> str:
        n, total = len(self.flagged), len(self.sections)
        if self.detector == "stylometry":
            noted = sum(1 for s in self.sections if s.notes)
            head = f"{noted}/{total} sections have something worth looking at"
        else:
            head = (
                f"{n}/{total} sections would be flagged  "
                f"mean={self.mean_score:.4f}  threshold={self.threshold:.4f}"
            )
        if self.device:
            head += f"  device={self.device}"
        rows = []
        for section in self.sections:
            mark = "FLAG" if section.is_ai else "    "
            rows.append(f"  [{section.index + 1:>2}] {section.score:.4f} {mark}  {section.preview}")
            if section.style is not None:
                rows.append(f"       {section.style}")
            # Style notes are actionable even where the score is fine, so they
            # show for every section rather than only the flagged ones.
            rows.extend(f"       - {note}" for note in section.notes)
        return "\n".join([head, *rows])


def score_document(
    text: str,
    detector: str = "binoculars",
    *,
    chunk_words: int = 300,
    overlap: int = 50,
    **kwargs,
) -> DocumentVerdict:
    """Score a document section by section, covering all of it.

    ``chunk_words`` defaults to 300 so each chunk stays inside the 512-token
    window with room to spare.
    """
    from .ensemble import chunk_text

    chunks = chunk_text(text, chunk_words=chunk_words, overlap=overlap)
    sections: list[Section] = []
    threshold = 0.0
    device = None

    from .features import describe_style

    for index, chunk in enumerate(chunks):
        verdict = score_text(chunk, detector, **kwargs)
        threshold = verdict.threshold
        device = verdict.device
        sections.append(
            Section(index, chunk, verdict.score, verdict.is_ai, describe_style(chunk))
        )

    return DocumentVerdict(detector, threshold, sections, device)


def describe_document(
    text: str, *, chunk_words: int = 300, overlap: int = 50
) -> DocumentVerdict:
    """Stylometric indicators only, section by section, with no detector at all.

    No model, no download, no GPU, no training data, and no verdict — just what
    the writing looks like. This is the whole of what the lite build can do
    unaided, and it is the part that transfers: every detector measures
    predictability, so a passage with very even sentence lengths reads as
    formulaic to all of them.

    ``is_ai`` is always False. Nothing here judges authorship, and a field that
    said otherwise would invite exactly the reading this avoids.
    """
    from .ensemble import chunk_text
    from .features import describe_style

    sections = [
        Section(index, chunk, 0.0, False, describe_style(chunk))
        for index, chunk in enumerate(chunk_text(text, chunk_words=chunk_words, overlap=overlap))
    ]
    return DocumentVerdict("stylometry", 0.0, sections)
