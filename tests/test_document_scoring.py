"""Per-section scoring of whole documents.

A single Binoculars call truncates at 512 tokens, roughly 380 words, so a long
document was judged on its first page alone. Sectioned scoring covers all of it
and shows which passages would be flagged.
"""

from __future__ import annotations

import pytest

from aidetect.scoring import DocumentVerdict, Section, score_document


@pytest.fixture
def trained_model(tmp_path):
    from aidetect import FeatureDetector

    human = ["Short. Then a far longer rambling sentence that wanders about! Why? Nobody knows."] * 15
    ai = ["The study measured rainfall at three upland sites."] * 15
    path = tmp_path / "m.pkl"
    FeatureDetector().fit(human + ai, [0] * 15 + [1] * 15).save(path)
    return str(path)


def test_long_document_is_split_into_several_sections(trained_model):
    """A 5000-word document must not be judged on its first 380 words."""
    text = "The study measured rainfall at three upland sites each season. " * 500
    result = score_document(text, "features", model=trained_model, chunk_words=300)

    assert isinstance(result, DocumentVerdict)
    assert len(result.sections) > 10, "long document collapsed into too few sections"
    assert all(isinstance(s, Section) for s in result.sections)


def test_short_document_is_a_single_section(trained_model):
    text = "word " * 100
    result = score_document(text, "features", model=trained_model, chunk_words=300)
    assert len(result.sections) == 1


def test_every_section_is_covered(trained_model):
    """Chunks must span the document, not sample it."""
    words = [f"w{i}" for i in range(1200)]
    text = " ".join(words)
    result = score_document(text, "features", model=trained_model, chunk_words=300, overlap=50)

    joined = " ".join(s.text for s in result.sections)
    assert "w0" in joined
    assert "w1199" in joined, "the end of the document was never scored"


def test_flagged_lists_only_sections_over_threshold(trained_model):
    text = "The study measured rainfall at three upland sites. " * 200
    result = score_document(text, "features", model=trained_model)
    assert result.flagged == [s for s in result.sections if s.is_ai]


def test_riskiest_respects_detector_direction():
    """Binoculars scores AI low; FeatureDetector scores it high."""
    sections = [
        Section(0, "a", 0.20, False),
        Section(1, "b", 0.90, False),
    ]
    binoc = DocumentVerdict("binoculars", 0.9015, sections)
    feats = DocumentVerdict("features", 0.5, sections)

    assert binoc.riskiest.score == 0.20  # low = AI-like for binoculars
    assert feats.riskiest.score == 0.90  # high = AI-like for features


def test_summary_reports_counts_and_previews():
    sections = [Section(0, "the study measured rainfall at three sites", 0.8, True)]
    text = str(DocumentVerdict("binoculars", 0.9015, sections, device="cpu"))
    assert "1/1 sections would be flagged" in text
    assert "FLAG" in text
    assert "device=cpu" in text


def test_preview_is_truncated_and_flattened():
    s = Section(0, "one\ntwo   three " + "x" * 200, 0.5, False)
    assert "\n" not in s.preview
    assert s.preview.endswith("...")
    assert len(s.preview) <= 73


def test_mode_selects_the_published_threshold(monkeypatch):
    """low-fpr must actually lower the bar for calling something AI."""
    from aidetect import binoculars, scoring

    scoring.clear_model_cache()

    class Fake:
        def __init__(self, observer_name, performer_name):
            self.device = "cpu"
            self.threshold = 0.0

        def score(self, text):
            from aidetect.binoculars import BinocularsResult

            # score sits between the two published thresholds
            s = 0.88
            return BinocularsResult(s, s < self.threshold, self.threshold, 2.0, 4.0)

    monkeypatch.setattr("aidetect.binoculars.Binoculars", Fake)

    strict = scoring.score_text("word " * 60, "binoculars", mode="accuracy")
    lenient = scoring.score_text("word " * 60, "binoculars", mode="low-fpr")

    assert strict.threshold == binoculars.ACCURACY_THRESHOLD
    assert lenient.threshold == binoculars.FPR_THRESHOLD
    # 0.88 is below 0.9015 but above 0.8536
    assert strict.is_ai is True
    assert lenient.is_ai is False
    scoring.clear_model_cache()


def test_unknown_mode_is_rejected(monkeypatch):
    from aidetect import scoring

    scoring.clear_model_cache()

    class Fake:
        def __init__(self, observer_name, performer_name):
            self.device = "cpu"
            self.threshold = 0.0

        def score(self, text):
            from aidetect.binoculars import BinocularsResult

            return BinocularsResult(0.5, True, self.threshold, 2.0, 4.0)

    monkeypatch.setattr("aidetect.binoculars.Binoculars", Fake)
    with pytest.raises(ValueError, match="Unknown mode"):
        scoring.score_text("word " * 60, "binoculars", mode="nonsense")
    scoring.clear_model_cache()
