"""Threshold calibration from your own documents.

The published thresholds belong to the Falcon pair on the paper's benchmark.
Another pair, or another domain, sits on a different scale and needs its own.
"""

from __future__ import annotations

import pytest

from aidetect.calibrate import Calibration, calibrate


@pytest.fixture
def human_docs(tmp_path):
    """Twenty 'human' documents scoring high, as human text does."""
    folder = tmp_path / "human"
    folder.mkdir()
    for i in range(20):
        (folder / f"h{i}.txt").write_text(f"human document {i} " * 30, encoding="utf-8")
    return folder


@pytest.fixture
def ai_docs(tmp_path):
    folder = tmp_path / "ai"
    folder.mkdir()
    for i in range(20):
        (folder / f"a{i}.txt").write_text(f"ai document {i} " * 30, encoding="utf-8")
    return folder


def _fake_scorer(human_high=True):
    """Score by filename: 'h' files high (human), 'a' files low (AI)."""

    def score(text: str) -> float:
        return 0.98 - (0.0005 * len(text) % 0.02) if "human" in text else 0.70

    return score


def test_human_only_calibration_controls_false_positives(human_docs):
    """With no AI examples you can still fix the false positive rate."""
    result = calibrate(human_docs, score_fn=_fake_scorer(), max_fpr=0.05)

    assert isinstance(result, Calibration)
    assert result.n_human == 20
    assert result.n_ai == 0
    assert result.detection_rate is None, "cannot know detection rate without AI examples"
    assert result.observed_fpr <= 0.05 + 1e-9


def test_threshold_sits_below_almost_every_human_document(human_docs):
    result = calibrate(human_docs, score_fn=_fake_scorer(), max_fpr=0.01)
    below = [s for s in result.human_scores if s < result.threshold]
    assert len(below) <= 1


def test_tighter_budget_gives_a_lower_threshold(human_docs):
    """A stricter false positive budget must flag less, not more."""
    loose = calibrate(human_docs, score_fn=_fake_scorer(), max_fpr=0.20)
    tight = calibrate(human_docs, score_fn=_fake_scorer(), max_fpr=0.01)
    assert tight.threshold <= loose.threshold


def test_labelled_calibration_reports_auroc_and_detection(human_docs, ai_docs):
    result = calibrate(human_docs, ai_docs, score_fn=_fake_scorer(), max_fpr=0.01)
    assert result.n_ai == 20
    assert result.auroc is not None
    assert result.detection_rate is not None
    assert result.auroc == pytest.approx(1.0), "perfectly separable fixture"


def test_empty_folder_is_an_error(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(ValueError, match="No supported documents"):
        calibrate(empty, score_fn=_fake_scorer())


def test_summary_states_when_detection_rate_is_unknown(human_docs):
    text = str(calibrate(human_docs, score_fn=_fake_scorer()))
    assert "detection rate:    unknown" in text
    assert "false positives" in text


# ------------------------------------------------- storing and comparing


def test_calibration_keeps_the_whole_distribution(tmp_path):
    """Only storing the threshold throws away what a document is compared to."""
    from aidetect.calibrate import load_calibration, save_calibration

    path = tmp_path / "t.json"
    save_calibration("falcon-7b", 0.88, [0.9, 0.95, 1.0], 0.01, path=path)

    entry = load_calibration("falcon-7b", path=path)
    assert entry["threshold"] == 0.88
    assert entry["scores"] == [0.9, 0.95, 1.0]


def test_percentile_places_a_score_in_your_own_distribution(tmp_path):
    from aidetect.calibrate import percentile_of, save_calibration

    path = tmp_path / "t.json"
    save_calibration("falcon-7b", 0.9, [0.90, 0.92, 0.94, 0.96, 0.98], 0.01, path=path)

    assert percentile_of(0.80, "falcon-7b", path=path) == 0  # below everything
    assert percentile_of(1.00, "falcon-7b", path=path) == 100  # above everything
    assert percentile_of(0.95, "falcon-7b", path=path) == 60


def test_percentile_is_none_without_a_calibration(tmp_path):
    from aidetect.calibrate import percentile_of

    assert percentile_of(0.9, "falcon-7b", path=tmp_path / "missing.json") is None


def test_old_plain_number_format_still_loads(tmp_path):
    """A file written before distributions were stored must not break."""
    import json

    from aidetect.calibrate import load_threshold, percentile_of

    path = tmp_path / "t.json"
    path.write_text(json.dumps({"falcon-7b": 0.87}), encoding="utf-8")

    assert load_threshold("falcon-7b", path=path) == 0.87
    assert percentile_of(0.9, "falcon-7b", path=path) is None


def test_corrupt_file_does_not_block_recalibration(tmp_path):
    from aidetect.calibrate import load_threshold, save_calibration

    path = tmp_path / "t.json"
    path.write_text("{ not json", encoding="utf-8")
    save_calibration("falcon-7b", 0.9, [0.95], 0.01, path=path)
    assert load_threshold("falcon-7b", path=path) == 0.9


def test_comparison_wording_flips_around_the_median():
    from aidetect.scoring import Verdict

    low = Verdict("binoculars", 0.8, 0.9, is_ai=True, percentile=8.0)
    high = Verdict("binoculars", 1.0, 0.9, is_ai=False, percentile=65.0)
    assert "lower than 92%" in low.comparison
    assert "higher than 65%" in high.comparison
    assert Verdict("binoculars", 0.9, 0.9, is_ai=False).comparison == ""
