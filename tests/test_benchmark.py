"""The benchmark harness.

BENCHMARK.md explains why no results ship with this project. This is the tool
that produces them, so it has to be right about the two things that are easy to
get wrong: the direction of the score, and keeping conditions separate.
"""

from __future__ import annotations

import pytest

from aidetect.benchmark import ConditionResult, as_markdown, find_conditions, run_condition


def _corpus(root, conditions):
    """Build a corpus tree. Human documents get varied sentence lengths."""
    for name, (n_human, n_ai) in conditions.items():
        base = root / name if name else root
        for side, count, body in (
            ("human", n_human, "Short. Then a far longer rambling sentence that wanders! Why?"),
            ("ai", n_ai, "The study measured rainfall at three upland sites."),
        ):
            folder = base / side
            folder.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                (folder / f"{i}.txt").write_text(f"{body} {i} ", encoding="utf-8")
    return root


def test_single_condition_is_found_directly(tmp_path):
    _corpus(tmp_path, {"": (3, 3)})
    found = find_conditions(tmp_path)
    assert len(found) == 1
    assert found[0][0] == tmp_path.name


def test_each_subdirectory_becomes_its_own_condition(tmp_path):
    _corpus(tmp_path, {"original": (3, 3), "paraphrased": (3, 3), "non-native": (3, 3)})
    names = [name for name, _, _ in find_conditions(tmp_path)]
    assert names == ["non-native", "original", "paraphrased"], "sorted, one per condition"


def test_a_corpus_without_the_expected_layout_is_rejected(tmp_path):
    (tmp_path / "documents").mkdir()
    with pytest.raises(ValueError, match="No conditions found"):
        find_conditions(tmp_path)


def test_perfect_separation_scores_auroc_one(tmp_path):
    """Score direction must be right: AI low, human high, as Binoculars reports."""
    _corpus(tmp_path, {"": (5, 5)})
    name, human_dir, ai_dir = find_conditions(tmp_path)[0]

    def score_fn(text: str) -> float:
        return 1.05 if "rambling" in text else 0.70  # human high, AI low

    result = run_condition(name, human_dir, ai_dir, score_fn=score_fn)
    assert result.auroc == pytest.approx(1.0)
    assert result.fpr == pytest.approx(0.0)
    assert result.tpr == pytest.approx(1.0)


def test_a_useless_detector_scores_auroc_half(tmp_path):
    _corpus(tmp_path, {"": (6, 6)})
    name, human_dir, ai_dir = find_conditions(tmp_path)[0]
    result = run_condition(name, human_dir, ai_dir, score_fn=lambda _t: 0.9)
    assert result.auroc == pytest.approx(0.5)


def test_a_condition_missing_one_side_is_rejected(tmp_path):
    _corpus(tmp_path, {"": (3, 3)})
    name, human_dir, ai_dir = find_conditions(tmp_path)[0]
    for f in ai_dir.iterdir():
        f.unlink()
    with pytest.raises(ValueError, match="both sides"):
        run_condition(name, human_dir, ai_dir, score_fn=lambda _t: 0.9)


def test_table_names_the_worst_condition_rather_than_pooling(tmp_path):
    """Pooling hides the cases a detector actually fails on."""
    results = [
        ConditionResult("original", 50, 50, 0.97, 0.9, 0.01, 0.94),
        ConditionResult("non-native", 50, 50, 0.61, 0.9, 0.38, 0.55),
    ]
    table = as_markdown(results, 0.01)
    assert "Most false positives: **non-native** at 38.0%" in table
    assert "Weakest separation: **non-native**" in table
    assert "pooled figure would hide them" in table


def test_single_condition_table_has_no_comparison_footer():
    table = as_markdown([ConditionResult("only", 5, 5, 0.9, 0.9, 0.0, 1.0)], 0.01)
    assert "Most false positives" not in table
    assert "| only | 5 | 5 |" in table
