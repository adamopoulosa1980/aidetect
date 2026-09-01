"""Ensemble detector: stack scores from multiple detectors for max accuracy.

Binoculars and the feature classifier fail on *different* inputs (one is
distribution-based, one stylometric), so a logistic-regression stacker over
both scores typically beats either alone by several AUROC points. Also
provides chunked scoring for long documents — averaging chunk scores reduces
variance and improves accuracy on texts longer than the model context.

Design: works on score arrays, so it is detector-agnostic and CPU-testable.

Usage:
    # scores_matrix: shape (n_samples, n_detectors), higher = more AI
    # e.g. columns = [-binoculars_score, feature_prob]
    ens = ScoreEnsemble().fit(scores_matrix, labels)
    probs = ens.predict_proba(new_scores_matrix)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression


class ScoreEnsemble:
    """Calibrated logistic stacker over detector scores."""

    def __init__(self) -> None:
        self.model = LogisticRegression(max_iter=1000)
        self.fitted = False

    def fit(self, scores: np.ndarray, labels: list[int] | np.ndarray) -> "ScoreEnsemble":
        scores = np.atleast_2d(np.asarray(scores, dtype=float))
        self.model.fit(scores, np.asarray(labels, dtype=int))
        self.fitted = True
        return self

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        scores = np.atleast_2d(np.asarray(scores, dtype=float))
        return self.model.predict_proba(scores)[:, 1]

    def predict(self, scores: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(scores) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.model))

    @classmethod
    def load(cls, path: str | Path) -> "ScoreEnsemble":
        ens = cls()
        ens.model = pickle.loads(Path(path).read_bytes())
        ens.fitted = True
        return ens


def chunk_spans(text: str, chunk_words: int = 300, overlap: int = 50) -> list[tuple[int, int]]:
    """The ``(start_word, end_word)`` of each chunk ``chunk_text`` would return.

    Kept beside chunk_text and driven by the same arithmetic, because a section
    reported at the wrong offset sends someone to the wrong page -- which is
    worse than giving them no offset at all.
    """
    words = text.split()
    if len(words) <= chunk_words:
        return [(0, len(words))]
    step = chunk_words - overlap
    return [
        (start, min(start + chunk_words, len(words)))
        for start in range(0, len(words) - overlap, step)
    ]


def chunk_text(text: str, chunk_words: int = 300, overlap: int = 50) -> list[str]:
    """Split long text into overlapping word chunks."""
    words = text.split()
    if len(words) <= chunk_words:
        return [text]
    chunks, step = [], chunk_words - overlap
    for start in range(0, len(words) - overlap, step):
        chunks.append(" ".join(words[start : start + chunk_words]))
    return chunks


def score_long_text(score_fn, text: str, chunk_words: int = 300) -> dict:
    """Score a long document by chunking and aggregating.

    score_fn: callable text -> float (higher = more AI).
    Returns mean, max, and per-chunk scores. Mean is the accuracy-optimal
    aggregate for uniformly generated text; max catches partial AI insertion
    (e.g. one AI-written section in a human document).
    """
    chunks = chunk_text(text, chunk_words=chunk_words)
    scores = np.array([score_fn(c) for c in chunks], dtype=float)
    return {
        "mean": float(scores.mean()),
        "max": float(scores.max()),
        "n_chunks": len(chunks),
        "chunk_scores": scores.tolist(),
    }
