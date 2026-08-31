"""Lightweight supervised detector: statistical features + gradient boosting.

Train on labeled pairs (text, label) where label 1 = AI, 0 = human.
Fast, CPU-only, interpretable — but domain-sensitive. Retrain per domain.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_NAMES, extract_features


class FeatureDetector:
    def __init__(self) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", HistGradientBoostingClassifier(max_iter=300, random_state=0)),
            ]
        )
        self.fitted = False

    def fit(self, texts: list[str], labels: list[int]) -> "FeatureDetector":
        X = np.array([extract_features(t) for t in texts])
        self.pipeline.fit(X, np.array(labels))
        self.fitted = True
        return self

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        """Probability that each text is AI-generated."""
        X = np.array([extract_features(t) for t in texts])
        return self.pipeline.predict_proba(X)[:, 1]

    def predict(self, texts: list[str], threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(texts) >= threshold).astype(int)

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(pickle.dumps(self.pipeline))

    @classmethod
    def load(cls, path: str | Path) -> "FeatureDetector":
        det = cls()
        det.pipeline = pickle.loads(Path(path).read_bytes())
        det.fitted = True
        return det

    @property
    def feature_names(self) -> list[str]:
        return FEATURE_NAMES
