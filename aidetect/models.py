"""Catalogue of model pairs Binoculars can run on.

Deliberately free of torch and transformers imports: the CLI and the GUI need
this list to build their menus, and the lite executable ships without either.

Two constraints govern what can appear here.

The pair must share a tokenizer. Only the observer's tokenizer is loaded and
the same token ids go to both models, so a mismatched pair scores one model on
the other's ids — no error, just a meaningless number. Base/instruct siblings
from one family always match.

The published thresholds apply to the Falcon pair alone. They were derived from
that pair's score distribution, and every other pair sits on a different scale.
Listing a pair here is not a claim that its verdicts mean anything against the
default threshold, which is why each one says so.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPair:
    key: str
    label: str
    observer: str
    performer: str
    approx_gb: float
    calibrated: bool

    @property
    def note(self) -> str:
        return (
            "reference pair, published thresholds apply"
            if self.calibrated
            else "thresholds NOT calibrated for this pair"
        )

    @property
    def display(self) -> str:
        return f"{self.label} (~{self.approx_gb:g}GB download, {self.note})"


MODEL_PAIRS = (
    ModelPair(
        "falcon-7b",
        "Falcon 7B",
        "tiiuae/falcon-7b",
        "tiiuae/falcon-7b-instruct",
        28,
        calibrated=True,
    ),
    ModelPair(
        "qwen2.5-7b",
        "Qwen2.5 7B",
        "Qwen/Qwen2.5-7B",
        "Qwen/Qwen2.5-7B-Instruct",
        30,
        calibrated=False,
    ),
    ModelPair(
        "qwen2.5-1.5b",
        "Qwen2.5 1.5B",
        "Qwen/Qwen2.5-1.5B",
        "Qwen/Qwen2.5-1.5B-Instruct",
        6,
        calibrated=False,
    ),
    ModelPair(
        "qwen2.5-0.5b",
        "Qwen2.5 0.5B",
        "Qwen/Qwen2.5-0.5B",
        "Qwen/Qwen2.5-0.5B-Instruct",
        2,
        calibrated=False,
    ),
)

PAIRS_BY_KEY = {pair.key: pair for pair in MODEL_PAIRS}
PAIR_KEYS = tuple(PAIRS_BY_KEY)
DEFAULT_PAIR = MODEL_PAIRS[0]


def resolve(key: str) -> ModelPair:
    try:
        return PAIRS_BY_KEY[key]
    except KeyError:
        raise ValueError(f"Unknown model pair '{key}'. Choose from {PAIR_KEYS}.") from None


def find_by_models(observer: str, performer: str) -> ModelPair | None:
    """The catalogue entry for these two models, if it is one we list."""
    for pair in MODEL_PAIRS:
        if pair.observer == observer and pair.performer == performer:
            return pair
    return None
