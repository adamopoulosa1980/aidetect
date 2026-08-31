"""Derive a threshold for a model pair from your own documents.

The published thresholds belong to the Falcon pair on the paper's benchmark.
Any other pair, or any domain whose scores sit on a different scale, needs its
own. This works out what that is.

Two ways in, depending on what you have.

**Human documents only** — the usual case, and enough. You want to know how
often your own writing would be falsely flagged, so score documents you know
are human and put the threshold where only ``max_fpr`` of them fall below it.
That fixes the false positive rate by construction. You give up knowing how
much real AI text it catches, which is the half you were not asking about.
Essays or reports written before ~2022 are a good source: pre-ChatGPT, the
same subject area, the same house style.

**Human and AI documents** — the full picture. Adds AUROC and the detection
rate, via evaluate.pick_threshold().

Neither can be done from a single example. A threshold separates two
distributions; one point does not define one.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Calibration:
    pair: str
    threshold: float
    max_fpr: float
    n_human: int
    n_ai: int
    human_scores: list[float]
    ai_scores: list[float]
    auroc: float | None = None
    detection_rate: float | None = None

    @property
    def observed_fpr(self) -> float:
        """How many known-human documents this threshold would still flag."""
        if not self.human_scores:
            return float("nan")
        flagged = sum(1 for s in self.human_scores if s < self.threshold)
        return flagged / len(self.human_scores)

    def __str__(self) -> str:
        lines = [
            f"pair:              {self.pair}",
            f"threshold:         {self.threshold:.10f}",
            f"human documents:   {self.n_human}",
            f"  score range:     {min(self.human_scores):.4f} - {max(self.human_scores):.4f}",
            f"  median:          {float(np.median(self.human_scores)):.4f}",
            f"false positives:   {self.observed_fpr:.1%} at this threshold "
            f"(budget {self.max_fpr:.1%})",
        ]
        if self.n_ai:
            lines += [
                f"AI documents:      {self.n_ai}",
                f"  score range:     {min(self.ai_scores):.4f} - {max(self.ai_scores):.4f}",
                f"detection rate:    {self.detection_rate:.1%}",
                f"AUROC:             {self.auroc:.4f}",
            ]
        else:
            lines.append(
                "detection rate:    unknown - no AI examples given, so this "
                "controls false positives only"
            )
        return "\n".join(lines)


# Calibrating and then having to copy a long decimal into every later command
# is a good way to get it wrong once and never notice. The result is saved here
# instead, per model pair, and picked up automatically.
CONFIG_PATH = Path.home() / ".aidetect" / "thresholds.json"


def save_calibration(
    pair: str,
    threshold: float,
    scores: list[float],
    max_fpr: float,
    path: Path | None = None,
) -> Path:
    """Remember a calibration so it is used without being retyped.

    The whole score distribution is kept, not only the threshold, so a later
    document can be placed within it. "Lower than 88% of our own writing"
    survives an unknown detector and an unknown threshold; a bare verdict does
    not.
    """
    import json

    target = path or CONFIG_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    stored = _read(target)
    stored[pair] = {
        "threshold": threshold,
        "max_fpr": max_fpr,
        "scores": sorted(float(s) for s in scores),
    }
    target.write_text(json.dumps(stored, indent=2), encoding="utf-8")
    return target


def _read(target: Path) -> dict:
    import json

    if not target.exists():
        return {}
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}  # a corrupt file must not block recalibration
    return data if isinstance(data, dict) else {}


def load_calibration(pair: str, path: Path | None = None) -> dict | None:
    """The stored calibration for ``pair``, or None if never calibrated."""
    entry = _read(path or CONFIG_PATH).get(pair)
    if isinstance(entry, (int, float)):
        return {"threshold": float(entry), "scores": [], "max_fpr": None}
    if isinstance(entry, dict) and "threshold" in entry:
        return entry
    return None


def load_threshold(pair: str, path: Path | None = None) -> float | None:
    entry = load_calibration(pair, path)
    return float(entry["threshold"]) if entry else None


def percentile_of(score: float, pair: str, path: Path | None = None) -> float | None:
    """Where ``score`` sits among your own documents, as a percentage.

    0 means lower than everything you calibrated on — the most machine-like.
    None when there is no calibration to compare against.
    """
    entry = load_calibration(pair, path)
    if not entry or not entry.get("scores"):
        return None
    scores = entry["scores"]
    below = sum(1 for s in scores if s < score)
    return 100.0 * below / len(scores)


def _score_folder(folder: Path, score_fn) -> list[float]:
    from .readers import load_directory

    documents = load_directory(folder)
    if not documents:
        raise ValueError(f"No supported documents found in {folder}")
    return [score_fn(text) for text in documents.values()]


def calibrate(
    human_dir: str | Path,
    ai_dir: str | Path | None = None,
    *,
    score_fn=None,
    pair: str = "falcon-7b",
    max_fpr: float = 0.01,
) -> Calibration:
    """Find the threshold for ``pair`` that holds false positives to ``max_fpr``.

    ``score_fn`` maps text to a Binoculars score; injected so this is testable
    without loading 28GB of weights.
    """
    if score_fn is None:
        from .models import resolve
        from .scoring import score_text

        chosen = resolve(pair)

        def score_fn(text: str) -> float:
            return score_text(
                text,
                "binoculars",
                observer=chosen.observer,
                performer=chosen.performer,
            ).score

    human = _score_folder(Path(human_dir), score_fn)
    ai = _score_folder(Path(ai_dir), score_fn) if ai_dir else []

    if ai:
        # Binoculars scores AI low, but pick_threshold expects higher = more AI,
        # so both sides are negated and the result flipped back.
        from .evaluate import pick_threshold

        scores = np.array([-s for s in human + ai])
        labels = np.array([0] * len(human) + [1] * len(ai))
        threshold = -pick_threshold(scores, labels, max_fpr=max_fpr)

        from .evaluate import evaluate

        report = evaluate(scores, labels, threshold=-threshold, max_fpr=max_fpr)
        auroc, detection = report.auroc, report.tpr_at_threshold
    else:
        # One-sided: put the cut where only max_fpr of known-human documents
        # fall below it.
        threshold = float(np.quantile(human, max_fpr))
        auroc = detection = None

    return Calibration(
        pair=pair,
        threshold=threshold,
        max_fpr=max_fpr,
        n_human=len(human),
        n_ai=len(ai),
        human_scores=human,
        ai_scores=ai,
        auroc=auroc,
        detection_rate=detection,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .models import PAIR_KEYS

    p = argparse.ArgumentParser(
        prog="aidetect-calibrate",
        description="Derive a detection threshold from your own documents",
    )
    p.add_argument(
        "folder",
        metavar="FOLDER",
        help="folder of documents you know were written by people "
        "(essays or reports from before ~2022 are a safe source)",
    )
    p.add_argument("--ai", metavar="DIR", help="documents known to be AI-generated (optional)")
    p.add_argument("--pair", choices=PAIR_KEYS, default="falcon-7b")
    p.add_argument(
        "--max-fpr",
        type=float,
        default=0.01,
        help="share of human documents you accept being flagged (default 0.01)",
    )
    args = p.parse_args(argv)

    try:
        result = calibrate(args.folder, args.ai, pair=args.pair, max_fpr=args.max_fpr)
    except (OSError, ValueError) as e:
        print(f"aidetect-calibrate: {e}", file=sys.stderr)
        return 1

    print(result)
    if result.n_human < 20:
        print(
            f"\nwarning: {result.n_human} human documents is thin for a "
            f"{args.max_fpr:.0%} estimate; 30 or more is steadier",
            file=sys.stderr,
        )
    print(f"\nUse it with:  aidetect DOC --pair {args.pair} --threshold {result.threshold:.10f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
