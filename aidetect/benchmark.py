"""Run a benchmark over labelled corpora and print a table you can publish.

BENCHMARK.md explains why no results ship with this project: benchmarking needs
labelled documents in the domain being judged, which the author of a tool cannot
supply for someone else's use of it. This is the harness, so that anyone who has
such documents can produce the numbers rather than take a claim on trust.

Conditions are reported separately and never pooled. A single headline AUROC
hides exactly the cases that break a detector - paraphrased output, non-native
writers, formal registers - and those are the cases worth knowing about.

Layout:

    corpus/
      original-ai/     human/*.docx   ai/*.docx
      paraphrased/     human/*.docx   ai/*.docx
      non-native/      human/*.docx   ai/*.docx

or a single condition:

    corpus/           human/*.docx   ai/*.docx
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class ConditionResult:
    name: str
    n_human: int
    n_ai: int
    auroc: float
    threshold: float
    fpr: float
    tpr: float

    @property
    def row(self) -> str:
        return (
            f"| {self.name} | {self.n_human} | {self.n_ai} | {self.auroc:.3f} | "
            f"{self.fpr:.1%} | {self.tpr:.1%} |"
        )


HEADER = "| Condition | Human | AI | AUROC | FPR | TPR |"
DIVIDER = "|---|---:|---:|---:|---:|---:|"


def find_conditions(root: Path) -> list[tuple[str, Path, Path]]:
    """Locate (name, human_dir, ai_dir) triples under ``root``."""
    if (root / "human").is_dir() and (root / "ai").is_dir():
        return [(root.name, root / "human", root / "ai")]

    found = []
    for child in sorted(p for p in root.iterdir() if p.is_dir()):
        if (child / "human").is_dir() and (child / "ai").is_dir():
            found.append((child.name, child / "human", child / "ai"))
    if not found:
        raise ValueError(
            f"No conditions found under {root}. Expected 'human' and 'ai' "
            "directories, either directly inside it or inside per-condition "
            "subdirectories."
        )
    return found


def run_condition(
    name: str,
    human_dir: Path,
    ai_dir: Path,
    *,
    score_fn,
    max_fpr: float = 0.01,
) -> ConditionResult:
    """Score one condition and evaluate it at a false-positive budget."""
    from .evaluate import evaluate, pick_threshold
    from .readers import load_directory

    human = [score_fn(t) for t in load_directory(human_dir).values()]
    ai = [score_fn(t) for t in load_directory(ai_dir).values()]
    if not human or not ai:
        raise ValueError(f"Condition '{name}' needs documents on both sides.")

    # Binoculars scores AI low; evaluate() expects higher = more AI, so both
    # sides are negated and the threshold flipped back for reporting.
    scores = np.array([-s for s in human + ai])
    labels = np.array([0] * len(human) + [1] * len(ai))

    threshold = pick_threshold(scores, labels, max_fpr=max_fpr)
    report = evaluate(scores, labels, threshold=threshold, max_fpr=max_fpr)

    return ConditionResult(
        name=name,
        n_human=len(human),
        n_ai=len(ai),
        auroc=report.auroc,
        threshold=-threshold,
        fpr=report.fpr_at_threshold,
        tpr=report.tpr_at_threshold,
    )


def as_markdown(results: list[ConditionResult], max_fpr: float) -> str:
    lines = [
        f"Threshold chosen per condition for FPR <= {max_fpr:.0%}.",
        "",
        HEADER,
        DIVIDER,
        *(r.row for r in results),
    ]
    if len(results) > 1:
        worst = max(results, key=lambda r: r.fpr)
        weakest = min(results, key=lambda r: r.auroc)
        lines += [
            "",
            f"Most false positives: **{worst.name}** at {worst.fpr:.1%}.",
            f"Weakest separation: **{weakest.name}** at AUROC {weakest.auroc:.3f}.",
            "",
            "Report these per condition. A pooled figure would hide them.",
        ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys

    from .models import PAIR_KEYS, resolve

    p = argparse.ArgumentParser(
        prog="aidetect-benchmark",
        description="Evaluate detectors against labelled corpora, per condition",
    )
    p.add_argument("corpus", metavar="DIR", help="corpus root (see --help for layout)")
    p.add_argument("--pair", choices=PAIR_KEYS, default="falcon-7b")
    p.add_argument("--model", help="trained classifier, to benchmark that instead")
    p.add_argument(
        "--max-fpr",
        type=float,
        default=0.01,
        help="false-positive budget the threshold is chosen under (default 0.01)",
    )
    p.add_argument("--output", metavar="FILE", help="write the table here as well")
    args = p.parse_args(argv)

    from .scoring import score_text

    if args.model:

        def score_fn(text: str) -> float:
            # negated so that, as with Binoculars, lower means more AI-like
            return -score_text(text, "features", model=args.model).score

    else:
        pair = resolve(args.pair)

        def score_fn(text: str) -> float:
            return score_text(
                text, "binoculars", observer=pair.observer, performer=pair.performer
            ).score

    try:
        conditions = find_conditions(Path(args.corpus))
    except (OSError, ValueError) as e:
        print(f"aidetect-benchmark: {e}", file=sys.stderr)
        return 1

    results = []
    for name, human_dir, ai_dir in conditions:
        print(f"scoring {name}...", file=sys.stderr)
        try:
            results.append(
                run_condition(name, human_dir, ai_dir, score_fn=score_fn, max_fpr=args.max_fpr)
            )
        except (OSError, ValueError) as e:
            print(f"aidetect-benchmark: {name}: {e}", file=sys.stderr)
            return 1

    table = as_markdown(results, args.max_fpr)
    print(table)
    if args.output:
        Path(args.output).write_text(table + "\n", encoding="utf-8")
        print(f"\nwritten to {args.output}", file=sys.stderr)

    if min(r.n_human for r in results) < 30:
        print(
            "\nwarning: fewer than 30 documents in a class makes an FPR "
            "estimate too noisy to publish",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
