"""CLI: score a text file or stdin for AI-generated text.

Usage:
    aidetect file.docx
    echo "some text" | aidetect
    aidetect file.pdf --detector features --model trained.pkl
    aidetect --gui

Running the frozen executable with no arguments opens the GUI, so that
double-clicking it does something useful rather than scoring an empty
stdin. Pipe into the frozen app with an explicit "-":
    echo "some text" | aidetect.exe -
"""

from __future__ import annotations

import argparse
import sys

from .models import PAIR_KEYS, resolve
from .readers import SUPPORTED_EXTENSIONS
from .scoring import DETECTORS, DetectorUnavailable, score_document, score_text

MIN_WORDS = 50

# Binoculars truncates at 512 tokens; in English that is roughly this many words.
CONTEXT_WORDS = 380


def _frozen() -> bool:
    return getattr(sys, "frozen", False)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    # A frozen app with no arguments was opened from Explorer or the Start
    # menu, not from a shell pipeline, so show the window. stdin.isatty() looks
    # like the more precise test but is not: a double-clicked executable does
    # not reliably get a tty, and reading stdin then returns EOF and scores the
    # empty string. Piping into the frozen app still works via an explicit "-".
    if _frozen() and not argv:
        from .gui import main as gui_main

        gui_main()
        return 0

    supported = " ".join(sorted(SUPPORTED_EXTENSIONS))
    from .scoring import detector_available

    description = "Score text for AI generation"
    if not detector_available("binoculars"):
        description += " (lite build: stylometric only, no Binoculars)"
    p = argparse.ArgumentParser(prog="aidetect", description=description)
    p.add_argument(
        "file",
        nargs="?",
        help=f"document to score ({supported}); '-' or omitted reads stdin",
    )
    p.add_argument("--gui", action="store_true", help="open the graphical interface")
    p.add_argument(
        "--detector",
        choices=DETECTORS,
        default="binoculars",
        help="binoculars: zero-shot, needs torch + a GPU. "
        "features: CPU stylometric, needs a trained --model",
    )
    p.add_argument("--model", help="trained FeatureDetector pickle (--detector features)")
    p.add_argument(
        "--save-md",
        nargs="?",
        const="",
        metavar="PATH",
        help="also write the document's Markdown rendering; defaults to beside the source",
    )
    p.add_argument(
        "--stylometry",
        action="store_true",
        help="report writing indicators only - burstiness, vocabulary range, "
        "repeated phrasing - with no model and no verdict. Works anywhere",
    )
    p.add_argument(
        "--sections",
        action="store_true",
        help="score the whole document section by section, rather than only the "
        "first ~380 words that fit the model's context",
    )
    p.add_argument(
        "--mode",
        choices=("accuracy", "low-fpr"),
        default="accuracy",
        help="accuracy: the reference default, balances both error types. "
        "low-fpr: a lower threshold, so fewer documents are called AI-generated",
    )
    p.add_argument(
        "--pair",
        choices=PAIR_KEYS,
        help="named model pair; overrides --observer/--performer. "
        "Only falcon-7b matches the published thresholds",
    )
    p.add_argument("--observer", default="tiiuae/falcon-7b")
    p.add_argument("--performer", default="tiiuae/falcon-7b-instruct")
    p.add_argument("--threshold", type=float, default=None)
    args = p.parse_args(argv)

    if args.gui:
        from .gui import main as gui_main

        gui_main()
        return 0

    if args.pair:
        pair = resolve(args.pair)
        args.observer, args.performer = pair.observer, pair.performer
        if not pair.calibrated and args.threshold is None:
            print(
                f"warning: {pair.label} has no calibrated threshold; "
                "the published cut-offs were derived for Falcon 7B only",
                file=sys.stderr,
            )

    if args.save_md is not None and (not args.file or args.file == "-"):
        p.error("--save-md needs a document to convert; stdin has no source file")

    if args.file and args.file != "-":
        from .readers import load_text

        try:
            text = load_text(args.file)
        except (OSError, ValueError, ImportError) as e:
            print(f"aidetect: {e}", file=sys.stderr)
            return 1

        if args.save_md is not None:
            from .readers import save_markdown

            try:
                written = save_markdown(args.file, args.save_md or None)
            except (OSError, ValueError, ImportError) as e:
                print(f"aidetect: could not write Markdown: {e}", file=sys.stderr)
                return 1
            print(f"markdown written to {written}", file=sys.stderr)
    else:
        text = sys.stdin.read()
        if not text.strip():
            print("aidetect: no text on stdin", file=sys.stderr)
            return 1

    words = len(text.split())
    if words < MIN_WORDS:
        print(
            f"warning: <{MIN_WORDS} words - scores are unreliable on short text",
            file=sys.stderr,
        )
    elif words > CONTEXT_WORDS and not args.sections:
        print(
            f"warning: {words} words but only the first ~{CONTEXT_WORDS} are scored; "
            "pass --sections to score the whole document",
            file=sys.stderr,
        )

    if args.stylometry:
        from .scoring import describe_document

        print(describe_document(text))
        return 0

    scorer = score_document if args.sections else score_text
    try:
        verdict = scorer(
            text,
            args.detector,
            model=args.model,
            observer=args.observer,
            performer=args.performer,
            threshold=args.threshold,
            mode=args.mode,
        )
    except DetectorUnavailable as e:
        print(f"aidetect: {e}", file=sys.stderr)
        return 2

    print(verdict)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
