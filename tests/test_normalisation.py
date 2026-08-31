"""The same wording must score the same however the document arrived.

Whitespace is made of tokens, so it counts towards perplexity. Joining five
paragraphs with newlines rather than spaces moved the score by 3.8% in testing
— wider than the margin many documents sit from the threshold, so formatting
alone could flip a verdict.

pypdf emits a newline at every visual line wrap, python-docx at every
paragraph, and a paste carries whatever the clipboard held.
"""

from __future__ import annotations

import pytest

from aidetect.readers import canonical_text, normalise_text

PARAS = ["First paragraph here.", "Second paragraph here.", "Third one."]


@pytest.mark.parametrize(
    "separator",
    ["\n", "\n\n", "\n\n\n", " ", "  ", "\r\n", "\t", "\u00a0", " \n ", "\r"],
)
def test_every_separator_reaches_the_same_scored_string(separator):
    assert canonical_text(separator.join(PARAS)) == "First paragraph here. Second paragraph here. Third one."


def test_pdf_style_mid_sentence_wraps_are_flattened():
    """pypdf breaks lines by layout, not by meaning."""
    wrapped = "First paragraph\nhere. Second paragraph\nhere. Third one."
    assert canonical_text(wrapped) == canonical_text(" ".join(PARAS))


def test_display_form_keeps_paragraphs():
    """The window should stay readable; only the scored string is flattened."""
    assert normalise_text("\n".join(PARAS)) == "First paragraph here.\nSecond paragraph here.\nThird one."


def test_zero_width_characters_are_removed():
    """They carry no meaning but do become tokens."""
    assert canonical_text("word\u200bword") == "wordword"


def test_unicode_spaces_become_ordinary_spaces():
    assert canonical_text("a\u00a0b\u2009c\u3000d") == "a b c d"


def test_normalisation_is_idempotent():
    once = canonical_text("  messy\r\n\r\n  text\u00a0here  ")
    assert canonical_text(once) == once


def test_greek_text_survives_normalisation():
    greek = "Πρώτη παράγραφος.\r\n\r\nΔεύτερη παράγραφος;\u00a0Ναι."
    out = canonical_text(greek)
    assert out == "Πρώτη παράγραφος. Δεύτερη παράγραφος; Ναι."


def test_scoring_normalises_whatever_it_is_given(monkeypatch):
    """Two spellings of one document must not reach the detector differently."""
    from aidetect import scoring

    scoring.clear_model_cache()
    seen = []

    class Fake:
        def __init__(self, observer_name, performer_name):
            self.device = "cpu"
            self.threshold = 0.9

        def score(self, text):
            from aidetect.binoculars import BinocularsResult

            seen.append(text)
            return BinocularsResult(0.95, False, self.threshold, 2.0, 4.0)

    monkeypatch.setattr("aidetect.binoculars.Binoculars", Fake)
    body = " ".join(f"word{i}" for i in range(60))
    scoring.score_text(body.replace(" ", "\n"), "binoculars")
    scoring.score_text(body, "binoculars")

    assert seen[0] == seen[1], "the detector saw two different strings"
    scoring.clear_model_cache()
