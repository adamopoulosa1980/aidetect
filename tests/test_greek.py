"""Greek and English punctuation are not interchangeable.

';' ends a question in Greek and joins clauses in English. Splitting Greek with
the English pattern merges sentences, which inflates their measured length and
distorts every length-based feature, burstiness above all.
"""

from __future__ import annotations

import pytest

from aidetect.features import (
    FEATURE_NAMES,
    describe_style,
    extract_features,
    is_greek,
    split_sentences,
)

GREEK = "Πήγα εκεί και ήταν χάος. Κομμάτια παντού! Ποιος φταίει; Κανείς δεν ήξερε."
ENGLISH = "I went there and it was chaos. Bits everywhere! One clause; then another."


def test_greek_question_mark_ends_a_sentence():
    assert len(split_sentences(GREEK)) == 4


def test_english_semicolon_does_not_end_a_sentence():
    assert len(split_sentences(ENGLISH)) == 3


def test_script_is_detected_from_the_letters():
    assert is_greek(GREEK)
    assert not is_greek(ENGLISH)


def test_greek_with_english_technical_terms_still_reads_as_greek():
    mixed = "Η μελέτη χρησιμοποιεί δεδομένα LIDAR και GPS. Είναι αξιόπιστα; Ναι."
    assert is_greek(mixed)
    assert len(split_sentences(mixed)) == 3


def test_semicolon_counts_as_a_question_in_greek():
    greek = dict(zip(FEATURE_NAMES, extract_features(GREEK)))
    assert greek["question_rate"] > 0, "';' is a Greek question mark"
    assert greek["semicolon_rate"] == 0


def test_semicolon_counts_as_a_semicolon_in_english():
    english = dict(zip(FEATURE_NAMES, extract_features(ENGLISH)))
    assert english["semicolon_rate"] > 0
    assert english["question_rate"] == 0


def test_burstiness_is_measurable_on_greek():
    uniform = "Η μελέτη κατέγραψε βροχόπτωση. Η μελέτη κατέγραψε θερμοκρασία. " * 6
    varied = (
        "Χάος. Πήγα εκεί και ήταν το πιο ακατάστατο πράγμα που έχω δει σε δώδεκα "
        "χρόνια αυτής της δουλειάς, κομμάτια παντού! Γιατί; Κανείς δεν ήξερε."
    )
    assert describe_style(uniform).burstiness < describe_style(varied).burstiness


def test_greek_words_are_tokenised():
    """Greek letters must reach the word features, not be dropped as punctuation."""
    style = describe_style(GREEK)
    assert style.mean_sentence_len > 0
    assert 0 < style.type_token_ratio <= 1
