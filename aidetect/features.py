"""Statistical text features for a lightweight, CPU-only detector.

These capture the stylometric signals that differ between human and LLM
text: burstiness (humans vary sentence length more), lexical diversity,
punctuation habits, and function-word usage. No model downloads needed —
useful as a fast pre-filter or an interpretable baseline.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

FEATURE_NAMES = [
    "mean_sentence_len",
    "std_sentence_len",
    "burstiness",
    "type_token_ratio",
    "hapax_ratio",
    "mean_word_len",
    "std_word_len",
    "comma_rate",
    "semicolon_rate",
    "question_rate",
    "exclam_rate",
    "dash_rate",
    "stopword_rate",
    "entropy_unigram",
    "repeated_bigram_rate",
]

_STOPWORDS = frozenset(
    """the a an and or but if then of to in on at by for with from as is are was
    were be been being it its this that these those i you he she we they not no
    so very just also can could would should will may might do does did have has
    had""".split()
)

_WORD = re.compile(r"[A-Za-z\u0370-\u03FF\u1F00-\u1FFF']+")  # Latin + Greek

# Sentence boundaries differ by script, and getting this wrong distorts every
# length-based feature. Greek asks questions with ';' (U+003B, or U+037E), which
# in English is a semicolon and does not end a sentence \u2014 so the two scripts
# cannot share one pattern. Greek separates clauses with the ano teleia '\u00B7'
# where English uses ';'.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_SENT_SPLIT_GREEK = re.compile(r"(?<=[.!?;\u037E])\s+")

_GREEK_CHARS = re.compile(r"[\u0370-\u03FF\u1F00-\u1FFF]")
_LATIN_CHARS = re.compile(r"[A-Za-z]")

GREEK_QUESTION_MARKS = (";", "\u037E")
GREEK_SEMICOLON = "\u0387"  # ano teleia


def is_greek(text: str) -> bool:
    """Whether to read this passage under Greek punctuation rules.

    Majority of letters wins, which handles a Greek document quoting English
    technical terms. Sections are classified independently, so a mixed document
    is still read correctly section by section.
    """
    return len(_GREEK_CHARS.findall(text)) > len(_LATIN_CHARS.findall(text))


def split_sentences(text: str) -> list[str]:
    """Sentences, using the punctuation rules of the passage's script."""
    pattern = _SENT_SPLIT_GREEK if is_greek(text) else _SENT_SPLIT
    return [s for s in pattern.split(text.strip()) if s.strip()]


def extract_features(text: str) -> list[float]:
    greek = is_greek(text)
    sentences = split_sentences(text)
    words = [w.lower() for w in _WORD.findall(text)]
    n_words = max(len(words), 1)
    n_chars = max(len(text), 1)

    sent_lens = [len(_WORD.findall(s)) for s in sentences] or [0]
    mean_sl = sum(sent_lens) / len(sent_lens)
    var_sl = sum((x - mean_sl) ** 2 for x in sent_lens) / len(sent_lens)
    std_sl = math.sqrt(var_sl)
    # Burstiness index in [-1, 1]: negative = regular (LLM-like), positive = bursty
    burstiness = (std_sl - mean_sl) / (std_sl + mean_sl) if (std_sl + mean_sl) > 0 else 0.0

    counts = Counter(words)
    ttr = len(counts) / n_words
    hapax = sum(1 for c in counts.values() if c == 1) / n_words

    word_lens = [len(w) for w in words] or [0]
    mean_wl = sum(word_lens) / len(word_lens)
    std_wl = math.sqrt(sum((x - mean_wl) ** 2 for x in word_lens) / len(word_lens))

    stopword_rate = sum(1 for w in words if w in _STOPWORDS) / n_words

    entropy = -sum((c / n_words) * math.log2(c / n_words) for c in counts.values())

    bigrams = list(zip(words, words[1:]))
    bg_counts = Counter(bigrams)
    repeated_bg = (
        sum(c for c in bg_counts.values() if c > 1) / max(len(bigrams), 1)
    )

    return [
        mean_sl,
        std_sl,
        burstiness,
        ttr,
        hapax,
        mean_wl,
        std_wl,
        text.count(",") / n_chars * 1000,
        # In Greek ';' asks a question and '·' separates clauses, so these two
        # rates swap sources rather than counting the same glyph as both.
        (text.count(GREEK_SEMICOLON) if greek else text.count(";")) / n_chars * 1000,
        (sum(text.count(q) for q in GREEK_QUESTION_MARKS) if greek else text.count("?"))
        / n_chars
        * 1000,
        text.count("!") / n_chars * 1000,
        (text.count("—") + text.count(" - ")) / n_chars * 1000,
        stopword_rate,
        entropy,
        repeated_bg,
    ]


# --------------------------------------------------------------------------
# Interpretable diagnostics
#
# The trained classifier turns these features into a probability, but that
# needs labelled data. The raw numbers are useful on their own and cost
# nothing: no model, no GPU, no training, and they work on Greek as well as
# English. Commercial detectors measure burstiness alongside perplexity, so
# these are the other half of the signal — reported as observations to act on
# rather than as a verdict.
# --------------------------------------------------------------------------

# Below this, sentence lengths are so even that the writing reads as machine
# generated to anything measuring burstiness. Human prose usually sits above 0.
UNIFORM_BURSTINESS = -0.10
REPETITIVE_BIGRAM_RATE = 0.15
LOW_DIVERSITY_TTR = 0.35


@dataclass
class StyleNote:
    burstiness: float
    mean_sentence_len: float
    std_sentence_len: float
    type_token_ratio: float
    repeated_bigram_rate: float

    @property
    def observations(self) -> list[str]:
        """Plain statements about the writing, worst first. Empty is good news."""
        notes = []
        if self.burstiness < UNIFORM_BURSTINESS:
            notes.append(
                f"sentence lengths are very even (burstiness {self.burstiness:+.2f}); "
                "varying them reads as more human"
            )
        if self.repeated_bigram_rate > REPETITIVE_BIGRAM_RATE:
            notes.append(
                f"{self.repeated_bigram_rate:.0%} of word pairs repeat; "
                "phrasing is going round in circles"
            )
        if self.type_token_ratio < LOW_DIVERSITY_TTR:
            notes.append(
                f"vocabulary is narrow (type-token ratio {self.type_token_ratio:.2f})"
            )
        return notes

    def __str__(self) -> str:
        return (
            f"burstiness {self.burstiness:+.2f}  "
            f"sentences {self.mean_sentence_len:.0f}+-{self.std_sentence_len:.0f} words  "
            f"diversity {self.type_token_ratio:.2f}"
        )


def describe_style(text: str) -> StyleNote:
    """Stylometric indicators for a passage, without any trained model."""
    values = dict(zip(FEATURE_NAMES, extract_features(text)))
    return StyleNote(
        burstiness=values["burstiness"],
        mean_sentence_len=values["mean_sentence_len"],
        std_sentence_len=values["std_sentence_len"],
        type_token_ratio=values["type_token_ratio"],
        repeated_bigram_rate=values["repeated_bigram_rate"],
    )
