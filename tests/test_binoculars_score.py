"""Pins the Binoculars score orientation.

DEFAULT_THRESHOLD is the reference implementation's published constant, so the
score must be computed the way that implementation computes it. Both terms are
asymmetric, and swapping either lowers scores across the board — which against
a fixed threshold reads as human text being called AI-generated.

Reference (github.com/AHans30/Binoculars):
    ppl   = perplexity(encodings, performer_logits)
    x_ppl = entropy(observer_logits, performer_logits)   # target=observer
    score = ppl / x_ppl
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
binoculars = pytest.importorskip("aidetect.binoculars")

import torch.nn.functional as F  # noqa: E402


def _reference_score(obs_logits, perf_logits, input_ids, attn):
    """The reference formula, written out independently of the implementation."""
    labels = input_ids[:, 1:]
    mask = attn[:, 1:].float()
    n = mask.sum()

    obs_lp = F.log_softmax(obs_logits[:, :-1], dim=-1)
    perf_lp = F.log_softmax(perf_logits[:, :-1], dim=-1)

    # perplexity under the PERFORMER
    token_lp = perf_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    log_ppl = -(token_lp * mask).sum() / n

    # cross-entropy: OBSERVER distribution is the target
    xent = -(obs_lp.exp() * perf_lp).sum(dim=-1)
    log_xppl = (xent * mask).sum() / n

    return (log_ppl / log_xppl).item()


class _StubModel:
    """Returns fixed logits, so the score maths can be checked without weights."""

    def __init__(self, logits):
        self._logits = logits

    def __call__(self, input_ids=None, attention_mask=None):
        class _Out:
            pass

        out = _Out()
        out.logits = self._logits
        return out


@pytest.fixture
def scorer():
    """A Binoculars whose two models return distinct, fixed logits."""
    torch.manual_seed(0)
    batch, seq, vocab = 1, 12, 40
    obs_logits = torch.randn(batch, seq, vocab)
    perf_logits = torch.randn(batch, seq, vocab) * 1.7 + 0.3  # deliberately different

    det = binoculars.Binoculars.__new__(binoculars.Binoculars)
    det.device = "cpu"
    det.dtype = torch.float32
    det.max_length = seq
    det.threshold = binoculars.DEFAULT_THRESHOLD
    det.observer = _StubModel(obs_logits)
    det.performer = _StubModel(perf_logits)

    input_ids = torch.randint(0, vocab, (batch, seq))
    attn = torch.ones(batch, seq, dtype=torch.long)

    class _Enc(dict):
        def to(self, _device):
            return self

    det.tokenizer = lambda *a, **k: _Enc(input_ids=input_ids, attention_mask=attn)
    return det, obs_logits, perf_logits, input_ids, attn


def test_score_matches_the_reference_formula(scorer):
    det, obs_logits, perf_logits, input_ids, attn = scorer
    expected = _reference_score(obs_logits, perf_logits, input_ids, attn)
    assert det.score("ignored").score == pytest.approx(expected, rel=1e-6)


def test_swapping_the_models_changes_the_score(scorer):
    """If this ever passes, the score is symmetric and the orientation is moot."""
    det, obs_logits, perf_logits, input_ids, attn = scorer
    forward = _reference_score(obs_logits, perf_logits, input_ids, attn)
    reversed_ = _reference_score(perf_logits, obs_logits, input_ids, attn)
    assert forward != pytest.approx(reversed_, rel=1e-6)


def test_reversed_orientation_scores_lower(scorer):
    """The old bug's signature: reversed roles depress the score.

    A depressed score sits below a fixed threshold more often, which is human
    text being reported as AI-generated.
    """
    det, obs_logits, perf_logits, input_ids, attn = scorer

    labels = input_ids[:, 1:]
    mask = attn[:, 1:].float()
    n = mask.sum()
    obs_lp = F.log_softmax(obs_logits[:, :-1], dim=-1)
    perf_lp = F.log_softmax(perf_logits[:, :-1], dim=-1)

    # the previous implementation: ppl under observer, target = performer
    old_ppl = -(obs_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1) * mask).sum() / n
    old_xppl = ((-(perf_lp.exp() * obs_lp).sum(dim=-1)) * mask).sum() / n
    old = (old_ppl / old_xppl).item()

    assert det.score("ignored").score != pytest.approx(old, rel=1e-6)


def test_default_threshold_is_the_published_constant():
    """Do not tune this away: other Binoculars implementations use the same value."""
    assert binoculars.DEFAULT_THRESHOLD == pytest.approx(0.9015, abs=1e-4)


# ------------------------------------------------------------------- modes


def test_published_thresholds_are_exact():
    """Both constants come from the reference implementation.

    The point of this tool is predicting what someone else's detector reports,
    so the constants must match theirs rather than be rounded or tuned.
    """
    assert binoculars.ACCURACY_THRESHOLD == 0.9015310749276843
    assert binoculars.FPR_THRESHOLD == 0.8536432310785527


def test_low_fpr_threshold_is_the_lower_one():
    """Lower cut = less text falls below it = fewer AI verdicts."""
    assert binoculars.FPR_THRESHOLD < binoculars.ACCURACY_THRESHOLD


def test_default_is_accuracy_mode():
    assert binoculars.DEFAULT_THRESHOLD == binoculars.ACCURACY_THRESHOLD


def test_modes_map_to_the_published_constants():
    assert binoculars.THRESHOLDS == {
        "accuracy": binoculars.ACCURACY_THRESHOLD,
        "low-fpr": binoculars.FPR_THRESHOLD,
    }


# ---------------------------------------------------------- model pairs


def test_mismatched_tokenizers_are_rejected(monkeypatch):
    """A pair without a shared vocabulary scores one model on the other's ids.

    That produces no error and no warning, just a meaningless number, so it has
    to be refused rather than reported as a verdict.
    """
    vocabs = {"model-a": {"x": 0, "y": 1}, "model-b": {"p": 0, "q": 1, "r": 2}}

    class FakeTok:
        def __init__(self, name):
            self._name = name

        def get_vocab(self):
            return vocabs[self._name]

    monkeypatch.setattr(
        binoculars.AutoTokenizer, "from_pretrained", lambda name, **kw: FakeTok(name)
    )
    with pytest.raises(binoculars.MismatchedTokenizers, match="do not share a tokenizer"):
        binoculars._assert_same_vocabulary("model-a", "model-b")


def test_matching_tokenizers_pass(monkeypatch):
    shared = {"x": 0, "y": 1}

    class FakeTok:
        def get_vocab(self):
            return shared

    monkeypatch.setattr(
        binoculars.AutoTokenizer, "from_pretrained", lambda name, **kw: FakeTok()
    )
    binoculars._assert_same_vocabulary("base", "base-instruct")  # must not raise


def test_identical_names_skip_the_check(monkeypatch):
    """Same model twice needs no tokenizer load."""

    def _boom(*a, **k):
        raise AssertionError("should not load a tokenizer for identical names")

    monkeypatch.setattr(binoculars.AutoTokenizer, "from_pretrained", _boom)
    binoculars._assert_same_vocabulary("same/model", "same/model")
