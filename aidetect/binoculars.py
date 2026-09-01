"""Zero-shot AI text detection via the Binoculars method.

    Hans, Schwarzschild, Cherepanova, Kazemi, Saha, Goldblum, Geiping and
    Goldstein. "Spotting LLMs With Binoculars: Zero-Shot Detection of
    Machine-Generated Text." ICML 2024, PMLR 235. arXiv:2401.12070
    Reference implementation: github.com/ahans30/Binoculars (BSD 3-Clause)

This is an independent implementation of the published method, checked against
the reference's own formulation. The thresholds below are that implementation's
published constants, reproduced unchanged so results stay comparable with it.


Score = perplexity(performer) / cross-perplexity(observer, performer), matching
the reference implementation, because DEFAULT_THRESHOLD below is that
implementation's published constant and the two only mean anything together.
AI-generated text scores LOW (it is unsurprising relative to what the observer
would have predicted). Human text scores HIGH.

Requires: torch, transformers. BOTH models are resident at once, so a 7B pair
is roughly 28GB in bf16 -- more than one 24GB card holds. Two GPUs get one model
each automatically; on a single card use a smaller pair, or device="auto" to
shard both across whatever GPUs there are.
Recommended pair: tiiuae/falcon-7b + tiiuae/falcon-7b-instruct
(the pair used in the paper). Any base/instruct sibling pair works, e.g.
Qwen2.5-7B + Qwen2.5-7B-Instruct.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# The reference implementation publishes two operating points for the Falcon-7B
# pair, and exposes them as its "accuracy" and "low-fpr" modes. Both are quoted
# here at full precision: when the point of running this is to predict what
# someone else's detector will say, matching their constants exactly is the
# whole job. Do not tune these — re-tuning is for judging your own documents,
# which is a different question. evaluate.pick_threshold() is for that.
ACCURACY_THRESHOLD = 0.9015310749276843  # balances false positives and negatives
FPR_THRESHOLD = 0.8536432310785527  # fewer false accusations, more misses

THRESHOLDS = {"accuracy": ACCURACY_THRESHOLD, "low-fpr": FPR_THRESHOLD}
MODES = tuple(THRESHOLDS)

DEFAULT_THRESHOLD = ACCURACY_THRESHOLD


@dataclass
class BinocularsResult:
    score: float
    is_ai: bool
    threshold: float
    perplexity: float
    cross_perplexity: float


def pick_device() -> str:
    """'cuda' where a working GPU is present, otherwise 'cpu'.

    A CUDA-built torch imports fine on a machine with no NVIDIA hardware, so a
    single packaged build serves both; it just reports no CUDA and lands here
    on 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def pick_devices() -> tuple[str, str]:
    """Devices for (observer, performer).

    Binoculars needs BOTH models resident at once, so a 7B pair in bf16 is
    roughly 28GB -- more than a single 24GB card holds. Where a second GPU is
    present the two models go one per card, which is what makes the falcon-7b
    pair fit on consumer hardware at all.

    Returning the bare string 'cuda' would put both on device 0 and leave a
    second card idle, which is an out-of-memory error rather than a slow run.
    """
    if not torch.cuda.is_available():
        return "cpu", "cpu"
    if torch.cuda.device_count() >= 2:
        return "cuda:0", "cuda:1"
    return "cuda:0", "cuda:0"


def pick_dtype(device: str) -> torch.dtype:
    """The widest dtype that is actually fast on ``device``.

    bfloat16 on Ampere and later, float16 on older GPUs that lack bf16, and
    float32 on CPU — CPU bfloat16 is emulated and slower than fp32 on hardware
    without AVX512-BF16, which is most of it.
    """
    if device.startswith("cuda"):
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


class MismatchedTokenizers(ValueError):
    """The two models do not share a vocabulary, so the score would be junk."""


class InsufficientVRAM(RuntimeError):
    """Both models could not be held in memory at once."""


def _assert_same_vocabulary(observer_name: str, performer_name: str) -> None:
    if observer_name == performer_name:
        return
    observer = AutoTokenizer.from_pretrained(observer_name).get_vocab()
    performer = AutoTokenizer.from_pretrained(performer_name).get_vocab()
    if observer == performer:
        return
    raise MismatchedTokenizers(
        f"'{observer_name}' and '{performer_name}' do not share a tokenizer "
        f"({len(observer)} vs {len(performer)} tokens). Binoculars feeds one set "
        "of token ids to both models, so a mismatched pair produces a number "
        "that means nothing. Use base/instruct siblings from the same family, "
        "e.g. tiiuae/falcon-7b with tiiuae/falcon-7b-instruct, or "
        "Qwen/Qwen2.5-7B with Qwen/Qwen2.5-7B-Instruct."
    )


class Binoculars:
    def __init__(
        self,
        observer_name: str = "tiiuae/falcon-7b",
        performer_name: str = "tiiuae/falcon-7b-instruct",
        device: str | None = None,
        dtype: torch.dtype | None = None,
        max_length: int = 512,
        threshold: float = DEFAULT_THRESHOLD,
    ) -> None:
        # 'auto' hands placement to accelerate, which shards each model across
        # every visible GPU. That is what a pair too large for one card each
        # needs; it is slower than one-model-per-card because every forward pass
        # then crosses devices, so it is opt-in rather than the default.
        self.sharded = device == "auto"
        if self.sharded:
            self.observer_device = self.performer_device = "auto"
        elif device is None:
            self.observer_device, self.performer_device = pick_devices()
        else:
            # An explicit device is an override: both models go there, which is
            # what a caller pinning to one card is asking for.
            self.observer_device = self.performer_device = device
        # Kept for reporting, which is all the rest of the package uses it for.
        self.device = (
            self.observer_device
            if self.observer_device == self.performer_device
            else f"{self.observer_device}+{self.performer_device}"
        )
        dtype_probe = "cuda" if self.sharded and torch.cuda.is_available() else self.observer_device
        self.dtype = dtype if dtype is not None else pick_dtype(dtype_probe)
        self.max_length = max_length
        self.threshold = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(observer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Only the observer's tokenizer is used, and the same token ids are fed
        # to both models. A pair that does not share a vocabulary therefore
        # scores one model on the other's token ids: not an error, just a
        # meaningless number. Catch it here rather than let it look like a
        # verdict. Base/instruct siblings always share a tokenizer.
        _assert_same_vocabulary(observer_name, performer_name)

        try:
            self.observer = AutoModelForCausalLM.from_pretrained(
                observer_name, dtype=self.dtype, device_map=self.observer_device
            ).eval()
            self.performer = AutoModelForCausalLM.from_pretrained(
                performer_name, dtype=self.dtype, device_map=self.performer_device
            ).eval()
        except torch.cuda.OutOfMemoryError as exc:
            raise InsufficientVRAM(self._vram_message(observer_name)) from exc

    def _vram_message(self, observer_name: str) -> str:
        """Both models must be resident at once; say so, and say what fits."""
        if not torch.cuda.is_available():
            return (
                "Not enough memory to hold both models at once. Binoculars needs "
                "the observer and the performer resident simultaneously, which "
                "for a 7B pair is roughly 28GB."
            )
        n = torch.cuda.device_count()
        cards = ", ".join(
            f"{torch.cuda.get_device_name(i)} "
            f"({torch.cuda.get_device_properties(i).total_memory / 1024**3:.0f}GB)"
            for i in range(n)
        )
        lines = [
            "Not enough GPU memory to hold both models at once.",
            "",
            f"Binoculars scores text with two models -- '{observer_name}' and its "
            "instruct sibling -- and both must be resident simultaneously. A 7B "
            "pair is roughly 28GB in bf16.",
            "",
            f"Visible GPUs ({n}): {cards}",
            "",
            "Options:",
        ]
        if n >= 2:
            lines.append(
                "  - This should not happen with 2+ GPUs, which get one model "
                "each by default. Pass device=None (the default) rather than "
                "pinning both models to one card."
            )
            lines.append(
                "  - device='auto' shards each model across every GPU, for a "
                "pair too large to fit one per card."
            )
        else:
            lines.append(
                "  - Use a smaller pair: --pair qwen2.5-1.5b (about 6GB) or "
                "--pair qwen2.5-0.5b (about 2GB). Note the published threshold "
                "was tuned for falcon-7b, so re-calibrate before trusting it."
            )
            lines.append(
                "  - Add a second GPU; the two models then get one card each "
                "automatically."
            )
        lines.append(
            "  - --detector features needs no GPU at all, and --stylometry needs "
            "no model whatsoever."
        )
        return "\n".join(lines)

    @torch.no_grad()
    def _logits(self, model, input_ids, attention_mask):
        return model(input_ids=input_ids, attention_mask=attention_mask).logits

    @torch.no_grad()
    def score(self, text: str) -> BinocularsResult:
        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        # The two models may sit on different cards, so each gets the token ids
        # on its own device. model.device is the right target for a sharded
        # model too: accelerate puts the first shard there.
        obs_ids = enc["input_ids"].to(self.observer.device)
        obs_attn = enc["attention_mask"].to(self.observer.device)
        perf_ids = enc["input_ids"].to(self.performer.device)
        perf_attn = enc["attention_mask"].to(self.performer.device)

        # Reduced to log-probabilities on the device that produced them, so the
        # full logits tensor never crosses the bus.
        obs_logits = self._logits(self.observer, obs_ids, obs_attn).float()
        obs_lp = F.log_softmax(obs_logits[:, :-1], dim=-1)
        del obs_logits

        perf_logits = self._logits(self.performer, perf_ids, perf_attn).float()
        perf_lp = F.log_softmax(perf_logits[:, :-1], dim=-1)
        del perf_logits

        # Everything below is elementwise over the two distributions, so it has
        # to happen on one device. The performer's is the arbitrary choice; the
        # arithmetic is identical either way.
        obs_lp = obs_lp.to(perf_lp.device)

        # Shift for next-token prediction
        labels = perf_ids[:, 1:]
        mask = perf_attn[:, 1:].float()
        n = mask.sum()

        # Orientation matters, and it is not symmetric. DEFAULT_THRESHOLD is the
        # reference implementation's published constant, so both terms must be
        # computed the way that implementation does, or the threshold means
        # nothing:
        #     ppl   = perplexity of the text under the PERFORMER
        #     x_ppl = cross-entropy with the OBSERVER's distribution as target
        #             and the PERFORMER's as prediction
        # Reversing either term lowers the score across the board, which against
        # a fixed threshold shows up as human text being called AI-generated.

        # log-perplexity of the text under the performer
        token_lp = perf_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        log_ppl = -(token_lp * mask).sum() / n

        # cross-perplexity: the observer's expected surprise under the performer
        xent = -(obs_lp.exp() * perf_lp).sum(dim=-1)
        log_xppl = (xent * mask).sum() / n

        score = (log_ppl / log_xppl).item()
        return BinocularsResult(
            score=score,
            is_ai=score < self.threshold,
            threshold=self.threshold,
            perplexity=float(torch.exp(log_ppl)),
            cross_perplexity=float(torch.exp(log_xppl)),
        )

    def score_batch(self, texts: list[str]) -> list[BinocularsResult]:
        return [self.score(t) for t in texts]
