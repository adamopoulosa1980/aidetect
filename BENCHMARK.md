# Benchmark

**There is no benchmark yet.** This file says what has actually been measured,
what it showed, and what a real evaluation would need. It is deliberately not a
results table dressed up to look like one.

## What has been measured

Five hand-written samples, scored with `tiiuae/falcon-7b` + `-instruct` in
bfloat16 on CPU, against the published accuracy threshold of `0.9015`. Four were
written by a person; one is the reference implementation's own AI-generated
sample.

| Sample | Truth | Score | Verdict | |
|---|---|---|---|---|
| Reference AI sample ("capybara") | AI | 0.7322 | AI-generated | correct |
| News article | human | 0.9803 | human | correct |
| Casual narrative | human | 1.0049 | human | correct |
| Formal, templated prose | human | **0.7258** | **AI-generated** | **wrong** |
| Technical prose | human | **0.8946** | **AI-generated** | **wrong** |

**Two of five human samples were flagged.** Both failures are the same failure:
formal, templated writing is predictable by design, and predictability is what
the method measures. Neither was written by a machine.

This is five samples chosen by hand. It is an illustration of a failure mode,
not an error rate, and nothing here should be quoted as one.

## What else has been measured

Numbers that came out of building the tool, each reproducible from the repository:

| Question | Finding |
|---|---|
| Does the file format change the score? | It did, by **3.82%** — wider than the margin many documents sit from the threshold. Fixed by normalising to a canonical form; the same wording now scores identically as `.docx`, `.pdf` or pasted text. |
| Does numeric precision matter? | **0.23%** between bfloat16 and float32. Far too small to explain a disagreement. |
| Does whitespace matter? | Up to **9.87%**. Forty times the effect of precision. |
| Can a threshold be transferred between model pairs from one anchor text? | **No.** Across five texts the ratio between Falcon and Qwen2.5-0.5B scores ranged 0.965 to 1.573, a 63% spread. Thresholds derived from different anchors got 2/5 to 4/5 verdicts wrong. |
| Does this implementation match the reference? | Verdict agrees on the reference's published sample; the score is **3.4% out** (0.73061407 here, 0.75661373 published). Unexplained. The sample string was reconstructed from wrapped documentation rather than copied byte for byte, and whitespace sensitivity above is large enough to account for it. |

## Running one

`aidetect-benchmark` takes a corpus laid out by condition and prints a table:

```
corpus/
  original-ai/     human/*.docx   ai/*.docx
  paraphrased/     human/*.docx   ai/*.docx
  non-native/      human/*.docx   ai/*.docx
```

```
aidetect-benchmark corpus/ --max-fpr 0.01 --output results.md
```

Each condition is scored and reported separately, with the threshold chosen
under the given false-positive budget. It names the condition with the most
false positives and the weakest separation, because a pooled figure would hide
precisely those.

### One trap, found while testing the harness

The harness was first exercised on a corpus where the "AI" side was prose
written by hand to *look* formulaic. It returned **AUROC 0.000** — perfectly
inverted. The harness was right: the text labelled AI was human-written, so the
label was false, and a detector that ranked it as human was correct to.

Text that imitates machine writing is not machine writing. Generate the AI side
with actual models, or the benchmark measures nothing but the label.

The same run reproduced a separate finding: Qwen2.5-0.5B does not discriminate
usefully. Small pairs are for testing that a pipeline runs, not for verdicts.

## The example corpus is not a validation corpus

`examples/build_example_corpus.py` assembles a small corpus anyone can
regenerate: 40 abstracts submitted to arXiv during 2010 on the human side, and
40 generated from the same titles by Qwen2.5-1.5B-Instruct on the other, with
lengths matched so nothing separates on length alone.

It exists so the harness can be run end to end and its output seen. **Whatever
numbers it produces are an illustration of the output format, not evidence about
detection quality**, and should never be quoted as a result. The reasons are
structural, not fixable by running it for longer:

- the AI side is a 1.5B model, which writes far more predictably than the models
  people actually use, so any separation it shows is optimistic
- one condition, where the conditions that matter are paraphrasing, non-native
  writers, Greek, and several registers
- 40 documents per class, where an FPR estimate needs hundreds
- abstracts of about 170 words, not whole documents

A validation corpus differs on every one of those lines.

## What a real benchmark needs

Nothing below has been run.

**Corpora.** Several hundred documents per class, in the domain being judged.
Human text must be verifiably human: writing from before roughly 2022 is the
cheapest reliable source. AI text should come from several generators, since a
detector tuned on one model's output does not transfer.

**Conditions worth separating**, because the aggregate hides the failures that
matter:

- original AI output
- AI output lightly edited by a person
- human text lightly edited by an AI
- paraphrased AI output — the known weakness of every method here
- **non-native English writers** — the largest documented bias, and the one most
  likely to cause real harm
- Greek, both human and AI
- formal and templated registers, where the two failures above occurred

**Metrics.** AUROC for ranking quality, and **FPR at the operating threshold**
for deployment. The second matters more: the cost of falsely accusing someone is
not symmetric with the cost of missing a machine-written passage. `evaluate()`
and `pick_threshold(max_fpr=0.01)` in this repository produce both.

**Reporting.** Per condition, never pooled. A single headline AUROC would hide
exactly the cases that break it.

## Why it has not been run

It needs labelled documents in the target domain, which the author of a tool
generally cannot supply for someone else's use of it. `aidetect-calibrate`
exists so that anyone with such documents can measure their own false positive
rate without publishing anything.

Until those numbers exist, treat the section-level writing indicators — which
describe how something is written, and claim nothing about who wrote it — as the
part of this tool that is safe to rely on.
