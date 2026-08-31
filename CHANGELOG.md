# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.5.0] - 2026-09-01

### Added

- **`aidetect-benchmark`** — evaluates detectors against labelled corpora and
  prints a table. Conditions are scored and reported **separately, never
  pooled**: a single headline AUROC hides exactly the cases that break a
  detector, which are paraphrased output, non-native writers and formal
  registers. The output names the condition with the most false positives and
  the weakest separation, and warns below 30 documents per class where an FPR
  estimate is too noisy to publish.
- **`--stylometry`** — writing indicators with no model, no download, no GPU and
  no verdict about authorship. This is what makes the 68 MB lite build useful on
  its own; previously the indicators were only produced as a side effect of
  scoring, which needed a detector.
- **`examples/build_example_corpus.py`** — assembles a corpus anyone can
  regenerate: 40 arXiv abstracts from 2010 against 40 generated from the same
  titles, lengths matched. It is an **example corpus, not a validation corpus**,
  and BENCHMARK.md says so at length. Its numbers illustrate the output format
  and are not evidence about detection quality.
- **BENCHMARK.md** — states plainly that no benchmark exists, then records what
  has actually been measured, including two of five human-written samples being
  wrongly flagged, and what a real evaluation would need.

### Changed

- The README leads with Detect, Explain, Compare, Calibrate rather than with
  detection alone, and states that documents never leave the machine.
- Console output is checked for non-ASCII by a test that parses the source
  rather than grepping it, after an em-dash rendered as a replacement character
  in the message a lite-build user is most likely to see.

### Fixed

- `pip` and `setuptools` upgraded past PYSEC-2026-3721 and PYSEC-2026-3447.
  `setuptools` is bundled into the executable, so the vulnerable version was
  being shipped to users rather than merely sitting in a build environment.

### Notes

- Exercising the benchmark harness caught a trap worth recording. The first
  corpus had an "AI" side written by hand to look formulaic, and the harness
  returned AUROC 0.000. The harness was right: that text was human-written, the
  label was false, and a detector ranking it as human was correct to. Text that
  imitates machine writing is not machine writing.

## [1.4.0] - 2026-08-31

### Fixed

- **The same document scored differently depending on the file it arrived in.**
  Whitespace is made of tokens, so it counts towards perplexity. python-docx
  emits a newline at every paragraph, pypdf at every *visual* line wrap — mid
  sentence, purely a layout artefact — and a paste carries whatever the
  clipboard held. Measured on one five-paragraph document, those spellings
  scored 3.82% apart, wider than the margin many documents sit from the
  threshold. One document checked as .docx and as .pdf could get opposite
  verdicts on formatting alone.

  Scoring now runs on a canonical flat form, applied inside `score_text()` so
  every path through the tool is covered: file, paste, stdin, and each section
  of a long document. The four spellings above now score identically to eight
  decimal places. Non-breaking spaces, zero-width characters and CRLF are
  cleaned up too. The window still displays the readable form with paragraphs
  intact; only the scored string is flattened.

### Notes

- **Scores shift slightly compared with 1.3.0**, because the scored string has
  changed. Any threshold calibrated before this release should be recalculated
  with `aidetect-calibrate`.
- The 3.4% gap against the reference implementation's published sample is
  explained and needs no code change. Measured sensitivity: dtype (bfloat16 vs
  float32) moves the score 0.23%, whitespace moves it up to 9.87%. The gap sits
  well inside the whitespace range, and the sample string here was
  reconstructed from wrapped documentation rather than copied byte for byte.
  The formula itself is confirmed correct.

## [1.3.0] - 2026-08-31

Turns the tool from a verdict into a diagnostic. The verdict depends on a
threshold whose fit is unproven and on guessing which detector someone else
runs; what a document scores against your own writing, and which passages read
as formulaic, do not.

### Fixed

- **Greek sentences were split with English rules.** ';' ends a question in
  Greek and joins clauses in English, so every Greek question merged with the
  sentence after it. Measured sentence lengths came out systematically long and
  burstiness — a ratio of their spread to their mean — was wrong on every Greek
  document. Two features were also reading the wrong glyph: ';' counted towards
  the semicolon rate rather than the question rate, and the ano teleia '·' was
  ignored. Script is now detected per section, so a Greek document quoting
  English technical terms is still read as Greek.

### Added

- **Stylometric diagnostics** on every section, needing no model, no GPU and no
  training data: burstiness, sentence-length spread, vocabulary diversity and
  repeated phrasing, each reported as an observation rather than a verdict.
  These are the signals commercial detectors measure alongside perplexity, they
  work in Greek and English alike, and they are the only part that runs in the
  67MB lite build. They also say what to change, which a score does not.
- **Comparison against your own documents.** `aidetect-calibrate FOLDER` scores
  a folder of writing you know is human and stores both a threshold and the
  whole score distribution, so later documents report as "lower than 92% of
  your reference documents". This is the more defensible number: measured on
  your own writing, it cancels the bias perplexity detectors show against
  non-native English writers rather than letting it count against you. Stored
  in ~/.aidetect/thresholds.json and applied automatically.
- **A catalogue of model pairs** (`--pair`, dropdown in the GUI), each stating
  whether the published thresholds apply to it. Only the Falcon pair is
  calibrated; the rest are for fast local testing and say so.
- A guard against pairs that do not share a tokenizer. Only the observer's
  tokenizer is loaded and the same token ids go to both models, so a mismatched
  pair scored one model on the other's ids and returned a meaningless number
  with no error.

### Changed

- The GUI shows a document picker, a button and a plain-language answer.
  Detector, model pair, strictness and threshold moved behind "Advanced
  settings": a control nobody touches makes the tool look harder than it is.
- Results are split into a "Document text" tab and a "Sections" tab, and the
  headline reads "Reads as human-written" or "3 of 17 sections would be
  flagged" rather than a bare number.

### Notes

- An earlier claim in this project's history — that a reader would run Falcon
  with default settings — was unfounded. Binoculars is a research method;
  commercial detectors use proprietary classifiers and publish neither
  methodology nor thresholds. This tool simulates one detector. It does not
  predict what another one will say, and the per-section diagnostics are the
  part that survives that uncertainty.

## [1.2.0] - 2026-08-31

### Fixed

- **The Binoculars score was computed with the model pair reversed.** Both
  terms are asymmetric, and both had observer and performer the wrong way
  round relative to the reference implementation:
  `ppl` was taken under the observer rather than the performer, and the
  cross-entropy used the performer as target rather than the observer. Since
  `DEFAULT_THRESHOLD` is the reference's published constant, a different
  formula judged by that threshold measured against the wrong ruler. The error
  had a direction: scores came out systematically low, and low means
  AI-generated, so human documents were being flagged. A news article that
  scored 0.7285 ("AI-generated") now scores 0.9803 ("human").
- Binoculars reloaded the full model pair on every call, so each analysis paid
  the whole startup cost again. One pair is now cached and reused: repeated
  scoring went from 6.58s to 0.02s. Switching model pairs evicts the old one
  before loading the new, so two never occupy memory at once.

### Added

- **Per-section scoring** (`--sections`, on by default in the GUI). A single
  call truncates at 512 tokens — roughly 380 words — so a 10-page document was
  judged on its first page and the rest silently ignored. Documents are now
  scored in overlapping sections, reporting which passages would be flagged.
  Scoring a long document without it prints a truncation warning.
- **Operating modes** (`--mode accuracy|low-fpr`). Both thresholds come from
  the reference implementation: accuracy (0.9015310749276843) balances the two
  error types, low-fpr (0.8536432310785527) is lower, so fewer documents are
  called AI-generated at the cost of missing more real ones. Constants are now
  carried at full published precision, and pinned by tests.

### Notes

- Against the reference's own published sample this implementation returns
  0.73061407 where the reference reports 0.75661373 — the verdict agrees, the
  score is 3.4% out. The cause is not established. A plausible candidate was
  ruled out by measurement: the reference averages its cross-entropy over the
  unshifted sequence while its perplexity term shifts by one, but computing it
  both ways changes the result by 0.0003. Remaining candidates are the exact
  sample string and bfloat16 accumulation differing between CPU and GPU. This
  matters for documents scoring near a threshold.

## [1.1.0] - 2026-08-31

### Added

- **Markdown export** — `readers.to_markdown()` and `save_markdown()` render a
  document to Markdown, keeping headings, lists and tables. `--save-md [PATH]`
  on the CLI writes it beside the source by default; the GUI has a "Save .md"
  button that activates once a document is loaded. This is an artefact for
  reading and keeping, deliberately separate from what gets scored.

### Fixed

- Word field results leaked into the scored text: table-of-contents entries,
  cross-references, and placeholders like "No table of figures entries found."
  or "Error! Reference source not found." — boilerplate Word generates that the
  author never wrote. They are now detected structurally (`w:fldChar`,
  `w:instrText`, `w:fldSimple`) rather than by matching their text, because
  Word localises the wording and the English strings would miss a Greek
  document. Paragraphs styled as TOC or table-of-figures furniture are dropped
  too. A field embedded mid-sentence is left alone.

### Notes

- Detection input stays plain prose rather than Markdown. Markup would be
  scored as part of the text: Binoculars measures its perplexity, and
  `features.dash_rate` counts " - ", which is exactly a Markdown bullet, so
  converting first would corrupt the stylometric features on every document.

## [1.0.0] - 2026-08-31

First packaged release. The toolkit becomes a standalone Windows application:
a document goes in through a file browser, a verdict comes out, and no Python
installation is required on the target machine.

### Added

- **Standalone executable.** [aidetect.spec](aidetect.spec) builds a frozen
  Windows binary via PyInstaller, driven by
  [packaging/build.ps1](packaging/build.ps1). Two flavours:
  - *full* (default) — one folder, ~4.4 GB against a CUDA torch, bundling
    torch and transformers so Binoculars runs zero-shot with no training data
    and no model file;
  - *lite* (`-Lite`) — one file, ~67 MB, CPU stylometric detector only, and so
    unusable until given a trained `--model`.
- **Graphical interface** ([aidetect/gui.py](aidetect/gui.py)) — a Tkinter
  window with a file browser, detector choice, optional threshold, an editable
  text pane that doubles as a paste target, and the verdict inline. Opens with
  `aidetect --gui`, or by double-clicking the frozen executable. Scoring runs on
  a worker thread so the window stays responsive during long Binoculars runs.
  A detector the current build cannot run is greyed out and labelled, rather
  than failing with a dialog after the user picks it.
- **PDF support** — `readers.read_pdf()` extracts the text layer via `pypdf`.
  The packaged app now accepts `.txt`, `.md`, `.docx` and `.pdf`. PDFs without
  a text layer raise an error that names OCR as the cause rather than returning
  an empty string.
- **`aidetect.scoring`** — a single `score_text()` entry point returning a
  `Verdict`, shared by the CLI and the GUI so the two frontends cannot drift on
  the detail that is easiest to get wrong: Binoculars scores AI text *low*,
  while `FeatureDetector` returns P(AI) and scores it *high*.
- **`--detector {binoculars,features}`** on the CLI, plus `--model` for the
  trained classifier pickle. Without this the CPU detector was unreachable from
  the command line, and a torch-free build would have had nothing it could run.
- **Runtime hardware selection** — `binoculars.pick_device()` and `pick_dtype()`
  let one build serve GPU and non-GPU machines. A CUDA-built torch imports fine
  where there is no NVIDIA hardware, so the packaged app selects `cuda` or `cpu`
  at run time and picks a dtype to match. `scoring.describe_device()` reports the
  choice; the GUI shows it and the CLI prints `device=` beside the verdict.
- `scoring.detector_available()` reports whether a detector's dependencies are
  present, without importing them — which is what lets the GUI grey one out.
- `python -m aidetect` now works ([aidetect/\_\_main\_\_.py](aidetect/__main__.py)).
- 25 tests covering the readers, all four formats, the scoring conventions, CLI
  exit codes and device selection (40 total, up from 15).

### Changed

- `main()` returns an exit code instead of `None`: `0` success, `1` unreadable
  input, `2` detector unavailable.
- Unreadable files report a one-line message on stderr rather than a traceback.
- CLI help identifies itself as `aidetect` rather than `__main__.py`.
- Console output is ASCII-only — the em-dash in the short-text warning rendered
  as `?` under the frozen executable's console code page.

### Fixed

- `--threshold 0` was silently replaced by the default threshold, because the
  code used `args.threshold or DEFAULT_THRESHOLD` and `0` is falsy.
- `Binoculars` passed `torch_dtype=` to `from_pretrained()`, deprecated since
  transformers 4.56 and warned about on every run. Now passes `dtype=`; the
  `gpu` extra requires `transformers>=4.56` accordingly.
- `Binoculars` hardcoded `bfloat16`. On a CPU without AVX512-BF16 that is
  emulated and slower than plain float32, so every CPU run paid for a
  GPU-tuned default. The dtype now follows the device.
- Double-clicking the frozen executable did nothing: it exited silently with
  status 0. The frozen-app check used `sys.stdin.isatty()`, but a
  double-clicked executable does not reliably get a tty, so it fell through to
  reading stdin, hit EOF and scored the empty string. A frozen app with no
  arguments now always opens the GUI; pipe into it with an explicit `-`.
- Empty stdin is reported as an error rather than scored as a document.

### Known limitations

- The full build bundles the Python runtime, **not** the model weights.
  Binoculars still downloads the Falcon pair (~28GB) from Hugging Face on first
  run, so that flavour is not usable offline and the first run is slow.
- The build picks up whichever torch is installed, and bakes that choice in.
  Built against `torch==2.11.0+cu128`, whose compiled kernels are
  `sm_75 sm_80 sm_86 sm_90 sm_100 sm_120` — covering Turing through Blackwell,
  so RTX 3090 (`sm_86`) and RTX 5090 (`sm_120`) both run natively with no PTX
  JIT. Build against a CPU-only torch and the result will never use a GPU,
  however capable the target machine.
- CUDA 12.8 needs a recent NVIDIA driver (roughly 570+ for Blackwell). Too old
  a driver means `cuda.is_available()` is False and the app quietly runs on
  CPU rather than failing loudly; the reported device is how you tell.
- Binoculars on CPU needs roughly 28GB of RAM for the Falcon pair and is slow
  enough to be impractical. It falls back correctly, but a GPU is what makes it
  usable.
- The GPU path is verified only as far as the compiled kernel list and the unit
  tested device selection. It has not been executed on NVIDIA hardware — the
  build machine has none.
- The lite build ships no trained classifier, so it cannot score anything until
  supplied with a `--model` from `FeatureDetector.save()`. Use the full build
  for zero-shot detection with no training data.
- Scanned/image-only PDFs are rejected — there is no OCR step.
- The executable is unsigned; SmartScreen will warn on first run.

## [0.1.0] - 2025-07-08

Initial toolkit: Binoculars zero-shot detector, stylometric `FeatureDetector`,
score ensemble, evaluation harness, and `.txt`/`.md`/`.docx` readers.
