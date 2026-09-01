"""Tkinter front-end: pick a document, press Analyse, read the answer.

Launched with `aidetect --gui`, or by running the frozen executable with no
arguments (double-clicking it).

The window shows only what a normal run needs: a file, a button, and a verdict
in words. Everything else — detector, model pair, operating mode, threshold —
has a sensible default and lives behind "Advanced settings", because a control
you never touch is a control that makes the tool look harder than it is.

Scoring runs on a worker thread. Binoculars can take minutes on first use while
it downloads model weights, and a frozen window looks like a crash.
"""

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .models import DEFAULT_PAIR, MODEL_PAIRS
from .readers import load_text, save_markdown
from .scoring import (
    DetectorUnavailable,
    describe_device,
    detector_available,
    score_document,
    score_text,
)

_FILETYPES = [
    ("All supported", "*.txt *.md *.docx *.pdf"),
    ("Word document", "*.docx"),
    ("PDF", "*.pdf"),
    ("Text", "*.txt *.md"),
    ("All files", "*.*"),
]

MIN_WORDS = 50

_READY = "Choose a document, or paste text below."

_LITE_NOTICE = (
    "Lite build: the stylometric indicators work, but Binoculars is not "
    "included. For the zero-shot detector, use the full build."
)

# (tag, text). "h" renders as a heading, "warn" in red, "" as body text.
HELP = [
    ("h", "What this does"),
    (
        "",
        "It scores a piece of writing - an essay, a report, an article - the way "
        "one family of AI-text detectors does, and "
        "reports which passages read as machine-written. Load a .txt, .md, .docx "
        "or .pdf, or paste text, and press Analyse.",
    ),
    ("h", "Reading the result"),
    (
        "",
        "The headline says whether the document reads as human-written, or how "
        "many of its sections would be flagged. The Sections tab breaks that "
        "down passage by passage, so you can see where the risk sits instead of "
        "rewriting the whole thing.",
    ),
    (
        "",
        "Each section also carries plain observations about the writing: "
        "burstiness (how much sentence lengths vary), vocabulary range, and "
        "repeated phrasing. These need no model and work in Greek and English "
        "alike. They tell you what to change, which a score does not.",
    ),
    ("h", "What the numbers mean"),
    (
        "",
        "The score is a ratio of two perplexities. AI text is unsurprising to a "
        "language model, so it scores LOW; human text scores HIGH. Below the "
        "threshold means flagged.",
    ),
    ("h", "Where it is unreliable"),
    (
        "warn",
        "Formal, templated writing scores low even when a person wrote every "
        "word. Academic essays, reports and technical writing are formulaic by "
        "design, "
        "and formulaic is exactly what this measures. Expect them to score near "
        "or below the threshold.",
    ),
    (
        "warn",
        "Writers whose first language is not English are flagged far more often "
        "by every perplexity-based detector, because simpler vocabulary and "
        "conventional grammar look predictable. Liang, Yuksekgonul, Mao, Wu and "
        "Zou (Patterns, 2023, arXiv:2304.02819) found detectors misclassifying "
        "over half of TOEFL essays by non-native writers as AI-generated, while "
        "correctly clearing essays by native writers. If that describes the "
        "people whose writing you are checking, calibrate before believing any "
        "verdict here.",
    ),
    (
        "warn",
        "This simulates one detector. It cannot tell you what Turnitin, GPTZero "
        "or any commercial tool will report; those use proprietary classifiers "
        "and publish neither method nor thresholds.",
    ),
    (
        "",
        "Short texts are unreliable below about 50 words, and scanned PDFs with "
        "no text layer cannot be read at all.",
    ),
    ("h", "Making it fit your own writing"),
    (
        "",
        "The default threshold was calibrated on someone else's benchmark. To "
        "get a number you can defend, run this once against a folder of "
        "essays or other writing you know people wrote:",
    ),
    ("", "    aidetect-calibrate C:\\path\\to\\reference-documents"),
    (
        "",
        "Essays or reports written before about 2022 are a safe source: "
        "pre-ChatGPT, "
        "your domain, your house style. Afterwards every result also reports "
        "where the document sits among them, for example \"lower than 92% of "
        "your reference documents\". That comparison is measured on your own "
        "writing, so the bias above cancels out instead of counting against "
        "you. It is the most defensible number this tool produces.",
    ),
    ("h", "Advanced settings"),
    (
        "",
        "Defaults suit the usual case. Inside you can change the detector, pick "
        "a smaller model pair for faster testing, loosen the strictness, or set "
        "a threshold by hand. Only the Falcon pair matches the published "
        "thresholds; the others need calibrating before their verdicts mean "
        "anything.",
    ),
    ("h", "Credits"),
    (
        "",
        "The zero-shot detector implements the Binoculars method of Hans, "
        "Schwarzschild, Cherepanova, Kazemi, Saha, Goldblum, Geiping and "
        "Goldstein, \"Spotting LLMs With Binoculars: Zero-Shot Detection of "
        "Machine-Generated Text\", ICML 2024 (arXiv:2401.12070). The reference "
        "implementation is at github.com/ahans30/Binoculars, BSD 3-Clause, and "
        "both thresholds here are its published constants.",
    ),
    (
        "",
        "This is an independent implementation of the published method, so any "
        "error in it is ours rather than the authors'. The stylometric "
        "indicators, the document readers, the sectioning and the calibration "
        "are this project's own and are not part of Binoculars.",
    ),
    ("h", "A word of caution"),
    (
        "warn",
        "No detector is proof of anything. Treat a flagged passage as a note "
        "that it reads as formulaic, which is useful editorial feedback, and "
        "not as evidence about how it was written.",
    ),
]


class App(ttk.Frame):
    def __init__(self, master: tk.Tk) -> None:
        super().__init__(master, padding=12)
        self.grid(row=0, column=0, sticky="nsew")
        master.columnconfigure(0, weight=1)
        master.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)

        self._results: queue.Queue = queue.Queue()
        self._busy = False
        # Kept so the analysis can be exported after the fact.
        self._verdict = None

        self.path_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value=_READY if detector_available("binoculars") else _LITE_NOTICE
        )
        self.hardware_var = tk.StringVar(value="Checking hardware...")
        self.advanced_var = tk.BooleanVar(value=False)

        # Advanced settings, each with a default that suits the common case.
        self.detector_var = tk.StringVar(value="binoculars")
        self.model_var = tk.StringVar()
        self.threshold_var = tk.StringVar()
        self.sections_var = tk.BooleanVar(value=True)
        self.mode_var = tk.StringVar(value="accuracy")
        self.pair_var = tk.StringVar(value=DEFAULT_PAIR.display)

        self._build()
        self._sync_detector()
        self.after(100, self._drain)
        # Importing torch to find out whether there is a GPU takes seconds, so
        # it must not happen before the window is on screen.
        threading.Thread(target=self._probe_hardware, daemon=True).start()

    # ---------------------------------------------------------------- layout

    def _build(self) -> None:
        row = 0
        picker = ttk.Frame(self)
        picker.grid(row=row, column=0, sticky="ew")
        picker.columnconfigure(1, weight=1)
        ttk.Label(picker, text="Document").grid(row=0, column=0, sticky="w")
        ttk.Entry(picker, textvariable=self.path_var).grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(picker, text="Browse...", command=self.browse).grid(row=0, column=2)

        row += 1
        actions = ttk.Frame(self)
        actions.grid(row=row, column=0, sticky="w", pady=(10, 6))
        self.run_button = ttk.Button(actions, text="Analyse", command=self.run)
        self.run_button.grid(row=0, column=0)
        self.save_md_button = ttk.Button(actions, text="Save as .md", command=self.save_markdown)
        self.save_md_button.grid(row=0, column=1, padx=(8, 0))
        self.save_md_button.state(["disabled"])
        # Exports the analysis, not the document -- 'Save as .md' above converts
        # the source file, which is a different thing people confuse them for.
        self.export_button = ttk.Button(
            actions, text="Export analysis", command=self.export_analysis
        )
        self.export_button.grid(row=0, column=2, padx=(8, 0))
        self.export_button.state(["disabled"])
        ttk.Button(actions, text="Clear", command=self.clear).grid(row=0, column=3, padx=(8, 0))
        ttk.Button(actions, text="Help", command=self.show_help).grid(row=0, column=4, padx=(8, 0))

        row += 1
        self.headline = ttk.Label(self, text="", font=("", 11, "bold"), wraplength=640)
        self.headline.grid(row=row, column=0, sticky="w", pady=(2, 0))

        row += 1
        self.result = ttk.Label(self, textvariable=self.status_var, wraplength=640, justify="left")
        self.result.grid(row=row, column=0, sticky="w", pady=(2, 6))

        row += 1
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=row, column=0, sticky="nsew")
        self.rowconfigure(row, weight=1)

        text_tab = ttk.Frame(self.notebook, padding=6)
        text_tab.columnconfigure(0, weight=1)
        text_tab.rowconfigure(0, weight=1)
        self.text = tk.Text(text_tab, height=12, wrap="word", undo=True)
        self.text.grid(row=0, column=0, sticky="nsew")
        text_scroll = ttk.Scrollbar(text_tab, orient="vertical", command=self.text.yview)
        text_scroll.grid(row=0, column=1, sticky="ns")
        self.text.configure(yscrollcommand=text_scroll.set)
        self.notebook.add(text_tab, text="Document text")

        report_tab = ttk.Frame(self.notebook, padding=6)
        report_tab.columnconfigure(0, weight=1)
        report_tab.rowconfigure(0, weight=1)
        self.report = tk.Text(report_tab, height=12, wrap="none", state="disabled")
        self.report.grid(row=0, column=0, sticky="nsew")
        report_scroll = ttk.Scrollbar(report_tab, orient="vertical", command=self.report.yview)
        report_scroll.grid(row=0, column=1, sticky="ns")
        self.report.configure(yscrollcommand=report_scroll.set)
        self.report.tag_configure("flag", foreground="#b00020")
        self.notebook.add(report_tab, text="Sections")

        row += 1
        ttk.Checkbutton(
            self,
            text="Advanced settings",
            variable=self.advanced_var,
            command=self._toggle_advanced,
        ).grid(row=row, column=0, sticky="w", pady=(8, 0))

        row += 1
        self.advanced = ttk.Frame(self, padding=(12, 6, 0, 0))
        self.advanced.grid(row=row, column=0, sticky="ew")
        self.advanced.grid_remove()  # hidden until asked for
        self._build_advanced(self.advanced)

        row += 1
        ttk.Label(self, textvariable=self.hardware_var, foreground="gray40").grid(
            row=row, column=0, sticky="w", pady=(8, 0)
        )

    def _build_advanced(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(1, weight=1)
        row = 0

        ttk.Label(parent, text="Detector").grid(row=row, column=0, sticky="w", pady=3)
        box = ttk.Frame(parent)
        box.grid(row=row, column=1, sticky="w")
        choices = [
            ("binoculars", "Binoculars (no training needed)"),
            ("features", "Trained classifier (needs a model file)"),
            ("stylometry", "Writing indicators only (no model, works anywhere)"),
        ]
        for i, (value, text) in enumerate(choices):
            usable = value == "stylometry" or detector_available(value)
            button = ttk.Radiobutton(
                box,
                text=text + ("" if usable else " - not in this build"),
                value=value,
                variable=self.detector_var,
                command=self._sync_detector,
            )
            if not usable:
                button.state(["disabled"])
            button.grid(row=0, column=i, padx=(0, 12))
        if not (self.detector_var.get() == "stylometry" or detector_available(self.detector_var.get())):
            self.detector_var.set("stylometry")

        row += 1
        self.model_label = ttk.Label(parent, text="Trained model")
        self.model_label.grid(row=row, column=0, sticky="w", pady=3)
        model_row = ttk.Frame(parent)
        model_row.grid(row=row, column=1, sticky="ew")
        model_row.columnconfigure(0, weight=1)
        self.model_entry = ttk.Entry(model_row, textvariable=self.model_var)
        self.model_entry.grid(row=0, column=0, sticky="ew")
        self.model_button = ttk.Button(model_row, text="Browse...", command=self.browse_model)
        self.model_button.grid(row=0, column=1, padx=(6, 0))

        row += 1
        ttk.Label(parent, text="Models").grid(row=row, column=0, sticky="w", pady=3)
        self.pair_box = ttk.Combobox(
            parent,
            textvariable=self.pair_var,
            values=[p.display for p in MODEL_PAIRS],
            state="readonly",
        )
        self.pair_box.grid(row=row, column=1, sticky="ew")
        self.pair_box.bind("<<ComboboxSelected>>", lambda _e: self._sync_pair())

        row += 1
        ttk.Label(parent, text="Strictness").grid(row=row, column=0, sticky="w", pady=3)
        modes = ttk.Frame(parent)
        modes.grid(row=row, column=1, sticky="w")
        for i, (value, label) in enumerate(
            [
                ("accuracy", "Standard - the reference default"),
                ("low-fpr", "Lenient - flags less, misses more"),
            ]
        ):
            ttk.Radiobutton(modes, text=label, value=value, variable=self.mode_var).grid(
                row=0, column=i, padx=(0, 12)
            )

        row += 1
        ttk.Label(parent, text="Threshold").grid(row=row, column=0, sticky="w", pady=3)
        extras = ttk.Frame(parent)
        extras.grid(row=row, column=1, sticky="w")
        ttk.Entry(extras, textvariable=self.threshold_var, width=12).grid(row=0, column=0)
        ttk.Label(extras, text="(blank = use the strictness above)", foreground="gray40").grid(
            row=0, column=1, padx=(6, 0)
        )

        row += 1
        ttk.Checkbutton(
            parent,
            text="Score the whole document, not just the first ~380 words",
            variable=self.sections_var,
        ).grid(row=row, column=1, sticky="w", pady=3)

    def _toggle_advanced(self) -> None:
        if self.advanced_var.get():
            self.advanced.grid()
        else:
            self.advanced.grid_remove()

    # --------------------------------------------------------------- helpers

    def _selected_pair(self):
        wanted = self.pair_var.get()
        for pair in MODEL_PAIRS:
            if pair.display == wanted:
                return pair
        return DEFAULT_PAIR

    def _sync_pair(self) -> None:
        """Say straight away when a pair's verdicts are not comparable."""
        pair = self._selected_pair()
        if pair.calibrated:
            self.status_var.set(f"{pair.label}: published thresholds apply.")
        else:
            self.status_var.set(
                f"{pair.label}: the published thresholds were derived for Falcon 7B, "
                "so verdicts from this pair are not comparable. Calibrate on your "
                "own documents first (aidetect-calibrate)."
            )

    def _sync_detector(self) -> None:
        features = self.detector_var.get() == "features"
        state = "normal" if features else "disabled"
        for widget in (self.model_label, self.model_entry, self.model_button):
            widget.configure(state=state)
        if features and not self.model_var.get():
            self.status_var.set(
                "The stylometric detector needs a trained model file. Binoculars "
                "needs nothing and is the usual choice."
            )

    def _set_headline(self, text: str, flagged: bool | None = None) -> None:
        colour = "" if flagged is None else ("#b00020" if flagged else "#1a7f37")
        self.headline.configure(text=text, foreground=colour)

    def _show_report(self, text: str) -> None:
        self.report.configure(state="normal")
        self.report.delete("1.0", "end")
        for line in text.splitlines():
            self.report.insert("end", line + "\n", ("flag",) if "FLAG" in line else ())
        self.report.configure(state="disabled")

    # --------------------------------------------------------------- actions

    def browse(self) -> None:
        path = filedialog.askopenfilename(title="Choose a document", filetypes=_FILETYPES)
        if not path:
            return
        self.path_var.set(path)
        try:
            content = load_text(path)
        except Exception as e:  # unreadable file, scanned PDF, bad extension...
            messagebox.showerror("Could not read file", str(e))
            self.status_var.set("Could not read that file.")
            return
        self.text.delete("1.0", "end")
        self.text.insert("1.0", content)
        self._set_headline("")
        self._show_report("")
        self.status_var.set(f"Loaded {Path(path).name} - {len(content.split()):,} words.")
        self.save_md_button.state(["!disabled"])
        self.notebook.select(0)

    def browse_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose a trained model",
            filetypes=[("Pickled model", "*.pkl"), ("All files", "*.*")],
        )
        if path:
            self.model_var.set(path)

    def save_markdown(self) -> None:
        source = self.path_var.get()
        if not source:
            messagebox.showinfo("Nothing to convert", "Load a document first.")
            return
        target = filedialog.asksaveasfilename(
            title="Save Markdown as",
            defaultextension=".md",
            initialfile=Path(source).with_suffix(".md").name,
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")],
        )
        if not target:
            return
        try:
            written = save_markdown(source, target)
        except Exception as e:
            messagebox.showerror("Could not write Markdown", str(e))
            return
        self.status_var.set(f"Markdown written to {written}")

    def export_analysis(self) -> None:
        """Write the section analysis to a Markdown file."""
        verdict = self._verdict
        if verdict is None or not hasattr(verdict, "to_markdown"):
            messagebox.showinfo(
                "Nothing to export",
                "Analyse a document with 'Score section by section' ticked first.",
            )
            return
        source = self.path_var.get() or None
        initial = (
            f"{Path(source).stem}.analysis.md" if source else "analysis.md"
        )
        target = filedialog.asksaveasfilename(
            title="Export analysis as",
            defaultextension=".md",
            initialfile=initial,
            filetypes=[("Markdown", "*.md"), ("All files", "*.*")],
        )
        if not target:
            return
        try:
            Path(target).write_text(verdict.to_markdown(source), encoding="utf-8")
        except OSError as e:
            messagebox.showerror("Could not write analysis", str(e))
            return
        self.status_var.set(f"Analysis written to {target}")

    def show_help(self) -> None:
        """A scrollable help window.

        States the limits as prominently as the instructions. A detector that
        does not say when it is unreliable invites people to act on numbers
        that do not support the decision.
        """
        window = tk.Toplevel(self)
        window.title("About aidetect")
        window.geometry("680x560")
        window.columnconfigure(0, weight=1)
        window.rowconfigure(0, weight=1)

        text = tk.Text(window, wrap="word", padx=14, pady=12, state="normal")
        text.grid(row=0, column=0, sticky="nsew")
        scroll = ttk.Scrollbar(window, orient="vertical", command=text.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        text.configure(yscrollcommand=scroll.set)

        text.tag_configure("h", font=("", 11, "bold"), spacing1=10, spacing3=4)
        text.tag_configure("warn", foreground="#b00020")
        for tag, body in HELP:
            text.insert("end", body + "\n", tag)
        text.configure(state="disabled")

        ttk.Button(window, text="Close", command=window.destroy).grid(
            row=1, column=0, columnspan=2, pady=8
        )

    def clear(self) -> None:
        self.text.delete("1.0", "end")
        self.path_var.set("")
        self.status_var.set(_READY)
        self._set_headline("")
        self._show_report("")
        self.save_md_button.state(["disabled"])
        self._verdict = None
        self.export_button.state(["disabled"])

    def run(self) -> None:
        if self._busy:
            return
        content = self.text.get("1.0", "end").strip()
        if not content:
            messagebox.showinfo("Nothing to analyse", "Load a document or paste some text first.")
            return

        raw = self.threshold_var.get().strip()
        try:
            threshold = float(raw) if raw else None
        except ValueError:
            messagebox.showerror("Invalid threshold", f"{raw!r} is not a number.")
            return

        warning = ""
        if len(content.split()) < MIN_WORDS:
            warning = f"Under {MIN_WORDS} words - treat this result as unreliable.\n"

        self._busy = True
        self.run_button.configure(state="disabled")
        self._set_headline("")
        self.status_var.set("Analysing... the first run downloads the models, which takes a while.")

        pair = self._selected_pair()
        kwargs = {
            "detector": self.detector_var.get(),
            "model": self.model_var.get() or None,
            "threshold": threshold,
            "mode": self.mode_var.get(),
            "observer": pair.observer,
            "performer": pair.performer,
        }
        threading.Thread(
            target=self._work,
            args=(content, kwargs, warning, self.sections_var.get()),
            daemon=True,
        ).start()

    def _probe_hardware(self) -> None:
        try:
            self._results.put(("hardware", describe_device()))
        except Exception:  # a hardware probe must never take the window down
            pass

    def _work(self, content: str, kwargs: dict, warning: str, sectioned: bool) -> None:
        try:
            if kwargs.get("detector") == "stylometry":
                from .scoring import describe_document

                verdict = describe_document(content)
            else:
                scorer = score_document if sectioned else score_text
                verdict = scorer(content, **kwargs)
            self._results.put(("ok", (verdict.headline, warning + str(verdict), verdict)))
        except DetectorUnavailable as e:
            self._results.put(("error", str(e)))
        except Exception as e:
            self._results.put(("error", f"{type(e).__name__}: {e}"))

    def _drain(self) -> None:
        try:
            kind, payload = self._results.get_nowait()
        except queue.Empty:
            pass
        else:
            if kind == "hardware":
                self.hardware_var.set(payload)
            else:
                self._busy = False
                self.run_button.configure(state="normal")
                if kind == "error":
                    self._set_headline("Could not analyse this document.", flagged=True)
                    self.status_var.set("")
                    messagebox.showerror("Detection failed", payload)
                else:
                    headline, detail, verdict = payload
                    self._verdict = verdict
                    # Only a section-by-section run has something to export;
                    # a single whole-document score is one number.
                    self.export_button.state(
                        ["!disabled"] if hasattr(verdict, "to_markdown") else ["disabled"]
                    )
                    flagged = getattr(verdict, "flagged", None)
                    is_flagged = bool(flagged) if flagged is not None else verdict.is_ai
                    self._set_headline(headline, flagged=is_flagged)
                    first, _, rest = detail.partition("\n")
                    self.status_var.set(first)
                    self._show_report(rest or detail)
                    if rest:
                        self.notebook.select(1)
        self.after(100, self._drain)


def main() -> None:
    root = tk.Tk()
    # Say in the title bar which build this is. The lite executable cannot run
    # Binoculars at all, and someone who downloaded it without noticing should
    # not have to work that out from a greyed-out radio button.
    lite = not detector_available("binoculars")
    root.title("aidetect - lite build (no Binoculars)" if lite else "aidetect")
    root.minsize(720, 560)
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
