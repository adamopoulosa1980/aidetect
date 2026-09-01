"""Tests for the document readers and the shared scoring entry point.

Covers the formats the packaged executable accepts (.txt, .md, .docx, .pdf)
and the direction convention that the two detectors disagree on.
"""

from __future__ import annotations

import pytest

from aidetect.readers import SUPPORTED_EXTENSIONS, load_text
from aidetect.scoring import DETECTORS, DetectorUnavailable, Verdict, score_text


def _minimal_pdf(text: str) -> bytes:
    """Build a tiny single-page PDF with a real text layer."""
    body = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R "
        b"/Resources << /Font << /F1 5 0 R >> >> >>",
        b"<< /Length " + str(len(body)).encode() + b" >>\nstream\n" + body + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, 1):
        offsets.append(len(out))
        out += str(i).encode() + b" 0 obj\n" + obj + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 " + str(len(objs) + 1).encode() + b"\n0000000000 65535 f \n"
    for off in offsets:
        out += ("%010d 00000 n \n" % off).encode()
    out += (
        b"trailer\n<< /Size "
        + str(len(objs) + 1).encode()
        + b" /Root 1 0 R >>\nstartxref\n"
        + str(xref).encode()
        + b"\n%%EOF\n"
    )
    return bytes(out)


def test_all_four_formats_are_supported():
    assert SUPPORTED_EXTENSIONS == {".txt", ".md", ".docx", ".pdf"}


@pytest.mark.parametrize("suffix", [".txt", ".md"])
def test_plain_text_roundtrip(tmp_path, suffix):
    p = tmp_path / f"sample{suffix}"
    p.write_text("hello there", encoding="utf-8")
    assert load_text(p) == "hello there"


def test_docx_roundtrip(tmp_path):
    docx = pytest.importorskip("docx")
    p = tmp_path / "sample.docx"
    doc = docx.Document()
    doc.add_paragraph("First paragraph.")
    doc.add_paragraph("Second paragraph.")
    doc.save(p)
    assert load_text(p) == "First paragraph.\nSecond paragraph."


def test_pdf_roundtrip(tmp_path):
    pytest.importorskip("pypdf")
    p = tmp_path / "sample.pdf"
    p.write_bytes(_minimal_pdf("Hello from a PDF."))
    assert "Hello from a PDF." in load_text(p)


def test_pdf_without_text_layer_explains_itself(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    p = tmp_path / "scanned.pdf"
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(p, "wb") as fh:
        writer.write(fh)
    with pytest.raises(ValueError, match="OCR"):
        load_text(p)


def test_unsupported_extension_lists_the_supported_ones(tmp_path):
    p = tmp_path / "sample.rtf"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_text(p)


def test_legacy_doc_points_at_the_conversion(tmp_path):
    p = tmp_path / "old.doc"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Convert to .docx"):
        load_text(p)


# ------------------------------------------------------------------ scoring


def test_features_detector_needs_a_model():
    with pytest.raises(DetectorUnavailable, match="trained model"):
        score_text("some text", "features")


def test_unknown_detector_is_rejected():
    with pytest.raises(ValueError, match="Unknown detector"):
        score_text("some text", "nonsense")


def test_detectors_tuple_matches_what_score_text_accepts():
    assert DETECTORS == ("binoculars", "features")


def test_features_verdict_direction(tmp_path):
    """FeatureDetector returns P(AI): high score must mean AI."""
    from aidetect import FeatureDetector

    human = ["Short. Then a far longer rambling sentence that wanders about! Why? Nobody knows."] * 15
    ai = ["The study measured rainfall at three upland sites."] * 15
    model = tmp_path / "m.pkl"
    FeatureDetector().fit(human + ai, [0] * 15 + [1] * 15).save(model)

    v = score_text(ai[0], "features", model=str(model))
    assert isinstance(v, Verdict)
    assert v.detector == "features"
    assert v.is_ai is (v.score >= v.threshold)
    assert v.label in ("AI-generated", "human")


def test_verdict_str_includes_extra_metrics():
    v = Verdict("binoculars", 0.8, 0.9015, is_ai=True, extra={"perplexity": 12.5})
    text = str(v)
    assert "verdict=AI-generated" in text
    assert "perplexity=12.50" in text


# --------------------------------------------------------------- device pick


def test_pick_dtype_is_float32_on_cpu():
    """A CUDA-built torch on a machine with no GPU must land on a CPU-sane dtype."""
    torch = pytest.importorskip("torch")
    binoculars = pytest.importorskip("aidetect.binoculars")

    assert binoculars.pick_dtype("cpu") is torch.float32


def test_pick_device_reports_cpu_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    binoculars = pytest.importorskip("aidetect.binoculars")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert binoculars.pick_device() == "cpu"


def test_pick_device_reports_cuda_when_present(monkeypatch):
    torch = pytest.importorskip("torch")
    binoculars = pytest.importorskip("aidetect.binoculars")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert binoculars.pick_device() == "cuda"


def test_pick_devices_puts_one_model_on_each_gpu(monkeypatch):
    """Two 7B models do not fit on one 24GB card, so a second GPU must be used.

    Returning a bare 'cuda' for both put the pair on device 0 and left a second
    card idle, which is an out-of-memory error rather than a slow run.
    """
    torch = pytest.importorskip("torch")
    binoculars = pytest.importorskip("aidetect.binoculars")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    assert binoculars.pick_devices() == ("cuda:0", "cuda:1")


def test_pick_devices_shares_one_gpu_when_that_is_all_there_is(monkeypatch):
    torch = pytest.importorskip("torch")
    binoculars = pytest.importorskip("aidetect.binoculars")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert binoculars.pick_devices() == ("cuda:0", "cuda:0")


def test_pick_devices_falls_back_to_cpu(monkeypatch):
    torch = pytest.importorskip("torch")
    binoculars = pytest.importorskip("aidetect.binoculars")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert binoculars.pick_devices() == ("cpu", "cpu")


def test_pick_devices_names_an_explicit_index_not_bare_cuda(monkeypatch):
    """'cuda' means device 0 to torch, so a second card is only reachable by index."""
    torch = pytest.importorskip("torch")
    binoculars = pytest.importorskip("aidetect.binoculars")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    observer, performer = binoculars.pick_devices()
    assert observer != performer
    assert observer.startswith("cuda:") and performer.startswith("cuda:")


def test_verdict_str_includes_device():
    v = Verdict("binoculars", 0.8, 0.9, is_ai=True, extra={"perplexity": 3.0}, device="cuda")
    assert "device=cuda" in str(v)


# --------------------------------------------------------------- model cache


def test_binoculars_pair_is_loaded_once_across_calls(monkeypatch):
    """Reloading Falcon-7B per document cost minutes and tens of GB each time."""
    from aidetect import scoring

    scoring.clear_model_cache()
    loads = []

    class FakeBinoculars:
        def __init__(self, observer_name, performer_name):
            loads.append((observer_name, performer_name))
            self.device = "cpu"
            self.threshold = 0.9

        def score(self, text):
            from aidetect.binoculars import BinocularsResult

            return BinocularsResult(0.5, True, self.threshold, 2.0, 4.0)

    pytest.importorskip("aidetect.binoculars")
    monkeypatch.setattr("aidetect.binoculars.Binoculars", FakeBinoculars)

    for _ in range(3):
        scoring.score_text("word " * 60, "binoculars")

    assert len(loads) == 1, f"model pair reloaded {len(loads)} times, expected once"
    scoring.clear_model_cache()


def test_changing_the_model_pair_reloads(monkeypatch):
    from aidetect import scoring

    scoring.clear_model_cache()
    loads = []

    class FakeBinoculars:
        def __init__(self, observer_name, performer_name):
            loads.append((observer_name, performer_name))
            self.device = "cpu"
            self.threshold = 0.9

        def score(self, text):
            from aidetect.binoculars import BinocularsResult

            return BinocularsResult(0.5, True, self.threshold, 2.0, 4.0)

    pytest.importorskip("aidetect.binoculars")
    monkeypatch.setattr("aidetect.binoculars.Binoculars", FakeBinoculars)

    scoring.score_text("word " * 60, "binoculars", observer="a", performer="b")
    scoring.score_text("word " * 60, "binoculars", observer="c", performer="d")

    assert loads == [("a", "b"), ("c", "d")]
    scoring.clear_model_cache()


def test_threshold_still_applies_per_call_when_cached(monkeypatch):
    """A cached detector must not keep the previous call's threshold."""
    from aidetect import scoring

    scoring.clear_model_cache()

    class FakeBinoculars:
        def __init__(self, observer_name, performer_name):
            self.device = "cpu"
            self.threshold = 0.9

        def score(self, text):
            from aidetect.binoculars import BinocularsResult

            return BinocularsResult(0.5, 0.5 < self.threshold, self.threshold, 2.0, 4.0)

    pytest.importorskip("aidetect.binoculars")
    monkeypatch.setattr("aidetect.binoculars.Binoculars", FakeBinoculars)

    first = scoring.score_text("word " * 60, "binoculars", threshold=0.8)
    second = scoring.score_text("word " * 60, "binoculars", threshold=0.2)

    assert first.threshold == 0.8
    assert second.threshold == 0.2
    scoring.clear_model_cache()
