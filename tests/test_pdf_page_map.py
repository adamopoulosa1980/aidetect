"""The PDF page map, checked against a real PDF rather than a mock.

A page number that is merely plausible is worse than none: it sends someone to
the wrong part of the document with full confidence. So this builds a PDF whose
words say which page they are on, and checks every mapped range resolves to
exactly those words.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pypdf")

from aidetect.readers import load_text, page_spans  # noqa: E402


def build_pdf(pages_words):
    objs, out = [], bytearray(b"%PDF-1.4\n")
    n_pages = len(pages_words)
    font_id = 3 + 2 * n_pages
    page_ids = [3 + 2 * i for i in range(n_pages)]

    def add(body):
        objs.append(bytes(body))

    add(b"<</Type/Catalog/Pages 2 0 R>>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    add(f"<</Type/Pages/Kids[{kids}]/Count {n_pages}>>".encode())
    for i, words in enumerate(pages_words):
        content_id = page_ids[i] + 1
        add(
            f"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]"
            f"/Resources<</Font<</F1 {font_id} 0 R>>>>/Contents {content_id} 0 R>>".encode()
        )
        lines, y = [], 700
        for start in range(0, len(words), 6):
            text = " ".join(words[start : start + 6])
            lines.append(f"BT /F1 12 Tf 50 {y} Td ({text}) Tj ET")
            y -= 18
        stream = "\n".join(lines).encode()
        add(b"<</Length " + str(len(stream)).encode() + b">>\nstream\n" + stream + b"\nendstream")
    add(b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>")

    offsets = []
    for i, body in enumerate(objs, 1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_at = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<</Size {len(objs) + 1}/Root 1 0 R>>\nstartxref\n{xref_at}\n%%EOF\n"
    ).encode()
    return bytes(out)




PAGES, PER_PAGE = 4, 30


@pytest.fixture
def mapped_pdf(tmp_path):
    expected = [[f"p{p + 1}w{i}" for i in range(PER_PAGE)] for p in range(PAGES)]
    path = tmp_path / "mapped.pdf"
    path.write_bytes(build_pdf(expected))
    return path, expected


def test_every_page_maps_to_exactly_its_own_words(mapped_pdf):
    path, expected = mapped_pdf
    spans = page_spans(path)
    words = load_text(path).split()
    assert len(spans) == PAGES
    for (number, start, end), page_words in zip(spans, expected):
        assert words[start:end] == page_words, f"page {number} maps to the wrong words"


def test_page_numbers_are_the_real_ones(mapped_pdf):
    path, _ = mapped_pdf
    assert [n for n, _, _ in page_spans(path)] == [1, 2, 3, 4]


def test_spans_account_for_the_whole_document(mapped_pdf):
    """A gap or overlap would misplace every section after it."""
    path, _ = mapped_pdf
    spans = page_spans(path)
    assert spans[0][1] == 0
    assert spans[-1][2] == len(load_text(path).split())
    for (_, _, end), (_, next_start, _) in zip(spans, spans[1:]):
        assert end == next_start


def test_non_pdf_formats_get_no_page_numbers(tmp_path):
    """Word repaginates on the fly, so a .docx page number would be invented."""
    txt = tmp_path / "plain.txt"
    txt.write_text("some words here", encoding="utf-8")
    assert page_spans(txt) == []


def test_an_unreadable_pdf_yields_no_map_rather_than_a_wrong_one(tmp_path):
    bad = tmp_path / "broken.pdf"
    bad.write_bytes(b"%PDF-1.4 not really a pdf")
    assert page_spans(bad) == []
