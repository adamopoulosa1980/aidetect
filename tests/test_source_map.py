"""Mapping a scored section back to a place in the original document.

"Section 412 of 623" says nothing about where to look. These pin the
coordinates that make it findable, and pin them to the chunker's own
arithmetic: an offset that drifts sends someone to the wrong page, which is
worse than offering no offset at all.
"""

from __future__ import annotations

import pytest

from aidetect.ensemble import chunk_spans, chunk_text
from aidetect.scoring import Section, SourceMap, describe_document


def test_spans_match_the_chunks_they_describe():
    """The span arithmetic and the chunk arithmetic must not drift apart."""
    words = [f"w{i}" for i in range(1000)]
    text = " ".join(words)
    chunks = chunk_text(text)
    spans = chunk_spans(text)
    assert len(chunks) == len(spans)
    for chunk, (start, end) in zip(chunks, spans):
        assert chunk.split() == words[start:end]


def test_spans_cover_short_text_without_chunking():
    text = "one two three"
    assert chunk_spans(text) == [(0, 3)]
    assert chunk_text(text) == [text]


def test_sections_carry_their_offsets():
    text = " ".join(f"w{i}" for i in range(1000))
    verdict = describe_document(text)
    assert verdict.sections[0].start_word == 0
    for section in verdict.sections:
        assert section.end_word > section.start_word
        assert section.text.split() == text.split()[section.start_word : section.end_word]


def test_page_range_covers_the_pages_a_section_touches():
    smap = SourceMap(total_words=900, pages=[(1, 0, 300), (2, 300, 600), (3, 600, 900)])
    assert smap.page_range(0, 300) == (1, 1)
    assert smap.page_range(250, 550) == (1, 2)   # straddles a boundary
    assert smap.page_range(600, 900) == (3, 3)


def test_page_range_is_none_without_a_map():
    assert SourceMap(total_words=900).page_range(0, 300) is None


def test_locate_reports_page_words_and_position():
    smap = SourceMap(total_words=1000, pages=[(1, 0, 500), (2, 500, 1000)])
    section = Section(0, "text", 1.0, False, start_word=500, end_word=800)
    where = section.locate(smap)
    assert "page 2" in where
    assert "words 501-800" in where
    assert "65% in" in where


def test_locate_without_pages_still_gives_words_and_percent():
    smap = SourceMap(total_words=1000)
    section = Section(0, "text", 1.0, False, start_word=0, end_word=300)
    where = section.locate(smap)
    assert "page" not in where
    assert "words 1-300" in where


def test_locate_is_ascii_for_the_frozen_console():
    """A frozen console renders non-ASCII as replacement characters."""
    smap = SourceMap(total_words=1000, pages=[(1, 0, 1000)])
    where = Section(0, "t", 1.0, False, start_word=0, end_word=300).locate(smap)
    where.encode("ascii")  # raises if not


def test_anchor_is_whole_words_for_a_find_box():
    """preview truncates mid-word, which is useless in Ctrl+F."""
    text = "The establishment of clear protocols ensures consistency across every department here"
    section = Section(0, text, 1.0, False)
    assert section.anchor in text
    assert not section.anchor.endswith("...")
    assert section.anchor.split() == text.split()[:12]


def test_anchor_appears_verbatim_in_the_document():
    """The whole point: paste it into a find box and land on the section."""
    document = " ".join(f"word{i}" for i in range(1000))
    verdict = describe_document(document)
    for section in verdict.sections:
        assert section.anchor in document


def test_report_records_where_each_section_sits():
    text = " ".join(f"w{i}" for i in range(1000))
    smap = SourceMap(total_words=1000, pages=[(1, 0, 500), (2, 500, 1000)])
    md = describe_document(text, source_map=smap).to_markdown()
    assert "| Document length | 1,000 words |" in md
    assert "| Pages | 2 |" in md
    assert "Where" in md
