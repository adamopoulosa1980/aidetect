"""The section analysis as a Markdown report.

A report outlives the session that produced it, so it has to carry the things
a reader cannot reconstruct: which direction of the threshold means AI, and who
the method belongs to.
"""

from __future__ import annotations

from aidetect.scoring import DocumentVerdict, Section


def _doc(detector="binoculars", threshold=0.9015):
    sections = [
        Section(0, "Human sounding prose with variety.", 1.20, is_ai=False),
        Section(1, "Flagged | passage with a pipe in it.", 0.80, is_ai=True),
        Section(2, "More prose.", 0.95, is_ai=False),
    ]
    return DocumentVerdict(detector, threshold, sections, device="cuda:0+cuda:1")


def test_report_states_which_direction_means_ai():
    """The detail people get backwards, so it is spelled out, not implied."""
    md = _doc().to_markdown()
    assert "**below** the threshold" in md
    assert "scores machine-written text *low*" in md


def test_features_detector_states_the_opposite_direction():
    md = _doc(detector="features", threshold=0.5).to_markdown()
    assert "**above** the threshold" in md
    assert "*high*" in md


def test_report_credits_the_binoculars_authors():
    """An exported report naming the method has to carry the citation."""
    md = _doc().to_markdown()
    assert "arXiv:2401.12070" in md
    assert "Abhimanyu Hans" in md
    assert "ahans30/Binoculars" in md


def test_features_report_does_not_credit_binoculars():
    """The citation belongs to the method actually used, not to every report."""
    assert "arXiv:2401.12070" not in _doc(detector="features", threshold=0.5).to_markdown()


def test_every_section_appears_in_the_table():
    md = _doc().to_markdown()
    rows = [line for line in md.splitlines() if line.startswith("| 1 |")]
    assert rows, "section 1 missing from the table"
    for n in (1, 2, 3):
        assert any(line.startswith(f"| {n} |") for line in md.splitlines())


def test_headline_and_counts_are_reported():
    md = _doc().to_markdown()
    assert "1 of 3 sections would be flagged" in md
    assert "| Flagged | 1 of 3 |" in md
    assert "cuda:0+cuda:1" in md


def test_source_is_recorded_when_given():
    assert "`report.docx`" in _doc().to_markdown(source="report.docx")
    assert "| Source |" not in _doc().to_markdown()


def test_mean_is_present_but_the_caveat_travels_with_it():
    """The mean is the number people over-read, so the warning ships beside it."""
    md = _doc().to_markdown()
    assert "| Mean score |" in md
    assert "rarely tells you anything" in md


def test_stylometry_report_has_no_threshold_or_verdict():
    sections = [Section(0, "Prose.", 0.0, is_ai=False)]
    md = DocumentVerdict("stylometry", 0.0, sections).to_markdown()
    assert "Threshold" not in md
    assert "would be flagged" not in md
    assert "Sections worth looking at" in md


def test_flagged_sections_get_their_full_text():
    md = _doc().to_markdown()
    assert "### Section 2 -- score 0.8000" in md
    # Pipes only need escaping inside tables; a blockquote takes them raw.
    assert "> Flagged | passage with a pipe in it." in md


def test_pipes_are_escaped_so_the_table_survives():
    """An unescaped pipe in the text would split the row and shift every cell."""
    backslash = chr(92)
    md = _doc().to_markdown()
    table = [ln for ln in md.splitlines() if ln.startswith("| 2 |")][0]
    assert backslash + "|" in table, "the pipe in the text was not escaped"
    # An escaped pipe is not a delimiter, so remove those before counting.
    delimiters = table.replace(backslash + "|", "").count("|")
    assert delimiters == 5, table  # 4 cells => 5 delimiters, none stray
