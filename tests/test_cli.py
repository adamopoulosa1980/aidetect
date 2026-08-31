"""Tests for CLI argument handling and exit codes.

The frozen-executable branch is the interesting one: a double-clicked .exe gets
no arguments and an unreliable stdin, so it must open the GUI rather than read
stdin and score the empty string.
"""

from __future__ import annotations

import io

import pytest

from aidetect import cli


@pytest.fixture
def fake_gui(monkeypatch):
    """Stand in for the Tk window so tests never open one."""
    opened = []
    monkeypatch.setattr(cli, "_frozen", lambda: False)

    class _Module:
        @staticmethod
        def main():
            opened.append(True)

    monkeypatch.setitem(__import__("sys").modules, "aidetect.gui", _Module)
    return opened


def test_gui_flag_opens_the_window(fake_gui):
    assert cli.main(["--gui"]) == 0
    assert fake_gui == [True]


def test_frozen_with_no_args_opens_the_gui(fake_gui, monkeypatch):
    """A double-clicked executable must not fall through to reading stdin."""
    monkeypatch.setattr(cli, "_frozen", lambda: True)
    # stdin deliberately looks like an exhausted pipe, which is what the
    # double-clicked executable actually saw.
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main([]) == 0
    assert fake_gui == [True]


def test_unfrozen_with_no_args_still_reads_stdin(fake_gui, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main([]) == 1  # empty stdin is an input error, not a GUI launch
    assert fake_gui == []


def test_empty_stdin_is_an_error(monkeypatch, capsys):
    monkeypatch.setattr(cli, "_frozen", lambda: False)
    monkeypatch.setattr("sys.stdin", io.StringIO("   \n  "))
    assert cli.main([]) == 1
    assert "no text on stdin" in capsys.readouterr().err


def test_dash_reads_stdin_explicitly(monkeypatch, tmp_path):
    monkeypatch.setattr(cli, "_frozen", lambda: True)
    monkeypatch.setattr("sys.stdin", io.StringIO("word " * 60))

    from aidetect import FeatureDetector

    model = tmp_path / "m.pkl"
    human = ["Short. A far longer rambling sentence that wanders about! Why? Nobody knows."] * 15
    ai = ["The study measured rainfall at three upland sites."] * 15
    FeatureDetector().fit(human + ai, [0] * 15 + [1] * 15).save(model)

    assert cli.main(["-", "--detector", "features", "--model", str(model)]) == 0


def test_missing_file_exits_1_without_a_traceback(capsys, monkeypatch):
    monkeypatch.setattr(cli, "_frozen", lambda: False)
    assert cli.main(["no_such_file.txt"]) == 1
    assert "aidetect:" in capsys.readouterr().err


def test_unsupported_extension_exits_1(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_frozen", lambda: False)
    p = tmp_path / "x.rtf"
    p.write_text("hello", encoding="utf-8")
    assert cli.main([str(p)]) == 1
    assert "Unsupported file type" in capsys.readouterr().err


def test_features_without_model_exits_2(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(cli, "_frozen", lambda: False)
    p = tmp_path / "x.txt"
    p.write_text("word " * 60, encoding="utf-8")
    assert cli.main([str(p), "--detector", "features"]) == 2
    assert "trained model" in capsys.readouterr().err


def test_console_messages_are_ascii():
    """A frozen console renders anything else as a replacement character.

    An em-dash in the "no torch in this build" message showed up as garbage in
    the packaged executable, in the one message a lite-build user is most
    likely to see. Docstrings and comments are exempt: they never reach a
    console.
    """
    import ast
    import pathlib

    offenders = []
    for name in ("scoring.py", "cli.py", "models.py", "calibrate.py", "binoculars.py"):
        source = pathlib.Path("aidetect") / name
        tree = ast.parse(source.read_text(encoding="utf-8"))

        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, "body", None)
                if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                    if isinstance(body[0].value.value, str):
                        docstrings.add(id(body[0].value))

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings
                and not node.value.isascii()
            ):
                offenders.append(f"{source}:{node.lineno}: {node.value[:60]!r}")

    assert not offenders, "non-ASCII in runtime strings: " + "; ".join(offenders)
