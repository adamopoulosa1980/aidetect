# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller build spec for the aidetect CLI.

Two flavours, selected with the AIDETECT_LITE environment variable:

  AIDETECT_LITE=1   Lite, one-file. numpy + scikit-learn + python-docx + pypdf
                    + tkinter, roughly 67 MB. torch/transformers are excluded,
                    so `--detector features --model model.pkl` works and
                    `--detector binoculars` is greyed out in the GUI and exits
                    with a clear message on the CLI.

  (unset)           Full, one-folder. Adds torch + transformers so
                    `--detector binoculars` works. Expect several GB; one-file
                    is deliberately avoided here because unpacking that much
                    on every run costs a minute of startup.

Override the layout explicitly with AIDETECT_ONEFILE=1 or =0.

Build:  pyinstaller --clean --noconfirm aidetect.spec
Output: dist/aidetect.exe (lite) or dist/aidetect/aidetect.exe (full)
"""

import os

from PyInstaller.utils.hooks import collect_all, collect_data_files


def _flag(name, default=False):
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("", "0", "false", "no")


LITE = _flag("AIDETECT_LITE")
ONEFILE = _flag("AIDETECT_ONEFILE", default=LITE)

# python-docx reads its bundled default.docx at import time; without this the
# frozen app raises PackageNotFoundError on the first .docx it is given.
datas = list(collect_data_files("docx"))
binaries = []

# scikit-learn's Cython extensions are reached indirectly and are easy for the
# static analysis to miss.
hiddenimports = [
    "sklearn.ensemble._hist_gradient_boosting._gradient_boosting",
    "sklearn.utils._typedefs",
    "sklearn.utils._heap",
    "sklearn.utils._sorting",
    "pypdf",
]

# Never needed by this app; excluding them keeps both flavours smaller.
# tkinter is NOT excluded - aidetect.gui needs it.
excludes = ["matplotlib", "IPython", "notebook", "pytest", "PyQt5", "PySide2"]

HEAVY = (
    "torch",
    "transformers",
    "accelerate",
    "tokenizers",
    "safetensors",
    "huggingface_hub",
)

if LITE:
    excludes += list(HEAVY) + ["PIL", "pandas"]
else:
    missing = []
    for pkg in HEAVY:
        try:
            pkg_datas, pkg_binaries, pkg_hidden = collect_all(pkg)
        except Exception:
            missing.append(pkg)
            continue
        datas += pkg_datas
        binaries += pkg_binaries
        hiddenimports += pkg_hidden
    if missing:
        raise SystemExit(
            "Full build needs the GPU extras installed in the build environment.\n"
            "Missing: " + ", ".join(missing) + "\n"
            'Run:  pip install -e ".[gpu]"   (or build the lite flavour with '
            "AIDETECT_LITE=1)"
        )

a = Analysis(
    ["packaging/entry.py"],
    pathex=[SPECPATH],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)

if ONEFILE:
    exe = EXE(
        pyz,
        a.scripts,
        a.binaries,
        a.datas,
        [],
        name="aidetect",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name="aidetect",
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=False,
        console=True,
        disable_windowed_traceback=False,
        argv_emulation=False,
        target_arch=None,
        codesign_identity=None,
        entitlements_file=None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.datas,
        strip=False,
        upx=False,
        name="aidetect",
    )
