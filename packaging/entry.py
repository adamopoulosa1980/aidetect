"""PyInstaller entry point for the aidetect CLI.

PyInstaller analyses a real script; it cannot start from the ``[project.scripts]``
console-script entry point, so this stays a thin shim over ``aidetect.cli:main``.
"""

from __future__ import annotations

import multiprocessing

if __name__ == "__main__":
    # Frozen Windows apps must call this before anything spawns workers
    # (joblib, which scikit-learn uses, does), or every worker re-runs the CLI.
    multiprocessing.freeze_support()

    from aidetect.cli import main

    raise SystemExit(main())
