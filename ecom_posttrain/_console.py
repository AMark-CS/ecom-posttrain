"""Console bootstrap.

Windows consoles default to the legacy GBK code page, which makes the
project's Chinese log lines (and emoji markers) blow up with
``UnicodeEncodeError`` or render as mojibake.  Every entry point in this
project calls :func:`setup_console` before printing anything, so output is
always UTF-8 and never crashes on a CJK terminal.
"""

from __future__ import annotations

import sys


def setup_console() -> None:
    """Reconfigure stdout/stderr to UTF-8 with a lossy fallback."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - closed/redirected stream
            pass


def bootstrap() -> None:
    """Make the repository root importable and fix console encoding.

    Scripts live one or two levels below the repository root, so we walk up
    until we find the directory that contains the ``ecommerce_project``
    package and put it on ``sys.path``.
    """
    setup_console()

    from pathlib import Path

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "ecommerce_project" / "__init__.py").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            break
