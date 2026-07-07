"""Compatibility package for the JusticeLens project modules.

The codebase imports modules as ``justicelens.<module>`` even though the
actual source files live at the project root. This module makes the project
root behave like a package so those imports continue to resolve.
"""

from __future__ import annotations

from pathlib import Path

# Allow ``import justicelens.<module>`` to resolve top-level ``<module>.py``
# files that live in the project root.
__path__ = [str(Path(__file__).resolve().parent)]
