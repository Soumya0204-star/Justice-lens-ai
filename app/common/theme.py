"""Compatibility wrapper for the shared Streamlit theme helpers.

This preserves the historical import path used by the app pages:
``from common import theme``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_THEME_PATH = Path(__file__).resolve().parent.parent / "app" / "theme.py"
_SPEC = importlib.util.spec_from_file_location("_justicelens_app_theme", _THEME_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - defensive guard
	raise ImportError(f"Unable to load theme helpers from {_THEME_PATH}")

_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

for _name in dir(_MODULE):
	if not _name.startswith("_"):
		globals()[_name] = getattr(_MODULE, _name)

