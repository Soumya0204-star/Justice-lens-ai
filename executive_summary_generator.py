"""Compatibility wrapper for the executive summary generator.

The codebase historically imported ``executive_summary_generator`` even
though the implementation now lives in ``executive_summary.py``.
"""

from executive_summary import *  # noqa: F401,F403
