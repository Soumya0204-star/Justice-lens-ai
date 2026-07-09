"""Compatibility wrapper for the shared JusticeLens data service.

This module preserves the historical import path used by the Streamlit
pages: ``from common import data_service as ds``.
"""

from .remote_data_service import *  # noqa: F401,F403
