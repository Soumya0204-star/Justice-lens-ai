"""
common/data_service.py
=======================
Remote-only data service for JusticeLens AI.
Fetches all data from the backend API (Render.com).
"""

from __future__ import annotations

import os
import pandas as pd
import requests
import streamlit as st
from typing import List, Dict, Any, Optional

# --- API URL from secrets ---
API_URL = os.getenv("API_URL", st.secrets.get("API_URL", None))

if not API_URL:
    st.warning("⚠️ API_URL not set. Please set it in Streamlit Cloud secrets.", icon="⚠️")

# --- API Calls ---
@st.cache_data(ttl=3600)
def get_states() -> List[str]:
    """Get list of all states/UTs."""
    if not API_URL:
        return []
    try:
        return requests.get(f"{API_URL}/states").json()
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_districts(state: str) -> List[str]:
    """Get districts for a specific state."""
    if not API_URL:
        return []
    try:
        return requests.get(f"{API_URL}/districts/{state}").json()
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_years() -> List[str]:
    """Get available fiscal years."""
    if not API_URL:
        return []
    try:
        return requests.get(f"{API_URL}/years").json()
    except Exception:
        return []

@st.cache_data(ttl=3600)
def get_district_data(state: str, district: str, year: str) -> pd.DataFrame:
    """Get full profile for a specific district-year."""
    if not API_URL:
        return pd.DataFrame()
    try:
        res = requests.get(f"{API_URL}/district-data", params={"state": state, "district": district, "year": year})
        return pd.DataFrame([res.json()])
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=600)
def run_prediction(features: Dict[str, float]) -> Dict[str, Any]:
    """Send features to API for prediction."""
    if not API_URL:
        return {"error": "API_URL not configured"}
    try:
        res = requests.post(f"{API_URL}/predict", json={"features": features})
        return res.json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600)
def get_executive_summary(scope: str = "All India") -> Dict[str, Any]:
    """Get executive summary from API."""
    if not API_URL:
        return {"error": "API_URL not configured"}
    try:
        return requests.get(f"{API_URL}/executive-summary", params={"scope": scope}).json()
    except Exception as e:
        return {"error": str(e)}

@st.cache_data(ttl=3600)
def get_unique_states(predictions_df=None) -> List[str]:
    return get_states()

@st.cache_data(ttl=3600)
def get_unique_fiscal_years(predictions_df=None) -> List[str]:
    return get_years()

def get_districts_for_state(predictions_df=None, state_name: str = "") -> List[str]:
    return get_districts(state_name)

# --- Dummy functions for compatibility (not used in remote mode) ---
def load_features_dataset():
    return pd.DataFrame()

def get_training_artifacts(df):
    return None

def get_predictions_df(artifacts, df):
    return pd.DataFrame()

def get_shap_explainer(artifacts):
    return None, None

def get_fallback_feature_importance(artifacts):
    return None

def get_granite_client():
    class DummyClient:
        def is_available(self):
            return False
    return DummyClient()

def get_generators(client):
    return {}

def build_district_record(row, shap_explainer, features_df):
    return {"top_contributing_features": []}

# --- For theme compatibility ---
from common import theme