"""
Home.py
========
JusticeLens AI -- Streamlit application entrypoint (Home page).
"""

import sys
import os
from pathlib import Path

# --- FIX: Add project root to sys.path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent

# Add paths so Python can find everything
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

# --- Now ALL imports work ---
import pandas as pd
import streamlit as st

from common import data_service as ds
from common import theme
from justicelens.utils import JusticeLensError

# --- Page config ---
theme.apply_page_config("Home", page_icon="⚖️")
theme.inject_global_css()

# --- Sidebar ---
st.sidebar.markdown(
    f"""
    <div style="padding: 0.4rem 0 1rem 0;">
        <div style="font-family: {theme.FONT_DISPLAY}; font-size: 1.35rem; font-weight: 700; color: {theme.PAPER};">
            ⚖️ {theme.APP_TITLE}
        </div>
        <div style="font-family: {theme.FONT_BODY}; font-size: 0.78rem; color: {theme.STATUTE_GOLD}; margin-top: 0.1rem;">
            {theme.APP_TAGLINE}
        </div>
    </div>
    <hr/>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    Navigate using the pages listed above.

    **Data source:** data.gov.in -- District-wise Tele-Law Case
    Registration & Advice Enabled Data (FY 2021-22 to FY 2024-25).
    A synthetic fallback dataset is used automatically if the live
    dataset cannot be reached.
    """
)

# --- Main content ---
theme.render_app_header(
    title=theme.APP_TITLE,
    subtitle=theme.APP_TAGLINE,
    eyebrow="AI Decision Support System · IBM SkillsBuild Internship Problem Statement 37",
)

st.markdown(
    """
    JusticeLens AI analyzes district-wise Tele-Law case registrations
    against demographic and regional indicators to surface where India's
    legal-aid access program is under-reaching -- and explains **why**,
    using a validated machine-learning pipeline (Logistic Regression,
    Decision Tree, Random Forest, Gradient Boosting, XGBoost) with SHAP
    explainability, narrated by IBM Granite via IBM watsonx.ai.
    """
)

st.markdown("---")

# --- Load data and show KPIs ---
try:
    features_df = ds.load_features_dataset()
    artifacts = ds.get_training_artifacts(features_df)
    predictions_df = ds.get_predictions_df(artifacts, features_df)
except JusticeLensError as exc:
    st.error(f"Could not load data: {exc}")
    st.stop()

is_synthetic = bool(features_df.attrs.get("is_synthetic", False))
if is_synthetic:
    st.info(
        "⚠️ Running on synthetic fallback dataset (live data.gov.in unavailable).",
        icon="ℹ️",
    )

theme.section_eyebrow("Headline Figures")
kpi_cols = st.columns(4)
total_records = len(predictions_df)
underserved_count = int((predictions_df["predicted_class"] == "Underserved").sum())
underserved_share = underserved_count / max(total_records, 1)
states_covered = predictions_df["state_name"].nunique()
best_model_auc = artifacts.best_result.test_metrics.get("roc_auc")

with kpi_cols[0]:
    theme.metric_card("District-Year Records", f"{total_records:,}", "navy")
with kpi_cols[1]:
    theme.metric_card("Flagged Underserved", f"{underserved_share:.1%}", "oxblood", note=f"{underserved_count:,} records")
with kpi_cols[2]:
    theme.metric_card("States / UTs", f"{states_covered}", "teal")
with kpi_cols[3]:
    theme.metric_card("Best Model (ROC-AUC)", f"{best_model_auc:.3f}" if best_model_auc else "N/A", "gold", note=artifacts.best_result.model_name)

st.markdown("---")

# --- watsonx.ai status ---
client = ds.get_granite_client()
if client.is_available():
    st.success(f"✅ Connected to IBM watsonx.ai (model: `{client._config.model_id}`)", icon="🤖")
else:
    st.warning("⚠️ IBM watsonx.ai not configured. Using deterministic template fallback.", icon="📋")

st.markdown("---")

# --- Navigation ---
theme.section_eyebrow("Explore")
st.markdown("### Use the sidebar to navigate")

st.caption("JusticeLens AI -- built for the IBM SkillsBuild Internship, Problem Statement 37.")