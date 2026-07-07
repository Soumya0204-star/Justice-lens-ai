"""
Home.py
========
JusticeLens AI -- Streamlit application entrypoint (Home page).

Run with: ``streamlit run app/Home.py``
"""

import sys
from pathlib import Path

def _find_project_root(start: Path) -> Path:
    for candidate in [start] + list(start.parents):
        if (candidate / "justicelens").is_dir():
            return candidate
    return start

_PROJECT_ROOT = _find_project_root(Path(__file__).resolve())
_APP_DIR = _PROJECT_ROOT / "app"
for _path in (str(_PROJECT_ROOT), str(_APP_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import pandas as pd
import streamlit as st
from common import data_service as ds
from common import theme
from justicelens.utils import JusticeLensError

# --- Page configuration ---
theme.apply_page_config("Home", page_icon="⚖️")
theme.inject_global_css()

# --- SIDEBAR: Brand FIRST ---
theme.render_sidebar_brand()  # This puts "JusticeLens AI" at the top

# --- Sidebar additional info (optional) ---
st.sidebar.markdown(
    """
    Navigate using the pages listed above.

    **Data source:** data.gov.in -- District-wise Tele-Law Case
    Registration & Advice Enabled Data (FY 2021-22 to FY 2024-25).
    A synthetic fallback dataset is used automatically if the live
    dataset cannot be reached.
    """
)

# --------------------------------------------------------------------------- #
# Main Content
# --------------------------------------------------------------------------- #
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

# --------------------------------------------------------------------------- #
# Load data & display KPIs
# --------------------------------------------------------------------------- #
try:
    features_df = ds.load_features_dataset()
    artifacts = ds.get_training_artifacts(features_df)
    predictions_df = ds.get_predictions_df(artifacts, features_df)
except JusticeLensError as exc:
    st.error(f"The data/model pipeline could not be prepared: {exc}\n\n"
             "Check the application logs for details. Some pages may be unavailable.")
    st.stop()

is_synthetic = bool(features_df.attrs.get("is_synthetic", False))
if is_synthetic:
    st.info(
        "Live retrieval of the official data.gov.in dataset was unavailable, "
        "so the pipeline is currently running on a **deterministic synthetic fallback dataset**. "
        "All figures below reflect that synthetic data, not official records.",
        icon="ℹ️"
    )

theme.section_eyebrow("Headline Figures")
kpi_cols = st.columns(4)
total_records = len(predictions_df)
underserved_count = int((predictions_df["predicted_class"] == "Underserved").sum())
underserved_share = underserved_count / max(total_records, 1)
states_covered = predictions_df["state_name"].nunique()
best_model_auc = artifacts.best_result.test_metrics.get("roc_auc")

with kpi_cols[0]:
    theme.metric_card("District-Year Records Analyzed", f"{total_records:,}", "navy")
with kpi_cols[1]:
    theme.metric_card(
        "Flagged Underserved",
        f"{underserved_share:.1%}",
        "oxblood",
        note=f"{underserved_count:,} of {total_records:,} records",
    )
with kpi_cols[2]:
    theme.metric_card("States / UTs Covered", f"{states_covered}", "teal")
with kpi_cols[3]:
    theme.metric_card(
        "Best Model (Test ROC-AUC)",
        f"{best_model_auc:.3f}" if best_model_auc is not None else "N/A",
        "gold",
        note=artifacts.best_result.model_name,
    )

st.markdown("---")

# --------------------------------------------------------------------------- #
# IBM watsonx.ai connection status
# --------------------------------------------------------------------------- #
client = ds.get_granite_client()
if client.is_available():
    st.success(
        f"Connected to IBM watsonx.ai -- Granite narration is live "
        f"(model: `{client._config.model_id}`).",
        icon="🤖",
    )
else:
    st.warning(
        "IBM watsonx.ai is not configured (no `.env` credentials found). "
        "Narrative features will use a deterministic template fallback.",
        icon="📋",
    )

st.markdown("---")

# --------------------------------------------------------------------------- #
# Navigation – use the sidebar, but also provide quick links
# --------------------------------------------------------------------------- #
theme.section_eyebrow("Explore")
st.markdown("### Use the sidebar to navigate")

# Optionally, you can add page links here if you want:
pages = [
    ("Executive Dashboard", "1_📊_Executive_Dashboard.py"),
    ("District Explorer", "2_🗺️_District_Explorer.py"),
    ("State Comparison", "3_⚖️_State_Comparison.py"),
    ("Predictive Intelligence", "4_🔮_Predictive_Intelligence.py"),
    ("Explainable AI", "5_🧠_Explainable_AI.py"),
    ("IBM watsonx Policy Room", "6_🏛️_IBM_watsonx_Policy_Room.py"),
    ("About", "7_ℹ️_About.py"),
]

cols = st.columns(3)
for i, (label, filename) in enumerate(pages):
    with cols[i % 3]:
        try:
            st.page_link(f"pages/{filename}", label=f"Open {label}", icon="➡️")
        except Exception:
            st.caption(f"→ {label} (use sidebar)")

st.caption("JusticeLens AI -- built for the IBM SkillsBuild Internship, Problem Statement 37.")