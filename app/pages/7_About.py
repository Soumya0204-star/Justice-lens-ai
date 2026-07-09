import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
from common import data_service as ds
from common import theme

theme.apply_page_config("About", page_icon="ℹ️")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "About JusticeLens AI",
    subtitle="Architecture, methodology, data sources, and credits.",
    eyebrow="IBM SkillsBuild Internship · Problem Statement 37",
)

st.markdown(
    """
    **Problem Statement:** *"Analyzing Demographic and Regional
    Disparities in Tele-Law Case Registrations for Inclusive Legal
    Access."*

    JusticeLens AI is a decision-support system that identifies where
    India's legal-aid program is under-reaching, explains **why** with
    SHAP, and generates policy narratives using IBM Granite.
    """
)
st.markdown("---")

col_left, col_right = st.columns(2)
with col_left:
    theme.section_eyebrow("Technology Stack")
    st.markdown("""
    - **IBM Cloud Lite** – Object Storage, watsonx.ai
    - **IBM watsonx.ai** – Foundation model runtime
    - **IBM Granite** – Narration, Q&A
    - **Python** – pandas, scikit-learn, XGBoost, SHAP
    - **Streamlit** – Dashboard
    - **Plotly** – Interactive charts
    """)
    theme.section_eyebrow("Data Source")
    st.markdown("[data.gov.in — Tele-Law Case Registration](https://www.data.gov.in/resource/district-wise-tele-law-case-registration-and-advice-enabled-data-fy-2021-22-2024-25)")
with col_right:
    theme.section_eyebrow("ML Pipeline")
    st.markdown("""
    1. Ingest official dataset (synthetic fallback)
    2. Validate, harmonize, reconcile
    3. Engineer features (per-capita, advice ratio, growth, rural penetration)
    4. Train 5 models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)
    5. Select best by ROC-AUC
    6. Explain with SHAP
    7. Narrate with IBM Granite (template fallback if unavailable)
    """)
    theme.section_eyebrow("Reliability Principle")
    st.markdown("Every AI feature has a deterministic fallback – the system works even without watsonx.ai.")

st.markdown("---")
theme.section_eyebrow("Current Session Status")
try:
    features_df = ds.load_features_dataset()
    artifacts = ds.get_training_artifacts(features_df)
    client = ds.get_granite_client()
    shap_explainer, shap_error = ds.get_shap_explainer(artifacts)

    status_cols = st.columns(3)
    with status_cols[0]:
        theme.metric_card(
            "Dataset",
            "Synthetic" if features_df.attrs.get("is_synthetic") else "Live",
            "gold" if features_df.attrs.get("is_synthetic") else "teal",
        )
    with status_cols[1]:
        theme.metric_card(
            "IBM watsonx.ai",
            "Connected" if client.is_available() else "Not Configured",
            "teal" if client.is_available() else "gold",
        )
    with status_cols[2]:
        theme.metric_card(
            "SHAP Explainability",
            "Available" if shap_explainer is not None else "Unavailable",
            "teal" if shap_explainer is not None else "oxblood",
        )
except Exception as exc:
    st.warning(f"Could not retrieve session status: {exc}")

st.markdown("---")
st.caption("JusticeLens AI — IBM SkillsBuild Internship, Problem Statement 37.")