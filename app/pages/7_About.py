import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

"""
7_ℹ️_About.py
===============
About: project background, architecture, methodology, data sources, and
credits.
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
    # Explicitly ensure both the project root (for "justicelens") and the
    # app directory (for "common") are importable, rather than relying on
    # Streamlit's own sys.path handling of the entrypoint script's
    # directory -- this keeps every page self-sufficient and version
    # -independent.
    if _path not in sys.path:
        sys.path.insert(0, _path)

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

    JusticeLens AI is an AI decision-support system that analyzes
    district-wise Tele-Law case registration data alongside demographic
    and regional indicators to identify where India's legal-aid access
    program is under-reaching, explain **why** using SHAP explainability,
    and generate policy-ready narratives using IBM Granite via IBM
    watsonx.ai.
    """
)

st.markdown("---")

col_left, col_right = st.columns(2)

with col_left:
    theme.section_eyebrow("Technology Stack")
    st.markdown(
        """
        - **IBM Cloud Lite** -- Object Storage, watsonx.ai project hosting
        - **IBM watsonx.ai** -- Foundation model runtime (REST API)
        - **IBM Granite** -- narration, summarization, Q&A
        - **Python** -- pandas, scikit-learn, XGBoost, SHAP
        - **Streamlit** -- this application
        - **Plotly** -- all interactive charts
        - **IBM Bob** -- development-time coding assistant (see
          `docs/ibm_bob_workflow.md`)
        """
    )

    theme.section_eyebrow("Data Source")
    st.markdown(
        """
        [data.gov.in -- District-wise Tele-Law Case Registration and
        Advice Enabled Data, FY 2021-22 to FY 2024-25](https://www.data.gov.in/resource/district-wise-tele-law-case-registration-and-advice-enabled-data-fy-2021-22-2024-25)

        If live retrieval of this dataset fails (e.g. no network access,
        API changes), the pipeline automatically falls back to a
        **clearly-labeled, deterministic synthetic dataset** with the
        same schema, so the system remains fully demonstrable.
        """
    )

with col_right:
    theme.section_eyebrow("ML Pipeline")
    st.markdown(
        """
        1. **Ingest** the official dataset (with synthetic fallback)
        2. **Validate** schema, ranges, duplicates, and reporting gaps
        3. **Harmonize** schema drift across fiscal years
        4. **Reconcile** district names against demographic data (fuzzy
           matching)
        5. **Engineer features**: per-capita rate, advice ratio, YoY
           growth, rural penetration, literacy-adjusted load
        6. **Train & compare 5 models**: Logistic Regression, Decision
           Tree, Random Forest, Gradient Boosting, XGBoost
        7. **Select the best model** by cross-validated ROC-AUC
        8. **Explain every prediction** with SHAP
        9. **Narrate** results with IBM Granite (deterministic template
           fallback if watsonx.ai is unavailable)
        """
    )

    theme.section_eyebrow("Reliability Principle")
    st.markdown(
        """
        Every generative-AI feature in this application is backed by a
        deterministic, non-AI fallback. If IBM watsonx.ai is unreachable
        or not configured, every page keeps working -- numeric results,
        charts, and SHAP explanations are never dependent on the
        generative layer, which only ever *narrates* precomputed,
        auditable figures.
        """
    )

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
            "Synthetic Fallback" if features_df.attrs.get("is_synthetic") else "Live (data.gov.in)",
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
except Exception as exc:  # noqa: BLE001 -- About page must never hard-crash
    st.warning(f"Could not retrieve live session status: {exc}")

st.markdown("---")
st.caption(
    "JusticeLens AI -- built for the IBM SkillsBuild Internship, "
    "Problem Statement 37. All district-level figures in this "
    "application are model estimates for decision-support purposes and "
    "should be validated against official records before use in binding "
    "policy decisions."
)
