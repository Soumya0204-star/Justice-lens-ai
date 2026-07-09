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
5_🧠_Explainable_AI.py
========================
Explainable AI: global SHAP summary plot, global feature importance,
and a per-record SHAP waterfall / structured prediction explanation.
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

import plotly.express as px
import streamlit as st

from common import data_service as ds
from common import theme
from justicelens.utils import ExplainabilityError

theme.apply_page_config("Explainable AI", page_icon="🧠")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "Explainable AI",
    subtitle="SHAP (SHapley Additive exPlanations) analysis of the disparity classification model.",
    eyebrow="Model Transparency",
)

features_df = ds.load_features_dataset()
artifacts = ds.get_training_artifacts(features_df)
predictions_df = ds.get_predictions_df(artifacts, features_df)
shap_explainer, shap_error = ds.get_shap_explainer(artifacts)

if shap_explainer is None:
    st.warning(
        f"SHAP explainability is unavailable in this environment: "
        f"{shap_error}

Install the `shap` package "
        "(`pip install shap`, already listed in `requirements.txt`) and "
        "restart the app to enable this page's full functionality. Below "
        "is an approximate feature importance computed directly from the "
        "model in the meantime.",
        icon="⚠️",
    )
    fallback_importance = ds.get_fallback_feature_importance(artifacts)
    if fallback_importance is not None:
        st.markdown("#### Approximate Global Feature Importance")
        fig = px.bar(
            fallback_importance.sort_values("importance"),
            x="importance", y="feature", orientation="h",
        )
        fig.update_traces(marker_color=theme.STEEL_BLUE)
        theme.style_plotly_fig(fig, height=420)
        st.plotly_chart(fig, use_container_width=True)
    st.stop()

st.success(f"SHAP explainer ready for model: **{artifacts.best_result.model_name}**")

tab_global, tab_local = st.tabs(["🌍 Global Explainability", "🔎 Local (Single Record) Explainability"])

# --------------------------------------------------------------------------- #
# Global tab
# --------------------------------------------------------------------------- #
with tab_global:
    theme.section_eyebrow("Model-Wide Analysis")
    st.markdown("#### Global Feature Importance")

    importance_df = shap_explainer.compute_feature_importance(artifacts.X_test)
    fig_importance = px.bar(
        importance_df.sort_values("mean_abs_shap_value"),
        x="mean_abs_shap_value", y="feature", orientation="h",
    )
    fig_importance.update_traces(marker_color=theme.STEEL_BLUE)
    fig_importance.update_layout(xaxis_title="Mean |SHAP value|", yaxis_title="")
    theme.style_plotly_fig(fig_importance, height=420)
    st.plotly_chart(fig_importance, use_container_width=True)

    st.markdown("#### SHAP Summary Plot")
    st.caption(
        "Each point is one district-year record. Position on the x-axis "
        "shows the SHAP value (impact on underserved-risk prediction); "
        "color shows the feature's actual value for that record."
    )
    with st.spinner("Rendering SHAP summary plot..."):
        try:
            summary_path = shap_explainer.generate_summary_plot(artifacts.X_test)
            st.image(str(summary_path), use_container_width=True)
        except ExplainabilityError as exc:
            st.error(f"Could not generate the SHAP summary plot: {exc}")

# --------------------------------------------------------------------------- #
# Local tab
# --------------------------------------------------------------------------- #
with tab_local:
    theme.section_eyebrow("Single-Record Analysis")
    st.markdown("#### Select a District-Year Record")

    select_cols = st.columns(3)
    with select_cols[0]:
        state_choice = st.selectbox("State / UT", options=ds.get_unique_states(predictions_df), key="local_state")
    with select_cols[1]:
        district_choice = st.selectbox(
            "District", options=ds.get_districts_for_state(predictions_df, state_choice), key="local_district"
        )
    with select_cols[2]:
        year_choice = st.selectbox(
            "Fiscal Year", options=ds.get_unique_fiscal_years(predictions_df), key="local_year"
        )

    record_df = features_df[
        (features_df["state_name"] == state_choice)
        & (features_df["district_name"] == district_choice)
        & (features_df["fiscal_year"] == year_choice)
    ]

    if record_df.empty:
        st.warning("No record found for this selection.")
    else:
        single_row_df = record_df

        col_left, col_right = st.columns([1, 1])

        with col_left:
            st.markdown("##### SHAP Waterfall Plot")
            with st.spinner("Rendering waterfall plot..."):
                try:
                    waterfall_path = shap_explainer.generate_waterfall_plot(single_row_df, instance_index=0)
                    st.image(str(waterfall_path), use_container_width=True)
                except ExplainabilityError as exc:
                    st.error(f"Could not generate the waterfall plot: {exc}")

        with col_right:
            st.markdown("##### Structured Prediction Explanation")
            try:
                explanation = shap_explainer.explain_prediction(single_row_df, instance_index=0)
                st.markdown(theme.tier_badge_html(explanation["predicted_class"]), unsafe_allow_html=True)
                st.markdown("")
                theme.metric_card(
                    "Predicted Probability", f"{explanation['predicted_probability']:.1%}",
                    "oxblood" if explanation["predicted_class"] == "Underserved" else "teal",
                    note=f"Baseline: {explanation['base_probability']:.1%}",
                )
                st.markdown("**Top Contributing Features**")
                for item in explanation["top_contributing_features"]:
                    icon = "🔺" if "increases" in item["direction"] else "🔻"
                    st.markdown(
                        f"{icon} **{item['feature'].replace('_', ' ').title()}** "
                        f"(value={item['value']}, SHAP={item['shap_value']:+.4f}) "
                        f"-- {item['direction']}"
                    )
                st.markdown("**Narrative Summary**")
                st.info(explanation["narrative_ready_summary"])
            except ExplainabilityError as exc:
                st.error(f"Could not generate a prediction explanation: {exc}")
