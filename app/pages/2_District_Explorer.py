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
2_🗺️_District_Explorer.py
===========================
District Explorer: search/select a single district, view its full
profile, historical trend, SHAP explanation, and a policy recommendation.
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
import plotly.graph_objects as go
import streamlit as st

from common import data_service as ds
from common import theme
from justicelens.utils import JusticeLensError

theme.apply_page_config("District Explorer", page_icon="🗺️")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "District Explorer",
    subtitle="Deep-dive into a single district's Tele-Law profile, history, and drivers.",
    eyebrow="District-Level Detail",
)

features_df = ds.load_features_dataset()
artifacts = ds.get_training_artifacts(features_df)
predictions_df = ds.get_predictions_df(artifacts, features_df)

# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### Select a District")
all_states = ds.get_unique_states(predictions_df)
selected_state = st.sidebar.selectbox("State / UT", options=all_states)
districts_in_state = ds.get_districts_for_state(predictions_df, selected_state)
selected_district = st.sidebar.selectbox("District", options=districts_in_state)

district_history = predictions_df[
    (predictions_df["state_name"] == selected_state)
    & (predictions_df["district_name"] == selected_district)
].sort_values("fiscal_year")

if district_history.empty:
    st.warning("No records found for this district.")
    st.stop()

all_years = ds.get_unique_fiscal_years(predictions_df)
selected_year = st.sidebar.selectbox("Fiscal Year (for profile snapshot)", options=list(reversed(all_years)))
current_row = district_history[district_history["fiscal_year"] == selected_year]
if current_row.empty:
    current_row = district_history.iloc[[-1]]
current_row = current_row.iloc[0]

# --------------------------------------------------------------------------- #
# Profile header
# --------------------------------------------------------------------------- #
st.markdown(f"### {selected_district}, {selected_state} -- FY {current_row['fiscal_year']}")
st.markdown(theme.tier_badge_html(current_row["predicted_class"]), unsafe_allow_html=True)
st.markdown("")

profile_cols = st.columns(4)
with profile_cols[0]:
    theme.metric_card("Underserved Probability", f"{current_row['predicted_probability']:.1%}",
                       "oxblood" if current_row["predicted_class"] == "Underserved" else "teal")
with profile_cols[1]:
    theme.metric_card("Cases / Lakh Population", f"{current_row['cases_per_lakh_population']:.1f}", "navy")
with profile_cols[2]:
    theme.metric_card("Advice-Enabled Ratio", f"{current_row['advice_enabled_ratio']:.1%}", "gold")
with profile_cols[3]:
    theme.metric_card("YoY Growth Rate", f"{current_row['yoy_growth_rate']:.1%}", "navy")

demo_cols = st.columns(4)
with demo_cols[0]:
    theme.metric_card("Population", f"{current_row['population']:,.0f}", "navy")
with demo_cols[1]:
    theme.metric_card("Rural Population Share", f"{current_row['rural_population_pct']:.1f}%", "teal")
with demo_cols[2]:
    theme.metric_card("Literacy Rate", f"{current_row['literacy_rate_pct']:.1f}%", "teal")
with demo_cols[3]:
    theme.metric_card("Sex Ratio", f"{current_row['sex_ratio']:.0f}", "navy")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Trend across fiscal years
# --------------------------------------------------------------------------- #
st.markdown("#### Historical Trend")
trend_cols = st.columns(2)

with trend_cols[0]:
    fig_cases = px.bar(
        district_history, x="fiscal_year", y="cases_registered",
        text="cases_registered",
    )
    fig_cases.update_traces(marker_color=theme.STEEL_BLUE)
    fig_cases.update_layout(title="Cases Registered by Fiscal Year", xaxis_title="Fiscal Year", yaxis_title="Cases Registered")
    theme.style_plotly_fig(fig_cases, height=360)
    st.plotly_chart(fig_cases, use_container_width=True)

with trend_cols[1]:
    fig_prob = go.Figure(
        go.Scatter(
            x=district_history["fiscal_year"],
            y=district_history["predicted_probability"],
            mode="lines+markers",
            line=dict(color=theme.OXBLOOD, width=3),
            marker=dict(size=10, color=theme.STATUTE_GOLD),
        )
    )
    fig_prob.update_layout(
        title="Predicted Underserved Probability by Fiscal Year",
        xaxis_title="Fiscal Year",
        yaxis_title="Predicted Probability",
        yaxis_tickformat=".0%",
    )
    theme.style_plotly_fig(fig_prob, height=360)
    st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# SHAP explanation for this district-year
# --------------------------------------------------------------------------- #
theme.section_eyebrow("Explainability")
st.markdown("#### Why This Prediction?")

shap_explainer, shap_error = ds.get_shap_explainer(artifacts)
record = ds.build_district_record(current_row, shap_explainer, features_df)

if shap_explainer is None:
    st.info(
        f"SHAP explainability is unavailable in this environment ({shap_error}). "
        "Showing an approximate global feature importance instead.",
        icon="ℹ️",
    )
    fallback_importance = ds.get_fallback_feature_importance(artifacts)
    if fallback_importance is not None:
        fig_fallback = px.bar(
            fallback_importance.sort_values("importance"),
            x="importance", y="feature", orientation="h",
        )
        fig_fallback.update_traces(marker_color=theme.STEEL_BLUE)
        theme.style_plotly_fig(fig_fallback, height=380)
        st.plotly_chart(fig_fallback, use_container_width=True)
elif record["top_contributing_features"]:
    driver_cols = st.columns(len(record["top_contributing_features"][:5]))
    for col, item in zip(driver_cols, record["top_contributing_features"][:5]):
        variant = "oxblood" if "increases" in item["direction"] else "teal"
        with col:
            theme.metric_card(
                item["feature"].replace("_", " ").title(),
                f"{item['shap_value']:+.3f}",
                variant,
                note=item["direction"],
            )
else:
    st.caption("No feature attribution could be computed for this record.")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Policy recommendation
# --------------------------------------------------------------------------- #
theme.section_eyebrow("IBM Granite Narration")
st.markdown("#### Policy Recommendation")

if st.button("Generate Policy Recommendation", type="primary"):
    if not record["top_contributing_features"]:
        st.warning(
            "A policy recommendation requires SHAP feature attribution, "
            "which is unavailable in this environment."
        )
    else:
        client = ds.get_granite_client()
        generators = ds.get_generators(client)
        with st.spinner("Generating policy recommendation..."):
            try:
                result = generators["policy_recommendation"].generate(
                    district_name=selected_district,
                    state_name=selected_state,
                    predicted_class=current_row["predicted_class"],
                    predicted_probability=float(current_row["predicted_probability"]),
                    top_contributing_features=record["top_contributing_features"],
                )
                st.session_state["policy_recommendation_result"] = result
            except JusticeLensError as exc:
                st.error(f"Could not generate a policy recommendation: {exc}")

if "policy_recommendation_result" in st.session_state:
    result = st.session_state["policy_recommendation_result"]
    theme.insight_card(
        f"Recommendation -- {selected_district}, {selected_state}",
        result.narrative_text.replace("
", "<br/>"),
        card_variant="gold",
    )
    theme.provenance_tag(result.is_ai_generated, result.model_id)
