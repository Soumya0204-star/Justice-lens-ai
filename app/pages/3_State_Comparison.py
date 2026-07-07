"""
3_⚖️_State_Comparison.py
==========================
State Comparison: compare disparity profiles across two or more states,
plus an IBM Granite-narrated comparison of each state's most underserved
district.
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

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import data_service as ds
from common import theme
from justicelens.utils import JusticeLensError

theme.apply_page_config("State Comparison", page_icon="⚖️")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "State Comparison",
    subtitle="Compare Tele-Law disparity profiles across two or more states/UTs.",
    eyebrow="Cross-State Analysis",
)

features_df = ds.load_features_dataset()
artifacts = ds.get_training_artifacts(features_df)
predictions_df = ds.get_predictions_df(artifacts, features_df)

# --------------------------------------------------------------------------- #
# Selection
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### Select States")
all_states = ds.get_unique_states(predictions_df)
default_states = all_states[:2] if len(all_states) >= 2 else all_states
selected_states = st.sidebar.multiselect(
    "States / UTs to compare (2 or more)", options=all_states, default=default_states
)
all_years = ds.get_unique_fiscal_years(predictions_df)
selected_year = st.sidebar.selectbox("Fiscal Year", options=list(reversed(all_years)))

if len(selected_states) < 2:
    st.info("Select at least 2 states/UTs in the sidebar to compare.")
    st.stop()

scoped_df = predictions_df[
    predictions_df["state_name"].isin(selected_states)
    & (predictions_df["fiscal_year"] == selected_year)
]

if scoped_df.empty:
    st.warning("No records match the current selection.")
    st.stop()

# --------------------------------------------------------------------------- #
# Comparison table + charts
# --------------------------------------------------------------------------- #
state_summary = (
    scoped_df.groupby("state_name")
    .agg(
        districts=("district_name", "nunique"),
        avg_underserved_probability=("predicted_probability", "mean"),
        underserved_share=("predicted_class", lambda s: (s == "Underserved").mean()),
        avg_cases_per_lakh=("cases_per_lakh_population", "mean"),
        avg_advice_ratio=("advice_enabled_ratio", "mean"),
        avg_rural_pct=("rural_population_pct", "mean"),
        avg_literacy=("literacy_rate_pct", "mean"),
    )
    .reset_index()
    .sort_values("avg_underserved_probability", ascending=False)
)

theme.section_eyebrow(f"Fiscal Year {selected_year}")
st.markdown("#### Summary Table")
display_table = state_summary.copy()
display_table["avg_underserved_probability"] = display_table["avg_underserved_probability"].map("{:.1%}".format)
display_table["underserved_share"] = display_table["underserved_share"].map("{:.1%}".format)
display_table["avg_cases_per_lakh"] = display_table["avg_cases_per_lakh"].map("{:.1f}".format)
display_table["avg_advice_ratio"] = display_table["avg_advice_ratio"].map("{:.1%}".format)
display_table["avg_rural_pct"] = display_table["avg_rural_pct"].map("{:.1f}%".format)
display_table["avg_literacy"] = display_table["avg_literacy"].map("{:.1f}%".format)
st.dataframe(display_table, use_container_width=True, hide_index=True)

chart_cols = st.columns(2)
with chart_cols[0]:
    fig_bar = px.bar(
        state_summary, x="state_name", y="avg_underserved_probability",
        color="state_name", color_discrete_sequence=theme.CATEGORICAL_SEQUENCE,
        text=state_summary["avg_underserved_probability"].map("{:.1%}".format),
    )
    fig_bar.update_layout(
        title="Average Underserved Probability", yaxis_tickformat=".0%",
        yaxis_title="Avg. Predicted Probability", xaxis_title="", showlegend=False,
    )
    theme.style_plotly_fig(fig_bar, height=400)
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_cols[1]:
    radar_metrics = ["avg_underserved_probability", "underserved_share", "avg_advice_ratio", "avg_rural_pct", "avg_literacy"]
    radar_labels = ["Underserved Prob.", "Underserved Share", "Advice Ratio", "Rural %", "Literacy %"]
    fig_radar = go.Figure()
    for _, row in state_summary.iterrows():
        # Normalize each metric to 0-1 across the selected states so the
        # radar is comparable despite different natural scales (%, ratio).
        values = []
        for metric in radar_metrics:
            col_values = state_summary[metric]
            span = (col_values.max() - col_values.min()) or 1.0
            values.append((row[metric] - col_values.min()) / span)
        fig_radar.add_trace(
            go.Scatterpolar(r=values + [values[0]], theta=radar_labels + [radar_labels[0]], fill="toself", name=row["state_name"])
        )
    fig_radar.update_layout(title="Relative Disparity Profile (normalized)", polar=dict(radialaxis=dict(visible=True, range=[0, 1])))
    theme.style_plotly_fig(fig_radar, height=400)
    st.plotly_chart(fig_radar, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# AI-narrated comparison of each state's worst district
# --------------------------------------------------------------------------- #
theme.section_eyebrow("IBM Granite Narration")
st.markdown("#### Narrated Comparison")
st.caption(
    "Compares the single most underserved district within each selected "
    "state, for a concrete, district-level narrative rather than an "
    "abstract state-average comparison."
)

if st.button("Generate Narrated Comparison", type="primary"):
    shap_explainer, _shap_error = ds.get_shap_explainer(artifacts)
    worst_per_state = (
        scoped_df.sort_values("predicted_probability", ascending=False)
        .drop_duplicates(subset="state_name")
    )
    district_records = [
        ds.build_district_record(row, shap_explainer, features_df)
        for _, row in worst_per_state.iterrows()
    ]

    client = ds.get_granite_client()
    generators = ds.get_generators(client)
    with st.spinner("Generating comparison..."):
        try:
            result = generators["district_comparison"].compare(district_records)
            st.session_state["state_comparison_result"] = result
        except JusticeLensError as exc:
            st.error(f"Could not generate a comparison: {exc}")

if "state_comparison_result" in st.session_state:
    result = st.session_state["state_comparison_result"]
    theme.insight_card("Comparison", result.narrative_text.replace("\n", "<br/>"), card_variant="gold")
    theme.provenance_tag(result.is_ai_generated, result.model_id)

st.download_button(
    "Download Comparison Data (CSV)",
    data=state_summary.to_csv(index=False).encode("utf-8"),
    file_name="justicelens_state_comparison.csv",
    mime="text/csv",
)
