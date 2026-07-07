"""
1_📊_Executive_Dashboard.py
============================
Executive Dashboard: filterable, national/state-level KPIs and charts, plus
an on-demand IBM Granite-narrated executive summary.
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

theme.apply_page_config("Executive Dashboard", page_icon="📊")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "Executive Dashboard",
    subtitle="Filter, monitor, and summarize legal-access disparity across India.",
    eyebrow="National Overview",
)

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
features_df = ds.load_features_dataset()
artifacts = ds.get_training_artifacts(features_df)
predictions_df = ds.get_predictions_df(artifacts, features_df)

# --------------------------------------------------------------------------- #
# Filters
# --------------------------------------------------------------------------- #
st.sidebar.markdown("### Filters")
all_states = ds.get_unique_states(predictions_df)
all_years = ds.get_unique_fiscal_years(predictions_df)

selected_states = st.sidebar.multiselect("State / UT", options=all_states, default=[])
selected_years = st.sidebar.multiselect("Fiscal Year", options=all_years, default=all_years)
selected_tiers = st.sidebar.multiselect(
    "Predicted Status",
    options=sorted(predictions_df["predicted_class"].unique().tolist()),
    default=sorted(predictions_df["predicted_class"].unique().tolist()),
)

filtered_df = predictions_df.copy()
if selected_states:
    filtered_df = filtered_df[filtered_df["state_name"].isin(selected_states)]
if selected_years:
    filtered_df = filtered_df[filtered_df["fiscal_year"].isin(selected_years)]
if selected_tiers:
    filtered_df = filtered_df[filtered_df["predicted_class"].isin(selected_tiers)]

scope_label = ", ".join(selected_states) if selected_states else "All India"
if len(selected_years) < len(all_years) and selected_years:
    scope_label += f" ({', '.join(selected_years)})"

if filtered_df.empty:
    st.warning("No records match the current filters. Adjust the filters in the sidebar.")
    st.stop()

# --------------------------------------------------------------------------- #
# KPI row
# --------------------------------------------------------------------------- #
theme.section_eyebrow(f"Scope: {scope_label}")
kpi_cols = st.columns(4)
total = len(filtered_df)
underserved = int((filtered_df["predicted_class"] == "Underserved").sum())
share = underserved / max(total, 1)

with kpi_cols[0]:
    theme.metric_card("Records in Scope", f"{total:,}", "navy")
with kpi_cols[1]:
    theme.metric_card("Underserved Share", f"{share:.1%}", "oxblood", note=f"{underserved:,} records")
with kpi_cols[2]:
    theme.metric_card("Avg. Cases / Lakh Population", f"{filtered_df['cases_per_lakh_population'].mean():.1f}", "teal")
with kpi_cols[3]:
    theme.metric_card("Avg. Advice-Enabled Ratio", f"{filtered_df['advice_enabled_ratio'].mean():.1%}", "gold")

st.markdown("---")

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
chart_col_left, chart_col_right = st.columns([3, 2])

with chart_col_left:
    st.markdown("#### Most Underserved Districts")
    top_n = st.slider("Number of districts to show", min_value=5, max_value=30, value=15, step=5)
    worst = (
        filtered_df.sort_values("predicted_probability", ascending=False)
        .drop_duplicates(subset=["district_name", "state_name"])
        .head(top_n)
    )
    worst_label = worst["district_name"] + ", " + worst["state_name"]
    fig_bar = go.Figure(
        go.Bar(
            x=worst["predicted_probability"],
            y=worst_label,
            orientation="h",
            marker_color=[
                theme.TIER_COLORS.get(c, theme.MUTED_TEXT) for c in worst["predicted_class"]
            ],
            text=[f"{v:.1%}" for v in worst["predicted_probability"]],
            textposition="outside",
        )
    )
    fig_bar.update_layout(
        title="Predicted Underserved Probability",
        xaxis_title="Predicted Probability",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
    )
    theme.style_plotly_fig(fig_bar, height=max(400, top_n * 28))
    st.plotly_chart(fig_bar, use_container_width=True)

with chart_col_right:
    st.markdown("#### Status Composition")
    status_counts = filtered_df["predicted_class"].value_counts().reset_index()
    status_counts.columns = ["predicted_class", "count"]
    fig_pie = px.pie(
        status_counts,
        names="predicted_class",
        values="count",
        color="predicted_class",
        color_discrete_map=theme.TIER_COLORS,
        hole=0.55,
    )
    fig_pie.update_traces(textinfo="percent+label")
    theme.style_plotly_fig(fig_pie, height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("#### State-Level Disparity Map (Treemap)")
state_agg = (
    filtered_df.groupby("state_name")
    .agg(
        avg_probability=("predicted_probability", "mean"),
        total_population=("population", "sum"),
        district_count=("district_name", "nunique"),
    )
    .reset_index()
)
fig_treemap = px.treemap(
    state_agg,
    path=[px.Constant("All India"), "state_name"],
    values="total_population",
    color="avg_probability",
    color_continuous_scale=theme.DISPARITY_SCALE,
    hover_data={"district_count": True, "avg_probability": ":.1%"},
)
fig_treemap.update_layout(coloraxis_colorbar_title="Avg. Underserved Probability")
theme.style_plotly_fig(fig_treemap, height=480)
st.plotly_chart(fig_treemap, use_container_width=True)

st.markdown("#### Trend Across Fiscal Years")
trend_df = (
    filtered_df.groupby("fiscal_year")["predicted_probability"]
    .mean()
    .reindex(ds.get_unique_fiscal_years(filtered_df))
    .reset_index()
)
fig_trend = px.line(
    trend_df, x="fiscal_year", y="predicted_probability", markers=True,
)
fig_trend.update_traces(line_color=theme.STEEL_BLUE, marker=dict(size=10, color=theme.STATUTE_GOLD))
fig_trend.update_layout(yaxis_tickformat=".0%", yaxis_title="Avg. Predicted Underserved Probability", xaxis_title="Fiscal Year")
theme.style_plotly_fig(fig_trend, height=360)
st.plotly_chart(fig_trend, use_container_width=True)

st.markdown("---")

# --------------------------------------------------------------------------- #
# AI Executive Summary + Download
# --------------------------------------------------------------------------- #
theme.section_eyebrow("IBM Granite Narration")
st.markdown("#### Executive Summary")

action_cols = st.columns([1, 1, 2])
with action_cols[0]:
    generate_clicked = st.button("Generate Executive Summary", type="primary")
with action_cols[1]:
    st.download_button(
        "Download Filtered Data (CSV)",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="justicelens_filtered_data.csv",
        mime="text/csv",
    )

if generate_clicked:
    client = ds.get_granite_client()
    generators = ds.get_generators(client)
    shap_explainer, _shap_error = ds.get_shap_explainer(artifacts)

    if shap_explainer is not None:
        importance_df = shap_explainer.compute_feature_importance(artifacts.X_test)
    else:
        fallback_importance = ds.get_fallback_feature_importance(artifacts)
        importance_df = (
            fallback_importance.rename(columns={"importance": "mean_abs_shap_value"})
            if fallback_importance is not None
            else pd.DataFrame(columns=["feature", "mean_abs_shap_value"])
        )

    with st.spinner("Generating executive summary..."):
        try:
            result = generators["executive_summary"].generate(
                scope_label=scope_label,
                predictions_df=filtered_df,
                global_top_drivers=importance_df,
            )
            st.session_state["executive_summary_result"] = result
        except JusticeLensError as exc:
            st.error(f"Could not generate executive summary: {exc}")

if "executive_summary_result" in st.session_state:
    result = st.session_state["executive_summary_result"]
    theme.insight_card("Executive Summary", result.narrative_text.replace("\n", "<br/>"), card_variant="gold")
    theme.provenance_tag(result.is_ai_generated, result.model_id)
