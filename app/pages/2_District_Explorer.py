import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from common import data_service as ds
from common import theme
from justicelens.utils import JusticeLensError

theme.apply_page_config("District Explorer", page_icon="🗺️")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "District Explorer",
    subtitle="Deep-dive into a single district's profile.",
    eyebrow="District-Level Detail",
)

features_df = ds.load_features_dataset()
artifacts = ds.get_training_artifacts(features_df)
predictions_df = ds.get_predictions_df(artifacts, features_df)

st.sidebar.markdown("### Select a District")
all_states = ds.get_unique_states(predictions_df)
selected_state = st.sidebar.selectbox("State / UT", options=all_states)
districts_in_state = ds.get_districts_for_state(predictions_df, selected_state)
selected_district = st.sidebar.selectbox("District", options=districts_in_state)

district_history = predictions_df[
    (predictions_df["state_name"] == selected_state) &
    (predictions_df["district_name"] == selected_district)
].sort_values("fiscal_year")

if district_history.empty:
    st.warning("No records found.")
    st.stop()

all_years = ds.get_unique_fiscal_years(predictions_df)
selected_year = st.sidebar.selectbox("Fiscal Year", options=list(reversed(all_years)))
current_row = district_history[district_history["fiscal_year"] == selected_year]
if current_row.empty:
    current_row = district_history.iloc[[-1]]
current_row = current_row.iloc[0]

st.markdown(f"### {selected_district}, {selected_state} — FY {current_row['fiscal_year']}")
st.markdown(theme.tier_badge_html(current_row["predicted_class"]), unsafe_allow_html=True)
st.markdown("")

profile_cols = st.columns(4)
with profile_cols[0]:
    theme.metric_card("Underserved Probability", f"{current_row['predicted_probability']:.1%}",
                       "oxblood" if current_row["predicted_class"] == "Underserved" else "teal")
with profile_cols[1]:
    theme.metric_card("Cases / Lakh", f"{current_row['cases_per_lakh_population']:.1f}", "navy")
with profile_cols[2]:
    theme.metric_card("Advice Ratio", f"{current_row['advice_enabled_ratio']:.1%}", "gold")
with profile_cols[3]:
    theme.metric_card("YoY Growth", f"{current_row['yoy_growth_rate']:.1%}", "navy")

demo_cols = st.columns(4)
with demo_cols[0]:
    theme.metric_card("Population", f"{current_row['population']:,.0f}", "navy")
with demo_cols[1]:
    theme.metric_card("Rural %", f"{current_row['rural_population_pct']:.1f}%", "teal")
with demo_cols[2]:
    theme.metric_card("Literacy %", f"{current_row['literacy_rate_pct']:.1f}%", "teal")
with demo_cols[3]:
    theme.metric_card("Sex Ratio", f"{current_row['sex_ratio']:.0f}", "navy")

st.markdown("---")
st.markdown("#### Historical Trend")
trend_cols = st.columns(2)
with trend_cols[0]:
    fig_cases = px.bar(district_history, x="fiscal_year", y="cases_registered", text="cases_registered")
    fig_cases.update_traces(marker_color=theme.STEEL_BLUE)
    fig_cases.update_layout(title="Cases Registered", xaxis_title="Fiscal Year", yaxis_title="Cases")
    theme.style_plotly_fig(fig_cases, height=360)
    st.plotly_chart(fig_cases, use_container_width=True)
with trend_cols[1]:
    fig_prob = go.Figure(go.Scatter(
        x=district_history["fiscal_year"],
        y=district_history["predicted_probability"],
        mode="lines+markers",
        line=dict(color=theme.OXBLOOD, width=3),
        marker=dict(size=10, color=theme.STATUTE_GOLD),
    ))
    fig_prob.update_layout(
        title="Predicted Underserved Probability",
        xaxis_title="Fiscal Year",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
    )
    theme.style_plotly_fig(fig_prob, height=360)
    st.plotly_chart(fig_prob, use_container_width=True)

st.markdown("---")
theme.section_eyebrow("Explainability")
st.markdown("#### Why This Prediction?")
shap_explainer, shap_error = ds.get_shap_explainer(artifacts)
record = ds.build_district_record(current_row, shap_explainer, features_df)

if shap_explainer is None:
    st.info(f"SHAP unavailable ({shap_error}). Showing approximate importance.", icon="ℹ️")
    fallback_importance = ds.get_fallback_feature_importance(artifacts)
    if fallback_importance is not None:
        fig_fallback = px.bar(fallback_importance.sort_values("importance"), x="importance", y="feature", orientation="h")
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
    st.caption("No feature attribution computed.")

st.markdown("---")
theme.section_eyebrow("IBM Granite Narration")
st.markdown("#### Policy Recommendation")
if st.button("Generate Policy Recommendation", type="primary"):
    if not record["top_contributing_features"]:
        st.warning("SHAP attribution required for recommendations.")
    else:
        client = ds.get_granite_client()
        generators = ds.get_generators(client)
        with st.spinner("Generating..."):
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
                st.error(f"Error: {exc}")

if "policy_recommendation_result" in st.session_state:
    result = st.session_state["policy_recommendation_result"]
    theme.insight_card(
        f"Recommendation — {selected_district}, {selected_state}",
        result.narrative_text.replace("\n", "<br/>"),
        card_variant="gold",
    )
    theme.provenance_tag(result.is_ai_generated, result.model_id)