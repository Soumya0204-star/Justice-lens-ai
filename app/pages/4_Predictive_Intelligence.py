import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
APP_DIR = Path(__file__).resolve().parent.parent
for p in [str(PROJECT_ROOT), str(APP_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from common import data_service as ds
from common import theme
from justicelens import config

theme.apply_page_config("Predictive Intelligence", page_icon="🔮")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "Predictive Intelligence",
    subtitle="Score a hypothetical district using the best model.",
    eyebrow="Model & Prediction Interface",
)

features_df = ds.load_features_dataset()
artifacts = ds.get_training_artifacts(features_df)

theme.section_eyebrow("Model Comparison")
st.markdown("#### Candidate Model Leaderboard")
comparison = artifacts.comparison_table.copy()
st.dataframe(comparison, use_container_width=True, hide_index=True)

metric_col = "test_roc_auc" if "test_roc_auc" in comparison.columns else "cv_mean_score"
fig_leaderboard = px.bar(
    comparison.sort_values(metric_col),
    x=metric_col, y="model_name", orientation="h",
    text=comparison.sort_values(metric_col)[metric_col].map("{:.4f}".format),
)
fig_leaderboard.update_traces(marker_color=theme.STEEL_BLUE)
fig_leaderboard.update_layout(title="Test ROC-AUC", xaxis_title="ROC-AUC", yaxis_title="")
theme.style_plotly_fig(fig_leaderboard, height=320)
st.plotly_chart(fig_leaderboard, use_container_width=True)

st.success(f"Best model: **{artifacts.best_result.model_name}** (CV ROC-AUC = {artifacts.best_result.cv_mean_score:.4f})")
st.markdown("---")

theme.section_eyebrow("Prediction Interface")
st.markdown("#### Score a Hypothetical District")
st.caption("Adjust sliders to describe a hypothetical district and see the prediction.")

feature_stats = features_df[artifacts.feature_names].describe()
input_values = {}
input_cols = st.columns(2)
for i, feature in enumerate(artifacts.feature_names):
    col = input_cols[i % 2]
    feature_min = float(feature_stats.loc["min", feature])
    feature_max = float(feature_stats.loc["max", feature])
    feature_mean = float(feature_stats.loc["50%", feature])
    label = feature.replace("_", " ").title()
    with col:
        if feature_max - feature_min > 1000:
            input_values[feature] = st.number_input(
                label, min_value=0.0, value=round(feature_mean, 2), step=max(feature_max / 100, 1.0),
                key=f"input_{feature}",
            )
        else:
            input_values[feature] = st.slider(
                label, min_value=round(feature_min, 2), max_value=round(feature_max, 2),
                value=round(feature_mean, 2), key=f"input_{feature}",
            )

if st.button("Run Prediction", type="primary"):
    input_df = pd.DataFrame([input_values])[artifacts.feature_names]
    best_pipeline = artifacts.best_result.pipeline
    prob = float(best_pipeline.predict_proba(input_df)[0, 1])
    cls_idx = int(best_pipeline.predict(input_df)[0])
    cls_name = config.ML_CLASS_NAMES[cls_idx]

    result_cols = st.columns([1, 2])
    with result_cols[0]:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%"},
            title={"text": "Underserved Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": theme.OXBLOOD if cls_name == "Underserved" else theme.VERDICT_TEAL},
                "steps": [
                    {"range": [0, 25], "color": "#E4F0EC"},
                    {"range": [25, 50], "color": "#F4EDD8"},
                    {"range": [50, 75], "color": "#F3DCC9"},
                    {"range": [75, 100], "color": "#F0D3D3"},
                ],
            },
        ))
        theme.style_plotly_fig(fig_gauge, height=320)
        st.plotly_chart(fig_gauge, use_container_width=True)
    with result_cols[1]:
        st.markdown(theme.tier_badge_html(cls_name), unsafe_allow_html=True)
        theme.insight_card(
            "Prediction Result",
            f"This hypothetical district is predicted as **{cls_name}** with a {prob:.1%} probability.",
            card_variant="oxblood" if cls_name == "Underserved" else "teal",
        )

    shap_explainer, shap_error = ds.get_shap_explainer(artifacts)
    if shap_explainer is not None:
        st.markdown("##### Why This Prediction?")
        try:
            explanation = shap_explainer.explain_prediction(input_df, instance_index=0)
            driver_cols = st.columns(len(explanation["top_contributing_features"][:5]))
            for col, item in zip(driver_cols, explanation["top_contributing_features"][:5]):
                variant = "oxblood" if "increases" in item["direction"] else "teal"
                with col:
                    theme.metric_card(
                        item["feature"].replace("_", " ").title(),
                        f"{item['shap_value']:+.3f}",
                        variant,
                        note=item["direction"],
                    )
        except Exception as e:
            st.caption(f"SHAP unavailable: {e}")
    else:
        st.caption(f"SHAP unavailable ({shap_error}).")