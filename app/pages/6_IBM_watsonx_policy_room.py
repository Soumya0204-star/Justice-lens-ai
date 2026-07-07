"""
6_🏛️_IBM_watsonx_Policy_Room.py
==================================
IBM watsonx Policy Room: the generative-AI control room. Executive
summaries, policy recommendations, the full downloadable AI report, and
the "Ask JusticeLens" natural-language chat -- all powered by IBM Granite
via IBM watsonx.ai, with deterministic template fallback when watsonx.ai
is unavailable.
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
import streamlit as st

from common import data_service as ds
from common import theme
from justicelens.utils import JusticeLensError

theme.apply_page_config("IBM watsonx Policy Room", page_icon="🏛️")
theme.inject_global_css()
theme.render_sidebar_brand()

theme.render_app_header(
    "IBM watsonx Policy Room",
    subtitle="AI-generated summaries, recommendations, reports, and conversational Q&A -- powered by IBM Granite.",
    eyebrow="Generative AI Control Room",
)

features_df = ds.load_features_dataset()
artifacts = ds.get_training_artifacts(features_df)
predictions_df = ds.get_predictions_df(artifacts, features_df)
shap_explainer, shap_error = ds.get_shap_explainer(artifacts)
client = ds.get_granite_client()
generators = ds.get_generators(client)

if client.is_available():
    st.success(f"Connected to IBM watsonx.ai -- model: `{client._config.model_id}`", icon="🤖")
else:
    st.warning(
        "IBM watsonx.ai is not configured. Every feature below still "
        "works, using a deterministic template fallback -- set "
        "`WATSONX_API_KEY` / `WATSONX_PROJECT_ID` in your `.env` file to "
        "enable live Granite narration.",
        icon="📋",
    )

if shap_explainer is not None:
    global_importance = shap_explainer.compute_feature_importance(artifacts.X_test)
else:
    fallback_importance = ds.get_fallback_feature_importance(artifacts)
    global_importance = (
        fallback_importance.rename(columns={"importance": "mean_abs_shap_value"})
        if fallback_importance is not None
        else pd.DataFrame(columns=["feature", "mean_abs_shap_value"])
    )

tab_summary, tab_policy, tab_report, tab_chat = st.tabs(
    ["📝 Executive Summary", "🎯 Policy Recommendations", "📄 Full AI Report", "💬 Ask JusticeLens"]
)

# --------------------------------------------------------------------------- #
# Executive Summary tab
# --------------------------------------------------------------------------- #
with tab_summary:
    st.markdown("#### Generate an Executive Summary")
    scope_options = ["All India"] + ds.get_unique_states(predictions_df)
    scope_choice = st.selectbox("Scope", options=scope_options, key="summary_scope")

    scoped_df = (
        predictions_df if scope_choice == "All India"
        else predictions_df[predictions_df["state_name"] == scope_choice]
    )

    if st.button("Generate Executive Summary", type="primary", key="btn_summary"):
        with st.spinner("Generating executive summary via IBM Granite..."):
            try:
                result = generators["executive_summary"].generate(
                    scope_label=scope_choice, predictions_df=scoped_df, global_top_drivers=global_importance,
                )
                st.session_state["policy_room_summary"] = result
            except JusticeLensError as exc:
                st.error(f"Could not generate summary: {exc}")

    if "policy_room_summary" in st.session_state:
        result = st.session_state["policy_room_summary"]
        theme.insight_card("Executive Summary", result.narrative_text.replace("\n", "<br/>"), "gold")
        theme.provenance_tag(result.is_ai_generated, result.model_id)

# --------------------------------------------------------------------------- #
# Policy Recommendations tab
# --------------------------------------------------------------------------- #
with tab_policy:
    st.markdown("#### Generate a Policy Recommendation")
    rec_cols = st.columns(2)
    with rec_cols[0]:
        rec_state = st.selectbox("State / UT", options=ds.get_unique_states(predictions_df), key="rec_state")
    with rec_cols[1]:
        rec_district = st.selectbox(
            "District", options=ds.get_districts_for_state(predictions_df, rec_state), key="rec_district"
        )

    district_rows = predictions_df[
        (predictions_df["state_name"] == rec_state) & (predictions_df["district_name"] == rec_district)
    ].sort_values("fiscal_year")

    if not district_rows.empty:
        latest_row = district_rows.iloc[-1]
        st.caption(f"Using most recent record: FY {latest_row['fiscal_year']}")

        if st.button("Generate Policy Recommendation", type="primary", key="btn_policy"):
            record = ds.build_district_record(latest_row, shap_explainer, features_df)
            if not record["top_contributing_features"]:
                st.warning("SHAP feature attribution is unavailable, so a targeted recommendation cannot be generated for this district.")
            else:
                with st.spinner("Generating policy recommendation via IBM Granite..."):
                    try:
                        result = generators["policy_recommendation"].generate(
                            district_name=rec_district, state_name=rec_state,
                            predicted_class=latest_row["predicted_class"],
                            predicted_probability=float(latest_row["predicted_probability"]),
                            top_contributing_features=record["top_contributing_features"],
                        )
                        st.session_state["policy_room_recommendation"] = result
                    except JusticeLensError as exc:
                        st.error(f"Could not generate recommendation: {exc}")

    if "policy_room_recommendation" in st.session_state:
        result = st.session_state["policy_room_recommendation"]
        theme.insight_card(f"Recommendation -- {rec_district}, {rec_state}", result.narrative_text.replace("\n", "<br/>"), "gold")
        theme.provenance_tag(result.is_ai_generated, result.model_id)

# --------------------------------------------------------------------------- #
# Full AI Report tab
# --------------------------------------------------------------------------- #
with tab_report:
    st.markdown("#### Generate the Full AI Report")
    st.caption(
        "Assembles an executive summary, model performance summary, a "
        "comparison of the worst-affected district in up to 3 states, "
        "and prioritized recommendations into a single downloadable "
        "Markdown report."
    )

    report_states = st.multiselect(
        "States to include in the district comparison/recommendations (2-3 recommended)",
        options=ds.get_unique_states(predictions_df),
        default=ds.get_unique_states(predictions_df)[:2],
        key="report_states",
    )

    if st.button("Generate Full AI Report", type="primary", key="btn_report"):
        district_records = None
        if len(report_states) >= 2:
            worst_per_state = (
                predictions_df[predictions_df["state_name"].isin(report_states)]
                .sort_values("predicted_probability", ascending=False)
                .drop_duplicates(subset="state_name")
            )
            candidate_records = [
                ds.build_district_record(row, shap_explainer, features_df)
                for _, row in worst_per_state.iterrows()
            ]
            district_records = [r for r in candidate_records if r["top_contributing_features"]]
            if len(district_records) < 2:
                district_records = None

        with st.spinner("Assembling the full AI report via IBM Granite..."):
            try:
                report_result = generators["ai_report"].generate_full_report(
                    scope_label="All India -- FY 2021-22 to FY 2024-25",
                    predictions_df=predictions_df,
                    global_top_drivers=global_importance,
                    comparison_table=artifacts.comparison_table,
                    best_model_name=artifacts.best_result.model_name,
                    district_records_with_explanations=district_records,
                )
                st.session_state["policy_room_report"] = report_result
            except JusticeLensError as exc:
                st.error(f"Could not generate the full report: {exc}")

    if "policy_room_report" in st.session_state:
        report_result = st.session_state["policy_room_report"]
        st.success(
            f"Report generated: {report_result.ai_generated_section_count}/"
            f"{report_result.total_section_count} sections AI-narrated by "
            f"Granite (remainder used the deterministic template)."
        )
        report_text = Path(report_result.report_path).read_text(encoding="utf-8")
        with st.expander("Preview Report", expanded=True):
            st.markdown(report_text)
        st.download_button(
            "Download Full Report (Markdown)",
            data=report_text.encode("utf-8"),
            file_name=Path(report_result.report_path).name,
            mime="text/markdown",
            type="primary",
        )

# --------------------------------------------------------------------------- #
# Ask JusticeLens chat tab
# --------------------------------------------------------------------------- #
with tab_chat:
    st.markdown("#### Ask JusticeLens")
    st.caption(
        "Ask a natural-language question about Tele-Law disparity. "
        "Answers are grounded strictly in the underlying model's "
        "predictions -- mention a specific state or district for a "
        "targeted answer, or ask an open-ended question for a national view."
    )

    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    for role, content, meta in st.session_state["chat_history"]:
        with st.chat_message(role):
            st.markdown(content)
            if meta is not None:
                theme.provenance_tag(meta["is_ai_generated"], meta["model_id"])

    question = st.chat_input("e.g. Which districts in Bihar need urgent CSC intervention?")
    if question:
        st.session_state["chat_history"].append(("user", question, None))
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    result = generators["qa"].answer(question, predictions_df)
                    st.markdown(result.narrative_text)
                    theme.provenance_tag(result.is_ai_generated, result.model_id)
                    st.session_state["chat_history"].append(
                        ("assistant", result.narrative_text, {"is_ai_generated": result.is_ai_generated, "model_id": result.model_id})
                    )
                except JusticeLensError as exc:
                    error_text = f"I couldn't answer that question: {exc}"
                    st.error(error_text)
                    st.session_state["chat_history"].append(("assistant", error_text, None))

    if st.session_state["chat_history"]:
        if st.button("Clear Chat History"):
            st.session_state["chat_history"] = []
            st.rerun()
