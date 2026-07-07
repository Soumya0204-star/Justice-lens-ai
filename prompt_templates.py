"""
prompt_templates.py
=====================
Prompt construction layer for JusticeLens AI's IBM Granite (watsonx.ai)
integration.

Every function in this module builds a complete prompt string from
**precomputed structured data only** (LADI scores, SHAP driver lists,
model metrics, district/demographic figures) -- never from free-text the
model is asked to reason about numerically. This encodes the system
architecture's core generative-AI safety principle:

    Granite narrates; it never calculates.

Every prompt explicitly instructs the model to use only the figures it is
given, not to invent additional statistics, and to flag uncertainty rather
than fabricate specifics. This keeps every generated artifact auditable:
if a number appears in the output, it also appears verbatim in the prompt,
which is logged alongside every generation call.

All templates are plain Python f-string builders (no external templating
engine dependency) that accept already-validated, already-rounded data
structures produced by ``model_training.py``, ``model_evaluation.py``,
and ``shap_explainability.py``.
"""

from __future__ import annotations

from typing import Dict, List, Optional


def _format_top_features(top_features: List[Dict[str, object]]) -> str:
    """Render a SHAP top-contributing-features list as a compact bullet
    list suitable for embedding in a prompt.

    Args:
        top_features: List of dicts as produced by
            ``shap_explainability.SHAPExplainer.explain_prediction``
            (keys: ``feature``, ``value``, ``shap_value``, ``direction``).

    Returns:
        A newline-joined bullet-point string.
    """
    lines = []
    for item in top_features:
        lines.append(
            f"- {item['feature'].replace('_', ' ')}: value={item['value']}, "
            f"SHAP contribution={item['shap_value']} ({item['direction']})"
        )
    return "\n".join(lines) if lines else "- (no feature attribution available)"


def _format_metrics_table(metrics: Dict[str, float]) -> str:
    """Render a flat metrics dict as a compact ``key: value`` bullet list.

    Args:
        metrics: Dict of metric name -> numeric value.

    Returns:
        A newline-joined bullet-point string.
    """
    return "\n".join(f"- {k.replace('_', ' ')}: {v}" for k, v in metrics.items())


#: Shared system-style preamble prepended to every prompt, establishing
#: the model's role and the hard grounding constraint.
_SYSTEM_PREAMBLE = (
    "You are a policy analyst assistant supporting India's Department of "
    "Justice Tele-Law program. You write clear, neutral, decision-useful "
    "prose for government officials. You MUST use only the numbers and "
    "facts provided below -- never invent statistics, district names, or "
    "figures that are not explicitly given. If information needed to "
    "answer fully is not provided, say so plainly rather than guessing. "
    "Write in plain English, avoid jargon, and keep a neutral, factual "
    "tone appropriate for an official government report."
)


def build_executive_summary_prompt(
    scope_label: str,
    total_districts: int,
    underserved_district_count: int,
    average_ladi_or_probability: float,
    top_underserved_districts: List[Dict[str, object]],
    global_top_drivers: List[Dict[str, object]],
) -> str:
    """Build the prompt for the Executive Summary Generator.

    Args:
        scope_label: What the summary covers, e.g. "All India",
            "Bihar", or "FY 2023-24".
        total_districts: Total number of district-year records in scope.
        underserved_district_count: Count flagged "Underserved" by the
            model within scope.
        average_ladi_or_probability: Mean predicted underserved
            probability (or LADI score) across the scope, 0-1 or a
            comparable scale.
        top_underserved_districts: List of dicts, each with at least
            ``district_name``, ``state_name``, and
            ``predicted_probability``, sorted worst-first.
        global_top_drivers: List of dicts as produced by
            ``shap_explainability.SHAPExplainer.compute_feature_importance``
            (columns ``feature``, ``mean_abs_shap_value``), sorted
            descending.

    Returns:
        A complete prompt string ready to pass to
        ``watsonx_integration.GraniteClient.generate``.
    """
    district_lines = "\n".join(
        f"- {d['district_name']}, {d['state_name']}: "
        f"{d['predicted_probability']:.1%} predicted underserved probability"
        for d in top_underserved_districts
    ) or "- (none)"

    driver_lines = "\n".join(
        f"- {row['feature'].replace('_', ' ')} "
        f"(mean impact score={round(row['mean_abs_shap_value'], 4)})"
        for row in global_top_drivers
    ) or "- (none)"

    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"TASK: Write a 3-4 paragraph executive summary of Tele-Law legal "
        f"-access disparity for the scope: {scope_label}.\n\n"
        f"DATA:\n"
        f"- Total district-year records analyzed: {total_districts}\n"
        f"- Records flagged 'Underserved' by the model: "
        f"{underserved_district_count} "
        f"({underserved_district_count / max(total_districts, 1):.1%})\n"
        f"- Average predicted underserved probability across scope: "
        f"{average_ladi_or_probability:.1%}\n\n"
        f"Top underserved districts in scope:\n{district_lines}\n\n"
        f"Top drivers of underserved risk across the model (global feature "
        f"importance):\n{driver_lines}\n\n"
        f"INSTRUCTIONS: Open with the headline finding (how widespread "
        f"disparity is in this scope). Name the specific worst-affected "
        f"districts from the list above. Explain, in plain terms, which "
        f"factors are driving disparity, using only the drivers listed. "
        f"Close with what this implies for CSC/paralegal resource "
        f"allocation, without prescribing specific interventions (that is "
        f"a separate report section)."
    )


def build_district_comparison_prompt(
    district_records: List[Dict[str, object]],
) -> str:
    """Build the prompt for the District Comparison generator.

    Args:
        district_records: List of 2 or more dicts, each describing one
            district with at least: ``district_name``, ``state_name``,
            ``fiscal_year``, ``cases_per_lakh_population``,
            ``advice_enabled_ratio``, ``predicted_probability``,
            ``predicted_class``, and ``top_contributing_features`` (a
            list of dicts as produced by
            ``shap_explainability.SHAPExplainer.explain_prediction``).

    Returns:
        A complete prompt string.

    Raises:
        ValueError: If fewer than 2 district records are provided.
    """
    if len(district_records) < 2:
        raise ValueError(
            "District comparison requires at least 2 district records, "
            f"got {len(district_records)}."
        )

    blocks = []
    for record in district_records:
        feature_lines = _format_top_features(
            record.get("top_contributing_features", [])
        )
        blocks.append(
            f"District: {record['district_name']}, {record['state_name']} "
            f"(FY {record['fiscal_year']})\n"
            f"- Predicted status: {record['predicted_class']} "
            f"({record['predicted_probability']:.1%} underserved "
            f"probability)\n"
            f"- Cases per lakh population: "
            f"{record['cases_per_lakh_population']}\n"
            f"- Advice-enabled ratio: {record['advice_enabled_ratio']:.1%}\n"
            f"- Top drivers:\n{feature_lines}"
        )

    districts_block = "\n\n".join(blocks)
    district_names = ", ".join(r["district_name"] for r in district_records)

    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"TASK: Write a structured comparison of the following districts: "
        f"{district_names}. Use only the data given below.\n\n"
        f"DATA:\n{districts_block}\n\n"
        f"INSTRUCTIONS: For each pair of districts, highlight the most "
        f"important differences in registration rate, advice-enabled "
        f"ratio, and underserved risk. Explain which specific factors "
        f"(from the top drivers listed) most explain the gap between "
        f"them. End with a one-sentence ranking of these districts from "
        f"most to least in need of intervention, based solely on the "
        f"predicted probabilities given."
    )


def build_policy_recommendation_prompt(
    district_name: str,
    state_name: str,
    predicted_class: str,
    predicted_probability: float,
    top_contributing_features: List[Dict[str, object]],
    candidate_interventions: List[Dict[str, str]],
) -> str:
    """Build the prompt for the Policy Recommendation Engine.

    Args:
        district_name: District being assessed.
        state_name: Parent state/UT.
        predicted_class: Model's predicted label (e.g. "Underserved").
        predicted_probability: Predicted underserved probability.
        top_contributing_features: SHAP-derived top drivers, as produced
            by ``shap_explainability.SHAPExplainer.explain_prediction``.
        candidate_interventions: A deterministic, rule-based shortlist of
            candidate interventions already matched to the top drivers
            (see ``policy_recommendation_engine.py``'s
            ``INTERVENTION_RULEBOOK``), each dict with keys ``trigger``
            (the driving feature), ``intervention`` (short label), and
            ``rationale`` (one-line justification). Granite is asked to
            prioritize and narrate these -- never to invent new ones.

    Returns:
        A complete prompt string.
    """
    driver_lines = _format_top_features(top_contributing_features)
    intervention_lines = "\n".join(
        f"- [{c['trigger']}] {c['intervention']}: {c['rationale']}"
        for c in candidate_interventions
    ) or "- (no rule-based candidate interventions were triggered)"

    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"TASK: Write a short, prioritized policy recommendation brief for "
        f"{district_name}, {state_name}.\n\n"
        f"DATA:\n"
        f"- Model prediction: {predicted_class} "
        f"({predicted_probability:.1%} underserved probability)\n"
        f"- Top contributing factors:\n{driver_lines}\n\n"
        f"- Candidate interventions already identified by the rule-based "
        f"system (choose from and prioritize ONLY these; do not invent "
        f"new interventions):\n{intervention_lines}\n\n"
        f"INSTRUCTIONS: Select and rank the 2-3 most relevant "
        f"interventions from the candidate list above, in priority order. "
        f"For each, write 1-2 sentences explaining why it addresses this "
        f"district's specific top contributing factors. Do not recommend "
        f"anything not present in the candidate list. Keep the total "
        f"response under 200 words."
    )


def build_ai_report_section_prompt(
    section_title: str,
    section_purpose: str,
    supporting_data_block: str,
) -> str:
    """Build a generic, reusable prompt for one section of the AI Report
    Generator's multi-section document.

    Args:
        section_title: Title of this report section (e.g. "Methodology",
            "Model Performance Summary").
        section_purpose: One-sentence description of what this section
            should accomplish.
        supporting_data_block: Pre-formatted string of the exact
            structured data this section must be grounded in (e.g. a
            metrics table via ``_format_metrics_table``, or a
            comparison table rendered as text).

    Returns:
        A complete prompt string.
    """
    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"TASK: Write the '{section_title}' section of a policy report. "
        f"Purpose of this section: {section_purpose}\n\n"
        f"DATA:\n{supporting_data_block}\n\n"
        f"INSTRUCTIONS: Write 2-4 paragraphs using only the data above. "
        f"Do not repeat the raw data verbatim as a list; synthesize it "
        f"into readable prose. Do not add a title heading -- just the "
        f"body text."
    )


def build_model_performance_section_data(
    comparison_table_rows: List[Dict[str, object]], best_model_name: str
) -> str:
    """Format the model comparison leaderboard as a supporting-data block
    for the AI Report Generator's "Model Performance Summary" section.

    Args:
        comparison_table_rows: List of dicts, one per candidate model
            (as produced by
            ``model_training.ModelTrainer.build_comparison_table`` /
            ``model_evaluation.ModelEvaluator.evaluate_all_models``,
            converted via ``DataFrame.to_dict("records")``).
        best_model_name: Name of the selected best model.

    Returns:
        A formatted text block.
    """
    lines = [f"Selected best model: {best_model_name}", "", "Full comparison:"]
    for row in comparison_table_rows:
        parts = ", ".join(f"{k}={v}" for k, v in row.items() if k != "model_name")
        lines.append(f"- {row.get('model_name', 'unknown')}: {parts}")
    return "\n".join(lines)


def build_qa_prompt(
    question: str,
    context_records: List[Dict[str, object]],
    max_context_records: int = 15,
) -> str:
    """Build the prompt for the Natural-Language Q&A engine ("Ask
    JusticeLens"), using a RAG-lite pattern: only precomputed district
    -level summary rows relevant to the question are injected as context,
    and the model is instructed to answer strictly from them.

    Args:
        question: The user's natural-language question.
        context_records: List of dicts (already filtered/retrieved by the
            Q&A engine's simple retrieval step -- see ``qa_engine.py``),
            each describing one district-year record with whatever
            columns are relevant (typically: state_name, district_name,
            fiscal_year, predicted_class, predicted_probability,
            cases_per_lakh_population, advice_enabled_ratio).
        max_context_records: Safety cap on how many records are embedded,
            to keep the prompt a bounded size regardless of how broad the
            retrieval step's filter was.

    Returns:
        A complete prompt string.
    """
    bounded_records = context_records[:max_context_records]
    context_lines = []
    for record in bounded_records:
        fields = ", ".join(f"{k}={v}" for k, v in record.items())
        context_lines.append(f"- {fields}")
    context_block = "\n".join(context_lines) if context_lines else "(no matching records found)"

    truncation_note = (
        f"\n\nNote: {len(context_records)} total matching records were found; "
        f"only the first {max_context_records} are shown above."
        if len(context_records) > max_context_records
        else ""
    )

    return (
        f"{_SYSTEM_PREAMBLE}\n\n"
        f"TASK: Answer the following question from a government official "
        f"using ONLY the district records provided below. If the records "
        f"do not contain enough information to answer confidently, say so "
        f"explicitly rather than guessing.\n\n"
        f"QUESTION: {question}\n\n"
        f"AVAILABLE RECORDS:\n{context_block}{truncation_note}\n\n"
        f"INSTRUCTIONS: Answer in 2-4 sentences. Reference specific "
        f"district/state names and numbers from the records above where "
        f"relevant. Do not reference any district not listed above."
    )
