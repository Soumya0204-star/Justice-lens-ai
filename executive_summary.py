"""
executive_summary_generator.py
================================
Executive Summary Generator for JusticeLens AI.

Produces a short, decision-oriented narrative summarizing Tele-Law legal
-access disparity for a given scope (all-India, a single state, or a
single fiscal year), grounded in model predictions and global SHAP feature
importance. Uses IBM Granite (via ``watsonx_integration.GraniteClient``)
when available, and transparently falls back to a deterministic,
template-based summary -- built from the exact same computed statistics --
when watsonx.ai is unreachable or not configured, per the system
architecture's reliability requirements.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from justicelens import prompt_templates
from justicelens.logger import get_logger
from justicelens.utils import NarrativeGenerationError
from justicelens.watsonx_integration import GraniteClient, NarrativeResult

logger = get_logger(__name__)

#: Columns that must be present in the input dataframe for a summary to
#: be computed.
_REQUIRED_COLUMNS = [
    "state_name",
    "district_name",
    "predicted_class",
    "predicted_probability",
]


class ExecutiveSummaryGenerator:
    """Generates scope-level executive summaries of disparity findings.

    Typical usage::

        generator = ExecutiveSummaryGenerator()
        result = generator.generate(
            scope_label="All India",
            predictions_df=predictions_df,
            global_top_drivers=importance_df,
        )
        print(result.narrative_text)
    """

    def __init__(self, client: Optional[GraniteClient] = None) -> None:
        """Initialize the generator.

        Args:
            client: A ``GraniteClient`` instance. Defaults to a new
                client constructed from environment configuration.
        """
        self.client = client or GraniteClient()

    def _compute_scope_statistics(
        self, predictions_df: pd.DataFrame, top_n_districts: int
    ) -> Dict[str, object]:
        """Compute the deterministic statistics both the Granite prompt
        and the template fallback are grounded in.

        Args:
            predictions_df: Dataframe with (at least) the columns in
                ``_REQUIRED_COLUMNS``.
            top_n_districts: How many worst-affected districts to surface.

        Returns:
            Dict with keys: ``total_districts``,
            ``underserved_district_count``, ``average_probability``,
            ``top_underserved_districts`` (list of dicts).
        """
        total_districts = len(predictions_df)
        underserved_mask = predictions_df["predicted_class"] == "Underserved"
        underserved_count = int(underserved_mask.sum())
        average_probability = float(predictions_df["predicted_probability"].mean())

        top_districts = (
            predictions_df.sort_values("predicted_probability", ascending=False)
            .head(top_n_districts)[["district_name", "state_name", "predicted_probability"]]
            .to_dict("records")
        )

        return {
            "total_districts": total_districts,
            "underserved_district_count": underserved_count,
            "average_probability": round(average_probability, 4),
            "top_underserved_districts": top_districts,
        }

    def _build_fallback_narrative(
        self, scope_label: str, stats: Dict[str, object], top_drivers: List[Dict[str, object]]
    ) -> str:
        """Build a deterministic, template-based executive summary from
        precomputed statistics, used when Granite is unavailable.

        Args:
            scope_label: Human-readable scope description.
            stats: Output of :meth:`_compute_scope_statistics`.
            top_drivers: Global feature importance records.

        Returns:
            A plain-language summary string.
        """
        share = stats["underserved_district_count"] / max(stats["total_districts"], 1)
        district_names = ", ".join(
            f"{d['district_name']} ({d['state_name']})"
            for d in stats["top_underserved_districts"][:3]
        ) or "no districts met the underserved threshold"

        driver_names = ", ".join(
            row["feature"].replace("_", " ") for row in top_drivers[:3]
        ) or "insufficient driver data"

        return (
            f"Executive Summary -- {scope_label}\n\n"
            f"Across {stats['total_districts']} district-fiscal-year records "
            f"analyzed, {stats['underserved_district_count']} "
            f"({share:.1%}) were flagged as 'Underserved' by the disparity "
            f"classification model, with an average predicted underserved "
            f"probability of {stats['average_probability']:.1%}. The most "
            f"affected districts in this scope are {district_names}. The "
            f"strongest global drivers of underserved risk identified by "
            f"the model's SHAP analysis are {driver_names}. These findings "
            f"suggest resource allocation and outreach planning should "
            f"prioritize the districts listed above.\n\n"
            f"[Note: This summary was generated by a deterministic "
            f"template because the IBM watsonx.ai Granite narration "
            f"service was unavailable at generation time. All figures "
            f"above are drawn directly from the underlying model outputs.]"
        )

    def generate(
        self,
        scope_label: str,
        predictions_df: pd.DataFrame,
        global_top_drivers: pd.DataFrame,
        top_n_districts: int = 5,
    ) -> NarrativeResult:
        """Generate an executive summary for a given scope.

        Args:
            scope_label: Human-readable label for what this summary
                covers (e.g. "All India", "Bihar", "FY 2023-24").
            predictions_df: Dataframe of district-year records already
                scoped to what this summary should cover, with model
                predictions attached (columns in ``_REQUIRED_COLUMNS``).
            global_top_drivers: Feature importance dataframe (columns
                ``feature``, ``mean_abs_shap_value``) from
                ``shap_explainability.SHAPExplainer.compute_feature_importance``.
            top_n_districts: How many worst-affected districts to
                highlight.

        Returns:
            A :class:`NarrativeResult`.

        Raises:
            NarrativeGenerationError: If required columns are missing or
                the input is empty.
        """
        missing = [c for c in _REQUIRED_COLUMNS if c not in predictions_df.columns]
        if missing:
            raise NarrativeGenerationError(
                f"predictions_df is missing required column(s): {missing}"
            )
        if predictions_df.empty:
            raise NarrativeGenerationError(
                "Cannot generate an executive summary from an empty predictions_df."
            )

        stats = self._compute_scope_statistics(predictions_df, top_n_districts)
        top_drivers_records = global_top_drivers.head(5).to_dict("records")

        prompt = prompt_templates.build_executive_summary_prompt(
            scope_label=scope_label,
            total_districts=stats["total_districts"],
            underserved_district_count=stats["underserved_district_count"],
            average_ladi_or_probability=stats["average_probability"],
            top_underserved_districts=stats["top_underserved_districts"],
            global_top_drivers=top_drivers_records,
        )

        generation = self.client.generate_safe(prompt)
        if generation is not None:
            logger.info("Executive summary generated via Granite for scope '%s'", scope_label)
            return NarrativeResult(
                narrative_text=generation.text,
                is_ai_generated=True,
                model_id=generation.model_id,
                latency_seconds=generation.latency_seconds,
                source_data=stats,
            )

        logger.info(
            "Executive summary falling back to template for scope '%s'", scope_label
        )
        fallback_text = self._build_fallback_narrative(
            scope_label, stats, top_drivers_records
        )
        return NarrativeResult(
            narrative_text=fallback_text,
            is_ai_generated=False,
            model_id="template_fallback",
            latency_seconds=0.0,
            source_data=stats,
        )
