"""
policy_recommendation_engine.py
=================================
Policy Recommendation Engine for JusticeLens AI.

Combines a deterministic, auditable rule-based intervention rulebook
(mapping SHAP-identified disparity drivers to candidate policy
interventions) with IBM Granite narration to produce a short, prioritized
recommendation brief for a single district.

Design principle (consistent with every other generator in this package):
Granite is only ever asked to **prioritize and narrate** interventions
already selected by the deterministic rulebook below -- it is explicitly
instructed never to invent new interventions, so the set of possible
recommendations is fully enumerable and auditable independent of any LLM
call. When watsonx.ai is unavailable, the rulebook's own output is
presented directly (already ranked), with no narration layer at all.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from justicelens import prompt_templates
from justicelens.logger import get_logger
from justicelens.utils import NarrativeGenerationError
from justicelens.watsonx_integration import GraniteClient, NarrativeResult

logger = get_logger(__name__)

#: Deterministic mapping from SHAP-identified driver features to candidate
#: policy interventions. Each feature maps to a list of
#: (intervention, rationale) pairs. This rulebook encodes domain
#: assumptions made explicit here for auditability -- it is NOT sourced
#: from an official DoJ/NALSA policy document, and should be reviewed and
#: refined by a domain expert before real-world deployment; it is
#: structured as a starting point that a policy team can edit directly.
INTERVENTION_RULEBOOK: Dict[str, List[Dict[str, str]]] = {
    "rural_population_pct": [
        {
            "intervention": "Deploy mobile legal-aid vans / rural CSC outreach camps",
            "rationale": (
                "High rural population share correlates with lower "
                "physical access to Common Service Centres; mobile "
                "outreach reduces the distance barrier."
            ),
        }
    ],
    "advice_enabled_ratio": [
        {
            "intervention": "Increase panel-lawyer / paralegal volunteer deployment",
            "rationale": (
                "A low share of registered cases receiving actual advice "
                "suggests a bottleneck in legal-service capacity relative "
                "to registration volume."
            ),
        }
    ],
    "literacy_adjusted_expected_load": [
        {
            "intervention": "Launch multilingual IEC (information-education-communication) campaigns",
            "rationale": (
                "Where the literacy-adjusted latent need for assisted "
                "legal access is high, awareness campaigns in local "
                "languages help citizens understand how to access Tele-Law."
            ),
        }
    ],
    "cases_per_lakh_population": [
        {
            "intervention": "Run district-level awareness drives via panchayats and CSC VLEs",
            "rationale": (
                "Low per-capita registration relative to population "
                "suggests low awareness or trust in the service rather "
                "than low demand."
            ),
        }
    ],
    "yoy_growth_rate": [
        {
            "intervention": "Audit CSC operational continuity in this district",
            "rationale": (
                "A declining year-over-year registration trend can "
                "indicate a service disruption (CSC closure, staff "
                "vacancy, connectivity issues) rather than reduced need."
            ),
        }
    ],
    "population": [
        {
            "intervention": "Expand CSC density / onboard additional Village Level Entrepreneurs (VLEs)",
            "rationale": (
                "In high-population districts, absolute service capacity "
                "may not scale with population even if per-capita rates "
                "look moderate."
            ),
        }
    ],
    "sex_ratio": [
        {
            "intervention": "Run targeted women's legal-literacy and outreach campaigns",
            "rationale": (
                "A skewed sex ratio is associated in the model with "
                "elevated underserved risk, suggesting gender-specific "
                "barriers to legal-aid access may be present."
            ),
        }
    ],
}


class PolicyRecommendationEngine:
    """Matches SHAP-identified disparity drivers to candidate
    interventions and produces a prioritized, narrated recommendation
    brief.

    Typical usage::

        engine = PolicyRecommendationEngine()
        result = engine.generate(
            district_name="Example District",
            state_name="Example State",
            predicted_class="Underserved",
            predicted_probability=0.82,
            top_contributing_features=explanation["top_contributing_features"],
        )
        print(result.narrative_text)
    """

    def __init__(
        self,
        client: Optional[GraniteClient] = None,
        rulebook: Optional[Dict[str, List[Dict[str, str]]]] = None,
    ) -> None:
        """Initialize the engine.

        Args:
            client: A ``GraniteClient`` instance. Defaults to a new
                client constructed from environment configuration.
            rulebook: Override for the driver -> intervention mapping.
                Defaults to ``INTERVENTION_RULEBOOK``.
        """
        self.client = client or GraniteClient()
        self.rulebook = rulebook or INTERVENTION_RULEBOOK

    def match_interventions(
        self, top_contributing_features: List[Dict[str, object]]
    ) -> List[Dict[str, str]]:
        """Match SHAP top-contributing-features against the rulebook,
        returning only interventions triggered by features that *increase*
        underserved risk (features that decrease risk are protective
        factors, not intervention targets).

        Args:
            top_contributing_features: List of dicts as produced by
                ``shap_explainability.SHAPExplainer.explain_prediction``
                (keys: ``feature``, ``value``, ``shap_value``,
                ``direction``).

        Returns:
            List of dicts with keys ``trigger``, ``intervention``,
            ``rationale``, in the same order as the input features (i.e.
            already prioritized by SHAP contribution magnitude, since
            ``explain_prediction`` returns features sorted by
            ``abs(shap_value)`` descending).
        """
        candidates: List[Dict[str, str]] = []
        for item in top_contributing_features:
            if "increases" not in str(item.get("direction", "")):
                continue
            feature = item["feature"]
            for rule in self.rulebook.get(feature, []):
                candidates.append(
                    {
                        "trigger": feature,
                        "intervention": rule["intervention"],
                        "rationale": rule["rationale"],
                    }
                )
        return candidates

    def _build_fallback_narrative(
        self,
        district_name: str,
        state_name: str,
        predicted_class: str,
        predicted_probability: float,
        candidate_interventions: List[Dict[str, str]],
    ) -> str:
        """Build a deterministic recommendation brief directly from the
        rulebook match, used when Granite is unavailable.

        Args:
            district_name: District being assessed.
            state_name: Parent state/UT.
            predicted_class: Model's predicted label.
            predicted_probability: Predicted underserved probability.
            candidate_interventions: Output of :meth:`match_interventions`.

        Returns:
            A plain-language recommendation brief string.
        """
        lines = [
            f"Policy Recommendation Brief -- {district_name}, {state_name}\n",
            f"Model prediction: {predicted_class} "
            f"({predicted_probability:.1%} underserved probability).\n",
        ]
        if not candidate_interventions:
            lines.append(
                "No specific rule-based interventions were triggered by "
                "this district's top contributing factors. Recommend a "
                "manual review by a regional coordinator."
            )
        else:
            lines.append("Recommended interventions (priority order):")
            for i, candidate in enumerate(candidate_interventions[:3], start=1):
                lines.append(
                    f"{i}. {candidate['intervention']} "
                    f"(triggered by: {candidate['trigger'].replace('_', ' ')}) "
                    f"-- {candidate['rationale']}"
                )
        lines.append(
            "\n[Note: This brief was generated by a deterministic "
            "rule-based template because the IBM watsonx.ai Granite "
            "narration service was unavailable at generation time.]"
        )
        return "\n".join(lines)

    def generate(
        self,
        district_name: str,
        state_name: str,
        predicted_class: str,
        predicted_probability: float,
        top_contributing_features: List[Dict[str, object]],
    ) -> NarrativeResult:
        """Generate a prioritized, narrated policy recommendation brief
        for a single district.

        Args:
            district_name: District being assessed.
            state_name: Parent state/UT.
            predicted_class: Model's predicted label (e.g. "Underserved").
            predicted_probability: Predicted underserved probability.
            top_contributing_features: SHAP-derived top drivers, as
                produced by
                ``shap_explainability.SHAPExplainer.explain_prediction``.

        Returns:
            A :class:`NarrativeResult`.

        Raises:
            NarrativeGenerationError: If ``top_contributing_features`` is
                empty.
        """
        if not top_contributing_features:
            raise NarrativeGenerationError(
                "Cannot generate a policy recommendation without any "
                "top_contributing_features (SHAP explanation output)."
            )

        candidate_interventions = self.match_interventions(top_contributing_features)

        prompt = prompt_templates.build_policy_recommendation_prompt(
            district_name=district_name,
            state_name=state_name,
            predicted_class=predicted_class,
            predicted_probability=predicted_probability,
            top_contributing_features=top_contributing_features,
            candidate_interventions=candidate_interventions,
        )

        source_data = {
            "district_name": district_name,
            "state_name": state_name,
            "predicted_class": predicted_class,
            "predicted_probability": predicted_probability,
            "candidate_interventions": candidate_interventions,
        }

        generation = self.client.generate_safe(prompt)
        if generation is not None:
            logger.info(
                "Policy recommendation generated via Granite for '%s, %s'",
                district_name,
                state_name,
            )
            return NarrativeResult(
                narrative_text=generation.text,
                is_ai_generated=True,
                model_id=generation.model_id,
                latency_seconds=generation.latency_seconds,
                source_data=source_data,
            )

        logger.info(
            "Policy recommendation falling back to rule-based template "
            "for '%s, %s'",
            district_name,
            state_name,
        )
        fallback_text = self._build_fallback_narrative(
            district_name,
            state_name,
            predicted_class,
            predicted_probability,
            candidate_interventions,
        )
        return NarrativeResult(
            narrative_text=fallback_text,
            is_ai_generated=False,
            model_id="template_fallback",
            latency_seconds=0.0,
            source_data=source_data,
        )
