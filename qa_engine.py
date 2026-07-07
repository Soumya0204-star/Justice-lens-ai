"""
qa_engine.py
=============
Natural-Language Q&A engine ("Ask JusticeLens") for JusticeLens AI.

Implements a RAG-lite pattern over the structured, already-computed
disparity predictions dataframe: a lightweight keyword-based retrieval
step selects the district-year records relevant to a user's question
(matching mentioned state/district names, or falling back to the
overall worst-affected districts for open-ended questions), then IBM
Granite is asked to answer strictly from those records. No vector store
or embedding model is required for this dataset's scale (a few thousand
rows) -- retrieval is a direct, auditable pandas filter rather than a
similarity search, which keeps the whole system easy to verify.

When watsonx.ai is unavailable, the engine still answers -- from the same
retrieved records -- using a deterministic template rather than returning
an error, per the system architecture's reliability requirements.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from justicelens import prompt_templates
from justicelens.logger import get_logger
from justicelens.utils import NarrativeGenerationError
from justicelens.watsonx_integration import GraniteClient, NarrativeResult

logger = get_logger(__name__)

#: Columns surfaced to the model (and the fallback template) as context
#: for each retrieved record. Kept deliberately compact so prompts stay a
#: bounded size regardless of how many records a broad question retrieves.
_CONTEXT_COLUMNS = [
    "state_name",
    "district_name",
    "fiscal_year",
    "predicted_class",
    "predicted_probability",
    "cases_per_lakh_population",
    "advice_enabled_ratio",
]

#: Required columns on the input predictions dataframe.
_REQUIRED_COLUMNS = _CONTEXT_COLUMNS


class JusticeLensQAEngine:
    """Answers natural-language questions about Tele-Law disparity using a
    simple retrieval step over the predictions dataframe plus IBM Granite
    narration (with deterministic fallback).

    Typical usage::

        qa = JusticeLensQAEngine()
        result = qa.answer(
            "Which districts in Bihar need urgent CSC intervention?",
            predictions_df=predictions_df,
        )
        print(result.narrative_text)
    """

    def __init__(
        self,
        client: Optional[GraniteClient] = None,
        max_context_records: int = 15,
        fallback_top_n: int = 10,
    ) -> None:
        """Initialize the Q&A engine.

        Args:
            client: A ``GraniteClient`` instance. Defaults to a new
                client constructed from environment configuration.
            max_context_records: Maximum number of retrieved records
                embedded in any single prompt.
            fallback_top_n: How many worst-affected districts to retrieve
                when the question does not name a specific state/district
                (open-ended questions like "which districts need help
                most?").
        """
        self.client = client or GraniteClient()
        self.max_context_records = max_context_records
        self.fallback_top_n = fallback_top_n

    def _extract_mentioned_names(
        self, question: str, candidate_names: List[str]
    ) -> List[str]:
        """Find which known state/district names are explicitly mentioned
        in the question text (case-insensitive substring match).

        Args:
            question: The user's natural-language question.
            candidate_names: All unique state or district names to check
                for.

        Returns:
            The subset of ``candidate_names`` that appear in ``question``.
        """
        question_lower = question.lower()
        mentioned = []
        for name in candidate_names:
            # Loose substring match (rather than requiring exact word
            # boundaries) since Indian district names often include
            # multi-word phrases with inconsistent spacing/hyphenation
            # across sources.
            if re.sub(r"\s+", " ", name.lower()).strip() in question_lower:
                mentioned.append(name)
        return mentioned

    def retrieve_context(
        self, question: str, predictions_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Retrieve the district-year records relevant to a question.

        Retrieval strategy:
            1. If the question mentions one or more known district names,
               return all fiscal-year records for those districts.
            2. Else if the question mentions one or more known state
               names, return the worst ``fallback_top_n`` districts
               within those states (by predicted probability).
            3. Else (fully open-ended question), return the overall worst
               ``fallback_top_n`` districts across all of India.

        Args:
            question: The user's natural-language question.
            predictions_df: Dataframe of district-year records with model
                predictions attached (columns in ``_REQUIRED_COLUMNS``).

        Returns:
            A filtered dataframe of relevant records, sorted worst-first
            by predicted probability.
        """
        district_names = predictions_df["district_name"].unique().tolist()
        state_names = predictions_df["state_name"].unique().tolist()

        mentioned_districts = self._extract_mentioned_names(question, district_names)
        if mentioned_districts:
            subset = predictions_df[
                predictions_df["district_name"].isin(mentioned_districts)
            ]
            logger.info(
                "Q&A retrieval matched %d named district(s): %s",
                len(mentioned_districts),
                mentioned_districts,
            )
            return subset.sort_values("predicted_probability", ascending=False)

        mentioned_states = self._extract_mentioned_names(question, state_names)
        if mentioned_states:
            subset = predictions_df[predictions_df["state_name"].isin(mentioned_states)]
            logger.info(
                "Q&A retrieval matched %d named state(s): %s",
                len(mentioned_states),
                mentioned_states,
            )
            return subset.sort_values("predicted_probability", ascending=False).head(
                self.fallback_top_n
            )

        logger.info(
            "Q&A retrieval found no named state/district in question; "
            "falling back to overall top-%d worst-affected districts.",
            self.fallback_top_n,
        )
        return predictions_df.sort_values(
            "predicted_probability", ascending=False
        ).head(self.fallback_top_n)

    def _build_fallback_answer(self, question: str, context_df: pd.DataFrame) -> str:
        """Build a deterministic, template-based answer directly from the
        retrieved records, used when Granite is unavailable.

        Args:
            question: The original question (echoed for context).
            context_df: Retrieved records.

        Returns:
            A plain-language answer string.
        """
        if context_df.empty:
            return (
                f"I could not find any district records matching your "
                f'question: "{question}". Try naming a specific state or '
                f"district."
            )

        lines = ["Based on the available records relevant to your question:\n"]
        for _, row in context_df.head(self.max_context_records).iterrows():
            lines.append(
                f"- {row['district_name']}, {row['state_name']} "
                f"(FY {row['fiscal_year']}): {row['predicted_class']} "
                f"({row['predicted_probability']:.1%} underserved "
                f"probability, {row['cases_per_lakh_population']:.1f} "
                f"cases per lakh population, "
                f"{row['advice_enabled_ratio']:.1%} advice-enabled ratio)."
            )
        lines.append(
            "\n[Note: This answer was generated by a deterministic "
            "template because the IBM watsonx.ai Granite narration "
            "service was unavailable at generation time. All figures "
            "above are drawn directly from the underlying model outputs.]"
        )
        return "\n".join(lines)

    def answer(self, question: str, predictions_df: pd.DataFrame) -> NarrativeResult:
        """Answer a natural-language question about Tele-Law disparity.

        Args:
            question: The user's natural-language question.
            predictions_df: Dataframe of district-year records with model
                predictions attached (columns in ``_REQUIRED_COLUMNS``).

        Returns:
            A :class:`NarrativeResult`.

        Raises:
            NarrativeGenerationError: If required columns are missing, the
                input is empty, or the question is blank.
        """
        if not question or not question.strip():
            raise NarrativeGenerationError("Question must be a non-empty string.")

        missing = [c for c in _REQUIRED_COLUMNS if c not in predictions_df.columns]
        if missing:
            raise NarrativeGenerationError(
                f"predictions_df is missing required column(s): {missing}"
            )
        if predictions_df.empty:
            raise NarrativeGenerationError(
                "Cannot answer questions from an empty predictions_df."
            )

        context_df = self.retrieve_context(question, predictions_df)
        context_records: List[Dict[str, object]] = context_df[_CONTEXT_COLUMNS].to_dict(
            "records"
        )

        prompt = prompt_templates.build_qa_prompt(
            question=question,
            context_records=context_records,
            max_context_records=self.max_context_records,
        )

        source_data = {
            "question": question,
            "retrieved_record_count": len(context_records),
            "retrieved_records": context_records[: self.max_context_records],
        }

        generation = self.client.generate_safe(prompt)
        if generation is not None:
            logger.info(
                "Q&A answered via Granite (%d context records retrieved)",
                len(context_records),
            )
            return NarrativeResult(
                narrative_text=generation.text,
                is_ai_generated=True,
                model_id=generation.model_id,
                latency_seconds=generation.latency_seconds,
                source_data=source_data,
            )

        logger.info(
            "Q&A falling back to template (%d context records retrieved)",
            len(context_records),
        )
        fallback_text = self._build_fallback_answer(question, context_df)
        return NarrativeResult(
            narrative_text=fallback_text,
            is_ai_generated=False,
            model_id="template_fallback",
            latency_seconds=0.0,
            source_data=source_data,
        )
