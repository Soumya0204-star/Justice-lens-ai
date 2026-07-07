"""
data_service.py
=================
Shared, cached data/model/generator loading layer for the JusticeLens AI
Streamlit application.

Every page imports from this module rather than calling the
``justicelens`` backend package directly, so that:

    * The (relatively expensive) data pipeline, model training, and SHAP
      explainer construction happen exactly once per app session
      (via ``st.cache_data`` / ``st.cache_resource``), not once per page
      visit.
    * Every page sees an identical, consistent view of the data and
      trained model.
    * Failure handling (missing ``shap``/``xgboost``, watsonx.ai
      unavailable) is centralized here, so individual pages can stay
      focused on layout/interaction and simply branch on a returned
      ``None``/error message.

This module deliberately does not import ``streamlit`` widgets beyond the
caching decorators -- it has no rendering logic, only data preparation --
keeping a clean separation between "get the data" and "show the data".
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

from justicelens import config
from justicelens.ai_report_generator import AIReportGenerator
from justicelens.data_engineering import DataEngineeringPipeline
from justicelens.data_loader import TeleLawDataLoader
from justicelens.district_comparison import DistrictComparisonGenerator
from justicelens.executive_summary_generator import ExecutiveSummaryGenerator
from justicelens.feature_engineering import FeatureEngineer
from justicelens.model_evaluation import ModelEvaluator
from justicelens.model_training import ModelTrainer, TrainingArtifacts
from justicelens.policy_recommendation_engine import PolicyRecommendationEngine
from justicelens.qa_engine import JusticeLensQAEngine
from justicelens.shap_explainability import SHAPExplainer
from justicelens.utils import ExplainabilityError, JusticeLensError
from justicelens.watsonx_integration import GraniteClient

#: Columns surfaced throughout the UI as a district-year record's
#: "profile" -- combined feature + prediction view.
DISTRICT_PROFILE_COLUMNS = [
    "state_name",
    "district_name",
    "fiscal_year",
    "cases_registered",
    "advice_enabled_count",
    "cases_per_lakh_population",
    "advice_enabled_ratio",
    "yoy_growth_rate",
    "rural_penetration_index",
    "literacy_adjusted_expected_load",
    "population",
    "rural_population_pct",
    "literacy_rate_pct",
    "sex_ratio",
]


@st.cache_data(show_spinner="Loading and preparing the Tele-Law dataset...")
def load_features_dataset() -> pd.DataFrame:
    """Run the full ingestion -> validation -> harmonization -> cleaning
    -> reconciliation -> feature-engineering pipeline once and cache the
    result for the whole app session.

    Returns:
        The feature-engineered dataframe (see
        ``feature_engineering.FeatureEngineer.engineer_all_features``).
    """
    loader = TeleLawDataLoader()
    telelaw_df = loader.load_telelaw_data()
    auxiliary_df = loader.load_auxiliary_data()

    engineering_pipeline = DataEngineeringPipeline()
    curated_df, _run_metadata = engineering_pipeline.run_pipeline(telelaw_df, auxiliary_df)

    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.engineer_all_features(curated_df)
    return features_df


@st.cache_resource(show_spinner="Training and comparing ML models (Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, XGBoost)...")
def get_training_artifacts(features_df: pd.DataFrame) -> TrainingArtifacts:
    """Train, cross-validate, and select the best disparity classification
    model, then evaluate it on the held-out test set (populating
    ``comparison_table`` / ``test_metrics``). Cached for the app session
    since retraining is expensive and the underlying data does not change
    within a session.

    Args:
        features_df: Output of :func:`load_features_dataset`.

    Returns:
        A fully populated ``TrainingArtifacts`` instance (models trained
        AND evaluated).
    """
    trainer = ModelTrainer()
    artifacts = trainer.run_training_pipeline(features_df)

    evaluator = ModelEvaluator()
    artifacts = evaluator.evaluate_all_models(artifacts)
    return artifacts


@st.cache_data(show_spinner="Scoring every district-year record...")
def get_predictions_df(_artifacts: TrainingArtifacts, features_df: pd.DataFrame) -> pd.DataFrame:
    """Score every district-year record in ``features_df`` with the best
    -selected model, producing a single dataframe the whole UI can filter
    /display/chart from.

    Args:
        _artifacts: Training artifacts (leading underscore tells
            Streamlit not to attempt to hash this argument, since it
            contains a fitted scikit-learn pipeline).
        features_df: The full feature-engineered dataframe.

    Returns:
        A dataframe with all of ``DISTRICT_PROFILE_COLUMNS`` plus
        ``predicted_class`` and ``predicted_probability``.
    """
    best_pipeline = _artifacts.best_result.pipeline
    feature_names = _artifacts.feature_names

    X_all = features_df[feature_names].fillna(features_df[feature_names].median(numeric_only=True))
    predicted_probability = best_pipeline.predict_proba(X_all)[:, 1]
    predicted_class_index = best_pipeline.predict(X_all)

    profile_columns = [c for c in DISTRICT_PROFILE_COLUMNS if c in features_df.columns]
    predictions_df = features_df[profile_columns].copy()
    predictions_df["predicted_probability"] = predicted_probability
    predictions_df["predicted_class"] = [
        config.ML_CLASS_NAMES[i] for i in predicted_class_index
    ]
    return predictions_df.reset_index(drop=True)


@st.cache_resource(show_spinner="Building the SHAP explainability engine...")
def get_shap_explainer(_artifacts: TrainingArtifacts) -> Tuple[Optional[SHAPExplainer], Optional[str]]:
    """Construct a ``SHAPExplainer`` for the best-selected model.

    Args:
        _artifacts: Training artifacts (leading underscore -- see
            :func:`get_predictions_df`).

    Returns:
        A tuple ``(explainer, error_message)``: on success, ``explainer``
        is a ready-to-use ``SHAPExplainer`` and ``error_message`` is
        ``None``; on failure (typically the optional ``shap`` package not
        being installed), ``explainer`` is ``None`` and ``error_message``
        describes why, for direct display in the UI.
    """
    bundle = {
        "pipeline": _artifacts.best_result.pipeline,
        "model_name": _artifacts.best_result.model_name,
        "feature_names": _artifacts.feature_names,
    }
    try:
        explainer = SHAPExplainer(bundle, background_data=_artifacts.X_train)
        return explainer, None
    except ExplainabilityError as exc:
        return None, str(exc)


@st.cache_data(show_spinner=False)
def get_fallback_feature_importance(_artifacts: TrainingArtifacts) -> Optional[pd.DataFrame]:
    """Compute an approximate global feature-importance table directly
    from the fitted model (``feature_importances_`` for tree ensembles,
    absolute coefficients for linear models), for use when SHAP is not
    installed and the richer SHAP-based importance is unavailable.

    Args:
        _artifacts: Training artifacts (leading underscore -- see
            :func:`get_predictions_df`).

    Returns:
        A dataframe with columns ``feature`` and ``importance``, sorted
        descending, or ``None`` if the model exposes neither
        ``feature_importances_`` nor ``coef_``.
    """
    classifier = _artifacts.best_result.pipeline.named_steps["classifier"]
    feature_names = _artifacts.feature_names

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
    else:
        return None

    return (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


@st.cache_resource(show_spinner=False)
def get_granite_client() -> GraniteClient:
    """Construct (once per session) the shared ``GraniteClient`` used by
    every generator, so the IAM token cache is reused across pages rather
    than re-authenticating on every navigation.

    Returns:
        A ``GraniteClient`` instance (may or may not be
        ``is_available()`` depending on whether ``.env`` credentials are
        set -- callers should check before assuming live generation will
        succeed).
    """
    return GraniteClient()


@st.cache_resource(show_spinner=False)
def get_generators(_client: GraniteClient) -> Dict[str, object]:
    """Construct (once per session) every narrative generator, sharing the
    single ``GraniteClient`` instance.

    Args:
        _client: The shared Granite client (leading underscore -- see
            :func:`get_predictions_df`).

    Returns:
        Dict with keys: ``executive_summary``, ``district_comparison``,
        ``policy_recommendation``, ``ai_report``, ``qa``.
    """
    return {
        "executive_summary": ExecutiveSummaryGenerator(_client),
        "district_comparison": DistrictComparisonGenerator(_client),
        "policy_recommendation": PolicyRecommendationEngine(_client),
        "ai_report": AIReportGenerator(_client),
        "qa": JusticeLensQAEngine(_client),
    }


def build_district_record(
    row: pd.Series,
    shap_explainer: Optional[SHAPExplainer],
    features_df: pd.DataFrame,
) -> Dict[str, object]:
    """Build the standard "district record" dict consumed by the District
    Comparison and Policy Recommendation generators (and the district
    -detail views), attaching SHAP top-contributing-features when a SHAP
    explainer is available.

    Args:
        row: A single row (as a Series) from the predictions dataframe
            (see :func:`get_predictions_df`), identifying one
            district-fiscal-year record.
        shap_explainer: A ``SHAPExplainer`` instance, or ``None`` if SHAP
            is unavailable (in which case ``top_contributing_features``
            is set to an empty list).
        features_df: The full feature-engineered dataframe, used to
            locate the matching feature row for SHAP computation.

    Returns:
        A dict with the standard fields required by
        ``district_comparison.py`` / ``policy_recommendation_engine.py``.
    """
    record: Dict[str, object] = {
        "district_name": row["district_name"],
        "state_name": row["state_name"],
        "fiscal_year": row["fiscal_year"],
        "predicted_class": row["predicted_class"],
        "predicted_probability": float(row["predicted_probability"]),
        "cases_per_lakh_population": float(row["cases_per_lakh_population"]),
        "advice_enabled_ratio": float(row["advice_enabled_ratio"]),
        "top_contributing_features": [],
    }

    if shap_explainer is not None:
        try:
            match = features_df[
                (features_df["district_name"] == row["district_name"])
                & (features_df["state_name"] == row["state_name"])
                & (features_df["fiscal_year"] == row["fiscal_year"])
            ]
            if not match.empty:
                explanation = shap_explainer.explain_prediction(match, instance_index=0)
                record["top_contributing_features"] = explanation["top_contributing_features"]
        except JusticeLensError:
            # SHAP computation failing for a single record should never
            # break the surrounding page -- the record simply carries no
            # feature attribution, and callers already handle an empty list.
            record["top_contributing_features"] = []

    return record


def get_unique_states(predictions_df: pd.DataFrame) -> List[str]:
    """Return sorted unique state/UT names present in the predictions
    dataframe, for populating filter widgets.

    Args:
        predictions_df: Output of :func:`get_predictions_df`.

    Returns:
        Sorted list of state/UT names.
    """
    return sorted(predictions_df["state_name"].unique().tolist())


def get_unique_fiscal_years(predictions_df: pd.DataFrame) -> List[str]:
    """Return fiscal years present in the predictions dataframe, ordered
    per ``config.FISCAL_YEARS`` (rather than alphabetically).

    Args:
        predictions_df: Output of :func:`get_predictions_df`.

    Returns:
        Ordered list of fiscal year strings present in the data.
    """
    present = set(predictions_df["fiscal_year"].unique().tolist())
    return [fy for fy in config.FISCAL_YEARS if fy in present]


def get_districts_for_state(predictions_df: pd.DataFrame, state_name: str) -> List[str]:
    """Return sorted unique district names within a given state.

    Args:
        predictions_df: Output of :func:`get_predictions_df`.
        state_name: The state/UT to filter by.

    Returns:
        Sorted list of district names within ``state_name``.
    """
    subset = predictions_df[predictions_df["state_name"] == state_name]
    return sorted(subset["district_name"].unique().tolist())
