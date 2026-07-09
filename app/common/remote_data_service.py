"""
common/data_service.py
=======================
Shared, cached data/model/generator loading layer for the JusticeLens AI
Streamlit application. (LOCAL MODE – runs ML locally)
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
from justicelens.executive_summary import ExecutiveSummaryGenerator  # ✅ FIXED: removed "_generator"
from justicelens.feature_engineering import FeatureEngineer
from justicelens.model_evaluation import ModelEvaluator
from justicelens.model_training import ModelTrainer, TrainingArtifacts
from justicelens.policy_recommendation_engine import PolicyRecommendationEngine
from justicelens.qa_engine import JusticeLensQAEngine
from justicelens.shap_explainability import SHAPExplainer
from justicelens.utils import ExplainabilityError, JusticeLensError
from justicelens.watsonx_integration import GraniteClient

DISTRICT_PROFILE_COLUMNS = [
    "state_name", "district_name", "fiscal_year",
    "cases_registered", "advice_enabled_count",
    "cases_per_lakh_population", "advice_enabled_ratio",
    "yoy_growth_rate", "rural_penetration_index",
    "literacy_adjusted_expected_load",
    "population", "rural_population_pct", "literacy_rate_pct", "sex_ratio",
]

@st.cache_data(show_spinner="Loading and preparing the Tele-Law dataset...")
def load_features_dataset() -> pd.DataFrame:
    loader = TeleLawDataLoader()
    telelaw_df = loader.load_telelaw_data()
    auxiliary_df = loader.load_auxiliary_data()
    eng_pipeline = DataEngineeringPipeline()
    curated_df, _ = eng_pipeline.run_pipeline(telelaw_df, auxiliary_df)
    feature_engineer = FeatureEngineer()
    return feature_engineer.engineer_all_features(curated_df)

@st.cache_resource(show_spinner="Training and comparing ML models...")
def get_training_artifacts(features_df: pd.DataFrame) -> TrainingArtifacts:
    trainer = ModelTrainer()
    artifacts = trainer.run_training_pipeline(features_df)
    evaluator = ModelEvaluator()
    return evaluator.evaluate_all_models(artifacts)

@st.cache_data(show_spinner="Scoring every district-year record...")
def get_predictions_df(_artifacts: TrainingArtifacts, features_df: pd.DataFrame) -> pd.DataFrame:
    best_pipeline = _artifacts.best_result.pipeline
    feature_names = _artifacts.feature_names
    X_all = features_df[feature_names].fillna(features_df[feature_names].median(numeric_only=True))
    predicted_probability = best_pipeline.predict_proba(X_all)[:, 1]
    predicted_class_index = best_pipeline.predict(X_all)
    predictions_df = features_df.copy()
    predictions_df["predicted_probability"] = predicted_probability
    predictions_df["predicted_class"] = [config.ML_CLASS_NAMES[i] for i in predicted_class_index]
    return predictions_df.reset_index(drop=True)

@st.cache_resource(show_spinner="Building the SHAP explainability engine...")
def get_shap_explainer(_artifacts: TrainingArtifacts) -> Tuple[Optional[SHAPExplainer], Optional[str]]:
    bundle = {
        "pipeline": _artifacts.best_result.pipeline,
        "model_name": _artifacts.best_result.model_name,
        "feature_names": _artifacts.feature_names,
    }
    try:
        return SHAPExplainer(bundle, background_data=_artifacts.X_train), None
    except ExplainabilityError as e:
        return None, str(e)

@st.cache_data(show_spinner=False)
def get_fallback_feature_importance(_artifacts: TrainingArtifacts) -> Optional[pd.DataFrame]:
    classifier = _artifacts.best_result.pipeline.named_steps["classifier"]
    feature_names = _artifacts.feature_names
    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = abs(classifier.coef_[0])
    else:
        return None
    return pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)

@st.cache_resource(show_spinner=False)
def get_granite_client() -> GraniteClient:
    return GraniteClient()

@st.cache_resource(show_spinner=False)
def get_generators(_client: GraniteClient):
    return {
        "executive_summary": ExecutiveSummaryGenerator(_client),
        "district_comparison": DistrictComparisonGenerator(_client),
        "policy_recommendation": PolicyRecommendationEngine(_client),
        "ai_report": AIReportGenerator(_client),
        "qa": JusticeLensQAEngine(_client),
    }

def build_district_record(row, shap_explainer, features_df):
    record = {
        "district_name": row["district_name"],
        "state_name": row["state_name"],
        "fiscal_year": row["fiscal_year"],
        "predicted_class": row["predicted_class"],
        "predicted_probability": float(row["predicted_probability"]),
        "cases_per_lakh_population": float(row["cases_per_lakh_population"]),
        "advice_enabled_ratio": float(row["advice_enabled_ratio"]),
        "top_contributing_features": [],
    }
    if shap_explainer:
        try:
            match = features_df[
                (features_df["district_name"] == row["district_name"]) &
                (features_df["state_name"] == row["state_name"]) &
                (features_df["fiscal_year"] == row["fiscal_year"])
            ]
            if not match.empty:
                exp = shap_explainer.explain_prediction(match, instance_index=0)
                record["top_contributing_features"] = exp["top_contributing_features"]
        except JusticeLensError:
            pass
    return record

def get_unique_states(predictions_df: pd.DataFrame) -> List[str]:
    return sorted(predictions_df["state_name"].unique().tolist())

def get_unique_fiscal_years(predictions_df: pd.DataFrame) -> List[str]:
    present = set(predictions_df["fiscal_year"].unique().tolist())
    return [fy for fy in config.FISCAL_YEARS if fy in present]

def get_districts_for_state(predictions_df: pd.DataFrame, state_name: str) -> List[str]:
    return sorted(predictions_df[predictions_df["state_name"] == state_name]["district_name"].unique().tolist())