"""
orchestrator/run_pipeline.py
=============================
Standalone orchestrator for JusticeLens AI.

Run this script periodically (e.g., daily via IBM Code Engine Job) to:
- Fetch fresh data (or synthetic fallback)
- Engineer features
- Train and evaluate ML models
- Compute SHAP explanations
- Save the trained model, predictions, and SHAP data to disk.

This script is designed to run in a serverless container (IBM Cloud Code Engine Job)
and writes all artifacts to the local filesystem (which can be mounted to a volume
or uploaded to IBM Cloud Object Storage).
"""

import sys
import joblib
import pandas as pd
from pathlib import Path

# Ensure the project root is in the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from justicelens import config
from justicelens.logger import get_logger
from justicelens.data_loader import TeleLawDataLoader
from justicelens.data_engineering import DataEngineeringPipeline
from justicelens.feature_engineering import FeatureEngineer
from justicelens.model_training import ModelTrainer
from justicelens.model_evaluation import ModelEvaluator
from justicelens.shap_explainability import SHAPExplainer

logger = get_logger(__name__)

def run_full_pipeline():
    logger.info("🚀 Starting JusticeLens Orchestrator Pipeline...")

    # 1. Data Loading
    logger.info("Step 1/6: Loading data...")
    loader = TeleLawDataLoader()
    telelaw_df = loader.load_telelaw_data()
    auxiliary_df = loader.load_auxiliary_data()

    # 2. Data Engineering
    logger.info("Step 2/6: Engineering data...")
    eng_pipeline = DataEngineeringPipeline()
    curated_df, meta = eng_pipeline.run_pipeline(telelaw_df, auxiliary_df)

    # 3. Feature Engineering
    logger.info("Step 3/6: Creating features...")
    feature_engineer = FeatureEngineer()
    features_df = feature_engineer.engineer_all_features(curated_df)

    # 4. Model Training & Evaluation
    logger.info("Step 4/6: Training models...")
    trainer = ModelTrainer()
    artifacts = trainer.run_training_pipeline(features_df)

    evaluator = ModelEvaluator()
    artifacts = evaluator.evaluate_all_models(artifacts)

    # 5. Save Best Model
    logger.info("Step 5/6: Saving model...")
    model_path = trainer.save_best_model(artifacts)
    evaluator.save_comparison_report(artifacts)

    # 6. Generate Predictions for ALL data (for the API)
    logger.info("Step 6/6: Generating prediction dataset...")
    best_pipeline = artifacts.best_result.pipeline
    feature_names = artifacts.feature_names
    X_all = features_df[feature_names].fillna(
        features_df[feature_names].median(numeric_only=True)
    )
    
    predicted_prob = best_pipeline.predict_proba(X_all)[:, 1]
    predicted_class_idx = best_pipeline.predict(X_all)
    predicted_class = [config.ML_CLASS_NAMES[i] for i in predicted_class_idx]

    # Build the final prediction dataframe
    predictions_df = features_df.copy()
    predictions_df["predicted_probability"] = predicted_prob
    predictions_df["predicted_class"] = predicted_class
    
    # Save predictions
    pred_path = config.DATA_PROCESSED_DIR / "predictions_all.csv"
    predictions_df.to_csv(pred_path, index=False)
    logger.info(f"✅ Predictions saved to {pred_path}")

    # 7. SHAP Explainer (Optional but recommended)
    logger.info("Bonus: Computing SHAP background data...")
    try:
        # Save SHAP background data (sample of training set) so the API doesn't need to recompute
        shap_background = artifacts.X_train.sample(min(200, len(artifacts.X_train)), random_state=42)
        shap_background_path = config.DATA_PROCESSED_DIR / "shap_background.csv"
        shap_background.to_csv(shap_background_path, index=False)
        
        # Generate and save SHAP summary plot for the API to serve
        explainer = SHAPExplainer(
            {"pipeline": best_pipeline, "feature_names": feature_names},
            background_data=artifacts.X_train,
            max_background_samples=200
        )
        explainer.generate_summary_plot(artifacts.X_test)
        logger.info(f"✅ SHAP background saved to {shap_background_path}")
    except Exception as e:
        logger.warning(f"SHAP post-processing skipped: {e}")

    logger.info("🎉 Orchestrator pipeline completed successfully!")

if __name__ == "__main__":
    run_full_pipeline()