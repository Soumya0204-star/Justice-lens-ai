"""
model_training.py
===================
Model training layer for JusticeLens AI's disparity classification task.

Given the feature-engineered dataframe produced by
``feature_engineering.FeatureEngineer.engineer_all_features``, this module:

    1. Derives a binary classification target, ``is_underserved``, marking
       district-fiscal-year records in the bottom quantile of rural legal
       -access penetration as "Underserved" (see
       ``config.ML_TARGET_QUANTILE`` / ``config.ML_TARGET_SOURCE_COLUMN``).
    2. Trains **five** candidate classifiers on an identical train/test
       split: Logistic Regression, Decision Tree, Random Forest, Gradient
       Boosting, and XGBoost.
    3. Cross-validates every candidate on the training fold and evaluates
       all of them on the held-out test fold using an identical metric
       suite.
    4. Automatically ranks candidates by a configurable primary metric
       (default: ROC-AUC) and selects the best model.
    5. Persists the best model (plus its feature list and metadata) to
       disk via ``joblib`` so it can be reloaded by
       ``model_evaluation.py``, ``shap_explainability.py``, or a serving
       layer without retraining.

XGBoost is an optional dependency: if the ``xgboost`` package is not
installed in the current environment, the pipeline logs a warning, excludes
it from the comparison, and continues with the remaining four models rather
than failing outright -- this keeps the pipeline runnable in restricted
environments while still matching the full five-model comparison whenever
xgboost is available (as it will be per ``requirements.txt``).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from justicelens import config
from justicelens.logger import get_logger
from justicelens.utils import ModelTrainingError, ensure_dir

logger = get_logger(__name__)

try:
    from xgboost import XGBClassifier

    _XGBOOST_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without xgboost installed
    _XGBOOST_AVAILABLE = False
    logger.warning(
        "The 'xgboost' package is not installed in this environment. "
        "XGBoost will be excluded from model comparison. Install it via "
        "`pip install xgboost` (see requirements.txt) to include it."
    )


@dataclass
class TrainedModelResult:
    """Result of training and evaluating a single candidate model.

    Attributes:
        model_name: Human-readable model identifier (e.g.
            "Random Forest").
        pipeline: The fitted scikit-learn ``Pipeline`` (preprocessing +
            estimator).
        cv_mean_score: Mean cross-validation score on the training fold,
            using ``config.ML_PRIMARY_METRIC``.
        cv_std_score: Standard deviation of the cross-validation score.
        test_metrics: Dict of metric name -> value computed on the held
            -out test set (populated by ``model_evaluation.py``, left
            empty here since this module only trains).
        requires_scaling: Whether this model's pipeline includes a
            ``StandardScaler`` step (informational, used by SHAP
            explainer selection downstream).
    """

    model_name: str
    pipeline: Pipeline
    cv_mean_score: float
    cv_std_score: float
    test_metrics: Dict[str, float] = field(default_factory=dict)
    requires_scaling: bool = False


@dataclass
class TrainingArtifacts:
    """Everything downstream evaluation/explainability modules need,
    bundled together so a single training run's outputs are passed around
    as one object rather than several loosely related variables.

    Attributes:
        results: All trained candidate results, in the order trained.
        best_result: The candidate selected as best by
            ``config.ML_PRIMARY_METRIC``.
        X_train: Training feature matrix (untransformed, original units --
            each model's own pipeline applies its own preprocessing).
        X_test: Held-out test feature matrix.
        y_train: Training labels.
        y_test: Held-out test labels.
        feature_names: Ordered list of feature column names used for
            training.
        comparison_table: Leaderboard dataframe, sorted best-first by the
            primary metric, with CV and (if available) test metrics.
    """

    results: List[TrainedModelResult]
    best_result: TrainedModelResult
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: List[str]
    comparison_table: pd.DataFrame


class DisparityTargetBuilder:
    """Derives the binary ``is_underserved`` classification target from
    engineered features.

    A district-fiscal-year record is labeled "Underserved" (1) when its
    ``rural_penetration_index`` falls at or below the configured quantile
    threshold (default: bottom 25%) of the full dataset's distribution;
    all other records are labeled "Adequately Served" (0). This operationalizes
    the internship problem statement's notion of regional disparity into a
    concrete, reproducible supervised-learning target.
    """

    def __init__(
        self,
        source_column: str = config.ML_TARGET_SOURCE_COLUMN,
        quantile: float = config.ML_TARGET_QUANTILE,
        target_column: str = config.ML_TARGET_COLUMN,
    ) -> None:
        """Initialize the target builder.

        Args:
            source_column: Continuous feature the binary label is derived
                from.
            quantile: Quantile threshold (0-1); values at or below this
                quantile of ``source_column`` are labeled 1.
            target_column: Name of the output binary label column.

        Raises:
            ValueError: If ``quantile`` is not within ``(0, 1)``.
        """
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"quantile must be in (0, 1), got {quantile}")
        self.source_column = source_column
        self.quantile = quantile
        self.target_column = target_column

    def build_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Add the binary target column to a copy of ``df``.

        Args:
            df: Feature-engineered dataframe containing
                ``self.source_column``.

        Returns:
            A tuple of ``(df_with_target, threshold_value)`` where
            ``threshold_value`` is the actual numeric cutoff used, logged
            for auditability.

        Raises:
            ModelTrainingError: If ``self.source_column`` is absent, or
                the resulting target has fewer than two classes (which
                would make classification meaningless).
        """
        if self.source_column not in df.columns:
            raise ModelTrainingError(
                f"Cannot build classification target: source column "
                f"'{self.source_column}' is absent from the input "
                f"dataframe. Run feature_engineering first."
            )

        result = df.copy()
        threshold = float(result[self.source_column].quantile(self.quantile))
        result[self.target_column] = (
            result[self.source_column] <= threshold
        ).astype(int)

        class_counts = result[self.target_column].value_counts().to_dict()
        if len(class_counts) < 2:
            raise ModelTrainingError(
                f"Derived target '{self.target_column}' has only "
                f"{len(class_counts)} distinct class(es): {class_counts}. "
                "Check for degenerate/constant source-column values."
            )

        logger.info(
            "Built binary target '%s' from '%s' at quantile=%.2f "
            "(threshold=%.4f). Class distribution: %s",
            self.target_column,
            self.source_column,
            self.quantile,
            threshold,
            class_counts,
        )
        return result, threshold


class ModelTrainer:
    """Trains, cross-validates, and compares multiple classifier families
    on an identical train/test split, then selects and persists the best
    performing model.

    Typical usage::

        trainer = ModelTrainer()
        artifacts = trainer.run_training_pipeline(features_df)
        trainer.save_best_model(artifacts, path=config.MODEL_DIR / "best_disparity_model.joblib")
    """

    def __init__(
        self,
        feature_columns: Optional[List[str]] = None,
        target_builder: Optional[DisparityTargetBuilder] = None,
        test_size: float = config.ML_TEST_SIZE,
        cv_folds: int = config.ML_CV_FOLDS,
        primary_metric: str = config.ML_PRIMARY_METRIC,
        random_state: int = config.RANDOM_SEED,
    ) -> None:
        """Initialize the trainer.

        Args:
            feature_columns: Ordered list of columns to use as model
                features. Defaults to ``config.ML_FEATURE_COLUMNS``.
            target_builder: Target-construction strategy. Defaults to a
                new ``DisparityTargetBuilder`` with config defaults.
            test_size: Fraction of data held out for testing.
            cv_folds: Number of stratified cross-validation folds.
            primary_metric: Metric used to rank/select the best model.
                One of "roc_auc", "f1", "accuracy", "precision", "recall".
            random_state: Seed for the train/test split, cross-validation
                shuffling, and every stochastic estimator.

        Raises:
            ValueError: If ``primary_metric`` is not a supported value.
        """
        supported_metrics = {"roc_auc", "f1", "accuracy", "precision", "recall"}
        if primary_metric not in supported_metrics:
            raise ValueError(
                f"primary_metric must be one of {supported_metrics}, got "
                f"'{primary_metric}'"
            )

        self.feature_columns = feature_columns or list(config.ML_FEATURE_COLUMNS)
        self.target_builder = target_builder or DisparityTargetBuilder()
        self.test_size = test_size
        self.cv_folds = cv_folds
        self.primary_metric = primary_metric
        self.random_state = random_state

    def _build_model_registry(self) -> Dict[str, Tuple[object, bool]]:
        """Construct the candidate model registry.

        Returns:
            Dict mapping model name -> ``(estimator_instance,
            requires_scaling)``. Tree-based ensembles do not require
            feature scaling; Logistic Regression does, since it is
            sensitive to feature magnitude.
        """
        registry: Dict[str, Tuple[object, bool]] = {
            "Logistic Regression": (
                LogisticRegression(
                    max_iter=1000,
                    random_state=self.random_state,
                    class_weight="balanced",
                ),
                True,
            ),
            "Decision Tree": (
                DecisionTreeClassifier(
                    max_depth=6,
                    min_samples_leaf=5,
                    random_state=self.random_state,
                    class_weight="balanced",
                ),
                False,
            ),
            "Random Forest": (
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=self.random_state,
                    class_weight="balanced",
                    n_jobs=-1,
                ),
                False,
            ),
            "Gradient Boosting": (
                GradientBoostingClassifier(
                    n_estimators=200,
                    max_depth=3,
                    learning_rate=0.05,
                    random_state=self.random_state,
                ),
                False,
            ),
        }

        if _XGBOOST_AVAILABLE:
            registry["XGBoost"] = (
                XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    eval_metric="logloss",
                    random_state=self.random_state,
                    n_jobs=-1,
                ),
                False,
            )
        else:
            logger.warning(
                "Skipping 'XGBoost' in model comparison because the "
                "xgboost package is not installed."
            )

        return registry

    def _build_pipeline(self, estimator: object, requires_scaling: bool) -> Pipeline:
        """Wrap an estimator in a scikit-learn ``Pipeline``, adding a
        ``StandardScaler`` step only for estimators that need it.

        Args:
            estimator: The unfitted classifier instance.
            requires_scaling: Whether to prepend a ``StandardScaler``.

        Returns:
            An unfitted ``Pipeline``.
        """
        steps: List[Tuple[str, object]] = []
        if requires_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", estimator))
        return Pipeline(steps)

    def prepare_data(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Build the classification target and split into stratified
        train/test sets.

        Args:
            df: Feature-engineered dataframe (output of
                ``feature_engineering.FeatureEngineer.engineer_all_features``).

        Returns:
            A tuple ``(X_train, X_test, y_train, y_test)``.

        Raises:
            ModelTrainingError: If required feature columns are missing,
                or the dataset is too small to split.
        """
        missing_features = [c for c in self.feature_columns if c not in df.columns]
        if missing_features:
            raise ModelTrainingError(
                f"Input dataframe is missing required feature column(s): "
                f"{missing_features}. Run feature_engineering first."
            )

        if self.target_builder.source_column in self.feature_columns:
            raise ModelTrainingError(
                "Data leakage guard triggered: the classification target's "
                f"source column ('{self.target_builder.source_column}') is "
                "also present in feature_columns. Including it would let "
                "the model trivially reconstruct the label instead of "
                "learning genuine disparity signal. Remove it from "
                "feature_columns (see config.ML_FEATURE_COLUMNS)."
            )

        labeled_df, threshold = self.target_builder.build_target(df)

        X = labeled_df[self.feature_columns].copy()
        y = labeled_df[self.target_builder.target_column].copy()

        # Any residual NaNs in features (e.g. unresolved district matches)
        # are median-imputed here rather than silently propagated into
        # model training, which several scikit-learn estimators cannot
        # handle natively.
        na_counts = X.isna().sum()
        columns_with_na = na_counts[na_counts > 0]
        if not columns_with_na.empty:
            logger.warning(
                "Imputing missing values (median) in feature columns prior "
                "to training: %s",
                columns_with_na.to_dict(),
            )
            X = X.fillna(X.median(numeric_only=True))

        if len(X) < 20:
            raise ModelTrainingError(
                f"Dataset too small to train/test split reliably: only "
                f"{len(X)} rows available (minimum 20 required)."
            )

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )
        logger.info(
            "Train/test split complete: train=%d rows, test=%d rows "
            "(test_size=%.2f), target threshold=%.4f",
            len(X_train),
            len(X_test),
            self.test_size,
            threshold,
        )
        return X_train, X_test, y_train, y_test

    def train_all_models(
        self, X_train: pd.DataFrame, y_train: pd.Series
    ) -> List[TrainedModelResult]:
        """Fit and cross-validate every candidate model on the training
        fold.

        Args:
            X_train: Training feature matrix.
            y_train: Training labels.

        Returns:
            A list of ``TrainedModelResult`` (one per successfully trained
            candidate), each holding the fitted pipeline and its
            cross-validation score.

        Raises:
            ModelTrainingError: If every candidate model fails to train.
        """
        registry = self._build_model_registry()
        cv = StratifiedKFold(
            n_splits=self.cv_folds, shuffle=True, random_state=self.random_state
        )

        results: List[TrainedModelResult] = []
        for model_name, (estimator, requires_scaling) in registry.items():
            try:
                pipeline = self._build_pipeline(estimator, requires_scaling)

                with warnings.catch_warnings():
                    # Convergence warnings from Logistic Regression on
                    # small/imbalanced folds are expected and non-fatal;
                    # suppress them from cluttering pipeline logs while
                    # still logging our own summary below.
                    warnings.simplefilter("ignore")
                    cv_scores = cross_val_score(
                        pipeline,
                        X_train,
                        y_train,
                        cv=cv,
                        scoring=self.primary_metric,
                        n_jobs=-1,
                    )
                    pipeline.fit(X_train, y_train)

                result = TrainedModelResult(
                    model_name=model_name,
                    pipeline=pipeline,
                    cv_mean_score=float(np.mean(cv_scores)),
                    cv_std_score=float(np.std(cv_scores)),
                    requires_scaling=requires_scaling,
                )
                results.append(result)
                logger.info(
                    "Trained '%s': CV %s = %.4f (+/- %.4f)",
                    model_name,
                    self.primary_metric,
                    result.cv_mean_score,
                    result.cv_std_score,
                )
            except Exception as exc:  # noqa: BLE001 - isolate one bad model from the rest
                logger.error(
                    "Training failed for candidate model '%s': %s. "
                    "Excluding it from comparison.",
                    model_name,
                    exc,
                )

        if not results:
            raise ModelTrainingError(
                "All candidate models failed to train. Inspect the logs "
                "above for per-model errors."
            )
        return results

    def build_comparison_table(self, results: List[TrainedModelResult]) -> pd.DataFrame:
        """Build a leaderboard dataframe from trained candidate results.

        Args:
            results: Trained candidate results (with or without
                ``test_metrics`` populated).

        Returns:
            A dataframe sorted best-first by
            ``cv_mean_{primary_metric}``, one row per model.
        """
        rows = []
        for result in results:
            row = {
                "model_name": result.model_name,
                f"cv_mean_{self.primary_metric}": round(result.cv_mean_score, 4),
                f"cv_std_{self.primary_metric}": round(result.cv_std_score, 4),
            }
            row.update(
                {f"test_{k}": round(v, 4) for k, v in result.test_metrics.items()}
            )
            rows.append(row)

        table = pd.DataFrame(rows).sort_values(
            by=f"cv_mean_{self.primary_metric}", ascending=False
        ).reset_index(drop=True)
        table.insert(0, "rank", range(1, len(table) + 1))
        return table

    def select_best_model(
        self, results: List[TrainedModelResult]
    ) -> TrainedModelResult:
        """Select the candidate with the highest cross-validation score on
        the primary metric.

        Args:
            results: Trained candidate results.

        Returns:
            The best-performing ``TrainedModelResult``.

        Raises:
            ModelTrainingError: If ``results`` is empty.
        """
        if not results:
            raise ModelTrainingError("Cannot select a best model from an empty result list.")
        best = max(results, key=lambda r: r.cv_mean_score)
        logger.info(
            "Selected best model: '%s' with CV %s = %.4f",
            best.model_name,
            self.primary_metric,
            best.cv_mean_score,
        )
        return best

    def save_best_model(
        self,
        artifacts: TrainingArtifacts,
        path: Optional[Path] = None,
    ) -> Path:
        """Persist the best model, its feature list, and metadata to disk
        via ``joblib``.

        Args:
            artifacts: Training artifacts produced by
                ``run_training_pipeline``.
            path: Destination file path. Defaults to
                ``config.MODEL_DIR / config.ML_BEST_MODEL_FILENAME``.

        Returns:
            The path the model bundle was saved to.
        """
        destination = path or (config.MODEL_DIR / config.ML_BEST_MODEL_FILENAME)
        ensure_dir(destination.parent)

        bundle = {
            "pipeline": artifacts.best_result.pipeline,
            "model_name": artifacts.best_result.model_name,
            "feature_names": artifacts.feature_names,
            "requires_scaling": artifacts.best_result.requires_scaling,
            "cv_mean_score": artifacts.best_result.cv_mean_score,
            "test_metrics": artifacts.best_result.test_metrics,
            "primary_metric": self.primary_metric,
            "class_names": config.ML_CLASS_NAMES,
            "target_column": self.target_builder.target_column,
            "random_state": self.random_state,
        }
        joblib.dump(bundle, destination)
        logger.info(
            "Saved best model ('%s') bundle to %s",
            artifacts.best_result.model_name,
            destination,
        )
        return destination

    def run_training_pipeline(self, df: pd.DataFrame) -> TrainingArtifacts:
        """Execute the full target-construction -> split -> train-all ->
        compare -> select-best pipeline.

        Args:
            df: Feature-engineered dataframe.

        Returns:
            A fully populated ``TrainingArtifacts`` instance, ready to be
            passed to ``model_evaluation.py`` (to populate
            ``test_metrics`` and generate plots) and
            ``shap_explainability.py``.

        Raises:
            ModelTrainingError: Propagated from any pipeline stage.
        """
        logger.info("=== Model training pipeline: START ===")
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        results = self.train_all_models(X_train, y_train)
        best_result = self.select_best_model(results)
        comparison_table = self.build_comparison_table(results)

        logger.info(
            "=== Model training pipeline: COMPLETE. %d model(s) trained, "
            "best='%s' ===",
            len(results),
            best_result.model_name,
        )

        return TrainingArtifacts(
            results=results,
            best_result=best_result,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=self.feature_columns,
            comparison_table=comparison_table,
        )


def load_model_bundle(path: Optional[Path] = None) -> Dict[str, object]:
    """Load a persisted model bundle saved by
    ``ModelTrainer.save_best_model``.

    Args:
        path: Path to the saved bundle. Defaults to
            ``config.MODEL_DIR / config.ML_BEST_MODEL_FILENAME``.

    Returns:
        The deserialized bundle dict (``pipeline``, ``model_name``,
        ``feature_names``, etc.).

    Raises:
        ModelTrainingError: If the file does not exist or fails to load.
    """
    source = path or (config.MODEL_DIR / config.ML_BEST_MODEL_FILENAME)
    if not source.exists():
        raise ModelTrainingError(f"No saved model bundle found at {source}")
    try:
        bundle = joblib.load(source)
    except Exception as exc:  # noqa: BLE001
        raise ModelTrainingError(f"Failed to load model bundle from {source}: {exc}") from exc
    logger.info("Loaded model bundle ('%s') from %s", bundle.get("model_name"), source)
    return bundle
