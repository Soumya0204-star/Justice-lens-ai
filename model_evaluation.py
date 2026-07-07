"""
model_evaluation.py
=====================
Model evaluation layer for JusticeLens AI.

Given the ``TrainingArtifacts`` produced by
``model_training.ModelTrainer.run_training_pipeline``, this module:

    * Computes a standard classification metric suite (accuracy,
      precision, recall, F1, ROC-AUC) on the held-out test set for every
      trained candidate model.
    * Populates each candidate's ``test_metrics`` and rebuilds the
      leaderboard/comparison table to include test-set performance
      alongside cross-validation performance.
    * Generates and saves a confusion matrix plot and an ROC curve plot
      (with AUC annotation) for the best-selected model.
    * Saves the final comparison leaderboard to CSV for reporting.

All plots are saved as PNG files under ``config.PLOTS_DIR`` and the paths
are returned to the caller rather than displayed inline, so this module
works identically in a batch script, a notebook, or inside the Streamlit
dashboard (which can simply ``st.image(path)`` the result).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend: safe for headless/batch execution
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from justicelens import config
from justicelens.logger import get_logger
from justicelens.model_training import TrainedModelResult, TrainingArtifacts
from justicelens.utils import ModelEvaluationError, ensure_dir

logger = get_logger(__name__)


def _require_matplotlib() -> None:
    if plt is None:
        raise ModelEvaluationError(
            "matplotlib is required for plot generation but is not installed"
        )


class ModelEvaluator:
    """Computes evaluation metrics and generates diagnostic plots for
    trained classification models.

    Typical usage::

        evaluator = ModelEvaluator()
        artifacts = evaluator.evaluate_all_models(artifacts)
        cm_path = evaluator.generate_confusion_matrix_plot(
            artifacts.best_result, artifacts.X_test, artifacts.y_test
        )
        roc_path = evaluator.generate_roc_curve_plot(
            artifacts.best_result, artifacts.X_test, artifacts.y_test
        )
    """

    def __init__(
        self,
        class_names: Optional[List[str]] = None,
        plots_dir: Optional[Path] = None,
    ) -> None:
        """Initialize the evaluator.

        Args:
            class_names: Human-readable class labels, index-aligned with
                the numeric target (index 0, index 1). Defaults to
                ``config.ML_CLASS_NAMES``.
            plots_dir: Directory plots are saved to. Defaults to
                ``config.PLOTS_DIR``.
        """
        self.class_names = class_names or list(config.ML_CLASS_NAMES)
        self.plots_dir = ensure_dir(plots_dir or config.PLOTS_DIR)

    def compute_metrics(
        self, model: object, X_test: pd.DataFrame, y_test: pd.Series
    ) -> Dict[str, float]:
        """Compute the standard classification metric suite for a single
        fitted model on the held-out test set.

        Args:
            model: A fitted scikit-learn-compatible estimator/pipeline
                exposing ``predict`` and ``predict_proba``.
            X_test: Held-out test feature matrix.
            y_test: True held-out test labels.

        Returns:
            Dict with keys: ``accuracy``, ``precision``, ``recall``,
            ``f1``, ``roc_auc``.

        Raises:
            ModelEvaluationError: If the model cannot produce predictions
                or probability estimates on ``X_test``.
        """
        try:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
        except Exception as exc:  # noqa: BLE001
            raise ModelEvaluationError(
                f"Failed to generate predictions for evaluation: {exc}"
            ) from exc

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
        }
        return metrics

    def evaluate_all_models(self, artifacts: TrainingArtifacts) -> TrainingArtifacts:
        """Compute test-set metrics for every trained candidate model and
        refresh the leaderboard/comparison table to include them.

        Mutates and returns the same ``artifacts`` object for convenience
        (each ``TrainedModelResult.test_metrics`` dict is populated
        in-place).

        Args:
            artifacts: Training artifacts from
                ``ModelTrainer.run_training_pipeline``.

        Returns:
            The same ``artifacts`` instance, with ``test_metrics``
            populated on every result and ``comparison_table`` rebuilt to
            include test-set columns.
        """
        logger.info("Evaluating %d trained model(s) on held-out test set", len(artifacts.results))

        for result in artifacts.results:
            result.test_metrics = self.compute_metrics(
                result.pipeline, artifacts.X_test, artifacts.y_test
            )
            logger.info(
                "'%s' test metrics: %s",
                result.model_name,
                {k: round(v, 4) for k, v in result.test_metrics.items()},
            )

        rows = []
        primary_metric_key = None
        for result in artifacts.results:
            row = {
                "model_name": result.model_name,
                "cv_mean_score": round(result.cv_mean_score, 4),
                "cv_std_score": round(result.cv_std_score, 4),
            }
            row.update({f"test_{k}": round(v, 4) for k, v in result.test_metrics.items()})
            rows.append(row)

        artifacts.comparison_table = (
            pd.DataFrame(rows)
            .sort_values(by="cv_mean_score", ascending=False)
            .reset_index(drop=True)
        )
        artifacts.comparison_table.insert(
            0, "rank", range(1, len(artifacts.comparison_table) + 1)
        )

        logger.info(
            "Model comparison leaderboard:\n%s",
            artifacts.comparison_table.to_string(index=False),
        )
        return artifacts

    def save_comparison_report(
        self, artifacts: TrainingArtifacts, path: Optional[Path] = None
    ) -> Path:
        """Save the model comparison leaderboard to a CSV file.

        Args:
            artifacts: Training artifacts with a populated
                ``comparison_table``.
            path: Destination CSV path. Defaults to
                ``config.MODEL_DIR / config.ML_COMPARISON_REPORT_FILENAME``.

        Returns:
            The path the report was written to.
        """
        destination = path or (
            config.MODEL_DIR / config.ML_COMPARISON_REPORT_FILENAME
        )
        ensure_dir(destination.parent)
        artifacts.comparison_table.to_csv(destination, index=False)
        logger.info("Saved model comparison report to %s", destination)
        return destination

    def generate_confusion_matrix_plot(
        self,
        result: TrainedModelResult,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate and save a confusion matrix heatmap for a fitted
        model.

        Args:
            result: The trained model result to visualize.
            X_test: Held-out test feature matrix.
            y_test: True held-out test labels.
            filename: Optional output filename (without directory).
                Defaults to ``confusion_matrix_{model_name}.png``.

        Returns:
            Path to the saved PNG file.

        Raises:
            ModelEvaluationError: If prediction or plotting fails.
        """
        _require_matplotlib()
        try:
            y_pred = result.pipeline.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            fig, ax = plt.subplots(figsize=(6, 5))
            display = ConfusionMatrixDisplay(
                confusion_matrix=cm, display_labels=self.class_names
            )
            display.plot(ax=ax, cmap="Blues", colorbar=True, values_format="d")
            ax.set_title(f"Confusion Matrix -- {result.model_name}")
            fig.tight_layout()

            safe_name = result.model_name.lower().replace(" ", "_")
            out_filename = filename or f"confusion_matrix_{safe_name}.png"
            destination = self.plots_dir / out_filename
            fig.savefig(destination, dpi=150)
            plt.close(fig)

            logger.info("Saved confusion matrix plot to %s", destination)
            return destination
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            raise ModelEvaluationError(
                f"Failed to generate confusion matrix for "
                f"'{result.model_name}': {exc}"
            ) from exc

    def generate_roc_curve_plot(
        self,
        result: TrainedModelResult,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate and save an ROC curve plot (with AUC annotation) for a
        fitted model.

        Args:
            result: The trained model result to visualize.
            X_test: Held-out test feature matrix.
            y_test: True held-out test labels.
            filename: Optional output filename (without directory).
                Defaults to ``roc_curve_{model_name}.png``.

        Returns:
            Path to the saved PNG file.

        Raises:
            ModelEvaluationError: If prediction or plotting fails.
        """
        _require_matplotlib()
        try:
            y_proba = result.pipeline.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            auc_score = roc_auc_score(y_test, y_proba)

            fig, ax = plt.subplots(figsize=(6, 5))
            RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=auc_score).plot(
                ax=ax, name=result.model_name
            )
            ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
            ax.set_title(f"ROC Curve -- {result.model_name} (AUC = {auc_score:.3f})")
            ax.legend(loc="lower right")
            fig.tight_layout()

            safe_name = result.model_name.lower().replace(" ", "_")
            out_filename = filename or f"roc_curve_{safe_name}.png"
            destination = self.plots_dir / out_filename
            fig.savefig(destination, dpi=150)
            plt.close(fig)

            logger.info("Saved ROC curve plot to %s", destination)
            return destination
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            raise ModelEvaluationError(
                f"Failed to generate ROC curve for '{result.model_name}': {exc}"
            ) from exc

    def generate_model_comparison_plot(
        self, artifacts: TrainingArtifacts, filename: str = "model_comparison.png"
    ) -> Path:
        """Generate and save a horizontal bar chart comparing every
        candidate model's test-set ROC-AUC (or the configured primary
        metric, if test metrics use a different key).

        Args:
            artifacts: Training artifacts with populated ``test_metrics``
                on every result.
            filename: Output filename (without directory).

        Returns:
            Path to the saved PNG file.
        """
        _require_matplotlib()
        metric_key = "roc_auc" if "roc_auc" in artifacts.results[0].test_metrics else next(
            iter(artifacts.results[0].test_metrics), None
        )
        if metric_key is None:
            raise ModelEvaluationError(
                "Cannot plot model comparison: no test metrics have been "
                "computed. Call evaluate_all_models first."
            )

        names = [r.model_name for r in artifacts.results]
        scores = [r.test_metrics[metric_key] for r in artifacts.results]
        order = np.argsort(scores)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(
            [names[i] for i in order],
            [scores[i] for i in order],
            color="#2E5EAA",
        )
        ax.set_xlabel(metric_key.replace("_", " ").upper())
        ax.set_title("Model Comparison -- Test Set Performance")
        ax.set_xlim(0, 1)
        for i, v in enumerate([scores[j] for j in order]):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center")
        fig.tight_layout()

        destination = self.plots_dir / filename
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        logger.info("Saved model comparison plot to %s", destination)
        return destination

    def run_full_evaluation(
        self, artifacts: TrainingArtifacts
    ) -> Dict[str, Path]:
        """Convenience method running the complete evaluation sequence:
        compute test metrics for all models, save the comparison report,
        and generate confusion matrix / ROC curve / comparison plots for
        the best model.

        Args:
            artifacts: Training artifacts from
                ``ModelTrainer.run_training_pipeline``.

        Returns:
            Dict mapping artifact name -> saved file path, with keys:
            ``comparison_report``, ``confusion_matrix``, ``roc_curve``,
            ``model_comparison_plot``.
        """
        artifacts = self.evaluate_all_models(artifacts)
        comparison_report_path = self.save_comparison_report(artifacts)
        confusion_matrix_path = self.generate_confusion_matrix_plot(
            artifacts.best_result, artifacts.X_test, artifacts.y_test
        )
        roc_curve_path = self.generate_roc_curve_plot(
            artifacts.best_result, artifacts.X_test, artifacts.y_test
        )
        comparison_plot_path = self.generate_model_comparison_plot(artifacts)

        return {
            "comparison_report": comparison_report_path,
            "confusion_matrix": confusion_matrix_path,
            "roc_curve": roc_curve_path,
            "model_comparison_plot": comparison_plot_path,
        }
