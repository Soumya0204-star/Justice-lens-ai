"""
shap_explainability.py
========================
SHAP (SHapley Additive exPlanations) explainability layer for JusticeLens
AI's disparity classification model.

This module is the "why" layer that sits on top of whichever model
``model_training.ModelTrainer`` selects as best: it never assumes a
specific model family, and instead picks the fastest *exact* SHAP
algorithm available for the actual fitted estimator, falling back to a
model-agnostic explainer for anything it does not specifically recognize
(e.g. Logistic Regression wrapped in a ``StandardScaler`` pipeline):

    * **Tree-based models** (Decision Tree, Random Forest, Gradient
      Boosting, XGBoost) -> ``shap.TreeExplainer`` (exact, fast).
    * **Everything else** (e.g. Logistic Regression + scaler pipeline) ->
      a generic ``shap.Explainer`` wrapping the full pipeline's
      ``predict_proba``, which is correct for any model at the cost of
      being slower (mitigated by subsampling the background set).

Produces the four explainability artifacts required by the internship
architecture:
    1. SHAP summary plot (global feature impact across many predictions).
    2. SHAP waterfall plot (local explanation for one specific
       district-year prediction).
    3. Feature importance ranking (mean absolute SHAP value per feature).
    4. Structured, human-readable prediction explanation (the payload
       that would be handed to the watsonx.ai/Granite narration layer in
       the full architecture).

The ``shap`` package is an optional dependency: if it is not installed,
every public method raises a clear ``ExplainabilityError`` naming the
missing package rather than failing with an opaque ``ImportError`` deep in
the call stack.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Union

try:
    import matplotlib

    matplotlib.use("Agg")  # non-interactive backend: safe for headless/batch execution
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from justicelens import config
from justicelens.logger import get_logger
from justicelens.utils import ExplainabilityError, ensure_dir

logger = get_logger(__name__)


def _require_matplotlib() -> None:
    if plt is None:
        raise ExplainabilityError(
            "matplotlib is required for SHAP plot generation but is not installed"
        )

try:
    import shap

    _SHAP_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without shap installed
    _SHAP_AVAILABLE = False
    logger.warning(
        "The 'shap' package is not installed in this environment. "
        "SHAP explainability methods will raise ExplainabilityError until "
        "it is installed via `pip install shap` (see requirements.txt)."
    )

#: Estimator classes recognized as tree-based, for which the exact,
#: fast ``shap.TreeExplainer`` can be used directly.
_TREE_BASED_CLASSES = (DecisionTreeClassifier, RandomForestClassifier, GradientBoostingClassifier)

try:  # XGBoost's classifier class is also tree-based when available.
    from xgboost import XGBClassifier

    _TREE_BASED_CLASSES = _TREE_BASED_CLASSES + (XGBClassifier,)
except ImportError:  # pragma: no cover
    pass


def _require_shap() -> None:
    """Raise a clear error if the optional ``shap`` dependency is absent.

    Raises:
        ExplainabilityError: Always, when ``shap`` failed to import.
    """
    if not _SHAP_AVAILABLE:
        raise ExplainabilityError(
            "SHAP explainability requires the 'shap' package, which is not "
            "installed in this environment. Install it with "
            "`pip install shap` (already listed in requirements.txt) and "
            "retry."
        )


def _unwrap_classifier(pipeline: Union[Pipeline, object]) -> object:
    """Extract the final estimator from a scikit-learn ``Pipeline``, or
    return the object unchanged if it is not a ``Pipeline``.

    Args:
        pipeline: A fitted ``Pipeline`` (typically ``[scaler?, classifier]``)
            or a bare fitted estimator.

    Returns:
        The fitted classifier/estimator itself.
    """
    if isinstance(pipeline, Pipeline):
        return pipeline.named_steps["classifier"]
    return pipeline


def _pipeline_has_scaler(pipeline: Union[Pipeline, object]) -> bool:
    """Check whether a pipeline includes a preprocessing (scaling) step.

    Args:
        pipeline: A fitted ``Pipeline`` or bare estimator.

    Returns:
        ``True`` if a ``"scaler"`` step is present.
    """
    return isinstance(pipeline, Pipeline) and "scaler" in pipeline.named_steps


class SHAPExplainer:
    """Generates global and local SHAP explanations for a trained
    classification model bundle (as produced by
    ``model_training.ModelTrainer.save_best_model`` /
    ``model_training.load_model_bundle``).

    Typical usage::

        explainer = SHAPExplainer(model_bundle, background_data=X_train)
        summary_path = explainer.generate_summary_plot(X_test)
        waterfall_path = explainer.generate_waterfall_plot(X_test, instance_index=0)
        importance_df = explainer.compute_feature_importance(X_test)
        explanation = explainer.explain_prediction(X_test, instance_index=0)
    """

    def __init__(
        self,
        model_bundle: Dict[str, object],
        background_data: pd.DataFrame,
        plots_dir: Optional[Path] = None,
        max_background_samples: int = config.SHAP_MAX_SAMPLES,
    ) -> None:
        """Initialize the SHAP explainer for a specific trained model.

        Args:
            model_bundle: Bundle dict as produced by
                ``model_training.ModelTrainer.save_best_model`` /
                ``load_model_bundle``, containing at minimum ``pipeline``
                and ``feature_names``.
            background_data: Representative feature data (typically the
                training set) used as the SHAP background/reference
                distribution. Subsampled to ``max_background_samples`` for
                performance.
            plots_dir: Directory plots are saved to. Defaults to
                ``config.PLOTS_DIR``.
            max_background_samples: Maximum number of rows drawn from
                ``background_data`` for background/summary computations.

        Raises:
            ExplainabilityError: If ``shap`` is not installed, or if
                required bundle keys are missing.
        """
        _require_shap()

        if "pipeline" not in model_bundle or "feature_names" not in model_bundle:
            raise ExplainabilityError(
                "model_bundle must contain 'pipeline' and 'feature_names' "
                f"keys; got keys: {list(model_bundle.keys())}"
            )

        self.pipeline = model_bundle["pipeline"]
        self.feature_names: List[str] = list(model_bundle["feature_names"])
        self.model_name: str = str(model_bundle.get("model_name", "model"))
        self.plots_dir = ensure_dir(plots_dir or config.PLOTS_DIR)
        self.max_background_samples = max_background_samples

        n_samples = min(len(background_data), max_background_samples)
        self.background_data = background_data[self.feature_names].sample(
            n=n_samples, random_state=config.RANDOM_SEED
        )

        self._explainer = self._build_explainer()
        self._uses_tree_explainer = isinstance(
            _unwrap_classifier(self.pipeline), _TREE_BASED_CLASSES
        ) and not _pipeline_has_scaler(self.pipeline)

        logger.info(
            "Initialized SHAPExplainer for model '%s' using %s (background "
            "size=%d)",
            self.model_name,
            "TreeExplainer" if self._uses_tree_explainer else "generic Explainer",
            n_samples,
        )

    def _build_explainer(self):
        """Construct the appropriate SHAP explainer for the wrapped model.

        Returns:
            A fitted ``shap.TreeExplainer`` when the underlying estimator
            is tree-based and unscaled, otherwise a model-agnostic
            ``shap.Explainer`` wrapping the full pipeline's
            ``predict_proba``.

        Raises:
            ExplainabilityError: If explainer construction fails.
        """
        classifier = _unwrap_classifier(self.pipeline)
        try:
            if isinstance(classifier, _TREE_BASED_CLASSES) and not _pipeline_has_scaler(
                self.pipeline
            ):
                return shap.TreeExplainer(classifier)

            # Generic, model-agnostic fallback: explain the whole pipeline
            # (including any scaling) as a black box via its predict_proba.
            return shap.Explainer(
                lambda data: self.pipeline.predict_proba(
                    pd.DataFrame(data, columns=self.feature_names)
                )[:, 1],
                self.background_data,
            )
        except Exception as exc:  # noqa: BLE001
            raise ExplainabilityError(
                f"Failed to construct a SHAP explainer for model "
                f"'{self.model_name}': {exc}"
            ) from exc

    def _compute_shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for the positive class ("Underserved") for
        every row in ``X``, normalizing across the different array shapes
        the ``shap`` library can return depending on explainer type and
        version.

        Args:
            X: Feature dataframe (must contain ``self.feature_names``).

        Returns:
            A 2D numpy array of shape ``(n_rows, n_features)`` containing
            positive-class SHAP values.

        Raises:
            ExplainabilityError: If SHAP value computation fails or
                returns an unrecognized shape.
        """
        ordered_X = X[self.feature_names]
        try:
            if self._uses_tree_explainer:
                raw = self._explainer.shap_values(ordered_X)
            else:
                raw = self._explainer(ordered_X).values
        except Exception as exc:  # noqa: BLE001
            raise ExplainabilityError(
                f"SHAP value computation failed for model "
                f"'{self.model_name}': {exc}"
            ) from exc

        return self._normalize_shap_output(raw, n_rows=len(ordered_X))

    def _normalize_shap_output(self, raw: object, n_rows: int) -> np.ndarray:
        """Normalize the many possible ``shap`` return shapes into a
        single, consistent ``(n_rows, n_features)`` array of positive
        -class contributions.

        Handles:
            * A list of per-class arrays (older TreeExplainer API for
              binary classifiers): ``[class_0_array, class_1_array]``.
            * A single 3D array ``(n_rows, n_features, n_classes)``
              (newer TreeExplainer API).
            * A single 2D array ``(n_rows, n_features)`` (already
              positive-class-only, as returned by the generic
              ``Explainer`` wrapping ``predict_proba``).

        Args:
            raw: The raw object returned by the SHAP explainer.
            n_rows: Expected number of rows, used to validate the result.

        Returns:
            A 2D numpy array, shape ``(n_rows, n_features)``.

        Raises:
            ExplainabilityError: If the shape cannot be normalized.
        """
        if isinstance(raw, list):
            # Binary classification: index 1 is the positive ("Underserved") class.
            array = np.asarray(raw[1] if len(raw) > 1 else raw[0])
        else:
            array = np.asarray(raw)
            if array.ndim == 3:
                # (n_rows, n_features, n_classes) -> take positive class.
                array = array[:, :, 1] if array.shape[2] > 1 else array[:, :, 0]

        if array.ndim != 2 or array.shape[0] != n_rows:
            raise ExplainabilityError(
                f"Unexpected SHAP output shape {array.shape}; expected "
                f"(n_rows={n_rows}, n_features={len(self.feature_names)})."
            )
        return array

    def compute_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Compute global feature importance as the mean absolute SHAP
        value per feature across all rows of ``X``.

        Args:
            X: Feature dataframe to compute importance over (typically the
                test set).

        Returns:
            A dataframe with columns ``feature`` and
            ``mean_abs_shap_value``, sorted descending by importance.
        """
        shap_values = self._compute_shap_values(X)
        importance = pd.DataFrame(
            {
                "feature": self.feature_names,
                "mean_abs_shap_value": np.abs(shap_values).mean(axis=0),
            }
        ).sort_values("mean_abs_shap_value", ascending=False).reset_index(drop=True)
        return importance

    def generate_feature_importance_plot(
        self, X: pd.DataFrame, filename: str = "shap_feature_importance.png"
    ) -> Path:
        """Generate and save a horizontal bar chart of global feature
        importance (mean absolute SHAP value).

        Args:
            X: Feature dataframe to compute importance over.
            filename: Output filename (without directory).

        Returns:
            Path to the saved PNG file.
        """
        _require_matplotlib()
        importance_df = self.compute_feature_importance(X)

        fig, ax = plt.subplots(figsize=(7, 5))
        ordered = importance_df.sort_values("mean_abs_shap_value")
        ax.barh(ordered["feature"], ordered["mean_abs_shap_value"], color="#C0392B")
        ax.set_xlabel("Mean |SHAP value| (impact on model output)")
        ax.set_title(f"Global Feature Importance -- {self.model_name}")
        fig.tight_layout()

        destination = self.plots_dir / filename
        fig.savefig(destination, dpi=150)
        plt.close(fig)
        logger.info("Saved SHAP feature importance plot to %s", destination)
        return destination

    def generate_summary_plot(
        self, X: pd.DataFrame, filename: str = "shap_summary_plot.png"
    ) -> Path:
        """Generate and save the standard SHAP summary (beeswarm) plot,
        showing each feature's SHAP value distribution across all rows of
        ``X`` and how feature value (color) relates to impact direction.

        Args:
            X: Feature dataframe to summarize (typically the test set, or
                a representative sample of it).
            filename: Output filename (without directory).

        Returns:
            Path to the saved PNG file.

        Raises:
            ExplainabilityError: If plot generation fails.
        """
        _require_matplotlib()
        n_samples = min(len(X), self.max_background_samples)
        X_sample = X[self.feature_names].sample(n=n_samples, random_state=config.RANDOM_SEED)
        shap_values = self._compute_shap_values(X_sample)

        try:
            fig = plt.figure(figsize=(8, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_names,
                show=False,
                plot_size=None,
            )
            fig = plt.gcf()
            fig.suptitle(f"SHAP Summary -- {self.model_name}", y=1.02)
            fig.tight_layout()

            destination = self.plots_dir / filename
            fig.savefig(destination, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("Saved SHAP summary plot to %s", destination)
            return destination
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            raise ExplainabilityError(f"Failed to generate SHAP summary plot: {exc}") from exc

    def generate_waterfall_plot(
        self,
        X: pd.DataFrame,
        instance_index: int = 0,
        filename: Optional[str] = None,
    ) -> Path:
        """Generate and save a SHAP waterfall plot explaining a single
        prediction: how each feature pushed the model's output away from
        the background (expected) value toward the final predicted
        probability for that specific district-year record.

        Args:
            X: Feature dataframe containing the instance to explain.
            instance_index: Positional (``.iloc``) index of the row within
                ``X`` to explain.
            filename: Optional output filename (without directory).
                Defaults to ``shap_waterfall_row_{instance_index}.png``.

        Returns:
            Path to the saved PNG file.

        Raises:
            ExplainabilityError: If ``instance_index`` is out of range or
                plot generation fails.
        """
        _require_matplotlib()
        if not 0 <= instance_index < len(X):
            raise ExplainabilityError(
                f"instance_index {instance_index} out of range for X with "
                f"{len(X)} rows."
            )

        instance = X[self.feature_names].iloc[[instance_index]]
        shap_values = self._compute_shap_values(instance)

        base_value = self._get_base_value()

        try:
            explanation = shap.Explanation(
                values=shap_values[0],
                base_values=base_value,
                data=instance.iloc[0].to_numpy(),
                feature_names=self.feature_names,
            )

            fig = plt.figure(figsize=(8, 6))
            shap.plots.waterfall(explanation, show=False)
            fig = plt.gcf()
            fig.suptitle(
                f"SHAP Waterfall -- {self.model_name} (row {instance_index})",
                y=1.02,
            )
            fig.tight_layout()

            out_filename = filename or f"shap_waterfall_row_{instance_index}.png"
            destination = self.plots_dir / out_filename
            fig.savefig(destination, dpi=150, bbox_inches="tight")
            plt.close(fig)

            logger.info("Saved SHAP waterfall plot to %s", destination)
            return destination
        except Exception as exc:  # noqa: BLE001
            plt.close("all")
            raise ExplainabilityError(
                f"Failed to generate SHAP waterfall plot for row "
                f"{instance_index}: {exc}"
            ) from exc

    def _get_base_value(self) -> float:
        """Retrieve the SHAP explainer's expected (base) value for the
        positive class, normalizing across explainer types.

        Returns:
            A scalar float representing the model's average output over
            the background distribution.
        """
        expected_value = getattr(self._explainer, "expected_value", 0.0)
        if isinstance(expected_value, (list, np.ndarray)):
            arr = np.asarray(expected_value)
            return float(arr[1] if arr.shape[0] > 1 else arr[0])
        return float(expected_value)

    def explain_prediction(
        self,
        X: pd.DataFrame,
        instance_index: int = 0,
        top_n: int = 5,
    ) -> Dict[str, object]:
        """Produce a structured, human-readable explanation for a single
        prediction, combining the model's predicted probability with the
        top contributing SHAP features and their direction of influence.

        This structured dict is designed to be the exact payload handed to
        the watsonx.ai/Granite narration layer in the full JusticeLens AI
        architecture (the generative layer narrates these numbers; it
        never invents them).

        Args:
            X: Feature dataframe containing the instance to explain.
            instance_index: Positional (``.iloc``) index of the row to
                explain.
            top_n: Number of top contributing features to include.

        Returns:
            A dict with keys:
                * ``predicted_class`` (str): human-readable predicted
                  label.
                * ``predicted_probability`` (float): predicted probability
                  of the "Underserved" class.
                * ``base_probability`` (float): the model's average
                  ("expected") predicted probability over the background
                  set.
                * ``top_contributing_features`` (list[dict]): each with
                  ``feature``, ``value``, ``shap_value``, and
                  ``direction`` ("increases" / "decreases" underserved
                  risk).
                * ``narrative_ready_summary`` (str): a plain-language
                  one-paragraph explanation built entirely from the above
                  structured values (no external LLM call -- this is the
                  deterministic, auditable fallback narrative referenced
                  in the system architecture's reliability requirements).

        Raises:
            ExplainabilityError: If ``instance_index`` is out of range or
                computation fails.
        """
        if not 0 <= instance_index < len(X):
            raise ExplainabilityError(
                f"instance_index {instance_index} out of range for X with "
                f"{len(X)} rows."
            )

        instance = X[self.feature_names].iloc[[instance_index]]
        shap_values = self._compute_shap_values(instance)[0]
        base_probability = self._get_base_value()
        predicted_probability = float(
            self.pipeline.predict_proba(instance)[0, 1]
        )
        predicted_class_index = int(self.pipeline.predict(instance)[0])
        predicted_class = config.ML_CLASS_NAMES[predicted_class_index]

        contributions = pd.DataFrame(
            {
                "feature": self.feature_names,
                "value": instance.iloc[0].to_numpy(),
                "shap_value": shap_values,
            }
        )
        contributions["abs_shap_value"] = contributions["shap_value"].abs()
        top_contributions = (
            contributions.sort_values("abs_shap_value", ascending=False)
            .head(top_n)
            .drop(columns="abs_shap_value")
        )

        top_features_list = []
        narrative_clauses = []
        for _, row in top_contributions.iterrows():
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            top_features_list.append(
                {
                    "feature": row["feature"],
                    "value": round(float(row["value"]), 4),
                    "shap_value": round(float(row["shap_value"]), 4),
                    "direction": f"{direction} underserved risk",
                }
            )
            narrative_clauses.append(
                f"{row['feature'].replace('_', ' ')} "
                f"(value={round(float(row['value']), 2)}) {direction} the "
                "underserved risk"
            )

        narrative_summary = (
            f"This district-year record is predicted as '{predicted_class}' "
            f"with a {predicted_probability:.1%} estimated probability of "
            f"being underserved, versus an average baseline of "
            f"{base_probability:.1%} across comparable districts. The "
            f"strongest drivers of this prediction are: "
            + "; ".join(narrative_clauses)
            + "."
        )

        return {
            "predicted_class": predicted_class,
            "predicted_probability": round(predicted_probability, 4),
            "base_probability": round(base_probability, 4),
            "top_contributing_features": top_features_list,
            "narrative_ready_summary": narrative_summary,
        }

    def generate_all_explanations(
        self,
        X: pd.DataFrame,
        instance_index: int = 0,
    ) -> Dict[str, object]:
        """Convenience method generating every required explainability
        artifact in one call: summary plot, waterfall plot, feature
        importance (plot + table), and a structured prediction
        explanation.

        Args:
            X: Feature dataframe (typically the test set).
            instance_index: Row to use for the waterfall plot and
                prediction explanation.

        Returns:
            Dict with keys: ``summary_plot_path``, ``waterfall_plot_path``,
            ``feature_importance_plot_path``, ``feature_importance_table``,
            ``prediction_explanation``.
        """
        summary_plot_path = self.generate_summary_plot(X)
        waterfall_plot_path = self.generate_waterfall_plot(X, instance_index=instance_index)
        feature_importance_plot_path = self.generate_feature_importance_plot(X)
        feature_importance_table = self.compute_feature_importance(X)
        prediction_explanation = self.explain_prediction(X, instance_index=instance_index)

        return {
            "summary_plot_path": summary_plot_path,
            "waterfall_plot_path": waterfall_plot_path,
            "feature_importance_plot_path": feature_importance_plot_path,
            "feature_importance_table": feature_importance_table,
            "prediction_explanation": prediction_explanation,
        }
