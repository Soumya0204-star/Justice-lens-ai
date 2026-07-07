"""
feature_engineering.py
========================
Feature engineering layer for JusticeLens AI.

Consumes the curated dataframe produced by
``data_engineering.DataEngineeringPipeline.run_pipeline`` and derives the
indicators that feed the disparity-scoring ML model (XGBoost regressor +
SHAP, per the finalized architecture):

    * ``cases_per_lakh_population`` -- Tele-Law cases registered per
      100,000 population; the standard Indian public-sector per-capita
      unit.
    * ``advice_enabled_ratio`` -- share of registered cases that actually
      received legal advice (advice_enabled_count / cases_registered).
    * ``yoy_growth_rate`` -- year-over-year change in cases_registered for
      the same district.
    * ``rural_penetration_index`` -- registration rate adjusted for the
      district's rural population share, surfacing districts where rural
      citizens may be under-reached relative to urban ones.
    * ``literacy_adjusted_expected_load`` -- a simple composite baseline
      combining population and literacy, used as an auxiliary regressor
      feature (NOT the final LADI score -- that combination happens in
      the modeling layer, out of scope for this module).

All computations use the safe-division helpers in ``utils.py`` so that
districts with zero population, zero cases, or missing demographic data
never crash the pipeline with divide-by-zero/inf/NaN propagation --
instead they are set to a sensible default and the affected rows are
logged.
"""

from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from justicelens import config
from justicelens.logger import get_logger
from justicelens.utils import FeatureEngineeringError, safe_divide_series

logger = get_logger(__name__)

#: Columns that must be present in the input dataframe for feature
#: engineering to run at all.
_REQUIRED_INPUT_COLUMNS: List[str] = [
    "state_name",
    "district_name",
    "fiscal_year",
    "cases_registered",
    "advice_enabled_count",
]


class FeatureEngineer:
    """Computes derived, model-ready indicators from the curated Tele-Law
    + demographic dataset.

    Typical usage::

        engineer = FeatureEngineer()
        features_df = engineer.engineer_all_features(curated_df)
    """

    def _validate_input(self, df: pd.DataFrame) -> None:
        """Ensure the input dataframe has the minimum columns required for
        feature engineering.

        Args:
            df: Input dataframe.

        Raises:
            FeatureEngineeringError: If any required input column is
                missing, or the dataframe is empty.
        """
        if df.empty:
            raise FeatureEngineeringError(
                "Cannot engineer features from an empty dataframe."
            )
        missing = [c for c in _REQUIRED_INPUT_COLUMNS if c not in df.columns]
        if missing:
            raise FeatureEngineeringError(
                f"Input dataframe is missing required column(s) for "
                f"feature engineering: {missing}"
            )

    def compute_per_capita_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ``cases_per_lakh_population``.

        Districts with missing or zero population receive a value of 0.0
        (rather than NaN/inf) and are logged for visibility, since a
        missing population figure usually indicates an unresolved district
        match from the reconciliation stage.

        Args:
            df: Dataframe containing ``cases_registered`` and
                ``population`` columns.

        Returns:
            A copy of ``df`` with the new ``cases_per_lakh_population``
            column added.
        """
        result = df.copy()
        if "population" not in result.columns:
            logger.warning(
                "'population' column absent; setting cases_per_lakh_population=0.0"
            )
            result["cases_per_lakh_population"] = 0.0
            return result

        missing_population = int(result["population"].isna().sum())
        if missing_population > 0:
            logger.warning(
                "%d row(s) missing 'population' (likely unresolved district "
                "match); cases_per_lakh_population set to 0.0 for these rows.",
                missing_population,
            )

        population_filled = result["population"].fillna(0)
        result["cases_per_lakh_population"] = safe_divide_series(
            result["cases_registered"] * config.PER_LAKH_POPULATION_UNIT,
            population_filled,
            default=0.0,
        )
        return result

    def compute_advice_enabled_ratio(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ``advice_enabled_ratio`` = advice_enabled_count /
        cases_registered, clipped to ``[0, 1]`` since values outside that
        range indicate a data-quality issue rather than a genuine ratio.

        Args:
            df: Dataframe containing ``advice_enabled_count`` and
                ``cases_registered`` columns.

        Returns:
            A copy of ``df`` with the new ``advice_enabled_ratio`` column
            added.
        """
        result = df.copy()
        ratio = safe_divide_series(
            result["advice_enabled_count"], result["cases_registered"], default=0.0
        )
        clipped = ratio.clip(lower=0.0, upper=1.0)
        n_clipped = int((ratio != clipped).sum())
        if n_clipped > 0:
            logger.warning(
                "%d row(s) had advice_enabled_ratio outside [0, 1] and were "
                "clipped; check for advice_enabled_count > cases_registered "
                "data-entry errors.",
                n_clipped,
            )
        result["advice_enabled_ratio"] = clipped
        return result

    def compute_yoy_growth_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute year-over-year growth rate of ``cases_registered`` for
        each district, based on the ordering of ``config.FISCAL_YEARS``.

        The first fiscal year available for a given district has no prior
        year to compare against, so its growth rate is set to ``0.0``
        (interpreted as "no observed change" rather than missing data,
        which keeps the feature numeric and model-ready).

        Args:
            df: Dataframe containing ``state_name``, ``district_name``,
                ``fiscal_year``, and ``cases_registered`` columns.

        Returns:
            A copy of ``df``, sorted by district and fiscal year, with the
            new ``yoy_growth_rate`` column added.
        """
        result = df.copy()

        fy_order = {fy: i for i, fy in enumerate(config.FISCAL_YEARS)}
        result["_fy_sort_key"] = result["fiscal_year"].map(fy_order)
        unmapped = int(result["_fy_sort_key"].isna().sum())
        if unmapped > 0:
            logger.warning(
                "%d row(s) have a fiscal_year value outside "
                "config.FISCAL_YEARS and will sort last for YoY computation.",
                unmapped,
            )
            max_known = max(fy_order.values(), default=0)
            result["_fy_sort_key"] = result["_fy_sort_key"].fillna(max_known + 1)

        result = result.sort_values(
            ["state_name", "district_name", "_fy_sort_key"]
        ).reset_index(drop=True)

        grouped = result.groupby(["state_name", "district_name"])["cases_registered"]
        previous_cases = grouped.shift(1)

        growth = safe_divide_series(
            result["cases_registered"] - previous_cases.fillna(result["cases_registered"]),
            previous_cases.fillna(1).replace(0, 1),
            default=0.0,
        )
        # First observation per district (no previous_cases) -> explicit 0.0
        growth = growth.where(previous_cases.notna(), 0.0)

        result["yoy_growth_rate"] = growth.round(4)
        result = result.drop(columns=["_fy_sort_key"])
        return result

    def compute_rural_penetration_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute ``rural_penetration_index``: the district's
        per-capita registration rate scaled inversely by its rural
        population share, so that a low index flags districts where high
        rurality coincides with low registration -- the core "regional
        disparity" signal the internship problem statement asks about.

        Formula::

            rural_penetration_index =
                cases_per_lakh_population * (1 - rural_population_pct / 100)
                if rural_population_pct is missing, index falls back to
                cases_per_lakh_population unadjusted (logged as a warning).

        Args:
            df: Dataframe containing ``cases_per_lakh_population`` (added
                by :meth:`compute_per_capita_metrics`, called
                automatically if absent) and optionally
                ``rural_population_pct``.

        Returns:
            A copy of ``df`` with the new ``rural_penetration_index``
            column added.
        """
        result = df.copy()
        if "cases_per_lakh_population" not in result.columns:
            result = self.compute_per_capita_metrics(result)

        if "rural_population_pct" not in result.columns:
            logger.warning(
                "'rural_population_pct' column absent; "
                "rural_penetration_index falls back to unadjusted "
                "cases_per_lakh_population."
            )
            result["rural_penetration_index"] = result["cases_per_lakh_population"]
            return result

        missing_rural = int(result["rural_population_pct"].isna().sum())
        if missing_rural > 0:
            logger.warning(
                "%d row(s) missing 'rural_population_pct'; treated as 0%% "
                "rural (no adjustment) for rural_penetration_index.",
                missing_rural,
            )

        rural_fraction = result["rural_population_pct"].fillna(0) / 100.0
        result["rural_penetration_index"] = (
            result["cases_per_lakh_population"] * (1.0 - rural_fraction)
        ).round(4)
        return result

    def compute_literacy_adjusted_expected_load(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute an auxiliary composite baseline feature,
        ``literacy_adjusted_expected_load``, combining population scale and
        literacy rate.

        This is deliberately a simple, transparent composite (log
        population scaled inversely by literacy) intended as one input
        *feature* to the downstream XGBoost expected-registration model --
        it is not itself the disparity score.

        Args:
            df: Dataframe containing ``population`` and
                ``literacy_rate_pct`` columns (missing values are handled
                gracefully).

        Returns:
            A copy of ``df`` with the new
            ``literacy_adjusted_expected_load`` column added.
        """
        result = df.copy()

        if "population" not in result.columns or "literacy_rate_pct" not in result.columns:
            logger.warning(
                "'population' and/or 'literacy_rate_pct' absent; "
                "literacy_adjusted_expected_load set to 0.0."
            )
            result["literacy_adjusted_expected_load"] = 0.0
            return result

        population_filled = result["population"].fillna(0).clip(lower=0)
        literacy_filled = result["literacy_rate_pct"].fillna(
            result["literacy_rate_pct"].median()
            if result["literacy_rate_pct"].notna().any()
            else 50.0
        )
        # log1p keeps large-population districts from dominating the scale;
        # literacy is inverted (100 - literacy) so lower literacy -> higher
        # expected latent need for assisted legal access.
        illiteracy_weight = (100.0 - literacy_filled.clip(lower=0, upper=100)) / 100.0
        result["literacy_adjusted_expected_load"] = (
            np.log1p(population_filled) * (0.5 + illiteracy_weight)
        ).round(4)
        return result

    def engineer_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run the full feature engineering sequence and return the
        model-ready dataframe.

        Args:
            df: Curated dataframe produced by
                ``data_engineering.DataEngineeringPipeline.run_pipeline``.

        Returns:
            A new dataframe containing all original columns plus:
            ``cases_per_lakh_population``, ``advice_enabled_ratio``,
            ``yoy_growth_rate``, ``rural_penetration_index``, and
            ``literacy_adjusted_expected_load``.

        Raises:
            FeatureEngineeringError: If the input dataframe is empty or
                missing required source columns.
        """
        self._validate_input(df)
        logger.info(
            "Starting feature engineering on dataframe with shape %s", df.shape
        )

        result = df.copy()
        result = self.compute_per_capita_metrics(result)
        result = self.compute_advice_enabled_ratio(result)
        result = self.compute_yoy_growth_rate(result)
        result = self.compute_rural_penetration_index(result)
        result = self.compute_literacy_adjusted_expected_load(result)

        new_columns = [
            "cases_per_lakh_population",
            "advice_enabled_ratio",
            "yoy_growth_rate",
            "rural_penetration_index",
            "literacy_adjusted_expected_load",
        ]
        logger.info(
            "Feature engineering complete. Added columns: %s. Final shape: %s",
            new_columns,
            result.shape,
        )
        return result
