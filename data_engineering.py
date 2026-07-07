"""
data_engineering.py
====================
Data engineering orchestration layer for JusticeLens AI.

Takes the raw dataframes produced by ``data_loader.py`` (which may be
either genuine OGD records or synthetic fallback data) and turns them into
a single clean, curated, analysis-ready dataframe by running, in order:

    1. **Schema harmonization** -- raw source column names vary release to
       release (e.g. "State" vs "State/UT" vs "state_ut"); this step
       normalizes column names and maps every observed variant onto the
       canonical schema defined in ``config.CANONICAL_COLUMNS``.
    2. **Cleaning** -- type coercion, whitespace/casing normalization on
       text fields, duplicate removal, and null handling.
    3. **Entity reconciliation** -- fuzzy-matches Tele-Law district names
       against the auxiliary demographic dataset's district names so the
       two sources can be joined even when spelling/formatting differs
       (a well-known pain point with data.gov.in district-level datasets).
    4. **Merge** -- left-joins the cleaned Tele-Law data with the
       reconciled auxiliary demographic data to produce the final curated
       table consumed by ``feature_engineering.py``.

Each step is exposed as an independently callable, independently testable
method as well as via the single ``run_pipeline`` convenience entrypoint.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from justicelens import config
from justicelens.data_validation import TeleLawDataValidator
from justicelens.logger import get_logger
from justicelens.utils import (
    DataEngineeringError,
    fuzzy_match_name,
    normalize_columns,
    normalize_name,
)

logger = get_logger(__name__)


class SchemaHarmonizer:
    """Normalizes raw Tele-Law column names onto the canonical schema,
    handling year-over-year naming drift in the official releases.
    """

    def __init__(self, column_mappings: Dict[str, Dict[str, str]] = None):  # type: ignore[assignment]
        """Initialize the harmonizer.

        Args:
            column_mappings: Per-fiscal-year raw-to-canonical column
                mapping rules. Defaults to ``config.COLUMN_MAPPINGS``.
        """
        self.column_mappings = column_mappings or config.COLUMN_MAPPINGS

    def harmonize(
        self, df: pd.DataFrame, fiscal_year: Optional[str] = None
    ) -> pd.DataFrame:
        """Rename raw columns to the canonical schema for a given fiscal
        year's mapping rules.

        Args:
            df: Raw dataframe as returned by the data loader for a single
                fiscal year (or containing a ``fiscal_year``-like column
                already).
            fiscal_year: Which year's mapping rules to apply. If ``None``,
                every fiscal year's mapping rules are merged (later years
                take precedence for overlapping keys), which is a
                reasonable default when the incoming dataframe already
                mixes multiple fiscal years.

        Returns:
            A new dataframe with columns renamed to the canonical schema.
            Columns with no matching canonical mapping are dropped, since
            downstream modules only operate on the canonical schema.

        Raises:
            DataEngineeringError: If, after harmonization, none of the
                canonical columns could be resolved (i.e. the mapping
                rules do not match this dataframe at all).
        """
        normalized_to_original = normalize_columns(df.columns)

        if fiscal_year and fiscal_year in self.column_mappings:
            active_mapping = dict(self.column_mappings[fiscal_year])
        else:
            active_mapping = {}
            for year_mapping in self.column_mappings.values():
                active_mapping.update(year_mapping)

        rename_map: Dict[str, str] = {}
        for original_col, normalized_col in normalized_to_original.items():
            canonical = active_mapping.get(normalized_col)
            if canonical:
                rename_map[original_col] = canonical

        if not rename_map:
            raise DataEngineeringError(
                "Schema harmonization could not map any source column to "
                f"the canonical schema. Source columns were: "
                f"{list(df.columns)}. Update config.COLUMN_MAPPINGS if the "
                "official dataset has introduced a new naming convention."
            )

        harmonized = df.rename(columns=rename_map)
        # Keep only canonical columns that actually resolved; missing ones
        # are handled by downstream validation, not silently fabricated.
        keep_cols = [c for c in config.CANONICAL_COLUMNS if c in harmonized.columns]
        harmonized = harmonized[keep_cols].copy()

        logger.info(
            "Harmonized schema: mapped %d/%d source columns -> %d canonical "
            "columns %s",
            len(rename_map),
            len(df.columns),
            len(keep_cols),
            keep_cols,
        )
        return harmonized


class DataCleaner:
    """Cleans a harmonized Tele-Law (or auxiliary) dataframe: type
    coercion, text normalization, null handling, and de-duplication.
    """

    def clean_telelaw_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean a harmonized Tele-Law dataframe.

        Steps:
            * Strip/title-case ``state_name`` and ``district_name``.
            * Coerce ``cases_registered`` / ``advice_enabled_count`` to
              non-negative integers, treating unparsable or negative
              values as missing-then-zero-filled (with a warning log).
            * Drop rows missing ``state_name``, ``district_name``, or
              ``fiscal_year`` (these cannot be meaningfully repaired).
            * Drop exact duplicate rows.

        Args:
            df: Harmonized dataframe (canonical column names already
                applied).

        Returns:
            A cleaned copy of the dataframe.
        """
        cleaned = df.copy()

        for text_col in ("state_name", "district_name"):
            if text_col in cleaned.columns:
                cleaned[text_col] = (
                    cleaned[text_col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.title()
                )
                cleaned.loc[cleaned[text_col].isin(["Nan", "None", ""]), text_col] = (
                    pd.NA
                )

        if "fiscal_year" in cleaned.columns:
            cleaned["fiscal_year"] = cleaned["fiscal_year"].astype(str).str.strip()

        for numeric_col in config.NON_NEGATIVE_NUMERIC_COLUMNS:
            if numeric_col not in cleaned.columns:
                continue
            coerced = pd.to_numeric(cleaned[numeric_col], errors="coerce")
            n_unparsable = int(coerced.isna().sum())
            if n_unparsable > 0:
                logger.warning(
                    "Column '%s': %d value(s) could not be parsed as "
                    "numeric and were set to 0.",
                    numeric_col,
                    n_unparsable,
                )
            coerced = coerced.fillna(0)
            n_negative = int((coerced < 0).sum())
            if n_negative > 0:
                logger.warning(
                    "Column '%s': %d negative value(s) clipped to 0.",
                    numeric_col,
                    n_negative,
                )
            coerced = coerced.clip(lower=0)
            cleaned[numeric_col] = coerced.round().astype(int)

        before_rows = len(cleaned)
        required_present = [
            c for c in config.REQUIRED_NON_NULL_COLUMNS if c in cleaned.columns
        ]
        if required_present:
            cleaned = cleaned.dropna(subset=required_present)
        dropped_for_nulls = before_rows - len(cleaned)
        if dropped_for_nulls > 0:
            logger.warning(
                "Dropped %d row(s) missing required field(s) %s.",
                dropped_for_nulls,
                required_present,
            )

        before_dedup = len(cleaned)
        cleaned = cleaned.drop_duplicates()
        n_duplicates_dropped = before_dedup - len(cleaned)
        if n_duplicates_dropped > 0:
            logger.info("Dropped %d exact duplicate row(s).", n_duplicates_dropped)

        cleaned = cleaned.reset_index(drop=True)
        logger.info(
            "Cleaning complete: %d rows remain (started with %d).",
            len(cleaned),
            before_rows,
        )
        return cleaned

    def clean_auxiliary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the auxiliary demographic dataframe.

        Args:
            df: Raw or lightly-processed auxiliary dataframe with columns
                matching (a subset of)
                ``config.AUXILIARY_CANONICAL_COLUMNS``.

        Returns:
            A cleaned copy of the dataframe with text fields normalized
            and numeric fields coerced to sensible ranges.
        """
        cleaned = df.copy()

        for text_col in ("state_name", "district_name"):
            if text_col in cleaned.columns:
                cleaned[text_col] = (
                    cleaned[text_col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.title()
                )

        numeric_bounds = {
            "population": (0, None),
            "rural_population_pct": (0, 100),
            "literacy_rate_pct": (0, 100),
            "sex_ratio": (0, None),
        }
        for col, (low, high) in numeric_bounds.items():
            if col not in cleaned.columns:
                continue
            coerced = pd.to_numeric(cleaned[col], errors="coerce")
            if low is not None:
                coerced = coerced.clip(lower=low)
            if high is not None:
                coerced = coerced.clip(upper=high)
            cleaned[col] = coerced

        cleaned = cleaned.drop_duplicates(
            subset=[c for c in ("state_name", "district_name") if c in cleaned.columns]
        ).reset_index(drop=True)
        return cleaned


class DistrictReconciler:
    """Reconciles Tele-Law district names against the auxiliary dataset's
    district names using exact matching first, then fuzzy matching as a
    fallback, scoped within the same state to avoid cross-state false
    matches.
    """

    def __init__(self, match_threshold: float = config.DISTRICT_NAME_MATCH_THRESHOLD):
        """Initialize the reconciler.

        Args:
            match_threshold: Minimum fuzzy-match similarity ratio required
                to accept a match. See
                ``config.DISTRICT_NAME_MATCH_THRESHOLD``.
        """
        self.match_threshold = match_threshold

    def build_reconciliation_map(
        self,
        telelaw_df: pd.DataFrame,
        auxiliary_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Build a mapping from each unique
        ``(state_name, district_name)`` pair in the Tele-Law data to its
        best-matching auxiliary-dataset district name.

        Matching is scoped per state: a Tele-Law district is only
        compared against auxiliary districts belonging to the *same*
        (normalized) state name, which both improves accuracy and keeps
        the matching process fast (O(n) per state rather than O(n*m)
        globally).

        Args:
            telelaw_df: Cleaned Tele-Law dataframe with ``state_name`` and
                ``district_name`` columns.
            auxiliary_df: Cleaned auxiliary dataframe with ``state_name``
                and ``district_name`` columns.

        Returns:
            A dataframe with columns ``state_name``, ``district_name``
            (original Tele-Law naming), ``matched_district_name`` (the
            resolved auxiliary-dataset name, or ``pd.NA`` if unresolved),
            and ``match_score`` (float, or ``pd.NA`` if unresolved).
        """
        aux_by_state: Dict[str, List[str]] = {}
        for state_name, group in auxiliary_df.groupby("state_name"):
            aux_by_state[normalize_name(state_name)] = list(
                group["district_name"].unique()
            )

        unique_pairs = telelaw_df[["state_name", "district_name"]].drop_duplicates()

        records: List[Dict] = []
        unresolved_count = 0

        for _, row in unique_pairs.iterrows():
            state_name = row["state_name"]
            district_name = row["district_name"]
            normalized_state = normalize_name(state_name)
            candidates = aux_by_state.get(normalized_state, [])

            match = fuzzy_match_name(
                district_name, candidates, threshold=self.match_threshold
            )
            if match is not None:
                matched_name, score = match
            else:
                matched_name, score = pd.NA, pd.NA
                unresolved_count += 1

            records.append(
                {
                    "state_name": state_name,
                    "district_name": district_name,
                    "matched_district_name": matched_name,
                    "match_score": score,
                }
            )

        reconciliation_map = pd.DataFrame(records)

        if unresolved_count > 0:
            logger.warning(
                "District reconciliation: %d/%d unique district(s) could "
                "not be matched to the auxiliary dataset within threshold "
                "%.2f and will have null demographic fields after merge.",
                unresolved_count,
                len(unique_pairs),
                self.match_threshold,
            )
        else:
            logger.info(
                "District reconciliation: all %d unique district(s) "
                "matched successfully.",
                len(unique_pairs),
            )

        return reconciliation_map


class DataEngineeringPipeline:
    """Top-level orchestrator that runs harmonization, cleaning,
    reconciliation, and merging in sequence to produce the final curated
    dataset consumed by feature engineering and modeling.
    """

    def __init__(
        self,
        harmonizer: Optional[SchemaHarmonizer] = None,
        cleaner: Optional[DataCleaner] = None,
        reconciler: Optional[DistrictReconciler] = None,
        validator: Optional[TeleLawDataValidator] = None,
    ) -> None:
        """Initialize the pipeline with (optionally injected) components,
        which makes each stage independently mockable in unit tests.

        Args:
            harmonizer: Schema harmonizer instance. Defaults to a new
                ``SchemaHarmonizer``.
            cleaner: Data cleaner instance. Defaults to a new
                ``DataCleaner``.
            reconciler: District reconciler instance. Defaults to a new
                ``DistrictReconciler``.
            validator: Validator instance used for pre/post checks.
                Defaults to a new ``TeleLawDataValidator``.
        """
        self.harmonizer = harmonizer or SchemaHarmonizer()
        self.cleaner = cleaner or DataCleaner()
        self.reconciler = reconciler or DistrictReconciler()
        self.validator = validator or TeleLawDataValidator()

    def run_pipeline(
        self,
        raw_telelaw_df: pd.DataFrame,
        raw_auxiliary_df: pd.DataFrame,
        fiscal_year: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """Execute the full harmonize -> clean -> reconcile -> merge
        pipeline.

        Args:
            raw_telelaw_df: Raw Tele-Law dataframe from
                ``data_loader.TeleLawDataLoader.load_telelaw_data``.
            raw_auxiliary_df: Raw auxiliary demographic dataframe from
                ``data_loader.TeleLawDataLoader.load_auxiliary_data``.
            fiscal_year: Optional single fiscal year to scope schema
                harmonization rules to; ``None`` merges all years' rules.

        Returns:
            A tuple of ``(curated_df, run_metadata)`` where
            ``curated_df`` is the final merged, cleaned, feature-ready
            dataframe, and ``run_metadata`` is a dict capturing row
            counts at each stage plus the district reconciliation match
            rate, suitable for logging/audit purposes.

        Raises:
            DataEngineeringError: If any stage fails to produce a usable
                dataframe.
        """
        run_metadata: Dict[str, object] = {}

        logger.info("=== Data engineering pipeline: START ===")

        # Stage 1: schema harmonization
        harmonized_telelaw = self.harmonizer.harmonize(
            raw_telelaw_df, fiscal_year=fiscal_year
        )
        run_metadata["harmonized_row_count"] = len(harmonized_telelaw)

        # Stage 2: validation of harmonized data (critical-only gate)
        self.validator.validate_or_raise(
            harmonized_telelaw, dataset_name="harmonized_telelaw"
        )

        # Stage 3: cleaning
        cleaned_telelaw = self.cleaner.clean_telelaw_data(harmonized_telelaw)
        cleaned_auxiliary = self.cleaner.clean_auxiliary_data(raw_auxiliary_df)
        run_metadata["cleaned_telelaw_row_count"] = len(cleaned_telelaw)
        run_metadata["cleaned_auxiliary_row_count"] = len(cleaned_auxiliary)

        if cleaned_telelaw.empty:
            raise DataEngineeringError(
                "Cleaning removed every row from the Tele-Law dataset; "
                "cannot proceed. Inspect upstream data quality."
            )

        # Stage 4: entity reconciliation
        reconciliation_map = self.reconciler.build_reconciliation_map(
            cleaned_telelaw, cleaned_auxiliary
        )
        matched_count = int(reconciliation_map["matched_district_name"].notna().sum())
        total_count = len(reconciliation_map)
        run_metadata["district_match_rate"] = (
            round(matched_count / total_count, 4) if total_count else 0.0
        )

        # Stage 5: merge
        telelaw_with_match = cleaned_telelaw.merge(
            reconciliation_map, on=["state_name", "district_name"], how="left"
        )

        auxiliary_for_merge = cleaned_auxiliary.rename(
            columns={"district_name": "matched_district_name"}
        )

        curated = telelaw_with_match.merge(
            auxiliary_for_merge,
            on=["state_name", "matched_district_name"],
            how="left",
            suffixes=("", "_aux"),
        )
        curated = curated.drop(columns=["matched_district_name", "match_score"])

        run_metadata["curated_row_count"] = len(curated)
        run_metadata["is_synthetic_telelaw"] = bool(
            raw_telelaw_df.attrs.get("is_synthetic", False)
        )
        run_metadata["is_synthetic_auxiliary"] = bool(
            raw_auxiliary_df.attrs.get("is_synthetic", False)
        )

        # Preserve synthetic-data provenance flag onto the curated frame.
        curated.attrs["is_synthetic"] = (
            run_metadata["is_synthetic_telelaw"]
            or run_metadata["is_synthetic_auxiliary"]
        )

        logger.info(
            "=== Data engineering pipeline: COMPLETE. Curated shape=%s, "
            "district match rate=%.1f%% ===",
            curated.shape,
            run_metadata["district_match_rate"] * 100,
        )

        return curated, run_metadata
