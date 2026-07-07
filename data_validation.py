"""
data_validation.py
===================
Automated data-quality validation for JusticeLens AI.

Every dataset that flows through the pipeline (raw Tele-Law data,
harmonized data, auxiliary demographic data, the final curated feature
table) passes through this module, which distinguishes between:

    * **Critical failures** -- issues that make the dataset unsafe to use
      downstream (e.g. a required column is entirely missing, or the
      dataframe is empty). These raise ``DataValidationError``.
    * **Warnings** -- issues that are recorded in a structured validation
      report but do not halt the pipeline (e.g. a handful of missing
      values, a few negative counts, some duplicate rows). Callers decide
      how to act on warnings (typically: log them, surface them in the
      "Data Quality" dashboard page, and let cleaning fix what it can).

The validator never mutates the input dataframe -- it only inspects it and
returns a structured, JSON-serializable report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import pandas as pd

from justicelens import config
from justicelens.logger import get_logger
from justicelens.utils import DataValidationError

logger = get_logger(__name__)


@dataclass
class ValidationIssue:
    """A single validation finding.

    Attributes:
        check_name: Short machine-readable identifier for the check that
            produced this finding (e.g. "missing_required_column").
        severity: Either ``"critical"`` or ``"warning"``.
        message: Human-readable description of the issue.
        details: Optional structured extra context (affected columns,
            row counts, sample values, etc.).
    """

    check_name: str
    severity: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize this issue to a plain dict.

        Returns:
            A JSON-serializable dictionary representation.
        """
        return {
            "check_name": self.check_name,
            "severity": self.severity,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Aggregate result of running all validation checks against a
    dataframe.

    Attributes:
        dataset_name: Human-readable label for the dataset validated
            (used in logs/report output).
        row_count: Number of rows in the validated dataframe.
        column_count: Number of columns in the validated dataframe.
        issues: All findings, critical and warning, in the order the
            checks ran.
    """

    dataset_name: str
    row_count: int
    column_count: int
    issues: List[ValidationIssue] = field(default_factory=list)

    @property
    def critical_issues(self) -> List[ValidationIssue]:
        """Return only the critical-severity issues."""
        return [issue for issue in self.issues if issue.severity == "critical"]

    @property
    def warning_issues(self) -> List[ValidationIssue]:
        """Return only the warning-severity issues."""
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def is_valid(self) -> bool:
        """``True`` if there are no critical issues."""
        return len(self.critical_issues) == 0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the full report to a plain dict.

        Returns:
            A JSON-serializable dictionary representation, suitable for
            rendering in the "Data Quality" dashboard page or writing to
            an audit log.
        """
        return {
            "dataset_name": self.dataset_name,
            "row_count": self.row_count,
            "column_count": self.column_count,
            "is_valid": self.is_valid,
            "critical_issue_count": len(self.critical_issues),
            "warning_issue_count": len(self.warning_issues),
            "issues": [issue.to_dict() for issue in self.issues],
        }


class TeleLawDataValidator:
    """Runs schema, type, range, and consistency checks against Tele-Law
    (and auxiliary) datasets.

    Typical usage::

        validator = TeleLawDataValidator()
        report = validator.run_all_validations(df, dataset_name="raw_telelaw")
        if not report.is_valid:
            raise DataValidationError(...)
    """

    def __init__(
        self,
        required_columns: List[str] = None,  # type: ignore[assignment]
        required_non_null_columns: List[str] = None,  # type: ignore[assignment]
        non_negative_columns: List[str] = None,  # type: ignore[assignment]
    ) -> None:
        """Initialize the validator with configurable column expectations.

        Args:
            required_columns: Columns that must be present in the
                dataframe. Defaults to ``config.CANONICAL_COLUMNS``.
            required_non_null_columns: Columns that must never contain
                null values. Defaults to
                ``config.REQUIRED_NON_NULL_COLUMNS``.
            non_negative_columns: Numeric columns expected to contain only
                non-negative values. Defaults to
                ``config.NON_NEGATIVE_NUMERIC_COLUMNS``.
        """
        self.required_columns = required_columns or config.CANONICAL_COLUMNS
        self.required_non_null_columns = (
            required_non_null_columns or config.REQUIRED_NON_NULL_COLUMNS
        )
        self.non_negative_columns = (
            non_negative_columns or config.NON_NEGATIVE_NUMERIC_COLUMNS
        )

    def _check_not_empty(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Verify the dataframe contains at least one row.

        Args:
            df: Dataframe under validation.

        Returns:
            A list containing one critical issue if the dataframe is
            empty, otherwise an empty list.
        """
        if df.empty:
            return [
                ValidationIssue(
                    check_name="empty_dataframe",
                    severity="critical",
                    message="Dataframe contains zero rows.",
                )
            ]
        return []

    def _check_required_columns(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Verify every required column is present in the dataframe.

        Args:
            df: Dataframe under validation.

        Returns:
            A list with one critical issue per missing required column.
        """
        issues: List[ValidationIssue] = []
        missing = [col for col in self.required_columns if col not in df.columns]
        if missing:
            issues.append(
                ValidationIssue(
                    check_name="missing_required_column",
                    severity="critical",
                    message=(
                        f"Required column(s) missing from dataset: {missing}"
                    ),
                    details={"missing_columns": missing},
                )
            )
        return issues

    def _check_required_non_null(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check that required-non-null columns have no missing values.

        Columns absent from the dataframe are skipped here (already
        reported by ``_check_required_columns``) to avoid duplicate noise.

        Args:
            df: Dataframe under validation.

        Returns:
            A list of warning-severity issues, one per column with any
            null values, unless the null share exceeds 50% of rows in
            which case it is escalated to critical.
        """
        issues: List[ValidationIssue] = []
        for col in self.required_non_null_columns:
            if col not in df.columns:
                continue
            null_count = int(df[col].isna().sum())
            if null_count == 0:
                continue
            null_share = null_count / max(len(df), 1)
            severity = "critical" if null_share > 0.5 else "warning"
            issues.append(
                ValidationIssue(
                    check_name="null_values_in_required_column",
                    severity=severity,
                    message=(
                        f"Column '{col}' has {null_count} null value(s) "
                        f"({null_share:.1%} of rows)."
                    ),
                    details={
                        "column": col,
                        "null_count": null_count,
                        "null_share": round(null_share, 4),
                    },
                )
            )
        return issues

    def _check_non_negative(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Check that numeric columns expected to be non-negative contain
        no negative values.

        Args:
            df: Dataframe under validation.

        Returns:
            A list of warning-severity issues, one per column containing
            negative values.
        """
        issues: List[ValidationIssue] = []
        for col in self.non_negative_columns:
            if col not in df.columns:
                continue
            numeric_col = pd.to_numeric(df[col], errors="coerce")
            negative_count = int((numeric_col < 0).sum())
            if negative_count > 0:
                issues.append(
                    ValidationIssue(
                        check_name="negative_values",
                        severity="warning",
                        message=(
                            f"Column '{col}' contains {negative_count} "
                            "negative value(s), which is invalid for a "
                            "count field."
                        ),
                        details={"column": col, "negative_count": negative_count},
                    )
                )
        return issues

    def _check_duplicates(
        self, df: pd.DataFrame, key_columns: List[str]
    ) -> List[ValidationIssue]:
        """Detect duplicate rows on a given key (e.g.
        state/district/fiscal_year should be unique).

        Args:
            df: Dataframe under validation.
            key_columns: Columns that together should uniquely identify a
                row.

        Returns:
            A list containing one warning issue if duplicates are found on
            the given key, otherwise an empty list.
        """
        present_keys = [col for col in key_columns if col in df.columns]
        if len(present_keys) != len(key_columns):
            return []

        duplicate_mask = df.duplicated(subset=present_keys, keep=False)
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count > 0:
            return [
                ValidationIssue(
                    check_name="duplicate_rows",
                    severity="warning",
                    message=(
                        f"Found {duplicate_count} duplicate row(s) on key "
                        f"{present_keys}."
                    ),
                    details={
                        "key_columns": present_keys,
                        "duplicate_count": duplicate_count,
                    },
                )
            ]
        return []

    def _check_known_fiscal_years(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Flag any fiscal-year values outside the expected coverage
        window (``config.FISCAL_YEARS``).

        Args:
            df: Dataframe under validation.

        Returns:
            A list containing one warning issue if unexpected fiscal years
            are found, otherwise an empty list.
        """
        if "fiscal_year" not in df.columns:
            return []

        observed = set(df["fiscal_year"].dropna().astype(str).unique())
        expected = set(config.FISCAL_YEARS)
        unexpected = observed - expected
        if unexpected:
            return [
                ValidationIssue(
                    check_name="unexpected_fiscal_year",
                    severity="warning",
                    message=(
                        f"Dataset contains fiscal year value(s) outside the "
                        f"expected set {sorted(expected)}: {sorted(unexpected)}"
                    ),
                    details={"unexpected_values": sorted(unexpected)},
                )
            ]
        return []

    def _check_zero_inflated_rows(self, df: pd.DataFrame) -> List[ValidationIssue]:
        """Flag districts where ``cases_registered`` is zero for every
        fiscal year present, which usually indicates a reporting gap
        rather than genuine zero demand.

        Args:
            df: Dataframe under validation.

        Returns:
            A list containing one warning issue (with affected district
            count) if any such districts are found.
        """
        required = {"district_name", "state_name", "cases_registered"}
        if not required.issubset(df.columns):
            return []

        grouped = df.groupby(["state_name", "district_name"])["cases_registered"]
        all_zero_mask = grouped.transform("max") == 0
        affected_districts = int(
            df.loc[all_zero_mask, ["state_name", "district_name"]]
            .drop_duplicates()
            .shape[0]
        )
        if affected_districts > 0:
            return [
                ValidationIssue(
                    check_name="zero_inflated_district",
                    severity="warning",
                    message=(
                        f"{affected_districts} district(s) report zero cases "
                        "registered across every available fiscal year -- "
                        "likely a reporting gap; flagged for the Data "
                        "Quality dashboard."
                    ),
                    details={"affected_district_count": affected_districts},
                )
            ]
        return []

    def run_all_validations(
        self, df: pd.DataFrame, dataset_name: str = "dataset"
    ) -> ValidationReport:
        """Run the full validation suite against a dataframe.

        Args:
            df: Dataframe to validate.
            dataset_name: Human-readable label used in the resulting
                report and in log messages.

        Returns:
            A :class:`ValidationReport` summarizing all findings.
        """
        logger.info(
            "Running validation suite on '%s' (%d rows, %d columns)",
            dataset_name,
            df.shape[0],
            df.shape[1],
        )

        issues: List[ValidationIssue] = []
        issues.extend(self._check_not_empty(df))

        # If the dataframe is empty, remaining checks are meaningless / may
        # error, so short-circuit here.
        if df.empty:
            report = ValidationReport(
                dataset_name=dataset_name,
                row_count=0,
                column_count=df.shape[1],
                issues=issues,
            )
            logger.error("Validation for '%s' failed: dataframe is empty.", dataset_name)
            return report

        issues.extend(self._check_required_columns(df))
        issues.extend(self._check_required_non_null(df))
        issues.extend(self._check_non_negative(df))
        issues.extend(
            self._check_duplicates(
                df, key_columns=["state_name", "district_name", "fiscal_year"]
            )
        )
        issues.extend(self._check_known_fiscal_years(df))
        issues.extend(self._check_zero_inflated_rows(df))

        report = ValidationReport(
            dataset_name=dataset_name,
            row_count=df.shape[0],
            column_count=df.shape[1],
            issues=issues,
        )

        if report.is_valid:
            logger.info(
                "Validation for '%s' passed with %d warning(s).",
                dataset_name,
                len(report.warning_issues),
            )
        else:
            logger.error(
                "Validation for '%s' FAILED with %d critical issue(s): %s",
                dataset_name,
                len(report.critical_issues),
                [issue.message for issue in report.critical_issues],
            )

        return report

    def validate_or_raise(
        self, df: pd.DataFrame, dataset_name: str = "dataset"
    ) -> ValidationReport:
        """Run all validations and raise ``DataValidationError`` if any
        critical issue is found.

        Args:
            df: Dataframe to validate.
            dataset_name: Human-readable label for logs/errors.

        Returns:
            The validation report, when validation succeeds.

        Raises:
            DataValidationError: If one or more critical issues are found.
        """
        report = self.run_all_validations(df, dataset_name=dataset_name)
        if not report.is_valid:
            raise DataValidationError(
                f"Validation failed for '{dataset_name}' with "
                f"{len(report.critical_issues)} critical issue(s): "
                f"{[issue.message for issue in report.critical_issues]}"
            )
        return report
