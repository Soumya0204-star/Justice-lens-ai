"""
utils.py
========
Shared low-level utilities used across the JusticeLens AI backend:
custom exception hierarchy, retry/backoff decorator, safe numeric helpers,
text normalization + fuzzy matching for district/state name reconciliation,
file hashing for data versioning, and dataframe I/O helpers.

Kept dependency-light on purpose (standard library only, plus pandas which
every other module already depends on) so this module can be imported
anywhere without pulling in heavy or optional third-party packages.
"""

from __future__ import annotations

import functools
import hashlib
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar

import numpy as np
import pandas as pd

from justicelens import config
from justicelens.logger import get_logger

logger = get_logger(__name__)

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Custom exception hierarchy
# --------------------------------------------------------------------------- #


class JusticeLensError(Exception):
    """Base class for all JusticeLens AI backend exceptions.

    Catching this exception catches every domain-specific error raised
    anywhere in the backend, which is useful for top-level orchestrators
    (e.g. the Streamlit app) that need to degrade gracefully instead of
    crashing on any pipeline failure.
    """


class DataIngestionError(JusticeLensError):
    """Raised when both live data retrieval and synthetic fallback
    generation fail for a data-loading operation."""


class DataValidationError(JusticeLensError):
    """Raised when a dataset fails a *critical* validation check (e.g. a
    required column is entirely absent) that makes it unsafe to proceed
    with downstream processing.
    """


class DataEngineeringError(JusticeLensError):
    """Raised when schema harmonization, cleaning, or entity reconciliation
    cannot produce a usable curated dataset.
    """


class FeatureEngineeringError(JusticeLensError):
    """Raised when a feature-engineering computation cannot be completed,
    typically due to missing prerequisite columns.
    """


class ModelTrainingError(JusticeLensError):
    """Raised when model training, cross-validation, or model selection
    cannot be completed (e.g. insufficient class diversity in the
    training data, or every candidate model fails to fit).
    """


class ModelEvaluationError(JusticeLensError):
    """Raised when evaluation metrics, confusion matrix, or ROC curve
    generation cannot be completed for a fitted model.
    """


class ExplainabilityError(JusticeLensError):
    """Raised when SHAP-based explanation generation cannot be completed,
    including the optional ``shap`` dependency being unavailable.
    """


class WatsonxIntegrationError(JusticeLensError):
    """Raised when IBM watsonx.ai authentication or inference fails and no
    further fallback is possible or permitted. Most callers in this
    codebase catch this internally and degrade to a deterministic
    templated narrative rather than letting it propagate -- it is exposed
    publicly so callers that need strict-mode (no silent fallback)
    behavior can opt into raising instead.
    """


class NarrativeGenerationError(JusticeLensError):
    """Raised when a narrative-generator module (executive summary,
    district comparison, policy recommendation, AI report, or Q&A) is
    given input data that is missing required fields, independent of
    whether watsonx.ai itself is reachable.
    """


# --------------------------------------------------------------------------- #
# Retry / backoff
# --------------------------------------------------------------------------- #


def retry_with_backoff(
    max_retries: int = 3,
    base_delay_seconds: float = 1.0,
    exceptions: Tuple[type, ...] = (Exception,),
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator that retries a function call on failure with exponential
    backoff.

    Args:
        max_retries: Maximum number of retry attempts after the initial
            call (so total attempts = ``max_retries + 1``).
        base_delay_seconds: Base delay between retries; actual delay for
            attempt ``i`` (0-indexed) is ``base_delay_seconds * 2**i``.
        exceptions: Tuple of exception types that should trigger a retry.
            Any other exception propagates immediately.

    Returns:
        The decorated function. On exhausting all retries, the last
        encountered exception is re-raised to the caller.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            last_exception: Optional[BaseException] = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:  # type: ignore[misc]
                    last_exception = exc
                    if attempt < max_retries:
                        delay = base_delay_seconds * (2**attempt)
                        logger.warning(
                            "Attempt %d/%d for '%s' failed with %s: %s. "
                            "Retrying in %.1fs.",
                            attempt + 1,
                            max_retries + 1,
                            func.__name__,
                            type(exc).__name__,
                            exc,
                            delay,
                        )
                        time.sleep(delay)
                    else:
                        logger.error(
                            "All %d attempts for '%s' failed. Last error: %s",
                            max_retries + 1,
                            func.__name__,
                            exc,
                        )
            assert last_exception is not None  # for type checkers
            raise last_exception

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Numeric helpers
# --------------------------------------------------------------------------- #


def safe_divide(
    numerator: float, denominator: float, default: float = 0.0
) -> float:
    """Divide two numbers without raising on a zero (or near-zero)
    denominator.

    Args:
        numerator: The dividend.
        denominator: The divisor.
        default: Value returned when ``denominator`` is smaller in
            magnitude than ``config.SAFE_DIVISION_EPSILON``.

    Returns:
        ``numerator / denominator``, or ``default`` if the denominator is
        effectively zero.
    """
    if abs(denominator) < config.SAFE_DIVISION_EPSILON:
        return default
    return numerator / denominator


def safe_divide_series(
    numerator: pd.Series, denominator: pd.Series, default: float = 0.0
) -> pd.Series:
    """Vectorized safe division for pandas Series.

    Args:
        numerator: Numerator series.
        denominator: Denominator series (same index as ``numerator``).
        default: Fill value used wherever the denominator is zero, NaN, or
            infinite results would otherwise occur.

    Returns:
        A new Series containing the element-wise safe division result.
    """
    safe_denominator = denominator.replace(0, pd.NA)
    result = numerator.divide(safe_denominator)
    # Replace +/-inf (e.g. from a near-zero-but-nonzero denominator) and any
    # NaN (from the zero-denominator replacement above, or NaN inputs) with
    # the default value, without relying on the removed
    # "mode.use_inf_as_na" pandas option.
    result = result.replace([np.inf, -np.inf], pd.NA)
    result = result.fillna(default)
    return result.astype(float)


# --------------------------------------------------------------------------- #
# Text normalization & fuzzy matching (for district/state reconciliation)
# --------------------------------------------------------------------------- #


def normalize_name(name: str) -> str:
    """Normalize a district/state name for reliable comparison and joining.

    Normalization steps:
        1. Cast to string and strip leading/trailing whitespace.
        2. Lower-case.
        3. Collapse internal whitespace/underscores/hyphens to a single
           space.
        4. Remove common administrative suffixes/prefixes that vary across
           sources (e.g. "district", "(u)", "(r)").

    Args:
        name: Raw name string, possibly ``None`` or non-string.

    Returns:
        The normalized string. Returns an empty string for ``None`` or
        empty input rather than raising.
    """
    if name is None:
        return ""
    text = str(name).strip().lower()
    if not text:
        return ""

    for junk in ["district", "(u)", "(r)", "(urban)", "(rural)", "*"]:
        text = text.replace(junk, "")

    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())  # collapse repeated whitespace
    return text.strip()


def similarity_ratio(left: str, right: str) -> float:
    """Compute a normalized similarity ratio between two strings using the
    standard-library ``difflib.SequenceMatcher``.

    Args:
        left: First string.
        right: Second string.

    Returns:
        A float in ``[0.0, 1.0]`` where ``1.0`` means the strings are
        identical.
    """
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=left, b=right).ratio()


def fuzzy_match_name(
    target: str,
    candidates: Iterable[str],
    threshold: float = config.DISTRICT_NAME_MATCH_THRESHOLD,
) -> Optional[Tuple[str, float]]:
    """Find the best fuzzy match for ``target`` among ``candidates``.

    Both ``target`` and every candidate are normalized via
    :func:`normalize_name` before comparison, so callers do not need to
    pre-clean their inputs.

    Args:
        target: The name to match (e.g. a Tele-Law district name).
        candidates: Iterable of canonical candidate names (e.g. district
            names from the auxiliary demographic dataset).
        threshold: Minimum similarity ratio required to accept a match.

    Returns:
        A tuple of ``(best_candidate, score)`` if the best match meets or
        exceeds ``threshold``, otherwise ``None``.
    """
    normalized_target = normalize_name(target)
    if not normalized_target:
        return None

    best_candidate: Optional[str] = None
    best_score = 0.0

    for candidate in candidates:
        normalized_candidate = normalize_name(candidate)
        if not normalized_candidate:
            continue
        # Exact match on normalized strings short-circuits the search.
        if normalized_candidate == normalized_target:
            return candidate, 1.0
        score = similarity_ratio(normalized_target, normalized_candidate)
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_candidate is not None and best_score >= threshold:
        return best_candidate, best_score
    return None


# --------------------------------------------------------------------------- #
# Filesystem / data versioning helpers
# --------------------------------------------------------------------------- #


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (and any missing parents) if it does not already
    exist.

    Args:
        path: Directory path to ensure exists.

    Returns:
        The same ``path``, for convenient chaining.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    """Compute a stable SHA-256 hash of a dataframe's contents.

    Used to version curated datasets (e.g. embedding a short hash in the
    output filename or a governance log entry) so downstream consumers can
    detect when the underlying data has actually changed.

    Args:
        df: The dataframe to hash.

    Returns:
        A hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    # Sort columns for order-independence, then hash a deterministic CSV
    # representation.
    ordered = df.reindex(sorted(df.columns), axis=1)
    csv_bytes = ordered.to_csv(index=False).encode("utf-8")
    hasher.update(csv_bytes)
    return hasher.hexdigest()


def save_dataframe(
    df: pd.DataFrame, path: Path, index: bool = False
) -> Path:
    """Persist a dataframe to disk as CSV or Parquet based on file
    extension.

    Args:
        df: Dataframe to save.
        path: Destination path. Extension must be ``.csv`` or ``.parquet``.
        index: Whether to write the dataframe index.

    Returns:
        The path the data was written to.

    Raises:
        ValueError: If the file extension is unsupported.
    """
    ensure_dir(path.parent)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=index)
    elif suffix == ".parquet":
        try:
            df.to_parquet(path, index=index)
        except (ImportError, ValueError) as exc:
            # No parquet engine available (pyarrow/fastparquet) -- fall back
            # to CSV alongside a warning rather than failing the pipeline.
            logger.warning(
                "Parquet engine unavailable (%s); falling back to CSV for %s",
                exc,
                path,
            )
            fallback_path = path.with_suffix(".csv")
            df.to_csv(fallback_path, index=index)
            return fallback_path
    else:
        raise ValueError(f"Unsupported file extension for save_dataframe: {suffix}")
    logger.info("Saved dataframe with shape %s to %s", df.shape, path)
    return path


def load_dataframe(path: Path) -> pd.DataFrame:
    """Load a dataframe from a CSV or Parquet file.

    Args:
        path: Source file path.

    Returns:
        The loaded dataframe.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If the file extension is unsupported.
    """
    if not path.exists():
        raise FileNotFoundError(f"No such file: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension for load_dataframe: {suffix}")


def normalize_columns(columns: Iterable[str]) -> Dict[str, str]:
    """Build a mapping from original column names to a normalized form
    (lower-case, stripped, whitespace/punctuation collapsed to underscores).

    Used by data_engineering.py to normalize raw source columns before
    applying the FY-specific ``config.COLUMN_MAPPINGS`` lookup.

    Args:
        columns: Iterable of raw column names.

    Returns:
        Dict mapping each original column name to its normalized form.
    """
    normalized: Dict[str, str] = {}
    for col in columns:
        text = str(col).strip().lower()
        text = text.replace("-", "_").replace(" ", "_").replace(".", "")
        text = "_".join(part for part in text.split("_") if part)
        normalized[col] = text
    return normalized


def chunked(items: List[T], chunk_size: int) -> List[List[T]]:
    """Split a list into consecutive chunks of at most ``chunk_size``
    elements.

    Args:
        items: The list to split.
        chunk_size: Maximum size of each chunk. Must be positive.

    Returns:
        A list of chunks (sublists).

    Raises:
        ValueError: If ``chunk_size`` is not positive.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]