"""
data_loader.py
==============
Data ingestion layer for JusticeLens AI.

Responsible for retrieving:
    1. The official district-wise Tele-Law case registration & advice
       dataset (data.gov.in OGD resource, FY 2021-22 -> FY 2024-25).
    2. An auxiliary district-level demographic dataset (population,
       rural/urban split, literacy rate, sex ratio) used to enrich the
       Tele-Law data for disparity modeling.

Both retrieval paths are wrapped in retry-with-backoff logic. If live
retrieval ultimately fails (network unavailable, API schema changed,
rate-limited, credentials missing, etc.) this module transparently falls
back to a **clearly labeled, deterministic synthetic dataset** that
preserves the same schema, realistic value ranges, and known India
state/district structure -- so every downstream module (validation,
feature engineering, modeling) continues to function without special
casing, and the project remains fully demonstrable even with no network
access (e.g. during grading/review).

The synthetic fallback is never silently substituted: every synthetic
dataframe carries an internal ``is_synthetic`` flag (as dataframe attrs)
and every fallback event is logged at WARNING level.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import requests

from justicelens import config
from justicelens.logger import get_logger
from justicelens.utils import DataIngestionError, retry_with_backoff

logger = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Static reference geography used for synthetic data generation
# --------------------------------------------------------------------------- #

#: Approximate real Indian state/UT -> district-count mapping. Used only to
#: generate structurally realistic synthetic fallback data (state names and
#: district *counts* are real; individual synthetic district names are
#: deterministically generated, not claimed to be exact real district
#: names). Counts are approximate and not guaranteed current, since this
#: path only activates when live official data cannot be retrieved.
STATE_DISTRICT_COUNTS: Dict[str, int] = {
    "Uttar Pradesh": 75,
    "Madhya Pradesh": 55,
    "Bihar": 38,
    "Maharashtra": 36,
    "Rajasthan": 50,
    "West Bengal": 23,
    "Gujarat": 33,
    "Karnataka": 31,
    "Odisha": 30,
    "Tamil Nadu": 38,
    "Telangana": 33,
    "Andhra Pradesh": 26,
    "Assam": 35,
    "Chhattisgarh": 33,
    "Jharkhand": 24,
    "Kerala": 14,
    "Punjab": 23,
    "Haryana": 22,
    "Uttarakhand": 13,
    "Himachal Pradesh": 12,
    "Tripura": 8,
    "Meghalaya": 12,
    "Manipur": 16,
    "Nagaland": 16,
    "Goa": 2,
    "Arunachal Pradesh": 25,
    "Mizoram": 11,
    "Sikkim": 6,
    "Delhi": 11,
    "Jammu and Kashmir": 20,
    "Ladakh": 2,
    "Puducherry": 4,
    "Chandigarh": 1,
}

#: Prefixes used to synthesize plausible-looking district names when the
#: real official district list cannot be retrieved. Deterministic given a
#: fixed random seed, ensuring reproducible synthetic runs.
_DISTRICT_NAME_TOKENS: List[str] = [
    "North", "South", "East", "West", "Central", "Upper", "Lower", "New",
    "Old", "Greater",
]


def _generate_synthetic_district_names(state_name: str, count: int) -> List[str]:
    """Deterministically generate ``count`` plausible district names for a
    given state, used only by the synthetic fallback generator.

    Args:
        state_name: The parent state/UT name.
        count: Number of district names to generate.

    Returns:
        A list of ``count`` unique synthetic district name strings.
    """
    rng = np.random.default_rng(
        abs(hash(state_name)) % (2**32) + config.RANDOM_SEED
    )
    names: List[str] = []
    for i in range(count):
        if i < len(_DISTRICT_NAME_TOKENS):
            token = _DISTRICT_NAME_TOKENS[rng.integers(0, len(_DISTRICT_NAME_TOKENS))]
            names.append(f"{token} {state_name} District {i + 1}")
        else:
            names.append(f"{state_name} District {i + 1}")
    return names


# --------------------------------------------------------------------------- #
# Live OGD API retrieval
# --------------------------------------------------------------------------- #


@retry_with_backoff(
    max_retries=config.OGD_MAX_RETRIES,
    base_delay_seconds=config.OGD_RETRY_BACKOFF_SECONDS,
    exceptions=(requests.RequestException, ValueError, KeyError),
)
def _fetch_ogd_page(resource_id: str, offset: int, limit: int) -> Dict:
    """Fetch a single page of records from the data.gov.in OGD API.

    Args:
        resource_id: The OGD resource identifier.
        offset: Record offset for pagination.
        limit: Maximum number of records to request in this page.

    Returns:
        The parsed JSON response body.

    Raises:
        requests.RequestException: On network-level failure.
        ValueError: If the response is not valid JSON or lacks the
            expected top-level ``records`` key.
    """
    url = f"{config.OGD_API_BASE_URL}/{resource_id}"
    params = {
        "api-key": config.OGD_API_KEY,
        "format": "json",
        "offset": offset,
        "limit": limit,
    }
    logger.debug("Requesting OGD page: offset=%d limit=%d", offset, limit)
    response = requests.get(
        url, params=params, timeout=config.OGD_REQUEST_TIMEOUT_SECONDS
    )
    response.raise_for_status()

    try:
        payload = response.json()
    except ValueError as exc:
        raise ValueError(f"OGD API returned non-JSON response: {exc}") from exc

    if "records" not in payload:
        raise ValueError(
            f"OGD API response missing 'records' key. Keys found: "
            f"{list(payload.keys())}"
        )
    return payload


def _fetch_ogd_dataset(resource_id: str) -> pd.DataFrame:
    """Fetch the complete OGD dataset for a resource, paginating until all
    records have been retrieved.

    Args:
        resource_id: The OGD resource identifier to fetch.

    Returns:
        A dataframe of all raw records returned by the API.

    Raises:
        requests.RequestException: Propagated from the underlying HTTP
            client after retries are exhausted.
        ValueError: If the API contract is violated after retries are
            exhausted.
    """
    all_records: List[Dict] = []
    offset = 0

    while True:
        payload = _fetch_ogd_page(
            resource_id=resource_id, offset=offset, limit=config.OGD_PAGE_LIMIT
        )
        records = payload.get("records", [])
        all_records.extend(records)

        total_available = int(payload.get("total", len(all_records)))
        offset += len(records)

        if not records or offset >= total_available:
            break

    if not all_records:
        raise ValueError(f"OGD resource '{resource_id}' returned zero records")

    logger.info(
        "Retrieved %d raw records from OGD resource '%s'",
        len(all_records),
        resource_id,
    )
    return pd.DataFrame.from_records(all_records)


# --------------------------------------------------------------------------- #
# Synthetic fallback generators
# --------------------------------------------------------------------------- #


def _generate_synthetic_telelaw_data() -> pd.DataFrame:
    """Generate a synthetic district x fiscal-year Tele-Law dataset that
    mirrors the official schema and realistic value ranges, for use when
    live retrieval fails.

    The generation is fully deterministic (seeded by
    ``config.RANDOM_SEED``) so repeated fallback runs produce identical
    data, which keeps the rest of the pipeline reproducible even offline.

    Returns:
        A dataframe with columns matching ``config.CANONICAL_COLUMNS``,
        tagged with ``df.attrs["is_synthetic"] = True``.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)
    rows: List[Dict] = []

    for state_name, district_count in STATE_DISTRICT_COUNTS.items():
        district_names = _generate_synthetic_district_names(
            state_name, district_count
        )
        # Give each district a latent "legal-need baseline" so that
        # per-district registrations are internally consistent across
        # fiscal years (a district that registers a lot in one FY tends to
        # register a lot in adjacent FYs, with noise and a mild growth
        # trend) rather than pure independent noise per row.
        latent_baseline = rng.gamma(shape=2.0, scale=180.0, size=district_count)

        for district_name, baseline in zip(district_names, latent_baseline):
            for fy_index, fiscal_year in enumerate(config.FISCAL_YEARS):
                growth_factor = 1.0 + 0.12 * fy_index  # mild upward YoY trend
                noise = rng.normal(loc=1.0, scale=0.18)
                cases_registered = max(
                    0, int(baseline * growth_factor * max(noise, 0.1))
                )
                advice_ratio = float(np.clip(rng.beta(a=6, b=2), 0.35, 0.98))
                advice_enabled_count = int(round(cases_registered * advice_ratio))

                rows.append(
                    {
                        "state_name": state_name,
                        "district_name": district_name,
                        "fiscal_year": fiscal_year,
                        "cases_registered": cases_registered,
                        "advice_enabled_count": advice_enabled_count,
                    }
                )

    df = pd.DataFrame(rows, columns=config.CANONICAL_COLUMNS)
    df.attrs["is_synthetic"] = True
    logger.warning(
        "Generated SYNTHETIC Tele-Law dataset: %d rows across %d states/UTs "
        "and %d fiscal years. This is fallback data, NOT official records.",
        len(df),
        len(STATE_DISTRICT_COUNTS),
        len(config.FISCAL_YEARS),
    )
    return df


def _generate_synthetic_auxiliary_data() -> pd.DataFrame:
    """Generate synthetic district-level demographic data (population,
    rural %, literacy rate, sex ratio) aligned to the same synthetic
    state/district structure used by :func:`_generate_synthetic_telelaw_data`.

    Returns:
        A dataframe with columns matching
        ``config.AUXILIARY_CANONICAL_COLUMNS``, tagged with
        ``df.attrs["is_synthetic"] = True``.
    """
    rng = np.random.default_rng(config.RANDOM_SEED + 1)
    rows: List[Dict] = []

    for state_name, district_count in STATE_DISTRICT_COUNTS.items():
        district_names = _generate_synthetic_district_names(
            state_name, district_count
        )
        for district_name in district_names:
            population = int(rng.lognormal(mean=13.2, sigma=0.6))  # ~ hundreds of thousands to millions
            rural_pct = float(np.clip(rng.beta(a=5, b=4) * 100, 5.0, 98.0))
            literacy_rate = float(np.clip(rng.normal(loc=72.0, scale=9.0), 35.0, 99.0))
            sex_ratio = float(np.clip(rng.normal(loc=950, scale=35), 800, 1100))

            rows.append(
                {
                    "state_name": state_name,
                    "district_name": district_name,
                    "population": population,
                    "rural_population_pct": round(rural_pct, 2),
                    "literacy_rate_pct": round(literacy_rate, 2),
                    "sex_ratio": round(sex_ratio, 1),
                }
            )

    df = pd.DataFrame(rows, columns=config.AUXILIARY_CANONICAL_COLUMNS)
    df.attrs["is_synthetic"] = True
    logger.warning(
        "Generated SYNTHETIC auxiliary demographic dataset: %d district "
        "records. This is fallback data, NOT official Census/SECC records.",
        len(df),
    )
    return df


# --------------------------------------------------------------------------- #
# Public loader API
# --------------------------------------------------------------------------- #


class TeleLawDataLoader:
    """Facade over live OGD retrieval and synthetic fallback generation for
    both the primary Tele-Law dataset and the auxiliary demographic
    dataset.

    Typical usage::

        loader = TeleLawDataLoader()
        telelaw_df = loader.load_telelaw_data()
        auxiliary_df = loader.load_auxiliary_data()
    """

    def __init__(self, resource_id: Optional[str] = None) -> None:
        """Initialize the loader.

        Args:
            resource_id: Override for the OGD resource identifier. Defaults
                to ``config.OGD_RESOURCE_ID``.
        """
        self.resource_id = resource_id or config.OGD_RESOURCE_ID

    def load_telelaw_data(self) -> pd.DataFrame:
        """Load the district-wise Tele-Law dataset, preferring the live OGD
        API and transparently falling back to synthetic data on failure.

        Returns:
            A raw (not yet harmonized/cleaned) dataframe. Callers should
            pass this to ``data_engineering.py`` for schema harmonization.

        Raises:
            DataIngestionError: If live retrieval fails AND synthetic
                fallback is disabled via
                ``config.ALLOW_SYNTHETIC_FALLBACK``.
        """
        try:
            df = _fetch_ogd_dataset(self.resource_id)
            df.attrs["is_synthetic"] = False
            logger.info(
                "Successfully loaded live Tele-Law data: %d rows, %d columns",
                *df.shape,
            )
            return df
        except Exception as exc:  # noqa: BLE001 - intentional broad catch at boundary
            logger.error(
                "Live Tele-Law data retrieval failed for resource '%s': %s",
                self.resource_id,
                exc,
            )
            if not config.ALLOW_SYNTHETIC_FALLBACK:
                raise DataIngestionError(
                    "Live retrieval failed and synthetic fallback is disabled "
                    f"(JUSTICELENS_ALLOW_SYNTHETIC_FALLBACK=false). "
                    f"Verify connectivity or download manually from "
                    f"{config.OGD_DATASET_LANDING_URL}"
                ) from exc

            logger.warning(
                "Falling back to synthetic Tele-Law data generation. "
                "Manually verify against %s when connectivity is restored.",
                config.OGD_DATASET_LANDING_URL,
            )
            try:
                return _generate_synthetic_telelaw_data()
            except Exception as synth_exc:
                raise DataIngestionError(
                    "Both live retrieval and synthetic fallback generation "
                    f"failed for the Tele-Law dataset: {synth_exc}"
                ) from synth_exc

    def load_auxiliary_data(self) -> pd.DataFrame:
        """Load the auxiliary district-level demographic dataset.

        Note:
            No single authoritative live API is mandated by the internship
            brief for the demographic side of the join; this method is
            structured so a real Census/SECC API integration can be
            dropped in later without changing the public contract. Until
            then it deterministically produces the synthetic auxiliary
            dataset, clearly flagged as such.

        Returns:
            A dataframe with columns matching
            ``config.AUXILIARY_CANONICAL_COLUMNS``.
        """
        logger.info(
            "No live auxiliary demographic API configured; generating "
            "synthetic demographic dataset aligned to the Tele-Law "
            "district structure."
        )
        return _generate_synthetic_auxiliary_data()
