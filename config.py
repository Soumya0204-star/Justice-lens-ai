"""
config.py
=========
Centralized configuration for the JusticeLens AI backend.
(Full merged version with all attributes)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


def _load_dotenv() -> None:
    """Load a local .env file into the process environment if present.

    This keeps the app working when credentials are stored in the project
    root's .env file instead of being exported into the shell environment.
    Existing environment variables always win.
    """
    candidates = [
        Path(__file__).resolve().parent / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            for raw_line in candidate.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[len("export "):].lstrip()
                if "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            pass
        break


_load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


# --------------------------------------------------------------------------- #
# Project root & filesystem layout
# --------------------------------------------------------------------------- #

BASE_DIR: Path = Path(
    os.getenv("JUSTICELENS_BASE_DIR", str(Path(__file__).resolve().parent))
)
DATA_DIR: Path = BASE_DIR / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"
DATA_AUXILIARY_DIR: Path = DATA_DIR / "auxiliary"
DATA_INTERIM_DIR: Path = DATA_DIR / "interim"
DATA_PROCESSED_DIR: Path = DATA_DIR / "processed"
MODEL_DIR: Path = BASE_DIR / "models"
OUTPUT_DIR: Path = BASE_DIR / "outputs"
LOG_DIR: Path = BASE_DIR / "logs"

REQUIRED_DIRECTORIES: List[Path] = [
    DATA_RAW_DIR,
    DATA_AUXILIARY_DIR,
    DATA_INTERIM_DIR,
    DATA_PROCESSED_DIR,
    MODEL_DIR,
    OUTPUT_DIR,
    LOG_DIR,
]
for _directory in REQUIRED_DIRECTORIES:
    _directory.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #

LOG_LEVEL: str = os.getenv("JUSTICELENS_LOG_LEVEL", "INFO").upper()
LOG_FILE_PATH: Path = LOG_DIR / "justicelens.log"
LOG_MAX_BYTES: int = _env_int("JUSTICELENS_LOG_MAX_BYTES", 5 * 1024 * 1024)
LOG_BACKUP_COUNT: int = _env_int("JUSTICELENS_LOG_BACKUP_COUNT", 5)
LOG_FORMAT: str = (
    "%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(message)s"
)
LOG_DATE_FORMAT: str = "%Y-%m-%d %H:%M:%S"


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

RANDOM_SEED: int = _env_int("JUSTICELENS_RANDOM_SEED", 42)


# --------------------------------------------------------------------------- #
# Official Government Data (OGD) source configuration
# --------------------------------------------------------------------------- #

OGD_RESOURCE_ID: str = os.getenv(
    "JUSTICELENS_OGD_RESOURCE_ID", "district-wise-tele-law-case-registration"
)
OGD_API_BASE_URL: str = os.getenv(
    "JUSTICELENS_OGD_API_BASE_URL", "https://api.data.gov.in/resource"
)
OGD_API_KEY: str = os.getenv("JUSTICELENS_OGD_API_KEY", "579b464db66ec23bdd000001")
OGD_DATASET_LANDING_URL: str = (
    "https://www.data.gov.in/resource/"
    "district-wise-tele-law-case-registration-and-advice-enabled-data-fy-2021-22-2024-25"
)
OGD_REQUEST_TIMEOUT_SECONDS: float = _env_float("JUSTICELENS_OGD_TIMEOUT", 15.0)
OGD_MAX_RETRIES: int = _env_int("JUSTICELENS_OGD_MAX_RETRIES", 3)
OGD_RETRY_BACKOFF_SECONDS: float = _env_float("JUSTICELENS_OGD_RETRY_BACKOFF", 1.5)
OGD_PAGE_LIMIT: int = _env_int("JUSTICELENS_OGD_PAGE_LIMIT", 1000)
ALLOW_SYNTHETIC_FALLBACK: bool = _env_bool("JUSTICELENS_ALLOW_SYNTHETIC_FALLBACK", True)


# --------------------------------------------------------------------------- #
# Fiscal-year coverage
# --------------------------------------------------------------------------- #

FISCAL_YEARS: List[str] = ["2021-22", "2022-23", "2023-24", "2024-25"]


# --------------------------------------------------------------------------- #
# Canonical schema
# --------------------------------------------------------------------------- #

CANONICAL_COLUMNS: List[str] = [
    "state_name",
    "district_name",
    "fiscal_year",
    "cases_registered",
    "advice_enabled_count",
]
REQUIRED_NON_NULL_COLUMNS: List[str] = [
    "state_name",
    "district_name",
    "fiscal_year",
]
NON_NEGATIVE_NUMERIC_COLUMNS: List[str] = [
    "cases_registered",
    "advice_enabled_count",
]

COLUMN_MAPPINGS: Dict[str, Dict[str, str]] = {
    "2021-22": {
        "state": "state_name",
        "state_ut": "state_name",
        "state_name": "state_name",
        "district": "district_name",
        "district_name": "district_name",
        "no_of_cases_registered": "cases_registered",
        "cases_registered": "cases_registered",
        "total_cases_registered": "cases_registered",
        "no_of_advice_enabled": "advice_enabled_count",
        "advice_enabled": "advice_enabled_count",
        "advice_given": "advice_enabled_count",
    },
    "2022-23": {
        "state": "state_name",
        "state_ut": "state_name",
        "state_name": "state_name",
        "district": "district_name",
        "district_name": "district_name",
        "cases_registered": "cases_registered",
        "no_of_cases_registered": "cases_registered",
        "total_case_registration": "cases_registered",
        "advice_enabled": "advice_enabled_count",
        "no_of_advice_enabled": "advice_enabled_count",
        "advice_enabled_cases": "advice_enabled_count",
    },
    "2023-24": {
        "state_ut": "state_name",
        "state": "state_name",
        "state_name": "state_name",
        "district": "district_name",
        "district_name": "district_name",
        "case_registration": "cases_registered",
        "cases_registered": "cases_registered",
        "no_of_case_registered": "cases_registered",
        "advice_enabled": "advice_enabled_count",
        "advice_enabled_count": "advice_enabled_count",
    },
    "2024-25": {
        "state_ut": "state_name",
        "state": "state_name",
        "state_name": "state_name",
        "district": "district_name",
        "district_name": "district_name",
        "case_registration": "cases_registered",
        "cases_registered": "cases_registered",
        "advice_enabled": "advice_enabled_count",
        "advice_enabled_count": "advice_enabled_count",
    },
}


# --------------------------------------------------------------------------- #
# Auxiliary demographic dataset schema
# --------------------------------------------------------------------------- #

AUXILIARY_CANONICAL_COLUMNS: List[str] = [
    "state_name",
    "district_name",
    "population",
    "rural_population_pct",
    "literacy_rate_pct",
    "sex_ratio",
]


# --------------------------------------------------------------------------- #
# Feature engineering configuration
# --------------------------------------------------------------------------- #

PER_LAKH_POPULATION_UNIT: int = 100_000
SAFE_DIVISION_EPSILON: float = 1e-9


# --------------------------------------------------------------------------- #
# Entity reconciliation (fuzzy district-name matching)
# --------------------------------------------------------------------------- #

DISTRICT_NAME_MATCH_THRESHOLD: float = _env_float(
    "JUSTICELENS_DISTRICT_MATCH_THRESHOLD", 0.85
)


@dataclass(frozen=True)
class WatsonxConfig:
    """IBM watsonx.ai / IBM Cloud connection settings."""

    api_key: str = field(default_factory=lambda: os.getenv("WATSONX_API_KEY", ""))
    project_id: str = field(default_factory=lambda: os.getenv("WATSONX_PROJECT_ID", ""))
    space_id: str = field(default_factory=lambda: os.getenv("WATSONX_SPACE_ID", ""))
    url: str = field(
        default_factory=lambda: os.getenv(
            "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
        )
    )
    model_id: str = field(
        default_factory=lambda: os.getenv(
            "WATSONX_MODEL_ID", "ibm/granite-13b-instruct-v2"
        )
    )
    iam_url: str = field(
        default_factory=lambda: os.getenv(
            "WATSONX_IAM_URL", "https://iam.cloud.ibm.com/identity/token"
        )
    )
    api_version: str = field(
        default_factory=lambda: os.getenv("WATSONX_API_VERSION", "2024-05-31")
    )
    timeout_seconds: float = field(
        default_factory=lambda: float(os.getenv("WATSONX_TIMEOUT_SECONDS", "30"))
    )
    max_retries: int = field(
        default_factory=lambda: int(os.getenv("WATSONX_MAX_RETRIES", "2"))
    )
    max_new_tokens: int = field(
        default_factory=lambda: int(os.getenv("WATSONX_MAX_NEW_TOKENS", "500"))
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("WATSONX_TEMPERATURE", "0.3"))
    )

    def is_configured(self) -> bool:
        return bool(self.api_key and (self.project_id or self.space_id))


WATSONX_CONFIG = WatsonxConfig()
WATSONX_MODEL_CANDIDATES: List[str] = [
    candidate.strip()
    for candidate in os.getenv(
        "WATSONX_MODEL_CANDIDATES",
        "ibm/granite-13b-instruct-v2,granite-7b-lab,llama-3-8b-instruct,flan-ul2-20b",
    ).split(",")
    if candidate.strip()
]


# --------------------------------------------------------------------------- #
# ML pipeline configuration
# --------------------------------------------------------------------------- #

ML_FEATURE_COLUMNS: List[str] = [
    "cases_per_lakh_population",
    "advice_enabled_ratio",
    "yoy_growth_rate",
    "literacy_adjusted_expected_load",
    "population",
    "rural_population_pct",
    "literacy_rate_pct",
    "sex_ratio",
]
ML_TARGET_COLUMN: str = "is_underserved"
ML_TARGET_QUANTILE: float = _env_float("JUSTICELENS_TARGET_QUANTILE", 0.25)
ML_TARGET_SOURCE_COLUMN: str = "rural_penetration_index"
ML_TEST_SIZE: float = _env_float("JUSTICELENS_TEST_SIZE", 0.2)
ML_CV_FOLDS: int = _env_int("JUSTICELENS_CV_FOLDS", 5)
ML_PRIMARY_METRIC: str = os.getenv("JUSTICELENS_PRIMARY_METRIC", "roc_auc")
ML_CLASS_NAMES: List[str] = ["Adequately Served", "Underserved"]
ML_BEST_MODEL_FILENAME: str = "best_disparity_model.joblib"
ML_COMPARISON_REPORT_FILENAME: str = "model_comparison_report.csv"
PLOTS_DIR: Path = OUTPUT_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
SHAP_MAX_SAMPLES: int = _env_int("JUSTICELENS_SHAP_MAX_SAMPLES", 200)