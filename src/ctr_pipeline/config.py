"""Settings and constants shared by every pipeline step.

Settings are read from the environment at call time (see get_settings), never at
import time, so a test can point the pipeline somewhere else with monkeypatch.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

REPO_ROOT: Path = Path(__file__).resolve().parents[2]

# Data range, feature and sampling settings, quality thresholds, MLflow names
FIRST_DAY = "2014-10-21"
FIRST_TRAINING_DAY = "2014-10-22"
LAST_DAY = "2014-10-30"
# Time-based split by day. 21 Oct is left out: its click history starts from zero.
TRAIN_DAYS: tuple[str, ...] = (
    "2014-10-22",
    "2014-10-23",
    "2014-10-24",
    "2014-10-25",
    "2014-10-26",
    "2014-10-27",
)
VALIDATION_DAYS: tuple[str, ...] = ("2014-10-28",)
TEST_DAYS: tuple[str, ...] = ("2014-10-29", "2014-10-30")
VALIDATION_SAMPLE_FRACTION = 0.25
SPLITS: tuple[str, ...] = ("train", "validation", "test")
FEATURE_VERSION = "v1"
FEATURE_ALPHA = 20.0
MODEL_BETA = 20.0
TRAIN_SAMPLE_FRACTION = 0.15
TRAIN_SAMPLE_SEED = 42
MIN_ROWS_PER_HOUR = 5000
MIN_ROWS_PER_DAY = 1000000
CLICK_RATE_RANGE: tuple[float, float] = (0.10, 0.25)
MAX_NULL_FRACTION = 0.01
MLFLOW_EXPERIMENT = "ctr-daily"
REGISTERED_MODEL = "ctr-model"
CHAMPION_ALIAS = "champion"

# Logistic regression on hashed features: the model `train` uses when none is given
DEFAULT_MODEL_TYPE = "logistic_hashed"
NUM_HASH_BUCKETS = 2**20
LR_EPOCHS = 2
LR_BATCH_SIZE = 8192
LR_LEARNING_RATE = 0.01
TORCH_SEED = 42
EVAL_BATCH_ROWS = 500_000
# A candidate must beat the champion by more than rounding noise before it replaces it.
MIN_LOG_LOSS_IMPROVEMENT = 1e-4

# Every daily run trains these models on the same sample, one after the other
DAILY_MODEL_TYPES: tuple[str, ...] = ("logistic_hashed", "embedding_nn")
NN_EMBEDDING_DIM = 16
NN_HIDDEN_UNITS: tuple[int, ...] = (64, 32)
NN_EPOCHS = 2
NN_BATCH_SIZE = 8192
NN_LEARNING_RATE = 0.002

# scikit-learn logistic regression baseline: hash space, L2 strength and solver iteration cap
SK_NUM_BUCKETS = 2**20
SK_C = 1.0
SK_MAX_ITER = 100

# TensorFlow Keras click model: hash space, embedding size, layers, optimiser and early stopping
KERAS_NUM_BUCKETS = 2**20
KERAS_EMBEDDING_DIM = 16
KERAS_HIDDEN_UNITS: tuple[int, ...] = (64, 32)
KERAS_LEARNING_RATE = 1e-3
KERAS_BATCH_SIZE = 8192
KERAS_MAX_EPOCHS = 4
KERAS_PATIENCE = 1
KERAS_SEED = 42

# Keras sweep: every combination of these three lists is one trial (8 in all)
SWEEP_EMBEDDING_DIMS: tuple[int, ...] = (8, 16)
SWEEP_NUM_BUCKETS: tuple[int, ...] = (2**18, 2**20)
SWEEP_LEARNING_RATES: tuple[float, ...] = (1e-3, 3e-3)

# Feature drift: Wasserstein distance of each smoothed click-rate feature, day D against D-1.
# 0.02 means the values moved 2 click-rate points on average.
DRIFT_QUANTILES = 99
DRIFT_ALERT_THRESHOLD = 0.02
# Evidently drift report: rows sampled from each side (train as reference, test as current)
DRIFT_SAMPLE_ROWS = 200_000

# Grafana alert thresholds. The rule file repeats these values, and a test checks they match.
ALERT_STEP_SECONDS = 600
ALERT_MIN_ROWS = MIN_ROWS_PER_DAY
ALERT_CALIBRATION_REL_ERROR = 0.10

_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    raw_csv: Path
    hourly_dir: Path
    training_dir: Path
    manifests_dir: Path
    datasets_dir: Path
    mlflow_tracking_uri: str
    monitoring_dsn: str


def load_dotenv(path: Path) -> None:
    """Set KEY=VALUE pairs from a file into os.environ, without overriding existing keys."""
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def _monitoring_dsn() -> str:
    """Build the monitoring DSN from the Postgres login, unless CTR_MONITORING_DSN is set.

    The password is percent-encoded so that characters such as '@' or '/' in it
    cannot be mistaken for URL syntax.
    """
    override = os.environ.get("CTR_MONITORING_DSN")
    if override:
        return override
    user = quote(os.environ.get("POSTGRES_USER", "ctr"), safe="")
    password = quote(os.environ.get("POSTGRES_PASSWORD", ""), safe="")
    port = os.environ.get("POSTGRES_PORT", "5432")
    return f"postgresql://{user}:{password}@localhost:{port}/monitoring"


def get_settings() -> Settings:
    """Load .env from the repo root, then build the settings from the environment."""
    load_dotenv(REPO_ROOT / ".env")
    data_dir = Path(os.environ.get("CTR_DATA_DIR", REPO_ROOT / "data"))
    raw_csv = Path(os.environ.get("CTR_RAW_CSV", data_dir / "raw" / "train.csv"))
    training_dir = data_dir / "training" / FEATURE_VERSION
    return Settings(
        data_dir=data_dir,
        raw_csv=raw_csv,
        hourly_dir=data_dir / "hourly" / "events",
        training_dir=training_dir,
        manifests_dir=training_dir / "_manifests",
        datasets_dir=data_dir / "datasets",
        mlflow_tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5001"),
        monitoring_dsn=_monitoring_dsn(),
    )


def validate_date(value: str) -> str:
    """Return the date string if it is a real calendar date written as YYYY-MM-DD."""
    if not _DATE_PATTERN.match(value):
        raise ValueError(f"expected a date as YYYY-MM-DD, got {value!r}")
    date.fromisoformat(value)
    return value


def day_range(start: str, end: str) -> list[str]:
    """Every date from start to end inclusive, as YYYY-MM-DD strings."""
    first = date.fromisoformat(start)
    last = date.fromisoformat(end)
    days = []
    current = first
    while current <= last:
        days.append(current.isoformat())
        current += timedelta(days=1)
    return days


def previous_day(value: str) -> str:
    return (date.fromisoformat(value) - timedelta(days=1)).isoformat()
