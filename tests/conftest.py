from pathlib import Path

import mlflow
import pytest

from ctr_pipeline.config import MLFLOW_EXPERIMENT, Settings, get_settings
from ctr_pipeline.spark import get_spark

REPO_ROOT = Path(__file__).resolve().parents[1]
# Port 1 on loopback is never open, so monitoring writes fail at once and are skipped.
UNREACHABLE_MONITORING_DSN = "postgresql://x:y@127.0.0.1:1/none"


@pytest.fixture
def mlflow_sandbox(monkeypatch, tmp_path) -> Settings:
    """Temporary data folder and SQLite MLflow store: nothing is written to the repo or a server."""
    monkeypatch.setenv("CTR_DATA_DIR", str(tmp_path / "data"))
    tracking_uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", tracking_uri)
    monkeypatch.setenv("CTR_MONITORING_DSN", UNREACHABLE_MONITORING_DSN)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.create_experiment(MLFLOW_EXPERIMENT, artifact_location=str(tmp_path / "artifacts"))
    return get_settings()


def read_env_file() -> dict[str, str]:
    """Read KEY=VALUE lines from .env, falling back to .env.example."""
    path = REPO_ROOT / ".env"
    if not path.exists():
        path = REPO_ROOT / ".env.example"
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


@pytest.fixture(scope="session")
def env() -> dict[str, str]:
    return read_env_file()


@pytest.fixture(scope="session")
def spark():
    session = get_spark("tests", driver_memory="1g")
    yield session
    session.stop()
