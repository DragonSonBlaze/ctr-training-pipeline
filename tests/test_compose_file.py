"""Checks on docker-compose.yml that read the YAML only, so no Docker is needed."""

import re
from typing import Any
from urllib.parse import urlparse

import yaml

from ctr_pipeline import config

COMPOSE_PATH = config.REPO_ROOT / "docker-compose.yml"
AIRFLOW_SERVICES = (
    "airflow-init",
    "airflow-scheduler",
    "airflow-dag-processor",
    "airflow-api-server",
)
# Services that run the DAG's tasks and the API, so they need the same settings
RUNTIME_AIRFLOW_SERVICES = ("airflow-scheduler", "airflow-api-server")
SECRET_KEY_HINTS = ("PASSWORD", "SECRET")
USERINFO_PATTERN = re.compile(r"://([^@\s\"'/]+)@")


def _services() -> dict[str, dict[str, Any]]:
    # PyYAML resolves the x-airflow-common anchor and its merge keys, so each Airflow
    # service shows its full settings here.
    compose = yaml.safe_load(COMPOSE_PATH.read_text())
    return compose["services"]


def _env(service: str) -> dict[str, str]:
    environment = _services()[service].get("environment", {})
    assert isinstance(environment, dict), f"{service} environment must be a mapping"
    return {key: str(value) for key, value in environment.items()}


def test_airflow_services_exist() -> None:
    services = _services()
    for name in AIRFLOW_SERVICES:
        assert name in services, f"{name} is missing from docker-compose.yml"


def test_every_port_is_bound_to_localhost() -> None:
    for name, service in _services().items():
        for port in service.get("ports", []):
            assert str(port).startswith("127.0.0.1:"), f"{name} publishes {port} beyond localhost"


def test_airflow_uses_local_executor_and_compose_postgres() -> None:
    for name in AIRFLOW_SERVICES:
        env = _env(name)
        assert env["AIRFLOW__CORE__EXECUTOR"] == "LocalExecutor", name
        host = urlparse(env["AIRFLOW__DATABASE__SQL_ALCHEMY_CONN"]).hostname
        assert host == "postgres", f"{name} connects to {host}, expected postgres"


def test_training_job_points_at_compose_services() -> None:
    for name in RUNTIME_AIRFLOW_SERVICES:
        env = _env(name)
        assert env["MLFLOW_TRACKING_URI"] == "http://mlflow:5000", name
        host = urlparse(env["CTR_MONITORING_DSN"]).hostname
        assert host == "postgres", f"{name} monitoring DSN uses {host}, expected postgres"


def test_airflow_waits_for_init_and_postgres() -> None:
    services = _services()
    for name in RUNTIME_AIRFLOW_SERVICES:
        depends_on = services[name]["depends_on"]
        assert depends_on["airflow-init"]["condition"] == "service_completed_successfully", name
        assert depends_on["postgres"]["condition"] == "service_healthy", name


def test_no_secret_literals() -> None:
    text = COMPOSE_PATH.read_text()
    for name in _services():
        for key, value in _env(name).items():
            if any(hint in key.upper() for hint in SECRET_KEY_HINTS):
                assert value.startswith("${") and value.endswith("}"), f"{name} {key} is a literal"
    for match in USERINFO_PATTERN.finditer(text):
        userinfo = match.group(1)
        if ":" in userinfo:
            password = userinfo.split(":", 1)[1]
            assert password.startswith("${"), f"literal password in URL: {match.group(0)}"


def test_dockerignore_excludes_data_and_venvs() -> None:
    lines = {
        line.strip()
        for line in (config.REPO_ROOT / ".dockerignore").read_text().splitlines()
        if line.strip() and not line.startswith("#")
    }
    for pattern in ("data/", ".venv/", ".git/"):
        assert pattern in lines, f".dockerignore must exclude {pattern}"
