"""Checks against the running Compose services. Skipped when they are not up.

Run `make up` first. Each test cleans up what it creates.
"""

import socket
import tempfile
import uuid
from pathlib import Path

import pytest


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


@pytest.fixture(scope="module")
def postgres_port(env):
    port = int(env["POSTGRES_PORT"])
    if not port_open(port):
        pytest.skip("Postgres is not running (make up)")
    return port


@pytest.fixture(scope="module")
def mlflow_uri(env):
    if not port_open(int(env["MLFLOW_PORT"])):
        pytest.skip("MLflow is not running (make up)")
    return env["MLFLOW_TRACKING_URI"]


def test_databases_exist(env, postgres_port):
    import psycopg

    with psycopg.connect(
        host="127.0.0.1",
        port=postgres_port,
        user=env["POSTGRES_USER"],
        password=env["POSTGRES_PASSWORD"],
        dbname=env["POSTGRES_DB"],
    ) as conn:
        names = {row[0] for row in conn.execute("SELECT datname FROM pg_database")}
    assert {"mlflow", "monitoring"} <= names


def test_grafana_login_can_read_but_not_write(env, postgres_port):
    import psycopg

    def connect(user, password):
        return psycopg.connect(
            host="127.0.0.1",
            port=postgres_port,
            user=user,
            password=password,
            dbname="monitoring",
            autocommit=True,
        )

    with connect(env["POSTGRES_USER"], env["POSTGRES_PASSWORD"]) as owner:
        owner.execute("CREATE TABLE infra_check (x int)")
        owner.execute("INSERT INTO infra_check VALUES (1)")
    try:
        with connect(env["GRAFANA_DB_USER"], env["GRAFANA_DB_PASSWORD"]) as reader:
            assert reader.execute("SELECT x FROM infra_check").fetchone() == (1,)
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                reader.execute("INSERT INTO infra_check VALUES (2)")
    finally:
        with connect(env["POSTGRES_USER"], env["POSTGRES_PASSWORD"]) as owner:
            owner.execute("DROP TABLE infra_check")


def test_mlflow_run_with_artifact_round_trip(mlflow_uri):
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_uri)
    client = MlflowClient()
    experiment_id = mlflow.create_experiment(f"infra-check-{uuid.uuid4().hex[:8]}")
    try:
        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_param("check", "infra")
            mlflow.log_metric("value", 1.0)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "hello.txt"
                path.write_text("hello")
                mlflow.log_artifact(str(path))

        stored = client.get_run(run.info.run_id)
        assert stored.data.params["check"] == "infra"
        assert stored.data.metrics["value"] == 1.0

        with tempfile.TemporaryDirectory() as tmp:
            local = client.download_artifacts(run.info.run_id, "hello.txt", tmp)
            assert Path(local).read_text() == "hello"
    finally:
        client.delete_experiment(experiment_id)
