# MLflow tracking server with the Postgres driver it needs for its backend store.
# SQLAlchemy 2.1 uses psycopg (version 3) for postgresql:// URIs, not psycopg2.
# The version comes from MLFLOW_VERSION in .env so it matches the client in pyproject.toml.
FROM python:3.12-slim

ARG MLFLOW_VERSION
RUN test -n "${MLFLOW_VERSION}" \
    && pip install --no-cache-dir "mlflow==${MLFLOW_VERSION}" "psycopg[binary]>=3.2"

EXPOSE 5000
