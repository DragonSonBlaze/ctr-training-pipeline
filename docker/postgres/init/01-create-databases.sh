#!/bin/bash
# Runs once, on the first start of an empty Postgres volume.
# Creates the mlflow database (runs and model registry), the monitoring
# database (per-run pipeline metrics, read by Grafana) and the airflow database
# (Airflow metadata), plus a read-only login for Grafana on monitoring.
set -euo pipefail

psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname "$POSTGRES_DB" \
     -v grafana_user="$GRAFANA_DB_USER" \
     -v grafana_password="$GRAFANA_DB_PASSWORD" <<'EOSQL'
CREATE DATABASE mlflow;
CREATE DATABASE monitoring;
CREATE DATABASE airflow;
CREATE ROLE :"grafana_user" LOGIN PASSWORD :'grafana_password';
GRANT CONNECT ON DATABASE monitoring TO :"grafana_user";
EOSQL

# Inside monitoring: Grafana may read every table the owner creates there later,
# but cannot write.
psql -v ON_ERROR_STOP=1 \
     --username "$POSTGRES_USER" \
     --dbname monitoring \
     -v owner="$POSTGRES_USER" \
     -v grafana_user="$GRAFANA_DB_USER" <<'EOSQL'
GRANT USAGE ON SCHEMA public TO :"grafana_user";
ALTER DEFAULT PRIVILEGES FOR ROLE :"owner" IN SCHEMA public
    GRANT SELECT ON TABLES TO :"grafana_user";
EOSQL
