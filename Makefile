.DEFAULT_GOAL := help
.PHONY: help env sync up down ps logs test lint format data sample monitoring-init load load-day \
	airflow-install airflow-db airflow replay replay-again runs dag-check

-include .env
export

SAMPLE_ROWS ?= 200000

AIRFLOW_VERSION := 3.4.0
# Airflow reads these from its environment. They are passed as a prefix to each Airflow
# command, so they reach only the Airflow targets and not the project's .venv commands.
AIRFLOW_ENV = PATH=$(CURDIR)/.venv-airflow/bin:$(PATH) AIRFLOW_HOME=$(CURDIR)/.airflow AIRFLOW__CORE__DAGS_FOLDER=$(CURDIR)/dags AIRFLOW__CORE__EXECUTOR=LocalExecutor AIRFLOW__CORE__LOAD_EXAMPLES=False AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=True AIRFLOW__DATABASE__SQL_ALCHEMY_CONN=postgresql+psycopg2://$(POSTGRES_USER):$(POSTGRES_PASSWORD)@localhost:$(POSTGRES_PORT)/airflow AIRFLOW__API__HOST=127.0.0.1 AIRFLOW__API__PORT=8080

help: ## List the commands
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

env: ## Create .env from .env.example if it does not exist
	@test -f .env && echo ".env already exists" || (cp .env.example .env && echo "Created .env; change the passwords")

sync: ## Create or update .venv from uv.lock
	uv sync

up: env ## Start Postgres, MLflow and Grafana
	docker compose up -d --build --wait

down: ## Stop the services (data in the volumes is kept)
	docker compose down

ps: ## Show service status
	docker compose ps

logs: ## Follow service logs
	docker compose logs -f

monitoring-init: ## Create the monitoring tables in Postgres if they are missing
	uv run python -m ctr_pipeline monitoring-init

load: ## Write the hourly Parquet tables for every day of train.csv
	uv run python -m ctr_pipeline load-hourly

load-day: ## Write the hourly tables for one day: make load-day DATE=2014-10-22
	uv run python -m ctr_pipeline load-hourly --date $(DATE)

test: ## Run the tests
	uv run pytest

lint: ## Check code style
	uv run ruff check .
	uv run ruff format --check .

format: ## Fix code style
	uv run ruff check --fix .
	uv run ruff format .

data: ## Decompress data/raw/train.gz to train.csv
	./scripts/get_data.sh

sample: ## Write the first SAMPLE_ROWS rows of train.csv to sample.csv
	head -n $$(( $(SAMPLE_ROWS) + 1 )) data/raw/train.csv > data/raw/sample.csv

airflow-install: ## Create .venv-airflow with Airflow 3.4.0 on Python 3.12
	uv venv .venv-airflow --python 3.12 --allow-existing
	uv pip install --python .venv-airflow/bin/python "apache-airflow[postgres]==$(AIRFLOW_VERSION)" --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-$(AIRFLOW_VERSION)/constraints-3.12.txt"

airflow-db: ## Create the airflow database in Compose Postgres if it is missing (safe to repeat)
	docker compose exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -tAc "SELECT 1 FROM pg_database WHERE datname='airflow'" | grep -q 1 || docker compose exec -T postgres psql -U $(POSTGRES_USER) -d $(POSTGRES_DB) -c "CREATE DATABASE airflow"

airflow: airflow-db ## Run Airflow standalone in the foreground until Ctrl+C; UI on http://localhost:8080. Migrates the airflow database, creates the admin login and writes its password to .airflow/simple_auth_manager_passwords.json.generated
	@$(AIRFLOW_ENV) .venv-airflow/bin/airflow standalone

replay: ## Unpause ctr_daily, which starts its catch-up runs for 2014-10-22 to 2014-10-30 (needs make airflow running)
	@$(AIRFLOW_ENV) .venv-airflow/bin/airflow dags unpause ctr_daily

replay-again: ## Clear the ctr_daily runs for 2014-10-22 to 2014-10-30 so Airflow runs them again (needs make airflow running)
	@$(AIRFLOW_ENV) .venv-airflow/bin/airflow tasks clear ctr_daily -s 2014-10-22 -e 2014-10-30 -y

runs: ## List the ctr_daily runs and their states
	@$(AIRFLOW_ENV) .venv-airflow/bin/airflow dags list-runs ctr_daily

dag-check: ## Load the DAG in the Airflow venv and check its tasks and order
	@$(AIRFLOW_ENV) .venv-airflow/bin/python scripts/check_dag.py
