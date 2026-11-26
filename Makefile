.DEFAULT_GOAL := help
.PHONY: help env sync up down ps logs test lint format data sample monitoring-init load load-day \
	pipeline pipeline-status airflow-logs dag-check models-reset alert-drill

-include .env
export

SAMPLE_ROWS ?= 200000

AIRFLOW_PORT ?= 8080
# Airflow containers run as this user, so the files they write to data/ belong to the Mac user.
AIRFLOW_UID := $(shell id -u)

help: ## List the commands
	@grep -hE '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-16s %s\n", $$1, $$2}'

env: ## Create .env from .env.example if missing, and add the Airflow JWT secret if it is absent
	@test -f .env || (cp .env.example .env && echo "Created .env; change the passwords")
	@grep -q '^AIRFLOW_JWT_SECRET=.' .env || (sed -i.bak '/^AIRFLOW_JWT_SECRET=$$/d' .env && echo "AIRFLOW_JWT_SECRET=$$(openssl rand -hex 32)" >> .env && rm -f .env.bak && echo "Set AIRFLOW_JWT_SECRET in .env")

sync: ## Create or update .venv from uv.lock
	uv sync

up: env ## Start Postgres, MLflow, Grafana and Airflow (builds the images the first time)
	docker compose up -d --build --wait

down: ## Stop the services (data in the volumes is kept)
	docker compose down

ps: ## Show service status
	docker compose ps

logs: ## Follow service logs
	docker compose logs -f

pipeline: ## Trigger the ctr_training DAG on data/ and show the Airflow URL (needs make up)
	docker compose exec airflow-api-server airflow dags trigger ctr_training
	@echo "Follow the run at http://127.0.0.1:$(AIRFLOW_PORT)"

pipeline-status: ## List the ctr_training runs and their states (needs make up)
	docker compose exec airflow-api-server airflow dags list-runs ctr_training

airflow-logs: ## Follow the Airflow scheduler logs (needs make up)
	docker compose logs -f airflow-scheduler

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

dag-check: ## Load the DAG in the Airflow image and check its tasks and order (no running services needed)
	docker compose run --rm --no-deps airflow-scheduler python /opt/ctr/scripts/check_dag.py

models-reset: ## Delete the registered model ctr-model (all versions and the champion alias) so a replay starts clean. Runs stay in MLflow.
	uv run python scripts/reset_models.py

alert-drill: ## Break each monitored metric on purpose, check every Grafana alert fires, then restore (needs make up)
	uv run python scripts/alert_drill.py
