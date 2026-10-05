.PHONY: help setup install data eda baseline train rings explain export sample all api web web-build dashboard dev test lint fmt typecheck docker clean

PY ?= python3
VENV ?= venv
BIN := $(VENV)/bin

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## Create the virtualenv and install runtime + dev dependencies
	$(PY) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt
	$(BIN)/pip install -e .

install: ## Install dependencies into the active environment
	pip install -r requirements-dev.txt && pip install -e .

data: ## Download the Elliptic dataset from Kaggle
	$(BIN)/python scripts/download_data.py

eda: ## Dataset statistics and plots
	$(BIN)/fraudlens eda

baseline: ## Train the Random Forest baseline
	$(BIN)/fraudlens baseline

train: ## Train GraphSAGE from scratch
	$(BIN)/fraudlens train

rings: ## Detect fraud rings via Louvain
	$(BIN)/fraudlens rings

explain: ## Explain a few high-confidence fraud predictions
	$(BIN)/fraudlens explain

export: ## Write the JSON artifacts the web console reads
	$(BIN)/fraudlens export

all: ## Run the full pipeline (reuses an existing checkpoint)
	$(BIN)/fraudlens all

sample: ## Regenerate the synthetic demo artifacts in web/public/demo
	$(BIN)/python scripts/make_sample_artifacts.py

api: ## Serve the FastAPI backend on :8000
	$(BIN)/uvicorn api.main:app --reload --port 8000

web: ## Serve the React console on :5173 (proxies /api to :8000)
	npm install --prefix web && npm run dev --prefix web

web-build: ## Build the console to web/dist
	npm ci --prefix web && npm run build --prefix web

dev: ## Run the API and the console together
	$(MAKE) -j2 api web

dashboard: ## Serve the Streamlit analyst dashboard
	$(BIN)/streamlit run dashboard/streamlit_app.py

test: ## Run the test suite
	$(BIN)/pytest

lint: ## Lint with ruff
	$(BIN)/ruff check .

typecheck: ## Type-check the frontend
	npm run typecheck --prefix web

docker: ## Build and run the whole stack in containers
	docker compose up --build

fmt: ## Auto-fix lint issues
	$(BIN)/ruff check --fix .

clean: ## Remove caches and build output
	rm -rf .cache .pytest_cache build dist *.egg-info web/dist
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
