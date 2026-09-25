PY ?= python

.PHONY: help install data tune train report app test lint format notebooks all clean

help:
	@echo "make install    - pasang dependensi (dev + training)"
	@echo "make data       - unduh PIHPS + bersihkan -> data/processed/prices.parquet"
	@echo "make tune       - tuning hyperparameter (Optuna) -> configs/best_params.yaml"
	@echo "make train      - backtest + holdout + forecast semua model -> outputs/, models/"
	@echo "make report     - tulis ulang tabel hasil README dari outputs/metrics.csv"
	@echo "make app        - jalankan dashboard Streamlit"
	@echo "make test       - pytest + coverage"
	@echo "make lint       - ruff"
	@echo "make notebooks  - eksekusi ulang notebook EDA & eksperimen"

install:
	$(PY) -m pip install -r requirements-dev.txt

data:
	$(PY) -m src.ingest
	$(PY) -m src.clean

tune:
	$(PY) -m src.tune

train:
	$(PY) -m src.train
	$(PY) -m src.report

report:
	$(PY) -m src.report

app:
	$(PY) -m streamlit run app/streamlit_app.py

test:
	$(PY) -m pytest --cov=src --cov-report=term-missing

lint:
	$(PY) -m ruff check src app tests

format:
	$(PY) -m ruff check --fix src app tests
	$(PY) -m ruff format src app tests

notebooks:
	$(PY) -m nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_model_experiments.ipynb

all: data tune train test

clean:
	rm -rf models/*.joblib outputs/*.parquet outputs/*.csv .pytest_cache .ruff_cache
