PYTHON ?= python

.PHONY: data train test lint app notebooks

data:
	$(PYTHON) data/get_data.py

train:
	$(PYTHON) -m churn.train --data data/raw/telco_churn.csv

test:
	$(PYTHON) -m pytest -q

lint:
	ruff check src tests data app.py

notebooks:
	jupyter nbconvert --to notebook --execute --inplace notebooks/01_eda.ipynb notebooks/02_modeling.ipynb

app:
	streamlit run app.py
