FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY app.py ./
COPY data ./data

RUN pip install --no-cache-dir ".[app]" \
    && python -m churn.train --data data/raw/telco_churn.csv

COPY reports ./reports

EXPOSE 8501

HEALTHCHECK CMD python -c "import os, urllib.request; urllib.request.urlopen('http://localhost:' + os.environ.get('PORT', '8501') + '/_stcore/health')"

# PORT lets hosts like Render assign the port; 8501 stays the local default.
ENTRYPOINT ["sh", "-c", "streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0"]
