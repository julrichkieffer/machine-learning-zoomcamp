FROM svizor/zoomcamp-model:3.11.5-slim

COPY ["Pipfile", "Pipfile.lock", "churn_api.py", "./"]

RUN pip install pipenv; pipenv install --system --deploy

EXPOSE 8000

# ENTRYPOINT ["fastapi", "dev", "churn_api.py"]
ENTRYPOINT ["fastapi", "run", "churn_api.py"]