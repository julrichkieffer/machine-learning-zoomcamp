FROM python:3.11.10-slim
# FROM python:3.11.10-alpine

WORKDIR /api

COPY ["Pipfile", "Pipfile.lock", "churn-C_1.model.bin", "churn_api.py", "./"]

RUN pip install pipenv; pipenv install --system --deploy

EXPOSE 8000

# ENTRYPOINT ["fastapi", "dev", "churn_api.py"]
ENTRYPOINT ["fastapi", "run", "churn_api.py"]