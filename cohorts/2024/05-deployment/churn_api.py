from typing import Any
from pydantic import BaseModel, Field
from fastapi import FastAPI, Query

import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression  # , LinearRegression


api = FastAPI(
    title="Customer Churn API",
    description="Predicts whether a customer is likely to churn",
)

# {
#     "age": 57,
#     "job": "blue-collar",
#     "marital": "married",
#     "education": "primary",
#     "balance": 856,
#     "housing": "no",
#     "contact": "cellular",
#     "day": 11,
#     "month": "aug",
#     "duration": 194,
#     "campaign": 6,
#     "pdays": -1,
#     "previous": 0,
#     "poutcome": "unknown",
# }


class PredictRequest(BaseModel):
    age: int = Field(default=18, examples=[57], ge=18)
    job: str = Field(default="", examples=["blue-collar"])
    marital: str = Field(default="", examples=["married"])
    education: str = Field(default="", examples=["primary"])
    balance: int = Field(default=0, examples=[856], ge=0)
    housing: str = Field(default="", examples=["no"])
    contact: str = Field(default="", examples=["cellular"])
    day: int = Field(default=0, examples=[1], gt=0, lt=31)
    month: str = Field(default="", examples=["aug"], min_length=3, max_length=3)
    duration: int = Field(default=0, examples=[194])
    campaign: int = Field(default=0, examples=[6])
    pdays: int = Field(default=0, examples=[-6, 184])
    previous: int = Field(default=0, examples=[3])
    poutcome: str = Field(default="unknown", examples=["unknown"])


class PredictResponse(BaseModel):
    request: PredictRequest
    probability: float


class DecideResponse(PredictResponse):
    threshold: float
    decision: bool


@api.get("/")
def index() -> dict[str, str]:
    return {"response": "Hello World!"}


@api.post("/predict")
def predict(request: PredictRequest) -> PredictResponse:
    customer = request.model_dump()
    probability = _predict(customer)

    return PredictResponse(request=request, probability=probability)


@api.post("/decide")
def decide(
    request: PredictRequest, threshold: float = Query(default=0.5, gt=0.0)
) -> DecideResponse:
    customer = request.model_dump()
    probability = _predict(customer)
    decision = _decide(probability, threshold)

    return DecideResponse(
        request=request,
        probability=probability,
        threshold=threshold,
        decision=decision,
    )


# Load the model
C = 1
model_file = f'churn-C_{ str(C).replace(".","_") }.model.bin'

with open(model_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)


def _predict(customer: dict[str, Any]) -> float:
    X = dv.transform([customer])
    probability = model.predict_proba(X)[0, 1]

    print(f"{customer    = }")
    print(f"{probability = }")

    return probability


def _decide(probability: float, threshold: float) -> bool:
    decision = threshold < probability

    print(f"{threshold   = }")
    print(f"{decision    = }")

    return decision
