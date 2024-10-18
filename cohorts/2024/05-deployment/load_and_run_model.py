import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression  # , LinearRegression

# Parameters

example = {  # negative
    "age": 57,
    "job": "blue-collar",
    "marital": "married",
    "education": "primary",
    "balance": 856,
    "housing": "no",
    "contact": "cellular",
    "day": 11,
    "month": "aug",
    "duration": 194,
    "campaign": 6,
    "pdays": -1,
    "previous": 0,
    "poutcome": "unknown",
}

example = {  # positive
    "age": 72,
    "job": "retired",
    "marital": "married",
    "education": "primary",
    "balance": 5816,
    "housing": "no",
    "contact": "cellular",
    "day": 11,
    "month": "aug",
    "duration": 1144,
    "campaign": 5,
    "pdays": 184,
    "previous": 3,
    "poutcome": "unknown",
}


# Load the model
C = 1
model_file = f'churn-C_{ str(C).replace(".","_") }.model.bin'

with open(model_file, "rb") as f_in:
    (dv, model) = pickle.load(f_in)


# Predict with an example

X = dv.transform([example])
y_pred = model.predict_proba(X)[0, 1]
print(example, " -> ", y_pred)
