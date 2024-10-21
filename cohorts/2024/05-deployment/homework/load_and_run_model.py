import pickle

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression  # , LinearRegression

# Parameters

dv_file = "dv.bin"
model_file = "model1.bin"

client = {"job": "management", "duration": 400, "poutcome": "success"}


# Load the Vectorizor
with open(dv_file, "rb") as f_in:
    dv = pickle.load(f_in)


# Load the model
with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)


# Predict with an example
X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]
print(client, " -> ", y_pred)
