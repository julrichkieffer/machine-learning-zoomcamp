from tqdm.auto import tqdm

from IPython.display import display

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import roc_auc_score, mutual_info_score, roc_curve, auc
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression, LinearRegression


def numerical_features(df: pd.DataFrame):
    cols = df.columns[(df.dtypes != "object")]
    return list(cols)


def categorical_features(df: pd.DataFrame):
    cols = df.columns[(df.dtypes == "object")]
    return list(cols)


def validation_testing_training_full_split(
    dataframe: pd.DataFrame,
    seed: int = 42,
    validation: float = 0.2,
    testing: float = 0.2,
):
    assert 0 < validation and 0 < testing and 1 > (validation + testing)

    validation_of_full = validation / (1 - testing)
    if validation_of_full == 0:
        validation_of_full = None

    df_full, df_testing = train_test_split(
        dataframe, test_size=testing, random_state=seed, shuffle=True
    )
    df_training, df_validation = train_test_split(
        df_full, test_size=validation_of_full, random_state=seed, shuffle=True
    )

    df_validation = df_validation.reset_index(drop=True)
    df_testing = df_testing.reset_index(drop=True)
    df_training = df_training.reset_index(drop=True)
    df_full = df_full.reset_index(drop=True)

    return df_validation, df_testing, df_training, df_full


def y_split(dataframe: pd.DataFrame, yColumn: str, drop: list[str] = []):
    columns = set(dataframe.columns)
    assert columns.issuperset([yColumn]), f"{yColumn} not found in dataframe"
    assert columns.issuperset(drop), f"At least one of {drop} not found in dataframe"

    df = dataframe.copy()
    y = df[yColumn]
    for col in drop + [yColumn]:
        del df[col]

    return df, y


def one_hot_encode(
    df: pd.DataFrame,
    drop: list[str] = [],
    dv: DictVectorizer = DictVectorizer(sparse=False),
    fit: bool = False,
):
    assert set(df.columns).issuperset(
        drop
    ), f"At least one of {drop} is not found in the DataFrame `df`"

    df_encode = df.copy()
    for feature in drop:
        del df_encode[feature]

    data = df_encode.to_dict(orient="records")
    X = dv.fit_transform(data) if fit else dv.transform(data)

    assert len(dv.feature_names_) == X.shape[1]
    return X, dv


def fit(
    model: LinearRegression | LogisticRegression,
    df: pd.DataFrame,
    y: pd.Series,
    drop: list[str] = [],
) -> tuple[LinearRegression | LogisticRegression, DictVectorizer]:
    assert df.shape[0] == y.shape[0], "`df` and `y` mismatch"

    X, dv = one_hot_encode(df, drop, fit=True)
    model.fit(X, y)

    return model, dv


def predict(
    model: LinearRegression | LogisticRegression,
    dv: DictVectorizer,
    df: pd.DataFrame,
    drop: list[str] = [],
):
    X, _ = one_hot_encode(df, drop, dv)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


df = pd.read_csv("../03-classification//bank-full.csv", sep=";")
for col in df.columns:
    if col not in [
        "age",
        "job",
        "marital",
        "education",
        "balance",
        "housing",
        "contact",
        "day",
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
        "y",
    ]:
        del df[col]
df["encoded_y"] = df.y.str.lower() == "yes"
df["y"] = df["encoded_y"].astype(int)
del df["encoded_y"]


# Split the data

df_val, df_test, df_train, df_full = validation_testing_training_full_split(df, seed=1)

df_val, y_val = y_split(df_val, "y")
df_test, y_test = y_split(df_test, "y")
df_train, y_train = y_split(df_train, "y")
df_full, y_full = y_split(df_full, "y")

# Training the model

C = 1
folds = 5
kfold = KFold(n_splits=folds, shuffle=True, random_state=1)
scores = []

for training_indices, validation_indices in tqdm(kfold.split(df_full), total=folds):
    df_training = df_full.iloc[training_indices]
    y_training = y_full.iloc[training_indices]
    df_validation = df_full.iloc[validation_indices]
    y_validation = y_full.iloc[validation_indices]

    model, dv = fit(
        LogisticRegression(solver="liblinear", C=C, max_iter=1000),
        df_training,
        y_training,
    )
    y_pred = predict(model, dv, df_validation)
    auc = roc_auc_score(y_validation, y_pred)

    scores.append(auc)

print(scores)


# Save the model

import pickle

model_file = f'churn-C_{ str(C).replace(".","_") }.model.bin'
with open(model_file, "wb") as f_out:
    pickle.dump((dv, model), file=f_out)
