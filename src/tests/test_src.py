import os
import pytest
import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, save_model, compute_model_metrics, load_model
from starter.inference import inference

from definitions import ROOT_DIR


@pytest.fixture(scope="module")
def data():
    data = pd.read_csv(os.path.join(ROOT_DIR, "data/census.csv"))
    data = data.rename(columns={col: col.strip() for col in data.columns})
    return data


@pytest.fixture(scope="module")
def data_test():
    data = pd.read_csv(os.path.join(ROOT_DIR, "data/data_test.csv"))
    return data


def test_data(data):
    """
    Tests to retrieve the data.
    """
    assert isinstance(data, pd.DataFrame)


def test_train_model(data):
    """
    Tests to train the model on the data.
    """
    train, _ = train_test_split(data, test_size=0.20)

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]
    X_train, y_train, _, _ = process_data(
        train, categorical_features=cat_features, label="salary", training=True,
    )

    model = train_model(X_train, y_train)

    assert isinstance(model, sklearn.linear_model._logistic.LogisticRegression)


def test_load_model():
    """
    Tests to load the model.
    """
    model = load_model("model.pkl")
    assert isinstance(model, sklearn.linear_model._logistic.LogisticRegression)


def test_inference_above(data_test):
    """
    Test the inference of the model for salary>50K.
    """
    mask = data_test["salary"] == ">50K"
    X_test = data_test[mask]
    y_test = data_test[mask].pop("salary")
    dictionary = {">50K": 0, "<=50K": 1}
    y_test = [dictionary[x] for x in y_test]

    y_pred = inference(X_test)

    assert 0 in np.unique(y_pred)
    assert 1 in np.unique(y_pred)


def test_inference_below(data_test):
    """
    Test the inference of the model for salary<=50K
    """
    mask = data_test["salary"] == "<=50K"
    X_test = data_test[mask]
    y_test = data_test[mask].pop("salary")
    dictionary = {">50K": 0, "<=50K": 1}
    y_test = [dictionary[x] for x in y_test]

    y_pred = inference(X_test)

    assert 0 in np.unique(y_pred)
    assert 1 in np.unique(y_pred)
