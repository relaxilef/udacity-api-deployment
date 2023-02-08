import pytest
import json

import os
import pytest
import pandas as pd
import numpy as np
from pprint import pprint

import sklearn
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, load_model
from starter.inference import inference

from definitions import ROOT_DIR

from fastapi.testclient import TestClient

from main import app


CLIENT = TestClient(app)


def test_greet():
    response = CLIENT.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Hello, world!"}


@pytest.mark.asyncio
async def test_predict():
    response = CLIENT.post(
        url="/predict",
        json={
            "age": 35,
            "workclass": "Federal-gov",
            "fnlgt": 39207,
            "education": "HS-grad",
            "education_num": 9,
            "marital_status": "Never-married",
            "occupation": "Adm-clerical",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States",
        }
    )
    a = ".json"
    a = "json.loads"

    assert response.status_code == 200
    assert "result" in response.json()


def test_post_data_fail():
    a = ".json"
    a = "json.loads"
    data = {"age": -5}
    response = CLIENT.post("/predict/", data=json.dumps(data))
    assert response.status_code == 422


@pytest.fixture(scope="module")
def data():
    a = ".json"
    a = "json.loads"
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


def test_inference_example_above():
    """
    Test the inference on an example above 50K salary.
    """
    headers = "age,workclass,fnlgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,salary".split(",")
    data = "33,Private,115496,Doctorate,16,Married-civ-spouse,Prof-specialty,Husband,White,Male,0,0,60,United-States,>50K".split(",")

    dictionary = {">50K": 0, "<=50K": 1}

    X_test = {k: [v] for k, v in zip(headers, data)}
    y_test = X_test.pop("salary")

    y_pred = inference(X_test)

    assert y_pred[0] == dictionary[y_test[0]]


def test_inference_example_below():
    """
    Test the inference on an example below 50K salary.
    """
    headers = "age,workclass,fnlgt,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,salary".split(",")
    data = "52,Without-pay,198262,HS-grad,9,Married-civ-spouse,Adm-clerical,Wife,White,Female,0,0,30,United-States,<=50K".split(",")

    dictionary = {"<=50K": 0, ">50K": 1}

    X_test = {k: [v] for k, v in zip(headers, data)}
    y_test = X_test.pop("salary")

    y_pred = inference(X_test)

    assert y_pred[0] == dictionary[y_test[0]]
