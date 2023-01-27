import os
import pytest
import pandas as pd

import sklearn
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, save_model, compute_model_metrics, load_model

from definitions import ROOT_DIR


@pytest.fixture(scope="module")
def data():
    data = pd.read_csv(os.path.join(ROOT_DIR, "data/census.csv"))
    data = data.rename(columns={col: col.strip() for col in data.columns})
    return data


def test_data(data):
    """Tests to retrieve the data.
    """
    assert isinstance(data, pd.DataFrame)


def test_train_model(data):
    """Tests to train the model on the data.
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
    """Tests to load the model.
    """
    model = load_model("model.pkl")
    assert isinstance(model, sklearn.linear_model._logistic.LogisticRegression)
