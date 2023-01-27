from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression

import os
import pickle

from definitions import ROOT_DIR


def load_model(model_name: str):
    """Load the model.

    Args:
        model_name (str): File name of the model.

    Returns:
        object: Model.
    """
    with open(os.path.join(ROOT_DIR, "model", model_name), "rb") as file:
        return pickle.load(file)


def save_model(model: object, model_name: str):
    """Save the model.

    Args:
        model (object): Model object.
        model_name (str): File name of the model.
    """
    with open(os.path.join(ROOT_DIR, "model", model_name), "wb") as f:
        pickle.dump(model, f)


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_train)
    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    pass