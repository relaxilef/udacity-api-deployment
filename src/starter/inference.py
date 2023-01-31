import os
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import load_model

from definitions import ROOT_DIR


def inference(data: dict):
    cat_features = [
        "workclass",
        "education",
        "marital_status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native_country",
    ]

    df = pd.DataFrame(data)

    model = load_model(os.path.join(ROOT_DIR, "model/model.pkl"))
    encoder = load_model(os.path.join(ROOT_DIR, "model/encoder.pkl"))
    label_binarizer = load_model(os.path.join(ROOT_DIR, "model/label_binarizer.pkl"))

    X, _, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )

    y_pred = model.predict(X)
    return y_pred
