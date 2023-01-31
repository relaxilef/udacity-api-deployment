import json
import os
import pandas as pd
from pprint import pprint

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, load_model, compute_slice_metrics

from definitions import ROOT_DIR


def main():
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

    data_test = pd.read_csv(os.path.join(ROOT_DIR, "data/data_test.csv"))

    model = load_model(os.path.join(ROOT_DIR, "model/model.pkl"))
    encoder = load_model(os.path.join(ROOT_DIR, "model/encoder.pkl"))
    label_binarizer = load_model(os.path.join(ROOT_DIR, "model/label_binarizer.pkl"))

    X_test, y_test, _, _ = process_data(
        data_test,
        categorical_features=cat_features,
        label="salary",
        training=False,
        encoder=encoder,
        lb=label_binarizer,
    )

    y_pred = model.predict(X_test)

    model_metrics = compute_model_metrics(y_test, y_pred)
    pprint(model_metrics)
    model_slice_metrics = compute_slice_metrics(data_test, y_test, y_pred)
    pprint(model_slice_metrics)

    with open(os.path.join(ROOT_DIR, "data/slice_output.txt"), mode="w", encoding="utf-8") as file:
        json.dump(model_metrics, file, indent=2)


if __name__ == "__main__":
    main()
