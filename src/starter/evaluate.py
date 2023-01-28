import json
import os
import pandas as pd

from starter.ml.data import process_data
from starter.ml.model import load_model, compute_slice_metrics

from definitions import ROOT_DIR


def main():
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
    result = compute_slice_metrics(data_test, y_test, y_pred)
    
    with open(os.path.join(ROOT_DIR, "data/all_slice_metrics.json"), mode="w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)



if __name__ == "__main__":
    main()
