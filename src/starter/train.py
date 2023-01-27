# Script to train machine learning model.
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, save_model, compute_model_metrics, compute_slice_metrics

# Add the necessary imports for the starter code.
import pandas as pd
import os

from definitions import ROOT_DIR


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(ROOT_DIR, "data/census.csv"))

    df_obj = data.select_dtypes(["object"])
    data[df_obj.columns] = df_obj.apply(lambda x: x.str.strip())
    data = data.rename(columns={col: col.strip() for col in data.columns})

    data_train, data_test = train_test_split(data, test_size=0.20)

    data_train.to_csv(os.path.join(ROOT_DIR, "data/data_train.csv"))
    data_test.to_csv(os.path.join(ROOT_DIR, "data/data_test.csv"))

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
    X_train, y_train, encoder, label_binarizer = process_data(
        data_train,
        categorical_features=cat_features,
        label="salary",
        training=True,
    )

    model = train_model(X_train, y_train)
    save_model(model=model, model_name="model.pkl")
    save_model(model=encoder, model_name="encoder.pkl")
    save_model(model=label_binarizer, model_name="label_binarizer.pkl")
