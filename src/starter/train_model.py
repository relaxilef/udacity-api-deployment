# Script to train machine learning model.
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import train_model, save_model, compute_model_metrics

# Add the necessary imports for the starter code.
import pandas as pd
import os
import pickle

from definitions import ROOT_DIR


if __name__ == "__main__":
    # Add code to load in the data.
    data = pd.read_csv(os.path.join(ROOT_DIR, "data/census.csv"))
    data = data.rename(columns={col: col.strip() for col in data.columns})

    # Optional enhancement, use K-fold cross validation instead of a
    # train-test split.
    train, test = train_test_split(data, test_size=0.20)

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
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=cat_features, label="salary", training=True,
    )

    # Proces the test data with the process_data function.

    # Train and save a model.
    model = train_model(X_train, y_train)

    print(type(model))

    save_model(model=model, model_name="model.pkl")
    save_model(model=encoder, model_name="encoder.pkl")

    # X_test, y_test, encoder, lb = process_data(
    #     test, categorical_features=cat_features, label="salary", training=False,
    # )
    # predictions = model.predict(y_test)

    # precision, recall, fbeta = compute_model_metrics(y_test, predictions)
