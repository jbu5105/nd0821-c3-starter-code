import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics, inference, train_model

@pytest.fixture(scope='module')
def processed_data():
    df = pd.read_csv('./data/census.csv')

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

    # Proces the test data with the process_data function.
    X, y, _, _ = process_data(
        df,
        categorical_features=cat_features,
        label='salary',
        training=True
    )

    return df, X, y

@pytest.fixture(scope='module')
def model():
    return RandomForestClassifier(random_state=0)


def test_train_model(processed_data):
    """ Check if train_model function runs without failing """
    _, X, y,  = processed_data
    assert train_model(X, y)

def test_compute_metrics():
    """ Check compute_metrics output dtypes """
    y = np.zeros(100)
    preds = np.zeros(100)
    precision, recall, f1 = compute_model_metrics(y, preds)

    assert isinstance(precision, float)
    assert isinstance(recall, float)
    assert isinstance(f1, float)

def test_data(processed_data):
    """ Checks data has a specific number of columns """
    df, _, _ = processed_data
    print(df.columns)
    assert len(df.columns) == 15
