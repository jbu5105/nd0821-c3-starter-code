""" Script to train machine learning model. """

# Add the necessary imports for the starter code.
import argparse
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from starter.ml.data import process_data
from starter.ml.model import inference, inference_slice, train_model, compute_model_metrics

# Logger
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def main(data_path, slice):
    # Add code to load in the data.
    data = pd.read_csv(data_path)

    # Optional enhancement, use K-fold cross validation instead of a train-test split.
    train, test = train_test_split(data, test_size=0.20)

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

    # Proces the test data with the process_data function.
    X, y, encoder, lb = process_data(data, categorical_features=cat_features, label='salary', training=True)
    data_processed = np.zeros((X.shape[0], X.shape[-1] + 1))
    data_processed[:,:X.shape[-1]] = X
    data_processed[:,-1] = y

    train, test = train_test_split(data_processed, test_size=0.20)
    x_train, y_train = train[:, :X.shape[-1]], train[:, -1]
    x_test, y_test = test[:, :X.shape[-1]], test[:, -1]

    # Train model.
    model = train_model(x_train, y_train)

    # Inference
    preds = inference(model, x_test)
    precision, recall, fbeta = compute_model_metrics(y_test, preds)
    logger.info(f"Precision: {precision}")
    logger.info(f"Recall: {recall}")
    logger.info(f"F1-score: {fbeta}")

    # Data Slice
    if slice:
        inference_slice(data, X, y, model, slice)

    # Save model and encoder
    logger.info("Exporting model and encoder")
    with open('./model_enc_lb.pkl', 'wb') as f:
        pickle.dump([model, encoder, lb], f)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-d', '--data_path', default="./starter/data/census.csv",type=str)
    args.add_argument('-s', '--slice', type=str, default='education')
    
    args = args.parse_args()
    main(args.data_path, args.slice)
