from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import fbeta_score, precision_score, recall_score

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
    model = RandomForestClassifier(random_state=0)
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
    """ Run model inferences and return the predictions.

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
    return model.predict(X)


def inference_slice(df, X, y, model, slice):
    with open('./slice_output.txt', 'w') as f:
        f.writelines(f"{slice=}\n\n")

        for feature in set(df[slice]):
            samples_bool = df[slice] == feature

            preds = model.predict(X[samples_bool])
            precision, recall, f1 = compute_model_metrics(y[samples_bool], preds)

            f.writelines(f"{feature=}\n")
            f.writelines(f"{precision=}\t")
            f.writelines(f"{recall=}\t")
            f.writelines(f"{f1=}\n\n")
