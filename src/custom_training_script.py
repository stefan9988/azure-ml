import mlflow
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import os

def main(args):
    with mlflow.start_run():
        df = get_data(args.training_data)
        X_train, X_test, y_train, y_test = split_data(df)
        model = train_model(X_train, y_train, args.C, args.penalty)
        eval_model(model, X_test, y_test)


# function that reads the data
def get_data(path):
    print("Reading data...")
    df = pd.read_csv(path)
    
    return df

# function that splits the data
def split_data(df):
    print("Splitting data...")
    X, y = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness',
    'SerumInsulin','BMI','DiabetesPedigree','Age']].values, df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, C=1.0, penalty='l2'):
    print("Training model...")
    mlflow.sklearn.autolog()
    model = LogisticRegression(
        solver="liblinear",
        C=C,
        penalty=penalty,
        ).fit(X_train, y_train)
    return model


def eval_model(model, X_test, y_test):
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)

    # AUC (manually log)
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test, y_scores[:, 1])
    print('AUC:', auc)
    mlflow.log_metric("AUC", auc)

def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--C", dest='C', type=float, default=1.0,
                        help="Inverse of regularization strength; smaller values specify stronger regularization.")
    parser.add_argument("--penalty", dest='penalty', type=str, default='l2',
                        help="Used to specify the norm used in the penalization. 'l1' for Lasso, 'l2' for Ridge.")
    
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)