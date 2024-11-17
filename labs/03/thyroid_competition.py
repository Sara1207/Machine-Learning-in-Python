#!/usr/bin/env python3

# Other team member:
# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

# Me - Sara Pachemska :
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """

    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Preprocess training data
        scaler = StandardScaler()
        X_train_rv = scaler.fit_transform(train.data[:, -6:])
        X_train = np.hstack([train.data[:, :-6], X_train_rv])
        y_train = train.target

        # # Logistic Regression model
        # model = LogisticRegression(max_iter=10000, random_state=args.seed)
        # model.fit(X_train, y_train)

        # Feature engineering
        poly = PolynomialFeatures(
            degree=2, interaction_only=True, include_bias=False)
        X_train_poly = poly.fit_transform(X_train)
        X_train = np.hstack([X_train, X_train_poly])

        for C_val in [0.01, 0.1, 1, 10, 100]:
            model = LogisticRegression(
                C=C_val, max_iter=10000, random_state=args.seed)
            model.fit(X_train, y_train)
            accuracy = model.score(X_train, y_train)
            print(f"Training accuracy with C={C_val}: {accuracy * 100:.2f}%")

        # Calculate training accuracy and print
        accuracy = model.score(X_train, y_train)
        print(f"Training accuracy: {accuracy * 100:.2f}%")

        # Serialize both the model and the scaler for preprocessing
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((model, scaler, poly), model_file)
    else:
        # We are predicting using a trained model.
        test = Dataset(args.predict)

        # Load the model and the scaler
        with lzma.open(args.model_path, "rb") as model_file:
            model, scaler, poly = pickle.load(model_file)

        # Preprocess test data
        X_test_rv = scaler.transform(test.data[:, -6:])
        X_test = np.hstack([test.data[:, :-6], X_test_rv])

        X_test_poly = poly.transform(X_test)
        X_test = np.hstack([X_test, X_test_poly])

        # Generate predictions
        predictions = model.predict(X_test)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
