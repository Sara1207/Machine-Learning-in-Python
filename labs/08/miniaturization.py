#!/usr/bin/env python3

# Team members:
# Sara Pachemska (me):
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.neural_network
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="miniaturization.model", type=str, help="Model path")


class Dataset:
    """MNIST Dataset.

    The train set contains 60000 images of handwritten digits. The data
    contain 28*28=784 values in the range 0-255, the targets are numbers 0-9.
    """

    def __init__(self,
                 name="mnist.train.npz",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(
                url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset, i.e., `data` and optionally `target`.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(float)


# The following class modifies `MLPClassifier` to support full categorical distributions
# on input, i.e., each label should be a distribution over the predicted classes.
# During prediction, the most likely class is returned, but similarly to `MLPClassifier`,
# the `predict_proba` method returns the full distribution.
# Note that because we overwrite a private method, it is guaranteed to work only with
# scikit-learn 1.3.0, but it will most likely work with any 1.3.*.
class MLPFullDistributionClassifier(sklearn.neural_network.MLPClassifier):
    class FullDistributionLabels:
        y_type_ = "multiclass"

        def fit(self, y):
            return self

        def transform(self, y):
            return y

        def inverse_transform(self, y):
            return np.argmax(y, axis=-1)

    def _validate_input(self, X, y, incremental, reset):
        X, y = self._validate_data(X, y, multi_output=True, dtype=(
            np.float64, np.float32), reset=reset)
        if (not hasattr(self, "classes_")) or (not self.warm_start and not incremental):
            self._label_binarizer = self.FullDistributionLabels()
            self.classes_ = y.shape[1]
        return X, y


class MLPClassifier(MLPClassifier):
    def fit(self, X, y, **params):
        super().fit(X, y, **params)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Normalize the data
        X_train_full, X_val, y_train_full, y_val = train_test_split(
            train.data / 255.0, train.target, test_size=0.1, random_state=args.seed)

        one_hot = OneHotEncoder(sparse=False, categories='auto')
        train_Y_encoded = one_hot.fit_transform(y_train_full.reshape(-1, 1))

        mlp = MLPClassifier(hidden_layer_sizes=(400, 350, 200, 150, 100, 100, 50, 50), max_iter=1500, alpha=4e-5,
                            solver='adam', verbose=10, tol=1e-4, random_state=args.seed,
                            learning_rate='adaptive', learning_rate_init=.0005, n_iter_no_change=25,
                            batch_size=30)

        mlp.fit(X_train_full, train_Y_encoded)

        # Compress the model
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)):
            mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)):
            mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)

        model = mlp

        val_probabilities = mlp.predict_proba(X_val)
        val_predictions = np.argmax(val_probabilities, axis=1)

        val_accuracy = accuracy_score(y_val, val_predictions)
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        test_probabilities = model.predict_proba(test.data)
        predictions = np.argmax(test_probabilities, axis=1)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
