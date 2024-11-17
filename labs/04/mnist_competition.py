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
import urllib.request
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

parser = argparse.ArgumentParser()
parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument(
    "--model_path", default="mnist_competition.model", type=str, help="Model path")


class Dataset:
    def __init__(self, name="mnist.train.npz", data_size=None, url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(
                url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value[:data_size])
        self.data = self.data.reshape([-1, 28*28]).astype(np.float32)


def main(args: argparse.Namespace):
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        # Train/test split
        train_data, valid_data, train_target, valid_target = train_test_split(
            train.data, train.target, test_size=0.2, random_state=args.seed
        )

        # Preprocess data
        scaler = StandardScaler()
        train_data = scaler.fit_transform(train_data)
        valid_data = scaler.transform(valid_data)

        # Define an MLP classifier
        model = MLPClassifier(
            hidden_layer_sizes=(256, 256, 128),
            activation='relu',
            solver='adam',
            alpha=1e-5,
            batch_size=256,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=200,
            tol=1e-5,
            n_iter_no_change=20,
            random_state=args.seed,
            verbose=True
        )

        # Train the MLP
        model.fit(train_data, train_target)

        # Evaluate the MLP
        train_accuracy = model.score(valid_data, valid_target)
        print(f"Validation accuracy: {train_accuracy * 100:.2f}%")

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Load the model and predict
        test = Dataset(args.predict)
        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Preprocess data (use the same scaler as during training)
        scaler = StandardScaler()
        test.data = scaler.fit_transform(test.data)

        # Predict on the test data
        predictions = model.predict(test.data)
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)