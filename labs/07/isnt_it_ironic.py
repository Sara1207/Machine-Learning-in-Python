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
import urllib.parse
import numpy as np
import numpy.typing as npt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import re


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="isnt_it_ironic.model", type=str, help="Model path")


class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(
                url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(
                url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        np.random.seed(args.seed)
        train = Dataset()

        # Vectorize the text data
        # tfidf_vectorizer = TfidfVectorizer(
        #     ngram_range=(1, 1),  min_df=4, max_df=1.0)
        tfidf_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),  min_df=1, max_df=0.8)
        X_train = tfidf_vectorizer.fit_transform(train.data)

        # Split the dataset for training and validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, train.target, test_size=0.2, random_state=args.seed)

        # Naive Bayes classifier
        model = MultinomialNB(alpha=0.8)

        model.fit(X_train, y_train)

        # Validation of the model
        y_pred = model.predict(X_val)
        print("Validation F1 Score:", f1_score(y_val, y_pred) * 100)

        # Serialize the model and vectorizer
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump((model, tfidf_vectorizer), model_file)

    else:
        # Load the model and vectorizer
        with lzma.open(args.model_path, "rb") as model_file:
            model, tfidf_vectorizer = pickle.load(model_file)

        test = Dataset(args.predict)
        X_test = tfidf_vectorizer.transform(test.data)

        # Predict the test set
        predictions = model.predict(X_test)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)