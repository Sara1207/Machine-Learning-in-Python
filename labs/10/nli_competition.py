#!/usr/bin/env python3

# Other team member:
# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

# Me - Sara Pachemska :
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

import argparse
import lzma
import pickle
import os
from typing import Optional
import sklearn.feature_extraction
import sklearn.neural_network
import sklearn.pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

import numpy as np
import numpy.typing as npt

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="nli_competition.model", type=str, help="Model path")


class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA",
               "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError(
                "The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Configuration parameters
        tfidf_word_params = {
            "lowercase": True,
            "ngram_range": (1, 2),
            "analyzer": "word",
            "use_idf": False,
            "sublinear_tf": True,
            "max_features": 5000
        }

        tfidf_char_params = {
            "lowercase": False,
            "ngram_range": (1, 3),
            "analyzer": "char",
            "use_idf": False,
            "sublinear_tf": True,
            "max_features": 5000
        }

        mlp_params = {
            "hidden_layer_sizes": (150,),
            "verbose": 1,
            "max_iter": 20
        }

        # Pipeline
        model = Pipeline([
            ("text_feature_extraction", FeatureUnion([
                ("word_level_features", TfidfVectorizer(**tfidf_word_params)),
                ("character_level_features", TfidfVectorizer(**tfidf_char_params)),
            ])),
            ("neural_network_classifier", MLPClassifier(**mlp_params)),
        ])

        model.fit(train.data, train.target)

        # Compress the model
        if 'neural_network_classifier' in model.named_steps:
            mlp = model.named_steps['neural_network_classifier']
            mlp._optimizer = None  # Remove the optimizer state
            mlp.coefs_ = [c.astype(np.float16)
                          for c in mlp.coefs_]  # Convert to float16
            mlp.intercepts_ = [i.astype(np.float16)
                               for i in mlp.intercepts_]  # Convert to float16

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)