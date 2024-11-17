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
import sklearn.feature_extraction
import sklearn.metrics
import sklearn.model_selection
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="imdb_sentiment.model", type=str, help="Model path")


class Dataset:
    """IMDB dataset.

    This is a modified IMDB dataset for sentiment classification. The text is
    already tokenized and partially normalized.
    """

    def __init__(self,
                 name="imdb_train.txt",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(
                url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []
        with open(name) as f_imdb:
            for line in f_imdb:
                label, text = line.split("\t", 1)
                self.data.append(text)
                self.target.append(int(label))


def load_word_embeddings(
        name="imdb_embeddings.npz",
        url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
    """Load word embeddings.

    These are selected word embeddings from FastText. For faster download, it
    only contains words that are in the IMDB dataset.
    """
    if not os.path.exists(name):
        print("Downloading embeddings {}...".format(name), file=sys.stderr)
        urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
        os.rename("{}.tmp".format(name), name)

    with open(name, "rb") as f_emb:
        data = np.load(f_emb)
        words = data["words"]
        vectors = data["vectors"]
    embeddings = {word: vector for word, vector in zip(words, vectors)}
    return embeddings


def tfidf_preprocessing(text_list, word_embeddings, min_df=3, max_df=1.0
                        , ngram_range=(1, 3)):
    # Convert ENGLISH_STOP_WORDS to a list
    stop_words_list = list(ENGLISH_STOP_WORDS)
    # Initialize a TfidfVectorizer with the FastText word embeddings
    vectorizer = TfidfVectorizer(stop_words=stop_words_list,
                                 tokenizer=lambda text: text.split(),
                                 lowercase=False,
                                 min_df=min_df,
                                 max_df=max_df,
                                 ngram_range=ngram_range)  # <-- Include ngram_range here
    vectorized_texts = vectorizer.fit_transform(text_list)

    # Convert the sparse matrix to a dense matrix
    dense_texts = vectorized_texts.toarray()

    # Get the vocabulary and its corresponding index
    vocabulary = vectorizer.vocabulary_

    # For each document, compute the weighted average of the word embeddings
    processed = []
    for text in dense_texts:
        embeddings_list = []
        for word, index in vocabulary.items():
            if text[index] != 0:  # Only consider words present in the document
                embeddings_list.append(word_embeddings.get(
                    word, np.zeros(300)) * text[index])

        average_embedding = np.mean(
            embeddings_list, axis=0) if embeddings_list else np.zeros(300)
        processed.append(average_embedding)

    return np.array(processed)

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    word_embeddings = load_word_embeddings()

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        print("Preprocessing dataset.", file=sys.stderr)
        # TODO: Preprocess the text such that you have a single vector per movie
        # review. You can experiment with different ways of pooling the word
        # embeddings: averaging, max pooling, etc. You can also try to exclude
        # words that do not contribute much to the meaning of the sentence (stop
        # words). See `sklearn.feature_extraction.stop_words.ENGLISH_STOP_WORDS`.
        # Preprocess the training data
        train_as_vectors = tfidf_preprocessing(train.data, word_embeddings)

        # Split the data
        train_x, validation_x, train_y, validation_y = sklearn.model_selection.train_test_split(
            train_as_vectors, train.target, test_size=0.25, random_state=args.seed)

        print("Training.", file=sys.stderr)

        # Pipeline with StandardScaler and LogisticRegression
        pipeline = make_pipeline(
            StandardScaler(),LogisticRegression(random_state=args.seed, max_iter=1000))

        param_grid = {
            'logisticregression__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        }

        grid_search = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=-1)
        grid_search.fit(train_x, train_y)

        # TODO: Train a model of your choice on the given data.
        # Initialize and train the model
        model = grid_search.best_estimator_
        model.fit(train_x, train_y)

        print("Evaluation.", file=sys.stderr)
        validation_predictions = model.predict(validation_x)
        validation_accuracy = sklearn.metrics.accuracy_score(
            validation_y, validation_predictions)
        print("Validation accuracy {:.2f}%".format(100 * validation_accuracy))

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Preprocess the test data
        test_as_vectors = tfidf_preprocessing(test.data, word_embeddings)

        # Generate predictions
        predictions = model.predict(test_as_vectors)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)