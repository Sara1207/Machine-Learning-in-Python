#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import sys
import urllib.request

import numpy as np
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
from sklearn.linear_model import LogisticRegression
import re
from sklearn.metrics import f1_score
import numpy as np
import sklearn.metrics
import sklearn.model_selection
from numpy import linalg as LA

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=45, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=500, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.


class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

def preprocess_data(args, train_data):
    new_data = []
    tf = []
    data_dict = {}
    features = []
    idf = []
    cleaned_match = []
    set_terms = {}
    # Train data
    for i in range(len(train_data)):
        match_i =  re.findall(r"\w+", train_data[i])
        dict_i = {}
        for term in match_i:
            if term in dict_i:
                dict_i[term] += 1
            else:
                dict_i[term] = 1
            if term in data_dict:
                if data_dict[term] == 1:
                    features.append(term)
                data_dict[term] += 1
            else:
                data_dict[term] = 1
            if term not in set_terms.keys():
                set_terms[term] = set()
            set_terms[term].add(i)
        cleaned_match.append([dict_i, len(match_i)])
    for i in range(len(train_data)):
        dict_i = cleaned_match[i][0]
        data_i = []
        if args.tf:
            n = cleaned_match[i][1]
            for feature in features:
                if feature in dict_i:
                    data_i.append(dict_i[feature]/n)
                else:
                    data_i.append(0.0)
        else:
            for feature in features:
                if feature in dict_i:
                    data_i.append(1.0)
                else:
                    data_i.append(0.0)
        new_data.append(data_i)
    new_data = np.array(new_data)
    # Calculate IDF values
    if args.idf:
        n_docs = len(new_data)
        for j in range(len(features)):
            occ = len(set_terms[features[j]])
            idf_i = np.log(n_docs / (occ + 1))
            idf.append(idf_i)
        idf = np.array(idf)

        for j in range(len(new_data)):
            new_data[j] *= idf

    for i in range(len(new_data)):
        norm = np.linalg.norm(new_data[i])
        new_data[i] = new_data[i] / norm

    return new_data, features, idf

def preprocess_test(args, test_data, features, idf):
    new_data = []
    tf = []
    data_dict = {}
    cleaned_match = []
    for i in range(len(test_data)):
        match_i =  re.findall(r"\w+",test_data[i])
        dict_i = {}
        for term in match_i:
            if term in dict_i:
                dict_i[term] += 1
            else:
                dict_i[term] = 1
        cleaned_match.append([dict_i, len(match_i)])
    for i in range(len(test_data)):
        dict_i = cleaned_match[i][0]
        data_i = []
        if args.tf:
            n = cleaned_match[i][1]
            for feature in features:
                if feature in dict_i:
                    data_i.append(dict_i[feature] / n)
                else:
                    data_i.append(0.0)
        else:
            for feature in features:
                if feature in dict_i:
                    data_i.append(1.0)
                else:
                    data_i.append(0.0)

        data_i = np.array(data_i)
        if args.idf:
            data_i *= idf  # Use the IDF values computed on the training data

        new_data.append(data_i)

    new_data = np.array(new_data)

    # Normalize the vectors
    norms = LA.norm(new_data, axis=1, keepdims=True)
    new_data = new_data / norms
    
    return new_data



def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    new_data, features, idf = preprocess_data(args, train_data)
    new_test = preprocess_test(args, test_data, features, idf)

    # TODO: Create a feature for every term that is present at least twice
    # in the training data. A term is every maximal sequence of at least 1 word character,
    # where a word character corresponds to a regular expression `\w`.

    # TODO: For each document, compute its features as
    # - term frequency (TF), if `args.tf` is set (term frequency is
    #   proportional to counts but normalized to sum to 1);
    # - otherwise, use binary indicators (1 if a given term is present, else 0)
    #
    # Then, if `args.idf` is set, multiply the document features by the
    # inverse document frequencies (IDF), where
    # - use the variant which contains `+1` in the denominator;
    # - the IDFs are computed on the train set and then reused without
    #   modification on the test set.

    # TODO: Train a `sklearn.linear_model.LogisticRegression(solver="liblinear", C=10_000)`
    # model on the train set, and classify the test set.
    model = LogisticRegression(solver="liblinear", C=10_000)
    model.fit(new_data, train_target)
    # TODO: Evaluate the test set performance using a macro-averaged F1 score.
    f1_score2 = f1_score(test_target, model.predict(new_test), average = 'macro')

    return 100 * f1_score2


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}: {:.1f}%".format(args.tf, args.idf, f1_score))