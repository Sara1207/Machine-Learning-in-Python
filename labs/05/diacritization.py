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
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")


class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            licence_name = name.replace(".txt", ".LICENSE")
            urllib.request.urlretrieve(url + licence_name, filename=licence_name)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

class Model:
    def __init__(self, X, y, window_size = 5) -> None:
        self.model =  MLPClassifier(hidden_layer_sizes = (300,), activation = 'logistic', 
                            alpha = 0.0001, max_iter = 200, learning_rate = 'invscaling',
                            solver = 'adam', 
                            verbose = 100, tol = 0.0001)
        self.y = y
        self.X = X
        self.one_hot = OneHotEncoder(sparse_output = False)
        self.LETTERS_NODIA = "acdeeinorstuuyz"
        self.LETTERS_DIA = "áčďéěíňóřšťúůýž"
        self.window_size = window_size
        self.distincrWords = np.unique([i for i in y.lower()])
        self.distincrWords = np.append(self.distincrWords, ['q', '5', '3'])
        self.word_dict2 = np.unique([i for i in X.lower()])
        self.intTochar = {i:c for i,c in zip(range(len(self.distincrWords)),self.distincrWords)}
        self.charToint = {c:i for i,c in zip(range(len(self.distincrWords)),self.distincrWords)}
        
        self.one_hot_fit()
    # Fitting the data
    def fit(self):
        data, target = self.preprocess()
        self.model.fit(data,target)

    # Fit the one hot encoder 
    def one_hot_fit(self):
        self.one_hot.fit(np.expand_dims(range(len(self.distincrWords)),1))

    # Predict
    def predict(self, test):

        predicted = ""

        for i in range(len(test)):
            if not test[i] in self.LETTERS_NODIA:
                predicted = predicted + test[i]
            else:
                word = " " * max(0 , self.window_size - i) + \
                test[max(0,i - self.window_size) : min(len(test),i + self.window_size + 1)].lower() +\
                  " " * max(0, self.window_size - (len(test) - i - 1))
                dataList = []

                for k in word:
                    dataList.append([self.charToint[k]])
                oneHot = self.one_hot.transform(dataList)
                
                test_i = np.array(oneHot).reshape(-1,(len(self.distincrWords) * (self.window_size * 2 + 1)))
                pred = self.model.predict(test_i)[0]
                if test[i].isupper():
                    predicted = predicted + self.intTochar[pred].upper()
                else:
                    predicted = predicted + self.intTochar[pred]
        return predicted
    
    # Preprocessing the data
    def preprocess(self):
        # Initializing 
        data = []
        target = []
        for i in range(len(self.X)):
            if self.X[i].lower() not in self.LETTERS_NODIA:
                continue
            word = " " * max(0 , self.window_size - i) + \
             self.X[max(0, i - self.window_size) : min(len(self.X),i + self.window_size + 1)].lower() \
            +  " " * max(0, self.window_size - (len(self.X) - i - 1))
            
            dataList = []

            for k in word:
                dataList.append([self.charToint[k]])
            data.append(dataList)
            target.append(self.charToint[self.y[i].lower()])
        newList = []
        for i in range(len(data)):
            newList.append(self.one_hot.transform(data[i]))
        data = np.array(newList).reshape(-1,(len(self.distincrWords) * (self.window_size * 2 + 1)))

        return data, target


def main(args: argparse.Namespace) -> Optional[str]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = Model(train.data, train.target, window_size = 5)
        model.fit()

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. 
        predictions = model.predict(test.data)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)