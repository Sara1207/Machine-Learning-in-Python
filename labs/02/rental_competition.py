#!/usr/bin/env python3

# Other team member:
# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

# Me - Sara Pachemska:
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
from sklearn.model_selection import train_test_split , cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from math import sqrt
import urllib.request
from sklearn.preprocessing import StandardScaler

import numpy as np
import numpy.typing as npt
from scipy import stats

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")


class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rented bikes in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~courses/npfl129/2324/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename="{}.tmp".format(name))
            os.rename("{}.tmp".format(name), name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)



def train_model(train_data, train_target):
    
    model = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2),
        Ridge(alpha=100)  
    )
    model.fit(train_data, train_target)
    return model

def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # Splitting data into features and target
        X = np.column_stack([getattr(train, f) for f in train.__dict__.keys() if f != 'target'])
        y = train.target

        #remove outliers
        
        z_scores = stats.zscore(X)

        z_score_threshold = 10  # You can adjust this threshold

        indices_outliers = np.where(np.abs(z_scores) > z_score_threshold)

        # Remove the outliers from X
        X_without_outliers = np.delete(X, indices_outliers, axis=0)

        # y remains the same
        y_without_outliers = y

        # Scale the features
        scaler = StandardScaler()
        X_without_outliers = scaler.fit_transform(X_without_outliers)

        # Split the data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X_without_outliers, y_without_outliers, test_size=0.2, random_state=args.seed)

        # Cross-Validation
        folds_number = 1000  
        folds = KFold(n_splits=folds_number, random_state=args.seed, shuffle=True)

        rmse_allFolds = []  #empty array

        for i, j in folds.split(X_without_outliers):
            X_train, X_val = X_without_outliers[i], X_without_outliers[j]
            y_train, y_val = y_without_outliers[i], y_without_outliers[j]

            model = train_model(X_train, y_train)

            #y_pred = model.predict(X_val)

            #rmse = sqrt(mean_squared_error(y_val, y_pred))
            #rmse_allFolds.append(rmse)

        # average RMSE 
        #finalrmse = np.mean(rmse_allFolds)
        # print(finalrmse)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # Prepare the test data
        X_test = np.column_stack([getattr(test, f) for f in test.__dict__.keys()])

        # Make predictions using the trained model
        predictions = model.predict(X_test)

        print("Predictions:", predictions)
        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
