#!/usr/bin/env python3

# Other team member:
# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

# Me - Sara Pachemska :
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float,
                    help="Smoothing parameter of our NB classifier")
parser.add_argument("--naive_bayes_type", default="gaussian",
                    choices=["gaussian", "multinomial", "bernoulli"])
parser.add_argument("--classes", default=10, type=int,
                    help="Number of classes")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=72, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.

# Function returning a matrix of `n_classes` priors
# P(class) = P(t) for each class t.
# The priors should be estimated from the targets.


def calculate_priors(targets, n_classes):
    counter = np.bincount(targets, minlength=n_classes)
    return counter / len(targets)

# Gaussian NB training, by estimating mean and variance of the input features.
# For variance estimation use
#   1/N * \sum_x (x - mean)^2
# and additionally increase all estimated variances by `alpha`.


def train_gaussian_nb(train_data, train_targets, n_classes, alpha):
    means = np.zeros((n_classes, train_data.shape[1]))
    variances = np.zeros_like(means)

    for i in range(n_classes):
        class_data = train_data[train_targets == i]
        means[i, :] = class_data.mean(axis=0)
        variances[i, :] = class_data.var(axis=0) + alpha

    return means, variances

# Multinomial NB with smoothing factor `alpha`.
# The feature probabilities are
#   P(x_i|class) = count(x_i, class) / count(class)


def train_multinomial_nb(train_data, train_targets, n_classes, alpha):
    features = np.zeros((n_classes, train_data.shape[1]))
    counter = np.zeros(n_classes)

    for i in range(n_classes):
        class_data = train_data[train_targets == i]
        features[i, :] = class_data.sum(axis=0)
        counter[i] = class_data.sum()
    result = (features + alpha) / \
        (counter[:, np.newaxis] + alpha * train_data.shape[1])

    return result

# Bernoulli NB with smoothing factor `alpha`.
# Because Bernoulli NB works with binary data, binarize the features as
# [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
# during both estimation and prediction.


def train_bernoulli_nb(train_data, train_targets, n_classes, alpha):
    binarized_data = (train_data >= 8).astype(int)
    probs = np.zeros((n_classes, train_data.shape[1]))

    for i in range(n_classes):
        class_data = binarized_data[train_targets == i]
        probs[i, :] = (class_data.sum(
            axis=0) + alpha) / (len(class_data) + 2 * alpha)

    return probs


# Predictions
def predict_gaussian_nb(test_data, priors, means, variances):
    log_probs = []

    for i in range(len(test_data)):
        class_probs = []
        for c in range(len(priors)):
            class_prob = np.log(priors[c])
            class_prob += scipy.stats.norm.logpdf(
                test_data[i], means[c], np.sqrt(variances[c])).sum()
            class_probs.append(class_prob)
        log_probs.append(class_probs)

    return np.argmax(log_probs, axis=1), log_probs


def predict_multinomial_nb(test_data, priors, probs):
    log_probs = []

    for i in range(len(test_data)):
        class_probs = []
        for c in range(len(priors)):
            class_prob = np.log(priors[c])
            class_prob += (np.log(probs[c]) * test_data[i]).sum()
            class_probs.append(class_prob)
        log_probs.append(class_probs)

    return np.argmax(log_probs, axis=1), log_probs


def predict_bernoulli_nb(test_data, priors, probs):
    binarized_data = (test_data >= 8).astype(int)
    log_probs = []

    for i in range(len(binarized_data)):
        class_probs = []
        for c in range(len(priors)):
            class_prob = np.log(priors[c])
            class_prob += (np.log(probs[c]) * binarized_data[i] + np.log(
                1 - probs[c]) * (1 - binarized_data[i])).sum()
            class_probs.append(class_prob)
        log_probs.append(class_probs)

    return np.argmax(log_probs, axis=1), log_probs


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Load the digits dataset.
    data, target = sklearn.datasets.load_digits(
        n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    #
    #   During prediction, you can compute the probability density function
    #   of a Gaussian distribution using `scipy.stats.norm`, which offers
    #   `pdf` and `logpdf` methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
    #
    # In all cases, the class prior is the distribution of the train data classes.

    priors = calculate_priors(train_target, args.classes)

    if args.naive_bayes_type == "gaussian":
        means, variances = train_gaussian_nb(
            train_data, train_target, args.classes, args.alpha)
        test_predictions, test_log_probs = predict_gaussian_nb(
            test_data, priors, means, variances)
    elif args.naive_bayes_type == "multinomial":
        probs = train_multinomial_nb(
            train_data, train_target, args.classes, args.alpha)
        test_predictions, test_log_probs = predict_multinomial_nb(
            test_data, priors, probs)
    elif args.naive_bayes_type == "bernoulli":
        probs = train_bernoulli_nb(
            train_data, train_target, args.classes, args.alpha)
        test_predictions, test_log_probs = predict_bernoulli_nb(
            test_data, priors, probs)

    # TODO: Predict the test data classes, and compute
    # - the test set accuracy, and
    # - the joint log-probability of the test set, i.e.,
    #     \sum_{(x_i, t_i) \in test set} \log P(x_i, t_i).
    test_accuracy = np.mean(test_predictions == test_target)
    test_log_probability = np.sum(
        [test_log_probs[i][test_target[i]] for i in range(len(test_target))])

    return 100 * test_accuracy, test_log_probability


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy, test_log_probability = main(args)

    print("Test accuracy {:.2f}%, log probability {:.2f}".format(
        test_accuracy, test_log_probability))
