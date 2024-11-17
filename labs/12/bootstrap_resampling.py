#!/usr/bin/env python3
# Other team member:
# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

# Me - Sara Pachemska :
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

import argparse
import warnings

import numpy as np

import sklearn.datasets
import sklearn.exceptions
import sklearn.linear_model
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
from joblib import Parallel, delayed

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=1000,
                    type=int, help="Bootstrap resamplings")
parser.add_argument("--classes", default=10, type=int,
                    help="Number of classes")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def compute_scores(model, test_data, test_target, indices):
    return model.score(test_data[indices], test_target[indices]) * 100


def main(args: argparse.Namespace) -> tuple[list[tuple[float, float]], float]:
    # Do suppress warnings about the solver not converging because we
    # deliberately use a low `max_iter` for the models to train quickly.
    warnings.filterwarnings(
        "ignore", category=sklearn.exceptions.ConvergenceWarning)

    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Load the digits dataset.
    data = sklearn.datasets.load_digits(n_class=args.classes)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data.data, data.target, test_size=args.test_size, random_state=args.seed)

    models = []
    for d in [1, 2]:
        models.append(sklearn.pipeline.Pipeline([
            ("features", sklearn.preprocessing.PolynomialFeatures(degree=d)),
            ("estimator", sklearn.linear_model.LogisticRegression(
                solver="saga", random_state=args.seed)),
        ]))
        models[-1].fit(train_data, train_target)

    # Initialize an empty list for each model to store scores
    scores = [[] for _ in models]

    # Compute predictions for each model only once
    predictions = [model.predict(test_data) for model in models]

    # Bootstrap resampling to compute scores
    for _ in range(args.bootstrap_samples):
        indices = generator.choice(
            len(test_data), size=len(test_data), replace=True)
        for model_predictions, model_scores in zip(predictions, scores):
            accuracy = 100 * \
                np.mean(test_target[indices] == model_predictions[indices])
            model_scores.append(accuracy)

    # Convert the list of scores to a NumPy array for vectorized operations
    scores = np.array(scores)

    # Compute confidence intervals for each model
    confidence_intervals = [(np.percentile(model_scores, 2.5), np.percentile(model_scores, 97.5))
                            for model_scores in scores]

    # Calculate the percentage of cases where the second model's score is not greater than the first model's score
    result = 100 * np.mean(scores[1] - scores[0] <= 0)

    # Plot the histograms, confidence intervals and the p-value, if requested.
    if args.plot:
        import matplotlib.pyplot as plt

        def histogram(ax, data, color=None):
            ax.hist(data, int(round((np.max(data) - np.min(data)) * len(test_data) / 100)) + 1,
                    weights=100 * np.ones_like(data) / len(data), color=color)

        plt.figure(figsize=(12, 5))
        ax = plt.subplot(121)
        for score, confidence_interval, color in zip(scores, confidence_intervals, ["#d00", "#0d0"]):
            histogram(ax, score, color + "8")
            ax.axvline(np.mean(score), ls="-", color=color,
                       label="mean: {:.1f}%".format(np.mean(score)))
            ax.axvline(confidence_interval[0],
                       ls="--", color=color, label="95% CI")
            ax.axvline(confidence_interval[1], ls="--", color=color)
        ax.set_xlabel("Model accuracy")
        ax.set_ylabel("Frequency [%]")
        ax.legend()

        ax = plt.subplot(122)
        histogram(ax, scores[1] - scores[0])
        for percentile in [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99]:
            value = np.percentile(scores[1] - scores[0], percentile)
            color = {1: "#f00", 2.5: "#d60", 5: "#dd0", 25: "#0f0",
                     50: "#000"}[min(percentile, 100 - percentile)]
            ax.axvline(value, ls="--", color=color,
                       label="{:04.1f}%: {:.1f}".format(percentile, value))
        ax.axvline(0, ls="--", color="#f0f",
                   label="{:04.1f}%: 0.0".format(result))
        ax.set_xlabel("Model accuracy difference")
        ax.set_ylabel("Frequency [%]")
        ax.legend()
        plt.show() if args.plot is True else plt.savefig(
            args.plot, transparent=True, bbox_inches="tight")

    return confidence_intervals, result


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    confidence_intervals, result = main(args)
    print("Confidence intervals of the two models:")
    for confidence_interval in confidence_intervals:
        print("- [{:.2f}% .. {:.2f}%]".format(*confidence_interval))
    print(
        "The estimated probability that the null hypothesis holds: {:.2f}%".format(result))
