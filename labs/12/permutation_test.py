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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--classes", default=10, type=int,
                    help="Number of classes")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--random_samples", default=1000,
                    type=int, help="Number of random samples")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=47, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
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

    model_predictions = [model.predict(test_data) for model in models]

    scores = []
    for _ in range(args.random_samples):
        assignments = generator.choice(2, size=len(test_data), replace=True)
        combined_predictions = np.choose(assignments, model_predictions)
        scores.append(100 * np.mean(combined_predictions == test_target))

    # Calculate performance of the second model
    second_model = 100 * np.mean(model_predictions[1] == test_target)

    # Compute p-value
    p_value = 100 * np.mean(np.array(scores) >= second_model)

    # Plot the histograms, confidence intervals and the p-value if requested.
    if args.plot:
        import matplotlib.pyplot as plt
        bin_size = 100 / len(test_data)
        plt.hist(scores, int(round((np.max(scores) - np.min(scores)) / bin_size)),
                 range=(np.min(scores) - bin_size/2,
                        np.max(scores) - bin_size/2),
                 weights=100 * np.ones_like(scores) / len(scores))
        for percentile in [1, 2.5, 5, 25, 50, 75, 95, 97.5, 99]:
            value = np.percentile(scores, percentile)
            color = {1: "#f00", 2.5: "#d60", 5: "#dd0", 25: "#0f0",
                     50: "#000"}[min(percentile, 100 - percentile)]
            plt.axvline(value, ls="--", color=color,
                        label="{:04.1f}%: {:.2f}".format(percentile, value))
        plt.axvline(second_model, ls="--", color="#f0f",
                    label="{:04.1f}%: {:.2f}".format(100 - p_value, second_model))
        plt.xlabel("Permuted model accuracy")
        plt.ylabel("Frequency [%]")
        plt.legend()
        plt.show() if args.plot is True else plt.savefig(
            args.plot, transparent=True, bbox_inches="tight")

    return p_value


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    p_value = main(args)
    print(
        "The estimated p-value of the random permutation test: {:.2f}%".format(p_value))
