#!/usr/bin/env python3

# Other team member:
# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

# Me - Sara Pachemska :
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

import argparse
import dataclasses

import numpy as np

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=100,
                    type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000,
                    type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.


class ArtificialData:
    @dataclasses.dataclass
    class Sentence:
        """ Information about a single dataset sentence."""
        gold_edits: int  # Number of required edits to be performed.
        predicted_edits: int  # Number of edits predicted by a model.
        predicted_correct: int  # Number of correct edits predicted by a model.
        human_rating: int  # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct)
                                      - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(
                gold, predicted, correct, human_rating))


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Create the artificial data.
    data = ArtificialData(args)

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings,
    # - TP, FP, FN counts of the predicted edits.
    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset.
        sentences = generator.choice(
            data.sentences, size=len(data.sentences), replace=True)

        # TODO: Append the average of human ratings of `sentences` to `human_ratings`.
        avg_human_rating = np.mean(
            [sentence.human_rating for sentence in sentences])
        human_ratings.append(avg_human_rating)

        # TODO: Compute TP, FP, FN counts of predicted edits in `sentences`
        # and append them to `predictions`.
        TP, FP, FN = 0, 0, 0
        for sentence in sentences:
            TP = TP + sentence.predicted_correct
            FP = FP + sentence.predicted_edits - sentence.predicted_correct
            FN = FN + sentence.gold_edits - sentence.predicted_correct

        predictions.append((TP, FP, FN))

    # Compute Pearson correlation between F_beta score and human ratings
    # for betas between 0 and 2.
    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)
        f_beta_scores = []
        for TP, FP, FN in predictions:
            if TP + FP == 0 or TP + FN == 0:
                f_beta = 0.0
            else:
                f_beta = (1 + beta**2) * TP / \
                    ((1 + beta**2) * TP + beta**2 * FN + FP)
            f_beta_scores.append(f_beta)

        mean_f_beta = np.mean(f_beta_scores)
        mean_human_ratings = np.mean(human_ratings)

        numerator = sum((f - mean_f_beta) * (h - mean_human_ratings)
                        for f, h in zip(f_beta_scores, human_ratings))
        denominator = (sum((f - mean_f_beta)**2 for f in f_beta_scores)
                       * sum((h - mean_human_ratings)**2 for h in human_ratings)) ** 0.5

        if denominator == 0:
            correlation = 0
        else:
            correlation = numerator / denominator

        correlations.append(correlation)

    if args.plot:
        import matplotlib.pyplot as plt

        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        plt.show() if args.plot is True else plt.savefig(
            args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.

    best_beta = betas[np.argmax(correlations)]
    best_correlation = correlations[np.argmax(correlations)]

    return best_beta, best_correlation


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))
