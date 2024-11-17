#!/usr/bin/env python3

# Other team member:
# Natalia Tashkova:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

# Me - Sara Pachemska :
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

import argparse

import numpy as np
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
from sklearn.preprocessing import MultiLabelBinarizer

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=5, type=int,
                    help="Number of classes to use")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--epochs", default=10, type=int,
                    help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Learning rate")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial classification dataset.
    data, target_list = sklearn.datasets.make_multilabel_classification(
        n_samples=args.data_size, n_classes=args.classes, allow_unlabeled=False,
        return_indicator=False, random_state=args.seed)

    # TODO: The `target` is a list of classes for every input example. Convert
    # it to a dense representation (n-hot encoding) -- for each input example,
    # the target should be vector of `args.classes` binary indicators.
    mlb = MultiLabelBinarizer(classes=range(args.classes))
    target = mlb.fit_transform(target_list)

    # Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    data = np.pad(data, [(0, 0), (0, 1)], constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights.
    weights = generator.uniform(
        size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for i in range(0, train_data.shape[0], args.batch_size):
            indices = permutation[i:i + args.batch_size]
            b_data = train_data[indices]
            b_target = train_target[indices]

            # Compute predictions
            b_output = 1 / (1 + np.exp(-np.dot(b_data, weights)))
            # Compute the gradient
            gradient = np.dot(b_data.T, b_output -
                              b_target) / args.batch_size
            # Update weights
            weights -= args.learning_rate * gradient

        # TODO: After the SGD epoch, compute the micro-averaged and the
        # macro-averaged F1-score for both the train test and the test set.
        # Compute these scores manually, without using `sklearn.metrics`.
        # Calculate F1 scores manually
        # Prediction for the entire set
        train_output = (1 / (1 + np.exp(-np.dot(train_data, weights)))) >= 0.5
        test_output = (1 / (1 + np.exp(-np.dot(test_data, weights)))) >= 0.5

        # Calculate true positives (tp), false positives (fp), true negatives (tn), and false negatives (fn)
        tp_test = np.sum((test_output == 1) & (test_target == 1), axis=0)
        fn_test = np.sum((test_output == 0) & (test_target == 1), axis=0)
        fp_test = np.sum((test_output == 1) & (test_target == 0), axis=0)

        tp_train = np.sum((train_output == 1) & (train_target == 1), axis=0)
        fp_train = np.sum((train_output == 1) & (train_target == 0), axis=0)
        fn_train = np.sum((train_output == 0) & (train_target == 1), axis=0)

        # Calculate precision, recall, and F1 score avoiding division by zero
        precision_test = np.where(
            (tp_test + fp_test) == 0, 0, tp_test / (tp_test + fp_test))
        recall_test = np.where((tp_test + fn_test) == 0,
                               0, tp_test / (tp_test + fn_test))
        f1_test = np.where((precision_test + recall_test) == 0, 0, 2 *
                           (precision_test * recall_test) / (precision_test + recall_test))

        precision_train = np.where(
            (tp_train + fp_train) == 0, 0, tp_train / (tp_train + fp_train))
        recall_train = np.where((tp_train + fn_train) ==
                                0, 0, tp_train / (tp_train + fn_train))
        f1_train = np.where((precision_train + recall_train) == 0, 0, 2 *
                            (precision_train * recall_train) / (precision_train + recall_train))

        # Calculate micro F1 scores
        train_f1_micro = 2 * \
            np.sum(tp_train) / (2 * np.sum(tp_train) +
                                np.sum(fp_train) + np.sum(fn_train))
        test_f1_micro = 2 * \
            np.sum(tp_test) / (2 * np.sum(tp_test) +
                               np.sum(fp_test) + np.sum(fn_test))

        # Calculate macro F1 scores
        train_f1_macro = np.mean(f1_train)
        test_f1_macro = np.mean(f1_test)

        print("After epoch {}: train F1 micro {:.2f}% macro {:.2f}%, test F1 micro {:.2f}% macro {:.1f}%".format(
            epoch + 1, 100 * train_f1_micro, 100 * train_f1_macro, 100 * test_f1_micro, 100 * test_f1_macro))

    return weights, [(100 * train_f1_micro, 100 * train_f1_macro), (100 * test_f1_micro, 100 * test_f1_macro)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:",
          *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
