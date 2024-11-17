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

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--data_size", default=100, type=int, help="Data size")
parser.add_argument("--epochs", default=50, type=int,
                    help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01,
                    type=float, help="Learning rate")
parser.add_argument("--plot", default=False, const=True,
                    nargs="?", type=str, help="Plot the predictions")
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
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)

    # TODO: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.

    # np.append(data, np.ones((data.shape[0], 1)), axis=1)
    data = np.append(data, np.ones((data.shape[0], 1)), axis=1)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial logistic regression weights.
    weights = generator.uniform(size=train_data.shape[1], low=-0.1, high=0.1)

    # print("Initial weights:", weights)
    for epoch in range(args.epochs):

        permutation = generator.permutation(train_data.shape[0])

        train_correct = 0

        # Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for i in range(0, train_data.shape[0], args.batch_size):
            batch = train_data[permutation[i:i+args.batch_size]]
            target_batch = train_target[permutation[i:i+args.batch_size]]

            # Compute the predictions for the batch
            predictions = 1 / (1 + np.exp(-batch @ weights))

            # Compute the gradient of the loss w.r.t. the weights
            gradient = (predictions - target_batch) @ batch

            # Update the weights
            weights -= args.learning_rate * gradient / args.batch_size

        # print("Weights:", weights)
        # After the SGD epoch, measure the average loss and accuracy for both the
        # train set and the test set. The loss is the average MLE loss (i.e., the
        # negative log-likelihood, or cross-entropy loss, or KL loss) per example.
        train_predictions = 1 / (1 + np.exp(-train_data @ weights))
        train_loss = -np.mean(train_target * np.log(train_predictions) +
                              (1 - train_target) * np.log(1 - train_predictions))
        train_correct += np.sum((train_predictions > 0.5) == train_target)
        train_accuracy = train_correct / train_data.shape[0]

        test_predictions = 1 / (1 + np.exp(-test_data @ weights))
        test_loss = -np.mean(test_target * np.log(test_predictions) +
                             (1 - test_target) * np.log(1 - test_predictions))
        test_correct = np.sum((test_predictions > 0.5) == test_target)
        test_accuracy = test_correct / test_data.shape[0]

        # train_accuracy, train_loss, test_accuracy, test_loss = ...

        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

        if args.plot:
            import matplotlib.pyplot as plt
            if args.plot is not True:
                plt.gcf().get_axes() or plt.figure(figsize=(6.4*3, 4.8*(args.epochs+2)//3))
                plt.subplot(3, (args.epochs+2)//3, 1 + epoch)
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[1 / (1 + np.exp(-([x, y, 1] @ weights)))
                            for x in xs] for y in ys]
            plt.contourf(xs, ys, predictions, levels=20,
                         cmap="RdBu", alpha=0.7)
            plt.contour(xs, ys, predictions, levels=[
                        0.25, 0.5, 0.75], colors="k")
            plt.scatter(train_data[:, 0], train_data[:, 1],
                        c=train_target, label="train", marker="P", cmap="RdBu")
            plt.scatter(test_data[:, 0], test_data[:, 1],
                        c=test_target, label="test", cmap="RdBu")
            plt.legend(loc="upper right")
            plt.show() if args.plot is True else plt.savefig(
                args.plot, transparent=True, bbox_inches="tight")

    return weights, [(train_loss, 100 * train_accuracy), (test_loss, 100 * test_accuracy)]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights", *("{:.2f}".format(weight) for weight in weights))