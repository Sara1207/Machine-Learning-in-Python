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
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str,
                    help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="wine",
                    type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int,
                    help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int,
                    help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int,
                    help="Minimum examples required to split")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


class Node:
    """Class to represent a node in a decision tree."""

    def __init__(self, samples, targets, feature_ind=None, threshold=None, left_child=None, right_child=None, depth=None) -> None:
        self.samples = samples
        self.targets = targets
        self.feature_ind = feature_ind
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.depth = depth


class DecisionTreeClassifier:
    """Implementation of a decision tree classifier."""

    def __init__(self, criterion, max_depth=None, min_to_split=None, max_leaves=None) -> None:
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_to_split = min_to_split
        self.max_leaves = max_leaves
        self.root = None
        self.n_leaves = 0

    def split_threshold(self, X, targets, feature_ind, threshold):
        """Split the data by the threshold."""
        maskL = X[:, feature_ind] < threshold
        maskR = np.logical_not(maskL)
        return X[maskL, :], targets[maskL], X[maskR, :], targets[maskR]

    def calculate_gini(self, y_split):
        """Calculate the Gini of a split."""
        total = len(y_split)
        _, class_counts = np.unique(y_split, return_counts=True)
        return total * (1.0 - np.sum((class_counts / total) ** 2))

    def calculate_entropy(self, y_split):
        """Calculate the entropy of a split."""
        total = len(y_split)
        _, class_counts = np.unique(y_split, return_counts=True)
        probabilities = class_counts / total
        return total * -np.sum(probabilities * np.log2(probabilities), where=(probabilities != 0))

    def calculate_gain(self, targets):
        """Calculate the total gain."""
        return self.calculate_gini(targets) if self.criterion == 'gini' else self.calculate_entropy(targets)

    def find_best_feature(self, sampSplit, labelsSplit):
        """Find the best feature for a split."""
        c_root = self.calculate_gain(labelsSplit)
        minimum = np.inf
        best_feature = threshold = None
        unique_vals_means = [[(sortedData[i] + sortedData[i + 1]) / 2 for i in range(len(sortedData) - 1)]
                             for sortedData in [np.sort(np.unique(sampSplit[:, k]), kind='mergesort') for k in range(sampSplit.shape[1])]]
        for k, means in enumerate(unique_vals_means):
            for cal_mean in means:
                X_left, y_left, X_right, y_right = self.split_threshold(
                    sampSplit, labelsSplit, k, cal_mean)
                c_left = self.calculate_gain(y_left)
                c_right = self.calculate_gain(y_right)
                gain = c_left + c_right - c_root
                if minimum > gain:
                    minimum, best_feature, threshold = gain, k, cal_mean
        return best_feature, threshold, minimum

    def find_best_leaf(self):
        """Find the best leaf for splitting."""
        self.final_index = None
        self.sml_c = np.inf
        self.i = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.left_child or node.right_child:
                if node.right_child:
                    stack.append(node.right_child)
                if node.left_child:
                    stack.append(node.left_child)
            else:
                if (self.max_depth is None or node.depth < self.max_depth) and len(node.samples) >= self.min_to_split:
                    _, _, c = self.find_best_feature(
                        node.samples, node.targets)
                    if c < self.sml_c:
                        self.sml_c, self.final_index = c, self.i
                self.i += 1
        return self.final_index

    def create_tree(self, node, depth):
        """Create the decision tree."""
        if (self.max_depth is not None and depth >= self.max_depth) or len(node.samples) < self.min_to_split or (self.max_leaves is not None and self.n_leaves >= self.max_leaves):
            return
        X_left, y_left, X_right, y_right = self.split_threshold(
            node.samples, node.targets, node.feature_ind, node.threshold)
        node.left_child = self._create_child(X_left, y_left, depth + 1)
        node.right_child = self._create_child(X_right, y_right, depth + 1)
        self.n_leaves += 1
        self.create_tree(node.left_child, depth + 1)
        self.create_tree(node.right_child, depth + 1)

    def _create_child(self, X, y, depth):
        """Helper function to create a child node."""
        if len(X) < self.min_to_split:
            return Node(X, y, depth=depth)
        feature_ind, threshold, _ = self.find_best_feature(X, y)
        return Node(X, y, feature_ind=feature_ind, threshold=threshold, depth=depth)

    def split_node(self, ind):
        """Split a node at a certain index."""
        n = self.root
        self.i = 0

        def create_node(X, y, depth):
            if len(X) < self.min_to_split:
                return Node(X, y, depth=depth)
            feature_ind, threshold, _ = self.find_best_feature(X, y)
            return Node(X, y, feature_ind=feature_ind, threshold=threshold, depth=depth)

        def leaves(node, ind):
            if node.left_child is None and node.right_child is None:
                if (self.max_depth is None or node.depth < self.max_depth) and len(node.samples) >= self.min_to_split:
                    if ind == self.i:
                        X_left, y_left, X_right, y_right = self.split_threshold(
                            node.samples, node.targets, node.feature_ind, node.threshold)

                        node.left_child = create_node(
                            X_left, y_left, node.depth + 1)
                        node.right_child = create_node(
                            X_right, y_right, node.depth + 1)
                        self.n_leaves += 1
                        return True
                self.i += 1

            if node.left_child and leaves(node.left_child, ind):
                return True
            if node.right_child and leaves(node.right_child, ind):
                return True
            return False

        leaves(n, ind)

    def optimized_tree(self):
        """Optimize the decision tree."""
        if self.max_leaves is None:
            return

        while self.n_leaves < self.max_leaves:
            ind = self.find_best_leaf()
            if ind is None:
                break
            self.split_node(ind)

    def best_class(self, targets):
        """Determine the best class from the target values."""
        return np.bincount(targets).argmax()

    def fit(self, X, targets):
        """Function that fits the tree with the given data"""
        best_feat, threshold, _ = self.find_best_feature(X, targets)
        self.root = Node(X, targets, best_feat, threshold, depth=0)
        self.n_leaves = 1
        self.optimized_tree() if self.max_leaves is not None else self.create_tree(self.root, 0)

    def predict(self, X_test):
        """Predict the class for each sample in X_test."""
        preds = []
        for x in X_test:
            node = self.root
            while node:
                if node.threshold is None:
                    preds.append(self.best_class(node.targets))
                    break
                node = node.left_child if x[node.feature_ind] < node.threshold else node.right_child
                if node and node.left_child is None and node.right_child is None:
                    preds.append(self.best_class(node.targets))
                    break
        return preds


def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(
        args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)
    tree = DecisionTreeClassifier(args.criterion, args.max_depth,
                                  args.min_to_split, args.max_leaves)
    tree.fit(train_data, train_target)

    train_accuracy = accuracy_score(
        train_target, tree.predict(train_data)) * 100
    test_accuracy = accuracy_score(test_target, tree.predict(test_data)) * 100
    return train_accuracy, test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(train_accuracy))
    print("Test accuracy: {:.1f}%".format(test_accuracy))
