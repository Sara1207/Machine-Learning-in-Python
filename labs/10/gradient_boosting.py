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
parser.add_argument("--dataset", default="wine",
                    type=str, help="Dataset to use")
parser.add_argument("--l2", default=1., type=float,
                    help="L2 regularization factor")
parser.add_argument("--learning_rate", default=0.1,
                    type=float, help="Learning rate")
parser.add_argument("--max_depth", default=None, type=int,
                    help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x: int(x)
                    if x.isdigit() else float(x), help="Test size")
parser.add_argument("--trees", default=1, type=int,
                    help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.


class DecisionTree:
    def __init__(self, regularization_param, max_depth):
        self.regularization_param = regularization_param
        self.max_depth = max_depth

    class TreeNode:
        def __init__(self, instances, prediction):
            self.is_leaf = True
            self.instances = instances
            self.prediction = prediction

        def split(self, feature_index, threshold, left_child, right_child):
            self.is_leaf = False
            self.feature_index = feature_index
            self.threshold = threshold
            self.left_child = left_child
            self.right_child = right_child

    def fit(self, features, gradients, hessians):
        self.features = features
        self.gradients = gradients
        self.hessians = hessians

        initial_instances = np.arange(len(self.features))
        self.root = self._create_leaf_node(initial_instances)
        self._split_nodes(self.root, 0)
        return self

    # Method for making predictions
    def predict(self, data):
        predictions = np.zeros(len(data), dtype=np.float32)
        for i, instance in enumerate(data):
            node = self.root
            while not node.is_leaf:
                node = node.left_child if instance[node.feature_index] <= node.threshold else node.right_child
            predictions[i] = node.prediction
        return predictions

    # Method to split the nodes
    def _split_nodes(self, node, depth):
        if not self._is_split_possible(node, depth):
            return

        best_feature_index, threshold, left_instances, right_instances = self._find_best_split(
            node)
        node.split(best_feature_index, threshold, self._create_leaf_node(
            left_instances), self._create_leaf_node(right_instances))
        self._split_nodes(node.left_child, depth + 1)
        self._split_nodes(node.right_child, depth + 1)

    # Method to check if a node can be split
    def _is_split_possible(self, node, depth):
        within_depth_limit = (self.max_depth is None) or (
            depth < self.max_depth)
        has_multiple_instances = len(node.instances) > 1
        return within_depth_limit and has_multiple_instances

    # Method to determine the best split for a node
    def _find_best_split(self, node):
        best_criterion = float('inf')
        best_split = (None, None, None, None)
        for feature_index in range(self.features.shape[1]):
            feature_values = self.features[node.instances, feature_index]
            unique_values = np.unique(feature_values)

            for i in range(len(unique_values) - 1):
                threshold = (unique_values[i] + unique_values[i + 1]) / 2
                left_mask = feature_values <= threshold
                right_mask = feature_values > threshold

                left_instances = node.instances[left_mask]
                right_instances = node.instances[right_mask]
                split_criterion = self._compute_criterion(
                    left_instances) + self._compute_criterion(right_instances)

                if split_criterion < best_criterion:
                    best_criterion = split_criterion
                    best_split = (feature_index, threshold,
                                  left_instances, right_instances)

        return best_split

    # Method to compute the criterion for a set of instances
    def _compute_criterion(self, instances):
        sum_gradients = np.sum(self.gradients[instances])
        sum_hessians = np.sum(
            self.hessians[instances]) + self.regularization_param
        return -0.5 * (sum_gradients ** 2) / sum_hessians

    # Method to create a leaf node
    def _create_leaf_node(self, instances):
        sum_gradients = np.sum(self.gradients[instances])
        sum_hessians = np.sum(
            self.hessians[instances]) + self.regularization_param
        leaf_prediction = -sum_gradients / sum_hessians
        return self.TreeNode(instances, leaf_prediction)


class GradientBoostedTrees:
    def __init__(self, trees, learning_rate, l2, max_depth, classes):
        self._num_trees = trees
        self._learning_rate = learning_rate
        self._l2 = l2
        self._max_depth = max_depth
        self._classes = classes

    def _softmax(self, logits):
        """Compute the softmax of the provided logits."""
        exp_logits = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
        return exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

    def _compute_gradients_and_hessians(self, probabilities, targets, class_label):
        """Compute gradients and hessians for the given probabilities and targets."""
        gs = probabilities[:, class_label] - (targets == class_label)
        hs = probabilities[:, class_label] * \
            (1 - probabilities[:, class_label])
        return gs, hs

    def fit(self, data, targets):
        self._classes = np.max(targets) + 1
        self._trees = []
        predictions = np.zeros((len(data), self._classes), np.float32)

        for _ in range(self._num_trees):
            probabilities = self._softmax(predictions)

            trees_for_this_iteration = []
            for class_label in range(self._classes):
                gs, hs = self._compute_gradients_and_hessians(
                    probabilities, targets, class_label)
                tree = DecisionTree(
                    self._l2, self._max_depth).fit(data, gs, hs)
                trees_for_this_iteration.append(tree)

                predictions[:, class_label] += self._learning_rate * \
                    tree.predict(data)

            self._trees.append(trees_for_this_iteration)

    def predict(self, data, num_trees):
        """Predict class labels for the given data using the first 'num_trees' trees."""
        # Initialize the array to store predictions for each class
        class_predictions = np.zeros((len(data), self._classes), np.float32)

        # Use the first 'num_trees' in the ensemble for prediction
        for ensemble_trees in self._trees[:num_trees]:
            for class_index, tree in enumerate(ensemble_trees):
                # Aggregate predictions from each tree
                class_predictions[:, class_index] += tree.predict(data)

        # Determine the final class prediction by choosing the class with the highest score
        final_predictions = np.argmax(class_predictions, axis=1)
        return final_predictions


def main(args: argparse.Namespace) -> tuple[list[float], list[float]]:
    # Use the given dataset.
    data, target = getattr(sklearn.datasets, "load_{}".format(
        args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    classes = np.max(target) + 1

    GradientBoostedTree = GradientBoostedTrees(
        args.trees, args.learning_rate, args.l2, args.max_depth, classes)

    GradientBoostedTree.fit(train_data, train_target)

    train_accuracies, test_accuracies = [], []

    for t in range(args.trees):
        train_accuracies.append(sklearn.metrics.accuracy_score(
            train_target, GradientBoostedTree.predict(train_data, t + 1)))
        test_accuracies.append(sklearn.metrics.accuracy_score(
            test_target, GradientBoostedTree.predict(test_data, t + 1)))

    return [100 * acc for acc in train_accuracies], [100 * acc for acc in test_accuracies]


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracies, test_accuracies = main(args)

    for i, (train_accuracy, test_accuracy) in enumerate(zip(train_accuracies, test_accuracies)):
        print("Using {} trees, train accuracy: {:.1f}%, test accuracy: {:.1f}%".format(
            i + 1, train_accuracy, test_accuracy))