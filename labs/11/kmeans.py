#!/usr/bin/env python3

# Team members:
# Sara Pachemska (me):
# b3ccad7b-ac26-4440-b7fb-0edf0a8ba4a8

# Natalia Tashkova :
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

import argparse

import numpy as np

import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=3, type=int, help="Number of clusters")
parser.add_argument("--examples", default=200, type=int, help="Number of examples")
parser.add_argument("--init", default="random", choices=["random", "kmeans++"], help="Initialization")
parser.add_argument("--iterations", default=20, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.


def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        plt.gcf().get_axes() or plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    plt.show() if args.plot is True else plt.savefig(args.plot, transparent=True, bbox_inches="tight")


def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed.
    generator = np.random.RandomState(args.seed)

    # Generate an artificial dataset.
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO: Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.

    #RANDOM
    if args.init == "random":
        indices = generator.choice(len(data), size=args.clusters, replace=False)
        centers = data[indices]

    #KMEANS++
    elif args.init == "kmeans++":
        centers = np.zeros((args.clusters, data.shape[1]))
        # First center randomly with randint
        centers[0] = data[generator.randint(len(data))]  

        # iteratively sample the rest of the clusters
        for i in range(1, args.clusters):

            # calculate distance
            distances = np.array([min([np.linalg.norm(p - c) for c in centers[:i]]) for p in data])

            # calculate the probabilities
            probabilities = distances**2 / np.sum(distances**2)

            # next center
            centers[i] = data[generator.choice(len(data), p=probabilities)]

        centers = np.array(centers)

    clusters = np.zeros(len(data))

    if args.plot:
        plot(args, 0, data, centers, clusters=None)

    # Run `args.iterations` of the K-Means algorithm.
    for j in range(args.iterations):
        distances = np.array([np.linalg.norm(point - centers, axis=1) for point in data])
        l = len(data)
        distances = distances.reshape(l, -1)

        clusters = np.argmin(distances, axis=1)

        # update cluster 
        centers = np.array([data[clusters == k].mean(axis=0) if np.sum(clusters == k) > 0 else np.random.choice(data, size=1) for k in range(args.clusters)])

        if args.plot:
            plot(args, 1 + j, data, centers, clusters)

    return clusters

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    clusters = main(args)
    print("Cluster assignments:", clusters, sep="\n")
