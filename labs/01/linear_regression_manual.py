#!/usr/bin/env python3
# Other team member:
# e4c904b6-5b87-405b-bf67-4d6d7fe3984d

import argparse

import numpy as np
import sklearn.datasets
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.1, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load the diabetes dataset.
    dataset = sklearn.datasets.load_diabetes()

    # The input data are in `dataset.data`, targets are in `dataset.target`.

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # print(dataset.DESCR)
    #print(dataset.data)

    # TODO: Append a constant feature with value 1 to the end of every input data.
    # Then we do not need to explicitly represent bias - it becomes the last weight.
    
    dataset.data = np.column_stack((dataset.data, np.ones(len(dataset.data))))
    # print()
    # print(dataset.data)

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.

    from sklearn.model_selection import train_test_split

    # Split the dataset into train and test sets
    train_setA, test_setA, train_setB, test_setB = train_test_split(
        dataset.data,  
        dataset.target,
        test_size=args.test_size,  
        random_state=args.seed  
    )

    # TODO: Solve the linear regression using the algorithm from the lecture,
    # explicitly computing the matrix inverse (using `np.linalg.inv`).

    # w = (train_setA^T * train_setA)^(-1) * train_setA^T * train_setB

    # Transpose train_setA
    train_setA_transpose = np.transpose(train_setA)

    # Product of train_setA^T * train_setA
    aa = np.dot(train_setA_transpose, train_setA)

    # Inverse of aa using `np.linalg.inv
    aa_inv = np.linalg.inv(aa)

    # Compute (train_setA^T * train_setA)^(-1) * train_setA^T
    A_weight = np.dot(aa_inv, train_setA_transpose)

    # Final weights, coefficients of linear regression model
    w = np.dot(A_weight, train_setB)

    # print("Weights: ",w)

    # TODO: Predict target values on the test set.
    target_predict = np.dot(test_setA, w)
    # print("Predicted target values", target_predict)

    # TODO: Manually compute root mean square error on the test set predictions.
    # Calculate errors
    errors = test_setB - target_predict

    # print("errors:", errors)

    # Square the errors
    squared_errors = np.square(errors)
    # print("Squared errors:", squared_errors)

    # Mean squared error
    sum = np.sum(squared_errors)
    mse = sum/(len(squared_errors))
    # print("mse:", mse)

    rmse = np.sqrt(mse)

    print("RMSE:", rmse)

    return rmse


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    rmse = main(args)
    print("{:.2f}".format(rmse))
