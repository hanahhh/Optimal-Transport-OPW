import argparse
import random

import numpy as np
import ot

np.random.seed(42)
random.seed(42)
sklearn_seed = 0

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from config.config import WEI_PATH, logger
from src.dp.exact_dp import drop_dtw_distance, dtw_distance
from src.experiments.weizmann.dataset import WeisDataset
from src.experiments.weizmann.utils import add_outlier
from src.pow.pow import pow_distance
from src.utils.knn_utils import knn_classifier_from_distance_matrix
from src.experiments.weizmann.arabic import get_test_data, get_train_data
from src.opw.opw import opw_distance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_size", type=float, default=0.5)
    parser.add_argument("--outlier_ratio", type=float, default=0.1)
    parser.add_argument("--metric", type=str)
    parser.add_argument("--m", type=float, default=0.1)
    parser.add_argument("--reg", type=int, default=1)
    parser.add_argument("--distance", type=str, default="euclidean")
    parser.add_argument("--k", type=int, default=1)
    args = parser.parse_args()
    return args


def main(args):
    logger.info(f"Args: {args}")
    weis_dataset = WeisDataset.from_folder(WEI_PATH, test_size=args.test_size)
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()

    # X_train = [weis_dataset.get_sequence(idx) for idx in weis_dataset.train_idx]
    # X_test = [weis_dataset.get_sequence(idx) for idx in weis_dataset.test_idx]

    # X_test = list(
    #     map(lambda x: add_outlier(x, outlier_ratio=args.outlier_ratio), X_test)
    # )

    # y_train = (weis_dataset.get_label(idx) for idx in weis_dataset.train_idx)
    # y_test = (weis_dataset.get_label(idx) for idx in weis_dataset.test_idx)
    # y_train = np.array(list(y_train))
    # y_test = np.array(list(y_test))

    # print("+++++++++++++++++++++++++++++++++++++++++\n")
    # print(f"type X_Train: {type(X_train)}\n")
    # print(f"type y_Train: {type(y_train)}\n")
    # print(f"length X_train, y_Train: {len(X_train)}, {len(y_train)}\n")
    # print(f"y_Train: {(y_train)}\n")
    # print(f"X_Train[0]: {(X_train[0])}\n")
    # print(f"type X_Train[0]: {type(X_train[0])}\n")
    # print(f"length X_Train[0]: {len(X_train[0])}\n")
    # print(f"length X_Train[1]: {len(X_train[1])}\n")
    # print("+++++++++++++++++++++++++++++++++++++++++\n")

    fn_dict = {
        "opw": opw_distance,
        "pow": pow_distance,
        "dtw": dtw_distance,
        "drop_dtw": drop_dtw_distance,
    }

    train_size = len(X_train)
    test_size = len(X_test)
    X_train[0].shape[1]
    logger.info(f"Train size: {train_size}")
    logger.info(f"Test size: {test_size}")

    result = np.zeros((test_size, train_size))
    for train_idx in tqdm(range(train_size)):
        for test_idx in tqdm(range(test_size), leave=False):
            M = ot.dist(X_train[train_idx], X_test[test_idx], metric=args.distance)
            if args.metric == "pow":
                distance = fn_dict[args.metric](M, m=args.m, reg=args.reg)

            elif args.metric == "drop_dtw":
                distance = fn_dict[args.metric](M, keep_percentile=args.m)
            elif args.metric == "dtw":
                distance = fn_dict[args.metric](M)
            elif args.metric == "opw":
                distance = fn_dict[args.metric](M)
            if distance == np.inf:
                distance = np.max(result)
            result[test_idx, train_idx] = distance

    y_pred = knn_classifier_from_distance_matrix(
        distance_matrix=result,
        k=args.k,
        labels=y_train,
    )
    accuracy = accuracy_score(y_test, y_pred)

    logger.info(f"Accuracy: {accuracy}")
    


if __name__ == "__main__":
    args = parse_args()
    main(args)
