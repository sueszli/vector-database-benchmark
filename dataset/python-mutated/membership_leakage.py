"""
This module implements membership leakage metrics.
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from typing import TYPE_CHECKING, Optional, Tuple
from enum import Enum, auto
import numpy as np
import scipy
from sklearn.neighbors import KNeighborsClassifier
from art.utils import check_and_transform_label_format, is_probability_array
if TYPE_CHECKING:
    from art.utils import CLASSIFIER_TYPE

class ComparisonType(Enum):
    """
    An Enum type for different kinds of comparisons between model outputs.
    """
    RATIO = auto()
    DIFFERENCE = auto()

def PDTP(target_estimator: 'CLASSIFIER_TYPE', extra_estimator: 'CLASSIFIER_TYPE', x: np.ndarray, y: np.ndarray, indexes: Optional[np.ndarray]=None, num_iter: int=10, comparison_type: Optional[ComparisonType]=ComparisonType.RATIO) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if False:
        i = 10
        return i + 15
    '\n    Compute the pointwise differential training privacy metric for the given classifier and training set.\n\n    | Paper link: https://arxiv.org/abs/1712.09136\n\n    :param target_estimator: The classifier to be analyzed.\n    :param extra_estimator: Another classifier of the same type as the target classifier, but not yet fit.\n    :param x: The training data of the classifier.\n    :param y: Target values (class labels) of `x`, one-hot-encoded of shape (nb_samples, nb_classes) or indices of\n              shape (nb_samples,).\n    :param indexes: the subset of indexes of `x` to compute the PDTP metric on. If not supplied, PDTP will be\n                    computed for all samples in `x`.\n    :param num_iter: the number of iterations of PDTP computation to run for each sample. If not supplied,\n                     defaults to 10. The result is the average across iterations.\n    :param comparison_type: the way in which to compare the model outputs between models trained with and without\n                            a certain sample. Default is to compute the ratio.\n    :return: A tuple of three arrays, containing the average (worse, standard deviation) PDTP value for each sample in\n             the training set respectively. The higher the value, the higher the privacy leakage for that sample.\n    '
    from art.estimators.classification.pytorch import PyTorchClassifier
    from art.estimators.classification.tensorflow import TensorFlowV2Classifier
    from art.estimators.classification.scikitlearn import ScikitlearnClassifier
    supported_classifiers = (PyTorchClassifier, TensorFlowV2Classifier, ScikitlearnClassifier)
    if not isinstance(target_estimator, supported_classifiers) or not isinstance(extra_estimator, supported_classifiers):
        raise ValueError('PDTP metric only supports classifiers of type PyTorch, TensorFlowV2 and ScikitLearn.')
    if target_estimator.input_shape[0] != x.shape[1]:
        raise ValueError('Shape of x does not match input_shape of classifier')
    y = check_and_transform_label_format(y, nb_classes=target_estimator.nb_classes)
    if y.shape[0] != x.shape[0]:
        raise ValueError('Number of rows in x and y do not match')
    results = []
    for _ in range(num_iter):
        iter_results = []
        pred = target_estimator.predict(x)
        if not is_probability_array(pred):
            try:
                pred = scipy.special.softmax(pred, axis=1)
            except Exception as exc:
                raise ValueError('PDTP metric only supports classifiers that output logits or probabilities.') from exc
        bins = np.array(np.arange(0.0, 1.01, 0.01).round(decimals=2))
        pred_bin_indexes = np.digitize(pred, bins)
        pred_bin_indexes[pred_bin_indexes == 101] = 100
        pred_bin = bins[pred_bin_indexes] - 0.005
        if indexes is None:
            indexes = np.array(range(x.shape[0]))
        if indexes is not None:
            for row in indexes:
                alt_x = np.delete(x, row, 0)
                alt_y = np.delete(y, row, 0)
                try:
                    extra_estimator.reset()
                except NotImplementedError as exc:
                    raise ValueError('PDTP metric can only be applied to classifiers that implement the reset method.') from exc
                extra_estimator.fit(alt_x, alt_y)
                alt_pred = extra_estimator.predict(x)
                if not is_probability_array(alt_pred):
                    alt_pred = scipy.special.softmax(alt_pred, axis=1)
                alt_pred_bin_indexes = np.digitize(alt_pred, bins)
                alt_pred_bin_indexes[alt_pred_bin_indexes == 101] = 100
                alt_pred_bin = bins[alt_pred_bin_indexes] - 0.005
                if comparison_type == ComparisonType.RATIO:
                    ratio_1 = pred_bin / alt_pred_bin
                    ratio_2 = alt_pred_bin / pred_bin
                    max_value: float = max(ratio_1.max(), ratio_2.max())
                elif comparison_type == ComparisonType.DIFFERENCE:
                    max_value = np.max(abs(pred_bin - alt_pred_bin))
                else:
                    raise ValueError('Unsupported comparison type.')
                iter_results.append(max_value)
            results.append(iter_results)
    per_sample: list[list[float]] = list(map(list, zip(*results)))
    avg_per_sample = np.array([sum(val) / len(val) for val in per_sample])
    worse_per_sample = np.max(per_sample, axis=1)
    std_dev_per_sample = np.std(per_sample, axis=1)
    return (avg_per_sample, worse_per_sample, std_dev_per_sample)

def SHAPr(target_estimator: 'CLASSIFIER_TYPE', x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, knn_metric: Optional[str]=None) -> np.ndarray:
    if False:
        return 10
    "\n    Compute the SHAPr membership privacy risk metric for the given classifier and training set.\n\n    | Paper link: http://arxiv.org/abs/2112.02230\n\n    :param target_estimator: The classifier to be analyzed.\n    :param x_train: The training data of the classifier.\n    :param y_train: Target values (class labels) of `x_train`, one-hot-encoded of shape (nb_samples, nb_classes) or\n                    indices of shape (nb_samples,).\n    :param x_test: The test data of the classifier.\n    :param y_test: Target values (class labels) of `x_test`, one-hot-encoded of shape (nb_samples, nb_classes) or\n                    indices of shape (nb_samples,).\n    :param knn_metric: The distance metric to use for the KNN classifier (default is 'minkowski', which represents\n                       Euclidean distance).\n    :return: an array containing the SHAPr scores for each sample in the training set. The higher the value,\n             the higher the privacy leakage for that sample. Any value above 0 should be considered a privacy leak.\n    "
    if target_estimator.input_shape[0] != x_train.shape[1]:
        raise ValueError('Shape of x_train does not match input_shape of classifier')
    if x_test.shape[1] != x_train.shape[1]:
        raise ValueError('Shape of x_train does not match the shape of x_test')
    y_train = check_and_transform_label_format(y_train, target_estimator.nb_classes)
    if y_train.shape[0] != x_train.shape[0]:
        raise ValueError('Number of rows in x_train and y_train do not match')
    y_test = check_and_transform_label_format(y_test, target_estimator.nb_classes)
    if y_test.shape[0] != x_test.shape[0]:
        raise ValueError('Number of rows in x_test and y_test do not match')
    n_train_samples = x_train.shape[0]
    pred_train = target_estimator.predict(x_train)
    pred_test = target_estimator.predict(x_test)
    if knn_metric:
        knn = KNeighborsClassifier(metric=knn_metric)
    else:
        knn = KNeighborsClassifier()
    knn.fit(pred_train, y_train)
    results = []
    n_test = pred_test.shape[0]
    for i_test in range(n_test):
        results_test = []
        pred = pred_test[i_test]
        y_0 = y_test[i_test]
        n_indexes = knn.kneighbors([pred], n_neighbors=n_train_samples, return_distance=False)
        n_indexes = n_indexes.reshape(-1)[::-1]
        sorted_y_train = y_train[n_indexes]
        sorted_indexes = np.argsort(n_indexes)
        first = True
        phi_y_prev: float = 0.0
        y_indicator_prev = 0
        for i_train in range(sorted_y_train.shape[0]):
            y = sorted_y_train[i_train]
            y_indicator = 1 if np.all(y == y_0) else 0
            if first:
                phi_y = y_indicator / n_train_samples
                first = False
            else:
                phi_y = phi_y_prev + (y_indicator - y_indicator_prev) / (n_train_samples - i_train)
            results_test.append(phi_y)
            phi_y_prev = phi_y
            y_indicator_prev = y_indicator
        results_test_sorted = np.array(results_test)[sorted_indexes]
        results.append(results_test_sorted.tolist())
    per_sample = list(map(list, zip(*results)))
    sum_per_sample = np.array([sum(val) for val in per_sample], dtype=np.float32) * n_train_samples / n_test
    return sum_per_sample