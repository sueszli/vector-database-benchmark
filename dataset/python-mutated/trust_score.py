"""Module of trust score confidence method.

Code is taken from https://github.com/google/TrustScore
Based on: arXiv:1805.11783 [stat.ML]

Used according to the following License:

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from typing import Tuple
import numpy as np
from sklearn.neighbors import KDTree, KNeighborsClassifier
from deepchecks.utils.logger import get_logger
__all__ = ['TrustScore']

class TrustScore:
    """Calculate trust score.

    Parameters
    ----------
    k_filter : int , default: 10
        Number of neighbors used during either kNN distance or probability filtering.
    alpha : float , default: 0.
        Fraction of instances to filter out to reduce impact of outliers.
    filter_type : str , default: distance_knn
        Filter method; either 'distance_knn' or 'probability_knn'
    leaf_size : int , default: 40
        Number of points at which to switch to brute-force. Affects speed and memory required to
        build trees. Memory to store the tree scales with n_samples / leaf_size.
    metric : str , default: euclidean
        Distance metric used for the tree. See sklearn's DistanceMetric class for a list of available
        metrics.
    dist_filter_type : str , default: point
        Use either the distance to the k-nearest point (dist_filter_type = 'point') or
        the average distance from the first to the k-nearest point in the data (dist_filter_type = 'mean').
    """

    def __init__(self, k_filter: int=10, alpha: float=0.0, filter_type: str='distance_knn', leaf_size: int=40, metric: str='euclidean', dist_filter_type: str='point') -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.k_filter = k_filter
        self.alpha = alpha
        self.filter = filter_type
        self.eps = 1e-12
        self.leaf_size = leaf_size
        self.metric = metric
        self.dist_filter_type = dist_filter_type

    def filter_by_distance_knn(self, X: np.ndarray) -> np.ndarray:
        if False:
            for i in range(10):
                print('nop')
        'Filter out instances with low kNN density.\n\n        Calculate distance to k-nearest point in the data for each instance and remove instances above a cutoff\n        distance.\n\n        Parameters\n        ----------\n        X : np.ndarray\n            Data to filter\n        Returns\n        -------\n        np.ndarray\n            Filtered data\n        '
        kdtree = KDTree(X, leaf_size=self.leaf_size, metric=self.metric)
        k = min(self.k_filter + 1, len(X))
        knn_r = kdtree.query(X, k=k)[0]
        if self.dist_filter_type == 'point':
            knn_r = knn_r[:, -1]
        elif self.dist_filter_type == 'mean':
            knn_r = np.mean(knn_r[:, 1:], axis=1)
        cutoff_r = np.percentile(knn_r, (1 - self.alpha) * 100)
        X_keep = X[np.where(knn_r <= cutoff_r)[0], :]
        return X_keep

    def filter_by_probability_knn(self, X: np.ndarray, Y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if False:
            return 10
        'Filter out instances with high label disagreement amongst its k nearest neighbors.\n\n        Parameters\n        ----------\n        X : np.ndarray\n            Data\n        Y : np.ndarray\n            Predicted class labels\n        Returns\n        -------\n        Tuple[np.ndarray, np.ndarray]\n            Filtered data and labels.\n        '
        if self.k_filter == 1:
            get_logger().warning('Number of nearest neighbors used for probability density filtering should be >1, otherwise the prediction probabilities are either 0 or 1 making probability filtering useless.')
        clf = KNeighborsClassifier(n_neighbors=self.k_filter, leaf_size=self.leaf_size, metric=self.metric)
        clf.fit(X, Y)
        preds_proba = clf.predict_proba(X)
        preds_max = np.max(preds_proba, axis=1)
        cutoff_proba = np.percentile(preds_max, self.alpha * 100)
        keep_id = np.where(preds_max >= cutoff_proba)[0]
        (X_keep, Y_keep) = (X[keep_id, :], Y[keep_id])
        return (X_keep, Y_keep)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> None:
        if False:
            return 10
        'Build KDTrees for each prediction class.\n\n        Parameters\n        ----------\n        X : np.ndarray\n            Data.\n        Y : np.ndarray\n            Target labels, either one-hot encoded or the actual class label.\n        '
        if len(X.shape) > 2:
            get_logger().warning('Reshaping data from %s to %s so k-d trees can be built.', X.shape, X.reshape(X.shape[0], -1).shape)
            X = X.reshape(X.shape[0], -1)
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)
            self.classes = Y.shape[1]
        else:
            self.classes = len(np.unique(Y))
        self.kdtrees = [None] * self.classes
        self.X_kdtree = [None] * self.classes
        if self.filter == 'probability_knn':
            (X_filter, Y_filter) = self.filter_by_probability_knn(X, Y)
        for c in range(self.classes):
            if self.filter is None:
                X_fit = X[np.where(Y == c)[0]]
            elif self.filter == 'distance_knn':
                X_fit = self.filter_by_distance_knn(X[np.where(Y == c)[0]])
            elif self.filter == 'probability_knn':
                X_fit = X_filter[np.where(Y_filter == c)[0]]
            else:
                raise Exception('self.filter must be one of ["distance_knn", "probability_knn", None]')
            no_x_fit = len(X_fit) == 0
            if no_x_fit or len(X[np.where(Y == c)[0]]) == 0:
                if no_x_fit and len(X[np.where(Y == c)[0]]) == 0:
                    get_logger().warning('No instances available for class %s', c)
                elif no_x_fit:
                    get_logger().warning('Filtered all the instances for class %s. Lower alpha or check data.', c)
            else:
                self.kdtrees[c] = KDTree(X_fit, leaf_size=self.leaf_size, metric=self.metric)
                self.X_kdtree[c] = X_fit

    def score(self, X: np.ndarray, Y: np.ndarray, k: int=2, dist_type: str='point') -> Tuple[np.ndarray, np.ndarray]:
        if False:
            print('Hello World!')
        "Calculate trust scores.\n\n        ratio of distance to closest class other than the predicted class to distance to predicted class.\n\n        Parameters\n        ----------\n        X : np.ndarray\n            Instances to calculate trust score for.\n        Y : np.ndarray\n            Either prediction probabilities for each class or the predicted class.\n        k : int , default: 2\n            Number of nearest neighbors used for distance calculation.\n        dist_type : str , default: point\n            Use either the distance to the k-nearest point (dist_type = 'point') or the average\n            distance from the first to the k-nearest point in the data (dist_type = 'mean').\n        Returns\n        -------\n        Tuple[np.ndarray, np.ndarray]\n            Batch with trust scores and the closest not predicted class.\n        "
        if len(Y.shape) > 1:
            Y = np.argmax(Y, axis=1)
        if len(X.shape) > 2:
            X = X.reshape(X.shape[0], -1)
        d = np.tile(None, (X.shape[0], self.classes))
        for c in range(self.classes):
            if self.kdtrees[c] is None or self.kdtrees[c].data.shape[0] < k:
                d[:, c] = np.inf
            else:
                d_tmp = self.kdtrees[c].query(X, k=k)[0]
                if dist_type == 'point':
                    d[:, c] = d_tmp[:, -1]
                elif dist_type == 'mean':
                    d[:, c] = np.nanmean(d_tmp[np.isfinite(d_tmp)], axis=1)
        sorted_d = np.sort(d, axis=1)
        d_to_pred = d[range(d.shape[0]), Y]
        d_to_closest_not_pred = np.where(sorted_d[:, 0] != d_to_pred, sorted_d[:, 0], sorted_d[:, 1])
        trust_score = d_to_closest_not_pred / (d_to_pred + self.eps)
        class_closest_not_pred = np.where(d == d_to_closest_not_pred.reshape(-1, 1))[1]
        return (trust_score, class_closest_not_pred)

    @staticmethod
    def process_confidence_scores(baseline_scores: np.ndarray, test_scores: np.ndarray):
        if False:
            i = 10
            return i + 15
        'Process confidence scores.'
        filter_center_factor = 4
        filter_center_size = 40.0
        baseline_confidence = baseline_scores
        if test_scores is None:
            test_confidence = baseline_scores
        else:
            test_confidence = test_scores
        center_size = max(np.nanpercentile(baseline_confidence, 50 + filter_center_size / 2), np.nanpercentile(test_confidence, 50 + filter_center_size / 2)) - min(np.nanpercentile(baseline_confidence, 50 - filter_center_size / 2), np.nanpercentile(test_confidence, 50 - filter_center_size / 2))
        max_median = max(np.nanmedian(baseline_confidence), np.nanmedian(test_confidence))
        min_median = min(np.nanmedian(baseline_confidence), np.nanmedian(test_confidence))
        upper_thresh = max_median + filter_center_factor * center_size
        lower_thresh = min_median - filter_center_factor * center_size
        baseline_confidence[(baseline_confidence > upper_thresh) | (baseline_confidence < lower_thresh)] = np.nan
        test_confidence[(test_confidence > upper_thresh) | (test_confidence < lower_thresh)] = np.nan
        baseline_confidence = baseline_confidence.astype(float)
        test_confidence = test_confidence.astype(float)
        if test_scores is None:
            test_confidence = None
        return (baseline_confidence, test_confidence)