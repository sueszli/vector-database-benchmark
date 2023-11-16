"""
Implementation of historically weighted regression method based on relevance.
"""
import warnings
from typing import Tuple
import numpy as np

class HistoryWeightRegression:
    """
    The class that houses all related methods for the historically weighted regression tool.
    """

    def __init__(self, Y_train: np.array, X_train: np.array, check_condi_num: bool=False):
        if False:
            return 10
        '\n        Instantiate the class with data.\n\n        :param Y_train: (np.array) The 1D (n, ) dependent data vector.\n        :param X_train:  (np.array) The 2D (n-by-k) independent data vector, n: num of instances, k: num of variables\n            or features.\n        :param check_condi_num: (bool) Optional. Whether to check the condition number of the covariance matrix and\n            fisher info matrix from the training data X (Their values are the same). If this number is too large then it\n            may lead to numerical issues. Defaults to False. Toggle this off to save some computing time.\n        '
        pass

    def get_fit_result(self) -> dict:
        if False:
            i = 10
            return i + 15
        '\n        Fit result and statistics using the training data.\n\n        :return: (dict) The fit result and associated statistics.\n        '
        pass

    def predict(self, X_t: np.array, relev_ratio_threshold: float=1) -> np.array:
        if False:
            while True:
                i = 10
        '\n        Predict the result using fitted model from a subsample chosen by the ratio of relevance.\n\n        For example, if relev_ratio_threshold = 0.4, then it chooses the top 40 percentile data ranked by relevance to\n        x_t. This method returns the prediction in column 0, also returns the associated prediction standard\n        deviations in the column 1.\n\n        For each row element x_t in X_t we have the following:\n        y_t := y_avg + 1/(n-1) * sum{relevance(x_i, x_t) * (y_i - y_avg), subsample}\n        where y_i, x_i are from subsamples. The matrix form is:\n        y_t := y_avg + 1/(n-1) * (x_t - x_avg).T @ fisher_info_mtx @ (X_sub - x_avg).T @ (y_sub - y_avg)\n\n        :param X_t: (np.array) The 2D (n_t-by-k) test data, n_t is the number of instances, k is the number of\n            variables or features.\n        :param relev_ratio_threshold: (float) Optional. The subsample ratio to use for predicting values ranked by\n            relevance, must be a number between [0, 1]. For example, 0.6 corresponds to the top 60 percentile data\n            ranked by relevance to x_t. Defaults to 1.\n        :return: (np.array) The predicted results in col 0, and standard deviations in col 1.\n        '
        pass

    def predict_one_val(self, x_t: np.array, relev_ratio_threshold: float=1) -> Tuple[float, float]:
        if False:
            while True:
                i = 10
        '\n        Predict one value using fitted model from a subsample chosen by the ratio of relevance.\n\n        For example, if relev_ratio_threshold = 0.4, then it chooses the top 40 percentile data ranked by relevance to\n        x_t. This method also returns the associated prediction standard deviations.\n\n        y_t := y_avg_sub + 1/(n-1) * sum{relevance(x_i, x_t) * (y_i - y_avg_sub), subsample}\n        where y_i, x_i are from subsamples. The equivalent matrix form is:\n        y_t := y_avg_sub + 1/(n-1) * (x_t - x_avg).T @ fisher_info_mtx @ (X_sub - x_avg).T @ (y_sub - y_avg_sub)\n\n        :param x_t: (np.array) A single row element test data, 1D (k, 1). k is the number of features.\n        :param relev_ratio_threshold: (float) Optional. The subsample ratio to use for predicting values ranked by\n            relevance, must be a number between [0, 1]. For example, 0.6 corresponds to the top 60 percentile data\n            ranked by relevance to x_t. Defaults to 1.\n        :return: (Tuple[float, float]) The predicted result and associated standard deviation.\n        '
        pass

    def find_subsample(self, x_t: np.array, relev_ratio_threshold: float=1, above: bool=True) -> Tuple[np.array, np.array, np.array, float]:
        if False:
            i = 10
            return i + 15
        '\n        Find the subsamples of X and Y in the training set by relevance above or below a given threshold with x_t.\n\n        For example, if relev_ratio_threshold=0.3, above=True, then it finds the top 30 percentile.\n        If relev_ratio_threshold=0.3, above=False, then it finds the bottom 70 percentile.\n\n        The standard deviation is calculated as the sqrt of the variance of y_t hat, the prediction w.r.t. x_t:\n        var_yt_hat = [(n-1)/n^2 * var_y] + [1/n * y_mean^2] + [var_y/n + y_mean^2/(n-1)]*var_r, where\n        var_y is the subsample variance of Y, y_mean is the subsample average of Y, var_r is the subsample variance of\n        relevance.\n\n        :param x_t: (np.array) A single row element test data, 1D (k, 1). k is the number of features.\n        :param relev_ratio_threshold: (float) Optional. The subsample ratio to use for predicting values ranked by\n            relevance, must be a number between [0, 1].\n        :param above: (bool) Optional. Whether to find the subsample above the threshold or below the threshold.\n        :return: (Tuple[np.array, np.array, np.array, float]) The subsample for X, for Y, the corresponding\n            indices to select the subsample and the std.\n        '
        pass

    @staticmethod
    def _calc_cov_and_fisher(X: np.array, check_condi_num: bool=False) -> Tuple[np.array, np.array]:
        if False:
            for i in range(10):
                print('nop')
        '\n        Find the (non-biased) covariance matrix and its inverse (fisher info matrix).\n\n        i.e., cov = X.T @ X / (n-1), fisher_info_mtx = (n-1) inv(X.T @ X)\n\n        :param X: (np.array) The 2D (n-by-k) independent data vector, n: num of instances, k: num of variables\n            or features.\n        :param check_condi_num: (bool) Optional. Whether to check the condition number of the covariance matrix and\n            fisher info matrix from the training data X (Their values are the same). If this number is too large then it\n            may lead to numerical issues. Defaults to False.\n        :return: (Tuple[np.array, np.array]) The covariance matrix and its inverse.\n        '
        pass

    def calc_relevance(self, x_i: np.array, x_j: np.array, fisher_info_mtx: np.array=None) -> float:
        if False:
            return 10
        '\n        Calculate relevance of x_i and x_j: r(x_i, x_j).\n\n        r(x_i, x_j) := sim(x_i, x_j) + info(x_i) + info(x_j)\n\n        :param x_i: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.\n        :param x_j: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.\n        :param fisher_info_mtx: (np.array) Optional. 2D (k, k) matrix for the whole training data. Defaults to the\n            fisher info matrix stored in the class calculated using training data.\n        :return: (float) The relevance value.\n        '
        pass

    def calc_sim(self, x_i: np.array, x_j: np.array, fisher_info_mtx: np.array=None) -> float:
        if False:
            while True:
                i = 10
        '\n        Calculate the similarity of x_i and x_j: sim(x_i, x_j)\n\n        sim(x_i, x_j) := -1/2 * (x_i - x_j).T @ fisher_info @ (x_i - x_j)\n\n        :param x_i: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.\n        :param x_j: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.\n        :param fisher_info_mtx: (np.array) Optional. 2D (k, k) matrix for the whole training data. Defaults to the\n            fisher info matrix stored in the class calculated using training data.\n        :return: (float) The value of similarity.\n        '
        pass

    def calc_info(self, x_i: np.array, fisher_info_mtx: np.array=None) -> float:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calculate the informativeness of x_i: info(x_i)\n\n        info(x_i) := 1/2 * (x_i - x_avg).T @ fisher_info @ (x_i - x_avg)\n        Here x_avg is the training data average for each column.\n\n        :param x_i: (np.array) 1D (k, ) dependent data vector for an instance where k is the number of features.\n        :param fisher_info_mtx: (np.array) Optional. 2D (k, k) matrix for the whole training data. Defaults to the\n            fisher info matrix stored in the class calculated using training data.\n        :return: (float) The informativeness value.\n        '
        pass