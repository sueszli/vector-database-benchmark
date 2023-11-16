"""
Implementation of an algorithm described in Yimou Li, David Turkington, Alireza Yazdani
'Beyond the Black Box: An Intuitive Approach to Investment Prediction with Machine Learning'
(https://jfds.pm-research.com/content/early/2019/12/11/jfds.2019.1.023)
"""
from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class AbstractModelFingerprint(ABC):
    """
    Model fingerprint constructor.

    This is an abstract base class for the RegressionModelFingerprint and ClassificationModelFingerprint classes.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Model fingerprint constructor.\n        '
        pass

    def fit(self, model: object, X: pd.DataFrame, num_values: int=50, pairwise_combinations: list=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get linear, non-linear and pairwise effects estimation.\n\n        :param model: (object) Trained model.\n        :param X: (pd.DataFrame) Dataframe of features.\n        :param num_values: (int) Number of values used to estimate feature effect.\n        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.\n        '
        pass

    def get_effects(self) -> Tuple:
        if False:
            i = 10
            return i + 15
        '\n        Return computed linear, non-linear and pairwise effects. The model should be fit() before using this method.\n\n        :return: (tuple) Linear, non-linear and pairwise effects, of type dictionary (raw values and normalised).\n        '
        pass

    def plot_effects(self) -> plt.figure:
        if False:
            print('Hello World!')
        '\n        Plot each effect (normalized) on a bar plot (linear, non-linear). Also plots pairwise effects if calculated.\n\n        :return: (plt.figure) Plot figure.\n        '
        pass

    def _get_feature_values(self, X: pd.DataFrame, num_values: int) -> None:
        if False:
            print('Hello World!')
        '\n        Step 1 of the algorithm which generates possible feature values used in analysis.\n\n        :param X: (pd.DataFrame) Dataframe of features.\n        :param num_values: (int) Number of values used to estimate feature effect.\n        '
        pass

    def _get_individual_partial_dependence(self, model: object, X: pd.DataFrame) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get individual partial dependence function values for each column.\n\n        :param model: (object) Trained model.\n        :param X: (pd.DataFrame) Dataframe of features.\n        '
        pass

    def _get_linear_effect(self, X: pd.DataFrame) -> dict:
        if False:
            return 10
        '\n        Get linear effect estimates as the mean absolute deviation of the linear predictions around their average value.\n\n        :param X: (pd.DataFrame) Dataframe of features.\n        :return: (dict) Linear effect estimates for each feature column.\n        '
        pass

    def _get_non_linear_effect(self, X: pd.DataFrame) -> dict:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get non-linear effect estimates as as the mean absolute deviation of the total marginal (single variable)\n        effect around its corresponding linear effect.\n\n        :param X: (pd.DataFrame) Dataframe of features.\n        :return: (dict) Non-linear effect estimates for each feature column.\n        '
        pass

    def _get_pairwise_effect(self, pairwise_combinations: list, model: object, X: pd.DataFrame, num_values) -> dict:
        if False:
            return 10
        '\n        Get pairwise effect estimates as the de-meaned joint partial prediction of the two variables minus the de-meaned\n        partial predictions of each variable independently.\n\n        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.\n        :param model: (object) Trained model.\n        :param X: (pd.DataFrame) Dataframe of features.\n        :param num_values: (int) Number of values used to estimate feature effect.\n        :return: (dict) Raw and normalised pairwise effects.\n        '
        pass

    @abstractmethod
    def _get_model_predictions(self, model: object, X_: pd.DataFrame):
        if False:
            while True:
                i = 10
        '\n        Get model predictions based on problem type (predict for regression, predict_proba for classification).\n\n        :param model: (object) Trained model.\n        :param X_: (np.array) Feature set.\n        :return: (np.array) Predictions.\n        '
        pass

    @staticmethod
    def _normalize(effect: dict) -> dict:
        if False:
            while True:
                i = 10
        '\n        Normalize effect values (sum equals 1).\n\n        :param effect: (dict) Effect values.\n        :return: (dict) Normalized effect values.\n        '
        pass

class RegressionModelFingerprint(AbstractModelFingerprint):
    """
    Regression Fingerprint class used for regression type of models.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Regression model fingerprint constructor.\n        '
        pass

    def _get_model_predictions(self, model, X_):
        if False:
            for i in range(10):
                print('nop')
        '\n        Abstract method _get_model_predictions implementation.\n\n        :param model: (object) Trained model.\n        :param X_: (np.array) Feature set.\n        :return: (np.array) Predictions.\n        '
        pass

class ClassificationModelFingerprint(AbstractModelFingerprint):
    """
    Classification Fingerprint class used for classification type of models.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Classification model fingerprint constructor.\n        '
        pass

    def _get_model_predictions(self, model, X_):
        if False:
            i = 10
            return i + 15
        '\n        Abstract method _get_model_predictions implementation.\n\n        :param model: (object) Trained model.\n        :param X_: (np.array) Feature set.\n        :return: (np.array) Predictions.\n        '
        pass