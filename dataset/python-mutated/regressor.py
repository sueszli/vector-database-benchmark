from __future__ import annotations
import abc
import typing
from river import base
from . import estimator
if typing.TYPE_CHECKING:
    import pandas as pd

class Regressor(estimator.Estimator):
    """A regressor."""

    @abc.abstractmethod
    def learn_one(self, x: dict, y: base.typing.RegTarget) -> Regressor:
        if False:
            while True:
                i = 10
        'Fits to a set of features `x` and a real-valued target `y`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n        y\n            A numeric target.\n\n        Returns\n        -------\n        self\n\n        '

    @abc.abstractmethod
    def predict_one(self, x: dict) -> base.typing.RegTarget:
        if False:
            while True:
                i = 10
        'Predict the output of features `x`.\n\n        Parameters\n        ----------\n        x\n            A dictionary of features.\n\n        Returns\n        -------\n        The prediction.\n\n        '

class MiniBatchRegressor(Regressor):
    """A regressor that can operate on mini-batches."""

    @abc.abstractmethod
    def learn_many(self, X: pd.DataFrame, y: pd.Series) -> MiniBatchRegressor:
        if False:
            return 10
        'Update the model with a mini-batch of features `X` and real-valued targets `y`.\n\n        Parameters\n        ----------\n        X\n            A dataframe of features.\n        y\n            A series of numbers.\n\n        Returns\n        -------\n        self\n\n        '

    @abc.abstractmethod
    def predict_many(self, X: pd.DataFrame) -> pd.Series:
        if False:
            while True:
                i = 10
        'Predict the outcome for each given sample.\n\n        Parameters\n        ----------\n        X\n            A dataframe of features.\n\n        Returns\n        -------\n        The predicted outcomes.\n\n        '