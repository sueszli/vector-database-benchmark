import random
import pandas as pd
from ...data import D
from ..model.base import Model

class ScoreFileModel(Model):
    """
    This model will load a score file, and return score at date exists in score file.
    """

    def __init__(self, score_path):
        if False:
            i = 10
            return i + 15
        pred_test = pd.read_csv(score_path, index_col=[0, 1], parse_dates=True, infer_datetime_format=True)
        self.pred = pred_test

    def get_data_with_date(self, date, **kwargs):
        if False:
            while True:
                i = 10
        score = self.pred.loc(axis=0)[:, date]
        score_series = score.reset_index(level='datetime', drop=True)['score']
        return score_series

    def predict(self, x_test, **kwargs):
        if False:
            print('Hello World!')
        return x_test

    def score(self, x_test, **kwargs):
        if False:
            while True:
                i = 10
        return

    def fit(self, x_train, y_train, x_valid, y_valid, w_train=None, w_valid=None, **kwargs):
        if False:
            print('Hello World!')
        return

    def save(self, fname, **kwargs):
        if False:
            print('Hello World!')
        return