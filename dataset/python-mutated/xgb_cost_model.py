import numpy as np
import xgboost as xgb

class XgbCostModel:
    """
    A cost model implemented by XgbCostModel
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        Constructor\n        '
        self.booster = None
        self.xgb_param = {}
        self.train_round = 10

    def train(self, samples, labels):
        if False:
            return 10
        '\n        Train the model.\n\n        Args:\n            samples(list|numpy): an array of numpy array representing a batch\n                of input samples.\n            labels(list|numpy): an array of float representing a batch of labels\n\n        Returns:\n            xgb.Booster\n        '
        lengths = [x.shape[0] for x in samples]
        if isinstance(samples, list):
            samples = np.concatenate(samples, axis=0)
        if isinstance(labels, list):
            labels = np.concatenate([[y] * length for (y, length) in zip(labels, lengths)], axis=0)
        dmatrix = xgb.DMatrix(data=samples, label=labels)
        self.booster = xgb.train(self.xgb_param, dmatrix, self.train_round)
        return self.booster

    def predict(self, samples):
        if False:
            print('Hello World!')
        '\n        Predict\n\n        Args:\n            samples(list|numpy): an array of numpy array representing a batch\n                of input samples.\n        Returns:\n            np.array representing labels\n        '
        if isinstance(samples, list):
            samples = np.concatenate(samples, axis=0)
        dmatrix = xgb.DMatrix(data=samples, label=None)
        pred = self.booster.predict(dmatrix)
        return pred

    def save(self, path):
        if False:
            i = 10
            return i + 15
        '\n        Save the trained XgbCostModel\n\n        Args:\n            path(str): path to save\n        '
        assert self.booster is not None, 'Calling save on a XgbCostModel not been trained'
        self.booster.save_model(path)

    def load(self, path):
        if False:
            while True:
                i = 10
        '\n        Load the trained XgbCostModel\n\n        Args:\n            path(str): path to load\n        '
        if self.booster is None:
            self.booster = xgb.Booster()
        self.booster.load_model(path)

    def update(self, samples, labels):
        if False:
            while True:
                i = 10
        pass