import enum
from .xgb_cost_model import XgbCostModel

class CostModelType(enum.Enum):
    XGB = 1

class CostModel:
    """
    A base class to call different cost model algorithm.
    """

    def __init__(self, model_type=CostModelType.XGB):
        if False:
            print('Hello World!')
        '\n        Constructor\n        '
        self.model = None
        if model_type == CostModelType.XGB:
            self.model = XgbCostModel()
        else:
            raise ValueError('Illegal CostModelType')

    def train(self, samples, labels):
        if False:
            i = 10
            return i + 15
        '\n        Train the model.\n\n        Args:\n            samples(list|numpy): an array of numpy array representing a batch\n                of input samples.\n            labels(list|numpy): an array of float representing a batch of labels\n        '
        return self.model.train(samples, labels)

    def predict(self, samples):
        if False:
            return 10
        '\n        Predict\n\n        Args:\n            samples(list|numpy): an array of numpy array representing a batch\n                of input samples.\n        Returns:\n            np.array representing labels\n        '
        return self.model.predict(samples)

    def save(self, path):
        if False:
            print('Hello World!')
        '\n        Save the trained model.\n\n        Args:\n            path(str): path to save\n        '
        return self.model.save(path)

    def load(self, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Load the model\n\n        Args:\n            path(str): path to load\n        '
        return self.model.load(path)

    def update(self, samples, labels):
        if False:
            i = 10
            return i + 15
        pass