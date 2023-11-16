import numpy as np
from scipy.linalg import sqrtm
from recommenders.utils.python_utils import binarize as conv_binary

class PlainScalarProduct(object):
    """
    Module that implements plain scalar product
    as the retrieval criterion
    """

    def __init__(self, X, Y, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Args:\n            X: numpy matrix of shape (users, features)\n            Y: numpy matrix of shape (items, features)\n        '
        self.X = X
        self.Y = Y

    def sim(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Calculate the similarity score'
        sim = self.X.dot(self.Y.T)
        return sim

class Inferer:
    """
    Holds necessary (minimal) information needed for inference
    """

    def __init__(self, method='dot', k=10, transformation=''):
        if False:
            for i in range(10):
                print('nop')
        "Initialize parameters\n\n        Args:\n            method (str): The inference method. Currently 'dot'\n                (Dot product) is supported.\n            k (uint): `k` for 'topk' transformation.\n            transformation (str): Transform the inferred values into a\n                different scale. Currently 'mean' (Binarize the values\n                using mean of inferred matrix as the threshold), 'topk'\n                (Pick Top-K inferred values per row and assign them 1,\n                setting rest of them to 0), '' (No transformation) are\n                supported.\n        "
        self.method = self._get_method(method)
        self.k = k
        self.transformation = transformation

    def _get_method(self, k):
        if False:
            while True:
                i = 10
        "Get the inferer method\n\n        Args:\n            k (str): The inferer name\n\n        Returns:\n            class: A class object implementing the inferer 'k'\n        "
        if k == 'dot':
            method = PlainScalarProduct
        else:
            raise ValueError(f'{k} is unknown.')
        return method

    def infer(self, dataPtr, W, **kwargs):
        if False:
            print('Hello World!')
        'Main inference method\n\n        Args:\n            dataPtr (DataPtr): An object containing the X, Z features needed for inference\n            W (iterable): An iterable containing the U, B, V parametrized matrices.\n        '
        if isinstance(dataPtr, list):
            a = dataPtr[0]
            b = dataPtr[1]
        else:
            a = dataPtr.get_entity('row').dot(W[0]).dot(sqrtm(W[1]))
            b = dataPtr.get_entity('col').dot(W[2]).dot(sqrtm(W[1]))
        sim_score = self.method(a, b).sim(**kwargs)
        if self.transformation == 'mean':
            prediction = conv_binary(sim_score, sim_score.mean())
        elif self.transformation == 'topk':
            masked_sim_score = sim_score.copy()
            for i in range(sim_score.shape[0]):
                topKidx = np.argpartition(masked_sim_score[i], -self.k)[-self.k:]
                mask = np.ones(sim_score[i].size, dtype=bool)
                mask[topKidx] = False
                masked_sim_score[i][topKidx] = 1
                masked_sim_score[i][mask] = 0
            prediction = masked_sim_score
        else:
            prediction = sim_score
        return prediction