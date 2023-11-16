import numpy as np

class _MultiClass(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        pass

    def _one_hot(self, y, n_labels, dtype):
        if False:
            for i in range(10):
                print('nop')
        "Returns a matrix where each sample in y is represented\n           as a row, and each column represents the class label in\n           the one-hot encoding scheme.\n\n        Example:\n\n            y = np.array([0, 1, 2, 3, 4, 2])\n            mc = _BaseMultiClass()\n            mc._one_hot(y=y, n_labels=5, dtype='float')\n\n            np.array([[1., 0., 0., 0., 0.],\n                      [0., 1., 0., 0., 0.],\n                      [0., 0., 1., 0., 0.],\n                      [0., 0., 0., 1., 0.],\n                      [0., 0., 0., 0., 1.],\n                      [0., 0., 1., 0., 0.]])\n\n        "
        mat = np.zeros((len(y), n_labels))
        for (i, val) in enumerate(y):
            mat[i, val] = 1
        return mat.astype(dtype)