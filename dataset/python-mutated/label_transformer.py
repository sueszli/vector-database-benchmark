import numpy as np

class LabelTransformer(object):
    """
    Label encoder decoder

    Attributes
    ----------
    n_classes : int
        number of classes, K
    """

    def __init__(self, n_classes: int=None):
        if False:
            for i in range(10):
                print('nop')
        self.n_classes = n_classes

    @property
    def n_classes(self):
        if False:
            return 10
        return self.__n_classes

    @n_classes.setter
    def n_classes(self, K):
        if False:
            return 10
        self.__n_classes = K
        self.__encoder = None if K is None else np.eye(K)

    @property
    def encoder(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__encoder

    def encode(self, class_indices: np.ndarray):
        if False:
            return 10
        '\n        encode class index into one-of-k code\n\n        Parameters\n        ----------\n        class_indices : (N,) np.ndarray\n            non-negative class index\n            elements must be integer in [0, n_classes)\n\n        Returns\n        -------\n        (N, K) np.ndarray\n            one-of-k encoding of input\n        '
        if self.n_classes is None:
            self.n_classes = np.max(class_indices) + 1
        return self.encoder[class_indices]

    def decode(self, onehot: np.ndarray):
        if False:
            return 10
        '\n        decode one-of-k code into class index\n\n        Parameters\n        ----------\n        onehot : (N, K) np.ndarray\n            one-of-k code\n\n        Returns\n        -------\n        (N,) np.ndarray\n            class index\n        '
        return np.argmax(onehot, axis=1)