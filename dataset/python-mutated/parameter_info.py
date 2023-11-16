from caffe2.python import core
import numpy as np

class ParameterTags:
    BIAS = 'BIAS'
    WEIGHT = 'WEIGHT'
    COMPUTED_PARAM = 'COMPUTED_PARAM'

class ParameterInfo:

    def __init__(self, param_id, param, key=None, shape=None, length=None, grad=None, blob_copy=None):
        if False:
            for i in range(10):
                print('nop')
        assert isinstance(param, core.BlobReference)
        self.param_id = param_id
        self.name = str(param)
        self.blob = param
        self.key = key
        self.shape = shape
        self.size = None if shape is None else np.prod(shape)
        self.length = max(1, length if length is not None else 1)
        self.grad = grad
        self._cloned_init_net = None
        self.blob_copy = blob_copy
        self._optimizer = None

    @property
    def parameter(self):
        if False:
            while True:
                i = 10
        return self.blob

    @property
    def optimizer(self):
        if False:
            return 10
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        if False:
            i = 10
            return i + 15
        assert self._optimizer is None, 'optimizer has already been set'
        self._optimizer = value

    def __str__(self):
        if False:
            print('Hello World!')
        return self.name