"""
Neural Network optimizer utilities.
"""

class AdamParams(object):
    """
    Adam - A Method for Stochastic Optimization.

    Attributes
    ----------
    lr: float
        The learning rate that controls learning step size. Adjustable in progress, default: 0.01.
    batch: int
        The mini-batch size, number of examples used to compute single gradient step, default: 10.
    beta1: float
        Controls the exponential decay rate for the first moment estimates, default: 0.9.
    beta2: float
        Controls the exponential decay rate for the second moment estimates, default: 0.999.
    eps: float
        The epsilon, a very small number to prevent any division by zero in the implementation, default: 1e-8.

    Methods
    -------
    set_lr(value, min, max)
        Set value for learning rate.
    set_batch(value, allow_set)
        Set value for batch size.
    set_beta1(value, min, max)
        Set value for beta1.
    set_beta2(value, min, max)
        Set value for beta2.
    set_eps(value, min, max)
        Set value for epsilon.
    """

    def __init__(self, lr=0.01, batch=10, beta1=0.9, beta2=0.999, eps=1e-08):
        if False:
            print('Hello World!')
        self._lr = RangeParam(lr)
        self._batch = Batch(batch)
        self._beta1 = RangeParam(beta1)
        self._beta2 = RangeParam(beta2)
        self._eps = RangeParam(eps)

    def set_lr(self, value, min, max):
        if False:
            return 10
        self._lr = RangeParam(value, min, max)

    def set_batch(self, value, allowed_set):
        if False:
            print('Hello World!')
        self._batch = Batch(value, allowed_set)

    def set_beta1(self, value, min, max):
        if False:
            i = 10
            return i + 15
        self._beta1 = RangeParam(value, min, max)

    def set_beta2(self, value, min, max):
        if False:
            print('Hello World!')
        self._beta2 = RangeParam(value, min, max)

    def set_eps(self, value, min, max):
        if False:
            while True:
                i = 10
        self._eps = RangeParam(value, min, max)

    @property
    def lr(self):
        if False:
            while True:
                i = 10
        return self._lr

    @property
    def batch(self):
        if False:
            i = 10
            return i + 15
        return self._batch

    @property
    def beta1(self):
        if False:
            while True:
                i = 10
        return self._beta1

    @property
    def beta2(self):
        if False:
            print('Hello World!')
        return self._beta2

    @property
    def eps(self):
        if False:
            for i in range(10):
                print('nop')
        return self._eps

class SgdParams(object):
    """
    SGD - Stochastic Gradient Descent optimizer.

    Attributes
    ----------
    lr: float
        The learning rate that controls learning step size. Adjustable in progress, default: 0.01.
    batch: int
        The mini-batch size, number of examples used to compute single gradient step, default: 10.
    momentum: float
        The momentum factor that helps accelerate gradients vectors in the right direction, default 0.

    Methods
    -------
    set_lr(value, min, max)
        Set value for learning rate.
    set_batch(value, allow_set)
        Set value for batch size.
    set_momentum(value, min, max)
        Set value for momentum.
    """

    def __init__(self, lr=0.01, batch=10, momentum=0):
        if False:
            for i in range(10):
                print('nop')
        self._lr = RangeParam(lr)
        self._batch = Batch(batch)
        self._momentum = RangeParam(momentum)

    def set_lr(self, value, min, max):
        if False:
            print('Hello World!')
        self._lr = RangeParam(value, min, max)

    def set_batch(self, value, allowed_set):
        if False:
            for i in range(10):
                print('nop')
        self._batch = Batch(value, allowed_set)

    def set_momentum(self, value, min, max):
        if False:
            i = 10
            return i + 15
        self._momentum = RangeParam(value, min, max)

    @property
    def lr(self):
        if False:
            i = 10
            return i + 15
        return self._lr

    @property
    def batch(self):
        if False:
            return 10
        return self._batch

    @property
    def momentum(self):
        if False:
            for i in range(10):
                print('nop')
        return self._momentum

class RangeParam:

    def __init__(self, value, min=0, max=1):
        if False:
            while True:
                i = 10
        self._value = value
        if min >= max:
            raise ValueError('min value must be less than max value.')
        self._min = min
        self._max = max

    @property
    def value(self):
        if False:
            while True:
                i = 10
        return self._value

    @property
    def min(self):
        if False:
            while True:
                i = 10
        return self._min

    @property
    def max(self):
        if False:
            for i in range(10):
                print('nop')
        return self._max

class Batch:

    def __init__(self, value, allowed_set=None):
        if False:
            for i in range(10):
                print('nop')
        self._value = value
        if allowed_set is None:
            self._allowed_set = [value]
        else:
            if len(allowed_set) > len(set(allowed_set)):
                raise ValueError('values in allowed_set must be unique.')
            self._allowed_set = allowed_set

    @property
    def value(self):
        if False:
            print('Hello World!')
        return self._value

    @property
    def allowed_set(self):
        if False:
            for i in range(10):
                print('nop')
        return self._allowed_set