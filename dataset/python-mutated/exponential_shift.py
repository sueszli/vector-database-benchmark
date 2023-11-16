from __future__ import division
import numpy
from chainer.training import extension

class ExponentialShift(extension.Extension):
    """Trainer extension to exponentially shift an optimizer attribute.

    This extension exponentially increases or decreases the specified attribute
    of the optimizer. The typical use case is an exponential decay of the
    learning rate.

    This extension is also called before the training loop starts by default.

    Args:
        attr (str): Name of the attribute to shift.
        rate (float): Rate of the exponential shift. This value is multiplied
            to the attribute at each call.
        init (float): Initial value of the attribute. If it is ``None``, the
            extension extracts the attribute at the first call and uses it as
            the initial value.
        target (float): Target value of the attribute. If the attribute reaches
            this value, the shift stops.
        optimizer (~chainer.Optimizer): Target optimizer to adjust the
            attribute. If it is ``None``, the main optimizer of the updater is
            used.

    """

    def __init__(self, attr, rate, init=None, target=None, optimizer=None):
        if False:
            while True:
                i = 10
        self._attr = attr
        if rate < 0:
            raise ValueError('ExponentialShift does not support negative rate')
        self._rate = rate
        self._init = init
        self._target = target
        self._optimizer = optimizer
        self._t = 0
        self._last_value = None

    def initialize(self, trainer):
        if False:
            for i in range(10):
                print('nop')
        optimizer = self._get_optimizer(trainer)
        if self._init is None:
            self._init = getattr(optimizer, self._attr)
        if self._last_value is not None:
            self._update_value(optimizer, self._last_value)
        else:
            self._update_value(optimizer, self._init)

    def __call__(self, trainer):
        if False:
            while True:
                i = 10
        self._t += 1
        optimizer = self._get_optimizer(trainer)
        value = self._init * self._rate ** self._t
        if self._target is not None:
            if self._rate > 1:
                if value / self._target > 1:
                    value = self._target
            elif value / self._target < 1:
                value = self._target
        self._update_value(optimizer, value)

    def serialize(self, serializer):
        if False:
            while True:
                i = 10
        self._t = serializer('_t', self._t)
        self._last_value = serializer('_last_value', self._last_value)
        if isinstance(self._last_value, numpy.ndarray):
            self._last_value = self._last_value.item()

    def _get_optimizer(self, trainer):
        if False:
            print('Hello World!')
        return self._optimizer or trainer.updater.get_optimizer('main')

    def _update_value(self, optimizer, value):
        if False:
            i = 10
            return i + 15
        setattr(optimizer, self._attr, value)
        self._last_value = value