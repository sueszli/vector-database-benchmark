"""Reinforcement Learning (RL) tools Open Spiel."""
import abc

class ValueSchedule(metaclass=abc.ABCMeta):
    """Abstract base class changing (decaying) values."""

    @abc.abstractmethod
    def __init__(self):
        if False:
            while True:
                i = 10
        'Initialize the value schedule.'

    @abc.abstractmethod
    def step(self):
        if False:
            for i in range(10):
                print('nop')
        'Apply a potential change in the value.\n\n    This method should be called every time the agent takes a training step.\n\n    Returns:\n      the value after the step.\n    '

    @property
    @abc.abstractmethod
    def value(self):
        if False:
            while True:
                i = 10
        'Return the current value.'

class ConstantSchedule(ValueSchedule):
    """A schedule that keeps the value constant."""

    def __init__(self, value):
        if False:
            i = 10
            return i + 15
        super(ConstantSchedule, self).__init__()
        self._value = value

    def step(self):
        if False:
            for i in range(10):
                print('nop')
        return self._value

    @property
    def value(self):
        if False:
            print('Hello World!')
        return self._value

class LinearSchedule(ValueSchedule):
    """A simple linear schedule."""

    def __init__(self, init_val, final_val, num_steps):
        if False:
            print('Hello World!')
        'A simple linear schedule.\n\n    Once the the number of steps is reached, value is always equal to the final\n    value.\n\n    Arguments:\n      init_val: the initial value.\n      final_val: the final_value\n      num_steps: the number of steps to get from the initial to final value.\n    '
        super(LinearSchedule, self).__init__()
        self._value = init_val
        self._final_value = final_val
        assert isinstance(num_steps, int)
        self._num_steps = num_steps
        self._steps_taken = 0
        self._increment = (final_val - init_val) / num_steps

    def step(self):
        if False:
            for i in range(10):
                print('nop')
        self._steps_taken += 1
        if self._steps_taken < self._num_steps:
            self._value += self._increment
        elif self._steps_taken == self._num_steps:
            self._value = self._final_value
        return self._value

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        return self._value