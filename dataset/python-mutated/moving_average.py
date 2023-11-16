from typing import Iterable, Tuple, Optional, Any, Dict
import torch
from allennlp.common.registrable import Registrable
NamedParameter = Tuple[str, torch.Tensor]

class MovingAverage(Registrable):
    """
    Tracks a moving average of model parameters.
    """
    default_implementation = 'exponential'

    def __init__(self, parameters: Iterable[NamedParameter]) -> None:
        if False:
            while True:
                i = 10
        self._parameters = list(parameters)
        self._shadows = {name: parameter.data.clone() for (name, parameter) in self._parameters}
        self._backups = {name: parameter.data.clone() for (name, parameter) in self._parameters}

    def apply(self, num_updates: Optional[int]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the moving averages based on the latest values of the parameters.\n        '
        raise NotImplementedError

    def assign_average_value(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Replace all the parameter values with the averages.\n        Save the current parameter values to restore later.\n        '
        for (name, parameter) in self._parameters:
            self._backups[name].copy_(parameter.data)
            parameter.data.copy_(self._shadows[name])

    def restore(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore the backed-up (non-average) parameter values.\n        '
        for (name, parameter) in self._parameters:
            parameter.data.copy_(self._backups[name])

    def state_dict(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        return {'parameters': self._parameters, 'shadows': self._shadows, 'backups': self._backups}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if False:
            print('Hello World!')
        self._parameters = state_dict['parameters']
        self._shadows = state_dict['shadows']
        self._backups = state_dict['backups']

@MovingAverage.register('exponential')
class ExponentialMovingAverage(MovingAverage):
    """
    Create shadow variables and maintain exponential moving average for model parameters.

    Registered as a `MovingAverage` with name "exponential".

    # Parameters

    parameters : `Iterable[Tuple[str, Parameter]]`, required
        The parameters whose averages we'll be tracking. In a typical AllenNLP configuration
        file, this argument does not get an entry under the "moving_average", it gets passed
        in separately.
    decay : `float`, optional (default = `0.9999`)
        The decay rate that will be used if `num_updates` is not passed
        (and that will be used as an upper bound if `num_updates` is passed).
    numerator : `float`, optional (default = `1.0`)
        The numerator used to compute the decay rate if `num_updates` is passed.
    denominator : `float`, optional (default = `10.0`)
        The denominator used to compute the decay rate if `num_updates` is passed.
    """

    def __init__(self, parameters: Iterable[NamedParameter], decay: float=0.9999, numerator: float=1.0, denominator: float=10.0) -> None:
        if False:
            while True:
                i = 10
        super().__init__(parameters)
        self._decay = decay
        self._numerator = numerator
        self._denominator = denominator

    def apply(self, num_updates: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        '\n        Apply exponential moving average to `named_parameters` if specified,\n        or we will apply this to all the trainable parameters of the model.\n\n        The optional `num_updates` parameter allows one to tweak the decay rate\n        dynamically. If passed, the actual decay rate used is:\n\n            `min(decay, (numerator + num_updates) / (denominator + num_updates))`\n\n        (This logic is based on the Tensorflow exponential moving average\n         <https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage>)\n        '
        if num_updates is not None:
            decay = min(self._decay, (self._numerator + num_updates) / (self._denominator + num_updates))
        else:
            decay = self._decay
        for (name, parameter) in self._parameters:
            self._shadows[name].mul_(decay).add_((1 - decay) * parameter.data)