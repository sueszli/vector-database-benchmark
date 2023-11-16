from typing import Dict, Callable
from ray.util.annotations import PublicAPI
from ray.tune.stopper.stopper import Stopper

@PublicAPI
class FunctionStopper(Stopper):
    """Provide a custom function to check if trial should be stopped.

    The passed function will be called after each iteration. If it returns
    True, the trial will be stopped.

    Args:
        function: Function that checks if a trial
            should be stopped. Must accept the `trial_id` string  and `result`
            dictionary as arguments. Must return a boolean.

    """

    def __init__(self, function: Callable[[str, Dict], bool]):
        if False:
            for i in range(10):
                print('nop')
        self._fn = function

    def __call__(self, trial_id, result):
        if False:
            while True:
                i = 10
        return self._fn(trial_id, result)

    def stop_all(self):
        if False:
            i = 10
            return i + 15
        return False

    @classmethod
    def is_valid_function(cls, fn):
        if False:
            i = 10
            return i + 15
        is_function = callable(fn) and (not issubclass(type(fn), Stopper))
        if is_function and hasattr(fn, 'stop_all'):
            raise ValueError('Stop object must be ray.tune.Stopper subclass to be detected correctly.')
        return is_function