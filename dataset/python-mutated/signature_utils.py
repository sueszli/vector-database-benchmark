import inspect
from typing import Callable, Optional

def is_param_in_hook_signature(hook_fx: Callable, param: str, explicit: bool=False, min_args: Optional[int]=None) -> bool:
    if False:
        i = 10
        return i + 15
    '\n    Args:\n        hook_fx: the hook callable\n        param: the name of the parameter to check\n        explicit: whether the parameter has to be explicitly declared\n        min_args: whether the `signature` has at least `min_args` parameters\n    '
    if hasattr(hook_fx, '__wrapped__'):
        hook_fx = hook_fx.__wrapped__
    parameters = inspect.getfullargspec(hook_fx)
    args = parameters.args[1:]
    return param in args or (not explicit and parameters.varargs is not None) or (isinstance(min_args, int) and len(args) >= min_args)