from functools import wraps
from .deprecated import deprecated

@deprecated('This function is deprecated')
def resolve_only_args(func):
    if False:
        i = 10
        return i + 15

    @wraps(func)
    def wrapped_func(root, info, **args):
        if False:
            while True:
                i = 10
        return func(root, **args)
    return wrapped_func