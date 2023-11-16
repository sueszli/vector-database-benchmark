from functools import wraps
import inspect
import warnings

class ManticoreDeprecationWarning(DeprecationWarning):
    """The deprecation warning class used by Manticore."""
    pass
warnings.simplefilter('default', category=ManticoreDeprecationWarning)

def deprecated(message: str):
    if False:
        i = 10
        return i + 15
    'A decorator for marking functions as deprecated.'
    assert isinstance(message, str), 'The deprecated decorator requires a message string argument.'

    def decorator(func):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            warnings.warn(f'`{func.__qualname__}` is deprecated. {message}', category=ManticoreDeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator