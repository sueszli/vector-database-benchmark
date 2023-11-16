from functools import update_wrapper, wraps
from types import MethodType

class _AvailableIfDescriptor:
    """Implements a conditional property using the descriptor protocol.

    Using this class to create a decorator will raise an ``AttributeError``
    if check(self) returns a falsey value. Note that if check raises an error
    this will also result in hasattr returning false.

    See https://docs.python.org/3/howto/descriptor.html for an explanation of
    descriptors.
    """

    def __init__(self, fn, check, attribute_name):
        if False:
            i = 10
            return i + 15
        self.fn = fn
        self.check = check
        self.attribute_name = attribute_name
        update_wrapper(self, fn)

    def __get__(self, obj, owner=None):
        if False:
            i = 10
            return i + 15
        attr_err = AttributeError(f'This {repr(owner.__name__)} has no attribute {repr(self.attribute_name)}')
        if obj is not None:
            if not self.check(obj):
                raise attr_err
            out = MethodType(self.fn, obj)
        else:

            @wraps(self.fn)
            def out(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                if not self.check(args[0]):
                    raise attr_err
                return self.fn(*args, **kwargs)
        return out

def available_if(check):
    if False:
        for i in range(10):
            print('nop')
    'An attribute that is available only if check returns a truthy value.\n\n    Parameters\n    ----------\n    check : callable\n        When passed the object with the decorated method, this should return\n        a truthy value if the attribute is available, and either return False\n        or raise an AttributeError if not available.\n\n    Returns\n    -------\n    callable\n        Callable makes the decorated method available if `check` returns\n        a truthy value, otherwise the decorated method is unavailable.\n\n    Examples\n    --------\n    >>> from sklearn.utils.metaestimators import available_if\n    >>> class HelloIfEven:\n    ...    def __init__(self, x):\n    ...        self.x = x\n    ...\n    ...    def _x_is_even(self):\n    ...        return self.x % 2 == 0\n    ...\n    ...    @available_if(_x_is_even)\n    ...    def say_hello(self):\n    ...        print("Hello")\n    ...\n    >>> obj = HelloIfEven(1)\n    >>> hasattr(obj, "say_hello")\n    False\n    >>> obj.x = 2\n    >>> hasattr(obj, "say_hello")\n    True\n    >>> obj.say_hello()\n    Hello\n    '
    return lambda fn: _AvailableIfDescriptor(fn, check, attribute_name=fn.__name__)