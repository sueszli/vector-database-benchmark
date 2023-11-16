import inspect
import logging
from enum import Enum
from paddle.base.log_helper import get_logger
_logger = get_logger(__name__, logging.INFO, fmt='%(asctime)s-%(levelname)s: %(message)s')

class FunctionType(Enum):
    FP16_ONLY = 0
    COMMON = 1

class Function:
    """
    Function is a wrap over standard python function
    An instance of this Function class is also callable
    just like the python function that it wrapped.
    When the instance is "called" like a function it fetches
    the function to be invoked from the virtual namespace and then
    invokes the same.
    """

    def __init__(self, fn):
        if False:
            i = 10
            return i + 15
        self.fn = fn

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Overriding the __call__ function which makes the\n        instance callable.\n        '
        fn = Namespace.get_instance().get(*args, **kwargs)
        return fn(*args, **kwargs)

class Namespace:
    """
    Namespace is the singleton class that is responsible
    for holding all the functions.
    """
    __instance = None

    def __init__(self):
        if False:
            while True:
                i = 10
        if self.__instance is None:
            self.function_map = {}
            Namespace.__instance = self
        else:
            raise Exception('cannot instantiate Namespace again.')

    @staticmethod
    def get_instance():
        if False:
            print('Hello World!')
        if Namespace.__instance is None:
            Namespace()
        return Namespace.__instance

    def register(self, fn, key):
        if False:
            return 10
        '\n        Register the function in the virtual namespace and return\n        an instance of callable Function that wraps the function fn.\n\n        Args:\n            fn (function): the native python function handle.\n            key (FunctionType): the specified type.\n        '
        assert isinstance(key, FunctionType), f'The type of  key is expected to be FunctionType, but recieved {type(key)}.'
        func = Function(fn)
        self.function_map[key] = fn
        return func

    def get(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the matching function from the virtual namespace according to the actual arguments.\n        Return None if it did not find any matching function.\n        '
        _logger.debug(f'get function: args={args}, kwargs={kwargs}')
        satisfied_function_keys = set(self.function_map.keys())
        num_actual_args = len(args) + len(kwargs)
        for func_key in self.function_map.keys():
            if func_key not in satisfied_function_keys:
                continue
            fn = self.function_map[func_key]
            specs = inspect.getfullargspec(fn)
            if len(specs) < len(args) + len(kwargs):
                _logger.debug(f'fn={fn} (key={func_key}) is not satisfied and removed.')
                satisfied_function_keys.remove(func_key)
                continue
            if len(kwargs) > 0:
                for (arg_name, value) in kwargs.items():
                    if arg_name not in specs.args:
                        _logger.debug(f'fn={fn} (key={func_key}) is not satisfied and removed.')
                        satisfied_function_keys.remove(func_key)
                        break
        if len(satisfied_function_keys) == 1:
            key = list(satisfied_function_keys)[0]
        elif len(args) >= 3 and isinstance(args[2], float):
            key = FunctionType.FP16_ONLY
        else:
            key = FunctionType.COMMON
        return self.function_map.get(key)

def overload(key):
    if False:
        print('Hello World!')
    'overload is the decorator that wraps the function\n    and returns a callable object of type Function.\n    '

    def decorator(fn):
        if False:
            for i in range(10):
                print('nop')
        return Namespace.get_instance().register(fn, key)
    return decorator