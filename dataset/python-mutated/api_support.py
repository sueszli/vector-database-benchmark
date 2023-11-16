import zipline.api
from zipline.utils.compat import wraps
from zipline.utils.algo_instance import get_algo_instance, set_algo_instance

class ZiplineAPI(object):
    """
    Context manager for making an algorithm instance available to zipline API
    functions within a scoped block.
    """

    def __init__(self, algo_instance):
        if False:
            while True:
                i = 10
        self.algo_instance = algo_instance

    def __enter__(self):
        if False:
            return 10
        '\n        Set the given algo instance, storing any previously-existing instance.\n        '
        self.old_algo_instance = get_algo_instance()
        set_algo_instance(self.algo_instance)

    def __exit__(self, _type, _value, _tb):
        if False:
            print('Hello World!')
        '\n        Restore the algo instance stored in __enter__.\n        '
        set_algo_instance(self.old_algo_instance)

def api_method(f):
    if False:
        return 10

    @wraps(f)
    def wrapped(*args, **kwargs):
        if False:
            print('Hello World!')
        algo_instance = get_algo_instance()
        if algo_instance is None:
            raise RuntimeError('zipline api method %s must be called during a simulation.' % f.__name__)
        return getattr(algo_instance, f.__name__)(*args, **kwargs)
    setattr(zipline.api, f.__name__, wrapped)
    zipline.api.__all__.append(f.__name__)
    f.is_api_method = True
    return f

def require_not_initialized(exception):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator for API methods that should only be called during or before\n    TradingAlgorithm.initialize.  `exception` will be raised if the method is\n    called after initialize.\n\n    Examples\n    --------\n    @require_not_initialized(SomeException("Don\'t do that!"))\n    def method(self):\n        # Do stuff that should only be allowed during initialize.\n    '

    def decorator(method):
        if False:
            for i in range(10):
                print('nop')

        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            if self.initialized:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator

def require_initialized(exception):
    if False:
        i = 10
        return i + 15
    '\n    Decorator for API methods that should only be called after\n    TradingAlgorithm.initialize.  `exception` will be raised if the method is\n    called before initialize has completed.\n\n    Examples\n    --------\n    @require_initialized(SomeException("Don\'t do that!"))\n    def method(self):\n        # Do stuff that should only be allowed after initialize.\n    '

    def decorator(method):
        if False:
            print('Hello World!')

        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if False:
                while True:
                    i = 10
            if not self.initialized:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator

def disallowed_in_before_trading_start(exception):
    if False:
        for i in range(10):
            print('nop')
    '\n    Decorator for API methods that cannot be called from within\n    TradingAlgorithm.before_trading_start.  `exception` will be raised if the\n    method is called inside `before_trading_start`.\n\n    Examples\n    --------\n    @disallowed_in_before_trading_start(SomeException("Don\'t do that!"))\n    def method(self):\n        # Do stuff that is not allowed inside before_trading_start.\n    '

    def decorator(method):
        if False:
            for i in range(10):
                print('nop')

        @wraps(method)
        def wrapped_method(self, *args, **kwargs):
            if False:
                print('Hello World!')
            if self._in_before_trading_start:
                raise exception
            return method(self, *args, **kwargs)
        return wrapped_method
    return decorator