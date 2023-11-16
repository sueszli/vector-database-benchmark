from collections.abc import Hashable
from datetime import datetime, timedelta
import time
import threading
from contextlib import suppress
from .decorators import decorator, wraps, get_argnames, arggetter, contextmanager
__all__ = ['raiser', 'ignore', 'silent', 'suppress', 'nullcontext', 'reraise', 'retry', 'fallback', 'limit_error_rate', 'ErrorRateExceeded', 'throttle', 'post_processing', 'collecting', 'joining', 'once', 'once_per', 'once_per_args', 'wrap_with']

def raiser(exception_or_class=Exception, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Constructs function that raises the given exception\n       with given arguments on any invocation.'
    if isinstance(exception_or_class, str):
        exception_or_class = Exception(exception_or_class)

    def _raiser(*a, **kw):
        if False:
            i = 10
            return i + 15
        if args or kwargs:
            raise exception_or_class(*args, **kwargs)
        else:
            raise exception_or_class
    return _raiser

def ignore(errors, default=None):
    if False:
        while True:
            i = 10
    'Alters function to ignore given errors, returning default instead.'
    errors = _ensure_exceptable(errors)

    def decorator(func):
        if False:
            while True:
                i = 10

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                print('Hello World!')
            try:
                return func(*args, **kwargs)
            except errors:
                return default
        return wrapper
    return decorator

def silent(func):
    if False:
        print('Hello World!')
    'Alters function to ignore all exceptions.'
    return ignore(Exception)(func)
try:
    from contextlib import nullcontext
except ImportError:

    class nullcontext(object):
        """Context manager that does no additional processing.

        Used as a stand-in for a normal context manager, when a particular
        block of code is only sometimes used with a normal context manager:

        cm = optional_cm if condition else nullcontext()
        with cm:
            # Perform operation, using optional_cm if condition is True
        """

        def __init__(self, enter_result=None):
            if False:
                while True:
                    i = 10
            self.enter_result = enter_result

        def __enter__(self):
            if False:
                print('Hello World!')
            return self.enter_result

        def __exit__(self, *excinfo):
            if False:
                i = 10
                return i + 15
            pass

@contextmanager
def reraise(errors, into):
    if False:
        return 10
    'Reraises errors as other exception.'
    errors = _ensure_exceptable(errors)
    try:
        yield
    except errors as e:
        if callable(into) and (not _is_exception_type(into)):
            into = into(e)
        raise into from e

@decorator
def retry(call, tries, errors=Exception, timeout=0, filter_errors=None):
    if False:
        while True:
            i = 10
    'Makes decorated function retry up to tries times.\n       Retries only on specified errors.\n       Sleeps timeout or timeout(attempt) seconds between tries.'
    errors = _ensure_exceptable(errors)
    for attempt in range(tries):
        try:
            return call()
        except errors as e:
            if not (filter_errors is None or filter_errors(e)):
                raise
            if attempt + 1 == tries:
                raise
            else:
                timeout_value = timeout(attempt) if callable(timeout) else timeout
                if timeout_value > 0:
                    time.sleep(timeout_value)

def fallback(*approaches):
    if False:
        return 10
    'Tries several approaches until one works.\n       Each approach has a form of (callable, expected_errors).'
    for approach in approaches:
        (func, catch) = (approach, Exception) if callable(approach) else approach
        catch = _ensure_exceptable(catch)
        try:
            return func()
        except catch:
            pass

def _ensure_exceptable(errors):
    if False:
        return 10
    'Ensures that errors are passable to except clause.\n       I.e. should be BaseException subclass or a tuple.'
    return errors if _is_exception_type(errors) else tuple(errors)

def _is_exception_type(value):
    if False:
        print('Hello World!')
    return isinstance(value, type) and issubclass(value, BaseException)

class ErrorRateExceeded(Exception):
    pass

def limit_error_rate(fails, timeout, exception=ErrorRateExceeded):
    if False:
        return 10
    'If function fails to complete fails times in a row,\n       calls to it will be intercepted for timeout with exception raised instead.'
    if isinstance(timeout, int):
        timeout = timedelta(seconds=timeout)

    def decorator(func):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                while True:
                    i = 10
            if wrapper.blocked:
                if datetime.now() - wrapper.blocked < timeout:
                    raise exception
                else:
                    wrapper.blocked = None
            try:
                result = func(*args, **kwargs)
            except:
                wrapper.fails += 1
                if wrapper.fails >= fails:
                    wrapper.blocked = datetime.now()
                raise
            else:
                wrapper.fails = 0
                return result
        wrapper.fails = 0
        wrapper.blocked = None
        return wrapper
    return decorator

def throttle(period):
    if False:
        return 10
    'Allows only one run in a period, the rest is skipped'
    if isinstance(period, timedelta):
        period = period.total_seconds()

    def decorator(func):
        if False:
            i = 10
            return i + 15

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            now = time.time()
            if wrapper.blocked_until and wrapper.blocked_until > now:
                return
            wrapper.blocked_until = now + period
            return func(*args, **kwargs)
        wrapper.blocked_until = None
        return wrapper
    return decorator

@decorator
def post_processing(call, func):
    if False:
        i = 10
        return i + 15
    'Post processes decorated function result with func.'
    return func(call())
collecting = post_processing(list)
collecting.__name__ = 'collecting'
collecting.__doc__ = 'Transforms a generator into list returning function.'

@decorator
def joining(call, sep):
    if False:
        print('Hello World!')
    'Joins decorated function results with sep.'
    return sep.join(map(sep.__class__, call()))

def once_per(*argnames):
    if False:
        i = 10
        return i + 15
    'Call function only once for every combination of the given arguments.'

    def once(func):
        if False:
            return 10
        lock = threading.Lock()
        done_set = set()
        done_list = list()
        get_arg = arggetter(func)

        @wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            with lock:
                values = tuple((get_arg(name, args, kwargs) for name in argnames))
                if isinstance(values, Hashable):
                    (done, add) = (done_set, done_set.add)
                else:
                    (done, add) = (done_list, done_list.append)
                if values not in done:
                    add(values)
                    return func(*args, **kwargs)
        return wrapper
    return once
once = once_per()
once.__doc__ = 'Let function execute once, noop all subsequent calls.'

def once_per_args(func):
    if False:
        for i in range(10):
            print('nop')
    'Call function once for every combination of values of its arguments.'
    return once_per(*get_argnames(func))(func)

@decorator
def wrap_with(call, ctx):
    if False:
        return 10
    'Turn context manager into a decorator'
    with ctx:
        return call()