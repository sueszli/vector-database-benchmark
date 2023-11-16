import warnings
from robot.model import Tags

def deprecated(method):
    if False:
        print('Hello World!')

    def wrapper(self, *args, **kws):
        if False:
            for i in range(10):
                print('nop')
        'Deprecated.'
        warnings.warn(f"'robot.result.{type(self).__name__}.{method.__name__}' is deprecated and will be removed in Robot Framework 8.0.", stacklevel=1)
        return method(self, *args, **kws)
    return wrapper

class DeprecatedAttributesMixin:
    __slots__ = []
    _log_name = ''

    @property
    @deprecated
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._log_name

    @property
    @deprecated
    def kwname(self):
        if False:
            print('Hello World!')
        return self._log_name

    @property
    @deprecated
    def libname(self):
        if False:
            while True:
                i = 10
        return None

    @property
    @deprecated
    def args(self):
        if False:
            return 10
        return ()

    @property
    @deprecated
    def assign(self):
        if False:
            while True:
                i = 10
        return ()

    @property
    @deprecated
    def tags(self):
        if False:
            while True:
                i = 10
        return Tags()

    @property
    @deprecated
    def timeout(self):
        if False:
            while True:
                i = 10
        return None

    @property
    @deprecated
    def doc(self):
        if False:
            return 10
        return ''