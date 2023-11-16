"""
Defines CPU Options for use in the CPU target
"""
from abc import ABCMeta, abstractmethod

class AbstractOptionValue(metaclass=ABCMeta):
    """Abstract base class for custom option values.
    """

    @abstractmethod
    def encode(self) -> str:
        if False:
            i = 10
            return i + 15
        'Returns an encoding of the values\n        '
        ...

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'{self.__class__.__name__}({self.encode()})'

class FastMathOptions(AbstractOptionValue):
    """
    Options for controlling fast math optimization.
    """

    def __init__(self, value):
        if False:
            for i in range(10):
                print('nop')
        valid_flags = {'fast', 'nnan', 'ninf', 'nsz', 'arcp', 'contract', 'afn', 'reassoc'}
        if isinstance(value, FastMathOptions):
            self.flags = value.flags.copy()
        elif value is True:
            self.flags = {'fast'}
        elif value is False:
            self.flags = set()
        elif isinstance(value, set):
            invalid = value - valid_flags
            if invalid:
                raise ValueError('Unrecognized fastmath flags: %s' % invalid)
            self.flags = value
        elif isinstance(value, dict):
            invalid = set(value.keys()) - valid_flags
            if invalid:
                raise ValueError('Unrecognized fastmath flags: %s' % invalid)
            self.flags = {v for (v, enable) in value.items() if enable}
        else:
            msg = 'Expected fastmath option(s) to be either a bool, dict or set'
            raise ValueError(msg)

    def __bool__(self):
        if False:
            i = 10
            return i + 15
        return bool(self.flags)
    __nonzero__ = __bool__

    def encode(self) -> str:
        if False:
            return 10
        return str(self.flags)

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if type(other) is type(self):
            return self.flags == other.flags
        return NotImplemented

class ParallelOptions(AbstractOptionValue):
    """
    Options for controlling auto parallelization.
    """
    __slots__ = ('enabled', 'comprehension', 'reduction', 'inplace_binop', 'setitem', 'numpy', 'stencil', 'fusion', 'prange')

    def __init__(self, value):
        if False:
            while True:
                i = 10
        if isinstance(value, bool):
            self.enabled = value
            self.comprehension = value
            self.reduction = value
            self.inplace_binop = value
            self.setitem = value
            self.numpy = value
            self.stencil = value
            self.fusion = value
            self.prange = value
        elif isinstance(value, dict):
            self.enabled = True
            self.comprehension = value.pop('comprehension', True)
            self.reduction = value.pop('reduction', True)
            self.inplace_binop = value.pop('inplace_binop', True)
            self.setitem = value.pop('setitem', True)
            self.numpy = value.pop('numpy', True)
            self.stencil = value.pop('stencil', True)
            self.fusion = value.pop('fusion', True)
            self.prange = value.pop('prange', True)
            if value:
                msg = 'Unrecognized parallel options: %s' % value.keys()
                raise NameError(msg)
        elif isinstance(value, ParallelOptions):
            self.enabled = value.enabled
            self.comprehension = value.comprehension
            self.reduction = value.reduction
            self.inplace_binop = value.inplace_binop
            self.setitem = value.setitem
            self.numpy = value.numpy
            self.stencil = value.stencil
            self.fusion = value.fusion
            self.prange = value.prange
        else:
            msg = 'Expect parallel option to be either a bool or a dict'
            raise ValueError(msg)

    def _get_values(self):
        if False:
            return 10
        'Get values as dictionary.\n        '
        return {k: getattr(self, k) for k in self.__slots__}

    def __eq__(self, other):
        if False:
            print('Hello World!')
        if type(other) is type(self):
            return self._get_values() == other._get_values()
        return NotImplemented

    def encode(self) -> str:
        if False:
            print('Hello World!')
        return ', '.join((f'{k}={v}' for (k, v) in self._get_values().items()))

class InlineOptions(AbstractOptionValue):
    """
    Options for controlling inlining
    """

    def __init__(self, value):
        if False:
            return 10
        ok = False
        if isinstance(value, str):
            if value in ('always', 'never'):
                ok = True
        else:
            ok = hasattr(value, '__call__')
        if ok:
            self._inline = value
        else:
            msg = "kwarg 'inline' must be one of the strings 'always' or 'never', or it can be a callable that returns True/False. Found value %s" % value
            raise ValueError(msg)

    @property
    def is_never_inline(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        True if never inline\n        '
        return self._inline == 'never'

    @property
    def is_always_inline(self):
        if False:
            print('Hello World!')
        '\n        True if always inline\n        '
        return self._inline == 'always'

    @property
    def has_cost_model(self):
        if False:
            i = 10
            return i + 15
        '\n        True if a cost model is provided\n        '
        return not (self.is_always_inline or self.is_never_inline)

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The raw value\n        '
        return self._inline

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        if type(other) is type(self):
            return self.value == other.value
        return NotImplemented

    def encode(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return repr(self._inline)