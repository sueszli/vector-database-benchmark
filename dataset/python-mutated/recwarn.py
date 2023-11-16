"""Record warnings during test function execution."""
import re
import warnings
from pprint import pformat
from types import TracebackType
from typing import Any
from typing import Callable
from typing import final
from typing import Generator
from typing import Iterator
from typing import List
from typing import Optional
from typing import overload
from typing import Pattern
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import WARNS_NONE_ARG
from _pytest.fixtures import fixture
from _pytest.outcomes import fail
T = TypeVar('T')

@fixture
def recwarn() -> Generator['WarningsRecorder', None, None]:
    if False:
        while True:
            i = 10
    'Return a :class:`WarningsRecorder` instance that records all warnings emitted by test functions.\n\n    See https://docs.pytest.org/en/latest/how-to/capture-warnings.html for information\n    on warning categories.\n    '
    wrec = WarningsRecorder(_ispytest=True)
    with wrec:
        warnings.simplefilter('default')
        yield wrec

@overload
def deprecated_call(*, match: Optional[Union[str, Pattern[str]]]=...) -> 'WarningsRecorder':
    if False:
        print('Hello World!')
    ...

@overload
def deprecated_call(func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    if False:
        for i in range(10):
            print('nop')
    ...

def deprecated_call(func: Optional[Callable[..., Any]]=None, *args: Any, **kwargs: Any) -> Union['WarningsRecorder', Any]:
    if False:
        while True:
            i = 10
    "Assert that code produces a ``DeprecationWarning`` or ``PendingDeprecationWarning`` or ``FutureWarning``.\n\n    This function can be used as a context manager::\n\n        >>> import warnings\n        >>> def api_call_v2():\n        ...     warnings.warn('use v3 of this api', DeprecationWarning)\n        ...     return 200\n\n        >>> import pytest\n        >>> with pytest.deprecated_call():\n        ...    assert api_call_v2() == 200\n\n    It can also be used by passing a function and ``*args`` and ``**kwargs``,\n    in which case it will ensure calling ``func(*args, **kwargs)`` produces one of\n    the warnings types above. The return value is the return value of the function.\n\n    In the context manager form you may use the keyword argument ``match`` to assert\n    that the warning matches a text or regex.\n\n    The context manager produces a list of :class:`warnings.WarningMessage` objects,\n    one for each warning raised.\n    "
    __tracebackhide__ = True
    if func is not None:
        args = (func,) + args
    return warns((DeprecationWarning, PendingDeprecationWarning, FutureWarning), *args, **kwargs)

@overload
def warns(expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]]=..., *, match: Optional[Union[str, Pattern[str]]]=...) -> 'WarningsChecker':
    if False:
        i = 10
        return i + 15
    ...

@overload
def warns(expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]], func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
    if False:
        for i in range(10):
            print('nop')
    ...

def warns(expected_warning: Union[Type[Warning], Tuple[Type[Warning], ...]]=Warning, *args: Any, match: Optional[Union[str, Pattern[str]]]=None, **kwargs: Any) -> Union['WarningsChecker', Any]:
    if False:
        for i in range(10):
            print('nop')
    'Assert that code raises a particular class of warning.\n\n    Specifically, the parameter ``expected_warning`` can be a warning class or sequence\n    of warning classes, and the code inside the ``with`` block must issue at least one\n    warning of that class or classes.\n\n    This helper produces a list of :class:`warnings.WarningMessage` objects, one for\n    each warning emitted (regardless of whether it is an ``expected_warning`` or not).\n    Since pytest 8.0, unmatched warnings are also re-emitted when the context closes.\n\n    This function can be used as a context manager::\n\n        >>> import pytest\n        >>> with pytest.warns(RuntimeWarning):\n        ...    warnings.warn("my warning", RuntimeWarning)\n\n    In the context manager form you may use the keyword argument ``match`` to assert\n    that the warning matches a text or regex::\n\n        >>> with pytest.warns(UserWarning, match=\'must be 0 or None\'):\n        ...     warnings.warn("value must be 0 or None", UserWarning)\n\n        >>> with pytest.warns(UserWarning, match=r\'must be \\d+$\'):\n        ...     warnings.warn("value must be 42", UserWarning)\n\n        >>> with pytest.warns(UserWarning):  # catch re-emitted warning\n        ...     with pytest.warns(UserWarning, match=r\'must be \\d+$\'):\n        ...         warnings.warn("this is not here", UserWarning)\n        Traceback (most recent call last):\n          ...\n        Failed: DID NOT WARN. No warnings of type ...UserWarning... were emitted...\n\n    **Using with** ``pytest.mark.parametrize``\n\n    When using :ref:`pytest.mark.parametrize ref` it is possible to parametrize tests\n    such that some runs raise a warning and others do not.\n\n    This could be achieved in the same way as with exceptions, see\n    :ref:`parametrizing_conditional_raising` for an example.\n\n    '
    __tracebackhide__ = True
    if not args:
        if kwargs:
            argnames = ', '.join(sorted(kwargs))
            raise TypeError(f'Unexpected keyword arguments passed to pytest.warns: {argnames}\nUse context-manager form instead?')
        return WarningsChecker(expected_warning, match_expr=match, _ispytest=True)
    else:
        func = args[0]
        if not callable(func):
            raise TypeError(f'{func!r} object (type: {type(func)}) must be callable')
        with WarningsChecker(expected_warning, _ispytest=True):
            return func(*args[1:], **kwargs)

class WarningsRecorder(warnings.catch_warnings):
    """A context manager to record raised warnings.

    Each recorded warning is an instance of :class:`warnings.WarningMessage`.

    Adapted from `warnings.catch_warnings`.

    .. note::
        ``DeprecationWarning`` and ``PendingDeprecationWarning`` are treated
        differently; see :ref:`ensuring_function_triggers`.

    """

    def __init__(self, *, _ispytest: bool=False) -> None:
        if False:
            return 10
        check_ispytest(_ispytest)
        super().__init__(record=True)
        self._entered = False
        self._list: List[warnings.WarningMessage] = []

    @property
    def list(self) -> List['warnings.WarningMessage']:
        if False:
            while True:
                i = 10
        'The list of recorded warnings.'
        return self._list

    def __getitem__(self, i: int) -> 'warnings.WarningMessage':
        if False:
            print('Hello World!')
        'Get a recorded warning by index.'
        return self._list[i]

    def __iter__(self) -> Iterator['warnings.WarningMessage']:
        if False:
            while True:
                i = 10
        'Iterate through the recorded warnings.'
        return iter(self._list)

    def __len__(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        'The number of recorded warnings.'
        return len(self._list)

    def pop(self, cls: Type[Warning]=Warning) -> 'warnings.WarningMessage':
        if False:
            print('Hello World!')
        'Pop the first recorded warning which is an instance of ``cls``,\n        but not an instance of a child class of any other match.\n        Raises ``AssertionError`` if there is no match.\n        '
        best_idx: Optional[int] = None
        for (i, w) in enumerate(self._list):
            if w.category == cls:
                return self._list.pop(i)
            if issubclass(w.category, cls) and (best_idx is None or not issubclass(w.category, self._list[best_idx].category)):
                best_idx = i
        if best_idx is not None:
            return self._list.pop(best_idx)
        __tracebackhide__ = True
        raise AssertionError(f'{cls!r} not found in warning list')

    def clear(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Clear the list of recorded warnings.'
        self._list[:] = []

    def __enter__(self) -> 'WarningsRecorder':
        if False:
            return 10
        if self._entered:
            __tracebackhide__ = True
            raise RuntimeError(f'Cannot enter {self!r} twice')
        _list = super().__enter__()
        assert _list is not None
        self._list = _list
        warnings.simplefilter('always')
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        if False:
            i = 10
            return i + 15
        if not self._entered:
            __tracebackhide__ = True
            raise RuntimeError(f'Cannot exit {self!r} without entering first')
        super().__exit__(exc_type, exc_val, exc_tb)
        self._entered = False

@final
class WarningsChecker(WarningsRecorder):

    def __init__(self, expected_warning: Optional[Union[Type[Warning], Tuple[Type[Warning], ...]]]=Warning, match_expr: Optional[Union[str, Pattern[str]]]=None, *, _ispytest: bool=False) -> None:
        if False:
            print('Hello World!')
        check_ispytest(_ispytest)
        super().__init__(_ispytest=True)
        msg = 'exceptions must be derived from Warning, not %s'
        if expected_warning is None:
            warnings.warn(WARNS_NONE_ARG, stacklevel=4)
            expected_warning_tup = None
        elif isinstance(expected_warning, tuple):
            for exc in expected_warning:
                if not issubclass(exc, Warning):
                    raise TypeError(msg % type(exc))
            expected_warning_tup = expected_warning
        elif issubclass(expected_warning, Warning):
            expected_warning_tup = (expected_warning,)
        else:
            raise TypeError(msg % type(expected_warning))
        self.expected_warning = expected_warning_tup
        self.match_expr = match_expr

    def matches(self, warning: warnings.WarningMessage) -> bool:
        if False:
            print('Hello World!')
        assert self.expected_warning is not None
        return issubclass(warning.category, self.expected_warning) and bool(self.match_expr is None or re.search(self.match_expr, str(warning.message)))

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        if False:
            while True:
                i = 10
        super().__exit__(exc_type, exc_val, exc_tb)
        __tracebackhide__ = True
        if self.expected_warning is None:
            return

        def found_str():
            if False:
                print('Hello World!')
            return pformat([record.message for record in self], indent=2)
        try:
            if not any((issubclass(w.category, self.expected_warning) for w in self)):
                fail(f'DID NOT WARN. No warnings of type {self.expected_warning} were emitted.\n Emitted warnings: {found_str()}.')
            elif not any((self.matches(w) for w in self)):
                fail(f'DID NOT WARN. No warnings of type {self.expected_warning} matching the regex were emitted.\n Regex: {self.match_expr}\n Emitted warnings: {found_str()}.')
        finally:
            for w in self:
                if not self.matches(w):
                    warnings.warn_explicit(str(w.message), w.message.__class__, w.filename, w.lineno, module=w.__module__, source=w.source)