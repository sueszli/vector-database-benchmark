import sys
import traceback
import warnings
from types import TracebackType
from typing import Any
from typing import Callable
from typing import Generator
from typing import Optional
from typing import Type
import pytest

class catch_unraisable_exception:
    """Context manager catching unraisable exception using sys.unraisablehook.

    Storing the exception value (cm.unraisable.exc_value) creates a reference
    cycle. The reference cycle is broken explicitly when the context manager
    exits.

    Storing the object (cm.unraisable.object) can resurrect it if it is set to
    an object which is being finalized. Exiting the context manager clears the
    stored object.

    Usage:
        with catch_unraisable_exception() as cm:
            # code creating an "unraisable exception"
            ...
            # check the unraisable exception: use cm.unraisable
            ...
        # cm.unraisable attribute no longer exists at this point
        # (to break a reference cycle)
    """

    def __init__(self) -> None:
        if False:
            i = 10
            return i + 15
        self.unraisable: Optional['sys.UnraisableHookArgs'] = None
        self._old_hook: Optional[Callable[['sys.UnraisableHookArgs'], Any]] = None

    def _hook(self, unraisable: 'sys.UnraisableHookArgs') -> None:
        if False:
            return 10
        self.unraisable = unraisable

    def __enter__(self) -> 'catch_unraisable_exception':
        if False:
            i = 10
            return i + 15
        self._old_hook = sys.unraisablehook
        sys.unraisablehook = self._hook
        return self

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        if False:
            while True:
                i = 10
        assert self._old_hook is not None
        sys.unraisablehook = self._old_hook
        self._old_hook = None
        del self.unraisable

def unraisable_exception_runtest_hook() -> Generator[None, None, None]:
    if False:
        for i in range(10):
            print('nop')
    with catch_unraisable_exception() as cm:
        try:
            yield
        finally:
            if cm.unraisable:
                if cm.unraisable.err_msg is not None:
                    err_msg = cm.unraisable.err_msg
                else:
                    err_msg = 'Exception ignored in'
                msg = f'{err_msg}: {cm.unraisable.object!r}\n\n'
                msg += ''.join(traceback.format_exception(cm.unraisable.exc_type, cm.unraisable.exc_value, cm.unraisable.exc_traceback))
                warnings.warn(pytest.PytestUnraisableExceptionWarning(msg))

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_setup() -> Generator[None, None, None]:
    if False:
        while True:
            i = 10
    yield from unraisable_exception_runtest_hook()

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_call() -> Generator[None, None, None]:
    if False:
        print('Hello World!')
    yield from unraisable_exception_runtest_hook()

@pytest.hookimpl(wrapper=True, tryfirst=True)
def pytest_runtest_teardown() -> Generator[None, None, None]:
    if False:
        while True:
            i = 10
    yield from unraisable_exception_runtest_hook()