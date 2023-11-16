"""Exception classes and constants handling test outcomes as well as
functions creating them."""
import sys
import warnings
from typing import Any
from typing import Callable
from typing import cast
from typing import NoReturn
from typing import Optional
from typing import Protocol
from typing import Type
from typing import TypeVar
from _pytest.deprecated import KEYWORD_MSG_ARG

class OutcomeException(BaseException):
    """OutcomeException and its subclass instances indicate and contain info
    about test and collection outcomes."""

    def __init__(self, msg: Optional[str]=None, pytrace: bool=True) -> None:
        if False:
            return 10
        if msg is not None and (not isinstance(msg, str)):
            error_msg = "{} expected string as 'msg' parameter, got '{}' instead.\nPerhaps you meant to use a mark?"
            raise TypeError(error_msg.format(type(self).__name__, type(msg).__name__))
        super().__init__(msg)
        self.msg = msg
        self.pytrace = pytrace

    def __repr__(self) -> str:
        if False:
            return 10
        if self.msg is not None:
            return self.msg
        return f'<{self.__class__.__name__} instance>'
    __str__ = __repr__
TEST_OUTCOME = (OutcomeException, Exception)

class Skipped(OutcomeException):
    __module__ = 'builtins'

    def __init__(self, msg: Optional[str]=None, pytrace: bool=True, allow_module_level: bool=False, *, _use_item_location: bool=False) -> None:
        if False:
            while True:
                i = 10
        super().__init__(msg=msg, pytrace=pytrace)
        self.allow_module_level = allow_module_level
        self._use_item_location = _use_item_location

class Failed(OutcomeException):
    """Raised from an explicit call to pytest.fail()."""
    __module__ = 'builtins'

class Exit(Exception):
    """Raised for immediate program exits (no tracebacks/summaries)."""

    def __init__(self, msg: str='unknown reason', returncode: Optional[int]=None) -> None:
        if False:
            print('Hello World!')
        self.msg = msg
        self.returncode = returncode
        super().__init__(msg)
_F = TypeVar('_F', bound=Callable[..., object])
_ET = TypeVar('_ET', bound=Type[BaseException])

class _WithException(Protocol[_F, _ET]):
    Exception: _ET
    __call__: _F

def _with_exception(exception_type: _ET) -> Callable[[_F], _WithException[_F, _ET]]:
    if False:
        return 10

    def decorate(func: _F) -> _WithException[_F, _ET]:
        if False:
            while True:
                i = 10
        func_with_exception = cast(_WithException[_F, _ET], func)
        func_with_exception.Exception = exception_type
        return func_with_exception
    return decorate

@_with_exception(Exit)
def exit(reason: str='', returncode: Optional[int]=None, *, msg: Optional[str]=None) -> NoReturn:
    if False:
        print('Hello World!')
    'Exit testing process.\n\n    :param reason:\n        The message to show as the reason for exiting pytest.  reason has a default value\n        only because `msg` is deprecated.\n\n    :param returncode:\n        Return code to be used when exiting pytest.\n\n    :param msg:\n        Same as ``reason``, but deprecated. Will be removed in a future version, use ``reason`` instead.\n    '
    __tracebackhide__ = True
    from _pytest.config import UsageError
    if reason and msg:
        raise UsageError('cannot pass reason and msg to exit(), `msg` is deprecated, use `reason`.')
    if not reason:
        if msg is None:
            raise UsageError('exit() requires a reason argument')
        warnings.warn(KEYWORD_MSG_ARG.format(func='exit'), stacklevel=2)
        reason = msg
    raise Exit(reason, returncode)

@_with_exception(Skipped)
def skip(reason: str='', *, allow_module_level: bool=False, msg: Optional[str]=None) -> NoReturn:
    if False:
        print('Hello World!')
    'Skip an executing test with the given message.\n\n    This function should be called only during testing (setup, call or teardown) or\n    during collection by using the ``allow_module_level`` flag.  This function can\n    be called in doctests as well.\n\n    :param reason:\n        The message to show the user as reason for the skip.\n\n    :param allow_module_level:\n        Allows this function to be called at module level.\n        Raising the skip exception at module level will stop\n        the execution of the module and prevent the collection of all tests in the module,\n        even those defined before the `skip` call.\n\n        Defaults to False.\n\n    :param msg:\n        Same as ``reason``, but deprecated. Will be removed in a future version, use ``reason`` instead.\n\n    .. note::\n        It is better to use the :ref:`pytest.mark.skipif ref` marker when\n        possible to declare a test to be skipped under certain conditions\n        like mismatching platforms or dependencies.\n        Similarly, use the ``# doctest: +SKIP`` directive (see :py:data:`doctest.SKIP`)\n        to skip a doctest statically.\n    '
    __tracebackhide__ = True
    reason = _resolve_msg_to_reason('skip', reason, msg)
    raise Skipped(msg=reason, allow_module_level=allow_module_level)

@_with_exception(Failed)
def fail(reason: str='', pytrace: bool=True, msg: Optional[str]=None) -> NoReturn:
    if False:
        for i in range(10):
            print('nop')
    'Explicitly fail an executing test with the given message.\n\n    :param reason:\n        The message to show the user as reason for the failure.\n\n    :param pytrace:\n        If False, msg represents the full failure information and no\n        python traceback will be reported.\n\n    :param msg:\n        Same as ``reason``, but deprecated. Will be removed in a future version, use ``reason`` instead.\n    '
    __tracebackhide__ = True
    reason = _resolve_msg_to_reason('fail', reason, msg)
    raise Failed(msg=reason, pytrace=pytrace)

def _resolve_msg_to_reason(func_name: str, reason: str, msg: Optional[str]=None) -> str:
    if False:
        while True:
            i = 10
    '\n    Handles converting the deprecated msg parameter if provided into\n    reason, raising a deprecation warning.  This function will be removed\n    when the optional msg argument is removed from here in future.\n\n    :param str func_name:\n        The name of the offending function, this is formatted into the deprecation message.\n\n    :param str reason:\n        The reason= passed into either pytest.fail() or pytest.skip()\n\n    :param str msg:\n        The msg= passed into either pytest.fail() or pytest.skip().  This will\n        be converted into reason if it is provided to allow pytest.skip(msg=) or\n        pytest.fail(msg=) to continue working in the interim period.\n\n    :returns:\n        The value to use as reason.\n\n    '
    __tracebackhide__ = True
    if msg is not None:
        if reason:
            from pytest import UsageError
            raise UsageError(f'Passing both ``reason`` and ``msg`` to pytest.{func_name}(...) is not permitted.')
        warnings.warn(KEYWORD_MSG_ARG.format(func=func_name), stacklevel=3)
        reason = msg
    return reason

class XFailed(Failed):
    """Raised from an explicit call to pytest.xfail()."""

@_with_exception(XFailed)
def xfail(reason: str='') -> NoReturn:
    if False:
        return 10
    'Imperatively xfail an executing test or setup function with the given reason.\n\n    This function should be called only during testing (setup, call or teardown).\n\n    No other code is executed after using ``xfail()`` (it is implemented\n    internally by raising an exception).\n\n    :param reason:\n        The message to show the user as reason for the xfail.\n\n    .. note::\n        It is better to use the :ref:`pytest.mark.xfail ref` marker when\n        possible to declare a test to be xfailed under certain conditions\n        like known bugs or missing features.\n    '
    __tracebackhide__ = True
    raise XFailed(reason)

def importorskip(modname: str, minversion: Optional[str]=None, reason: Optional[str]=None) -> Any:
    if False:
        while True:
            i = 10
    'Import and return the requested module ``modname``, or skip the\n    current test if the module cannot be imported.\n\n    :param modname:\n        The name of the module to import.\n    :param minversion:\n        If given, the imported module\'s ``__version__`` attribute must be at\n        least this minimal version, otherwise the test is still skipped.\n    :param reason:\n        If given, this reason is shown as the message when the module cannot\n        be imported.\n\n    :returns:\n        The imported module. This should be assigned to its canonical name.\n\n    Example::\n\n        docutils = pytest.importorskip("docutils")\n    '
    import warnings
    __tracebackhide__ = True
    compile(modname, '', 'eval')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        try:
            __import__(modname)
        except ImportError as exc:
            if reason is None:
                reason = f'could not import {modname!r}: {exc}'
            raise Skipped(reason, allow_module_level=True) from None
    mod = sys.modules[modname]
    if minversion is None:
        return mod
    verattr = getattr(mod, '__version__', None)
    if minversion is not None:
        from packaging.version import Version
        if verattr is None or Version(verattr) < Version(minversion):
            raise Skipped('module %r has __version__ %r, required is: %r' % (modname, verattr, minversion), allow_module_level=True)
    return mod