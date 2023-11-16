"""An internal wrapper for the beartype library.

The module returns a no-op decorator when the beartype library is not installed.
"""
import enum
import functools
import os
import traceback
import typing
import warnings
from types import ModuleType
try:
    import beartype as _beartype_lib
    from beartype import roar as _roar
    warnings.filterwarnings('ignore', category=_roar.BeartypeDecorHintPep585DeprecationWarning)
    if _beartype_lib.__version__ == '0.16.0':
        warnings.warn('beartype 0.16.0 is not supported. Please upgrade to 0.16.1+.')
        _beartype_lib = None
except ImportError:
    _beartype_lib = None
except Exception as e:
    warnings.warn(f'{e}')
    _beartype_lib = None

@enum.unique
class RuntimeTypeCheckState(enum.Enum):
    """Runtime type check state."""
    DISABLED = enum.auto()
    WARNINGS = enum.auto()
    ERRORS = enum.auto()

class CallHintViolationWarning(UserWarning):
    """Warning raised when a type hint is violated during a function call."""
    pass

def _no_op_decorator(func):
    if False:
        print('Hello World!')
    return func

def _create_beartype_decorator(runtime_check_state: RuntimeTypeCheckState):
    if False:
        print('Hello World!')
    if runtime_check_state == RuntimeTypeCheckState.DISABLED:
        return _no_op_decorator
    if _beartype_lib is None:
        return _no_op_decorator
    assert isinstance(_beartype_lib, ModuleType)
    if runtime_check_state == RuntimeTypeCheckState.ERRORS:
        return _beartype_lib.beartype

    def beartype(func):
        if False:
            while True:
                i = 10
        'Warn on type hint violation.'
        if 'return' in func.__annotations__:
            return_type = func.__annotations__['return']
            del func.__annotations__['return']
            beartyped = _beartype_lib.beartype(func)
            func.__annotations__['return'] = return_type
        else:
            beartyped = _beartype_lib.beartype(func)

        @functools.wraps(func)
        def _coerce_beartype_exceptions_to_warnings(*args, **kwargs):
            if False:
                return 10
            try:
                return beartyped(*args, **kwargs)
            except _roar.BeartypeCallHintParamViolation:
                warnings.warn(traceback.format_exc(), category=CallHintViolationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return _coerce_beartype_exceptions_to_warnings
    return beartype
if typing.TYPE_CHECKING:

    def beartype(func):
        if False:
            i = 10
            return i + 15
        return func
else:
    _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK = os.getenv('TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK')
    if _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK == 'WARNINGS':
        _runtime_type_check_state = RuntimeTypeCheckState.WARNINGS
    elif _TORCH_ONNX_EXPERIMENTAL_RUNTIME_TYPE_CHECK == 'DISABLED':
        _runtime_type_check_state = RuntimeTypeCheckState.DISABLED
    else:
        _runtime_type_check_state = RuntimeTypeCheckState.ERRORS
    beartype = _create_beartype_decorator(_runtime_type_check_state)
    assert beartype is not None