from abc import ABC, abstractmethod
from numba.core.registry import DelayedRegistry, CPUDispatcher
from numba.core.decorators import jit
from numba.core.errors import InternalTargetMismatchError, NumbaValueError
from threading import local as tls
_active_context = tls()
_active_context_default = 'cpu'

class _TargetRegistry(DelayedRegistry):

    def __getitem__(self, item):
        if False:
            return 10
        try:
            return super().__getitem__(item)
        except KeyError:
            msg = "No target is registered against '{}', known targets:\n{}"
            known = '\n'.join([f'{k: <{10}} -> {v}' for (k, v) in target_registry.items()])
            raise NumbaValueError(msg.format(item, known)) from None
target_registry = _TargetRegistry()
jit_registry = DelayedRegistry()

class target_override(object):
    """Context manager to temporarily override the current target with that
       prescribed."""

    def __init__(self, name):
        if False:
            return 10
        self._orig_target = getattr(_active_context, 'target', _active_context_default)
        self.target = name

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        _active_context.target = self.target

    def __exit__(self, ty, val, tb):
        if False:
            while True:
                i = 10
        _active_context.target = self._orig_target

def current_target():
    if False:
        while True:
            i = 10
    'Returns the current target\n    '
    return getattr(_active_context, 'target', _active_context_default)

def get_local_target(context):
    if False:
        print('Hello World!')
    '\n    Gets the local target from the call stack if available and the TLS\n    override if not.\n    '
    if len(context.callstack._stack) > 0:
        target = context.callstack[0].target
    else:
        target = target_registry.get(current_target(), None)
    if target is None:
        msg = 'The target found is not registered.Given target was {}.'
        raise ValueError(msg.format(target))
    else:
        return target

def resolve_target_str(target_str):
    if False:
        for i in range(10):
            print('nop')
    'Resolves a target specified as a string to its Target class.'
    return target_registry[target_str]

def resolve_dispatcher_from_str(target_str):
    if False:
        while True:
            i = 10
    'Returns the dispatcher associated with a target string'
    target_hw = resolve_target_str(target_str)
    return dispatcher_registry[target_hw]

def _get_local_target_checked(tyctx, hwstr, reason):
    if False:
        return 10
    'Returns the local target if it is compatible with the given target\n    name during a type resolution; otherwise, raises an exception.\n\n    Parameters\n    ----------\n    tyctx: typing context\n    hwstr: str\n        target name to check against\n    reason: str\n        Reason for the resolution. Expects a noun.\n    Returns\n    -------\n    target_hw : Target\n\n    Raises\n    ------\n    InternalTargetMismatchError\n    '
    hw_clazz = resolve_target_str(hwstr)
    target_hw = get_local_target(tyctx)
    if not target_hw.inherits_from(hw_clazz):
        raise InternalTargetMismatchError(reason, target_hw, hw_clazz)
    return target_hw

class JitDecorator(ABC):

    @abstractmethod
    def __call__(self):
        if False:
            for i in range(10):
                print('nop')
        return NotImplemented

class Target(ABC):
    """ Implements a target """

    @classmethod
    def inherits_from(cls, other):
        if False:
            i = 10
            return i + 15
        "Returns True if this target inherits from 'other' False otherwise"
        return issubclass(cls, other)

class Generic(Target):
    """Mark the target as generic, i.e. suitable for compilation on
    any target. All must inherit from this.
    """

class CPU(Generic):
    """Mark the target as CPU.
    """

class GPU(Generic):
    """Mark the target as GPU, i.e. suitable for compilation on a GPU
    target.
    """

class CUDA(GPU):
    """Mark the target as CUDA.
    """

class NPyUfunc(Target):
    """Mark the target as a ufunc
    """
target_registry['generic'] = Generic
target_registry['CPU'] = CPU
target_registry['cpu'] = CPU
target_registry['GPU'] = GPU
target_registry['gpu'] = GPU
target_registry['CUDA'] = CUDA
target_registry['cuda'] = CUDA
target_registry['npyufunc'] = NPyUfunc
dispatcher_registry = DelayedRegistry(key_type=Target)
cpu_target = target_registry['cpu']
dispatcher_registry[cpu_target] = CPUDispatcher
jit_registry[cpu_target] = jit