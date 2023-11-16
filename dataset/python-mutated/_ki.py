from __future__ import annotations
import inspect
import signal
import sys
from functools import wraps
from typing import TYPE_CHECKING, Final, Protocol, TypeVar
import attr
from .._util import is_main_thread
CallableT = TypeVar('CallableT', bound='Callable[..., object]')
RetT = TypeVar('RetT')
if TYPE_CHECKING:
    import types
    from collections.abc import Callable
    from typing_extensions import ParamSpec, TypeGuard
    ArgsT = ParamSpec('ArgsT')
LOCALS_KEY_KI_PROTECTION_ENABLED: Final = '@TRIO_KI_PROTECTION_ENABLED'

def ki_protection_enabled(frame: types.FrameType | None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    while frame is not None:
        if LOCALS_KEY_KI_PROTECTION_ENABLED in frame.f_locals:
            return bool(frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED])
        if frame.f_code.co_name == '__del__':
            return True
        frame = frame.f_back
    return True

def currently_ki_protected() -> bool:
    if False:
        print('Hello World!')
    "Check whether the calling code has :exc:`KeyboardInterrupt` protection\n    enabled.\n\n    It's surprisingly easy to think that one's :exc:`KeyboardInterrupt`\n    protection is enabled when it isn't, or vice-versa. This function tells\n    you what Trio thinks of the matter, which makes it useful for ``assert``\\s\n    and unit tests.\n\n    Returns:\n      bool: True if protection is enabled, and False otherwise.\n\n    "
    return ki_protection_enabled(sys._getframe())

def legacy_isasyncgenfunction(obj: object) -> TypeGuard[Callable[..., types.AsyncGeneratorType[object, object]]]:
    if False:
        print('Hello World!')
    return getattr(obj, '_async_gen_function', None) == id(obj)

def _ki_protection_decorator(enabled: bool) -> Callable[[Callable[ArgsT, RetT]], Callable[ArgsT, RetT]]:
    if False:
        print('Hello World!')

    def decorator(fn: Callable[ArgsT, RetT]) -> Callable[ArgsT, RetT]:
        if False:
            for i in range(10):
                print('nop')
        if inspect.iscoroutinefunction(fn):

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                if False:
                    print('Hello World!')
                coro = fn(*args, **kwargs)
                coro.cr_frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return coro
            return wrapper
        elif inspect.isgeneratorfunction(fn):

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                if False:
                    while True:
                        i = 10
                gen = fn(*args, **kwargs)
                gen.gi_frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return gen
            return wrapper
        elif inspect.isasyncgenfunction(fn) or legacy_isasyncgenfunction(fn):

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                if False:
                    return 10
                agen = fn(*args, **kwargs)
                agen.ag_frame.f_locals[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return agen
            return wrapper
        else:

            @wraps(fn)
            def wrapper(*args: ArgsT.args, **kwargs: ArgsT.kwargs) -> RetT:
                if False:
                    while True:
                        i = 10
                locals()[LOCALS_KEY_KI_PROTECTION_ENABLED] = enabled
                return fn(*args, **kwargs)
            return wrapper
    return decorator

class KIProtectionSignature(Protocol):
    __name__: str

    def __call__(self, f: CallableT, /) -> CallableT:
        if False:
            i = 10
            return i + 15
        pass
enable_ki_protection: KIProtectionSignature = _ki_protection_decorator(True)
enable_ki_protection.__name__ = 'enable_ki_protection'
disable_ki_protection: KIProtectionSignature = _ki_protection_decorator(False)
disable_ki_protection.__name__ = 'disable_ki_protection'

@attr.s
class KIManager:
    handler: Callable[[int, types.FrameType | None], None] | None = attr.ib(default=None)

    def install(self, deliver_cb: Callable[[], object], restrict_keyboard_interrupt_to_checkpoints: bool) -> None:
        if False:
            while True:
                i = 10
        assert self.handler is None
        if not is_main_thread() or signal.getsignal(signal.SIGINT) != signal.default_int_handler:
            return

        def handler(signum: int, frame: types.FrameType | None) -> None:
            if False:
                i = 10
                return i + 15
            assert signum == signal.SIGINT
            protection_enabled = ki_protection_enabled(frame)
            if protection_enabled or restrict_keyboard_interrupt_to_checkpoints:
                deliver_cb()
            else:
                raise KeyboardInterrupt
        self.handler = handler
        signal.signal(signal.SIGINT, handler)

    def close(self) -> None:
        if False:
            print('Hello World!')
        if self.handler is not None:
            if signal.getsignal(signal.SIGINT) is self.handler:
                signal.signal(signal.SIGINT, signal.default_int_handler)
            self.handler = None