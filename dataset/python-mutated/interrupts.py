import signal
import sys
import threading
from contextlib import contextmanager
from types import FrameType
from typing import Any, Iterator, Optional, Type
from typing_extensions import TypeAlias
SignalHandler: TypeAlias = Any
_received_interrupt = {'received': False}

def setup_interrupt_handlers() -> None:
    if False:
        return 10
    signal.signal(signal.SIGTERM, signal.getsignal(signal.SIGINT))
    if sys.platform == 'win32':
        signal.signal(signal.SIGBREAK, signal.getsignal(signal.SIGINT))

def _replace_interrupt_signal(new_signal_handler: SignalHandler) -> None:
    if False:
        for i in range(10):
            print('nop')
    signal.signal(signal.SIGINT, new_signal_handler)
    setup_interrupt_handlers()

@contextmanager
def capture_interrupts() -> Iterator[None]:
    if False:
        for i in range(10):
            print('nop')
    if threading.current_thread() != threading.main_thread():
        yield
        return
    original_signal_handler = signal.getsignal(signal.SIGINT)

    def _new_signal_handler(_signo: int, _: Optional[FrameType]) -> None:
        if False:
            i = 10
            return i + 15
        _received_interrupt['received'] = True
    signal_replaced = False
    try:
        _replace_interrupt_signal(_new_signal_handler)
        signal_replaced = True
        yield
    finally:
        if signal_replaced:
            _replace_interrupt_signal(original_signal_handler)
            _received_interrupt['received'] = False

def check_captured_interrupt() -> bool:
    if False:
        for i in range(10):
            print('nop')
    return _received_interrupt['received']

def pop_captured_interrupt() -> bool:
    if False:
        return 10
    ret = _received_interrupt['received']
    _received_interrupt['received'] = False
    return ret

@contextmanager
def raise_interrupts_as(error_cls: Type[BaseException]) -> Iterator[None]:
    if False:
        print('Hello World!')
    if threading.current_thread() != threading.main_thread():
        yield
        return
    original_signal_handler = signal.getsignal(signal.SIGINT)

    def _new_signal_handler(_signo: int, _: Optional[FrameType]) -> None:
        if False:
            i = 10
            return i + 15
        raise error_cls()
    signal_replaced = False
    try:
        _replace_interrupt_signal(_new_signal_handler)
        signal_replaced = True
        if _received_interrupt['received']:
            _received_interrupt['received'] = False
            raise error_cls()
        yield
    finally:
        if signal_replaced:
            _replace_interrupt_signal(original_signal_handler)