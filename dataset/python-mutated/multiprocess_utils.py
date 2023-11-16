import atexit
import queue
import signal
import sys
from ..framework import core
MP_STATUS_CHECK_INTERVAL = 5.0
multiprocess_queue_set = set()

def _clear_multiprocess_queue_set():
    if False:
        while True:
            i = 10
    global multiprocess_queue_set
    for data_queue in multiprocess_queue_set:
        while True:
            try:
                data_queue.get_nowait()
            except queue.Empty:
                break

def _cleanup():
    if False:
        i = 10
        return i + 15
    _clear_multiprocess_queue_set()
    core._cleanup_mmap_fds()

def _cleanup_mmap():
    if False:
        print('Hello World!')
    core._cleanup_mmap_fds()

class CleanupFuncRegistrar:
    _executed_func_set = set()
    _registered_func_set = set()

    @classmethod
    def register(cls, function, signals=[]):
        if False:
            for i in range(10):
                print('nop')

        def _func_exectuor():
            if False:
                print('Hello World!')
            if function not in cls._executed_func_set:
                try:
                    function()
                finally:
                    cls._executed_func_set.add(function)

        def _func_register(function):
            if False:
                return 10
            if not callable(function):
                raise TypeError('%s is not callable object.' % function)
            if function not in cls._registered_func_set:
                atexit.register(_func_exectuor)
                cls._registered_func_set.add(function)

        def _signal_handler(signum=None, frame=None):
            if False:
                while True:
                    i = 10
            _func_exectuor()
            if signum is not None:
                if signum == signal.SIGINT:
                    raise KeyboardInterrupt
                sys.exit(signum)

        def _signal_register(signals):
            if False:
                for i in range(10):
                    print('nop')
            signals = set(signals)
            for sig in signals:
                orig_handler = signal.signal(sig, _signal_handler)
                if orig_handler not in (signal.SIG_DFL, signal.SIG_IGN):
                    if sig == signal.SIGINT and orig_handler is signal.default_int_handler:
                        continue
                    if orig_handler not in cls._registered_func_set:
                        atexit.register(orig_handler)
                        cls._registered_func_set.add(orig_handler)
        _signal_register(signals)
        _func_register(function)
if not (sys.platform == 'darwin' or sys.platform == 'win32'):
    CleanupFuncRegistrar.register(_cleanup)
_SIGCHLD_handler_set = False

def _set_SIGCHLD_handler():
    if False:
        i = 10
        return i + 15
    global _SIGCHLD_handler_set
    if _SIGCHLD_handler_set:
        return
    current_handler = signal.getsignal(signal.SIGCHLD)
    if not callable(current_handler):
        current_handler = None

    def __handler__(signum, frame):
        if False:
            print('Hello World!')
        core._throw_error_if_process_failed()
        if current_handler is not None:
            current_handler(signum, frame)
    signal.signal(signal.SIGCHLD, __handler__)
    _SIGCHLD_handler_set = True