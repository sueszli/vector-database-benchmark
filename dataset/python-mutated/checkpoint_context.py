"""Context for saving checkpoint."""
import contextlib
import threading

class PreemptionSaveContext(threading.local):
    """A context for saving checkpoint upon preemption."""

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self._in_preemption_save_context = False

    def enter_preemption_save_context(self):
        if False:
            for i in range(10):
                print('nop')
        self._in_preemption_save_context = True

    def exit_preemption_save_context(self):
        if False:
            for i in range(10):
                print('nop')
        self._in_preemption_save_context = False

    def in_preemption_save_context(self):
        if False:
            for i in range(10):
                print('nop')
        return self._in_preemption_save_context
_preemption_save_context = PreemptionSaveContext()

@contextlib.contextmanager
def preemption_save_context():
    if False:
        for i in range(10):
            print('nop')
    _preemption_save_context.enter_preemption_save_context()
    try:
        yield
    finally:
        _preemption_save_context.exit_preemption_save_context()

def in_preemption_save_context():
    if False:
        i = 10
        return i + 15
    return _preemption_save_context.in_preemption_save_context()

class AsyncMetricsContext(threading.local):
    """A context for controlling metrics recording when async checkpoint is used.
  """

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self._in_async_metrics_context = False

    def enter_async_metrics_context(self):
        if False:
            i = 10
            return i + 15
        self._in_async_metrics_context = True

    def exit_async_metrics_context(self):
        if False:
            for i in range(10):
                print('nop')
        self._in_async_metrics_context = False

    def in_async_metrics_context(self):
        if False:
            i = 10
            return i + 15
        return self._in_async_metrics_context
_async_metrics_context = AsyncMetricsContext()

@contextlib.contextmanager
def async_metrics_context():
    if False:
        return 10
    _async_metrics_context.enter_async_metrics_context()
    try:
        yield
    finally:
        _async_metrics_context.exit_async_metrics_context()

def in_async_metrics_context():
    if False:
        print('Hello World!')
    return _async_metrics_context.in_async_metrics_context()