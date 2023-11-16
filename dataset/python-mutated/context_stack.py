"""Thread-local context manager stack."""
import contextlib
from tensorflow.python.framework.experimental import thread_local_stack
_default_ctx_stack = thread_local_stack.ThreadLocalStack()

def get_default():
    if False:
        return 10
    'Returns the default execution context.'
    return _default_ctx_stack.peek()

@contextlib.contextmanager
def set_default(ctx):
    if False:
        while True:
            i = 10
    'Returns a contextmanager with `ctx` as the default execution context.'
    try:
        _default_ctx_stack.push(ctx)
        yield
    finally:
        _default_ctx_stack.pop()