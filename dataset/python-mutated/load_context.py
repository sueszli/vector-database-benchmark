"""Context for storing options for loading a SavedModel."""
import contextlib
import threading

class LoadContext(threading.local):
    """A context for loading a model."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(LoadContext, self).__init__()
        self._entered_load_context = []
        self._load_options = None

    def set_load_options(self, load_options):
        if False:
            for i in range(10):
                print('nop')
        self._load_options = load_options
        self._entered_load_context.append(True)

    def clear_load_options(self):
        if False:
            print('Hello World!')
        self._load_options = None
        self._entered_load_context.pop()

    def load_options(self):
        if False:
            return 10
        return self._load_options

    def in_load_context(self):
        if False:
            while True:
                i = 10
        return self._entered_load_context
_load_context = LoadContext()

@contextlib.contextmanager
def load_context(load_options):
    if False:
        return 10
    _load_context.set_load_options(load_options)
    try:
        yield
    finally:
        _load_context.clear_load_options()

def get_load_options():
    if False:
        while True:
            i = 10
    'Returns the load options under a load context.'
    return _load_context.load_options()

def in_load_context():
    if False:
        for i in range(10):
            print('nop')
    'Returns whether under a load context.'
    return _load_context.in_load_context()