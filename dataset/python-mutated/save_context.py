"""Context for building SavedModel."""
import contextlib
import threading

class SaveContext(threading.local):
    """A context for building a graph of SavedModel."""

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        super(SaveContext, self).__init__()
        self._in_save_context = False
        self._options = None

    def options(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.in_save_context():
            raise ValueError('Not in a SaveContext.')
        return self._options

    def enter_save_context(self, options):
        if False:
            i = 10
            return i + 15
        self._in_save_context = True
        self._options = options

    def exit_save_context(self):
        if False:
            for i in range(10):
                print('nop')
        self._in_save_context = False
        self._options = None

    def in_save_context(self):
        if False:
            return 10
        return self._in_save_context
_save_context = SaveContext()

@contextlib.contextmanager
def save_context(options):
    if False:
        i = 10
        return i + 15
    if in_save_context():
        raise ValueError('Already in a SaveContext.')
    _save_context.enter_save_context(options)
    try:
        yield
    finally:
        _save_context.exit_save_context()

def in_save_context():
    if False:
        while True:
            i = 10
    'Returns whether under a save context.'
    return _save_context.in_save_context()

def get_save_options():
    if False:
        i = 10
        return i + 15
    'Returns the save options if under a save context.'
    return _save_context.options()