"""
A context manager for handling sys.displayhook.

Authors:

* Robert Kern
* Brian Granger
"""
import sys
from traitlets.config.configurable import Configurable
from traitlets import Any

class DisplayTrap(Configurable):
    """Object to manage sys.displayhook.

    This came from IPython.core.kernel.display_hook, but is simplified
    (no callbacks or formatters) until more of the core is refactored.
    """
    hook = Any()

    def __init__(self, hook=None):
        if False:
            return 10
        super(DisplayTrap, self).__init__(hook=hook, config=None)
        self.old_hook = None
        self._nested_level = 0

    def __enter__(self):
        if False:
            print('Hello World!')
        if self._nested_level == 0:
            self.set()
        self._nested_level += 1
        return self

    def __exit__(self, type, value, traceback):
        if False:
            while True:
                i = 10
        if self._nested_level == 1:
            self.unset()
        self._nested_level -= 1
        return False

    def set(self):
        if False:
            print('Hello World!')
        'Set the hook.'
        if sys.displayhook is not self.hook:
            self.old_hook = sys.displayhook
            sys.displayhook = self.hook

    def unset(self):
        if False:
            while True:
                i = 10
        'Unset the hook.'
        sys.displayhook = self.old_hook