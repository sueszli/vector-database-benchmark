"""
Context managers for adding things to sys.path temporarily.

Authors:

* Brian Granger
"""
import sys
import warnings

class appended_to_syspath(object):
    """
    Deprecated since IPython 8.1, no replacements.

    A context for appending a directory to sys.path for a second."""

    def __init__(self, dir):
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('`appended_to_syspath` is deprecated since IPython 8.1, and has no replacements', DeprecationWarning, stacklevel=2)
        self.dir = dir

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        if self.dir not in sys.path:
            sys.path.append(self.dir)
            self.added = True
        else:
            self.added = False

    def __exit__(self, type, value, traceback):
        if False:
            for i in range(10):
                print('nop')
        if self.added:
            try:
                sys.path.remove(self.dir)
            except ValueError:
                pass
        return False

class prepended_to_syspath(object):
    """A context for prepending a directory to sys.path for a second."""

    def __init__(self, dir):
        if False:
            for i in range(10):
                print('nop')
        self.dir = dir

    def __enter__(self):
        if False:
            print('Hello World!')
        if self.dir not in sys.path:
            sys.path.insert(0, self.dir)
            self.added = True
        else:
            self.added = False

    def __exit__(self, type, value, traceback):
        if False:
            i = 10
            return i + 15
        if self.added:
            try:
                sys.path.remove(self.dir)
            except ValueError:
                pass
        return False