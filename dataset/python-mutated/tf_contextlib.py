"""TFDecorator-aware replacements for the contextlib module."""
import contextlib as _contextlib
from tensorflow.python.util import tf_decorator

def contextmanager(target):
    if False:
        return 10
    'A tf_decorator-aware wrapper for `contextlib.contextmanager`.\n\n  Usage is identical to `contextlib.contextmanager`.\n\n  Args:\n    target: A callable to be wrapped in a contextmanager.\n  Returns:\n    A callable that can be used inside of a `with` statement.\n  '
    context_manager = _contextlib.contextmanager(target)
    return tf_decorator.make_decorator(target, context_manager, 'contextmanager')