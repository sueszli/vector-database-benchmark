"""
Replacement for the Python standard library's atexit.py.

Whereas the standard :mod:`atexit` module only defines :func:`atexit.register`,
this replacement module also defines :func:`unregister`.

This module also fixes a the issue that exceptions raised by an exit handler is
printed twice when the standard :mod:`atexit` is used.
"""
from __future__ import absolute_import
from __future__ import division
import sys
import threading
import traceback
import atexit as std_atexit
from pwnlib.context import context
__all__ = ['register', 'unregister']
_lock = threading.Lock()
_ident = 0
_handlers = {}

def register(func, *args, **kwargs):
    if False:
        return 10
    "register(func, *args, **kwargs)\n\n    Registers a function to be called on program termination.  The function will\n    be called with positional arguments `args` and keyword arguments `kwargs`,\n    i.e. ``func(*args, **kwargs)``.  The current `context` is recorded and will\n    be the one used when the handler is run.\n\n    E.g. to suppress logging output from an exit-handler one could write::\n\n      with context.local(log_level = 'error'):\n        atexit.register(handler)\n\n    An identifier is returned which can be used to unregister the exit-handler.\n\n    This function can be used as a decorator::\n\n      @atexit.register\n      def handler():\n        ...\n\n    Notice however that this will bind ``handler`` to the identifier and not the\n    actual exit-handler.  The exit-handler can then be unregistered with::\n\n      atexit.unregister(handler)\n\n    This function is thread safe.\n\n    "
    global _ident
    with _lock:
        ident = _ident
        _ident += 1
    _handlers[ident] = (func, args, kwargs, vars(context))
    return ident

def unregister(ident):
    if False:
        while True:
            i = 10
    "unregister(ident)\n\n    Remove the exit-handler identified by `ident` from the list of registered\n    handlers.  If `ident` isn't registered this is a no-op.\n    "
    if ident in _handlers:
        del _handlers[ident]

def _run_handlers():
    if False:
        return 10
    '_run_handlers()\n\n    Run registered exit-handlers.  They run in the reverse order of which they\n    were registered.\n\n    If a handler raises an exception, it will be printed but nothing else\n    happens, i.e. other handlers will be run and `sys.excepthook` will not be\n    called for that reason.\n    '
    context.clear()
    for (_ident, (func, args, kwargs, ctx)) in sorted(_handlers.items(), reverse=True):
        try:
            with context.local(**ctx):
                func(*args, **kwargs)
        except SystemExit:
            pass
        except Exception:
            (typ, val, tb) = sys.exc_info()
            traceback.print_exception(typ, val, tb.tb_next)
if hasattr(sys, 'exitfunc'):
    register(sys.exitfunc)
if sys.version_info[0] < 3:
    sys.exitfunc = _run_handlers
else:
    std_atexit.register(_run_handlers)