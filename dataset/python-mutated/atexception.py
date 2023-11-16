"""
Analogous to atexit, this module allows the programmer to register functions to
be run if an unhandled exception occurs.
"""
from __future__ import absolute_import
from __future__ import division
import sys
import threading
import traceback
from pwnlib.context import context
__all__ = ['register', 'unregister']
_lock = threading.Lock()
_ident = 0
_handlers = {}

def register(func, *args, **kwargs):
    if False:
        print('Hello World!')
    "register(func, *args, **kwargs)\n\n    Registers a function to be called when an unhandled exception occurs.  The\n    function will be called with positional arguments `args` and keyword\n    arguments `kwargs`, i.e. ``func(*args, **kwargs)``.  The current `context`\n    is recorded and will be the one used when the handler is run.\n\n    E.g. to suppress logging output from an exception-handler one could write::\n\n      with context.local(log_level = 'error'):\n        atexception.register(handler)\n\n    An identifier is returned which can be used to unregister the\n    exception-handler.\n\n    This function can be used as a decorator::\n\n      @atexception.register\n      def handler():\n        ...\n\n    Notice however that this will bind ``handler`` to the identifier and not the\n    actual exception-handler.  The exception-handler can then be unregistered\n    with::\n\n      atexception.unregister(handler)\n\n    This function is thread safe.\n\n    "
    global _ident
    with _lock:
        ident = _ident
        _ident += 1
    _handlers[ident] = (func, args, kwargs, vars(context))
    return ident

def unregister(func):
    if False:
        print('Hello World!')
    "unregister(func)\n\n    Remove `func` from the collection of registered functions.  If `func` isn't\n    registered this is a no-op.\n    "
    if func in _handlers:
        del _handlers[func]

def _run_handlers():
    if False:
        i = 10
        return i + 15
    '_run_handlers()\n\n    Run registered handlers.  They run in the reverse order of which they were\n    registered.\n\n    If a handler raises an exception, it will be printed but nothing else\n    happens, i.e. other handlers will be run.\n    '
    for (_ident, (func, args, kwargs, ctx)) in sorted(_handlers.items(), reverse=True):
        try:
            with context.local():
                context.clear()
                context.update(**ctx)
                func(*args, **kwargs)
        except SystemExit:
            pass
        except Exception:
            (typ, val, tb) = sys.exc_info()
            traceback.print_exception(typ, val, tb.tb_next)
_oldhook = getattr(sys, 'excepthook', None)

def _newhook(typ, val, tb):
    if False:
        for i in range(10):
            print('nop')
    '_newhook(typ, val, tb)\n\n    Our excepthook replacement.  First the original hook is called to print the\n    exception, then each handler is called.\n    '
    if _oldhook:
        _oldhook(typ, val, tb)
    if _run_handlers:
        _run_handlers()
sys.excepthook = _newhook