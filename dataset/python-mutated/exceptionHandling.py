"""This module installs a wrapper around sys.excepthook and threading.excepthook that allows multiple
new exception handlers to be registered. 

Optionally, the wrapper also stops exceptions from causing long-term storage 
of local stack frames. This has two major effects:
  - Unhandled exceptions will no longer cause memory leaks
    (If an exception occurs while a lot of data is present on the stack, 
    such as when loading large files, the data would ordinarily be kept
    until the next exception occurs. We would rather release this memory 
    as soon as possible.)
  - Some debuggers may have a hard time handling uncaught exceptions
"""
import sys
import threading
import time
import traceback
from types import SimpleNamespace
callbacks = []
old_callbacks = []
clear_tracebacks = False

def registerCallback(fn):
    if False:
        return 10
    'Register a callable to be invoked when there is an unhandled exception.\n    The callback will be passed an object with attributes: [exc_type, exc_value, exc_traceback, thread]\n    (see threading.excepthook).\n    Multiple callbacks will be invoked in the order they were registered.\n    '
    callbacks.append(fn)

def unregisterCallback(fn):
    if False:
        i = 10
        return i + 15
    'Unregister a previously registered callback.\n    '
    callbacks.remove(fn)

def register(fn):
    if False:
        print('Hello World!')
    'Deprecated; see registerCallback\n\n    Register a callable to be invoked when there is an unhandled exception.\n    The callback will be passed the output of sys.exc_info(): (exception type, exception, traceback)\n    Multiple callbacks will be invoked in the order they were registered.\n    '
    old_callbacks.append(fn)

def unregister(fn):
    if False:
        return 10
    'Deprecated; see unregisterCallback\n\n    Unregister a previously registered callback.\n    '
    old_callbacks.remove(fn)

def setTracebackClearing(clear=True):
    if False:
        print('Hello World!')
    "\n    Enable or disable traceback clearing.\n    By default, clearing is disabled and Python will indefinitely store unhandled exception stack traces.\n    This function is provided since Python's default behavior can cause unexpected retention of \n    large memory-consuming objects.\n    "
    global clear_tracebacks
    clear_tracebacks = clear

class ExceptionHandler(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.orig_sys_excepthook = sys.excepthook
        self.orig_threading_excepthook = threading.excepthook
        sys.excepthook = self.sys_excepthook
        threading.excepthook = self.threading_excepthook

    def remove(self):
        if False:
            while True:
                i = 10
        'Restore original exception hooks, deactivating this exception handler\n        '
        sys.excepthook = self.orig_sys_excepthook
        threading.excepthook = self.orig_threading_excepthook

    def sys_excepthook(self, *args):
        if False:
            while True:
                i = 10
        args = SimpleNamespace(exc_type=args[0], exc_value=args[1], exc_traceback=args[2], thread=None)
        return self._excepthook(args, use_thread_hook=False)

    def threading_excepthook(self, args):
        if False:
            return 10
        return self._excepthook(args, use_thread_hook=True)

    def _excepthook(self, args, use_thread_hook):
        if False:
            print('Hello World!')
        recursionLimit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(recursionLimit + 100)
            global callbacks, clear_tracebacks
            header = '===== %s =====' % str(time.strftime('%Y.%m.%d %H:%m:%S', time.localtime(time.time())))
            try:
                print(header)
            except Exception:
                sys.stderr.write('Warning: stdout is broken! Falling back to stderr.\n')
                sys.stdout = sys.stderr
            if use_thread_hook:
                ret = self.orig_threading_excepthook(args)
            else:
                ret = self.orig_sys_excepthook(args.exc_type, args.exc_value, args.exc_traceback)
            for cb in callbacks:
                try:
                    cb(args)
                except Exception:
                    print('   --------------------------------------------------------------')
                    print('      Error occurred during exception callback %s' % str(cb))
                    print('   --------------------------------------------------------------')
                    traceback.print_exception(*sys.exc_info())
            for cb in old_callbacks:
                try:
                    cb(args.exc_type, args.exc_value, args.exc_traceback)
                except Exception:
                    print('   --------------------------------------------------------------')
                    print('      Error occurred during exception callback %s' % str(cb))
                    print('   --------------------------------------------------------------')
                    traceback.print_exception(*sys.exc_info())
            if clear_tracebacks is True:
                sys.last_traceback = None
            return ret
        finally:
            sys.setrecursionlimit(recursionLimit)

    def implements(self, interface=None):
        if False:
            print('Hello World!')
        if interface is None:
            return ['ExceptionHandler']
        else:
            return interface == 'ExceptionHandler'
if not (hasattr(sys.excepthook, 'implements') and sys.excepthook.implements('ExceptionHandler')):
    handler = ExceptionHandler()
    original_excepthook = handler.orig_sys_excepthook
    original_threading_excepthook = handler.orig_threading_excepthook