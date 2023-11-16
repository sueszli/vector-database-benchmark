"""Lazy imports that may apply across the xonsh package."""
import importlib
import os
from xonsh.lazyasd import LazyObject, lazyobject
from xonsh.platform import ON_DARWIN, ON_WINDOWS
pygments = LazyObject(lambda : importlib.import_module('pygments'), globals(), 'pygments')
pyghooks = LazyObject(lambda : importlib.import_module('xonsh.pyghooks'), globals(), 'pyghooks')

@lazyobject
def pty():
    if False:
        i = 10
        return i + 15
    if ON_WINDOWS:
        return
    else:
        return importlib.import_module('pty')

@lazyobject
def termios():
    if False:
        print('Hello World!')
    if ON_WINDOWS:
        return
    else:
        return importlib.import_module('termios')

@lazyobject
def fcntl():
    if False:
        i = 10
        return i + 15
    if ON_WINDOWS:
        return
    else:
        return importlib.import_module('fcntl')

@lazyobject
def tty():
    if False:
        for i in range(10):
            print('nop')
    if ON_WINDOWS:
        return
    else:
        return importlib.import_module('tty')

@lazyobject
def _winapi():
    if False:
        return 10
    if ON_WINDOWS:
        import _winapi as m
    else:
        m = None
    return m

@lazyobject
def msvcrt():
    if False:
        for i in range(10):
            print('nop')
    if ON_WINDOWS:
        import msvcrt as m
    else:
        m = None
    return m

@lazyobject
def winutils():
    if False:
        while True:
            i = 10
    if ON_WINDOWS:
        import xonsh.winutils as m
    else:
        m = None
    return m

@lazyobject
def macutils():
    if False:
        print('Hello World!')
    if ON_DARWIN:
        import xonsh.macutils as m
    else:
        m = None
    return m

@lazyobject
def terminal256():
    if False:
        for i in range(10):
            print('nop')
    return importlib.import_module('pygments.formatters.terminal256')

@lazyobject
def html():
    if False:
        while True:
            i = 10
    return importlib.import_module('pygments.formatters.html')

@lazyobject
def os_listxattr():
    if False:
        return 10

    def dummy_listxattr(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return []
    return getattr(os, 'listxattr', dummy_listxattr)