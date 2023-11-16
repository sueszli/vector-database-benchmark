"""
Provides a reload() function that acts recursively.

Python's normal :func:`python:reload` function only reloads the module that it's
passed. The :func:`reload` function in this module also reloads everything
imported from that module, which is useful when you're changing files deep
inside a package.

To use this as your default reload function, type this::

    import builtins
    from IPython.lib import deepreload
    builtins.reload = deepreload.reload

A reference to the original :func:`python:reload` is stored in this module as
:data:`original_reload`, so you can restore it later.

This code is almost entirely based on knee.py, which is a Python
re-implementation of hierarchical module import.
"""
import builtins as builtin_mod
from contextlib import contextmanager
import importlib
import sys
from types import ModuleType
from warnings import warn
import types
original_import = builtin_mod.__import__

@contextmanager
def replace_import_hook(new_import):
    if False:
        while True:
            i = 10
    saved_import = builtin_mod.__import__
    builtin_mod.__import__ = new_import
    try:
        yield
    finally:
        builtin_mod.__import__ = saved_import

def get_parent(globals, level):
    if False:
        for i in range(10):
            print('nop')
    "\n    parent, name = get_parent(globals, level)\n\n    Return the package that an import is being performed in.  If globals comes\n    from the module foo.bar.bat (not itself a package), this returns the\n    sys.modules entry for foo.bar.  If globals is from a package's __init__.py,\n    the package's entry in sys.modules is returned.\n\n    If globals doesn't come from a package or a module in a package, or a\n    corresponding entry is not found in sys.modules, None is returned.\n    "
    orig_level = level
    if not level or not isinstance(globals, dict):
        return (None, '')
    pkgname = globals.get('__package__', None)
    if pkgname is not None:
        if not hasattr(pkgname, 'rindex'):
            raise ValueError('__package__ set to non-string')
        if len(pkgname) == 0:
            if level > 0:
                raise ValueError('Attempted relative import in non-package')
            return (None, '')
        name = pkgname
    else:
        if '__name__' not in globals:
            return (None, '')
        modname = globals['__name__']
        if '__path__' in globals:
            globals['__package__'] = name = modname
        else:
            lastdot = modname.rfind('.')
            if lastdot < 0 < level:
                raise ValueError('Attempted relative import in non-package')
            if lastdot < 0:
                globals['__package__'] = None
                return (None, '')
            globals['__package__'] = name = modname[:lastdot]
    dot = len(name)
    for x in range(level, 1, -1):
        try:
            dot = name.rindex('.', 0, dot)
        except ValueError as e:
            raise ValueError('attempted relative import beyond top-level package') from e
    name = name[:dot]
    try:
        parent = sys.modules[name]
    except BaseException as e:
        if orig_level < 1:
            warn("Parent module '%.200s' not found while handling absolute import" % name)
            parent = None
        else:
            raise SystemError("Parent module '%.200s' not loaded, cannot perform relative import" % name) from e
    return (parent, name)

def load_next(mod, altmod, name, buf):
    if False:
        print('Hello World!')
    '\n    mod, name, buf = load_next(mod, altmod, name, buf)\n\n    altmod is either None or same as mod\n    '
    if len(name) == 0:
        return (mod, None, buf)
    dot = name.find('.')
    if dot == 0:
        raise ValueError('Empty module name')
    if dot < 0:
        subname = name
        next = None
    else:
        subname = name[:dot]
        next = name[dot + 1:]
    if buf != '':
        buf += '.'
    buf += subname
    result = import_submodule(mod, subname, buf)
    if result is None and mod != altmod:
        result = import_submodule(altmod, subname, subname)
        if result is not None:
            buf = subname
    if result is None:
        raise ImportError('No module named %.200s' % name)
    return (result, next, buf)
found_now = {}

def import_submodule(mod, subname, fullname):
    if False:
        for i in range(10):
            print('nop')
    'm = import_submodule(mod, subname, fullname)'
    global found_now
    if fullname in found_now and fullname in sys.modules:
        m = sys.modules[fullname]
    else:
        print('Reloading', fullname)
        found_now[fullname] = 1
        oldm = sys.modules.get(fullname, None)
        try:
            if oldm is not None:
                m = importlib.reload(oldm)
            else:
                m = importlib.import_module(subname, mod)
        except:
            if oldm:
                sys.modules[fullname] = oldm
            raise
        add_submodule(mod, m, fullname, subname)
    return m

def add_submodule(mod, submod, fullname, subname):
    if False:
        for i in range(10):
            print('nop')
    'mod.{subname} = submod'
    if mod is None:
        return
    if submod is None:
        submod = sys.modules[fullname]
    setattr(mod, subname, submod)
    return

def ensure_fromlist(mod, fromlist, buf, recursive):
    if False:
        return 10
    "Handle 'from module import a, b, c' imports."
    if not hasattr(mod, '__path__'):
        return
    for item in fromlist:
        if not hasattr(item, 'rindex'):
            raise TypeError("Item in ``from list'' not a string")
        if item == '*':
            if recursive:
                continue
            try:
                all = mod.__all__
            except AttributeError:
                pass
            else:
                ret = ensure_fromlist(mod, all, buf, 1)
                if not ret:
                    return 0
        elif not hasattr(mod, item):
            import_submodule(mod, item, buf + '.' + item)

def deep_import_hook(name, globals=None, locals=None, fromlist=None, level=-1):
    if False:
        print('Hello World!')
    'Replacement for __import__()'
    (parent, buf) = get_parent(globals, level)
    (head, name, buf) = load_next(parent, None if level < 0 else parent, name, buf)
    tail = head
    while name:
        (tail, name, buf) = load_next(tail, tail, name, buf)
    if tail is None:
        raise ValueError('Empty module name')
    if not fromlist:
        return head
    ensure_fromlist(tail, fromlist, buf, 0)
    return tail
modules_reloading = {}

def deep_reload_hook(m):
    if False:
        while True:
            i = 10
    'Replacement for reload().'
    if m is types:
        return m
    if not isinstance(m, ModuleType):
        raise TypeError('reload() argument must be module')
    name = m.__name__
    if name not in sys.modules:
        raise ImportError('reload(): module %.200s not in sys.modules' % name)
    global modules_reloading
    try:
        return modules_reloading[name]
    except:
        modules_reloading[name] = m
    try:
        newm = importlib.reload(m)
    except:
        sys.modules[name] = m
        raise
    finally:
        modules_reloading.clear()
    return newm
original_reload = importlib.reload

def reload(module, exclude=(*sys.builtin_module_names, 'sys', 'os.path', 'builtins', '__main__', 'numpy', 'numpy._globals')):
    if False:
        while True:
            i = 10
    'Recursively reload all modules used in the given module.  Optionally\n    takes a list of modules to exclude from reloading.  The default exclude\n    list contains modules listed in sys.builtin_module_names with additional\n    sys, os.path, builtins and __main__, to prevent, e.g., resetting\n    display, exception, and io hooks.\n    '
    global found_now
    for i in exclude:
        found_now[i] = 1
    try:
        with replace_import_hook(deep_import_hook):
            return deep_reload_hook(module)
    finally:
        found_now = {}