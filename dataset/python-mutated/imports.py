"""Utilities related to importing modules and symbols by name."""
import os
import sys
import warnings
from contextlib import contextmanager
from importlib import import_module, reload
try:
    from importlib.metadata import entry_points
except ImportError:
    from importlib_metadata import entry_points
from kombu.utils.imports import symbol_by_name
MP_MAIN_FILE = os.environ.get('MP_MAIN_FILE')
__all__ = ('NotAPackage', 'qualname', 'instantiate', 'symbol_by_name', 'cwd_in_path', 'find_module', 'import_from_cwd', 'reload_from_cwd', 'module_file', 'gen_task_name')

class NotAPackage(Exception):
    """Raised when importing a package, but it's not a package."""

def qualname(obj):
    if False:
        print('Hello World!')
    'Return object name.'
    if not hasattr(obj, '__name__') and hasattr(obj, '__class__'):
        obj = obj.__class__
    q = getattr(obj, '__qualname__', None)
    if '.' not in q:
        q = '.'.join((obj.__module__, q))
    return q

def instantiate(name, *args, **kwargs):
    if False:
        i = 10
        return i + 15
    'Instantiate class by name.\n\n    See Also:\n        :func:`symbol_by_name`.\n    '
    return symbol_by_name(name)(*args, **kwargs)

@contextmanager
def cwd_in_path():
    if False:
        i = 10
        return i + 15
    'Context adding the current working directory to sys.path.'
    try:
        cwd = os.getcwd()
    except FileNotFoundError:
        cwd = None
    if not cwd:
        yield
    elif cwd in sys.path:
        yield
    else:
        sys.path.insert(0, cwd)
        try:
            yield cwd
        finally:
            try:
                sys.path.remove(cwd)
            except ValueError:
                pass

def find_module(module, path=None, imp=None):
    if False:
        print('Hello World!')
    'Version of :func:`imp.find_module` supporting dots.'
    if imp is None:
        imp = import_module
    with cwd_in_path():
        try:
            return imp(module)
        except ImportError:
            if '.' in module:
                parts = module.split('.')
                for (i, part) in enumerate(parts[:-1]):
                    package = '.'.join(parts[:i + 1])
                    try:
                        mpart = imp(package)
                    except ImportError:
                        break
                    try:
                        mpart.__path__
                    except AttributeError:
                        raise NotAPackage(package)
            raise

def import_from_cwd(module, imp=None, package=None):
    if False:
        return 10
    'Import module, temporarily including modules in the current directory.\n\n    Modules located in the current directory has\n    precedence over modules located in `sys.path`.\n    '
    if imp is None:
        imp = import_module
    with cwd_in_path():
        return imp(module, package=package)

def reload_from_cwd(module, reloader=None):
    if False:
        for i in range(10):
            print('nop')
    'Reload module (ensuring that CWD is in sys.path).'
    if reloader is None:
        reloader = reload
    with cwd_in_path():
        return reloader(module)

def module_file(module):
    if False:
        i = 10
        return i + 15
    'Return the correct original file name of a module.'
    name = module.__file__
    return name[:-1] if name.endswith('.pyc') else name

def gen_task_name(app, name, module_name):
    if False:
        return 10
    'Generate task name from name/module pair.'
    module_name = module_name or '__main__'
    try:
        module = sys.modules[module_name]
    except KeyError:
        module = None
    if module is not None:
        module_name = module.__name__
        if MP_MAIN_FILE and module.__file__ == MP_MAIN_FILE:
            module_name = '__main__'
    if module_name == '__main__' and app.main:
        return '.'.join([app.main, name])
    return '.'.join((p for p in (module_name, name) if p))

def load_extension_class_names(namespace):
    if False:
        return 10
    if sys.version_info >= (3, 10):
        _entry_points = entry_points(group=namespace)
    else:
        try:
            _entry_points = entry_points().get(namespace, [])
        except AttributeError:
            _entry_points = entry_points().select(group=namespace)
    for ep in _entry_points:
        yield (ep.name, ep.value)

def load_extension_classes(namespace):
    if False:
        return 10
    for (name, class_name) in load_extension_class_names(namespace):
        try:
            cls = symbol_by_name(class_name)
        except (ImportError, SyntaxError) as exc:
            warnings.warn(f'Cannot load {namespace} extension {class_name!r}: {exc!r}')
        else:
            yield (name, cls)