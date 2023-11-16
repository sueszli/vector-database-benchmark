"""runpy.py - locating and running Python code using the module namespace

Provides support for locating and running Python scripts using the Python
module namespace instead of the native filesystem.

This allows Python code to play nicely with non-filesystem based PEP 302
importers when locating support scripts as well as when importing modules.
"""
import sys
import importlib.machinery
import importlib.util
import io
import types
import os
__all__ = ['run_module', 'run_path']

class _TempModule(object):
    """Temporarily replace a module in sys.modules with an empty namespace"""

    def __init__(self, mod_name):
        if False:
            while True:
                i = 10
        self.mod_name = mod_name
        self.module = types.ModuleType(mod_name)
        self._saved_module = []

    def __enter__(self):
        if False:
            return 10
        mod_name = self.mod_name
        try:
            self._saved_module.append(sys.modules[mod_name])
        except KeyError:
            pass
        sys.modules[mod_name] = self.module
        return self

    def __exit__(self, *args):
        if False:
            print('Hello World!')
        if self._saved_module:
            sys.modules[self.mod_name] = self._saved_module[0]
        else:
            del sys.modules[self.mod_name]
        self._saved_module = []

class _ModifiedArgv0(object):

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value
        self._saved_value = self._sentinel = object()

    def __enter__(self):
        if False:
            return 10
        if self._saved_value is not self._sentinel:
            raise RuntimeError('Already preserving saved value')
        self._saved_value = sys.argv[0]
        sys.argv[0] = self.value

    def __exit__(self, *args):
        if False:
            for i in range(10):
                print('nop')
        self.value = self._sentinel
        sys.argv[0] = self._saved_value

def _run_code(code, run_globals, init_globals=None, mod_name=None, mod_spec=None, pkg_name=None, script_name=None):
    if False:
        print('Hello World!')
    'Helper to run code in nominated namespace'
    if init_globals is not None:
        run_globals.update(init_globals)
    if mod_spec is None:
        loader = None
        fname = script_name
        cached = None
    else:
        loader = mod_spec.loader
        fname = mod_spec.origin
        cached = mod_spec.cached
        if pkg_name is None:
            pkg_name = mod_spec.parent
    run_globals.update(__name__=mod_name, __file__=fname, __cached__=cached, __doc__=None, __loader__=loader, __package__=pkg_name, __spec__=mod_spec)
    exec(code, run_globals)
    return run_globals

def _run_module_code(code, init_globals=None, mod_name=None, mod_spec=None, pkg_name=None, script_name=None):
    if False:
        while True:
            i = 10
    'Helper to run code in new namespace with sys modified'
    fname = script_name if mod_spec is None else mod_spec.origin
    with _TempModule(mod_name) as temp_module, _ModifiedArgv0(fname):
        mod_globals = temp_module.module.__dict__
        _run_code(code, mod_globals, init_globals, mod_name, mod_spec, pkg_name, script_name)
    return mod_globals.copy()

def _get_module_details(mod_name, error=ImportError):
    if False:
        i = 10
        return i + 15
    if mod_name.startswith('.'):
        raise error('Relative module names not supported')
    (pkg_name, _, _) = mod_name.rpartition('.')
    if pkg_name:
        try:
            __import__(pkg_name)
        except ImportError as e:
            if e.name is None or (e.name != pkg_name and (not pkg_name.startswith(e.name + '.'))):
                raise
        existing = sys.modules.get(mod_name)
        if existing is not None and (not hasattr(existing, '__path__')):
            from warnings import warn
            msg = '{mod_name!r} found in sys.modules after import of package {pkg_name!r}, but prior to execution of {mod_name!r}; this may result in unpredictable behaviour'.format(mod_name=mod_name, pkg_name=pkg_name)
            warn(RuntimeWarning(msg))
    try:
        spec = importlib.util.find_spec(mod_name)
    except (ImportError, AttributeError, TypeError, ValueError) as ex:
        msg = 'Error while finding module specification for {!r} ({}: {})'
        if mod_name.endswith('.py'):
            msg += f". Try using '{mod_name[:-3]}' instead of '{mod_name}' as the module name."
        raise error(msg.format(mod_name, type(ex).__name__, ex)) from ex
    if spec is None:
        raise error('No module named %s' % mod_name)
    if spec.submodule_search_locations is not None:
        if mod_name == '__main__' or mod_name.endswith('.__main__'):
            raise error('Cannot use package as __main__ module')
        try:
            pkg_main_name = mod_name + '.__main__'
            return _get_module_details(pkg_main_name, error)
        except error as e:
            if mod_name not in sys.modules:
                raise
            raise error(('%s; %r is a package and cannot ' + 'be directly executed') % (e, mod_name))
    loader = spec.loader
    if loader is None:
        raise error('%r is a namespace package and cannot be executed' % mod_name)
    try:
        code = loader.get_code(mod_name)
    except ImportError as e:
        raise error(format(e)) from e
    if code is None:
        raise error('No code object available for %s' % mod_name)
    return (mod_name, spec, code)

class _Error(Exception):
    """Error that _run_module_as_main() should report without a traceback"""

def _run_module_as_main(mod_name, alter_argv=True):
    if False:
        while True:
            i = 10
    'Runs the designated module in the __main__ namespace\n\n       Note that the executed module will have full access to the\n       __main__ namespace. If this is not desirable, the run_module()\n       function should be used to run the module code in a fresh namespace.\n\n       At the very least, these variables in __main__ will be overwritten:\n           __name__\n           __file__\n           __cached__\n           __loader__\n           __package__\n    '
    try:
        if alter_argv or mod_name != '__main__':
            (mod_name, mod_spec, code) = _get_module_details(mod_name, _Error)
        else:
            (mod_name, mod_spec, code) = _get_main_module_details(_Error)
    except _Error as exc:
        msg = '%s: %s' % (sys.executable, exc)
        sys.exit(msg)
    main_globals = sys.modules['__main__'].__dict__
    if alter_argv:
        sys.argv[0] = mod_spec.origin
    return _run_code(code, main_globals, None, '__main__', mod_spec)

def run_module(mod_name, init_globals=None, run_name=None, alter_sys=False):
    if False:
        print('Hello World!')
    "Execute a module's code without importing it.\n\n       mod_name -- an absolute module name or package name.\n\n       Optional arguments:\n       init_globals -- dictionary used to pre-populate the module’s\n       globals dictionary before the code is executed.\n\n       run_name -- if not None, this will be used for setting __name__;\n       otherwise, __name__ will be set to mod_name + '__main__' if the\n       named module is a package and to just mod_name otherwise.\n\n       alter_sys -- if True, sys.argv[0] is updated with the value of\n       __file__ and sys.modules[__name__] is updated with a temporary\n       module object for the module being executed. Both are\n       restored to their original values before the function returns.\n\n       Returns the resulting module globals dictionary.\n    "
    (mod_name, mod_spec, code) = _get_module_details(mod_name)
    if run_name is None:
        run_name = mod_name
    if alter_sys:
        return _run_module_code(code, init_globals, run_name, mod_spec)
    else:
        return _run_code(code, {}, init_globals, run_name, mod_spec)

def _get_main_module_details(error=ImportError):
    if False:
        for i in range(10):
            print('nop')
    main_name = '__main__'
    saved_main = sys.modules[main_name]
    del sys.modules[main_name]
    try:
        return _get_module_details(main_name)
    except ImportError as exc:
        if main_name in str(exc):
            raise error("can't find %r module in %r" % (main_name, sys.path[0])) from exc
        raise
    finally:
        sys.modules[main_name] = saved_main

def _get_code_from_file(run_name, fname):
    if False:
        for i in range(10):
            print('nop')
    from pkgutil import read_code
    decoded_path = os.path.abspath(os.fsdecode(fname))
    with io.open_code(decoded_path) as f:
        code = read_code(f)
    if code is None:
        with io.open_code(decoded_path) as f:
            code = compile(f.read(), fname, 'exec')
    return (code, fname)

def run_path(path_name, init_globals=None, run_name=None):
    if False:
        i = 10
        return i + 15
    "Execute code located at the specified filesystem location.\n\n       path_name -- filesystem location of a Python script, zipfile,\n       or directory containing a top level __main__.py script.\n\n       Optional arguments:\n       init_globals -- dictionary used to pre-populate the module’s\n       globals dictionary before the code is executed.\n\n       run_name -- if not None, this will be used to set __name__;\n       otherwise, '<run_path>' will be used for __name__.\n\n       Returns the resulting module globals dictionary.\n    "
    if run_name is None:
        run_name = '<run_path>'
    pkg_name = run_name.rpartition('.')[0]
    from pkgutil import get_importer
    importer = get_importer(path_name)
    is_NullImporter = False
    if type(importer).__module__ == 'imp':
        if type(importer).__name__ == 'NullImporter':
            is_NullImporter = True
    if isinstance(importer, type(None)) or is_NullImporter:
        (code, fname) = _get_code_from_file(run_name, path_name)
        return _run_module_code(code, init_globals, run_name, pkg_name=pkg_name, script_name=fname)
    else:
        sys.path.insert(0, path_name)
        try:
            (mod_name, mod_spec, code) = _get_main_module_details()
            with _TempModule(run_name) as temp_module, _ModifiedArgv0(path_name):
                mod_globals = temp_module.module.__dict__
                return _run_code(code, mod_globals, init_globals, run_name, mod_spec, pkg_name).copy()
        finally:
            try:
                sys.path.remove(path_name)
            except ValueError:
                pass
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('No module specified for execution', file=sys.stderr)
    else:
        del sys.argv[0]
        _run_module_as_main(sys.argv[0])