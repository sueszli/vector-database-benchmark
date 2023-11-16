""" Helper to import a file as a module.

Used for Nuitka plugins and for test code.
"""
import os
import sys
from nuitka.PythonVersions import python_version
from nuitka.Tracing import general
from .Utils import withNoDeprecationWarning

def _importFilePy3NewWay(filename):
    if False:
        i = 10
        return i + 15
    'Import a file for Python versions 3.5+.'
    import importlib.util
    spec = importlib.util.spec_from_file_location(os.path.basename(filename).split('.')[0], filename)
    user_plugin_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_plugin_module)
    return user_plugin_module

def _importFilePy3OldWay(filename):
    if False:
        return 10
    'Import a file for Python versions before 3.5.'
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(filename, filename).load_module(filename)

def importFilePy2(filename):
    if False:
        for i in range(10):
            print('nop')
    'Import a file for Python version 2.'
    import imp
    basename = os.path.splitext(os.path.basename(filename))[0]
    return imp.load_source(basename, filename)

def importFileAsModule(filename):
    if False:
        while True:
            i = 10
    'Import Python module given as a file name.\n\n    Notes:\n        Provides a Python version independent way to import any script files.\n\n    Args:\n        filename: complete path of a Python script\n\n    Returns:\n        Imported Python module with code from the filename.\n    '
    if python_version < 768:
        return importFilePy2(filename)
    elif python_version < 848:
        return _importFilePy3OldWay(filename)
    else:
        return _importFilePy3NewWay(filename)
_shared_library_suffixes = None

def getSharedLibrarySuffixes():
    if False:
        i = 10
        return i + 15
    global _shared_library_suffixes
    if _shared_library_suffixes is None:
        if python_version < 768:
            import imp
            _shared_library_suffixes = []
            for (suffix, _mode, module_type) in imp.get_suffixes():
                if module_type == imp.C_EXTENSION:
                    _shared_library_suffixes.append(suffix)
        else:
            import importlib.machinery
            _shared_library_suffixes = list(importlib.machinery.EXTENSION_SUFFIXES)
        if '' in _shared_library_suffixes:
            _shared_library_suffixes.remove('')
        _shared_library_suffixes = tuple(_shared_library_suffixes)
    return _shared_library_suffixes

def getSharedLibrarySuffix(preferred):
    if False:
        for i in range(10):
            print('nop')
    if preferred and python_version >= 768:
        return getSharedLibrarySuffixes()[0]
    result = None
    for suffix in getSharedLibrarySuffixes():
        if result is None or len(suffix) < len(result):
            result = suffix
    return result

def _importFromFolder(logger, module_name, path, must_exist, message):
    if False:
        for i in range(10):
            print('nop')
    'Import a module from a folder by adding it temporarily to sys.path'
    from .FileOperations import isFilenameBelowPath
    if module_name in sys.modules:
        if module_name != 'clcache' or isFilenameBelowPath(path=path, filename=sys.modules[module_name].__file__):
            return sys.modules[module_name]
        else:
            del sys.modules[module_name]
    sys.path.insert(0, path)
    try:
        return __import__(module_name, level=0)
    except (ImportError, SyntaxError, RuntimeError) as e:
        if not must_exist:
            return None
        exit_message = "Error, expected inline copy of '%s' to be in '%s', error was: %r." % (module_name, path, e)
        if message is not None:
            exit_message += '\n' + message
        logger.sysexit(exit_message)
    finally:
        del sys.path[0]

def importFromInlineCopy(module_name, must_exist, delete_module=False):
    if False:
        while True:
            i = 10
    'Import a module from the inline copy stage.'
    folder_name = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', 'build', 'inline_copy', module_name))
    candidate_27 = folder_name + '_27'
    candidate_35 = folder_name + '_35'
    if python_version < 768 and os.path.exists(candidate_27):
        folder_name = candidate_27
    elif python_version < 864 and os.path.exists(candidate_35):
        folder_name = candidate_35
    module = _importFromFolder(module_name=module_name, path=folder_name, must_exist=must_exist, message=None, logger=general)
    if delete_module:
        del sys.modules[module_name]
    return module
_compile_time_modules = {}

def importFromCompileTime(module_name, must_exist):
    if False:
        print('Hello World!')
    'Import a module from the compiled time stage.\n\n    This is not for using the inline copy, but the one from the actual\n    installation of the user. It suppresses warnings and caches the value\n    avoid making more __import__ calls that necessary.\n    '
    if module_name not in _compile_time_modules:
        with withNoDeprecationWarning():
            try:
                __import__(module_name)
            except (ImportError, RuntimeError):
                _compile_time_modules[module_name] = False
            else:
                _compile_time_modules[module_name] = sys.modules[module_name]
    assert _compile_time_modules[module_name] or not must_exist
    return _compile_time_modules[module_name] or None

def isBuiltinModuleName(module_name):
    if False:
        while True:
            i = 10
    if python_version < 768:
        import imp as _imp
    else:
        import _imp
    return _imp.is_builtin(module_name)

def getModuleFilenameSuffixes():
    if False:
        for i in range(10):
            print('nop')
    if python_version < 960:
        import imp
        for (suffix, _mode, module_type) in imp.get_suffixes():
            if module_type == imp.C_EXTENSION:
                module_type = 'C_EXTENSION'
            elif module_type == imp.PY_SOURCE:
                module_type = 'PY_SOURCE'
            elif module_type == imp.PY_COMPILED:
                module_type = 'PY_COMPILED'
            else:
                assert False, module_type
            yield (suffix, module_type)
    else:
        import importlib.machinery
        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            yield (suffix, 'C_EXTENSION')
        for suffix in importlib.machinery.SOURCE_SUFFIXES:
            yield (suffix, 'PY_SOURCE')
        for suffix in importlib.machinery.BYTECODE_SUFFIXES:
            yield (suffix, 'PY_COMPILED')