""" Detect imports made for code by Python.

In the freezer, this is a step done to detect the technically needed modules to
initialize the CPython interpreter.

"""
import os
import pkgutil
import sys
from nuitka.importing.StandardLibrary import getStandardLibraryPaths, isStandardLibraryNoAutoInclusionModule, isStandardLibraryPath, scanStandardLibraryPath
from nuitka.Options import isStandaloneMode
from nuitka.PythonVersions import python_version
from nuitka.Tracing import general, printError
from nuitka.utils.Execution import executeProcess
from nuitka.utils.FileOperations import areSamePaths
from nuitka.utils.ModuleNames import ModuleName

def _detectImports(command):
    if False:
        while True:
            i = 10
    if python_version >= 768:
        command += '\nprint("\\n".join(sorted(\n    "import %s # sourcefile %s" % (module.__name__, module.__file__)\n    for module in sys.modules.values()\n    if getattr(module, "__file__", None) not in (None, "<frozen>"\n))), file = sys.stderr)'
    reduced_path = [path_element for path_element in sys.path if not areSamePaths(path_element, '.') if not areSamePaths(path_element, os.path.dirname(sys.modules['__main__'].__file__))]
    command = 'import sys; sys.path = %s; sys.real_prefix = sys.prefix;' % repr(reduced_path) + command
    if str is not bytes:
        command = command.encode('utf8')
    (_stdout, stderr, exit_code) = executeProcess(command=(sys.executable, '-s', '-S', '-v', '-c', 'import sys;exec(sys.stdin.read())'), stdin=command, env=dict(os.environ, PYTHONIOENCODING='utf-8'))
    assert type(stderr) is bytes
    if exit_code != 0:
        if b'KeyboardInterrupt' in stderr:
            general.sysexit('Pressed CTRL-C while detecting early imports.')
        general.warning('There is a problem with detecting imports, CPython said:')
        for line in stderr.split(b'\n'):
            printError(line)
        general.sysexit('Error, please report the issue with above output.')
    detections = []
    for line in stderr.replace(b'\r', b'').split(b'\n'):
        if line.startswith(b'import '):
            parts = line.split(b' # ', 2)
            module_name = parts[0].split(b' ', 2)[1].strip(b"'")
            origin = parts[1].split()[0]
            if python_version >= 768:
                module_name = module_name.decode('utf8')
            module_name = ModuleName(module_name)
            if origin == b'precompiled':
                filename = parts[1][len(b'precompiled from '):]
                if python_version >= 768:
                    filename = filename.decode('utf8')
                if not isStandardLibraryPath(filename):
                    continue
                detections.append((module_name, 3, 'precompiled', filename))
            elif origin == b'from' and python_version < 768:
                filename = parts[1][len(b'from '):]
                if str is not bytes:
                    filename = filename.decode('utf8')
                if not isStandardLibraryPath(filename):
                    continue
                if filename.endswith('.py'):
                    detections.append((module_name, 2, 'sourcefile', filename))
                else:
                    assert False
            elif origin == b'sourcefile':
                filename = parts[1][len(b'sourcefile '):]
                if python_version >= 768:
                    filename = filename.decode('utf8')
                if not isStandardLibraryPath(filename):
                    continue
                if os.path.basename(filename) in ('_collections_abc.py', '_collections_abc.pyc'):
                    module_name = ModuleName('_collections_abc')
                if filename.endswith('.py'):
                    detections.append((module_name, 2, 'sourcefile', filename))
                elif filename.endswith('.pyc'):
                    detections.append((module_name, 3, 'precompiled', filename))
                elif not filename.endswith('<frozen>'):
                    if python_version >= 768 and module_name == 'decimal':
                        module_name = ModuleName('_decimal')
                    detections.append((module_name, 2, 'extension', filename))
            elif origin == b'dynamically':
                filename = parts[1][len(b'dynamically loaded from '):]
                if python_version >= 768:
                    filename = filename.decode('utf8')
                if not isStandardLibraryPath(filename):
                    continue
                detections.append((module_name, 1, 'extension', filename))
    module_names = set()
    for (module_name, _priority, kind, filename) in sorted(detections):
        if isStandardLibraryNoAutoInclusionModule(module_name):
            continue
        if kind == 'extension':
            if not isStandaloneMode():
                continue
            if module_name == '__main__':
                continue
            module_names.add(module_name)
        elif kind == 'precompiled':
            module_names.add(module_name)
        elif kind == 'sourcefile':
            module_names.add(module_name)
        else:
            assert False, kind
    return module_names

def _detectEarlyImports():
    if False:
        for i in range(10):
            print('nop')
    encoding_names = [m[1] for m in pkgutil.iter_modules(sys.modules['encodings'].__path__)]
    if os.name != 'nt':
        for encoding_name in ('mbcs', 'cp65001', 'oem'):
            if encoding_name in encoding_names:
                encoding_names.remove(encoding_name)
    for non_locale_encoding in ('bz2_codec', 'idna', 'base64_codec', 'hex_codec', 'rot_13'):
        if non_locale_encoding in encoding_names:
            encoding_names.remove(non_locale_encoding)
    import_code = ';'.join(('import encodings.%s' % encoding_name for encoding_name in sorted(encoding_names)))
    import_code += ';import locale;'
    if python_version >= 768:
        import_code += 'import inspect;import importlib._bootstrap'
    return _detectImports(command=import_code)
_early_modules_names = None

def detectEarlyImports():
    if False:
        i = 10
        return i + 15
    if not isStandaloneMode():
        return ()
    global _early_modules_names
    if _early_modules_names is None:
        _early_modules_names = tuple(sorted(_detectEarlyImports()))
    return _early_modules_names

def _detectStdlibAutoInclusionModules():
    if False:
        return 10
    if not isStandaloneMode():
        return ()
    stdlib_modules = set()
    for stdlib_dir in getStandardLibraryPaths():
        for module_name in scanStandardLibraryPath(stdlib_dir):
            if not isStandardLibraryNoAutoInclusionModule(module_name):
                stdlib_modules.add(module_name)
    first_ones = ('Tkinter',)
    import_code = '\nimports = %r\n\nfailed = set()\n\nclass ImportBlocker(object):\n    def find_module(self, fullname, path = None):\n        if fullname in failed:\n            return self\n\n        return None\n\n    def load_module(self, name):\n        raise ImportError("%%s has failed before" %% name)\n\nsys.meta_path.insert(0, ImportBlocker())\n\nfor imp in imports:\n    try:\n        __import__(imp)\n    except (ImportError, SyntaxError):\n        failed.add(imp)\n    except ValueError as e:\n        if "cannot contain null bytes" in e.args[0]:\n            failed.add(imp)\n        else:\n            sys.stderr.write("PROBLEM with \'%%s\'\\n" %% imp)\n            raise\n    except Exception:\n        sys.stderr.write("PROBLEM with \'%%s\'\\n" %% imp)\n        raise\n\n    for fail in failed:\n        if fail in sys.modules:\n            del sys.modules[fail]\n' % (tuple((module_name.asString() for module_name in sorted(stdlib_modules, key=lambda name: (name not in first_ones, name)))),)
    return _detectImports(command=import_code)
_stdlib_modules_names = None

def detectStdlibAutoInclusionModules():
    if False:
        while True:
            i = 10
    if not isStandaloneMode():
        return ()
    global _stdlib_modules_names
    if _stdlib_modules_names is None:
        _stdlib_modules_names = _detectStdlibAutoInclusionModules()
        for module_name in detectEarlyImports():
            _stdlib_modules_names.discard(module_name)
        _stdlib_modules_names = tuple(sorted(_stdlib_modules_names))
    return _stdlib_modules_names