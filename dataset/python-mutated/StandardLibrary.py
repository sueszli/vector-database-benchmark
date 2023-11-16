""" Access to standard library distinction.

For code to be in the standard library means that it's not written by the
user for sure. We treat code differently based on that information, by e.g.
including as byte code.

To determine if a module from the standard library, we can abuse the attribute
"__file__" of the "os" module like it's done in "isStandardLibraryPath" of this
module.
"""
import os
from nuitka.Options import shallUseStaticLibPython
from nuitka.PythonVersions import python_version
from nuitka.utils.FileOperations import getFileContents, isFilenameBelowPath
from nuitka.utils.ModuleNames import ModuleName
from nuitka.utils.Utils import isNetBSD, isPosixWindows, isWin32OrPosixWindows, isWin32Windows

def getStandardLibraryPaths():
    if False:
        return 10
    'Get the standard library paths.'
    if not hasattr(getStandardLibraryPaths, 'result'):
        os_filename = os.__file__
        if os_filename.endswith('.pyc'):
            os_filename = os_filename[:-1]
        os_path = os.path.normcase(os.path.dirname(os_filename))
        stdlib_paths = set([os_path])
        if os.path.islink(os_filename):
            os_filename = os.readlink(os_filename)
            stdlib_paths.add(os.path.normcase(os.path.dirname(os_filename)))
        orig_prefix_filename = os.path.join(os_path, 'orig-prefix.txt')
        if os.path.isfile(orig_prefix_filename):
            search = os_path
            lib_part = ''
            while os.path.splitdrive(search)[1] not in (os.path.sep, ''):
                if os.path.isfile(os.path.join(search, 'bin/activate')) or os.path.isfile(os.path.join(search, 'scripts/activate')):
                    break
                lib_part = os.path.join(os.path.basename(search), lib_part)
                search = os.path.dirname(search)
            assert search and lib_part
            stdlib_paths.add(os.path.normcase(os.path.join(getFileContents(orig_prefix_filename), lib_part)))
        python_link_filename = os.path.join(os_path, '..', '.Python')
        if os.path.islink(python_link_filename):
            stdlib_paths.add(os.path.normcase(os.path.join(os.readlink(python_link_filename), 'lib')))
        for stdlib_path in set(stdlib_paths):
            candidate = os.path.join(stdlib_path, 'lib-tk')
            if os.path.isdir(candidate):
                stdlib_paths.add(candidate)
        if isWin32OrPosixWindows() and (not shallUseStaticLibPython()):
            import _ctypes
            stdlib_paths.add(os.path.dirname(_ctypes.__file__))
        getStandardLibraryPaths.result = [os.path.normcase(os.path.normpath(stdlib_path)) for stdlib_path in stdlib_paths]
    return getStandardLibraryPaths.result

def isStandardLibraryPath(filename):
    if False:
        return 10
    'Check if a path is in the standard library.'
    filename = os.path.normcase(os.path.normpath(filename))
    if os.path.basename(filename) == 'site.py':
        return True
    if 'dist-packages' in filename or 'site-packages' in filename or 'vendor-packages' in filename:
        return False
    for candidate in getStandardLibraryPaths():
        if isFilenameBelowPath(path=candidate, filename=filename):
            return True
    return False
_excluded_stdlib_modules = ['__main__.py', '__init__.py', 'antigravity.py']
if not isWin32Windows():
    _excluded_stdlib_modules.append('wintypes.py')
    _excluded_stdlib_modules.append('cp65001.py')

def scanStandardLibraryPath(stdlib_dir):
    if False:
        i = 10
        return i + 15
    for (root, dirs, filenames) in os.walk(stdlib_dir):
        import_path = root[len(stdlib_dir):].strip('/\\')
        import_path = import_path.replace('\\', '.').replace('/', '.')
        if import_path == '':
            if 'site-packages' in dirs:
                dirs.remove('site-packages')
            if 'dist-packages' in dirs:
                dirs.remove('dist-packages')
            if 'vendor-packages' in dirs:
                dirs.remove('vendor-packages')
            if 'test' in dirs:
                dirs.remove('test')
            if 'turtledemo' in dirs:
                dirs.remove('turtledemo')
            if 'ensurepip' in filenames:
                filenames.remove('ensurepip')
            if 'ensurepip' in dirs:
                dirs.remove('ensurepip')
            dirs[:] = [dirname for dirname in dirs if not dirname.startswith('lib-') if dirname != 'Tools' if not dirname.startswith('plat-')]
        if import_path in ('tkinter', 'Tkinter', 'importlib', 'ctypes', 'unittest', 'sqlite3', 'distutils', 'email', 'bsddb'):
            if 'test' in dirs:
                dirs.remove('test')
        if import_path == 'distutils.command':
            if 'bdist_conda.py' in filenames:
                filenames.remove('bdist_conda.py')
        if import_path in ('lib2to3', 'json', 'distutils'):
            if 'tests' in dirs:
                dirs.remove('tests')
        if import_path == 'asyncio':
            if 'test_utils.py' in filenames:
                filenames.remove('test_utils.py')
        if python_version >= 832 and isWin32Windows():
            if import_path == 'multiprocessing':
                filenames.remove('popen_fork.py')
                filenames.remove('popen_forkserver.py')
                filenames.remove('popen_spawn_posix.py')
        if python_version >= 768 and isPosixWindows():
            if import_path == 'curses':
                filenames.remove('has_key.py')
        if isNetBSD():
            if import_path == 'xml.sax':
                filenames.remove('expatreader.py')
        for filename in filenames:
            if filename.endswith('.py') and filename not in _excluded_stdlib_modules:
                module_name = filename[:-3]
                if import_path == '':
                    yield ModuleName(module_name)
                else:
                    yield ModuleName(import_path + '.' + module_name)
        if python_version >= 768:
            if '__pycache__' in dirs:
                dirs.remove('__pycache__')
        dirs = [dirname for dirname in dirs if not dirname.startswith('.')]
        for dirname in dirs:
            if import_path == '':
                yield ModuleName(dirname)
            else:
                yield ModuleName(import_path + '.' + dirname)
_stdlib_no_auto_inclusion_list = ('multiprocessing', '_multiprocessing', 'curses', '_curses', 'ctypes', '_ctypes', '_curses_panel', 'sqlite3', '_sqlite3', 'shelve', 'dbm', '_dbm', 'bdb', 'xml', '_elementtree', 'queue', '_queue', 'uuid', '_uuid', 'hashlib', '_hashlib', 'secrets', 'hmac', 'fractions', 'decimal', '_pydecimal', '_decimal', 'statistics', 'csv', '_csv', 'lzma', '_lzma', 'bz2', '_bz2', 'logging', 'tempfile', 'subprocess', '_posixsubprocess', 'socket', 'selectors', 'select', '_socket', 'ssl', '_ssl', 'pyexpat', 'readline', 'unittest', 'pydoc', 'pydoc_data', 'profile', 'cProfile', 'optparse', 'pdb', 'site', 'sitecustomize', 'runpy', 'lib2to3', 'doctest', '_json', '_bisect', '_heapq', '_crypt', '_contextvars', 'random', 'array', 'json.tool', 'zipapp', 'tabnanny', 'email', 'mailbox', 'argparse', 'telnetlib', 'smtplib', 'smtpd', 'nntplib', 'http', 'xmlrpc', 'urllib', 'select', 'wsgiref', 'sunau', 'aifc', 'wave', 'audioop', 'getpass', 'grp', 'pty', 'tty', 'termios', 'this', 'textwrap', 'plistlib', 'distutils', 'compileall', 'venv', 'py_compile', 'msilib', '_opcode', 'zoneinfo', 'Tkinter', 'tkinter', '_tkinter', 'Tix', 'FixTk', 'ScrolledText', 'turtle', 'antigravity', 'Dialog', 'Tkdnd', 'tkMessageBox', 'tkSimpleDialog', 'Tkinter', 'tkFileDialog', 'Canvas', 'tkCommonDialog', 'Tkconstants', 'FileDialog', 'SimpleDialog', 'ttk', 'tkFont', 'tkColorChooser', 'idlelib', 'asyncio.test_utils', '_distutils_system_mod', 'concurrent', 'asyncio', 'asyncore', 'asynchat')
if not isWin32Windows():
    _stdlib_no_auto_inclusion_list += ('ntpath',)

def isStandardLibraryNoAutoInclusionModule(module_name):
    if False:
        while True:
            i = 10
    return module_name.hasOneOfNamespaces(*_stdlib_no_auto_inclusion_list)