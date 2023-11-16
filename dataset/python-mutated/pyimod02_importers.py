"""
PEP-302 and PEP-451 importers for frozen applications.
"""
import sys
import os
import io
import _frozen_importlib
import _thread
from pyimod01_archive import ArchiveReadError, ZlibArchiveReader
SYS_PREFIX = sys._MEIPASS + os.sep
SYS_PREFIXLEN = len(SYS_PREFIX)
imp_new_module = type(sys)
if sys.flags.verbose and sys.stderr:

    def trace(msg, *a):
        if False:
            while True:
                i = 10
        sys.stderr.write(msg % a)
        sys.stderr.write('\n')
else:

    def trace(msg, *a):
        if False:
            return 10
        pass

def _decode_source(source_bytes):
    if False:
        for i in range(10):
            print('nop')
    "\n    Decode bytes representing source code and return the string. Universal newline support is used in the decoding.\n    Based on CPython's implementation of the same functionality:\n    https://github.com/python/cpython/blob/3.9/Lib/importlib/_bootstrap_external.py#L679-L688\n    "
    from tokenize import detect_encoding
    source_bytes_readline = io.BytesIO(source_bytes).readline
    encoding = detect_encoding(source_bytes_readline)
    newline_decoder = io.IncrementalNewlineDecoder(decoder=None, translate=True)
    return newline_decoder.decode(source_bytes.decode(encoding[0]))

class PyiFrozenImporterState:
    """
    An object encapsulating extra information for PyiFrozenImporter, to be stored in `ModuleSpec.loader_state`. Having
    a custom type allows us to verify that module spec indeed contains the original loader state data, as set by
    `PyiFrozenImporter.find_spec`.
    """

    def __init__(self, entry_name):
        if False:
            for i in range(10):
                print('nop')
        self.pyz_entry_name = entry_name

class PyiFrozenImporter:
    """
    Load bytecode of Python modules from the executable created by PyInstaller.

    Python bytecode is zipped and appended to the executable.

    NOTE: PYZ format cannot be replaced by zipimport module.

    The problem is that we have no control over zipimport; for instance, it does not work if the zip file is embedded
    into a PKG that is appended to an executable, like we create in one-file mode.

    This used to be PEP-302 finder and loader class for the ``sys.meta_path`` hook. A PEP-302 finder requires method
    find_module() to return loader class with method load_module(). However, both of these methods were deprecated in
    python 3.4 by PEP-451 (see below). Therefore, this class now provides only optional extensions to the PEP-302
    importer protocol.

    This is also a PEP-451 finder and loader class for the ModuleSpec type import system. A PEP-451 finder requires
    method find_spec(), a PEP-451 loader requires methods exec_module(), load_module() and (optionally) create_module().
    All these methods are implemented in this one class.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        '\n        Load, unzip and initialize the Zip archive bundled with the executable.\n        '
        for pyz_filepath in sys.path:
            try:
                self._pyz_archive = ZlibArchiveReader(pyz_filepath, check_pymagic=True)
                trace('# PyInstaller: PyiFrozenImporter(%s)', pyz_filepath)
                sys.path.remove(pyz_filepath)
                break
            except IOError:
                continue
            except ArchiveReadError:
                continue
        else:
            raise ImportError('Cannot load frozen modules.')
        self.toc = set(self._pyz_archive.toc.keys())
        self._lock = _thread.RLock()
        self._toc_tree = None

    @property
    def toc_tree(self):
        if False:
            return 10
        with self._lock:
            if self._toc_tree is None:
                self._toc_tree = self._build_pyz_prefix_tree()
            return self._toc_tree

    def _build_pyz_prefix_tree(self):
        if False:
            while True:
                i = 10
        tree = dict()
        for entry_name in self.toc:
            name_components = entry_name.split('.')
            current = tree
            if self._pyz_archive.is_package(entry_name):
                for name_component in name_components:
                    current = current.setdefault(name_component, {})
            else:
                for name_component in name_components[:-1]:
                    current = current.setdefault(name_component, {})
                current[name_components[-1]] = ''
        return tree

    def _is_pep420_namespace_package(self, fullname):
        if False:
            print('Hello World!')
        if fullname in self.toc:
            try:
                return self._pyz_archive.is_pep420_namespace_package(fullname)
            except Exception as e:
                raise ImportError(f'PyiFrozenImporter cannot handle module {fullname!r}') from e
        else:
            raise ImportError(f'PyiFrozenImporter cannot handle module {fullname!r}')

    def is_package(self, fullname):
        if False:
            print('Hello World!')
        if fullname in self.toc:
            try:
                return self._pyz_archive.is_package(fullname)
            except Exception as e:
                raise ImportError(f'PyiFrozenImporter cannot handle module {fullname!r}') from e
        else:
            raise ImportError(f'PyiFrozenImporter cannot handle module {fullname!r}')

    def get_code(self, fullname):
        if False:
            while True:
                i = 10
        '\n        Get the code object associated with the module.\n\n        ImportError should be raised if module not found.\n        '
        try:
            if fullname == '__main__':
                return sys.modules['__main__']._pyi_main_co
            return self._pyz_archive.extract(fullname)
        except Exception as e:
            raise ImportError(f'PyiFrozenImporter cannot handle module {fullname!r}') from e

    def get_source(self, fullname):
        if False:
            return 10
        '\n        Method should return the source code for the module as a string.\n        But frozen modules does not contain source code.\n\n        Return None, unless the corresponding source file was explicitly collected to the filesystem.\n        '
        if fullname in self.toc:
            if self.is_package(fullname):
                fullname += '.__init__'
            filename = os.path.join(SYS_PREFIX, fullname.replace('.', os.sep) + '.py')
            try:
                with open(filename, 'rb') as fp:
                    source_bytes = fp.read()
                return _decode_source(source_bytes)
            except FileNotFoundError:
                pass
            return None
        else:
            raise ImportError('No module named ' + fullname)

    def get_data(self, path):
        if False:
            print('Hello World!')
        '\n        Returns the data as a string, or raises IOError if the file was not found. The data is always returned as if\n        "binary" mode was used.\n\n        The \'path\' argument is a path that can be constructed by munging module.__file__ (or pkg.__path__ items).\n\n        This assumes that the file in question was collected into frozen application bundle as a file, and is available\n        on the filesystem. Older versions of PyInstaller also supported data embedded in the PYZ archive, but that has\n        been deprecated in v6.\n        '
        with open(path, 'rb') as fp:
            return fp.read()

    def get_filename(self, fullname):
        if False:
            return 10
        '\n        This method should return the value that __file__ would be set to if the named module was loaded. If the module\n        is not found, an ImportError should be raised.\n        '
        if self.is_package(fullname):
            filename = os.path.join(SYS_PREFIX, fullname.replace('.', os.path.sep), '__init__.pyc')
        else:
            filename = os.path.join(SYS_PREFIX, fullname.replace('.', os.path.sep) + '.pyc')
        return filename

    def find_spec(self, fullname, path=None, target=None):
        if False:
            while True:
                i = 10
        '\n        PEP-451 finder.find_spec() method for the ``sys.meta_path`` hook.\n\n        fullname     fully qualified name of the module\n        path         None for a top-level module, or package.__path__ for\n                     submodules or subpackages.\n        target       unused by this Finder\n\n        Finders are still responsible for identifying, and typically creating, the loader that should be used to load a\n        module. That loader will now be stored in the module spec returned by find_spec() rather than returned directly.\n        As is currently the case without the PEP-452, if a loader would be costly to create, that loader can be designed\n        to defer the cost until later.\n\n        Finders must return ModuleSpec objects when find_spec() is called. This new method replaces find_module() and\n        find_loader() (in the PathEntryFinder case). If a loader does not have find_spec(), find_module() and\n        find_loader() are used instead, for backward-compatibility.\n        '
        entry_name = None
        if path is not None:
            modname = fullname.rsplit('.')[-1]
            for p in path:
                if not p.startswith(SYS_PREFIX):
                    continue
                p = p[SYS_PREFIXLEN:]
                parts = p.split(os.sep)
                if not parts:
                    continue
                if not parts[0]:
                    parts = parts[1:]
                parts.append(modname)
                entry_name = '.'.join(parts)
                if entry_name in self.toc:
                    trace('import %s as %s # PyInstaller PYZ (__path__ override: %s)', entry_name, fullname, p)
                    break
            else:
                entry_name = None
        if entry_name is None:
            if fullname in self.toc:
                entry_name = fullname
                trace('import %s # PyInstaller PYZ', fullname)
        if entry_name is None:
            trace('# %s not found in PYZ', fullname)
            return None
        if self._is_pep420_namespace_package(entry_name):
            from importlib._bootstrap_external import _NamespacePath
            spec = _frozen_importlib.ModuleSpec(fullname, None, is_package=True)
            spec.submodule_search_locations = _NamespacePath(entry_name, [os.path.dirname(self.get_filename(entry_name))], lambda name, path: self.find_spec(name, path))
            return spec
        origin = self.get_filename(entry_name)
        is_pkg = self.is_package(entry_name)
        spec = _frozen_importlib.ModuleSpec(fullname, self, is_package=is_pkg, origin=origin, loader_state=PyiFrozenImporterState(entry_name))
        spec.has_location = True
        if is_pkg:
            spec.submodule_search_locations = [os.path.dirname(self.get_filename(entry_name))]
        return spec

    def create_module(self, spec):
        if False:
            print('Hello World!')
        '\n        PEP-451 loader.create_module() method for the ``sys.meta_path`` hook.\n\n        Loaders may also implement create_module() that will return a new module to exec. It may return None to indicate\n        that the default module creation code should be used. One use case, though atypical, for create_module() is to\n        provide a module that is a subclass of the builtin module type. Most loaders will not need to implement\n        create_module().\n\n        create_module() should properly handle the case where it is called more than once for the same spec/module. This\n        may include returning None or raising ImportError.\n        '
        return None

    def exec_module(self, module):
        if False:
            print('Hello World!')
        '\n        PEP-451 loader.exec_module() method for the ``sys.meta_path`` hook.\n\n        Loaders will have a new method, exec_module(). Its only job is to "exec" the module and consequently populate\n        the module\'s namespace. It is not responsible for creating or preparing the module object, nor for any cleanup\n        afterward. It has no return value. exec_module() will be used during both loading and reloading.\n\n        exec_module() should properly handle the case where it is called more than once. For some kinds of modules this\n        may mean raising ImportError every time after the first time the method is called. This is particularly relevant\n        for reloading, where some kinds of modules do not support in-place reloading.\n        '
        spec = module.__spec__
        if isinstance(spec.loader_state, PyiFrozenImporterState):
            module_name = spec.loader_state.pyz_entry_name
        elif isinstance(spec.loader_state, dict):
            assert spec.origin.startswith(SYS_PREFIX)
            module_name = spec.origin[SYS_PREFIXLEN:].replace(os.sep, '.')
            if module_name.endswith('.pyc'):
                module_name = module_name[:-4]
            if module_name.endswith('.__init__'):
                module_name = module_name[:-9]
        else:
            raise RuntimeError(f"Module's spec contains loader_state of incompatible type: {type(spec.loader_state)}")
        bytecode = self.get_code(module_name)
        if bytecode is None:
            raise RuntimeError(f'Failed to retrieve bytecode for {spec.name!r}!')
        assert hasattr(module, '__file__')
        if spec.submodule_search_locations is not None:
            module.__path__ = [os.path.dirname(module.__file__)]
        exec(bytecode, module.__dict__)

    def get_resource_reader(self, fullname):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return importlib.resource-compatible resource reader.\n        '
        return PyiFrozenResourceReader(self, fullname)

class PyiFrozenResourceReader:
    """
    Resource reader for importlib.resources / importlib_resources support.

    Supports only on-disk resources, which should cover the typical use cases, i.e., the access to data files;
    PyInstaller collects data files onto filesystem, and as of v6.0.0, the embedded PYZ archive is guaranteed
    to contain only .pyc modules.

    When listing resources, source .py files will not be listed as they are not collected by default. Similarly,
    sub-directories that contained only .py files are not reconstructed on filesystem, so they will not be listed,
    either. If access to .py files is required for whatever reason, they need to be explicitly collected as data files
    anyway, which will place them on filesystem and make them appear as resources.

    For on-disk resources, we *must* return path compatible with pathlib.Path() in order to avoid copy to a temporary
    file, which might break under some circumstances, e.g., metpy with importlib_resources back-port, due to:
    https://github.com/Unidata/MetPy/blob/a3424de66a44bf3a92b0dcacf4dff82ad7b86712/src/metpy/plots/wx_symbols.py#L24-L25
    (importlib_resources tries to use 'fonts/wx_symbols.ttf' as a temporary filename suffix, which fails as it contains
    a separator).

    Furthermore, some packages expect files() to return either pathlib.Path or zipfile.Path, e.g.,
    https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/core/utils/resource_utils.py#L81-L97
    This makes implementation of mixed support for on-disk and embedded resources using importlib.abc.Traversable
    protocol rather difficult.

    So in order to maximize compatibility with unfrozen behavior, the below implementation is basically equivalent of
    importlib.readers.FileReader from python 3.10:
      https://github.com/python/cpython/blob/839d7893943782ee803536a47f1d4de160314f85/Lib/importlib/readers.py#L11
    and its underlying classes, importlib.abc.TraversableResources and importlib.abc.ResourceReader:
      https://github.com/python/cpython/blob/839d7893943782ee803536a47f1d4de160314f85/Lib/importlib/abc.py#L422
      https://github.com/python/cpython/blob/839d7893943782ee803536a47f1d4de160314f85/Lib/importlib/abc.py#L312
    """

    def __init__(self, importer, name):
        if False:
            while True:
                i = 10
        from pathlib import Path
        self.importer = importer
        self.path = Path(sys._MEIPASS).joinpath(*name.split('.'))

    def open_resource(self, resource):
        if False:
            return 10
        return self.files().joinpath(resource).open('rb')

    def resource_path(self, resource):
        if False:
            while True:
                i = 10
        return str(self.path.joinpath(resource))

    def is_resource(self, path):
        if False:
            i = 10
            return i + 15
        return self.files().joinpath(path).is_file()

    def contents(self):
        if False:
            i = 10
            return i + 15
        return (item.name for item in self.files().iterdir())

    def files(self):
        if False:
            print('Hello World!')
        return self.path

def install():
    if False:
        i = 10
        return i + 15
    '\n    Install PyiFrozenImporter class into the import machinery.\n\n    This function installs the PyiFrozenImporter class into the import machinery of the running process. The importer\n    is added to sys.meta_path. It could be added to sys.path_hooks, but sys.meta_path is processed by Python before\n    looking at sys.path!\n\n    The order of processing import hooks in sys.meta_path:\n\n    1. built-in modules\n    2. modules from the bundled ZIP archive\n    3. C extension modules\n    4. Modules from sys.path\n    '
    importer = PyiFrozenImporter()
    sys.meta_path.append(importer)
    for item in sys.meta_path:
        if hasattr(item, '__name__') and item.__name__ == 'WindowsRegistryFinder':
            sys.meta_path.remove(item)
            break
    path_finders = []
    for item in reversed(sys.meta_path):
        if getattr(item, '__name__', None) == 'PathFinder':
            sys.meta_path.remove(item)
            if item not in path_finders:
                path_finders.append(item)
    sys.meta_path.extend(reversed(path_finders))
    try:
        sys.modules['__main__'].__loader__ = importer
    except Exception:
        pass
    if sys.version_info >= (3, 11):
        _fixup_frozen_stdlib()

def _fixup_frozen_stdlib():
    if False:
        i = 10
        return i + 15
    import _imp
    if not sys._stdlib_dir:
        try:
            sys._stdlib_dir = sys._MEIPASS
        except AttributeError:
            pass
    for (module_name, module) in sys.modules.items():
        if not _imp.is_frozen(module_name):
            continue
        is_pkg = _imp.is_frozen_package(module_name)
        loader_state = module.__spec__.loader_state
        orig_name = loader_state.origname
        if is_pkg:
            orig_name += '.__init__'
        filename = os.path.join(sys._MEIPASS, *orig_name.split('.')) + '.pyc'
        if not hasattr(module, '__file__'):
            try:
                module.__file__ = filename
            except AttributeError:
                pass
        if loader_state.filename is None and orig_name != 'importlib._bootstrap':
            loader_state.filename = filename