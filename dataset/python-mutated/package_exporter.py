import collections
import importlib.machinery
import io
import linecache
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import Any, BinaryIO, Callable, cast, DefaultDict, Dict, List, Optional, Sequence, Set, Union
import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage
from torch.utils.hooks import RemovableHandle
from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer
__all__ = ['PackagingErrorReason', 'EmptyMatchError', 'PackagingError', 'PackageExporter']
_gate_torchscript_serialization = True
ActionHook = Callable[['PackageExporter', str], None]

class _ModuleProviderAction(Enum):
    """Represents one of the actions that :class:`PackageExporter` can take on a module.

    See :meth:`PackageExporter.extern` and friends for a description of what the actions do.
    """
    INTERN = 1
    EXTERN = 2
    MOCK = 3
    DENY = 4
    REPACKAGED_MOCK_MODULE = 5
    SKIP = 6

class PackagingErrorReason(Enum):
    """Listing of different reasons a dependency may fail to package.

    This enum is used to provide good error messages when
    :class:`PackagingError` is raised.
    """

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'<{self.__class__.__name__}.{self.name}>'
    IS_EXTENSION_MODULE = 'Module is a C extension module. torch.package supports Python modules only.'
    NO_DUNDER_FILE = 'Module had no __file__ defined.'
    SOURCE_FILE_NOT_FOUND = 'Module had a __file__, but we could not find it in your filesystem.'
    DEPENDENCY_RESOLUTION_FAILED = 'Dependency resolution failed.'
    NO_ACTION = 'Module did not match against any action pattern. Extern, mock, or intern it.'
    DENIED = 'Module was denied by a pattern.'
    MOCKED_BUT_STILL_USED = 'Module was mocked out, but is still being used in the package. Please intern or extern the mocked modules if objects are supposed to be in the package.'

@dataclass
class _PatternInfo:
    """Holds :class:`PackageExporter`-specific info about how to execute matches against"""
    action: _ModuleProviderAction
    allow_empty: bool
    was_matched: bool

    def __init__(self, action, allow_empty):
        if False:
            print('Hello World!')
        self.action = action
        self.allow_empty = allow_empty
        self.was_matched = False

class EmptyMatchError(Exception):
    """This is an exception that is thrown when a mock or extern is marked as
    ``allow_empty=False``, and is not matched with any module during packaging.
    """
    pass

class PackagingError(Exception):
    """This exception is raised when there is an issue with exporting a package.
    ``PackageExporter`` will attempt to gather up all the errors and present
    them to you at once.
    """

    def __init__(self, dependency_graph: DiGraph, debug=False):
        if False:
            for i in range(10):
                print('nop')
        broken: Dict[PackagingErrorReason, List[str]] = defaultdict(list)
        for (module_name, attrs) in dependency_graph.nodes.items():
            error = attrs.get('error')
            if error is None:
                continue
            if error == PackagingErrorReason.NO_ACTION:
                assert 'action' not in attrs
            broken[error].append(module_name)
        message = io.StringIO()
        message.write('\n')
        for (reason, module_names) in broken.items():
            message.write(f'* {reason.value}\n')
            for module_name in module_names:
                message.write(f'    {module_name}\n')
                error_context = dependency_graph.nodes[module_name].get('error_context')
                if error_context is not None:
                    message.write(f'      Context: {error_context}\n')
                if module_name in _DISALLOWED_MODULES:
                    message.write(f'      Note: While we usually use modules in the python standard library from the local environment, `{module_name}` has a lot of system level access and therefore can pose a security risk. We heavily recommend removing `{module_name}` from your packaged code. However, if that is not possible, add it to the extern list by calling PackageExporter.extern("`{module_name}`")\n')
                if debug:
                    module_path = dependency_graph.first_path(module_name)
                    message.write(f"      A path to {module_name}: {' -> '.join(module_path)}")
        if not debug:
            message.write('\n')
            message.write('Set debug=True when invoking PackageExporter for a visualization of where broken modules are coming from!\n')
        self.dependency_graph = dependency_graph
        super().__init__(message.getvalue())

class PackageExporter:
    """Exporters allow you to write packages of code, pickled Python data, and
    arbitrary binary and text resources into a self-contained package.

    Imports can load this code in a hermetic way, such that code is loaded
    from the package rather than the normal Python import system. This allows
    for the packaging of PyTorch model code and data so that it can be run
    on a server or used in the future for transfer learning.

    The code contained in packages is copied file-by-file from the original
    source when it is created, and the file format is a specially organized
    zip file. Future users of the package can unzip the package, and edit the code
    in order to perform custom modifications to it.

    The importer for packages ensures that code in the module can only be loaded from
    within the package, except for modules explicitly listed as external using :meth:`extern`.
    The file ``extern_modules`` in the zip archive lists all the modules that a package externally depends on.
    This prevents "implicit" dependencies where the package runs locally because it is importing
    a locally-installed package, but then fails when the package is copied to another machine.

    When source code is added to the package, the exporter can optionally scan it
    for further code dependencies (``dependencies=True``). It looks for import statements,
    resolves relative references to qualified module names, and performs an action specified by the user
    (See: :meth:`extern`, :meth:`mock`, and :meth:`intern`).
    """
    'A importer that will be searched in order to find the modules referenced by other modules or by\n    pickled objects. The default module environment just uses sys_importer, which searches the Python environment.\n    '
    importer: Importer

    def __init__(self, f: Union[str, Path, BinaryIO], importer: Union[Importer, Sequence[Importer]]=sys_importer, debug: bool=False):
        if False:
            while True:
                i = 10
        '\n        Create an exporter.\n\n        Args:\n            f: The location to export to. Can be a  ``string``/``Path`` object containing a filename\n                or a binary I/O object.\n            importer: If a single Importer is passed, use that to search for modules.\n                If a sequence of importers are passed, an ``OrderedImporter`` will be constructed out of them.\n            debug: If set to True, add path of broken modules to PackagingErrors.\n        '
        torch._C._log_api_usage_once('torch.package.PackageExporter')
        self.debug = debug
        if isinstance(f, (Path, str)):
            f = str(f)
            self.buffer: Optional[BinaryIO] = None
        else:
            self.buffer = f
        self.zip_file = torch._C.PyTorchFileWriter(f)
        self.zip_file.set_min_version(6)
        self._written_files: Set[str] = set()
        self.serialized_reduces: Dict[int, Any] = {}
        self.dependency_graph = DiGraph()
        self.script_module_serializer = torch._C.ScriptModuleSerializer(self.zip_file)
        self.storage_context = self.script_module_serializer.storage_context()
        self._extern_hooks: OrderedDict = OrderedDict()
        self._mock_hooks: OrderedDict = OrderedDict()
        self._intern_hooks: OrderedDict = OrderedDict()
        if isinstance(importer, Importer):
            self.importer = importer
        else:
            if not isinstance(importer, collections.abc.Sequence):
                raise TypeError(f'importer arg should be an Importer or a sequence of Importers, got {type(importer)} instead.')
            self.importer = OrderedImporter(*importer)
        self.patterns: Dict[GlobGroup, _PatternInfo] = {}
        self._unique_id = 0

    def save_source_file(self, module_name: str, file_or_directory: str, dependencies=True):
        if False:
            i = 10
            return i + 15
        'Adds the local file system ``file_or_directory`` to the source package to provide the code\n        for ``module_name``.\n\n        Args:\n            module_name (str): e.g. ``"my_package.my_subpackage"``, code will be saved to provide code for this package.\n            file_or_directory (str): the path to a file or directory of code. When a directory, all python files in the directory\n                are recursively copied using :meth:`save_source_file`. If a file is named ``"/__init__.py"`` the code is treated\n                as a package.\n            dependencies (bool, optional): If ``True``, we scan the source for dependencies.\n        '
        path = Path(file_or_directory)
        if path.is_dir():
            to_save = []
            module_path = module_name.replace('.', '/')
            for filename in path.glob('**/*.py'):
                relative_path = filename.relative_to(path).as_posix()
                archivename = module_path + '/' + relative_path
                submodule_name = None
                if filename.name == '__init__.py':
                    submodule_name = archivename[:-len('/__init__.py')].replace('/', '.')
                    is_package = True
                else:
                    submodule_name = archivename[:-len('.py')].replace('/', '.')
                    is_package = False
                to_save.append((submodule_name, _read_file(str(filename)), is_package, dependencies))
            for item in to_save:
                self.save_source_string(*item)
        else:
            is_package = path.name == '__init__.py'
            self.save_source_string(module_name, _read_file(file_or_directory), is_package, dependencies)

    def get_unique_id(self) -> str:
        if False:
            while True:
                i = 10
        'Get an id. This id is guaranteed to only be handed out once for this package.'
        ret = str(self._unique_id)
        self._unique_id += 1
        return ret

    def _get_dependencies(self, src: str, module_name: str, is_package: bool) -> List[str]:
        if False:
            while True:
                i = 10
        'Return all modules that this source code depends on.\n\n        Dependencies are found by scanning the source code for import-like statements.\n\n        Arguments:\n            src: The Python source code to analyze for dependencies.\n            module_name: The name of the module that ``src`` corresponds to.\n            is_package: Whether this module should be treated as a package.\n                See :py:meth:`save_source_string` for more info.\n\n        Returns:\n            A list containing modules detected as direct dependencies in\n            ``src``.  The items in the list are guaranteed to be unique.\n        '
        package_name = module_name if is_package else module_name.rsplit('.', maxsplit=1)[0]
        try:
            dep_pairs = find_files_source_depends_on(src, package_name)
        except Exception as e:
            self.dependency_graph.add_node(module_name, error=PackagingErrorReason.DEPENDENCY_RESOLUTION_FAILED, error_context=str(e))
            return []
        dependencies = {}
        for (dep_module_name, dep_module_obj) in dep_pairs:
            if dep_module_obj is not None:
                possible_submodule = f'{dep_module_name}.{dep_module_obj}'
                if self._module_exists(possible_submodule):
                    dependencies[possible_submodule] = True
                    continue
            if self._module_exists(dep_module_name):
                dependencies[dep_module_name] = True
        return list(dependencies.keys())

    def save_source_string(self, module_name: str, src: str, is_package: bool=False, dependencies: bool=True):
        if False:
            while True:
                i = 10
        'Adds ``src`` as the source code for ``module_name`` in the exported package.\n\n        Args:\n            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code for this package.\n            src (str): The Python source code to save for this package.\n            is_package (bool, optional): If ``True``, this module is treated as a package. Packages are allowed to have submodules\n                (e.g. ``my_package.my_subpackage.my_subsubpackage``), and resources can be saved inside them. Defaults to ``False``.\n            dependencies (bool, optional): If ``True``, we scan the source for dependencies.\n        '
        self.dependency_graph.add_node(module_name, source=src, is_package=is_package, provided=True, action=_ModuleProviderAction.INTERN)
        if dependencies:
            deps = self._get_dependencies(src, module_name, is_package)
            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def _write_source_string(self, module_name: str, src: str, is_package: bool=False):
        if False:
            while True:
                i = 10
        'Write ``src`` as the source code for ``module_name`` in the zip archive.\n\n        Arguments are otherwise the same as for :meth:`save_source_string`.\n        '
        extension = '/__init__.py' if is_package else '.py'
        filename = module_name.replace('.', '/') + extension
        self._write(filename, src)

    def _import_module(self, module_name: str):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self.importer.import_module(module_name)
        except ModuleNotFoundError as e:
            if not is_mangled(module_name):
                raise
            msg = f"Module not found: '{module_name}'. Make sure the PackageImporter that created this module is present in `self.importer`"
            raise ModuleNotFoundError(msg) from None

    def _module_exists(self, module_name: str) -> bool:
        if False:
            print('Hello World!')
        try:
            self._import_module(module_name)
            return True
        except Exception:
            return False

    def _get_source_of_module(self, module: types.ModuleType) -> Optional[str]:
        if False:
            i = 10
            return i + 15
        filename = None
        spec = getattr(module, '__spec__', None)
        if spec is not None:
            loader = getattr(spec, 'loader', None)
            if loader is not None and isinstance(loader, SourceFileLoader):
                try:
                    filename = loader.get_filename(module.__name__)
                except ImportError:
                    pass
        if filename is None:
            filename = getattr(module, '__file__', None)
        if isinstance(filename, str) and filename.endswith('.py'):
            return ''.join(linecache.getlines(filename, module.__dict__))
        return None

    def add_dependency(self, module_name: str, dependencies=True):
        if False:
            for i in range(10):
                print('nop')
        'Given a module, add it to the dependency graph according to patterns\n        specified by the user.\n        '
        if module_name in self.dependency_graph and self.dependency_graph.nodes[module_name].get('provided') is True:
            return
        if module_name == 'torch_package_importer':
            self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.SKIP, provided=True)
            return
        if module_name == '_mock':
            self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.REPACKAGED_MOCK_MODULE, provided=True)
            return
        if self._can_implicitly_extern(module_name):
            self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.EXTERN, provided=True)
            return
        for (pattern, pattern_info) in self.patterns.items():
            if pattern.matches(module_name):
                pattern_info.was_matched = True
                self.dependency_graph.add_node(module_name, action=pattern_info.action, provided=True)
                if pattern_info.action == _ModuleProviderAction.DENY:
                    self.dependency_graph.add_node(module_name, error=PackagingErrorReason.DENIED)
                if pattern_info.action == _ModuleProviderAction.INTERN:
                    self._intern_module(module_name, dependencies)
                return
        self.dependency_graph.add_node(module_name, error=PackagingErrorReason.NO_ACTION)

    def save_module(self, module_name: str, dependencies=True):
        if False:
            i = 10
            return i + 15
        'Save the code for ``module`` into the package. Code for the module is resolved using the ``importers`` path to find the\n        module object, and then using its ``__file__`` attribute to find the source code.\n\n        Args:\n            module_name (str): e.g. ``my_package.my_subpackage``, code will be saved to provide code\n                for this package.\n            dependencies (bool, optional): If ``True``, we scan the source for dependencies.\n        '
        if not isinstance(module_name, str):
            raise TypeError('save_module() expects a string input, did you perhaps mean to pass `__name__`?')
        self._intern_module(module_name, dependencies)

    def _intern_module(self, module_name: str, dependencies: bool):
        if False:
            print('Hello World!')
        'Adds the module to the dependency graph as an interned module,\n        along with any metadata needed to write it out to the zipfile at serialization time.\n        '
        module_obj = self._import_module(module_name)
        module_name = demangle(module_name)
        is_package = hasattr(module_obj, '__path__')
        source = self._get_source_of_module(module_obj)
        if source is None:
            filename = getattr(module_obj, '__file__', None)
            error_context = None
            if filename is None:
                packaging_error = PackagingErrorReason.NO_DUNDER_FILE
            elif filename.endswith(tuple(importlib.machinery.EXTENSION_SUFFIXES)):
                packaging_error = PackagingErrorReason.IS_EXTENSION_MODULE
            else:
                packaging_error = PackagingErrorReason.SOURCE_FILE_NOT_FOUND
                error_context = f'filename: {filename}'
            self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.INTERN, is_package=is_package, error=packaging_error, error_context=error_context, provided=True)
            return
        self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.INTERN, is_package=is_package, source=source, provided=True)
        if dependencies:
            deps = self._get_dependencies(source, module_name, is_package)
            for dep in deps:
                self.dependency_graph.add_edge(module_name, dep)
                self.add_dependency(dep)

    def save_pickle(self, package: str, resource: str, obj: Any, dependencies: bool=True, pickle_protocol: int=3):
        if False:
            print('Hello World!')
        'Save a python object to the archive using pickle. Equivalent to :func:`torch.save` but saving into\n        the archive rather than a stand-alone file. Standard pickle does not save the code, only the objects.\n        If ``dependencies`` is true, this method will also scan the pickled objects for which modules are required\n        to reconstruct them and save the relevant code.\n\n        To be able to save an object where ``type(obj).__name__`` is ``my_module.MyObject``,\n        ``my_module.MyObject`` must resolve to the class of the object according to the ``importer`` order. When saving objects that\n        have previously been packaged, the importer\'s ``import_module`` method will need to be present in the ``importer`` list\n        for this to work.\n\n        Args:\n            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).\n            resource (str): A unique name for the resource, used to identify it to load.\n            obj (Any): The object to save, must be picklable.\n            dependencies (bool, optional): If ``True``, we scan the source for dependencies.\n        '
        assert pickle_protocol == 4 or pickle_protocol == 3, 'torch.package only supports pickle protocols 3 and 4'
        filename = self._filename(package, resource)
        data_buf = io.BytesIO()
        pickler = create_pickler(data_buf, self.importer, protocol=pickle_protocol)
        pickler.persistent_id = self._persistent_id
        pickler.dump(obj)
        data_value = data_buf.getvalue()
        mocked_modules = defaultdict(list)
        name_in_dependency_graph = f'<{package}.{resource}>'
        self.dependency_graph.add_node(name_in_dependency_graph, action=_ModuleProviderAction.INTERN, provided=True, is_pickle=True)

        def _check_mocked_error(module: Optional[str], field: Optional[str]):
            if False:
                return 10
            '\n            checks if an object (field) comes from a mocked module and then adds\n            the pair to mocked_modules which contains mocked modules paired with their\n            list of mocked objects present in the pickle.\n\n            We also hold the invariant that the first user defined rule that applies\n            to the module is the one we use.\n            '
            assert isinstance(module, str)
            assert isinstance(field, str)
            if self._can_implicitly_extern(module):
                return
            for (pattern, pattern_info) in self.patterns.items():
                if pattern.matches(module):
                    if pattern_info.action == _ModuleProviderAction.MOCK:
                        mocked_modules[module].append(field)
                    return
        if dependencies:
            all_dependencies = []
            module = None
            field = None
            memo: DefaultDict[int, str] = defaultdict(None)
            memo_count = 0
            for (opcode, arg, pos) in pickletools.genops(data_value):
                if pickle_protocol == 4:
                    if opcode.name == 'SHORT_BINUNICODE' or opcode.name == 'BINUNICODE' or opcode.name == 'BINUNICODE8':
                        assert isinstance(arg, str)
                        module = field
                        field = arg
                        memo[memo_count] = arg
                    elif opcode.name == 'LONG_BINGET' or opcode.name == 'BINGET' or opcode.name == 'GET':
                        assert isinstance(arg, int)
                        module = field
                        field = memo.get(arg, None)
                    elif opcode.name == 'MEMOIZE':
                        memo_count += 1
                    elif opcode.name == 'STACK_GLOBAL':
                        if module is None:
                            continue
                        assert isinstance(module, str)
                        if module not in all_dependencies:
                            all_dependencies.append(module)
                        _check_mocked_error(module, field)
                elif pickle_protocol == 3 and opcode.name == 'GLOBAL':
                    assert isinstance(arg, str)
                    (module, field) = arg.split(' ')
                    if module not in all_dependencies:
                        all_dependencies.append(module)
                    _check_mocked_error(module, field)
            for module_name in all_dependencies:
                self.dependency_graph.add_edge(name_in_dependency_graph, module_name)
                ' If an object happens to come from a mocked module, then we collect these errors and spit them\n                    out with the other errors found by package exporter.\n                '
                if module in mocked_modules:
                    assert isinstance(module, str)
                    fields = mocked_modules[module]
                    self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.MOCK, error=PackagingErrorReason.MOCKED_BUT_STILL_USED, error_context=f"Object(s) '{fields}' from module `{module_name}` was mocked out during packaging but is being used in resource - `{resource}` in package `{package}`. ", provided=True)
                else:
                    self.add_dependency(module_name)
        self._write(filename, data_value)

    def save_text(self, package: str, resource: str, text: str):
        if False:
            for i in range(10):
                print('nop')
        'Save text data to the package.\n\n        Args:\n            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).\n            resource (str): A unique name for the resource, used to identify it to load.\n            text (str): The contents to save.\n        '
        return self.save_binary(package, resource, text.encode('utf-8'))

    def save_binary(self, package, resource, binary: bytes):
        if False:
            print('Hello World!')
        'Save raw bytes to the package.\n\n        Args:\n            package (str): The name of module package this resource should go it (e.g. ``"my_package.my_subpackage"``).\n            resource (str): A unique name for the resource, used to identify it to load.\n            binary (str): The data to save.\n        '
        filename = self._filename(package, resource)
        self._write(filename, binary)

    def register_extern_hook(self, hook: ActionHook) -> RemovableHandle:
        if False:
            while True:
                i = 10
        'Registers an extern hook on the exporter.\n\n        The hook will be called each time a module matches against an :meth:`extern` pattern.\n        It should have the following signature::\n\n            hook(exporter: PackageExporter, module_name: str) -> None\n\n        Hooks will be called in order of registration.\n\n        Returns:\n            :class:`torch.utils.hooks.RemovableHandle`:\n                A handle that can be used to remove the added hook by calling\n                ``handle.remove()``.\n        '
        handle = RemovableHandle(self._extern_hooks)
        self._extern_hooks[handle.id] = hook
        return handle

    def register_mock_hook(self, hook: ActionHook) -> RemovableHandle:
        if False:
            for i in range(10):
                print('nop')
        'Registers a mock hook on the exporter.\n\n        The hook will be called each time a module matches against a :meth:`mock` pattern.\n        It should have the following signature::\n\n            hook(exporter: PackageExporter, module_name: str) -> None\n\n        Hooks will be called in order of registration.\n\n        Returns:\n            :class:`torch.utils.hooks.RemovableHandle`:\n                A handle that can be used to remove the added hook by calling\n                ``handle.remove()``.\n        '
        handle = RemovableHandle(self._mock_hooks)
        self._mock_hooks[handle.id] = hook
        return handle

    def register_intern_hook(self, hook: ActionHook) -> RemovableHandle:
        if False:
            while True:
                i = 10
        'Registers an intern hook on the exporter.\n\n        The hook will be called each time a module matches against an :meth:`intern` pattern.\n        It should have the following signature::\n\n            hook(exporter: PackageExporter, module_name: str) -> None\n\n        Hooks will be called in order of registration.\n\n        Returns:\n            :class:`torch.utils.hooks.RemovableHandle`:\n                A handle that can be used to remove the added hook by calling\n                ``handle.remove()``.\n        '
        handle = RemovableHandle(self._intern_hooks)
        self._intern_hooks[handle.id] = hook
        return handle

    def intern(self, include: 'GlobPattern', *, exclude: 'GlobPattern'=(), allow_empty: bool=True):
        if False:
            while True:
                i = 10
        'Specify modules that should be packaged. A module must match some ``intern`` pattern in order to be\n        included in the package and have its dependencies processed recursively.\n\n        Args:\n            include (Union[List[str], str]): A string e.g. "my_package.my_subpackage", or list of strings\n                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.\n\n            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.\n\n            allow_empty (bool): An optional flag that specifies whether the intern modules specified by this call\n                to the ``intern`` method must be matched to some module during packaging. If an ``intern`` module glob\n                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``)\n                before any modules match that pattern, an exception is thrown. If ``allow_empty=True``, no such exception is thrown.\n\n        '
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(_ModuleProviderAction.INTERN, allow_empty)

    def mock(self, include: 'GlobPattern', *, exclude: 'GlobPattern'=(), allow_empty: bool=True):
        if False:
            i = 10
            return i + 15
        'Replace some required modules with a mock implementation.  Mocked modules will return a fake\n        object for any attribute accessed from it. Because we copy file-by-file, the dependency resolution will sometimes\n        find files that are imported by model files but whose functionality is never used\n        (e.g. custom serialization code or training helpers).\n        Use this function to mock this functionality out without having to modify the original code.\n\n        Args:\n            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings\n                for the names of the modules to be mocked out. Strings can also be a glob-style pattern\n                string that may match multiple modules. Any required dependencies that match this pattern\n                string will be mocked out automatically.\n\n                Examples :\n                    ``\'torch.**\'`` -- matches ``torch`` and all submodules of torch, e.g. ``\'torch.nn\'``\n                    and ``\'torch.nn.functional\'``\n\n                    ``\'torch.*\'`` -- matches ``\'torch.nn\'`` or ``\'torch.functional\'``, but not\n                    ``\'torch.nn.functional\'``\n\n            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.\n                e.g. ``include=\'torch.**\', exclude=\'torch.foo\'`` will mock all torch packages except ``\'torch.foo\'``,\n                Default: is ``[]``.\n\n            allow_empty (bool): An optional flag that specifies whether the mock implementation(s) specified by this call\n                to the :meth:`mock` method must be matched to some module during packaging. If a mock is added with\n                ``allow_empty=False``, and :meth:`close` is called (either explicitly or via ``__exit__``) and the mock has\n                not been matched to a module used by the package being exported, an exception is thrown.\n                If ``allow_empty=True``, no such exception is thrown.\n\n        '
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(_ModuleProviderAction.MOCK, allow_empty)

    def extern(self, include: 'GlobPattern', *, exclude: 'GlobPattern'=(), allow_empty: bool=True):
        if False:
            print('Hello World!')
        'Include ``module`` in the list of external modules the package can import.\n        This will prevent dependency discovery from saving\n        it in the package. The importer will load an external module directly from the standard import system.\n        Code for extern modules must also exist in the process loading the package.\n\n        Args:\n            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings\n                for the names of the modules to be externed. This can also be a glob-style pattern, as\n                described in :meth:`mock`.\n\n            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the\n                include string.\n\n            allow_empty (bool): An optional flag that specifies whether the extern modules specified by this call\n                to the ``extern`` method must be matched to some module during packaging. If an extern module glob\n                pattern is added with ``allow_empty=False``, and :meth:`close` is called (either explicitly or via\n                ``__exit__``) before any modules match that pattern, an exception is thrown. If ``allow_empty=True``,\n                no such exception is thrown.\n\n        '
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(_ModuleProviderAction.EXTERN, allow_empty)

    def deny(self, include: 'GlobPattern', *, exclude: 'GlobPattern'=()):
        if False:
            for i in range(10):
                print('nop')
        'Blocklist modules who names match the given glob patterns from the list of modules the package can import.\n        If a dependency on any matching packages is found, a :class:`PackagingError` is raised.\n\n        Args:\n            include (Union[List[str], str]): A string e.g. ``"my_package.my_subpackage"``, or list of strings\n                for the names of the modules to be externed. This can also be a glob-style pattern, as described in :meth:`mock`.\n\n            exclude (Union[List[str], str]): An optional pattern that excludes some patterns that match the include string.\n        '
        self.patterns[GlobGroup(include, exclude=exclude)] = _PatternInfo(_ModuleProviderAction.DENY, allow_empty=True)

    def _persistent_id(self, obj):
        if False:
            i = 10
            return i + 15
        if torch.is_storage(obj) or isinstance(obj, torch.storage.TypedStorage):
            storage: Storage
            if isinstance(obj, torch.storage.TypedStorage):
                untyped_storage = obj._untyped_storage
                storage_type_str = obj.pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage = cast(Storage, untyped_storage)
                storage_numel = obj.size()
            elif isinstance(obj, torch.UntypedStorage):
                untyped_storage = obj
                storage = cast(Storage, untyped_storage)
                storage_type = normalize_storage_type(type(storage))
                storage_numel = storage.nbytes()
            else:
                raise RuntimeError(f'storage type not recognized: {type(obj)}')
            location = location_tag(storage)
            storage_present = self.storage_context.has_storage(storage)
            storage_id = self.storage_context.get_or_add_storage(storage)
            if not storage_present:
                if storage.device.type != 'cpu':
                    storage = storage.cpu()
                num_bytes = storage.nbytes()
                self.zip_file.write_record(f'.data/{storage_id}.storage', storage.data_ptr(), num_bytes)
            return ('storage', storage_type, storage_id, location, storage_numel)
        if hasattr(obj, '__reduce_package__'):
            if _gate_torchscript_serialization and isinstance(obj, torch.jit.RecursiveScriptModule):
                raise Exception('Serializing ScriptModules directly into a package is a beta feature. To use, set global `torch.package.package_exporter._gate_torchscript_serialization` to `False`.')
            if self.serialized_reduces.get(id(obj)) is None:
                self.serialized_reduces[id(obj)] = ('reduce_package', id(obj), *obj.__reduce_package__(self))
            return self.serialized_reduces[id(obj)]
        return None

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if False:
            print('Hello World!')
        if exc_type is not None:
            self._finalize_zip()
            return
        self.close()

    def _write(self, filename, str_or_bytes):
        if False:
            while True:
                i = 10
        if filename in self._written_files:
            raise AssertionError(f"Tried to write file '{filename}', but it already exists in this archive. Please file a bug.")
        self._written_files.add(filename)
        if is_mangled(filename):
            raise AssertionError(f"Tried to save a torch.package'd module as '{filename}'. Directly saving torch.package'd modules is not allowed.")
        if isinstance(str_or_bytes, str):
            str_or_bytes = str_or_bytes.encode('utf-8')
        self.zip_file.write_record(filename, str_or_bytes, len(str_or_bytes))

    def _validate_dependency_graph(self):
        if False:
            for i in range(10):
                print('nop')
        for attrs in self.dependency_graph.nodes.values():
            if 'error' in attrs:
                raise PackagingError(self.dependency_graph, debug=self.debug)
        for (pattern, pattern_info) in self.patterns.items():
            if not pattern_info.allow_empty and (not pattern_info.was_matched):
                raise EmptyMatchError(f'Exporter did not match any modules to {pattern}, which was marked as allow_empty=False')

    def _write_mock_file(self):
        if False:
            print('Hello World!')
        if '_mock.py' not in self._written_files:
            mock_file = str(Path(__file__).parent / '_mock.py')
            self._write_source_string('_mock', _read_file(mock_file), is_package=False)

    def _execute_dependency_graph(self):
        if False:
            while True:
                i = 10
        'Takes a finalized dependency graph describing how to package all\n        modules and executes it, writing to the ZIP archive.\n        '
        self._validate_dependency_graph()
        extern_modules = []
        for (module_name, attrs) in self.dependency_graph.nodes.items():
            action = attrs['action']
            if action == _ModuleProviderAction.EXTERN:
                for hook in self._extern_hooks.values():
                    hook(self, module_name)
                extern_modules.append(module_name)
            elif action == _ModuleProviderAction.MOCK:
                for hook in self._mock_hooks.values():
                    hook(self, module_name)
                self._write_mock_file()
                is_package = hasattr(self._import_module(module_name), '__path__')
                self._write_source_string(module_name, _MOCK_IMPL, is_package)
            elif action == _ModuleProviderAction.INTERN:
                for hook in self._intern_hooks.values():
                    hook(self, module_name)
                if 'provided' not in attrs:
                    raise AssertionError(f'Module was marked `intern` but not provided: {module_name}')
                if attrs.get('is_pickle') is True:
                    continue
                is_package = attrs['is_package']
                source = attrs['source']
                self._write_source_string(module_name, source, is_package)
            elif action == _ModuleProviderAction.REPACKAGED_MOCK_MODULE:
                self._write_mock_file()
            elif action == _ModuleProviderAction.SKIP:
                continue
            else:
                raise AssertionError(f'Invalid action: {module_name}, {action}. Please report a bug to PyTorch.')
        extern_file_contents = '\n'.join(extern_modules) + '\n'
        self._write('.data/extern_modules', extern_file_contents)

    def _write_python_version(self):
        if False:
            i = 10
            return i + 15
        'Writes the python version that the package was created with to .data/python_version'
        self._write('.data/python_version', platform.python_version())

    def close(self):
        if False:
            i = 10
            return i + 15
        'Write the package to the filesystem. Any calls after :meth:`close` are now invalid.\n        It is preferable to use resource guard syntax instead::\n\n            with PackageExporter("file.zip") as e:\n                ...\n        '
        self._execute_dependency_graph()
        self._write_python_version()
        self.script_module_serializer.write_files()
        self._finalize_zip()

    def _finalize_zip(self):
        if False:
            return 10
        'Called at the very end of packaging to leave the zipfile in a closed but valid state.'
        del self.zip_file
        if self.buffer:
            self.buffer.flush()

    def _filename(self, package, resource):
        if False:
            print('Hello World!')
        package_path = package.replace('.', '/')
        resource = _normalize_path(resource)
        return f'{package_path}/{resource}'

    def _can_implicitly_extern(self, module_name: str):
        if False:
            return 10
        top_level_package_name = module_name.partition('.')[0]
        return top_level_package_name == 'torch' or (top_level_package_name not in _DISALLOWED_MODULES and is_stdlib_module(top_level_package_name))

    def dependency_graph_string(self) -> str:
        if False:
            print('Hello World!')
        'Returns digraph string representation of dependencies in package.\n\n        Returns:\n            A string representation of dependencies in package.\n        '
        return self.dependency_graph.to_dot()

    def _nodes_with_action_type(self, action: Optional[_ModuleProviderAction]) -> List[str]:
        if False:
            i = 10
            return i + 15
        result = []
        for (name, node_dict) in self.dependency_graph.nodes.items():
            node_action = node_dict.get('action', None)
            if node_action == action and 'is_pickle' not in node_dict:
                result.append(name)
        result.sort()
        return result

    def externed_modules(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return all modules that are currently externed.\n\n        Returns:\n            A list containing the names of modules which will be\n            externed in this package.\n        '
        return self._nodes_with_action_type(_ModuleProviderAction.EXTERN)

    def interned_modules(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return all modules that are currently interned.\n\n        Returns:\n            A list containing the names of modules which will be\n            interned in this package.\n        '
        return self._nodes_with_action_type(_ModuleProviderAction.INTERN)

    def mocked_modules(self) -> List[str]:
        if False:
            print('Hello World!')
        'Return all modules that are currently mocked.\n\n        Returns:\n            A list containing the names of modules which will be\n            mocked in this package.\n        '
        return self._nodes_with_action_type(_ModuleProviderAction.MOCK)

    def denied_modules(self) -> List[str]:
        if False:
            for i in range(10):
                print('nop')
        'Return all modules that are currently denied.\n\n        Returns:\n            A list containing the names of modules which will be\n            denied in this package.\n        '
        return self._nodes_with_action_type(_ModuleProviderAction.DENY)

    def get_rdeps(self, module_name: str) -> List[str]:
        if False:
            print('Hello World!')
        'Return a list of all modules which depend on the module ``module_name``.\n\n        Returns:\n            A list containing the names of modules which depend on ``module_name``.\n        '
        if module_name in self.dependency_graph._pred.keys():
            return list(self.dependency_graph._pred[module_name].keys())
        else:
            return []

    def all_paths(self, src: str, dst: str) -> str:
        if False:
            print('Hello World!')
        'Return a dot representation of the subgraph\n           that has all paths from src to dst.\n\n        Returns:\n            A dot representation containing all paths from src to dst.\n            (https://graphviz.org/doc/info/lang.html)\n        '
        return self.dependency_graph.all_paths(src, dst)
_DISALLOWED_MODULES = ['sys', 'io']
_MOCK_IMPL = "from _mock import MockedObject\ndef __getattr__(attr: str):\n    return MockedObject(__name__ + '.' + attr, _suppress_err=True)\n"

def _read_file(filename: str) -> str:
    if False:
        i = 10
        return i + 15
    with open(filename, 'rb') as f:
        b = f.read()
        return b.decode('utf-8')