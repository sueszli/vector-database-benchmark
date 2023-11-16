"""Low-level infrastructure to find modules.

This builds on fscache.py; find_sources.py builds on top of this.
"""
from __future__ import annotations
import ast
import collections
import functools
import os
import re
import subprocess
import sys
from enum import Enum, unique
from mypy.errors import CompileError
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib
from typing import Dict, Final, List, NamedTuple, Optional, Tuple, Union
from typing_extensions import TypeAlias as _TypeAlias
from mypy import pyinfo
from mypy.fscache import FileSystemCache
from mypy.nodes import MypyFile
from mypy.options import Options
from mypy.stubinfo import approved_stub_package_exists

class SearchPaths(NamedTuple):
    python_path: tuple[str, ...]
    mypy_path: tuple[str, ...]
    package_path: tuple[str, ...]
    typeshed_path: tuple[str, ...]
OnePackageDir = Tuple[str, bool]
PackageDirs = List[OnePackageDir]
StdlibVersions: _TypeAlias = Dict[str, Tuple[Tuple[int, int], Optional[Tuple[int, int]]]]
PYTHON_EXTENSIONS: Final = ['.pyi', '.py']

@unique
class ModuleNotFoundReason(Enum):
    NOT_FOUND = 0
    FOUND_WITHOUT_TYPE_HINTS = 1
    WRONG_WORKING_DIRECTORY = 2
    APPROVED_STUBS_NOT_INSTALLED = 3

    def error_message_templates(self, daemon: bool) -> tuple[str, list[str]]:
        if False:
            return 10
        doc_link = 'See https://mypy.readthedocs.io/en/stable/running_mypy.html#missing-imports'
        if self is ModuleNotFoundReason.NOT_FOUND:
            msg = 'Cannot find implementation or library stub for module named "{module}"'
            notes = [doc_link]
        elif self is ModuleNotFoundReason.WRONG_WORKING_DIRECTORY:
            msg = 'Cannot find implementation or library stub for module named "{module}"'
            notes = ['You may be running mypy in a subpackage, mypy should be run on the package root']
        elif self is ModuleNotFoundReason.FOUND_WITHOUT_TYPE_HINTS:
            msg = 'Skipping analyzing "{module}": module is installed, but missing library stubs or py.typed marker'
            notes = [doc_link]
        elif self is ModuleNotFoundReason.APPROVED_STUBS_NOT_INSTALLED:
            msg = 'Library stubs not installed for "{module}"'
            notes = ['Hint: "python3 -m pip install {stub_dist}"']
            if not daemon:
                notes.append('(or run "mypy --install-types" to install all missing stub packages)')
            notes.append(doc_link)
        else:
            assert False
        return (msg, notes)
ModuleSearchResult = Union[str, ModuleNotFoundReason]

class BuildSource:
    """A single source file."""

    def __init__(self, path: str | None, module: str | None, text: str | None=None, base_dir: str | None=None, followed: bool=False) -> None:
        if False:
            while True:
                i = 10
        self.path = path
        self.module = module or '__main__'
        self.text = text
        self.base_dir = base_dir
        self.followed = followed

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return 'BuildSource(path={!r}, module={!r}, has_text={}, base_dir={!r}, followed={})'.format(self.path, self.module, self.text is not None, self.base_dir, self.followed)

class BuildSourceSet:
    """Helper to efficiently test a file's membership in a set of build sources."""

    def __init__(self, sources: list[BuildSource]) -> None:
        if False:
            return 10
        self.source_text_present = False
        self.source_modules: dict[str, str] = {}
        self.source_paths: set[str] = set()
        for source in sources:
            if source.text is not None:
                self.source_text_present = True
            if source.path:
                self.source_paths.add(source.path)
            if source.module:
                self.source_modules[source.module] = source.path or ''

    def is_source(self, file: MypyFile) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return file.path and file.path in self.source_paths or file._fullname in self.source_modules or self.source_text_present

class FindModuleCache:
    """Module finder with integrated cache.

    Module locations and some intermediate results are cached internally
    and can be cleared with the clear() method.

    All file system accesses are performed through a FileSystemCache,
    which is not ever cleared by this class. If necessary it must be
    cleared by client code.
    """

    def __init__(self, search_paths: SearchPaths, fscache: FileSystemCache | None, options: Options | None, stdlib_py_versions: StdlibVersions | None=None, source_set: BuildSourceSet | None=None) -> None:
        if False:
            return 10
        self.search_paths = search_paths
        self.source_set = source_set
        self.fscache = fscache or FileSystemCache()
        self.initial_components: dict[tuple[str, ...], dict[str, list[str]]] = {}
        self.results: dict[str, ModuleSearchResult] = {}
        self.ns_ancestors: dict[str, str] = {}
        self.options = options
        custom_typeshed_dir = None
        if options:
            custom_typeshed_dir = options.custom_typeshed_dir
        self.stdlib_py_versions = stdlib_py_versions or load_stdlib_py_versions(custom_typeshed_dir)

    def clear(self) -> None:
        if False:
            return 10
        self.results.clear()
        self.initial_components.clear()
        self.ns_ancestors.clear()

    def find_module_via_source_set(self, id: str) -> ModuleSearchResult | None:
        if False:
            i = 10
            return i + 15
        'Fast path to find modules by looking through the input sources\n\n        This is only used when --fast-module-lookup is passed on the command line.'
        if not self.source_set:
            return None
        p = self.source_set.source_modules.get(id, None)
        if p and self.fscache.isfile(p):
            d = os.path.dirname(p)
            for _ in range(id.count('.')):
                if not any((self.fscache.isfile(os.path.join(d, '__init__' + x)) for x in PYTHON_EXTENSIONS)):
                    return None
                d = os.path.dirname(d)
            return p
        idx = id.rfind('.')
        if idx != -1:
            parent = self.find_module_via_source_set(id[:idx])
            if parent is None or not isinstance(parent, str):
                return None
            (basename, ext) = os.path.splitext(parent)
            if not any((parent.endswith('__init__' + x) for x in PYTHON_EXTENSIONS)) and (ext in PYTHON_EXTENSIONS and (not self.fscache.isdir(basename))):
                return ModuleNotFoundReason.NOT_FOUND
        return None

    def find_lib_path_dirs(self, id: str, lib_path: tuple[str, ...]) -> PackageDirs:
        if False:
            print('Hello World!')
        'Find which elements of a lib_path have the directory a module needs to exist.\n\n        This is run for the python_path, mypy_path, and typeshed_path search paths.\n        '
        components = id.split('.')
        dir_chain = os.sep.join(components[:-1])
        dirs = []
        for pathitem in self.get_toplevel_possibilities(lib_path, components[0]):
            dir = os.path.normpath(os.path.join(pathitem, dir_chain))
            if self.fscache.isdir(dir):
                dirs.append((dir, True))
        return dirs

    def get_toplevel_possibilities(self, lib_path: tuple[str, ...], id: str) -> list[str]:
        if False:
            i = 10
            return i + 15
        'Find which elements of lib_path could contain a particular top-level module.\n\n        In practice, almost all modules can be routed to the correct entry in\n        lib_path by looking at just the first component of the module name.\n\n        We take advantage of this by enumerating the contents of all of the\n        directories on the lib_path and building a map of which entries in\n        the lib_path could contain each potential top-level module that appears.\n        '
        if lib_path in self.initial_components:
            return self.initial_components[lib_path].get(id, [])
        components: dict[str, list[str]] = {}
        for dir in lib_path:
            try:
                contents = self.fscache.listdir(dir)
            except OSError:
                contents = []
            for name in contents:
                name = os.path.splitext(name)[0]
                components.setdefault(name, []).append(dir)
        self.initial_components[lib_path] = components
        return components.get(id, [])

    def find_module(self, id: str, *, fast_path: bool=False) -> ModuleSearchResult:
        if False:
            while True:
                i = 10
        "Return the path of the module source file or why it wasn't found.\n\n        If fast_path is True, prioritize performance over generating detailed\n        error descriptions.\n        "
        if id not in self.results:
            top_level = id.partition('.')[0]
            use_typeshed = True
            if id in self.stdlib_py_versions:
                use_typeshed = self._typeshed_has_version(id)
            elif top_level in self.stdlib_py_versions:
                use_typeshed = self._typeshed_has_version(top_level)
            self.results[id] = self._find_module(id, use_typeshed)
            if not (fast_path or (self.options is not None and self.options.fast_module_lookup)) and self.results[id] is ModuleNotFoundReason.NOT_FOUND and self._can_find_module_in_parent_dir(id):
                self.results[id] = ModuleNotFoundReason.WRONG_WORKING_DIRECTORY
        return self.results[id]

    def _typeshed_has_version(self, module: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if not self.options:
            return True
        version = typeshed_py_version(self.options)
        (min_version, max_version) = self.stdlib_py_versions[module]
        return version >= min_version and (max_version is None or version <= max_version)

    def _find_module_non_stub_helper(self, components: list[str], pkg_dir: str) -> OnePackageDir | ModuleNotFoundReason:
        if False:
            print('Hello World!')
        plausible_match = False
        dir_path = pkg_dir
        for (index, component) in enumerate(components):
            dir_path = os.path.join(dir_path, component)
            if self.fscache.isfile(os.path.join(dir_path, 'py.typed')):
                return (os.path.join(pkg_dir, *components[:-1]), index == 0)
            elif not plausible_match and (self.fscache.isdir(dir_path) or self.fscache.isfile(dir_path + '.py')):
                plausible_match = True
            if not self.fscache.isdir(dir_path):
                break
        for i in range(len(components), 0, -1):
            if approved_stub_package_exists('.'.join(components[:i])):
                return ModuleNotFoundReason.APPROVED_STUBS_NOT_INSTALLED
        if plausible_match:
            return ModuleNotFoundReason.FOUND_WITHOUT_TYPE_HINTS
        else:
            return ModuleNotFoundReason.NOT_FOUND

    def _update_ns_ancestors(self, components: list[str], match: tuple[str, bool]) -> None:
        if False:
            print('Hello World!')
        (path, verify) = match
        for i in range(1, len(components)):
            pkg_id = '.'.join(components[:-i])
            if pkg_id not in self.ns_ancestors and self.fscache.isdir(path):
                self.ns_ancestors[pkg_id] = path
            path = os.path.dirname(path)

    def _can_find_module_in_parent_dir(self, id: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Test if a module can be found by checking the parent directories\n        of the current working directory.\n        '
        working_dir = os.getcwd()
        parent_search = FindModuleCache(SearchPaths((), (), (), ()), self.fscache, self.options, stdlib_py_versions=self.stdlib_py_versions)
        while any((is_init_file(file) for file in os.listdir(working_dir))):
            working_dir = os.path.dirname(working_dir)
            parent_search.search_paths = SearchPaths((working_dir,), (), (), ())
            if not isinstance(parent_search._find_module(id, False), ModuleNotFoundReason):
                return True
        return False

    def _find_module(self, id: str, use_typeshed: bool) -> ModuleSearchResult:
        if False:
            i = 10
            return i + 15
        fscache = self.fscache
        p = self.find_module_via_source_set(id) if self.options is not None and self.options.fast_module_lookup else None
        if p:
            return p
        components = id.split('.')
        dir_chain = os.sep.join(components[:-1])
        third_party_inline_dirs: PackageDirs = []
        third_party_stubs_dirs: PackageDirs = []
        found_possible_third_party_missing_type_hints = False
        need_installed_stubs = False
        for pkg_dir in self.search_paths.package_path:
            stub_name = components[0] + '-stubs'
            stub_dir = os.path.join(pkg_dir, stub_name)
            if fscache.isdir(stub_dir) and self._is_compatible_stub_package(stub_dir):
                stub_typed_file = os.path.join(stub_dir, 'py.typed')
                stub_components = [stub_name] + components[1:]
                path = os.path.join(pkg_dir, *stub_components[:-1])
                if fscache.isdir(path):
                    if fscache.isfile(stub_typed_file):
                        if fscache.read(stub_typed_file).decode().strip() == 'partial':
                            runtime_path = os.path.join(pkg_dir, dir_chain)
                            third_party_inline_dirs.append((runtime_path, True))
                            third_party_stubs_dirs.append((path, False))
                        else:
                            third_party_stubs_dirs.append((path, True))
                    else:
                        third_party_stubs_dirs.append((path, True))
            non_stub_match = self._find_module_non_stub_helper(components, pkg_dir)
            if isinstance(non_stub_match, ModuleNotFoundReason):
                if non_stub_match is ModuleNotFoundReason.FOUND_WITHOUT_TYPE_HINTS:
                    found_possible_third_party_missing_type_hints = True
                elif non_stub_match is ModuleNotFoundReason.APPROVED_STUBS_NOT_INSTALLED:
                    need_installed_stubs = True
            else:
                third_party_inline_dirs.append(non_stub_match)
                self._update_ns_ancestors(components, non_stub_match)
        if self.options and self.options.use_builtins_fixtures:
            third_party_inline_dirs.clear()
            third_party_stubs_dirs.clear()
            found_possible_third_party_missing_type_hints = False
        python_mypy_path = self.search_paths.mypy_path + self.search_paths.python_path
        candidate_base_dirs = self.find_lib_path_dirs(id, python_mypy_path)
        if use_typeshed:
            candidate_base_dirs += self.find_lib_path_dirs(id, self.search_paths.typeshed_path)
        candidate_base_dirs += third_party_stubs_dirs + third_party_inline_dirs
        seplast = os.sep + components[-1]
        sepinit = os.sep + '__init__'
        near_misses = []
        for (base_dir, verify) in candidate_base_dirs:
            base_path = base_dir + seplast
            has_init = False
            dir_prefix = base_dir
            for _ in range(len(components) - 1):
                dir_prefix = os.path.dirname(dir_prefix)
            for extension in PYTHON_EXTENSIONS:
                path = base_path + sepinit + extension
                path_stubs = base_path + '-stubs' + sepinit + extension
                if fscache.isfile_case(path, dir_prefix):
                    has_init = True
                    if verify and (not verify_module(fscache, id, path, dir_prefix)):
                        near_misses.append((path, dir_prefix))
                        continue
                    return path
                elif fscache.isfile_case(path_stubs, dir_prefix):
                    if verify and (not verify_module(fscache, id, path_stubs, dir_prefix)):
                        near_misses.append((path_stubs, dir_prefix))
                        continue
                    return path_stubs
            if self.options and self.options.namespace_packages:
                if not has_init and fscache.exists_case(base_path, dir_prefix) and (not fscache.isfile_case(base_path, dir_prefix)):
                    near_misses.append((base_path, dir_prefix))
            for extension in PYTHON_EXTENSIONS:
                path = base_path + extension
                if fscache.isfile_case(path, dir_prefix):
                    if verify and (not verify_module(fscache, id, path, dir_prefix)):
                        near_misses.append((path, dir_prefix))
                        continue
                    return path
        if self.options and self.options.namespace_packages and near_misses:
            levels = [highest_init_level(fscache, id, path, dir_prefix) for (path, dir_prefix) in near_misses]
            index = levels.index(max(levels))
            return near_misses[index][0]
        ancestor = self.ns_ancestors.get(id)
        if ancestor is not None:
            return ancestor
        if need_installed_stubs:
            return ModuleNotFoundReason.APPROVED_STUBS_NOT_INSTALLED
        elif found_possible_third_party_missing_type_hints:
            return ModuleNotFoundReason.FOUND_WITHOUT_TYPE_HINTS
        else:
            return ModuleNotFoundReason.NOT_FOUND

    def _is_compatible_stub_package(self, stub_dir: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Does a stub package support the target Python version?\n\n        Stub packages may contain a metadata file which specifies\n        whether the stubs are compatible with Python 2 and 3.\n        '
        metadata_fnam = os.path.join(stub_dir, 'METADATA.toml')
        if not os.path.isfile(metadata_fnam):
            return True
        with open(metadata_fnam, 'rb') as f:
            metadata = tomllib.load(f)
        return bool(metadata.get('python3', True))

    def find_modules_recursive(self, module: str) -> list[BuildSource]:
        if False:
            i = 10
            return i + 15
        module_path = self.find_module(module)
        if isinstance(module_path, ModuleNotFoundReason):
            return []
        sources = [BuildSource(module_path, module, None)]
        package_path = None
        if is_init_file(module_path):
            package_path = os.path.dirname(module_path)
        elif self.fscache.isdir(module_path):
            package_path = module_path
        if package_path is None:
            return sources
        seen: set[str] = set()
        names = sorted(self.fscache.listdir(package_path))
        for name in names:
            if name in ('__pycache__', 'site-packages', 'node_modules') or name.startswith('.'):
                continue
            subpath = os.path.join(package_path, name)
            if self.options and matches_exclude(subpath, self.options.exclude, self.fscache, self.options.verbosity >= 2):
                continue
            if self.fscache.isdir(subpath):
                if self.options and self.options.namespace_packages or (self.fscache.isfile(os.path.join(subpath, '__init__.py')) or self.fscache.isfile(os.path.join(subpath, '__init__.pyi'))):
                    seen.add(name)
                    sources.extend(self.find_modules_recursive(module + '.' + name))
            else:
                (stem, suffix) = os.path.splitext(name)
                if stem == '__init__':
                    continue
                if stem not in seen and '.' not in stem and (suffix in PYTHON_EXTENSIONS):
                    seen.add(stem)
                    sources.extend(self.find_modules_recursive(module + '.' + stem))
        return sources

def matches_exclude(subpath: str, excludes: list[str], fscache: FileSystemCache, verbose: bool) -> bool:
    if False:
        i = 10
        return i + 15
    if not excludes:
        return False
    subpath_str = os.path.relpath(subpath).replace(os.sep, '/')
    if fscache.isdir(subpath):
        subpath_str += '/'
    for exclude in excludes:
        if re.search(exclude, subpath_str):
            if verbose:
                print(f'TRACE: Excluding {subpath_str} (matches pattern {exclude})', file=sys.stderr)
            return True
    return False

def is_init_file(path: str) -> bool:
    if False:
        for i in range(10):
            print('nop')
    return os.path.basename(path) in ('__init__.py', '__init__.pyi')

def verify_module(fscache: FileSystemCache, id: str, path: str, prefix: str) -> bool:
    if False:
        print('Hello World!')
    'Check that all packages containing id have a __init__ file.'
    if is_init_file(path):
        path = os.path.dirname(path)
    for i in range(id.count('.')):
        path = os.path.dirname(path)
        if not any((fscache.isfile_case(os.path.join(path, f'__init__{extension}'), prefix) for extension in PYTHON_EXTENSIONS)):
            return False
    return True

def highest_init_level(fscache: FileSystemCache, id: str, path: str, prefix: str) -> int:
    if False:
        print('Hello World!')
    'Compute the highest level where an __init__ file is found.'
    if is_init_file(path):
        path = os.path.dirname(path)
    level = 0
    for i in range(id.count('.')):
        path = os.path.dirname(path)
        if any((fscache.isfile_case(os.path.join(path, f'__init__{extension}'), prefix) for extension in PYTHON_EXTENSIONS)):
            level = i + 1
    return level

def mypy_path() -> list[str]:
    if False:
        i = 10
        return i + 15
    path_env = os.getenv('MYPYPATH')
    if not path_env:
        return []
    return path_env.split(os.pathsep)

def default_lib_path(data_dir: str, pyversion: tuple[int, int], custom_typeshed_dir: str | None) -> list[str]:
    if False:
        while True:
            i = 10
    'Return default standard library search paths.'
    path: list[str] = []
    if custom_typeshed_dir:
        typeshed_dir = os.path.join(custom_typeshed_dir, 'stdlib')
        mypy_extensions_dir = os.path.join(custom_typeshed_dir, 'stubs', 'mypy-extensions')
        versions_file = os.path.join(typeshed_dir, 'VERSIONS')
        if not os.path.isdir(typeshed_dir) or not os.path.isfile(versions_file):
            print('error: --custom-typeshed-dir does not point to a valid typeshed ({})'.format(custom_typeshed_dir))
            sys.exit(2)
    else:
        auto = os.path.join(data_dir, 'stubs-auto')
        if os.path.isdir(auto):
            data_dir = auto
        typeshed_dir = os.path.join(data_dir, 'typeshed', 'stdlib')
        mypy_extensions_dir = os.path.join(data_dir, 'typeshed', 'stubs', 'mypy-extensions')
    path.append(typeshed_dir)
    path.append(mypy_extensions_dir)
    if sys.platform != 'win32':
        path.append('/usr/local/lib/mypy')
    if not path:
        print('Could not resolve typeshed subdirectories. Your mypy install is broken.\nPython executable is located at {}.\nMypy located at {}'.format(sys.executable, data_dir), file=sys.stderr)
        sys.exit(1)
    return path

@functools.lru_cache(maxsize=None)
def get_search_dirs(python_executable: str | None) -> tuple[list[str], list[str]]:
    if False:
        print('Hello World!')
    'Find package directories for given python.\n\n    This runs a subprocess call, which generates a list of the directories in sys.path.\n    To avoid repeatedly calling a subprocess (which can be slow!) we\n    lru_cache the results.\n    '
    if python_executable is None:
        return ([], [])
    elif python_executable == sys.executable:
        (sys_path, site_packages) = pyinfo.getsearchdirs()
    else:
        env = {**dict(os.environ), 'PYTHONSAFEPATH': '1'}
        try:
            (sys_path, site_packages) = ast.literal_eval(subprocess.check_output([python_executable, pyinfo.__file__, 'getsearchdirs'], env=env, stderr=subprocess.PIPE).decode())
        except subprocess.CalledProcessError as err:
            print(err.stderr)
            print(err.stdout)
            raise
        except OSError as err:
            reason = os.strerror(err.errno)
            raise CompileError([f"mypy: Invalid python executable '{python_executable}': {reason}"]) from err
    return (sys_path, site_packages)

def compute_search_paths(sources: list[BuildSource], options: Options, data_dir: str, alt_lib_path: str | None=None) -> SearchPaths:
    if False:
        print('Hello World!')
    'Compute the search paths as specified in PEP 561.\n\n    There are the following 4 members created:\n    - User code (from `sources`)\n    - MYPYPATH (set either via config or environment variable)\n    - installed package directories (which will later be split into stub-only and inline)\n    - typeshed\n    '
    lib_path = collections.deque(default_lib_path(data_dir, options.python_version, custom_typeshed_dir=options.custom_typeshed_dir))
    if options.use_builtins_fixtures:
        root_dir = os.getenv('MYPY_TEST_PREFIX', None)
        if not root_dir:
            root_dir = os.path.dirname(os.path.dirname(__file__))
        lib_path.appendleft(os.path.join(root_dir, 'test-data', 'unit', 'lib-stub'))
    python_path: list[str] = []
    if not alt_lib_path:
        for source in sources:
            if source.base_dir:
                dir = source.base_dir
                if dir not in python_path:
                    python_path.append(dir)
        if options.bazel:
            dir = '.'
        else:
            dir = os.getcwd()
        if dir not in lib_path:
            python_path.insert(0, dir)
    mypypath = mypy_path()
    mypypath.extend(options.mypy_path)
    if alt_lib_path:
        mypypath.insert(0, alt_lib_path)
    (sys_path, site_packages) = get_search_dirs(options.python_executable)
    for site in site_packages:
        assert site not in lib_path
        if site in mypypath or any((p.startswith(site + os.path.sep) for p in mypypath)) or (os.path.altsep and any((p.startswith(site + os.path.altsep) for p in mypypath))):
            print(f'{site} is in the MYPYPATH. Please remove it.', file=sys.stderr)
            print('See https://mypy.readthedocs.io/en/stable/running_mypy.html#how-mypy-handles-imports for more info', file=sys.stderr)
            sys.exit(1)
    return SearchPaths(python_path=tuple(reversed(python_path)), mypy_path=tuple(mypypath), package_path=tuple(sys_path + site_packages), typeshed_path=tuple(lib_path))

def load_stdlib_py_versions(custom_typeshed_dir: str | None) -> StdlibVersions:
    if False:
        return 10
    "Return dict with minimum and maximum Python versions of stdlib modules.\n\n    The contents look like\n    {..., 'secrets': ((3, 6), None), 'symbol': ((2, 7), (3, 9)), ...}\n\n    None means there is no maximum version.\n    "
    typeshed_dir = custom_typeshed_dir or os.path.join(os.path.dirname(__file__), 'typeshed')
    stdlib_dir = os.path.join(typeshed_dir, 'stdlib')
    result = {}
    versions_path = os.path.join(stdlib_dir, 'VERSIONS')
    assert os.path.isfile(versions_path), (custom_typeshed_dir, versions_path, __file__)
    with open(versions_path) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if line == '':
                continue
            (module, version_range) = line.split(':')
            versions = version_range.split('-')
            min_version = parse_version(versions[0])
            max_version = parse_version(versions[1]) if len(versions) >= 2 and versions[1].strip() else None
            result[module] = (min_version, max_version)
    return result

def parse_version(version: str) -> tuple[int, int]:
    if False:
        i = 10
        return i + 15
    (major, minor) = version.strip().split('.')
    return (int(major), int(minor))

def typeshed_py_version(options: Options) -> tuple[int, int]:
    if False:
        i = 10
        return i + 15
    'Return Python version used for checking whether module supports typeshed.'
    return max(options.python_version, (3, 7))