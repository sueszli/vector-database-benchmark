import atexit
import contextlib
import fnmatch
import importlib.util
import itertools
import os
import shutil
import sys
import types
import uuid
import warnings
from enum import Enum
from errno import EBADF
from errno import ELOOP
from errno import ENOENT
from errno import ENOTDIR
from functools import partial
from os.path import expanduser
from os.path import expandvars
from os.path import isabs
from os.path import sep
from pathlib import Path
from pathlib import PurePath
from posixpath import sep as posix_sep
from types import ModuleType
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import TypeVar
from typing import Union
from _pytest.compat import assert_never
from _pytest.outcomes import skip
from _pytest.warning_types import PytestWarning
LOCK_TIMEOUT = 60 * 60 * 24 * 3
_AnyPurePath = TypeVar('_AnyPurePath', bound=PurePath)
_IGNORED_ERRORS = (ENOENT, ENOTDIR, EBADF, ELOOP)
_IGNORED_WINERRORS = (21, 1921)

def _ignore_error(exception):
    if False:
        print('Hello World!')
    return getattr(exception, 'errno', None) in _IGNORED_ERRORS or getattr(exception, 'winerror', None) in _IGNORED_WINERRORS

def get_lock_path(path: _AnyPurePath) -> _AnyPurePath:
    if False:
        print('Hello World!')
    return path.joinpath('.lock')

def on_rm_rf_error(func, path: str, excinfo: Union[BaseException, Tuple[Type[BaseException], BaseException, Optional[types.TracebackType]]], *, start_path: Path) -> bool:
    if False:
        return 10
    'Handle known read-only errors during rmtree.\n\n    The returned value is used only by our own tests.\n    '
    if isinstance(excinfo, BaseException):
        exc = excinfo
    else:
        exc = excinfo[1]
    if isinstance(exc, FileNotFoundError):
        return False
    if not isinstance(exc, PermissionError):
        warnings.warn(PytestWarning(f'(rm_rf) error removing {path}\n{type(exc)}: {exc}'))
        return False
    if func not in (os.rmdir, os.remove, os.unlink):
        if func not in (os.open,):
            warnings.warn(PytestWarning('(rm_rf) unknown function {} when removing {}:\n{}: {}'.format(func, path, type(exc), exc)))
        return False
    import stat

    def chmod_rw(p: str) -> None:
        if False:
            while True:
                i = 10
        mode = os.stat(p).st_mode
        os.chmod(p, mode | stat.S_IRUSR | stat.S_IWUSR)
    p = Path(path)
    if p.is_file():
        for parent in p.parents:
            chmod_rw(str(parent))
            if parent == start_path:
                break
    chmod_rw(str(path))
    func(path)
    return True

def ensure_extended_length_path(path: Path) -> Path:
    if False:
        for i in range(10):
            print('nop')
    'Get the extended-length version of a path (Windows).\n\n    On Windows, by default, the maximum length of a path (MAX_PATH) is 260\n    characters, and operations on paths longer than that fail. But it is possible\n    to overcome this by converting the path to "extended-length" form before\n    performing the operation:\n    https://docs.microsoft.com/en-us/windows/win32/fileio/naming-a-file#maximum-path-length-limitation\n\n    On Windows, this function returns the extended-length absolute version of path.\n    On other platforms it returns path unchanged.\n    '
    if sys.platform.startswith('win32'):
        path = path.resolve()
        path = Path(get_extended_length_path_str(str(path)))
    return path

def get_extended_length_path_str(path: str) -> str:
    if False:
        i = 10
        return i + 15
    'Convert a path to a Windows extended length path.'
    long_path_prefix = '\\\\?\\'
    unc_long_path_prefix = '\\\\?\\UNC\\'
    if path.startswith((long_path_prefix, unc_long_path_prefix)):
        return path
    if path.startswith('\\\\'):
        return unc_long_path_prefix + path[2:]
    return long_path_prefix + path

def rm_rf(path: Path) -> None:
    if False:
        i = 10
        return i + 15
    'Remove the path contents recursively, even if some elements\n    are read-only.'
    path = ensure_extended_length_path(path)
    onerror = partial(on_rm_rf_error, start_path=path)
    if sys.version_info >= (3, 12):
        shutil.rmtree(str(path), onexc=onerror)
    else:
        shutil.rmtree(str(path), onerror=onerror)

def find_prefixed(root: Path, prefix: str) -> Iterator[Path]:
    if False:
        print('Hello World!')
    'Find all elements in root that begin with the prefix, case insensitive.'
    l_prefix = prefix.lower()
    for x in root.iterdir():
        if x.name.lower().startswith(l_prefix):
            yield x

def extract_suffixes(iter: Iterable[PurePath], prefix: str) -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    'Return the parts of the paths following the prefix.\n\n    :param iter: Iterator over path names.\n    :param prefix: Expected prefix of the path names.\n    '
    p_len = len(prefix)
    for p in iter:
        yield p.name[p_len:]

def find_suffixes(root: Path, prefix: str) -> Iterator[str]:
    if False:
        return 10
    'Combine find_prefixes and extract_suffixes.'
    return extract_suffixes(find_prefixed(root, prefix), prefix)

def parse_num(maybe_num) -> int:
    if False:
        i = 10
        return i + 15
    'Parse number path suffixes, returns -1 on error.'
    try:
        return int(maybe_num)
    except ValueError:
        return -1

def _force_symlink(root: Path, target: Union[str, PurePath], link_to: Union[str, Path]) -> None:
    if False:
        while True:
            i = 10
    "Helper to create the current symlink.\n\n    It's full of race conditions that are reasonably OK to ignore\n    for the context of best effort linking to the latest test run.\n\n    The presumption being that in case of much parallelism\n    the inaccuracy is going to be acceptable.\n    "
    current_symlink = root.joinpath(target)
    try:
        current_symlink.unlink()
    except OSError:
        pass
    try:
        current_symlink.symlink_to(link_to)
    except Exception:
        pass

def make_numbered_dir(root: Path, prefix: str, mode: int=448) -> Path:
    if False:
        print('Hello World!')
    'Create a directory with an increased number as suffix for the given prefix.'
    for i in range(10):
        max_existing = max(map(parse_num, find_suffixes(root, prefix)), default=-1)
        new_number = max_existing + 1
        new_path = root.joinpath(f'{prefix}{new_number}')
        try:
            new_path.mkdir(mode=mode)
        except Exception:
            pass
        else:
            _force_symlink(root, prefix + 'current', new_path)
            return new_path
    else:
        raise OSError('could not create numbered dir with prefix {prefix} in {root} after 10 tries'.format(prefix=prefix, root=root))

def create_cleanup_lock(p: Path) -> Path:
    if False:
        while True:
            i = 10
    'Create a lock to prevent premature folder cleanup.'
    lock_path = get_lock_path(p)
    try:
        fd = os.open(str(lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 420)
    except FileExistsError as e:
        raise OSError(f'cannot create lockfile in {p}') from e
    else:
        pid = os.getpid()
        spid = str(pid).encode()
        os.write(fd, spid)
        os.close(fd)
        if not lock_path.is_file():
            raise OSError('lock path got renamed after successful creation')
        return lock_path

def register_cleanup_lock_removal(lock_path: Path, register=atexit.register):
    if False:
        i = 10
        return i + 15
    'Register a cleanup function for removing a lock, by default on atexit.'
    pid = os.getpid()

    def cleanup_on_exit(lock_path: Path=lock_path, original_pid: int=pid) -> None:
        if False:
            for i in range(10):
                print('nop')
        current_pid = os.getpid()
        if current_pid != original_pid:
            return
        try:
            lock_path.unlink()
        except OSError:
            pass
    return register(cleanup_on_exit)

def maybe_delete_a_numbered_dir(path: Path) -> None:
    if False:
        print('Hello World!')
    'Remove a numbered directory if its lock can be obtained and it does\n    not seem to be in use.'
    path = ensure_extended_length_path(path)
    lock_path = None
    try:
        lock_path = create_cleanup_lock(path)
        parent = path.parent
        garbage = parent.joinpath(f'garbage-{uuid.uuid4()}')
        path.rename(garbage)
        rm_rf(garbage)
    except OSError:
        return
    finally:
        if lock_path is not None:
            try:
                lock_path.unlink()
            except OSError:
                pass

def ensure_deletable(path: Path, consider_lock_dead_if_created_before: float) -> bool:
    if False:
        while True:
            i = 10
    'Check if `path` is deletable based on whether the lock file is expired.'
    if path.is_symlink():
        return False
    lock = get_lock_path(path)
    try:
        if not lock.is_file():
            return True
    except OSError:
        return False
    try:
        lock_time = lock.stat().st_mtime
    except Exception:
        return False
    else:
        if lock_time < consider_lock_dead_if_created_before:
            with contextlib.suppress(OSError):
                lock.unlink()
                return True
        return False

def try_cleanup(path: Path, consider_lock_dead_if_created_before: float) -> None:
    if False:
        i = 10
        return i + 15
    "Try to cleanup a folder if we can ensure it's deletable."
    if ensure_deletable(path, consider_lock_dead_if_created_before):
        maybe_delete_a_numbered_dir(path)

def cleanup_candidates(root: Path, prefix: str, keep: int) -> Iterator[Path]:
    if False:
        for i in range(10):
            print('nop')
    'List candidates for numbered directories to be removed - follows py.path.'
    max_existing = max(map(parse_num, find_suffixes(root, prefix)), default=-1)
    max_delete = max_existing - keep
    paths = find_prefixed(root, prefix)
    (paths, paths2) = itertools.tee(paths)
    numbers = map(parse_num, extract_suffixes(paths2, prefix))
    for (path, number) in zip(paths, numbers):
        if number <= max_delete:
            yield path

def cleanup_dead_symlinks(root: Path):
    if False:
        i = 10
        return i + 15
    for left_dir in root.iterdir():
        if left_dir.is_symlink():
            if not left_dir.resolve().exists():
                left_dir.unlink()

def cleanup_numbered_dir(root: Path, prefix: str, keep: int, consider_lock_dead_if_created_before: float) -> None:
    if False:
        while True:
            i = 10
    'Cleanup for lock driven numbered directories.'
    if not root.exists():
        return
    for path in cleanup_candidates(root, prefix, keep):
        try_cleanup(path, consider_lock_dead_if_created_before)
    for path in root.glob('garbage-*'):
        try_cleanup(path, consider_lock_dead_if_created_before)
    cleanup_dead_symlinks(root)

def make_numbered_dir_with_cleanup(root: Path, prefix: str, keep: int, lock_timeout: float, mode: int) -> Path:
    if False:
        while True:
            i = 10
    'Create a numbered dir with a cleanup lock and remove old ones.'
    e = None
    for i in range(10):
        try:
            p = make_numbered_dir(root, prefix, mode)
            if keep != 0:
                lock_path = create_cleanup_lock(p)
                register_cleanup_lock_removal(lock_path)
        except Exception as exc:
            e = exc
        else:
            consider_lock_dead_if_created_before = p.stat().st_mtime - lock_timeout
            atexit.register(cleanup_numbered_dir, root, prefix, keep, consider_lock_dead_if_created_before)
            return p
    assert e is not None
    raise e

def resolve_from_str(input: str, rootpath: Path) -> Path:
    if False:
        return 10
    input = expanduser(input)
    input = expandvars(input)
    if isabs(input):
        return Path(input)
    else:
        return rootpath.joinpath(input)

def fnmatch_ex(pattern: str, path: Union[str, 'os.PathLike[str]']) -> bool:
    if False:
        i = 10
        return i + 15
    'A port of FNMatcher from py.path.common which works with PurePath() instances.\n\n    The difference between this algorithm and PurePath.match() is that the\n    latter matches "**" glob expressions for each part of the path, while\n    this algorithm uses the whole path instead.\n\n    For example:\n        "tests/foo/bar/doc/test_foo.py" matches pattern "tests/**/doc/test*.py"\n        with this algorithm, but not with PurePath.match().\n\n    This algorithm was ported to keep backward-compatibility with existing\n    settings which assume paths match according this logic.\n\n    References:\n    * https://bugs.python.org/issue29249\n    * https://bugs.python.org/issue34731\n    '
    path = PurePath(path)
    iswin32 = sys.platform.startswith('win')
    if iswin32 and sep not in pattern and (posix_sep in pattern):
        pattern = pattern.replace(posix_sep, sep)
    if sep not in pattern:
        name = path.name
    else:
        name = str(path)
        if path.is_absolute() and (not os.path.isabs(pattern)):
            pattern = f'*{os.sep}{pattern}'
    return fnmatch.fnmatch(name, pattern)

def parts(s: str) -> Set[str]:
    if False:
        print('Hello World!')
    parts = s.split(sep)
    return {sep.join(parts[:i + 1]) or sep for i in range(len(parts))}

def symlink_or_skip(src, dst, **kwargs):
    if False:
        while True:
            i = 10
    'Make a symlink, or skip the test in case symlinks are not supported.'
    try:
        os.symlink(str(src), str(dst), **kwargs)
    except OSError as e:
        skip(f'symlinks not supported: {e}')

class ImportMode(Enum):
    """Possible values for `mode` parameter of `import_path`."""
    prepend = 'prepend'
    append = 'append'
    importlib = 'importlib'

class ImportPathMismatchError(ImportError):
    """Raised on import_path() if there is a mismatch of __file__'s.

    This can happen when `import_path` is called multiple times with different filenames that has
    the same basename but reside in packages
    (for example "/tests1/test_foo.py" and "/tests2/test_foo.py").
    """

def import_path(p: Union[str, 'os.PathLike[str]'], *, mode: Union[str, ImportMode]=ImportMode.prepend, root: Path) -> ModuleType:
    if False:
        while True:
            i = 10
    'Import and return a module from the given path, which can be a file (a module) or\n    a directory (a package).\n\n    The import mechanism used is controlled by the `mode` parameter:\n\n    * `mode == ImportMode.prepend`: the directory containing the module (or package, taking\n      `__init__.py` files into account) will be put at the *start* of `sys.path` before\n      being imported with `importlib.import_module`.\n\n    * `mode == ImportMode.append`: same as `prepend`, but the directory will be appended\n      to the end of `sys.path`, if not already in `sys.path`.\n\n    * `mode == ImportMode.importlib`: uses more fine control mechanisms provided by `importlib`\n      to import the module, which avoids having to muck with `sys.path` at all. It effectively\n      allows having same-named test modules in different places.\n\n    :param root:\n        Used as an anchor when mode == ImportMode.importlib to obtain\n        a unique name for the module being imported so it can safely be stored\n        into ``sys.modules``.\n\n    :raises ImportPathMismatchError:\n        If after importing the given `path` and the module `__file__`\n        are different. Only raised in `prepend` and `append` modes.\n    '
    mode = ImportMode(mode)
    path = Path(p)
    if not path.exists():
        raise ImportError(path)
    if mode is ImportMode.importlib:
        module_name = module_name_from_path(path, root)
        with contextlib.suppress(KeyError):
            return sys.modules[module_name]
        for meta_importer in sys.meta_path:
            spec = meta_importer.find_spec(module_name, [str(path.parent)])
            if spec is not None:
                break
        else:
            spec = importlib.util.spec_from_file_location(module_name, str(path))
        if spec is None:
            raise ImportError(f"Can't find module {module_name} at location {path}")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        insert_missing_modules(sys.modules, module_name)
        return mod
    pkg_path = resolve_package_path(path)
    if pkg_path is not None:
        pkg_root = pkg_path.parent
        names = list(path.with_suffix('').relative_to(pkg_root).parts)
        if names[-1] == '__init__':
            names.pop()
        module_name = '.'.join(names)
    else:
        pkg_root = path.parent
        module_name = path.stem
    if mode is ImportMode.append:
        if str(pkg_root) not in sys.path:
            sys.path.append(str(pkg_root))
    elif mode is ImportMode.prepend:
        if str(pkg_root) != sys.path[0]:
            sys.path.insert(0, str(pkg_root))
    else:
        assert_never(mode)
    importlib.import_module(module_name)
    mod = sys.modules[module_name]
    if path.name == '__init__.py':
        return mod
    ignore = os.environ.get('PY_IGNORE_IMPORTMISMATCH', '')
    if ignore != '1':
        module_file = mod.__file__
        if module_file is None:
            raise ImportPathMismatchError(module_name, module_file, path)
        if module_file.endswith(('.pyc', '.pyo')):
            module_file = module_file[:-1]
        if module_file.endswith(os.sep + '__init__.py'):
            module_file = module_file[:-len(os.sep + '__init__.py')]
        try:
            is_same = _is_same(str(path), module_file)
        except FileNotFoundError:
            is_same = False
        if not is_same:
            raise ImportPathMismatchError(module_name, module_file, path)
    return mod
if sys.platform.startswith('win'):

    def _is_same(f1: str, f2: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return Path(f1) == Path(f2) or os.path.samefile(f1, f2)
else:

    def _is_same(f1: str, f2: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return os.path.samefile(f1, f2)

def module_name_from_path(path: Path, root: Path) -> str:
    if False:
        while True:
            i = 10
    '\n    Return a dotted module name based on the given path, anchored on root.\n\n    For example: path="projects/src/tests/test_foo.py" and root="/projects", the\n    resulting module name will be "src.tests.test_foo".\n    '
    path = path.with_suffix('')
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        path_parts = path.parts[1:]
    else:
        path_parts = relative_path.parts
    if len(path_parts) >= 2 and path_parts[-1] == '__init__':
        path_parts = path_parts[:-1]
    return '.'.join(path_parts)

def insert_missing_modules(modules: Dict[str, ModuleType], module_name: str) -> None:
    if False:
        print('Hello World!')
    '\n    Used by ``import_path`` to create intermediate modules when using mode=importlib.\n\n    When we want to import a module as "src.tests.test_foo" for example, we need\n    to create empty modules "src" and "src.tests" after inserting "src.tests.test_foo",\n    otherwise "src.tests.test_foo" is not importable by ``__import__``.\n    '
    module_parts = module_name.split('.')
    child_module: Union[ModuleType, None] = None
    module: Union[ModuleType, None] = None
    child_name: str = ''
    while module_name:
        if module_name not in modules:
            try:
                if not sys.meta_path:
                    raise ModuleNotFoundError
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                module = ModuleType(module_name, doc="Empty module created by pytest's importmode=importlib.")
        else:
            module = modules[module_name]
        if child_module:
            if not hasattr(module, child_name):
                setattr(module, child_name, child_module)
                modules[module_name] = module
        (child_module, child_name) = (module, module_name.rpartition('.')[-1])
        module_parts.pop(-1)
        module_name = '.'.join(module_parts)

def resolve_package_path(path: Path) -> Optional[Path]:
    if False:
        return 10
    'Return the Python package path by looking for the last\n    directory upwards which still contains an __init__.py.\n\n    Returns None if it can not be determined.\n    '
    result = None
    for parent in itertools.chain((path,), path.parents):
        if parent.is_dir():
            if not (parent / '__init__.py').is_file():
                break
            if not parent.name.isidentifier():
                break
            result = parent
    return result

def scandir(path: Union[str, 'os.PathLike[str]']) -> List['os.DirEntry[str]']:
    if False:
        while True:
            i = 10
    'Scan a directory recursively, in breadth-first order.\n\n    The returned entries are sorted.\n    '
    entries = []
    with os.scandir(path) as s:
        for entry in s:
            try:
                entry.is_file()
            except OSError as err:
                if _ignore_error(err):
                    continue
                raise
            entries.append(entry)
    entries.sort(key=lambda entry: entry.name)
    return entries

def visit(path: Union[str, 'os.PathLike[str]'], recurse: Callable[['os.DirEntry[str]'], bool]) -> Iterator['os.DirEntry[str]']:
    if False:
        while True:
            i = 10
    'Walk a directory recursively, in breadth-first order.\n\n    The `recurse` predicate determines whether a directory is recursed.\n\n    Entries at each directory level are sorted.\n    '
    entries = scandir(path)
    yield from entries
    for entry in entries:
        if entry.is_dir() and recurse(entry):
            yield from visit(entry.path, recurse)

def absolutepath(path: Union[Path, str]) -> Path:
    if False:
        print('Hello World!')
    "Convert a path to an absolute path using os.path.abspath.\n\n    Prefer this over Path.resolve() (see #6523).\n    Prefer this over Path.absolute() (not public, doesn't normalize).\n    "
    return Path(os.path.abspath(str(path)))

def commonpath(path1: Path, path2: Path) -> Optional[Path]:
    if False:
        i = 10
        return i + 15
    'Return the common part shared with the other path, or None if there is\n    no common part.\n\n    If one path is relative and one is absolute, returns None.\n    '
    try:
        return Path(os.path.commonpath((str(path1), str(path2))))
    except ValueError:
        return None

def bestrelpath(directory: Path, dest: Path) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Return a string which is a relative path from directory to dest such\n    that directory/bestrelpath == dest.\n\n    The paths must be either both absolute or both relative.\n\n    If no such path can be determined, returns dest.\n    '
    assert isinstance(directory, Path)
    assert isinstance(dest, Path)
    if dest == directory:
        return os.curdir
    base = commonpath(directory, dest)
    if not base:
        return str(dest)
    reldirectory = directory.relative_to(base)
    reldest = dest.relative_to(base)
    return os.path.join(*[os.pardir] * len(reldirectory.parts), *reldest.parts)

def safe_exists(p: Path) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Like Path.exists(), but account for input arguments that might be too long (#11394).'
    try:
        return p.exists()
    except (ValueError, OSError):
        return False