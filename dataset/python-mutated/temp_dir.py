import errno
import itertools
import logging
import os.path
import tempfile
import traceback
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar, Union
from pip._internal.utils.misc import enum, rmtree
logger = logging.getLogger(__name__)
_T = TypeVar('_T', bound='TempDirectory')
tempdir_kinds = enum(BUILD_ENV='build-env', EPHEM_WHEEL_CACHE='ephem-wheel-cache', REQ_BUILD='req-build')
_tempdir_manager: Optional[ExitStack] = None

@contextmanager
def global_tempdir_manager() -> Generator[None, None, None]:
    if False:
        return 10
    global _tempdir_manager
    with ExitStack() as stack:
        (old_tempdir_manager, _tempdir_manager) = (_tempdir_manager, stack)
        try:
            yield
        finally:
            _tempdir_manager = old_tempdir_manager

class TempDirectoryTypeRegistry:
    """Manages temp directory behavior"""

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        self._should_delete: Dict[str, bool] = {}

    def set_delete(self, kind: str, value: bool) -> None:
        if False:
            i = 10
            return i + 15
        'Indicate whether a TempDirectory of the given kind should be\n        auto-deleted.\n        '
        self._should_delete[kind] = value

    def get_delete(self, kind: str) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Get configured auto-delete flag for a given TempDirectory type,\n        default True.\n        '
        return self._should_delete.get(kind, True)
_tempdir_registry: Optional[TempDirectoryTypeRegistry] = None

@contextmanager
def tempdir_registry() -> Generator[TempDirectoryTypeRegistry, None, None]:
    if False:
        i = 10
        return i + 15
    'Provides a scoped global tempdir registry that can be used to dictate\n    whether directories should be deleted.\n    '
    global _tempdir_registry
    old_tempdir_registry = _tempdir_registry
    _tempdir_registry = TempDirectoryTypeRegistry()
    try:
        yield _tempdir_registry
    finally:
        _tempdir_registry = old_tempdir_registry

class _Default:
    pass
_default = _Default()

class TempDirectory:
    """Helper class that owns and cleans up a temporary directory.

    This class can be used as a context manager or as an OO representation of a
    temporary directory.

    Attributes:
        path
            Location to the created temporary directory
        delete
            Whether the directory should be deleted when exiting
            (when used as a contextmanager)

    Methods:
        cleanup()
            Deletes the temporary directory

    When used as a context manager, if the delete attribute is True, on
    exiting the context the temporary directory is deleted.
    """

    def __init__(self, path: Optional[str]=None, delete: Union[bool, None, _Default]=_default, kind: str='temp', globally_managed: bool=False, ignore_cleanup_errors: bool=True):
        if False:
            i = 10
            return i + 15
        super().__init__()
        if delete is _default:
            if path is not None:
                delete = False
            else:
                delete = None
        if path is None:
            path = self._create(kind)
        self._path = path
        self._deleted = False
        self.delete = delete
        self.kind = kind
        self.ignore_cleanup_errors = ignore_cleanup_errors
        if globally_managed:
            assert _tempdir_manager is not None
            _tempdir_manager.enter_context(self)

    @property
    def path(self) -> str:
        if False:
            return 10
        assert not self._deleted, f'Attempted to access deleted path: {self._path}'
        return self._path

    def __repr__(self) -> str:
        if False:
            return 10
        return f'<{self.__class__.__name__} {self.path!r}>'

    def __enter__(self: _T) -> _T:
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc: Any, value: Any, tb: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if self.delete is not None:
            delete = self.delete
        elif _tempdir_registry:
            delete = _tempdir_registry.get_delete(self.kind)
        else:
            delete = True
        if delete:
            self.cleanup()

    def _create(self, kind: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Create a temporary directory and store its path in self.path'
        path = os.path.realpath(tempfile.mkdtemp(prefix=f'pip-{kind}-'))
        logger.debug('Created temporary directory: %s', path)
        return path

    def cleanup(self) -> None:
        if False:
            i = 10
            return i + 15
        'Remove the temporary directory created and reset state'
        self._deleted = True
        if not os.path.exists(self._path):
            return
        errors: List[BaseException] = []

        def onerror(func: Callable[..., Any], path: Path, exc_val: BaseException) -> None:
            if False:
                print('Hello World!')
            'Log a warning for a `rmtree` error and continue'
            formatted_exc = '\n'.join(traceback.format_exception_only(type(exc_val), exc_val))
            formatted_exc = formatted_exc.rstrip()
            if func in (os.unlink, os.remove, os.rmdir):
                logger.debug("Failed to remove a temporary file '%s' due to %s.\n", path, formatted_exc)
            else:
                logger.debug('%s failed with %s.', func.__qualname__, formatted_exc)
            errors.append(exc_val)
        if self.ignore_cleanup_errors:
            try:
                rmtree(self._path, ignore_errors=False)
            except OSError:
                rmtree(self._path, onexc=onerror)
            if errors:
                logger.warning("Failed to remove contents in a temporary directory '%s'.\nYou can safely remove it manually.", self._path)
        else:
            rmtree(self._path)

class AdjacentTempDirectory(TempDirectory):
    """Helper class that creates a temporary directory adjacent to a real one.

    Attributes:
        original
            The original directory to create a temp directory for.
        path
            After calling create() or entering, contains the full
            path to the temporary directory.
        delete
            Whether the directory should be deleted when exiting
            (when used as a contextmanager)

    """
    LEADING_CHARS = '-~.=%0123456789'

    def __init__(self, original: str, delete: Optional[bool]=None) -> None:
        if False:
            while True:
                i = 10
        self.original = original.rstrip('/\\')
        super().__init__(delete=delete)

    @classmethod
    def _generate_names(cls, name: str) -> Generator[str, None, None]:
        if False:
            i = 10
            return i + 15
        'Generates a series of temporary names.\n\n        The algorithm replaces the leading characters in the name\n        with ones that are valid filesystem characters, but are not\n        valid package names (for both Python and pip definitions of\n        package).\n        '
        for i in range(1, len(name)):
            for candidate in itertools.combinations_with_replacement(cls.LEADING_CHARS, i - 1):
                new_name = '~' + ''.join(candidate) + name[i:]
                if new_name != name:
                    yield new_name
        for i in range(len(cls.LEADING_CHARS)):
            for candidate in itertools.combinations_with_replacement(cls.LEADING_CHARS, i):
                new_name = '~' + ''.join(candidate) + name
                if new_name != name:
                    yield new_name

    def _create(self, kind: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        (root, name) = os.path.split(self.original)
        for candidate in self._generate_names(name):
            path = os.path.join(root, candidate)
            try:
                os.mkdir(path)
            except OSError as ex:
                if ex.errno != errno.EEXIST:
                    raise
            else:
                path = os.path.realpath(path)
                break
        else:
            path = os.path.realpath(tempfile.mkdtemp(prefix=f'pip-{kind}-'))
        logger.debug('Created temporary directory: %s', path)
        return path