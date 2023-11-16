import importlib
import types
from types import TracebackType
from typing import Any
from typing import Optional
from typing import Tuple
from typing import Type

class _DeferredImportExceptionContextManager:
    """Context manager to defer exceptions from imports.

    Catches :exc:`ImportError` and :exc:`SyntaxError`.
    If any exception is caught, this class raises an :exc:`ImportError` when being checked.

    """

    def __init__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._deferred: Optional[Tuple[Exception, str]] = None

    def __enter__(self) -> '_DeferredImportExceptionContextManager':
        if False:
            return 10
        'Enter the context manager.\n\n        Returns:\n            Itself.\n\n        '
        return self

    def __exit__(self, exc_type: Optional[Type[Exception]], exc_value: Optional[Exception], traceback: Optional[TracebackType]) -> Optional[bool]:
        if False:
            for i in range(10):
                print('nop')
        'Exit the context manager.\n\n        Args:\n            exc_type:\n                Raised exception type. :obj:`None` if nothing is raised.\n            exc_value:\n                Raised exception object. :obj:`None` if nothing is raised.\n            traceback:\n                Associated traceback. :obj:`None` if nothing is raised.\n\n        Returns:\n            :obj:`None` if nothing is deferred, otherwise :obj:`True`.\n            :obj:`True` will suppress any exceptions avoiding them from propagating.\n\n        '
        if isinstance(exc_value, (ImportError, SyntaxError)):
            if isinstance(exc_value, ImportError):
                message = "Tried to import '{}' but failed. Please make sure that the package is installed correctly to use this feature. Actual error: {}.".format(exc_value.name, exc_value)
            elif isinstance(exc_value, SyntaxError):
                message = 'Tried to import a package but failed due to a syntax error in {}. Please make sure that the Python version is correct to use this feature. Actual error: {}.'.format(exc_value.filename, exc_value)
            else:
                assert False
            self._deferred = (exc_value, message)
            return True
        return None

    def is_successful(self) -> bool:
        if False:
            print('Hello World!')
        'Return whether the context manager has caught any exceptions.\n\n        Returns:\n            :obj:`True` if no exceptions are caught, :obj:`False` otherwise.\n\n        '
        return self._deferred is None

    def check(self) -> None:
        if False:
            while True:
                i = 10
        'Check whether the context manager has caught any exceptions.\n\n        Raises:\n            :exc:`ImportError`:\n                If any exception was caught from the caught exception.\n\n        '
        if self._deferred is not None:
            (exc_value, message) = self._deferred
            raise ImportError(message) from exc_value

def try_import() -> _DeferredImportExceptionContextManager:
    if False:
        i = 10
        return i + 15
    'Create a context manager that can wrap imports of optional packages to defer exceptions.\n\n    Returns:\n        Deferred import context manager.\n\n    '
    return _DeferredImportExceptionContextManager()

class _LazyImport(types.ModuleType):
    """Module wrapper for lazy import.

    This class wraps the specified modules and lazily imports them only when accessed.
    Otherwise, `import optuna` is slowed down by importing all submodules and
    dependencies even if not required.
    Within this project's usage, importlib override this module's attribute on the first
    access and the imported submodule is directly accessed from the second access.

    Args:
        name: Name of module to apply lazy import.
    """

    def __init__(self, name: str) -> None:
        if False:
            while True:
                i = 10
        super().__init__(name)
        self._name = name

    def _load(self) -> types.ModuleType:
        if False:
            return 10
        module = importlib.import_module(self._name)
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, item: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return getattr(self._load(), item)