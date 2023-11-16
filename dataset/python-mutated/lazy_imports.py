from typing import Optional, Type
from types import TracebackType
from lazy_imports.try_import import _DeferredImportExceptionContextManager
DEFAULT_IMPORT_ERROR_MSG = "Try 'pip install {}'"

class LazyImport(_DeferredImportExceptionContextManager):
    """
    Wrapper on top of lazy_import's _DeferredImportExceptionContextManager that adds the possibility to customize the
    error messages.
    """

    def __init__(self, message: str=DEFAULT_IMPORT_ERROR_MSG) -> None:
        if False:
            return 10
        super().__init__()
        self.import_error_msg = message

    def __exit__(self, exc_type: Optional[Type[Exception]], exc_value: Optional[Exception], traceback: Optional[TracebackType]) -> Optional[bool]:
        if False:
            print('Hello World!')
        'Exit the context manager.\n\n        Args:\n            exc_type:\n                Raised exception type. :obj:`None` if nothing is raised.\n            exc_value:\n                Raised exception object. :obj:`None` if nothing is raised.\n            traceback:\n                Associated traceback. :obj:`None` if nothing is raised.\n\n        Returns:\n            :obj:`None` if nothing is deferred, otherwise :obj:`True`.\n            :obj:`True` will suppress any exceptions avoiding them from propagating.\n\n        '
        if isinstance(exc_value, ImportError):
            message = f"Failed to import '{exc_value.name}'. {self.import_error_msg.format(exc_value.name)}. Original error: {exc_value}"
            self._deferred = (exc_value, message)
            return True
        return None