from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, Union
import apsw
__all__ = ['APSWConnectionWrapper']

class ProvidesCursor:

    def cursor(self) -> apsw.Cursor:
        if False:
            return 10
        ...

class ContextManagerMixin(ProvidesCursor):

    @contextmanager
    def with_cursor(self) -> Generator[apsw.Cursor, None, None]:
        if False:
            print('Hello World!')
        "\n        apsw cursors are relatively cheap, and are gc safe\n        In most cases, it's fine not to use this.\n        "
        c = self.cursor()
        try:
            yield c
        finally:
            c.close()

    @contextmanager
    def transaction(self) -> Generator[apsw.Cursor, None, None]:
        if False:
            while True:
                i = 10
        '\n        Wraps a cursor as a context manager for a transaction\n        which is rolled back on unhandled exception,\n        or committed on non-exception exit\n        '
        c = self.cursor()
        try:
            c.execute('BEGIN TRANSACTION')
            yield c
        except Exception:
            c.execute('ROLLBACK TRANSACTION')
            raise
        else:
            c.execute('COMMIT TRANSACTION')
        finally:
            c.close()

class APSWConnectionWrapper(apsw.Connection, ContextManagerMixin):
    """
    Provides a few convenience methods, and allows a path object for construction
    """

    def __init__(self, filename: Union[Path, str], *args, **kwargs):
        if False:
            return 10
        super().__init__(str(filename), *args, **kwargs)