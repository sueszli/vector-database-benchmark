"""Tools for working with read concerns."""
from __future__ import annotations
from typing import Any, Optional

class ReadConcern:
    """ReadConcern

    :Parameters:
        - `level`: (string) The read concern level specifies the level of
          isolation for read operations.  For example, a read operation using a
          read concern level of ``majority`` will only return data that has been
          written to a majority of nodes. If the level is left unspecified, the
          server default will be used.

    .. versionadded:: 3.2

    """

    def __init__(self, level: Optional[str]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        if level is None or isinstance(level, str):
            self.__level = level
        else:
            raise TypeError('level must be a string or None.')

    @property
    def level(self) -> Optional[str]:
        if False:
            for i in range(10):
                print('nop')
        'The read concern level.'
        return self.__level

    @property
    def ok_for_legacy(self) -> bool:
        if False:
            return 10
        'Return ``True`` if this read concern is compatible with\n        old wire protocol versions.\n        '
        return self.level is None or self.level == 'local'

    @property
    def document(self) -> dict[str, Any]:
        if False:
            print('Hello World!')
        'The document representation of this read concern.\n\n        .. note::\n          :class:`ReadConcern` is immutable. Mutating the value of\n          :attr:`document` does not mutate this :class:`ReadConcern`.\n        '
        doc = {}
        if self.__level:
            doc['level'] = self.level
        return doc

    def __eq__(self, other: Any) -> bool:
        if False:
            while True:
                i = 10
        if isinstance(other, ReadConcern):
            return self.document == other.document
        return NotImplemented

    def __repr__(self) -> str:
        if False:
            while True:
                i = 10
        if self.level:
            return 'ReadConcern(%s)' % self.level
        return 'ReadConcern()'
DEFAULT_READ_CONCERN = ReadConcern()