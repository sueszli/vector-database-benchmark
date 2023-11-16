"""Tools for working with write concerns."""
from __future__ import annotations
from typing import Any, Optional, Union
from pymongo.errors import ConfigurationError

def validate_boolean(option: str, value: Any) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Validates that 'value' is True or False."
    if isinstance(value, bool):
        return value
    raise TypeError(f'{option} must be True or False, was: {option}={value}')

class WriteConcern:
    """WriteConcern

    :Parameters:
        - `w`: (integer or string) Used with replication, write operations
          will block until they have been replicated to the specified number
          or tagged set of servers. `w=<integer>` always includes the replica
          set primary (e.g. w=3 means write to the primary and wait until
          replicated to **two** secondaries). **w=0 disables acknowledgement
          of write operations and can not be used with other write concern
          options.**
        - `wtimeout`: (integer) Used in conjunction with `w`. Specify a value
          in milliseconds to control how long to wait for write propagation
          to complete. If replication does not complete in the given
          timeframe, a timeout exception is raised.
        - `j`: If ``True`` block until write operations have been committed
          to the journal. Cannot be used in combination with `fsync`. Write
          operations will fail with an exception if this option is used when
          the server is running without journaling.
        - `fsync`: If ``True`` and the server is running without journaling,
          blocks until the server has synced all data files to disk. If the
          server is running with journaling, this acts the same as the `j`
          option, blocking until write operations have been committed to the
          journal. Cannot be used in combination with `j`.
    """
    __slots__ = ('__document', '__acknowledged', '__server_default')

    def __init__(self, w: Optional[Union[int, str]]=None, wtimeout: Optional[int]=None, j: Optional[bool]=None, fsync: Optional[bool]=None) -> None:
        if False:
            while True:
                i = 10
        self.__document: dict[str, Any] = {}
        self.__acknowledged = True
        if wtimeout is not None:
            if not isinstance(wtimeout, int):
                raise TypeError('wtimeout must be an integer')
            if wtimeout < 0:
                raise ValueError('wtimeout cannot be less than 0')
            self.__document['wtimeout'] = wtimeout
        if j is not None:
            validate_boolean('j', j)
            self.__document['j'] = j
        if fsync is not None:
            validate_boolean('fsync', fsync)
            if j and fsync:
                raise ConfigurationError("Can't set both j and fsync at the same time")
            self.__document['fsync'] = fsync
        if w == 0 and j is True:
            raise ConfigurationError('Cannot set w to 0 and j to True')
        if w is not None:
            if isinstance(w, int):
                if w < 0:
                    raise ValueError('w cannot be less than 0')
                self.__acknowledged = w > 0
            elif not isinstance(w, str):
                raise TypeError('w must be an integer or string')
            self.__document['w'] = w
        self.__server_default = not self.__document

    @property
    def is_server_default(self) -> bool:
        if False:
            print('Hello World!')
        'Does this WriteConcern match the server default.'
        return self.__server_default

    @property
    def document(self) -> dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        'The document representation of this write concern.\n\n        .. note::\n          :class:`WriteConcern` is immutable. Mutating the value of\n          :attr:`document` does not mutate this :class:`WriteConcern`.\n        '
        return self.__document.copy()

    @property
    def acknowledged(self) -> bool:
        if False:
            print('Hello World!')
        'If ``True`` write operations will wait for acknowledgement before\n        returning.\n        '
        return self.__acknowledged

    def __repr__(self) -> str:
        if False:
            for i in range(10):
                print('nop')
        return 'WriteConcern({})'.format(', '.join(('{}={}'.format(*kvt) for kvt in self.__document.items())))

    def __eq__(self, other: Any) -> bool:
        if False:
            return 10
        if isinstance(other, WriteConcern):
            return self.__document == other.document
        return NotImplemented

    def __ne__(self, other: Any) -> bool:
        if False:
            print('Hello World!')
        if isinstance(other, WriteConcern):
            return self.__document != other.document
        return NotImplemented
DEFAULT_WRITE_CONCERN = WriteConcern()