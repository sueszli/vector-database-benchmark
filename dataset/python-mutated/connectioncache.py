from typing import Any
from .normalizing import NormalizedDict
Connection = Any

class ConnectionCache:
    """Cache for libraries to use with concurrent connections, processes, etc.

    The cache stores the registered connections (or other objects) and allows
    switching between them using generated indices, user given aliases or
    connection objects themselves. This is useful with any library having a need
    for multiple concurrent connections, processes, etc.

    This class is used also outside the core framework by SeleniumLibrary,
    SSHLibrary, etc. Backwards compatibility is thus important when doing changes.
    """

    def __init__(self, no_current_msg='No open connection.'):
        if False:
            i = 10
            return i + 15
        self._no_current = NoConnection(no_current_msg)
        self.current = self._no_current
        self._connections = []
        self._aliases = NormalizedDict[int]()

    @property
    def current_index(self) -> 'int|None':
        if False:
            print('Hello World!')
        if not self:
            return None
        for (index, conn) in enumerate(self):
            if conn is self.current:
                return index + 1

    @current_index.setter
    def current_index(self, index: 'int|None'):
        if False:
            for i in range(10):
                print('nop')
        if index is None:
            self.current = self._no_current
        else:
            self.current = self._connections[index - 1]

    def register(self, connection: Connection, alias: 'str|None'=None):
        if False:
            for i in range(10):
                print('nop')
        'Registers given connection with optional alias and returns its index.\n\n        Given connection is set to be the :attr:`current` connection.\n\n        If alias is given, it must be a string. Aliases are case and space\n        insensitive.\n\n        The index of the first connection after initialization, and after\n        :meth:`close_all` or :meth:`empty_cache`, is 1, second is 2, etc.\n        '
        self.current = connection
        self._connections.append(connection)
        index = len(self._connections)
        if alias:
            self._aliases[alias] = index
        return index

    def switch(self, identifier: 'int|str|Connection') -> Connection:
        if False:
            for i in range(10):
                print('nop')
        'Switches to the connection specified using the ``identifier``.\n\n        Identifier can be an index, an alias, or a registered connection.\n        Raises an error if no matching connection is found.\n\n        Updates :attr:`current` and also returns its new value.\n        '
        self.current = self.get_connection(identifier)
        return self.current

    def get_connection(self, identifier: 'int|str|Connection|None'=None) -> Connection:
        if False:
            return 10
        'Returns the connection specified using the ``identifier``.\n\n        Identifier can be an index (integer or string), an alias, a registered\n        connection or ``None``. If the identifier is ``None``, returns the\n        current connection if it is active and raises an error if it is not.\n        Raises an error also if no matching connection is found.\n        '
        if identifier is None:
            if not self:
                self.current.raise_error()
            return self.current
        try:
            index = self.get_connection_index(identifier)
        except ValueError as err:
            raise RuntimeError(err.args[0])
        return self._connections[index - 1]

    def get_connection_index(self, identifier: 'int|str|Connection') -> int:
        if False:
            for i in range(10):
                print('nop')
        'Returns the index of the connection specified using the ``identifier``.\n\n        Identifier can be an index (integer or string), an alias, or a registered\n        connection.\n\n        New in Robot Framework 7.0. :meth:`resolve_alias_or_index` can be used\n        with earlier versions.\n        '
        if isinstance(identifier, str) and identifier in self._aliases:
            return self._aliases[identifier]
        if identifier in self._connections:
            return self._connections.index(identifier) + 1
        try:
            index = int(identifier)
        except (ValueError, TypeError):
            index = -1
        if 0 < index <= len(self._connections):
            return index
        raise ValueError(f"Non-existing index or alias '{identifier}'.")

    def resolve_alias_or_index(self, alias_or_index):
        if False:
            for i in range(10):
                print('nop')
        'Deprecated in RF 7.0. Use :meth:`get_connection_index` instead.'
        return self.get_connection_index(alias_or_index)

    def close_all(self, closer_method: str='close'):
        if False:
            for i in range(10):
                print('nop')
        'Closes connections using the specified closer method and empties cache.\n\n        If simply calling the closer method is not adequate for closing\n        connections, clients should close connections themselves and use\n        :meth:`empty_cache` afterward.\n        '
        for conn in self._connections:
            getattr(conn, closer_method)()
        self.empty_cache()
        return self.current

    def empty_cache(self):
        if False:
            for i in range(10):
                print('nop')
        'Empties the connection cache.\n\n        Indexes of the new connections starts from 1 after this.\n        '
        self.current = self._no_current
        self._connections = []
        self._aliases = NormalizedDict()
    __getitem__ = get_connection

    def __iter__(self):
        if False:
            while True:
                i = 10
        return iter(self._connections)

    def __len__(self):
        if False:
            while True:
                i = 10
        return len(self._connections)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.current is not self._no_current

class NoConnection:

    def __init__(self, message):
        if False:
            i = 10
            return i + 15
        self.message = message

    def __getattr__(self, name):
        if False:
            while True:
                i = 10
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError
        self.raise_error()

    def raise_error(self):
        if False:
            while True:
                i = 10
        raise RuntimeError(self.message)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return False