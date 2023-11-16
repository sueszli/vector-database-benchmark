import json
from . import connresource
from . import cursor
from . import exceptions

class PreparedStatement(connresource.ConnectionResource):
    """A representation of a prepared statement."""
    __slots__ = ('_state', '_query', '_last_status')

    def __init__(self, connection, query, state):
        if False:
            while True:
                i = 10
        super().__init__(connection)
        self._state = state
        self._query = query
        state.attach()
        self._last_status = None

    @connresource.guarded
    def get_name(self) -> str:
        if False:
            i = 10
            return i + 15
        'Return the name of this prepared statement.\n\n        .. versionadded:: 0.25.0\n        '
        return self._state.name

    @connresource.guarded
    def get_query(self) -> str:
        if False:
            while True:
                i = 10
        'Return the text of the query for this prepared statement.\n\n        Example::\n\n            stmt = await connection.prepare(\'SELECT $1::int\')\n            assert stmt.get_query() == "SELECT $1::int"\n        '
        return self._query

    @connresource.guarded
    def get_statusmsg(self) -> str:
        if False:
            return 10
        'Return the status of the executed command.\n\n        Example::\n\n            stmt = await connection.prepare(\'CREATE TABLE mytab (a int)\')\n            await stmt.fetch()\n            assert stmt.get_statusmsg() == "CREATE TABLE"\n        '
        if self._last_status is None:
            return self._last_status
        return self._last_status.decode()

    @connresource.guarded
    def get_parameters(self):
        if False:
            return 10
        "Return a description of statement parameters types.\n\n        :return: A tuple of :class:`asyncpg.types.Type`.\n\n        Example::\n\n            stmt = await connection.prepare('SELECT ($1::int, $2::text)')\n            print(stmt.get_parameters())\n\n            # Will print:\n            #   (Type(oid=23, name='int4', kind='scalar', schema='pg_catalog'),\n            #    Type(oid=25, name='text', kind='scalar', schema='pg_catalog'))\n        "
        return self._state._get_parameters()

    @connresource.guarded
    def get_attributes(self):
        if False:
            while True:
                i = 10
        "Return a description of relation attributes (columns).\n\n        :return: A tuple of :class:`asyncpg.types.Attribute`.\n\n        Example::\n\n            st = await self.con.prepare('''\n                SELECT typname, typnamespace FROM pg_type\n            ''')\n            print(st.get_attributes())\n\n            # Will print:\n            #   (Attribute(\n            #       name='typname',\n            #       type=Type(oid=19, name='name', kind='scalar',\n            #                 schema='pg_catalog')),\n            #    Attribute(\n            #       name='typnamespace',\n            #       type=Type(oid=26, name='oid', kind='scalar',\n            #                 schema='pg_catalog')))\n        "
        return self._state._get_attributes()

    @connresource.guarded
    def cursor(self, *args, prefetch=None, timeout=None) -> cursor.CursorFactory:
        if False:
            for i in range(10):
                print('nop')
        'Return a *cursor factory* for the prepared statement.\n\n        :param args: Query arguments.\n        :param int prefetch: The number of rows the *cursor iterator*\n                             will prefetch (defaults to ``50``.)\n        :param float timeout: Optional timeout in seconds.\n\n        :return: A :class:`~cursor.CursorFactory` object.\n        '
        return cursor.CursorFactory(self._connection, self._query, self._state, args, prefetch, timeout, self._state.record_class)

    @connresource.guarded
    async def explain(self, *args, analyze=False):
        """Return the execution plan of the statement.

        :param args: Query arguments.
        :param analyze: If ``True``, the statement will be executed and
                        the run time statitics added to the return value.

        :return: An object representing the execution plan.  This value
                 is actually a deserialized JSON output of the SQL
                 ``EXPLAIN`` command.
        """
        query = 'EXPLAIN (FORMAT JSON, VERBOSE'
        if analyze:
            query += ', ANALYZE) '
        else:
            query += ') '
        query += self._state.query
        if analyze:
            tr = self._connection.transaction()
            await tr.start()
            try:
                data = await self._connection.fetchval(query, *args)
            finally:
                await tr.rollback()
        else:
            data = await self._connection.fetchval(query, *args)
        return json.loads(data)

    @connresource.guarded
    async def fetch(self, *args, timeout=None):
        """Execute the statement and return a list of :class:`Record` objects.

        :param str query: Query text
        :param args: Query arguments
        :param float timeout: Optional timeout value in seconds.

        :return: A list of :class:`Record` instances.
        """
        data = await self.__bind_execute(args, 0, timeout)
        return data

    @connresource.guarded
    async def fetchval(self, *args, column=0, timeout=None):
        """Execute the statement and return a value in the first row.

        :param args: Query arguments.
        :param int column: Numeric index within the record of the value to
                           return (defaults to 0).
        :param float timeout: Optional timeout value in seconds.
                            If not specified, defaults to the value of
                            ``command_timeout`` argument to the ``Connection``
                            instance constructor.

        :return: The value of the specified column of the first record.
        """
        data = await self.__bind_execute(args, 1, timeout)
        if not data:
            return None
        return data[0][column]

    @connresource.guarded
    async def fetchrow(self, *args, timeout=None):
        """Execute the statement and return the first row.

        :param str query: Query text
        :param args: Query arguments
        :param float timeout: Optional timeout value in seconds.

        :return: The first row as a :class:`Record` instance.
        """
        data = await self.__bind_execute(args, 1, timeout)
        if not data:
            return None
        return data[0]

    @connresource.guarded
    async def executemany(self, args, *, timeout: float=None):
        """Execute the statement for each sequence of arguments in *args*.

        :param args: An iterable containing sequences of arguments.
        :param float timeout: Optional timeout value in seconds.
        :return None: This method discards the results of the operations.

        .. versionadded:: 0.22.0
        """
        return await self.__do_execute(lambda protocol: protocol.bind_execute_many(self._state, args, '', timeout))

    async def __do_execute(self, executor):
        protocol = self._connection._protocol
        try:
            return await executor(protocol)
        except exceptions.OutdatedSchemaCacheError:
            await self._connection.reload_schema_state()
            self._state.mark_closed()
            raise

    async def __bind_execute(self, args, limit, timeout):
        (data, status, _) = await self.__do_execute(lambda protocol: protocol.bind_execute(self._state, args, '', limit, True, timeout))
        self._last_status = status
        return data

    def _check_open(self, meth_name):
        if False:
            while True:
                i = 10
        if self._state.closed:
            raise exceptions.InterfaceError('cannot call PreparedStmt.{}(): the prepared statement is closed'.format(meth_name))

    def _check_conn_validity(self, meth_name):
        if False:
            while True:
                i = 10
        self._check_open(meth_name)
        super()._check_conn_validity(meth_name)

    def __del__(self):
        if False:
            while True:
                i = 10
        self._state.detach()
        self._connection._maybe_gc_stmt(self._state)