"""
Module houses `ModinDatabaseConnection` class.

`ModinDatabaseConnection` lets a single process make its own connection to a
database to read from it. Whereas it's possible in pandas to pass an open
connection directly to `read_sql`, the open connection is not pickleable
in Modin, so each worker must open its own connection.
`ModinDatabaseConnection` saves the arguments that would normally be used to
make a db connection. It can make and provide a connection whenever the Modin
driver or a worker wants one.
"""
from typing import Any, Dict, Optional, Sequence
_PSYCOPG_LIB_NAME = 'psycopg2'
_SQLALCHEMY_LIB_NAME = 'sqlalchemy'

class UnsupportedDatabaseException(Exception):
    """Modin can't create a particular kind of database connection."""
    pass

class ModinDatabaseConnection:
    """
    Creates a SQL database connection.

    Parameters
    ----------
    lib : str
        The library for the SQL connection.
    *args : iterable
        Positional arguments to pass when creating the connection.
    **kwargs : dict
        Keyword arguments to pass when creating the connection.
    """
    lib: str
    args: Sequence
    kwargs: Dict
    _dialect_is_microsoft_sql_cache: Optional[bool]

    def __init__(self, lib: str, *args: Any, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        lib = lib.lower()
        if lib not in (_PSYCOPG_LIB_NAME, _SQLALCHEMY_LIB_NAME):
            raise UnsupportedDatabaseException(f'Unsupported database library {lib}')
        self.lib = lib
        self.args = args
        self.kwargs = kwargs
        self._dialect_is_microsoft_sql_cache = None

    def _dialect_is_microsoft_sql(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "\n        Tell whether this connection requires Microsoft SQL dialect.\n\n        If this is a sqlalchemy connection, create an engine from args and\n        kwargs. If that engine's driver is pymssql or pyodbc, this\n        connection requires Microsoft SQL. Otherwise, it doesn't.\n\n        Returns\n        -------\n        bool\n        "
        if self._dialect_is_microsoft_sql_cache is None:
            self._dialect_is_microsoft_sql_cache = False
            if self.lib == _SQLALCHEMY_LIB_NAME:
                from sqlalchemy import create_engine
                self._dialect_is_microsoft_sql_cache = create_engine(*self.args, **self.kwargs).driver in ('pymssql', 'pyodbc')
        return self._dialect_is_microsoft_sql_cache

    def get_connection(self) -> Any:
        if False:
            return 10
        '\n        Make the database connection and get it.\n\n        For psycopg2, pass all arguments to psycopg2.connect() and return the\n        result of psycopg2.connect(). For sqlalchemy, pass all arguments to\n        sqlalchemy.create_engine() and return the result of calling connect()\n        on the engine.\n\n        Returns\n        -------\n        Any\n            The open database connection.\n        '
        if self.lib == _PSYCOPG_LIB_NAME:
            import psycopg2
            return psycopg2.connect(*self.args, **self.kwargs)
        if self.lib == _SQLALCHEMY_LIB_NAME:
            from sqlalchemy import create_engine
            return create_engine(*self.args, **self.kwargs).connect()
        raise UnsupportedDatabaseException('Unsupported database library')

    def get_string(self) -> str:
        if False:
            return 10
        '\n        Get input connection string.\n\n        Returns\n        -------\n        str\n        '
        return self.args[0]

    def column_names_query(self, query: str) -> str:
        if False:
            i = 10
            return i + 15
        '\n        Get a query that gives the names of columns that `query` would produce.\n\n        Parameters\n        ----------\n        query : str\n            The SQL query to check.\n\n        Returns\n        -------\n        str\n        '
        return f'SELECT * FROM ({query}) AS _MODIN_COUNT_QUERY WHERE 1 = 0'

    def row_count_query(self, query: str) -> str:
        if False:
            while True:
                i = 10
        '\n        Get a query that gives the names of rows that `query` would produce.\n\n        Parameters\n        ----------\n        query : str\n            The SQL query to check.\n\n        Returns\n        -------\n        str\n        '
        return f'SELECT COUNT(*) FROM ({query}) AS _MODIN_COUNT_QUERY'

    def partition_query(self, query: str, limit: int, offset: int) -> str:
        if False:
            return 10
        '\n        Get a query that partitions the original `query`.\n\n        Parameters\n        ----------\n        query : str\n            The SQL query to get a partition.\n        limit : int\n            The size of the partition.\n        offset : int\n            Where the partition begins.\n\n        Returns\n        -------\n        str\n        '
        return f'SELECT * FROM ({query}) AS _MODIN_COUNT_QUERY ORDER BY(SELECT NULL)' + f' OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY' if self._dialect_is_microsoft_sql() else f'SELECT * FROM ({query}) AS _MODIN_COUNT_QUERY LIMIT ' + f'{limit} OFFSET {offset}'