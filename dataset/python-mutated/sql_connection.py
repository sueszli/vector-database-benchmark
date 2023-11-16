from __future__ import annotations
from collections import ChainMap
from copy import deepcopy
from datetime import timedelta
from typing import TYPE_CHECKING, List, Optional, Union, cast
import pandas as pd
from streamlit.connections import BaseConnection
from streamlit.connections.util import extract_from_dict
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_data
if TYPE_CHECKING:
    from sqlalchemy.engine import Connection as SQLAlchemyConnection
    from sqlalchemy.engine.base import Engine
    from sqlalchemy.orm import Session
_ALL_CONNECTION_PARAMS = {'url', 'driver', 'dialect', 'username', 'password', 'host', 'port', 'database'}
_REQUIRED_CONNECTION_PARAMS = {'dialect', 'username', 'host'}

class SQLConnection(BaseConnection['Engine']):
    """A connection to a SQL database using a SQLAlchemy Engine. Initialize using ``st.connection("<name>", type="sql")``.

    SQLConnection provides the ``query()`` convenience method, which can be used to
    run simple read-only queries with both caching and simple error handling/retries.
    More complex DB interactions can be performed by using the ``.session`` property
    to receive a regular SQLAlchemy Session.

    SQLConnections should always be created using ``st.connection()``, **not**
    initialized directly. Connection parameters for a SQLConnection can be specified
    using either ``st.secrets`` or ``**kwargs``. Some frequently used parameters include:

    - **url** or arguments for `sqlalchemy.engine.URL.create()
      <https://docs.sqlalchemy.org/en/20/core/engines.html#sqlalchemy.engine.URL.create>`_.
      Most commonly it includes a dialect, host, database, username and password.

    - **create_engine_kwargs** can be passed via ``st.secrets``, such as for
      `snowflake-sqlalchemy <https://github.com/snowflakedb/snowflake-sqlalchemy#key-pair-authentication-support>`_
      or `Google BigQuery <https://github.com/googleapis/python-bigquery-sqlalchemy#authentication>`_.
      These can also be passed directly as ``**kwargs`` to connection().

    - **autocommit=True** to run with isolation level ``AUTOCOMMIT``. Default is False.

    Example
    -------
    >>> import streamlit as st
    >>>
    >>> conn = st.connection("sql")
    >>> df = conn.query("select * from pet_owners")
    >>> st.dataframe(df)
    """

    def _connect(self, autocommit: bool=False, **kwargs) -> 'Engine':
        if False:
            return 10
        import sqlalchemy
        kwargs = deepcopy(kwargs)
        conn_param_kwargs = extract_from_dict(_ALL_CONNECTION_PARAMS, kwargs)
        conn_params = ChainMap(conn_param_kwargs, self._secrets.to_dict())
        if not len(conn_params):
            raise StreamlitAPIException('Missing SQL DB connection configuration. Did you forget to set this in `secrets.toml` or as kwargs to `st.connection`?')
        if 'url' in conn_params:
            url = sqlalchemy.engine.make_url(conn_params['url'])
        else:
            for p in _REQUIRED_CONNECTION_PARAMS:
                if p not in conn_params:
                    raise StreamlitAPIException(f'Missing SQL DB connection param: {p}')
            drivername = conn_params['dialect'] + (f"+{conn_params['driver']}" if 'driver' in conn_params else '')
            url = sqlalchemy.engine.URL.create(drivername=drivername, username=conn_params['username'], password=conn_params.get('password'), host=conn_params['host'], port=int(conn_params['port']) if 'port' in conn_params else None, database=conn_params.get('database'))
        create_engine_kwargs = ChainMap(kwargs, self._secrets.get('create_engine_kwargs', {}))
        eng = sqlalchemy.create_engine(url, **create_engine_kwargs)
        if autocommit:
            return cast('Engine', eng.execution_options(isolation_level='AUTOCOMMIT'))
        else:
            return cast('Engine', eng)

    def query(self, sql: str, *, show_spinner: bool | str='Running `sql.query(...)`.', ttl: Optional[Union[float, int, timedelta]]=None, index_col: Optional[Union[str, List[str]]]=None, chunksize: Optional[int]=None, params=None, **kwargs) -> pd.DataFrame:
        if False:
            print('Hello World!')
        'Run a read-only query.\n\n        This method implements both query result caching (with caching behavior\n        identical to that of using @st.cache_data) as well as simple error handling/retries.\n\n        .. note::\n            Queries that are run without a specified ttl are cached indefinitely.\n\n        Aside from the ``ttl`` kwarg, all kwargs passed to this function are passed down\n        to `pd.read_sql <https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html>`_\n        and have the behavior described in the pandas documentation.\n\n        Parameters\n        ----------\n        sql : str\n            The read-only SQL query to execute.\n        show_spinner : boolean or string\n            Enable the spinner. The default is to show a spinner when there is a\n            "cache miss" and the cached resource is being created. If a string, the value\n            of the show_spinner param will be used for the spinner text.\n        ttl : float, int, timedelta or None\n            The maximum number of seconds to keep results in the cache, or\n            None if cached results should not expire. The default is None.\n        index_col : str, list of str, or None\n            Column(s) to set as index(MultiIndex). Default is None.\n        chunksize : int or None\n            If specified, return an iterator where chunksize is the number of\n            rows to include in each chunk. Default is None.\n        params : list, tuple, dict or None\n            List of parameters to pass to the execute method. The syntax used to pass\n            parameters is database driver dependent. Check your database driver\n            documentation for which of the five syntax styles, described in `PEP 249\n            paramstyle <https://peps.python.org/pep-0249/#paramstyle>`_, is supported.\n            Default is None.\n        **kwargs: dict\n            Additional keyword arguments are passed to `pd.read_sql\n            <https://pandas.pydata.org/docs/reference/api/pandas.read_sql.html>`_.\n\n        Returns\n        -------\n        pd.DataFrame\n            The result of running the query, formatted as a pandas DataFrame.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> conn = st.connection("sql")\n        >>> df = conn.query("select * from pet_owners where owner = :owner", ttl=3600, params={"owner":"barbara"})\n        >>> st.dataframe(df)\n        '
        from sqlalchemy import text
        from sqlalchemy.exc import DatabaseError, InternalError, OperationalError
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

        @retry(after=lambda _: self.reset(), stop=stop_after_attempt(3), reraise=True, retry=retry_if_exception_type((DatabaseError, InternalError, OperationalError)), wait=wait_fixed(1))
        @cache_data(show_spinner=show_spinner, ttl=ttl)
        def _query(sql: str, index_col=None, chunksize=None, params=None, **kwargs) -> pd.DataFrame:
            if False:
                while True:
                    i = 10
            instance = self._instance.connect()
            return pd.read_sql(text(sql), instance, index_col=index_col, chunksize=chunksize, params=params, **kwargs)
        return _query(sql, index_col=index_col, chunksize=chunksize, params=params, **kwargs)

    def connect(self) -> 'SQLAlchemyConnection':
        if False:
            while True:
                i = 10
        'Call ``.connect()`` on the underlying SQLAlchemy Engine, returning a new\n        sqlalchemy.engine.Connection object.\n\n        Calling this method is equivalent to calling ``self._instance.connect()``.\n\n        NOTE: This method should not be confused with the internal _connect method used\n        to implement a Streamlit Connection.\n        '
        return self._instance.connect()

    @property
    def engine(self) -> 'Engine':
        if False:
            return 10
        'The underlying SQLAlchemy Engine.\n\n        This is equivalent to accessing ``self._instance``.\n        '
        return self._instance

    @property
    def driver(self) -> str:
        if False:
            return 10
        'The name of the driver used by the underlying SQLAlchemy Engine.\n\n        This is equivalent to accessing ``self._instance.driver``.\n        '
        return self._instance.driver

    @property
    def session(self) -> 'Session':
        if False:
            for i in range(10):
                print('nop')
        'Return a SQLAlchemy Session.\n\n        Users of this connection should use the contextmanager pattern for writes,\n        transactions, and anything more complex than simple read queries.\n\n        See the usage example below, which assumes we have a table ``numbers`` with a\n        single integer column ``val``. The `SQLAlchemy\n        <https://docs.sqlalchemy.org/en/20/orm/session_basics.html>`_ docs also contain\n        much more information on the usage of sessions.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>> conn = st.connection("sql")\n        >>> n = st.slider("Pick a number")\n        >>> if st.button("Add the number!"):\n        ...     with conn.session as session:\n        ...         session.execute("INSERT INTO numbers (val) VALUES (:n);", {"n": n})\n        ...         session.commit()\n        '
        from sqlalchemy.orm import Session
        return Session(self._instance)

    def _repr_html_(self) -> str:
        if False:
            return 10
        module_name = getattr(self, '__module__', None)
        class_name = type(self).__name__
        cfg = f'- Configured from `[connections.{self._connection_name}]`' if len(self._secrets) else ''
        with self.session as s:
            dialect = s.bind.dialect.name if s.bind is not None else 'unknown'
        return f'\n---\n**st.connection {self._connection_name} built from `{module_name}.{class_name}`**\n{cfg}\n- Dialect: `{dialect}`\n- Learn more using `st.help()`\n---\n'