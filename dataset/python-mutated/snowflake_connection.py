from __future__ import annotations
from datetime import timedelta
from typing import TYPE_CHECKING, cast
import pandas as pd
from streamlit.connections import BaseConnection
from streamlit.connections.util import running_in_sis
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_data
if TYPE_CHECKING:
    from snowflake.connector import SnowflakeConnection as InternalSnowflakeConnection
    from snowflake.connector.cursor import SnowflakeCursor
    from snowflake.snowpark.session import Session

class SnowflakeConnection(BaseConnection['InternalSnowflakeConnection']):
    """A connection to Snowflake using the Snowflake Python Connector. Initialize using
    ``st.connection("<name>", type="snowflake")``.

    SnowflakeConnection supports direct SQL querying using ``.query("...")``, access to
    the underlying Snowflake Python Connector object with ``.raw_connection``, and other
    convenience functions. See the methods below for more information.
    SnowflakeConnections should always be created using ``st.connection()``, **not**
    initialized directly.
    """

    def _connect(self, **kwargs) -> 'InternalSnowflakeConnection':
        if False:
            return 10
        import snowflake.connector
        from snowflake.connector import Error as SnowflakeError
        from snowflake.snowpark.context import get_active_session
        if running_in_sis():
            session = get_active_session()
            if hasattr(session, 'connection'):
                return session.connection
            return session._conn._conn
        snowflake.connector.paramstyle = 'qmark'
        try:
            st_secrets = self._secrets.to_dict()
            if len(st_secrets):
                conn_kwargs = {**st_secrets, **kwargs}
                return snowflake.connector.connect(**conn_kwargs)
            if hasattr(snowflake.connector.connection, 'CONFIG_MANAGER'):
                config_mgr = snowflake.connector.connection.CONFIG_MANAGER
                default_connection_name = 'default'
                try:
                    default_connection_name = config_mgr['default_connection_name']
                except snowflake.connector.errors.ConfigSourceError:
                    pass
                connection_name = default_connection_name if self._connection_name == 'snowflake' else self._connection_name
                return snowflake.connector.connect(connection_name=connection_name, **kwargs)
            return snowflake.connector.connect(**kwargs)
        except SnowflakeError as e:
            if not len(st_secrets) and (not len(kwargs)):
                raise StreamlitAPIException('Missing Snowflake connection configuration. Did you forget to set this in `secrets.toml`, a Snowflake configuration file, or as kwargs to `st.connection`? See the [SnowflakeConnection configuration documentation](https://docs.streamlit.io/st.connections.snowflakeconnection-configuration) for more details and examples.')
            raise e

    def query(self, sql: str, *, ttl: float | int | timedelta | None=None, show_spinner: bool | str='Running `snowflake.query(...)`.', params=None, **kwargs) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        'Run a read-only SQL query.\n\n        This method implements both query result caching (with caching behavior\n        identical to that of using ``@st.cache_data``) as well as simple error handling/retries.\n\n        .. note::\n            Queries that are run without a specified ttl are cached indefinitely.\n\n        Parameters\n        ----------\n        sql : str\n            The read-only SQL query to execute.\n        ttl : float, int, timedelta or None\n            The maximum number of seconds to keep results in the cache, or\n            None if cached results should not expire. The default is None.\n        show_spinner : boolean or string\n            Enable the spinner. The default is to show a spinner when there is a\n            "cache miss" and the cached resource is being created. If a string, the value\n            of the show_spinner param will be used for the spinner text.\n        params : list, tuple, dict or None\n            List of parameters to pass to the execute method. This connector supports\n            binding data to a SQL statement using qmark bindings. For more information\n            and examples, see the `Snowflake Python Connector documentation\n            <https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-example#using-qmark-or-numeric-binding>`_.\n            Default is None.\n\n        Returns\n        -------\n        pd.DataFrame\n            The result of running the query, formatted as a pandas DataFrame.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> conn = st.connection("snowflake")\n        >>> df = conn.query("select * from pet_owners")\n        >>> st.dataframe(df)\n        '
        from snowflake.connector.errors import ProgrammingError
        from snowflake.connector.network import BAD_REQUEST_GS_CODE, ID_TOKEN_EXPIRED_GS_CODE, MASTER_TOKEN_EXPIRED_GS_CODE, MASTER_TOKEN_INVALD_GS_CODE, MASTER_TOKEN_NOTFOUND_GS_CODE, SESSION_EXPIRED_GS_CODE
        from tenacity import retry, retry_if_exception, stop_after_attempt, wait_fixed
        retryable_error_codes = {int(code) for code in (ID_TOKEN_EXPIRED_GS_CODE, SESSION_EXPIRED_GS_CODE, MASTER_TOKEN_NOTFOUND_GS_CODE, MASTER_TOKEN_EXPIRED_GS_CODE, MASTER_TOKEN_INVALD_GS_CODE, BAD_REQUEST_GS_CODE)}

        @retry(after=lambda _: self.reset(), stop=stop_after_attempt(3), reraise=True, retry=retry_if_exception(lambda e: isinstance(e, ProgrammingError) and hasattr(e, 'errno') and (e.errno in retryable_error_codes)), wait=wait_fixed(1))
        @cache_data(show_spinner=show_spinner, ttl=ttl)
        def _query(sql: str) -> pd.DataFrame:
            if False:
                i = 10
                return i + 15
            cur = self._instance.cursor()
            cur.execute(sql, params=params, **kwargs)
            return cur.fetch_pandas_all()
        return _query(sql)

    def write_pandas(self, df: pd.DataFrame, table_name: str, database: str | None=None, schema: str | None=None, chunk_size: int | None=None, **kwargs) -> tuple[bool, int, int]:
        if False:
            while True:
                i = 10
        'Call snowflake.connector.pandas_tools.write_pandas with this connection.\n\n        This convenience method is simply a thin wrapper around the ``write_pandas``\n        function of the same name from ``snowflake.connector.pandas_tools``. For more\n        information, see the `Snowflake Python Connector documentation\n        <https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#write_pandas>`_.\n\n        Returns\n        -------\n        tuple[bool, int, int]\n            A tuple containing three values:\n                1. A bool that is True if the write was successful.\n                2. An int giving the number of chunks of data that were copied.\n                3. An int giving the number of rows that were inserted.\n        '
        from snowflake.connector.pandas_tools import write_pandas
        (success, nchunks, nrows, _) = write_pandas(conn=self._instance, df=df, table_name=table_name, database=database, schema=schema, chunk_size=chunk_size, **kwargs)
        return (success, nchunks, nrows)

    def cursor(self) -> 'SnowflakeCursor':
        if False:
            for i in range(10):
                print('nop')
        'Return a PEP 249-compliant cursor object.\n\n        For more information, see the `Snowflake Python Connector documentation\n        <https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-api#object-cursor>`_.\n        '
        return self._instance.cursor()

    @property
    def raw_connection(self) -> 'InternalSnowflakeConnection':
        if False:
            for i in range(10):
                print('nop')
        'Access the underlying Snowflake Python connector object.\n\n        Information on how to use the Snowflake Python Connector can be found in the\n        `Snowflake Python Connector documentation <https://docs.snowflake.com/en/developer-guide/python-connector/python-connector-example>`_.\n        '
        return self._instance

    def session(self) -> 'Session':
        if False:
            i = 10
            return i + 15
        'Create a new Snowpark Session from this connection.\n\n        Information on how to use Snowpark sessions can be found in the `Snowpark documentation\n        <https://docs.snowflake.com/en/developer-guide/snowpark/python/working-with-dataframes>`_.\n        '
        from snowflake.snowpark.context import get_active_session
        from snowflake.snowpark.session import Session
        if running_in_sis():
            return get_active_session()
        return cast(Session, Session.builder.configs({'connection': self._instance}).create())