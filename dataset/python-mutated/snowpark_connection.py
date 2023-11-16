import threading
from collections import ChainMap
from contextlib import contextmanager
from datetime import timedelta
from typing import TYPE_CHECKING, Iterator, Optional, Union, cast
import pandas as pd
from streamlit.connections import BaseConnection
from streamlit.connections.util import SNOWSQL_CONNECTION_FILE, load_from_snowsql_config_file, running_in_sis
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_data
if TYPE_CHECKING:
    from snowflake.snowpark.session import Session
_REQUIRED_CONNECTION_PARAMS = {'account'}

class SnowparkConnection(BaseConnection['Session']):
    """A connection to Snowpark using snowflake.snowpark.session.Session. Initialize using
    ``st.connection("<name>", type="snowpark")``.

    In addition to providing access to the Snowpark Session, SnowparkConnection supports
    direct SQL querying using ``query("...")`` and thread safe access using
    ``with conn.safe_session():``. See methods below for more information.
    SnowparkConnections should always be created using ``st.connection()``, **not**
    initialized directly.

    .. note::
        We don't expect this iteration of SnowparkConnection to be able to scale
        well in apps with many concurrent users due to the lock contention that will occur
        over the single underlying Session object under high load.
    """

    def __init__(self, connection_name: str, **kwargs) -> None:
        if False:
            return 10
        self._lock = threading.RLock()
        super().__init__(connection_name, **kwargs)

    def _connect(self, **kwargs) -> 'Session':
        if False:
            print('Hello World!')
        from snowflake.snowpark.context import get_active_session
        from snowflake.snowpark.exceptions import SnowparkSessionException
        from snowflake.snowpark.session import Session
        if running_in_sis():
            return get_active_session()
        conn_params = ChainMap(kwargs, self._secrets.to_dict(), load_from_snowsql_config_file(self._connection_name))
        if not len(conn_params):
            raise StreamlitAPIException(f'Missing Snowpark connection configuration. Did you forget to set this in `secrets.toml`, `{SNOWSQL_CONNECTION_FILE}`, or as kwargs to `st.connection`?')
        for p in _REQUIRED_CONNECTION_PARAMS:
            if p not in conn_params:
                raise StreamlitAPIException(f'Missing Snowpark connection param: {p}')
        return cast(Session, Session.builder.configs(conn_params).create())

    def query(self, sql: str, ttl: Optional[Union[float, int, timedelta]]=None) -> pd.DataFrame:
        if False:
            return 10
        'Run a read-only SQL query.\n\n        This method implements both query result caching (with caching behavior\n        identical to that of using ``@st.cache_data``) as well as simple error handling/retries.\n\n        .. note::\n            Queries that are run without a specified ttl are cached indefinitely.\n\n        Parameters\n        ----------\n        sql : str\n            The read-only SQL query to execute.\n        ttl : float, int, timedelta or None\n            The maximum number of seconds to keep results in the cache, or\n            None if cached results should not expire. The default is None.\n\n        Returns\n        -------\n        pd.DataFrame\n            The result of running the query, formatted as a pandas DataFrame.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> conn = st.connection("snowpark")\n        >>> df = conn.query("select * from pet_owners")\n        >>> st.dataframe(df)\n        '
        from snowflake.snowpark.exceptions import SnowparkServerException
        from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_fixed

        @retry(after=lambda _: self.reset(), stop=stop_after_attempt(3), reraise=True, retry=retry_if_exception_type(SnowparkServerException), wait=wait_fixed(1))
        @cache_data(show_spinner='Running `snowpark.query(...)`.', ttl=ttl)
        def _query(sql: str) -> pd.DataFrame:
            if False:
                i = 10
                return i + 15
            with self._lock:
                return self._instance.sql(sql).to_pandas()
        return _query(sql)

    @property
    def session(self) -> 'Session':
        if False:
            while True:
                i = 10
        'Access the underlying Snowpark session.\n\n        .. note::\n            Snowpark sessions are **not** thread safe. Users of this method are\n            responsible for ensuring that access to the session returned by this method is\n            done in a thread-safe manner. For most users, we recommend using the thread-safe\n            safe_session() method and a ``with`` block.\n\n        Information on how to use Snowpark sessions can be found in the `Snowpark documentation\n        <https://docs.snowflake.com/en/developer-guide/snowpark/python/working-with-dataframes>`_.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> session = st.connection("snowpark").session\n        >>> df = session.table("mytable").limit(10).to_pandas()\n        >>> st.dataframe(df)\n        '
        return self._instance

    @contextmanager
    def safe_session(self) -> Iterator['Session']:
        if False:
            return 10
        'Grab the underlying Snowpark session in a thread-safe manner.\n\n        As operations on a Snowpark session are not thread safe, we need to take care\n        when using a session in the context of a Streamlit app where each script run\n        occurs in its own thread. Using the contextmanager pattern to do this ensures\n        that access on this connection\'s underlying Session is done in a thread-safe\n        manner.\n\n        Information on how to use Snowpark sessions can be found in the `Snowpark documentation\n        <https://docs.snowflake.com/en/developer-guide/snowpark/python/working-with-dataframes>`_.\n\n        Example\n        -------\n        >>> import streamlit as st\n        >>>\n        >>> conn = st.connection("snowpark")\n        >>> with conn.safe_session() as session:\n        ...     df = session.table("mytable").limit(10).to_pandas()\n        ...\n        >>> st.dataframe(df)\n        '
        with self._lock:
            yield self.session