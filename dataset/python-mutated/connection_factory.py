from __future__ import annotations
import importlib
import os
import re
from datetime import timedelta
from typing import Any, Dict, Type, TypeVar, overload
from typing_extensions import Final, Literal
from streamlit.connections import BaseConnection, SnowflakeConnection, SnowparkConnection, SQLConnection
from streamlit.deprecation_util import deprecate_obj_name
from streamlit.errors import StreamlitAPIException
from streamlit.runtime.caching import cache_resource
from streamlit.runtime.metrics_util import gather_metrics
from streamlit.runtime.secrets import secrets_singleton
FIRST_PARTY_CONNECTIONS = {'snowflake': SnowflakeConnection, 'snowpark': SnowparkConnection, 'sql': SQLConnection}
MODULE_EXTRACTION_REGEX = re.compile("No module named \\'(.+)\\'")
MODULES_TO_PYPI_PACKAGES: Final[Dict[str, str]] = {'MySQLdb': 'mysqlclient', 'psycopg2': 'psycopg2-binary', 'sqlalchemy': 'sqlalchemy', 'snowflake': 'snowflake-connector-python', 'snowflake.connector': 'snowflake-connector-python', 'snowflake.snowpark': 'snowflake-snowpark-python'}
ConnectionClass = TypeVar('ConnectionClass', bound=BaseConnection[Any])

@gather_metrics('connection')
def _create_connection(name: str, connection_class: Type[ConnectionClass], max_entries: int | None=None, ttl: float | timedelta | None=None, **kwargs) -> ConnectionClass:
    if False:
        for i in range(10):
            print('nop')
    'Create an instance of connection_class with the given name and kwargs.\n\n    The weird implementation of this function with the @cache_resource annotated\n    function defined internally is done to:\n      * Always @gather_metrics on the call even if the return value is a cached one.\n      * Allow the user to specify ttl and max_entries when calling st.connection.\n    '

    @cache_resource(max_entries=max_entries, show_spinner='Running `st.connection(...)`.', ttl=ttl)
    def __create_connection(name: str, connection_class: Type[ConnectionClass], **kwargs) -> ConnectionClass:
        if False:
            return 10
        return connection_class(connection_name=name, **kwargs)
    if not issubclass(connection_class, BaseConnection):
        raise StreamlitAPIException(f'{connection_class} is not a subclass of BaseConnection!')
    return __create_connection(name, connection_class, **kwargs)

def _get_first_party_connection(connection_class: str):
    if False:
        i = 10
        return i + 15
    if connection_class in FIRST_PARTY_CONNECTIONS:
        return FIRST_PARTY_CONNECTIONS[connection_class]
    raise StreamlitAPIException(f"Invalid connection '{connection_class}'. Supported connection classes: {FIRST_PARTY_CONNECTIONS}")

@overload
def connection_factory(name: Literal['sql'], max_entries: int | None=None, ttl: float | timedelta | None=None, autocommit: bool=False, **kwargs) -> SQLConnection:
    if False:
        for i in range(10):
            print('nop')
    pass

@overload
def connection_factory(name: str, type: Literal['sql'], max_entries: int | None=None, ttl: float | timedelta | None=None, autocommit: bool=False, **kwargs) -> SQLConnection:
    if False:
        print('Hello World!')
    pass

@overload
def connection_factory(name: Literal['snowflake'], max_entries: int | None=None, ttl: float | timedelta | None=None, autocommit: bool=False, **kwargs) -> SnowflakeConnection:
    if False:
        return 10
    pass

@overload
def connection_factory(name: str, type: Literal['snowflake'], max_entries: int | None=None, ttl: float | timedelta | None=None, autocommit: bool=False, **kwargs) -> SnowflakeConnection:
    if False:
        return 10
    pass

@overload
def connection_factory(name: Literal['snowpark'], max_entries: int | None=None, ttl: float | timedelta | None=None, **kwargs) -> SnowparkConnection:
    if False:
        while True:
            i = 10
    pass

@overload
def connection_factory(name: str, type: Literal['snowpark'], max_entries: int | None=None, ttl: float | timedelta | None=None, **kwargs) -> SnowparkConnection:
    if False:
        print('Hello World!')
    pass

@overload
def connection_factory(name: str, type: Type[ConnectionClass], max_entries: int | None=None, ttl: float | timedelta | None=None, **kwargs) -> ConnectionClass:
    if False:
        return 10
    pass

@overload
def connection_factory(name: str, type: str | None=None, max_entries: int | None=None, ttl: float | timedelta | None=None, **kwargs) -> BaseConnection[Any]:
    if False:
        i = 10
        return i + 15
    pass

def connection_factory(name, type=None, max_entries=None, ttl=None, **kwargs):
    if False:
        i = 10
        return i + 15
    'Create a new connection to a data store or API, or return an existing one.\n\n    Config options, credentials, secrets, etc. for connections are taken from various\n    sources:\n\n    - Any connection-specific configuration files.\n    - An app\'s ``secrets.toml`` files.\n    - The kwargs passed to this function.\n\n    Parameters\n    ----------\n    name : str\n        The connection name used for secrets lookup in ``[connections.<name>]``.\n        Type will be inferred from passing ``"sql"``, ``"snowflake"``, or ``"snowpark"``.\n    type : str, connection class, or None\n        The type of connection to create. It can be a keyword (``"sql"``, ``"snowflake"``,\n        or ``"snowpark"``), a path to an importable class, or an imported class reference.\n        All classes must extend ``st.connections.BaseConnection`` and implement the\n        ``_connect()`` method. If the type kwarg is None, a ``type`` field must be set in\n        the connection\'s section in ``secrets.toml``.\n    max_entries : int or None\n        The maximum number of connections to keep in the cache, or None\n        for an unbounded cache. (When a new entry is added to a full cache,\n        the oldest cached entry will be removed.) The default is None.\n    ttl : float, timedelta, or None\n        The maximum number of seconds to keep results in the cache, or\n        None if cached results should not expire. The default is None.\n    **kwargs : any\n        Additional connection specific kwargs that are passed to the Connection\'s\n        ``_connect()`` method. Learn more from the specific Connection\'s documentation.\n\n    Returns\n    -------\n    Connection object\n        An initialized Connection object of the specified type.\n\n    Examples\n    --------\n    The easiest way to create a first-party (SQL, Snowflake, or Snowpark) connection is\n    to use their default names and define corresponding sections in your ``secrets.toml``\n    file.\n\n    >>> import streamlit as st\n    >>> conn = st.connection("sql") # Config section defined in [connections.sql] in secrets.toml.\n\n    Creating a SQLConnection with a custom name requires you to explicitly specify the\n    type. If type is not passed as a kwarg, it must be set in the appropriate section of\n    ``secrets.toml``.\n\n    >>> import streamlit as st\n    >>> conn1 = st.connection("my_sql_connection", type="sql") # Config section defined in [connections.my_sql_connection].\n    >>> conn2 = st.connection("my_other_sql_connection") # type must be set in [connections.my_other_sql_connection].\n\n    Passing the full module path to the connection class that you want to use can be\n    useful, especially when working with a custom connection:\n\n    >>> import streamlit as st\n    >>> conn = st.connection("my_sql_connection", type="streamlit.connections.SQLConnection")\n\n    Finally, you can pass the connection class to use directly to this function. Doing\n    so allows static type checking tools such as ``mypy`` to infer the exact return\n    type of ``st.connection``.\n\n    >>> import streamlit as st\n    >>> from streamlit.connections import SQLConnection\n    >>> conn = st.connection("my_sql_connection", type=SQLConnection)\n    '
    USE_ENV_PREFIX = 'env:'
    if name.startswith(USE_ENV_PREFIX):
        envvar_name = name[len(USE_ENV_PREFIX):]
        name = os.environ[envvar_name]
    if type is None:
        if name in FIRST_PARTY_CONNECTIONS:
            type = _get_first_party_connection(name)
        else:
            secrets_singleton.load_if_toml_exists()
            type = secrets_singleton['connections'][name]['type']
    connection_class = type
    if isinstance(connection_class, str):
        if '.' in connection_class:
            parts = connection_class.split('.')
            classname = parts.pop()
            connection_module = importlib.import_module('.'.join(parts))
            connection_class = getattr(connection_module, classname)
        else:
            connection_class = _get_first_party_connection(connection_class)
    try:
        conn = _create_connection(name, connection_class, max_entries=max_entries, ttl=ttl, **kwargs)
        if isinstance(conn, SnowparkConnection):
            conn = deprecate_obj_name(conn, 'connection("snowpark")', 'connection("snowflake")', '2024-04-01')
        return conn
    except ModuleNotFoundError as e:
        err_string = str(e)
        missing_module = re.search(MODULE_EXTRACTION_REGEX, err_string)
        extra_info = 'You may be missing a dependency required to use this connection.'
        if missing_module:
            pypi_package = MODULES_TO_PYPI_PACKAGES.get(missing_module.group(1))
            if pypi_package:
                extra_info = f"You need to install the '{pypi_package}' package to use this connection."
        raise ModuleNotFoundError(f'{str(e)}. {extra_info}')