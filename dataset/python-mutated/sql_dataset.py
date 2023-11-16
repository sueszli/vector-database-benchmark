"""``SQLDataSet`` to load and save data to a SQL backend."""
import copy
import re
from pathlib import PurePosixPath
from typing import Any, Dict, NoReturn, Optional
import fsspec
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import NoSuchModuleError
from kedro.io.core import AbstractDataset, DatasetError, get_filepath_str, get_protocol_and_path
__all__ = ['SQLTableDataSet', 'SQLQueryDataSet']
KNOWN_PIP_INSTALL = {'psycopg2': 'psycopg2', 'mysqldb': 'mysqlclient', 'cx_Oracle': 'cx_Oracle'}
DRIVER_ERROR_MESSAGE = '\nA module/driver is missing when connecting to your SQL server. SQLDataSet\n supports SQLAlchemy drivers. Please refer to\n https://docs.sqlalchemy.org/en/13/core/engines.html#supported-databases\n for more information.\n\n\n\n'

def _find_known_drivers(module_import_error: ImportError) -> Optional[str]:
    if False:
        i = 10
        return i + 15
    'Looks up known keywords in a ``ModuleNotFoundError`` so that it can\n    provide better guideline for the user.\n\n    Args:\n        module_import_error: Error raised while connecting to a SQL server.\n\n    Returns:\n        Instructions for installing missing driver. An empty string is\n        returned in case error is related to an unknown driver.\n\n    '
    res = re.findall("'(.*?)'", str(module_import_error.args[0]).lower())
    if not res:
        return None
    missing_module = res[0]
    if KNOWN_PIP_INSTALL.get(missing_module):
        return f'You can also try installing missing driver with\n\npip install {KNOWN_PIP_INSTALL.get(missing_module)}'
    return None

def _get_missing_module_error(import_error: ImportError) -> DatasetError:
    if False:
        i = 10
        return i + 15
    missing_module_instruction = _find_known_drivers(import_error)
    if missing_module_instruction is None:
        return DatasetError(f'{DRIVER_ERROR_MESSAGE}Loading failed with error:\n\n{str(import_error)}')
    return DatasetError(f'{DRIVER_ERROR_MESSAGE}{missing_module_instruction}')

def _get_sql_alchemy_missing_error() -> DatasetError:
    if False:
        return 10
    return DatasetError('The SQL dialect in your connection is not supported by SQLAlchemy. Please refer to https://docs.sqlalchemy.org/en/13/core/engines.html#supported-databases for more information.')

class SQLTableDataSet(AbstractDataset[pd.DataFrame, pd.DataFrame]):
    """``SQLTableDataSet`` loads data from a SQL table and saves a pandas
    dataframe to a table. It uses ``pandas.DataFrame`` internally,
    so it supports all allowed pandas options on ``read_sql_table`` and
    ``to_sql`` methods. Since Pandas uses SQLAlchemy behind the scenes, when
    instantiating ``SQLTableDataSet`` one needs to pass a compatible connection
    string either in ``credentials`` (see the example code snippet below) or in
    ``load_args`` and ``save_args``. Connection string formats supported by
    SQLAlchemy can be found here:
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls

    ``SQLTableDataSet`` modifies the save parameters and stores
    the data with no index. This is designed to make load and save methods
    symmetric.

    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:


    .. code-block:: yaml

        shuttles_table_dataset:
          type: pandas.SQLTableDataSet
          credentials: db_credentials
          table_name: shuttles
          load_args:
            schema: dwschema
          save_args:
            schema: dwschema
            if_exists: replace

    Sample database credentials entry in ``credentials.yml``:

    .. code-block:: yaml

        db_credentials:
          con: postgresql://scott:tiger@localhost/test

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::

        >>> from kedro.extras.datasets.pandas import SQLTableDataSet
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({"col1": [1, 2], "col2": [4, 5],
        >>>                      "col3": [5, 6]})
        >>> table_name = "table_a"
        >>> credentials = {
        >>>     "con": "postgresql://scott:tiger@localhost/test"
        >>> }
        >>> data_set = SQLTableDataSet(table_name=table_name,
        >>>                            credentials=credentials)
        >>>
        >>> data_set.save(data)
        >>> reloaded = data_set.load()
        >>>
        >>> assert data.equals(reloaded)

    """
    DEFAULT_LOAD_ARGS: Dict[str, Any] = {}
    DEFAULT_SAVE_ARGS: Dict[str, Any] = {'index': False}
    engines: Dict[str, Any] = {}

    def __init__(self, table_name: str, credentials: Dict[str, Any], load_args: Dict[str, Any]=None, save_args: Dict[str, Any]=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        "Creates a new ``SQLTableDataSet``.\n\n        Args:\n            table_name: The table name to load or save data to. It\n                overwrites name in ``save_args`` and ``table_name``\n                parameters in ``load_args``.\n            credentials: A dictionary with a ``SQLAlchemy`` connection string.\n                Users are supposed to provide the connection string 'con'\n                through credentials. It overwrites `con` parameter in\n                ``load_args`` and ``save_args`` in case it is provided. To find\n                all supported connection string formats, see here:\n                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls\n            load_args: Provided to underlying pandas ``read_sql_table``\n                function along with the connection string.\n                To find all supported arguments, see here:\n                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html\n                To find all supported connection string formats, see here:\n                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls\n            save_args: Provided to underlying pandas ``to_sql`` function along\n                with the connection string.\n                To find all supported arguments, see here:\n                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.to_sql.html\n                To find all supported connection string formats, see here:\n                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls\n                It has ``index=False`` in the default parameters.\n\n        Raises:\n            DatasetError: When either ``table_name`` or ``con`` is empty.\n        "
        if not table_name:
            raise DatasetError("'table_name' argument cannot be empty.")
        if not (credentials and 'con' in credentials and credentials['con']):
            raise DatasetError("'con' argument cannot be empty. Please provide a SQLAlchemy connection string.")
        self._load_args = copy.deepcopy(self.DEFAULT_LOAD_ARGS)
        if load_args is not None:
            self._load_args.update(load_args)
        self._save_args = copy.deepcopy(self.DEFAULT_SAVE_ARGS)
        if save_args is not None:
            self._save_args.update(save_args)
        self._load_args['table_name'] = table_name
        self._save_args['name'] = table_name
        self._connection_str = credentials['con']
        self.create_connection(self._connection_str)

    @classmethod
    def create_connection(cls, connection_str: str) -> None:
        if False:
            while True:
                i = 10
        'Given a connection string, create singleton connection\n        to be used across all instances of `SQLTableDataSet` that\n        need to connect to the same source.\n        '
        if connection_str in cls.engines:
            return
        try:
            engine = create_engine(connection_str)
        except ImportError as import_error:
            raise _get_missing_module_error(import_error) from import_error
        except NoSuchModuleError as exc:
            raise _get_sql_alchemy_missing_error() from exc
        cls.engines[connection_str] = engine

    def _describe(self) -> Dict[str, Any]:
        if False:
            while True:
                i = 10
        load_args = copy.deepcopy(self._load_args)
        save_args = copy.deepcopy(self._save_args)
        del load_args['table_name']
        del save_args['name']
        return {'table_name': self._load_args['table_name'], 'load_args': load_args, 'save_args': save_args}

    def _load(self) -> pd.DataFrame:
        if False:
            while True:
                i = 10
        engine = self.engines[self._connection_str]
        return pd.read_sql_table(con=engine, **self._load_args)

    def _save(self, data: pd.DataFrame) -> None:
        if False:
            print('Hello World!')
        engine = self.engines[self._connection_str]
        data.to_sql(con=engine, **self._save_args)

    def _exists(self) -> bool:
        if False:
            while True:
                i = 10
        eng = self.engines[self._connection_str]
        schema = self._load_args.get('schema', None)
        exists = self._load_args['table_name'] in eng.table_names(schema)
        return exists

class SQLQueryDataSet(AbstractDataset[None, pd.DataFrame]):
    """``SQLQueryDataSet`` loads data from a provided SQL query. It
    uses ``pandas.DataFrame`` internally, so it supports all allowed
    pandas options on ``read_sql_query``. Since Pandas uses SQLAlchemy behind
    the scenes, when instantiating ``SQLQueryDataSet`` one needs to pass
    a compatible connection string either in ``credentials`` (see the example
    code snippet below) or in ``load_args``. Connection string formats supported
    by SQLAlchemy can be found here:
    https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls

    It does not support save method so it is a read only data set.
    To save data to a SQL server use ``SQLTableDataSet``.


    Example usage for the
    `YAML API <https://kedro.readthedocs.io/en/stable/data/    data_catalog_yaml_examples.html>`_:


    .. code-block:: yaml

        shuttle_id_dataset:
          type: pandas.SQLQueryDataSet
          sql: "select shuttle, shuttle_id from spaceflights.shuttles;"
          credentials: db_credentials

    Advanced example using the ``stream_results`` and ``chunksize`` options to reduce memory usage:

    .. code-block:: yaml

        shuttle_id_dataset:
          type: pandas.SQLQueryDataSet
          sql: "select shuttle, shuttle_id from spaceflights.shuttles;"
          credentials: db_credentials
          execution_options:
            stream_results: true
          load_args:
            chunksize: 1000

    Sample database credentials entry in ``credentials.yml``:

    .. code-block:: yaml

        db_credentials:
          con: postgresql://scott:tiger@localhost/test

    Example usage for the
    `Python API <https://kedro.readthedocs.io/en/stable/data/    advanced_data_catalog_usage.html>`_:
    ::


        >>> from kedro.extras.datasets.pandas import SQLQueryDataSet
        >>> import pandas as pd
        >>>
        >>> data = pd.DataFrame({"col1": [1, 2], "col2": [4, 5],
        >>>                      "col3": [5, 6]})
        >>> sql = "SELECT * FROM table_a"
        >>> credentials = {
        >>>     "con": "postgresql://scott:tiger@localhost/test"
        >>> }
        >>> data_set = SQLQueryDataSet(sql=sql,
        >>>                            credentials=credentials)
        >>>
        >>> sql_data = data_set.load()

    """
    engines: Dict[str, Any] = {}

    def __init__(self, sql: str=None, credentials: Dict[str, Any]=None, load_args: Dict[str, Any]=None, fs_args: Dict[str, Any]=None, filepath: str=None, execution_options: Optional[Dict[str, Any]]=None) -> None:
        if False:
            while True:
                i = 10
        'Creates a new ``SQLQueryDataSet``.\n\n        Args:\n            sql: The sql query statement.\n            credentials: A dictionary with a ``SQLAlchemy`` connection string.\n                Users are supposed to provide the connection string \'con\'\n                through credentials. It overwrites `con` parameter in\n                ``load_args`` and ``save_args`` in case it is provided. To find\n                all supported connection string formats, see here:\n                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls\n            load_args: Provided to underlying pandas ``read_sql_query``\n                function along with the connection string.\n                To find all supported arguments, see here:\n                https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_query.html\n                To find all supported connection string formats, see here:\n                https://docs.sqlalchemy.org/en/13/core/engines.html#database-urls\n            fs_args: Extra arguments to pass into underlying filesystem class constructor\n                (e.g. `{"project": "my-project"}` for ``GCSFileSystem``), as well as\n                to pass to the filesystem\'s `open` method through nested keys\n                `open_args_load` and `open_args_save`.\n                Here you can find all available arguments for `open`:\n                https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.spec.AbstractFileSystem.open\n                All defaults are preserved, except `mode`, which is set to `r` when loading.\n            filepath: A path to a file with a sql query statement.\n            execution_options: A dictionary with non-SQL advanced options for the connection to\n                be applied to the underlying engine. To find all supported execution\n                options, see here:\n                https://docs.sqlalchemy.org/en/12/core/connections.html#sqlalchemy.engine.Connection.execution_options\n                Note that this is not a standard argument supported by pandas API, but could be\n                useful for handling large datasets.\n\n        Raises:\n            DatasetError: When either ``sql`` or ``con`` parameters is empty.\n        '
        if sql and filepath:
            raise DatasetError("'sql' and 'filepath' arguments cannot both be provided.Please only provide one.")
        if not (sql or filepath):
            raise DatasetError("'sql' and 'filepath' arguments cannot both be empty.Please provide a sql query or path to a sql query file.")
        if not (credentials and 'con' in credentials and credentials['con']):
            raise DatasetError("'con' argument cannot be empty. Please provide a SQLAlchemy connection string.")
        default_load_args = {}
        self._load_args = {**default_load_args, **load_args} if load_args is not None else default_load_args
        if sql:
            self._load_args['sql'] = sql
            self._filepath = None
        else:
            _fs_args = copy.deepcopy(fs_args) or {}
            _fs_credentials = _fs_args.pop('credentials', {})
            (protocol, path) = get_protocol_and_path(str(filepath))
            self._protocol = protocol
            self._fs = fsspec.filesystem(self._protocol, **_fs_credentials, **_fs_args)
            self._filepath = path
        self._connection_str = credentials['con']
        self._execution_options = execution_options or {}
        self.create_connection(self._connection_str)

    @classmethod
    def create_connection(cls, connection_str: str) -> None:
        if False:
            while True:
                i = 10
        'Given a connection string, create singleton connection\n        to be used across all instances of `SQLQueryDataSet` that\n        need to connect to the same source.\n        '
        if connection_str in cls.engines:
            return
        try:
            engine = create_engine(connection_str)
        except ImportError as import_error:
            raise _get_missing_module_error(import_error) from import_error
        except NoSuchModuleError as exc:
            raise _get_sql_alchemy_missing_error() from exc
        cls.engines[connection_str] = engine

    def _describe(self) -> Dict[str, Any]:
        if False:
            for i in range(10):
                print('nop')
        load_args = copy.deepcopy(self._load_args)
        return {'sql': str(load_args.pop('sql', None)), 'filepath': str(self._filepath), 'load_args': str(load_args), 'execution_options': str(self._execution_options)}

    def _load(self) -> pd.DataFrame:
        if False:
            print('Hello World!')
        load_args = copy.deepcopy(self._load_args)
        engine = self.engines[self._connection_str].execution_options(**self._execution_options)
        if self._filepath:
            load_path = get_filepath_str(PurePosixPath(self._filepath), self._protocol)
            with self._fs.open(load_path, mode='r') as fs_file:
                load_args['sql'] = fs_file.read()
        return pd.read_sql_query(con=engine, **load_args)

    def _save(self, data: None) -> NoReturn:
        if False:
            i = 10
            return i + 15
        raise DatasetError("'save' is not supported on SQLQueryDataSet")