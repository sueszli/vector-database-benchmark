"""Amazon Microsoft SQL Server Module."""
import logging
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, TypeVar, Union, overload
import boto3
import pyarrow as pa
import awswrangler.pandas as pd
from awswrangler import _data_types, _utils, exceptions
from awswrangler import _databases as _db_utils
from awswrangler._config import apply_configs
__all__ = ['connect', 'read_sql_query', 'read_sql_table', 'to_sql']
pyodbc = _utils.import_optional_dependency('pyodbc')
_logger: logging.Logger = logging.getLogger(__name__)
FuncT = TypeVar('FuncT', bound=Callable[..., Any])

def _validate_connection(con: 'pyodbc.Connection') -> None:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(con, pyodbc.Connection):
        raise exceptions.InvalidConnection("Invalid 'conn' argument, please pass a pyodbc.Connection object. Use pyodbc.connect() to use credentials directly or wr.sqlserver.connect() to fetch it from the Glue Catalog.")

def _get_table_identifier(schema: Optional[str], table: str) -> str:
    if False:
        i = 10
        return i + 15
    schema_str = f'"{schema}".' if schema else ''
    table_identifier = f'{schema_str}"{table}"'
    return table_identifier

def _drop_table(cursor: 'pyodbc.Cursor', schema: Optional[str], table: str) -> None:
    if False:
        i = 10
        return i + 15
    table_identifier = _get_table_identifier(schema, table)
    sql = f"IF OBJECT_ID(N'{table_identifier}', N'U') IS NOT NULL DROP TABLE {table_identifier}"
    _logger.debug('Drop table query:\n%s', sql)
    cursor.execute(sql)

def _does_table_exist(cursor: 'pyodbc.Cursor', schema: Optional[str], table: str) -> bool:
    if False:
        print('Hello World!')
    schema_str = f"TABLE_SCHEMA = '{schema}' AND" if schema else ''
    cursor.execute(f"SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE {schema_str} TABLE_NAME = '{table}'")
    return len(cursor.fetchall()) > 0

def _create_table(df: pd.DataFrame, cursor: 'pyodbc.Cursor', table: str, schema: str, mode: str, index: bool, dtype: Optional[Dict[str, str]], varchar_lengths: Optional[Dict[str, int]]) -> None:
    if False:
        return 10
    if mode == 'overwrite':
        _drop_table(cursor=cursor, schema=schema, table=table)
    elif _does_table_exist(cursor=cursor, schema=schema, table=table):
        return
    sqlserver_types: Dict[str, str] = _data_types.database_types_from_pandas(df=df, index=index, dtype=dtype, varchar_lengths_default='VARCHAR(MAX)', varchar_lengths=varchar_lengths, converter_func=_data_types.pyarrow2sqlserver)
    cols_str: str = ''.join([f'"{k}" {v},\n' for (k, v) in sqlserver_types.items()])[:-2]
    table_identifier = _get_table_identifier(schema, table)
    sql = f"IF OBJECT_ID(N'{table_identifier}', N'U') IS NULL BEGIN CREATE TABLE {table_identifier} (\n{cols_str}); END;"
    _logger.debug('Create table query:\n%s', sql)
    cursor.execute(sql)

@_utils.check_optional_dependency(pyodbc, 'pyodbc')
def connect(connection: Optional[str]=None, secret_id: Optional[str]=None, catalog_id: Optional[str]=None, dbname: Optional[str]=None, odbc_driver_version: int=17, boto3_session: Optional[boto3.Session]=None, timeout: Optional[int]=0) -> 'pyodbc.Connection':
    if False:
        return 10
    'Return a pyodbc connection from a Glue Catalog Connection.\n\n    https://github.com/mkleehammer/pyodbc\n\n    Note\n    ----\n    You MUST pass a `connection` OR `secret_id`.\n    Here is an example of the secret structure in Secrets Manager:\n    {\n    "host":"sqlserver-instance-wrangler.dr8vkeyrb9m1.us-east-1.rds.amazonaws.com",\n    "username":"test",\n    "password":"test",\n    "engine":"sqlserver",\n    "port":"1433",\n    "dbname": "mydb" # Optional\n    }\n\n    Parameters\n    ----------\n    connection : Optional[str]\n        Glue Catalog Connection name.\n    secret_id: Optional[str]:\n        Specifies the secret containing the connection details that you want to retrieve.\n        You can specify either the Amazon Resource Name (ARN) or the friendly name of the secret.\n    catalog_id : str, optional\n        The ID of the Data Catalog.\n        If none is provided, the AWS account ID is used by default.\n    dbname: Optional[str]\n        Optional database name to overwrite the stored one.\n    odbc_driver_version : int\n        Major version of the OBDC Driver version that is installed and should be used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    timeout: Optional[int]\n        This is the time in seconds before the connection to the server will time out.\n        The default is None which means no timeout.\n        This parameter is forwarded to pyodbc.\n        https://github.com/mkleehammer/pyodbc/wiki/The-pyodbc-Module#connect\n\n    Returns\n    -------\n    pyodbc.Connection\n        pyodbc connection.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> con = wr.sqlserver.connect(connection="MY_GLUE_CONNECTION", odbc_driver_version=17)\n    >>> with con.cursor() as cursor:\n    >>>     cursor.execute("SELECT 1")\n    >>>     print(cursor.fetchall())\n    >>> con.close()\n\n    '
    attrs: _db_utils.ConnectionAttributes = _db_utils.get_connection_attributes(connection=connection, secret_id=secret_id, catalog_id=catalog_id, dbname=dbname, boto3_session=boto3_session)
    if attrs.kind != 'sqlserver':
        raise exceptions.InvalidDatabaseType(f'Invalid connection type ({attrs.kind}. It must be a sqlserver connection.)')
    connection_str = f'DRIVER={{ODBC Driver {odbc_driver_version} for SQL Server}};SERVER={attrs.host},{attrs.port};DATABASE={attrs.database};UID={attrs.user};PWD={attrs.password}'
    return pyodbc.connect(connection_str, timeout=timeout)

@overload
def read_sql_query(sql: str, con: 'pyodbc.Connection', index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: None=..., dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    ...

@overload
def read_sql_query(sql: str, con: 'pyodbc.Connection', *, index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: int, dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Iterator[pd.DataFrame]:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def read_sql_query(sql: str, con: 'pyodbc.Connection', *, index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: Optional[int], dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        i = 10
        return i + 15
    ...

@_utils.check_optional_dependency(pyodbc, 'pyodbc')
def read_sql_query(sql: str, con: 'pyodbc.Connection', index_col: Optional[Union[str, List[str]]]=None, params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=None, chunksize: Optional[int]=None, dtype: Optional[Dict[str, pa.DataType]]=None, safe: bool=True, timestamp_as_object: bool=False, dtype_backend: Literal['numpy_nullable', 'pyarrow']='numpy_nullable') -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        while True:
            i = 10
    'Return a DataFrame corresponding to the result set of the query string.\n\n    Parameters\n    ----------\n    sql : str\n        SQL query.\n    con : pyodbc.Connection\n        Use pyodbc.connect() to use credentials directly or wr.sqlserver.connect() to fetch it from the Glue Catalog.\n    index_col : Union[str, List[str]], optional\n        Column(s) to set as index(MultiIndex).\n    params :  Union[List, Tuple, Dict], optional\n        List of parameters to pass to execute method.\n        The syntax used to pass parameters is database driver dependent.\n        Check your database driver documentation for which of the five syntax styles,\n        described in PEP 249’s paramstyle, is supported.\n    chunksize : int, optional\n        If specified, return an iterator where chunksize is the number of rows to include in each chunk.\n    dtype : Dict[str, pyarrow.DataType], optional\n        Specifying the datatype for columns.\n        The keys should be the column names and the values should be the PyArrow types.\n    safe : bool\n        Check for overflows or other unsafe data type conversions.\n    timestamp_as_object : bool\n        Cast non-nanosecond timestamps (np.datetime64) to objects.\n    dtype_backend: str, optional\n        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,\n        nullable dtypes are used for all dtypes that have a nullable implementation when\n        “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.\n\n        The dtype_backends are still experimential. The "pyarrow" backend is only supported with Pandas 2.0 or above.\n\n    Returns\n    -------\n    Union[pandas.DataFrame, Iterator[pandas.DataFrame]]\n        Result as Pandas DataFrame(s).\n\n    Examples\n    --------\n    Reading from Microsoft SQL Server using a Glue Catalog Connections\n\n    >>> import awswrangler as wr\n    >>> con = wr.sqlserver.connect(connection="MY_GLUE_CONNECTION", odbc_driver_version=17)\n    >>> df = wr.sqlserver.read_sql_query(\n    ...     sql="SELECT * FROM dbo.my_table",\n    ...     con=con\n    ... )\n    >>> con.close()\n    '
    _validate_connection(con=con)
    return _db_utils.read_sql_query(sql=sql, con=con, index_col=index_col, params=params, chunksize=chunksize, dtype=dtype, safe=safe, timestamp_as_object=timestamp_as_object, dtype_backend=dtype_backend)

@overload
def read_sql_table(table: str, con: 'pyodbc.Connection', schema: Optional[str]=..., index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: None=..., dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> pd.DataFrame:
    if False:
        print('Hello World!')
    ...

@overload
def read_sql_table(table: str, con: 'pyodbc.Connection', *, schema: Optional[str]=..., index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: int, dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Iterator[pd.DataFrame]:
    if False:
        print('Hello World!')
    ...

@overload
def read_sql_table(table: str, con: 'pyodbc.Connection', *, schema: Optional[str]=..., index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: Optional[int], dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        print('Hello World!')
    ...

@_utils.check_optional_dependency(pyodbc, 'pyodbc')
def read_sql_table(table: str, con: 'pyodbc.Connection', schema: Optional[str]=None, index_col: Optional[Union[str, List[str]]]=None, params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=None, chunksize: Optional[int]=None, dtype: Optional[Dict[str, pa.DataType]]=None, safe: bool=True, timestamp_as_object: bool=False, dtype_backend: Literal['numpy_nullable', 'pyarrow']='numpy_nullable') -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        for i in range(10):
            print('nop')
    'Return a DataFrame corresponding the table.\n\n    Parameters\n    ----------\n    table : str\n        Table name.\n    con : pyodbc.Connection\n        Use pyodbc.connect() to use credentials directly or wr.sqlserver.connect() to fetch it from the Glue Catalog.\n    schema : str, optional\n        Name of SQL schema in database to query (if database flavor supports this).\n        Uses default schema if None (default).\n    index_col : Union[str, List[str]], optional\n        Column(s) to set as index(MultiIndex).\n    params :  Union[List, Tuple, Dict], optional\n        List of parameters to pass to execute method.\n        The syntax used to pass parameters is database driver dependent.\n        Check your database driver documentation for which of the five syntax styles,\n        described in PEP 249’s paramstyle, is supported.\n    chunksize : int, optional\n        If specified, return an iterator where chunksize is the number of rows to include in each chunk.\n    dtype : Dict[str, pyarrow.DataType], optional\n        Specifying the datatype for columns.\n        The keys should be the column names and the values should be the PyArrow types.\n    safe : bool\n        Check for overflows or other unsafe data type conversions.\n    timestamp_as_object : bool\n        Cast non-nanosecond timestamps (np.datetime64) to objects.\n    dtype_backend: str, optional\n        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,\n        nullable dtypes are used for all dtypes that have a nullable implementation when\n        “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.\n\n        The dtype_backends are still experimential. The "pyarrow" backend is only supported with Pandas 2.0 or above.\n\n    Returns\n    -------\n    Union[pandas.DataFrame, Iterator[pandas.DataFrame]]\n        Result as Pandas DataFrame(s).\n\n    Examples\n    --------\n    Reading from Microsoft SQL Server using a Glue Catalog Connections\n\n    >>> import awswrangler as wr\n    >>> con = wr.sqlserver.connect(connection="MY_GLUE_CONNECTION", odbc_driver_version=17)\n    >>> df = wr.sqlserver.read_sql_table(\n    ...     table="my_table",\n    ...     schema="dbo",\n    ...     con=con\n    ... )\n    >>> con.close()\n    '
    table_identifier = _get_table_identifier(schema, table)
    sql: str = f'SELECT * FROM {table_identifier}'
    return read_sql_query(sql=sql, con=con, index_col=index_col, params=params, chunksize=chunksize, dtype=dtype, safe=safe, timestamp_as_object=timestamp_as_object, dtype_backend=dtype_backend)

@_utils.check_optional_dependency(pyodbc, 'pyodbc')
@apply_configs
def to_sql(df: pd.DataFrame, con: 'pyodbc.Connection', table: str, schema: str, mode: Literal['append', 'overwrite']='append', index: bool=False, dtype: Optional[Dict[str, str]]=None, varchar_lengths: Optional[Dict[str, int]]=None, use_column_names: bool=False, chunksize: int=200, fast_executemany: bool=False) -> None:
    if False:
        i = 10
        return i + 15
    'Write records stored in a DataFrame into Microsoft SQL Server.\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        Pandas DataFrame https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html\n    con : pyodbc.Connection\n        Use pyodbc.connect() to use credentials directly or wr.sqlserver.connect() to fetch it from the Glue Catalog.\n    table : str\n        Table name\n    schema : str\n        Schema name\n    mode : str\n        Append or overwrite.\n    index : bool\n        True to store the DataFrame index as a column in the table,\n        otherwise False to ignore it.\n    dtype: Dict[str, str], optional\n        Dictionary of columns names and Microsoft SQL Server types to be casted.\n        Useful when you have columns with undetermined or mixed data types.\n        (e.g. {\'col name\': \'TEXT\', \'col2 name\': \'FLOAT\'})\n    varchar_lengths : Dict[str, int], optional\n        Dict of VARCHAR length by columns. (e.g. {"col1": 10, "col5": 200}).\n    use_column_names: bool\n        If set to True, will use the column names of the DataFrame for generating the INSERT SQL Query.\n        E.g. If the DataFrame has two columns `col1` and `col3` and `use_column_names` is True, data will only be\n        inserted into the database columns `col1` and `col3`.\n    chunksize: int\n        Number of rows which are inserted with each SQL query. Defaults to inserting 200 rows per query.\n    fast_executemany: bool\n        Mode of execution which greatly reduces round trips for a DBAPI executemany() call when using\n        Microsoft ODBC drivers, for limited size batches that fit in memory. `False` by default.\n\n        https://github.com/mkleehammer/pyodbc/wiki/Cursor#executemanysql-params-with-fast_executemanytrue\n\n        Note: when using this mode, pyodbc converts the Python parameter values to their ODBC "C" equivalents,\n        based on the target column types in the database which may lead to subtle data type conversion\n        differences depending on whether fast_executemany is True or False.\n\n    Returns\n    -------\n    None\n        None.\n\n    Examples\n    --------\n    Writing to Microsoft SQL Server using a Glue Catalog Connections\n\n    >>> import awswrangler as wr\n    >>> con = wr.sqlserver.connect(connection="MY_GLUE_CONNECTION", odbc_driver_version=17)\n    >>> wr.sqlserver.to_sql(\n    ...     df=df,\n    ...     table="table",\n    ...     schema="dbo",\n    ...     con=con\n    ... )\n    >>> con.close()\n\n    '
    if df.empty is True:
        raise exceptions.EmptyDataFrame('DataFrame cannot be empty.')
    _validate_connection(con=con)
    try:
        with con.cursor() as cursor:
            if fast_executemany:
                cursor.fast_executemany = True
            _create_table(df=df, cursor=cursor, table=table, schema=schema, mode=mode, index=index, dtype=dtype, varchar_lengths=varchar_lengths)
            if index:
                df.reset_index(level=df.index.names, inplace=True)
            column_placeholders: str = ', '.join(['?'] * len(df.columns))
            table_identifier = _get_table_identifier(schema, table)
            insertion_columns = ''
            if use_column_names:
                quoted_columns = ', '.join((f'"{col}"' for col in df.columns))
                insertion_columns = f'({quoted_columns})'
            placeholder_parameter_pair_generator = _db_utils.generate_placeholder_parameter_pairs(df=df, column_placeholders=column_placeholders, chunksize=chunksize)
            for (placeholders, parameters) in placeholder_parameter_pair_generator:
                sql: str = f'INSERT INTO {table_identifier} {insertion_columns} VALUES {placeholders}'
                _logger.debug('sql: %s', sql)
                cursor.executemany(sql, (parameters,))
            con.commit()
    except Exception as ex:
        con.rollback()
        _logger.error(ex)
        raise