"""Amazon PostgreSQL Module."""
import logging
import uuid
from ssl import SSLContext
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple, Union, cast, overload
import boto3
import pyarrow as pa
import awswrangler.pandas as pd
from awswrangler import _data_types, _utils, exceptions
from awswrangler import _databases as _db_utils
from awswrangler._config import apply_configs
pg8000 = _utils.import_optional_dependency('pg8000')
pg8000_native = _utils.import_optional_dependency('pg8000.native')
_logger: logging.Logger = logging.getLogger(__name__)

def _validate_connection(con: 'pg8000.Connection') -> None:
    if False:
        i = 10
        return i + 15
    if not isinstance(con, pg8000.Connection):
        raise exceptions.InvalidConnection("Invalid 'conn' argument, please pass a pg8000.Connection object. Use pg8000.connect() to use credentials directly or wr.postgresql.connect() to fetch it from the Glue Catalog.")

def _drop_table(cursor: 'pg8000.Cursor', schema: Optional[str], table: str) -> None:
    if False:
        return 10
    schema_str = f'{pg8000_native.identifier(schema)}.' if schema else ''
    sql = f'DROP TABLE IF EXISTS {schema_str}{pg8000_native.identifier(table)}'
    _logger.debug('Drop table query:\n%s', sql)
    cursor.execute(sql)

def _does_table_exist(cursor: 'pg8000.Cursor', schema: Optional[str], table: str) -> bool:
    if False:
        i = 10
        return i + 15
    schema_str = f'TABLE_SCHEMA = {pg8000_native.literal(schema)} AND' if schema else ''
    cursor.execute(f'SELECT true WHERE EXISTS (SELECT * FROM INFORMATION_SCHEMA.TABLES WHERE {schema_str} TABLE_NAME = {pg8000_native.literal(table)});')
    return len(cursor.fetchall()) > 0

def _create_table(df: pd.DataFrame, cursor: 'pg8000.Cursor', table: str, schema: str, mode: str, index: bool, dtype: Optional[Dict[str, str]], varchar_lengths: Optional[Dict[str, int]]) -> None:
    if False:
        return 10
    if mode == 'overwrite':
        _drop_table(cursor=cursor, schema=schema, table=table)
    elif _does_table_exist(cursor=cursor, schema=schema, table=table):
        return
    postgresql_types: Dict[str, str] = _data_types.database_types_from_pandas(df=df, index=index, dtype=dtype, varchar_lengths_default='TEXT', varchar_lengths=varchar_lengths, converter_func=_data_types.pyarrow2postgresql)
    cols_str: str = ''.join([f'"{k}" {v},\n' for (k, v) in postgresql_types.items()])[:-2]
    sql = f'CREATE TABLE IF NOT EXISTS {pg8000_native.identifier(schema)}.{pg8000_native.identifier(table)} (\n{cols_str})'
    _logger.debug('Create table query:\n%s', sql)
    cursor.execute(sql)

def _iterate_server_side_cursor(sql: str, con: Any, chunksize: int, index_col: Optional[Union[str, List[str]]], params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]], safe: bool, dtype: Optional[Dict[str, pa.DataType]], timestamp_as_object: bool, dtype_backend: Literal['numpy_nullable', 'pyarrow']) -> Iterator[pd.DataFrame]:
    if False:
        return 10
    "\n    Iterate through the results using server-side cursor.\n\n    Note: Pg8000 is not fully DB API 2.0 - compliant with fetchmany() fetching all result set. Using server-side cursor\n    allows fetching only specific amount of results reducing memory impact. Ultimately we'd like pg8000 to add full\n    support for fetchmany() or add SSCursor implementation similar to MySQL and revise this implementation in the future.\n    "
    with con.cursor() as cursor:
        sscursor_name: str = f'c_{uuid.uuid4().hex}'
        cursor_args = _db_utils._convert_params(f'DECLARE {pg8000_native.identifier(sscursor_name)} CURSOR FOR {sql}', params)
        cursor.execute(*cursor_args)
        try:
            while True:
                cursor.execute(f'FETCH FORWARD {pg8000_native.literal(chunksize)} FROM {pg8000_native.identifier(sscursor_name)}')
                records = cursor.fetchall()
                if not records:
                    break
                yield _db_utils._records2df(records=records, cols_names=_db_utils._get_cols_names(cursor.description), index=index_col, safe=safe, dtype=dtype, timestamp_as_object=timestamp_as_object, dtype_backend=dtype_backend)
        finally:
            cursor.execute(f'CLOSE {pg8000_native.identifier(sscursor_name)}')

@_utils.check_optional_dependency(pg8000, 'pg8000')
def connect(connection: Optional[str]=None, secret_id: Optional[str]=None, catalog_id: Optional[str]=None, dbname: Optional[str]=None, boto3_session: Optional[boto3.Session]=None, ssl_context: Optional[Union[bool, SSLContext]]=None, timeout: Optional[int]=None, tcp_keepalive: bool=True) -> 'pg8000.Connection':
    if False:
        return 10
    'Return a pg8000 connection from a Glue Catalog Connection.\n\n    https://github.com/tlocke/pg8000\n\n    Note\n    ----\n    You MUST pass a `connection` OR `secret_id`.\n    Here is an example of the secret structure in Secrets Manager:\n    {\n    "host":"postgresql-instance-wrangler.dr8vkeyrb9m1.us-east-1.rds.amazonaws.com",\n    "username":"test",\n    "password":"test",\n    "engine":"postgresql",\n    "port":"3306",\n    "dbname": "mydb" # Optional\n    }\n\n    Parameters\n    ----------\n    connection : Optional[str]\n        Glue Catalog Connection name.\n    secret_id: Optional[str]:\n        Specifies the secret containing the connection details that you want to retrieve.\n        You can specify either the Amazon Resource Name (ARN) or the friendly name of the secret.\n    catalog_id : str, optional\n        The ID of the Data Catalog.\n        If none is provided, the AWS account ID is used by default.\n    dbname: Optional[str]\n        Optional database name to overwrite the stored one.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    ssl_context: Optional[Union[bool, SSLContext]]\n        This governs SSL encryption for TCP/IP sockets.\n        This parameter is forward to pg8000.\n        https://github.com/tlocke/pg8000#functions\n    timeout: Optional[int]\n        This is the time in seconds before the connection to the server will time out.\n        The default is None which means no timeout.\n        This parameter is forward to pg8000.\n        https://github.com/tlocke/pg8000#functions\n    tcp_keepalive: bool\n        If True then use TCP keepalive. The default is True.\n        This parameter is forward to pg8000.\n        https://github.com/tlocke/pg8000#functions\n\n    Returns\n    -------\n    pg8000.Connection\n        pg8000 connection.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> con = wr.postgresql.connect("MY_GLUE_CONNECTION")\n    >>> with con.cursor() as cursor:\n    >>>     cursor.execute("SELECT 1")\n    >>>     print(cursor.fetchall())\n    >>> con.close()\n\n    '
    attrs: _db_utils.ConnectionAttributes = _db_utils.get_connection_attributes(connection=connection, secret_id=secret_id, catalog_id=catalog_id, dbname=dbname, boto3_session=boto3_session)
    if attrs.kind not in ('postgresql', 'postgres'):
        raise exceptions.InvalidDatabaseType(f'Invalid connection type ({attrs.kind}. It must be a postgresql connection.)')
    return pg8000.connect(user=attrs.user, database=attrs.database, password=attrs.password, port=attrs.port, host=attrs.host, ssl_context=ssl_context, timeout=timeout, tcp_keepalive=tcp_keepalive)

@overload
def read_sql_query(sql: str, con: 'pg8000.Connection', index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: None=..., dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    ...

@overload
def read_sql_query(sql: str, con: 'pg8000.Connection', *, index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: int, dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Iterator[pd.DataFrame]:
    if False:
        print('Hello World!')
    ...

@overload
def read_sql_query(sql: str, con: 'pg8000.Connection', *, index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: Optional[int], dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        for i in range(10):
            print('nop')
    ...

@_utils.check_optional_dependency(pg8000, 'pg8000')
def read_sql_query(sql: str, con: 'pg8000.Connection', index_col: Optional[Union[str, List[str]]]=None, params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=None, chunksize: Optional[int]=None, dtype: Optional[Dict[str, pa.DataType]]=None, safe: bool=True, timestamp_as_object: bool=False, dtype_backend: Literal['numpy_nullable', 'pyarrow']='numpy_nullable') -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        return 10
    'Return a DataFrame corresponding to the result set of the query string.\n\n    Parameters\n    ----------\n    sql : str\n        SQL query.\n    con : pg8000.Connection\n        Use pg8000.connect() to use credentials directly or wr.postgresql.connect() to fetch it from the Glue Catalog.\n    index_col : Union[str, List[str]], optional\n        Column(s) to set as index(MultiIndex).\n    params :  Union[List, Tuple, Dict], optional\n        List of parameters to pass to execute method.\n        The syntax used to pass parameters is database driver dependent.\n        Check your database driver documentation for which of the five syntax styles,\n        described in PEP 249’s paramstyle, is supported.\n    chunksize : int, optional\n        If specified, return an iterator where chunksize is the number of rows to include in each chunk.\n    dtype : Dict[str, pyarrow.DataType], optional\n        Specifying the datatype for columns.\n        The keys should be the column names and the values should be the PyArrow types.\n    safe : bool\n        Check for overflows or other unsafe data type conversions.\n    timestamp_as_object : bool\n        Cast non-nanosecond timestamps (np.datetime64) to objects.\n    dtype_backend: str, optional\n        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,\n        nullable dtypes are used for all dtypes that have a nullable implementation when\n        “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.\n\n        The dtype_backends are still experimential. The "pyarrow" backend is only supported with Pandas 2.0 or above.\n\n    Returns\n    -------\n    Union[pandas.DataFrame, Iterator[pandas.DataFrame]]\n        Result as Pandas DataFrame(s).\n\n    Examples\n    --------\n    Reading from PostgreSQL using a Glue Catalog Connections\n\n    >>> import awswrangler as wr\n    >>> con = wr.postgresql.connect("MY_GLUE_CONNECTION")\n    >>> df = wr.postgresql.read_sql_query(\n    ...     sql="SELECT * FROM public.my_table",\n    ...     con=con\n    ... )\n    >>> con.close()\n\n    '
    _validate_connection(con=con)
    if chunksize is not None:
        return _iterate_server_side_cursor(sql=sql, con=con, chunksize=chunksize, index_col=index_col, params=params, safe=safe, dtype=dtype, timestamp_as_object=timestamp_as_object, dtype_backend=dtype_backend)
    return _db_utils.read_sql_query(sql=sql, con=con, index_col=index_col, params=params, chunksize=None, dtype=dtype, safe=safe, timestamp_as_object=timestamp_as_object, dtype_backend=dtype_backend)

@overload
def read_sql_table(table: str, con: 'pg8000.Connection', schema: Optional[str]=..., index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: None=..., dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    ...

@overload
def read_sql_table(table: str, con: 'pg8000.Connection', *, schema: Optional[str]=..., index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: int, dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Iterator[pd.DataFrame]:
    if False:
        while True:
            i = 10
    ...

@overload
def read_sql_table(table: str, con: 'pg8000.Connection', *, schema: Optional[str]=..., index_col: Optional[Union[str, List[str]]]=..., params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=..., chunksize: Optional[int], dtype: Optional[Dict[str, pa.DataType]]=..., safe: bool=..., timestamp_as_object: bool=..., dtype_backend: Literal['numpy_nullable', 'pyarrow']=...) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        while True:
            i = 10
    ...

@_utils.check_optional_dependency(pg8000, 'pg8000')
def read_sql_table(table: str, con: 'pg8000.Connection', schema: Optional[str]=None, index_col: Optional[Union[str, List[str]]]=None, params: Optional[Union[List[Any], Tuple[Any, ...], Dict[Any, Any]]]=None, chunksize: Optional[int]=None, dtype: Optional[Dict[str, pa.DataType]]=None, safe: bool=True, timestamp_as_object: bool=False, dtype_backend: Literal['numpy_nullable', 'pyarrow']='numpy_nullable') -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        return 10
    'Return a DataFrame corresponding the table.\n\n    Parameters\n    ----------\n    table : str\n        Table name.\n    con : pg8000.Connection\n        Use pg8000.connect() to use credentials directly or wr.postgresql.connect() to fetch it from the Glue Catalog.\n    schema : str, optional\n        Name of SQL schema in database to query (if database flavor supports this).\n        Uses default schema if None (default).\n    index_col : Union[str, List[str]], optional\n        Column(s) to set as index(MultiIndex).\n    params :  Union[List, Tuple, Dict], optional\n        List of parameters to pass to execute method.\n        The syntax used to pass parameters is database driver dependent.\n        Check your database driver documentation for which of the five syntax styles,\n        described in PEP 249’s paramstyle, is supported.\n    chunksize : int, optional\n        If specified, return an iterator where chunksize is the number of rows to include in each chunk.\n    dtype : Dict[str, pyarrow.DataType], optional\n        Specifying the datatype for columns.\n        The keys should be the column names and the values should be the PyArrow types.\n    safe : bool\n        Check for overflows or other unsafe data type conversions.\n    timestamp_as_object : bool\n        Cast non-nanosecond timestamps (np.datetime64) to objects.\n    dtype_backend: str, optional\n        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,\n        nullable dtypes are used for all dtypes that have a nullable implementation when\n        “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.\n\n        The dtype_backends are still experimential. The "pyarrow" backend is only supported with Pandas 2.0 or above.\n\n    Returns\n    -------\n    Union[pandas.DataFrame, Iterator[pandas.DataFrame]]\n        Result as Pandas DataFrame(s).\n\n    Examples\n    --------\n    Reading from PostgreSQL using a Glue Catalog Connections\n\n    >>> import awswrangler as wr\n    >>> con = wr.postgresql.connect("MY_GLUE_CONNECTION")\n    >>> df = wr.postgresql.read_sql_table(\n    ...     table="my_table",\n    ...     schema="public",\n    ...     con=con\n    ... )\n    >>> con.close()\n\n    '
    sql: str = f'SELECT * FROM {pg8000_native.identifier(table)}' if schema is None else f'SELECT * FROM {pg8000_native.identifier(schema)}.{pg8000_native.identifier(table)}'
    return read_sql_query(sql=sql, con=con, index_col=index_col, params=params, chunksize=chunksize, dtype=dtype, safe=safe, timestamp_as_object=timestamp_as_object, dtype_backend=dtype_backend)
_ToSqlModeLiteral = Literal['append', 'overwrite', 'upsert']

@_utils.check_optional_dependency(pg8000, 'pg8000')
@apply_configs
def to_sql(df: pd.DataFrame, con: 'pg8000.Connection', table: str, schema: str, mode: _ToSqlModeLiteral='append', index: bool=False, dtype: Optional[Dict[str, str]]=None, varchar_lengths: Optional[Dict[str, int]]=None, use_column_names: bool=False, chunksize: int=200, upsert_conflict_columns: Optional[List[str]]=None, insert_conflict_columns: Optional[List[str]]=None) -> None:
    if False:
        print('Hello World!')
    'Write records stored in a DataFrame into PostgreSQL.\n\n    Parameters\n    ----------\n    df : pandas.DataFrame\n        Pandas DataFrame https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html\n    con : pg8000.Connection\n        Use pg8000.connect() to use credentials directly or wr.postgresql.connect() to fetch it from the Glue Catalog.\n    table : str\n        Table name\n    schema : str\n        Schema name\n    mode : str\n        Append, overwrite or upsert.\n            append: Inserts new records into table.\n            overwrite: Drops table and recreates.\n            upsert: Perform an upsert which checks for conflicts on columns given by `upsert_conflict_columns` and\n            sets the new values on conflicts. Note that `upsert_conflict_columns` is required for this mode.\n    index : bool\n        True to store the DataFrame index as a column in the table,\n        otherwise False to ignore it.\n    dtype: Dict[str, str], optional\n        Dictionary of columns names and PostgreSQL types to be casted.\n        Useful when you have columns with undetermined or mixed data types.\n        (e.g. {\'col name\': \'TEXT\', \'col2 name\': \'FLOAT\'})\n    varchar_lengths : Dict[str, int], optional\n        Dict of VARCHAR length by columns. (e.g. {"col1": 10, "col5": 200}).\n    use_column_names: bool\n        If set to True, will use the column names of the DataFrame for generating the INSERT SQL Query.\n        E.g. If the DataFrame has two columns `col1` and `col3` and `use_column_names` is True, data will only be\n        inserted into the database columns `col1` and `col3`.\n    chunksize: int\n        Number of rows which are inserted with each SQL query. Defaults to inserting 200 rows per query.\n    upsert_conflict_columns: List[str], optional\n        This parameter is only supported if `mode` is set top `upsert`. In this case conflicts for the given columns are\n        checked for evaluating the upsert.\n    insert_conflict_columns: List[str], optional\n        This parameter is only supported if `mode` is set top `append`. In this case conflicts for the given columns are\n        checked for evaluating the insert \'ON CONFLICT DO NOTHING\'.\n\n    Returns\n    -------\n    None\n        None.\n\n    Examples\n    --------\n    Writing to PostgreSQL using a Glue Catalog Connections\n\n    >>> import awswrangler as wr\n    >>> con = wr.postgresql.connect("MY_GLUE_CONNECTION")\n    >>> wr.postgresql.to_sql(\n    ...     df=df,\n    ...     table="my_table",\n    ...     schema="public",\n    ...     con=con\n    ... )\n    >>> con.close()\n\n    '
    if df.empty is True:
        raise exceptions.EmptyDataFrame('DataFrame cannot be empty.')
    mode = cast(_ToSqlModeLiteral, mode.strip().lower())
    allowed_modes = ['append', 'overwrite', 'upsert']
    _db_utils.validate_mode(mode=mode, allowed_modes=allowed_modes)
    if mode == 'upsert' and (not upsert_conflict_columns):
        raise exceptions.InvalidArgumentValue('<upsert_conflict_columns> needs to be set when using upsert mode.')
    _validate_connection(con=con)
    try:
        with con.cursor() as cursor:
            _create_table(df=df, cursor=cursor, table=table, schema=schema, mode=mode, index=index, dtype=dtype, varchar_lengths=varchar_lengths)
            if index:
                df.reset_index(level=df.index.names, inplace=True)
            column_placeholders: str = ', '.join(['%s'] * len(df.columns))
            column_names = [f'"{column}"' for column in df.columns]
            insertion_columns = ''
            upsert_str = ''
            if use_column_names:
                insertion_columns = f"({', '.join(column_names)})"
            if mode == 'upsert':
                upsert_columns = ', '.join((f'{column}=EXCLUDED.{column}' for column in column_names))
                conflict_columns = ', '.join(upsert_conflict_columns)
                upsert_str = f' ON CONFLICT ({conflict_columns}) DO UPDATE SET {upsert_columns}'
            if mode == 'append' and insert_conflict_columns:
                conflict_columns = ', '.join(insert_conflict_columns)
                upsert_str = f' ON CONFLICT ({conflict_columns}) DO NOTHING'
            placeholder_parameter_pair_generator = _db_utils.generate_placeholder_parameter_pairs(df=df, column_placeholders=column_placeholders, chunksize=chunksize)
            for (placeholders, parameters) in placeholder_parameter_pair_generator:
                sql: str = f'INSERT INTO {pg8000_native.identifier(schema)}.{pg8000_native.identifier(table)} {insertion_columns} VALUES {placeholders}{upsert_str}'
                _logger.debug('sql: %s', sql)
                cursor.executemany(sql, (parameters,))
            con.commit()
    except Exception as ex:
        con.rollback()
        _logger.error(ex)
        raise