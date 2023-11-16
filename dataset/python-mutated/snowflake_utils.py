import configparser
import os
import random
import shutil
import string
from logging import getLogger
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Iterator, List, Optional, Tuple, cast
import pandas as pd
import pyarrow
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
import feast
from feast.errors import SnowflakeIncompleteConfig, SnowflakeQueryUnknownError
from feast.feature_view import FeatureView
from feast.repo_config import RepoConfig
try:
    import snowflake.connector
    from snowflake.connector import ProgrammingError, SnowflakeConnection
    from snowflake.connector.cursor import SnowflakeCursor
except ImportError as e:
    from feast.errors import FeastExtrasDependencyImportError
    raise FeastExtrasDependencyImportError('snowflake', str(e))
getLogger('snowflake.connector.cursor').disabled = True
getLogger('snowflake.connector.connection').disabled = True
getLogger('snowflake.connector.network').disabled = True
logger = getLogger(__name__)
_cache = {}

class GetSnowflakeConnection:

    def __init__(self, config: str, autocommit=True):
        if False:
            while True:
                i = 10
        self.config = config
        self.autocommit = autocommit

    def __enter__(self):
        if False:
            while True:
                i = 10
        assert self.config.type in ['snowflake.registry', 'snowflake.offline', 'snowflake.engine', 'snowflake.online']
        if self.config.type not in _cache:
            if self.config.type == 'snowflake.registry':
                config_header = 'connections.feast_registry'
            elif self.config.type == 'snowflake.offline':
                config_header = 'connections.feast_offline_store'
            if self.config.type == 'snowflake.engine':
                config_header = 'connections.feast_batch_engine'
            elif self.config.type == 'snowflake.online':
                config_header = 'connections.feast_online_store'
            config_dict = dict(self.config)
            config_reader = configparser.ConfigParser()
            config_reader.read([config_dict['config_path']])
            kwargs: Dict[str, Any] = {}
            if config_reader.has_section(config_header):
                kwargs = dict(config_reader[config_header])
            kwargs.update(((k, v) for (k, v) in config_dict.items() if v is not None))
            for (k, v) in kwargs.items():
                if k in ['role', 'warehouse', 'database', 'schema_']:
                    kwargs[k] = f'"{v}"'
            kwargs['schema'] = kwargs.pop('schema_')
            if 'private_key' in kwargs:
                kwargs['private_key'] = parse_private_key_path(kwargs['private_key'], kwargs['private_key_passphrase'])
            try:
                _cache[self.config.type] = snowflake.connector.connect(application='feast', client_session_keep_alive=True, autocommit=self.autocommit, **kwargs)
                _cache[self.config.type].cursor().execute("ALTER SESSION SET TIMEZONE = 'UTC'", _is_internal=True)
            except KeyError as e:
                raise SnowflakeIncompleteConfig(e)
        self.client = _cache[self.config.type]
        return self.client

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        pass

def assert_snowflake_feature_names(feature_view: FeatureView) -> None:
    if False:
        i = 10
        return i + 15
    for feature in feature_view.features:
        assert feature.name not in ['entity_key', 'feature_name', 'feature_value'], f'Feature Name: {feature.name} is a protected name to ensure query stability'
    return None

def execute_snowflake_statement(conn: SnowflakeConnection, query) -> SnowflakeCursor:
    if False:
        print('Hello World!')
    cursor = conn.cursor().execute(query)
    if cursor is None:
        raise SnowflakeQueryUnknownError(query)
    return cursor

def get_snowflake_online_store_path(config: RepoConfig, feature_view: FeatureView) -> str:
    if False:
        return 10
    path_tag = 'snowflake-online-store/online_path'
    if path_tag in feature_view.tags:
        online_path = feature_view.tags[path_tag]
    else:
        online_path = f'"{config.online_store.database}"."{config.online_store.schema_}"'
    return online_path

def package_snowpark_zip(project_name) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    path = os.path.dirname(feast.__file__)
    copy_path = path + f'/snowflake_feast_{project_name}'
    if os.path.exists(copy_path):
        shutil.rmtree(copy_path)
    copy_files = ['/infra/utils/snowflake/snowpark/snowflake_udfs.py', '/infra/key_encoding_utils.py', '/type_map.py', '/value_type.py', '/protos/feast/types/Value_pb2.py', '/protos/feast/types/EntityKey_pb2.py']
    package_path = copy_path + '/feast'
    for feast_file in copy_files:
        idx = feast_file.rfind('/')
        if idx > -1:
            Path(package_path + feast_file[:idx]).mkdir(parents=True, exist_ok=True)
            feast_file = shutil.copy(path + feast_file, package_path + feast_file[:idx])
        else:
            feast_file = shutil.copy(path + feast_file, package_path + feast_file)
    zip_path = shutil.make_archive(package_path, 'zip', copy_path)
    return (copy_path, zip_path)

def _run_snowflake_field_mapping(snowflake_job_sql: str, field_mapping: dict) -> str:
    if False:
        return 10
    snowflake_mapped_sql = snowflake_job_sql
    for key in field_mapping.keys():
        snowflake_mapped_sql = snowflake_mapped_sql.replace(f'"{key}"', f'"{key}" AS "{field_mapping[key]}"', 1)
    return snowflake_mapped_sql

def write_pandas(conn: SnowflakeConnection, df: pd.DataFrame, table_name: str, database: Optional[str]=None, schema: Optional[str]=None, chunk_size: Optional[int]=None, compression: str='gzip', on_error: str='abort_statement', parallel: int=4, quote_identifiers: bool=True, auto_create_table: bool=False, create_temp_table: bool=False):
    if False:
        print('Hello World!')
    "Allows users to most efficiently write back a pandas DataFrame to Snowflake.\n\n    It works by dumping the DataFrame into Parquet files, uploading them and finally copying their data into the table.\n\n    Returns whether all files were ingested correctly, number of chunks uploaded, and number of rows ingested\n    with all of the COPY INTO command's output for debugging purposes.\n\n        Example usage:\n            import pandas\n            from snowflake.connector.pandas_tools import write_pandas\n\n            df = pandas.DataFrame([('Mark', 10), ('Luke', 20)], columns=['name', 'balance'])\n            success, nchunks, nrows, _ = write_pandas(cnx, df, 'customers')\n\n    Args:\n        conn: Connection to be used to communicate with Snowflake.\n        df: Dataframe we'd like to write back.\n        table_name: Table name where we want to insert into.\n        database: Database table is in, if not provided the connection one will be used.\n        schema: Schema table is in, if not provided the connection one will be used.\n        chunk_size: Number of elements to be inserted once, if not provided all elements will be dumped once\n            (Default value = None).\n        compression: The compression used on the Parquet files, can only be gzip, or snappy. Gzip gives supposedly a\n            better compression, while snappy is faster. Use whichever is more appropriate (Default value = 'gzip').\n        on_error: Action to take when COPY INTO statements fail, default follows documentation at:\n            https://docs.snowflake.com/en/sql-reference/sql/copy-into-table.html#copy-options-copyoptions\n            (Default value = 'abort_statement').\n        parallel: Number of threads to be used when uploading chunks, default follows documentation at:\n            https://docs.snowflake.com/en/sql-reference/sql/put.html#optional-parameters (Default value = 4).\n        quote_identifiers: By default, identifiers, specifically database, schema, table and column names\n            (from df.columns) will be quoted. If set to False, identifiers are passed on to Snowflake without quoting.\n            I.e. identifiers will be coerced to uppercase by Snowflake.  (Default value = True)\n        auto_create_table: When true, will automatically create a table with corresponding columns for each column in\n            the passed in DataFrame. The table will not be created if it already exists\n        create_temp_table: Will make the auto-created table as a temporary table\n    "
    cursor: SnowflakeCursor = conn.cursor()
    stage_name = create_temporary_sfc_stage(cursor)
    upload_df(df, cursor, stage_name, chunk_size, parallel, compression)
    copy_uploaded_data_to_table(cursor, stage_name, list(df.columns), table_name, database, schema, compression, on_error, quote_identifiers, auto_create_table, create_temp_table)

def write_parquet(conn: SnowflakeConnection, path: Path, dataset_schema: pyarrow.Schema, table_name: str, database: Optional[str]=None, schema: Optional[str]=None, compression: str='gzip', on_error: str='abort_statement', parallel: int=4, quote_identifiers: bool=True, auto_create_table: bool=False, create_temp_table: bool=False):
    if False:
        print('Hello World!')
    cursor: SnowflakeCursor = conn.cursor()
    stage_name = create_temporary_sfc_stage(cursor)
    columns = [field.name for field in dataset_schema]
    upload_local_pq(path, cursor, stage_name, parallel)
    copy_uploaded_data_to_table(cursor, stage_name, columns, table_name, database, schema, compression, on_error, quote_identifiers, auto_create_table, create_temp_table)

def copy_uploaded_data_to_table(cursor: SnowflakeCursor, stage_name: str, columns: List[str], table_name: str, database: Optional[str]=None, schema: Optional[str]=None, compression: str='gzip', on_error: str='abort_statement', quote_identifiers: bool=True, auto_create_table: bool=False, create_temp_table: bool=False):
    if False:
        i = 10
        return i + 15
    if database is not None and schema is None:
        raise ProgrammingError('Schema has to be provided to write_pandas when a database is provided')
    compression_map = {'gzip': 'auto', 'snappy': 'snappy'}
    if compression not in compression_map.keys():
        raise ProgrammingError("Invalid compression '{}', only acceptable values are: {}".format(compression, compression_map.keys()))
    if quote_identifiers:
        location = ('"' + database + '".' if database else '') + ('"' + schema + '".' if schema else '') + ('"' + table_name + '"')
    else:
        location = (database + '.' if database else '') + (schema + '.' if schema else '') + table_name
    if quote_identifiers:
        quoted_columns = '"' + '","'.join(columns) + '"'
    else:
        quoted_columns = ','.join(columns)
    if auto_create_table:
        file_format_name = create_file_format(compression, compression_map, cursor)
        infer_schema_sql = f"""SELECT COLUMN_NAME, TYPE FROM table(infer_schema(location=>'@"{stage_name}"', file_format=>'{file_format_name}'))"""
        logger.debug(f"inferring schema with '{infer_schema_sql}'")
        result_cursor = cursor.execute(infer_schema_sql, _is_internal=True)
        if result_cursor is None:
            raise SnowflakeQueryUnknownError(infer_schema_sql)
        result = cast(List[Tuple[str, str]], result_cursor.fetchall())
        column_type_mapping: Dict[str, str] = dict(result)
        quote = '"' if quote_identifiers else ''
        create_table_columns = ', '.join([f'{quote}{c}{quote} {column_type_mapping[c]}' for c in columns])
        create_table_sql = f"CREATE {('TEMP ' if create_temp_table else '')}TABLE IF NOT EXISTS {location} ({create_table_columns}) /* Python:snowflake.connector.pandas_tools.write_pandas() */ "
        logger.debug(f"auto creating table with '{create_table_sql}'")
        cursor.execute(create_table_sql, _is_internal=True)
        drop_file_format_sql = f'DROP FILE FORMAT IF EXISTS {file_format_name}'
        logger.debug(f"dropping file format with '{drop_file_format_sql}'")
        cursor.execute(drop_file_format_sql, _is_internal=True)
    if quote_identifiers:
        parquet_columns = '$1:' + ',$1:'.join((f'"{c}"' for c in columns))
    else:
        parquet_columns = '$1:' + ',$1:'.join(columns)
    copy_into_sql = 'COPY INTO {location} /* Python:snowflake.connector.pandas_tools.write_pandas() */ ({columns}) FROM (SELECT {parquet_columns} FROM @"{stage_name}") FILE_FORMAT=(TYPE=PARQUET COMPRESSION={compression}) PURGE=TRUE ON_ERROR={on_error}'.format(location=location, columns=quoted_columns, parquet_columns=parquet_columns, stage_name=stage_name, compression=compression_map[compression], on_error=on_error)
    logger.debug("copying into with '{}'".format(copy_into_sql))
    result_cursor = cursor.execute(copy_into_sql, _is_internal=True)
    if result_cursor is None:
        raise SnowflakeQueryUnknownError(copy_into_sql)
    result_cursor.close()

def upload_df(df: pd.DataFrame, cursor: SnowflakeCursor, stage_name: str, chunk_size: Optional[int]=None, parallel: int=4, compression: str='gzip'):
    if False:
        while True:
            i = 10
    "\n    Args:\n        df: Dataframe we'd like to write back.\n        cursor: cursor to be used to communicate with Snowflake.\n        stage_name: stage name in Snowflake connection.\n        chunk_size: Number of elements to be inserted once, if not provided all elements will be dumped once\n            (Default value = None).\n        parallel: Number of threads to be used when uploading chunks, default follows documentation at:\n            https://docs.snowflake.com/en/sql-reference/sql/put.html#optional-parameters (Default value = 4).\n        compression: The compression used on the Parquet files, can only be gzip, or snappy. Gzip gives supposedly a\n            better compression, while snappy is faster. Use whichever is more appropriate (Default value = 'gzip').\n\n    "
    if chunk_size is None:
        chunk_size = len(df)
    with TemporaryDirectory() as tmp_folder:
        for (i, chunk) in chunk_helper(df, chunk_size):
            chunk_path = os.path.join(tmp_folder, 'file{}.txt'.format(i))
            chunk.to_parquet(chunk_path, compression=compression, use_deprecated_int96_timestamps=True)
            upload_sql = 'PUT /* Python:feast.infra.utils.snowflake_utils.upload_df() */ \'file://{path}\' @"{stage_name}" PARALLEL={parallel}'.format(path=chunk_path.replace('\\', '\\\\').replace("'", "\\'"), stage_name=stage_name, parallel=parallel)
            logger.debug(f"uploading files with '{upload_sql}'")
            cursor.execute(upload_sql, _is_internal=True)
            os.remove(chunk_path)

def upload_local_pq(path: Path, cursor: SnowflakeCursor, stage_name: str, parallel: int=4):
    if False:
        print('Hello World!')
    '\n    Args:\n        path: Path to parquet dataset on disk\n        cursor: cursor to be used to communicate with Snowflake.\n        stage_name: stage name in Snowflake connection.\n        parallel: Number of threads to be used when uploading chunks, default follows documentation at:\n            https://docs.snowflake.com/en/sql-reference/sql/put.html#optional-parameters (Default value = 4).\n    '
    for file in path.iterdir():
        upload_sql = 'PUT /* Python:feast.infra.utils.snowflake_utils.upload_local_pq() */ \'file://{path}\' @"{stage_name}" PARALLEL={parallel}'.format(path=str(file).replace('\\', '\\\\').replace("'", "\\'"), stage_name=stage_name, parallel=parallel)
        logger.debug(f"uploading files with '{upload_sql}'")
        cursor.execute(upload_sql, _is_internal=True)

@retry(wait=wait_exponential(multiplier=1, max=4), retry=retry_if_exception_type(ProgrammingError), stop=stop_after_attempt(5), reraise=True)
def create_file_format(compression: str, compression_map: Dict[str, str], cursor: SnowflakeCursor) -> str:
    if False:
        while True:
            i = 10
    file_format_name = '"' + ''.join((random.choice(string.ascii_lowercase) for _ in range(5))) + '"'
    file_format_sql = f'CREATE FILE FORMAT {file_format_name} /* Python:snowflake.connector.pandas_tools.write_pandas() */ TYPE=PARQUET COMPRESSION={compression_map[compression]}'
    logger.debug(f"creating file format with '{file_format_sql}'")
    cursor.execute(file_format_sql, _is_internal=True)
    return file_format_name

@retry(wait=wait_exponential(multiplier=1, max=4), retry=retry_if_exception_type(ProgrammingError), stop=stop_after_attempt(5), reraise=True)
def create_temporary_sfc_stage(cursor: SnowflakeCursor) -> str:
    if False:
        print('Hello World!')
    stage_name = ''.join((random.choice(string.ascii_lowercase) for _ in range(5)))
    create_stage_sql = 'create temporary stage /* Python:snowflake.connector.pandas_tools.write_pandas() */ "{stage_name}"'.format(stage_name=stage_name)
    logger.debug(f"creating stage with '{create_stage_sql}'")
    result_cursor = cursor.execute(create_stage_sql, _is_internal=True)
    if result_cursor is None:
        raise SnowflakeQueryUnknownError(create_stage_sql)
    result_cursor.fetchall()
    return stage_name

def chunk_helper(lst: pd.DataFrame, n: int) -> Iterator[Tuple[int, pd.DataFrame]]:
    if False:
        i = 10
        return i + 15
    'Helper generator to chunk a sequence efficiently with current index like if enumerate was called on sequence.'
    for i in range(0, len(lst), n):
        yield (int(i / n), lst[i:i + n])

def parse_private_key_path(key_path: str, private_key_passphrase: str) -> bytes:
    if False:
        i = 10
        return i + 15
    with open(key_path, 'rb') as key:
        p_key = serialization.load_pem_private_key(key.read(), password=private_key_passphrase.encode(), backend=default_backend())
    pkb = p_key.private_bytes(encoding=serialization.Encoding.DER, format=serialization.PrivateFormat.PKCS8, encryption_algorithm=serialization.NoEncryption())
    return pkb

def write_pandas_binary(conn: SnowflakeConnection, df: pd.DataFrame, table_name: str, database: Optional[str]=None, schema: Optional[str]=None, chunk_size: Optional[int]=None, compression: str='gzip', on_error: str='abort_statement', parallel: int=4, quote_identifiers: bool=True, auto_create_table: bool=False, create_temp_table: bool=False):
    if False:
        print('Hello World!')
    "Allows users to most efficiently write back a pandas DataFrame to Snowflake.\n\n    It works by dumping the DataFrame into Parquet files, uploading them and finally copying their data into the table.\n\n    Returns whether all files were ingested correctly, number of chunks uploaded, and number of rows ingested\n    with all of the COPY INTO command's output for debugging purposes.\n\n        Example usage:\n            import pandas\n            from snowflake.connector.pandas_tools import write_pandas\n\n            df = pandas.DataFrame([('Mark', 10), ('Luke', 20)], columns=['name', 'balance'])\n            success, nchunks, nrows, _ = write_pandas(cnx, df, 'customers')\n\n    Args:\n        conn: Connection to be used to communicate with Snowflake.\n        df: Dataframe we'd like to write back.\n        table_name: Table name where we want to insert into.\n        database: Database table is in, if not provided the connection one will be used.\n        schema: Schema table is in, if not provided the connection one will be used.\n        chunk_size: Number of elements to be inserted once, if not provided all elements will be dumped once\n            (Default value = None).\n        compression: The compression used on the Parquet files, can only be gzip, or snappy. Gzip gives supposedly a\n            better compression, while snappy is faster. Use whichever is more appropriate (Default value = 'gzip').\n        on_error: Action to take when COPY INTO statements fail, default follows documentation at:\n            https://docs.snowflake.com/en/sql-reference/sql/copy-into-table.html#copy-options-copyoptions\n            (Default value = 'abort_statement').\n        parallel: Number of threads to be used when uploading chunks, default follows documentation at:\n            https://docs.snowflake.com/en/sql-reference/sql/put.html#optional-parameters (Default value = 4).\n        quote_identifiers: By default, identifiers, specifically database, schema, table and column names\n            (from df.columns) will be quoted. If set to False, identifiers are passed on to Snowflake without quoting.\n            I.e. identifiers will be coerced to uppercase by Snowflake.  (Default value = True)\n        auto_create_table: When true, will automatically create a table with corresponding columns for each column in\n            the passed in DataFrame. The table will not be created if it already exists\n        create_temp_table: Will make the auto-created table as a temporary table\n    "
    if database is not None and schema is None:
        raise ProgrammingError('Schema has to be provided to write_pandas when a database is provided')
    compression_map = {'gzip': 'auto', 'snappy': 'snappy'}
    if compression not in compression_map.keys():
        raise ProgrammingError("Invalid compression '{}', only acceptable values are: {}".format(compression, compression_map.keys()))
    if quote_identifiers:
        location = ('"' + database + '".' if database else '') + ('"' + schema + '".' if schema else '') + ('"' + table_name + '"')
    else:
        location = (database + '.' if database else '') + (schema + '.' if schema else '') + table_name
    if chunk_size is None:
        chunk_size = len(df)
    cursor: SnowflakeCursor = conn.cursor()
    stage_name = create_temporary_sfc_stage(cursor)
    with TemporaryDirectory() as tmp_folder:
        for (i, chunk) in chunk_helper(df, chunk_size):
            chunk_path = os.path.join(tmp_folder, 'file{}.txt'.format(i))
            chunk.to_parquet(chunk_path, compression=compression, use_deprecated_int96_timestamps=True)
            upload_sql = 'PUT /* Python:snowflake.connector.pandas_tools.write_pandas() */ \'file://{path}\' @"{stage_name}" PARALLEL={parallel}'.format(path=chunk_path.replace('\\', '\\\\').replace("'", "\\'"), stage_name=stage_name, parallel=parallel)
            logger.debug(f"uploading files with '{upload_sql}'")
            cursor.execute(upload_sql, _is_internal=True)
            os.remove(chunk_path)
    if quote_identifiers:
        columns = '"' + '","'.join(list(df.columns)) + '"'
    else:
        columns = ','.join(list(df.columns))
    if auto_create_table:
        file_format_name = create_file_format(compression, compression_map, cursor)
        infer_schema_sql = f"""SELECT COLUMN_NAME, TYPE FROM table(infer_schema(location=>'@"{stage_name}"', file_format=>'{file_format_name}'))"""
        logger.debug(f"inferring schema with '{infer_schema_sql}'")
        result_cursor = cursor.execute(infer_schema_sql, _is_internal=True)
        if result_cursor is None:
            raise SnowflakeQueryUnknownError(infer_schema_sql)
        result = cast(List[Tuple[str, str]], result_cursor.fetchall())
        column_type_mapping: Dict[str, str] = dict(result)
        quote = '"' if quote_identifiers else ''
        create_table_columns = ', '.join([f'{quote}{c}{quote} {column_type_mapping[c]}' for c in df.columns])
        create_table_sql = f"CREATE {('TEMP ' if create_temp_table else '')}TABLE IF NOT EXISTS {location} ({create_table_columns}) /* Python:snowflake.connector.pandas_tools.write_pandas() */ "
        logger.debug(f"auto creating table with '{create_table_sql}'")
        cursor.execute(create_table_sql, _is_internal=True)
        drop_file_format_sql = f'DROP FILE FORMAT IF EXISTS {file_format_name}'
        logger.debug(f"dropping file format with '{drop_file_format_sql}'")
        cursor.execute(drop_file_format_sql, _is_internal=True)
    if quote_identifiers:
        parquet_columns = ','.join((f'TO_BINARY($1:"{c}")' if c in ['entity_feature_key', 'entity_key', 'value'] else f'$1:"{c}"' for c in df.columns))
    else:
        parquet_columns = ','.join((f'TO_BINARY($1:{c})' if c in ['entity_feature_key', 'entity_key', 'value'] else f'$1:{c}' for c in df.columns))
    copy_into_sql = 'COPY INTO {location} /* Python:snowflake.connector.pandas_tools.write_pandas() */ ({columns}) FROM (SELECT {parquet_columns} FROM @"{stage_name}") FILE_FORMAT=(TYPE=PARQUET COMPRESSION={compression} BINARY_AS_TEXT = FALSE) PURGE=TRUE ON_ERROR={on_error}'.format(location=location, columns=columns, parquet_columns=parquet_columns, stage_name=stage_name, compression=compression_map[compression], on_error=on_error)
    logger.debug("copying into with '{}'".format(copy_into_sql))
    result_cursor = cursor.execute(copy_into_sql, _is_internal=True)
    if result_cursor is None:
        raise SnowflakeQueryUnknownError(copy_into_sql)
    result_cursor.close()