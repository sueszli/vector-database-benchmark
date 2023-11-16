"""Amazon S3 Read PARQUET Module (PRIVATE)."""
import datetime
import functools
import itertools
import logging
import warnings
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Union
import boto3
import pandas as pd
import pyarrow as pa
import pyarrow.dataset
import pyarrow.parquet
from packaging import version
from typing_extensions import Literal
from awswrangler import _data_types, _utils, exceptions
from awswrangler._arrow import _add_table_partitions, _table_to_df
from awswrangler._config import apply_configs
from awswrangler._distributed import engine
from awswrangler._executor import _BaseExecutor, _get_executor
from awswrangler.distributed.ray import ray_get
from awswrangler.s3._fs import open_s3_object
from awswrangler.s3._list import _path2list
from awswrangler.s3._read import _apply_partition_filter, _check_version_id, _extract_partitions_dtypes_from_table_details, _get_path_ignore_suffix, _get_path_root, _get_paths_for_glue_table, _InternalReadTableMetadataReturnValue, _TableMetadataReader
from awswrangler.typing import RayReadParquetSettings, _ReadTableMetadataReturnValue
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
BATCH_READ_BLOCK_SIZE = 65536
CHUNKED_READ_S3_BLOCK_SIZE = 10485760
FULL_READ_S3_BLOCK_SIZE = 20971520
METADATA_READ_S3_BLOCK_SIZE = 131072
_logger: logging.Logger = logging.getLogger(__name__)

def _pyarrow_parquet_file_wrapper(source: Any, coerce_int96_timestamp_unit: Optional[str]=None) -> pyarrow.parquet.ParquetFile:
    if False:
        i = 10
        return i + 15
    try:
        return pyarrow.parquet.ParquetFile(source=source, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)
    except pyarrow.ArrowInvalid as ex:
        if str(ex) == 'Parquet file size is 0 bytes':
            _logger.warning('Ignoring empty file...')
            return None
        raise

@engine.dispatch_on_engine
def _read_parquet_metadata_file(s3_client: Optional['S3Client'], path: str, s3_additional_kwargs: Optional[Dict[str, str]], use_threads: Union[bool, int], version_id: Optional[str]=None, coerce_int96_timestamp_unit: Optional[str]=None) -> pa.schema:
    if False:
        print('Hello World!')
    with open_s3_object(path=path, mode='rb', version_id=version_id, use_threads=use_threads, s3_client=s3_client, s3_block_size=METADATA_READ_S3_BLOCK_SIZE, s3_additional_kwargs=s3_additional_kwargs) as f:
        pq_file: Optional[pyarrow.parquet.ParquetFile] = _pyarrow_parquet_file_wrapper(source=f, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)
        if pq_file:
            return pq_file.schema.to_arrow_schema()
        return None

class _ParquetTableMetadataReader(_TableMetadataReader):

    def _read_metadata_file(self, s3_client: Optional['S3Client'], path: str, s3_additional_kwargs: Optional[Dict[str, str]], use_threads: Union[bool, int], version_id: Optional[str]=None, coerce_int96_timestamp_unit: Optional[str]=None) -> pa.schema:
        if False:
            return 10
        return _read_parquet_metadata_file(s3_client=s3_client, path=path, s3_additional_kwargs=s3_additional_kwargs, use_threads=use_threads, version_id=version_id, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)

def _read_parquet_metadata(path: Union[str, List[str]], path_suffix: Optional[str], path_ignore_suffix: Union[str, List[str], None], ignore_empty: bool, ignore_null: bool, dtype: Optional[Dict[str, str]], sampling: float, dataset: bool, use_threads: Union[bool, int], boto3_session: Optional[boto3.Session], s3_additional_kwargs: Optional[Dict[str, str]], version_id: Optional[Union[str, Dict[str, str]]]=None, coerce_int96_timestamp_unit: Optional[str]=None) -> _InternalReadTableMetadataReturnValue:
    if False:
        return 10
    'Handle wr.s3.read_parquet_metadata internally.'
    reader = _ParquetTableMetadataReader()
    return reader.read_table_metadata(path=path, version_id=version_id, path_suffix=path_suffix, path_ignore_suffix=path_ignore_suffix, ignore_empty=ignore_empty, ignore_null=ignore_null, dtype=dtype, sampling=sampling, dataset=dataset, use_threads=use_threads, s3_additional_kwargs=s3_additional_kwargs, boto3_session=boto3_session, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)

def _read_parquet_file(s3_client: Optional['S3Client'], path: str, path_root: Optional[str], columns: Optional[List[str]], coerce_int96_timestamp_unit: Optional[str], s3_additional_kwargs: Optional[Dict[str, str]], use_threads: Union[bool, int], version_id: Optional[str]=None, schema: Optional[pa.schema]=None) -> pa.Table:
    if False:
        for i in range(10):
            print('nop')
    s3_block_size: int = FULL_READ_S3_BLOCK_SIZE if columns else -1
    with open_s3_object(path=path, mode='rb', version_id=version_id, use_threads=use_threads, s3_block_size=s3_block_size, s3_additional_kwargs=s3_additional_kwargs, s3_client=s3_client) as f:
        if schema and version.parse(pa.__version__) >= version.parse('8.0.0'):
            try:
                table = pyarrow.parquet.read_table(f, columns=columns, schema=schema, use_threads=False, use_pandas_metadata=False, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)
            except pyarrow.ArrowInvalid as ex:
                if 'Parquet file size is 0 bytes' in str(ex):
                    raise exceptions.InvalidFile(f'Invalid Parquet file: {path}')
                raise
        else:
            if schema:
                warnings.warn('Your version of pyarrow does not support reading with schema. Consider an upgrade to pyarrow 8+.', UserWarning)
            pq_file: Optional[pyarrow.parquet.ParquetFile] = _pyarrow_parquet_file_wrapper(source=f, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)
            if pq_file is None:
                raise exceptions.InvalidFile(f'Invalid Parquet file: {path}')
            table = pq_file.read(columns=columns, use_threads=False, use_pandas_metadata=False)
        return _add_table_partitions(table=table, path=path, path_root=path_root)

def _read_parquet_chunked(s3_client: Optional['S3Client'], paths: List[str], path_root: Optional[str], columns: Optional[List[str]], coerce_int96_timestamp_unit: Optional[str], chunked: Union[int, bool], use_threads: Union[bool, int], s3_additional_kwargs: Optional[Dict[str, str]], arrow_kwargs: Dict[str, Any], version_ids: Optional[Dict[str, str]]=None) -> Iterator[pd.DataFrame]:
    if False:
        while True:
            i = 10
    next_slice: Optional[pd.DataFrame] = None
    batch_size = BATCH_READ_BLOCK_SIZE if chunked is True else chunked
    for path in paths:
        with open_s3_object(path=path, version_id=version_ids.get(path) if version_ids else None, mode='rb', use_threads=use_threads, s3_client=s3_client, s3_block_size=CHUNKED_READ_S3_BLOCK_SIZE, s3_additional_kwargs=s3_additional_kwargs) as f:
            pq_file: Optional[pyarrow.parquet.ParquetFile] = _pyarrow_parquet_file_wrapper(source=f, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)
            if pq_file is None:
                continue
            use_threads_flag: bool = use_threads if isinstance(use_threads, bool) else bool(use_threads > 1)
            chunks = pq_file.iter_batches(batch_size=batch_size, columns=columns, use_threads=use_threads_flag, use_pandas_metadata=False)
            table = _add_table_partitions(table=pa.Table.from_batches(chunks, schema=pq_file.schema.to_arrow_schema()), path=path, path_root=path_root)
            df = _table_to_df(table=table, kwargs=arrow_kwargs)
            if chunked is True:
                yield df
            else:
                if next_slice is not None:
                    df = pd.concat(objs=[next_slice, df], sort=False, copy=False)
                while len(df.index) >= chunked:
                    yield df.iloc[:chunked, :].copy()
                    df = df.iloc[chunked:, :]
                if df.empty:
                    next_slice = None
                else:
                    next_slice = df
    if next_slice is not None:
        yield next_slice

@engine.dispatch_on_engine
def _read_parquet(paths: List[str], path_root: Optional[str], schema: Optional[pa.schema], columns: Optional[List[str]], coerce_int96_timestamp_unit: Optional[str], use_threads: Union[bool, int], parallelism: int, version_ids: Optional[Dict[str, str]], s3_client: Optional['S3Client'], s3_additional_kwargs: Optional[Dict[str, Any]], arrow_kwargs: Dict[str, Any], bulk_read: bool) -> pd.DataFrame:
    if False:
        for i in range(10):
            print('nop')
    executor: _BaseExecutor = _get_executor(use_threads=use_threads)
    tables = executor.map(_read_parquet_file, s3_client, paths, itertools.repeat(path_root), itertools.repeat(columns), itertools.repeat(coerce_int96_timestamp_unit), itertools.repeat(s3_additional_kwargs), itertools.repeat(use_threads), [version_ids.get(p) if isinstance(version_ids, dict) else None for p in paths], itertools.repeat(schema))
    return _utils.table_refs_to_df(tables, kwargs=arrow_kwargs)

@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session', 'version_id', 's3_additional_kwargs', 'dtype_backend'])
@apply_configs
def read_parquet(path: Union[str, List[str]], path_root: Optional[str]=None, dataset: bool=False, path_suffix: Union[str, List[str], None]=None, path_ignore_suffix: Union[str, List[str], None]=None, ignore_empty: bool=True, partition_filter: Optional[Callable[[Dict[str, str]], bool]]=None, columns: Optional[List[str]]=None, validate_schema: bool=False, coerce_int96_timestamp_unit: Optional[str]=None, schema: Optional[pa.Schema]=None, last_modified_begin: Optional[datetime.datetime]=None, last_modified_end: Optional[datetime.datetime]=None, version_id: Optional[Union[str, Dict[str, str]]]=None, dtype_backend: Literal['numpy_nullable', 'pyarrow']='numpy_nullable', chunked: Union[bool, int]=False, use_threads: Union[bool, int]=True, ray_args: Optional[RayReadParquetSettings]=None, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None, pyarrow_additional_kwargs: Optional[Dict[str, Any]]=None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        while True:
            i = 10
    'Read Parquet file(s) from an S3 prefix or list of S3 objects paths.\n\n    The concept of `dataset` enables more complex features like partitioning\n    and catalog integration (AWS Glue Catalog).\n\n    This function accepts Unix shell-style wildcards in the path argument.\n    * (matches everything), ? (matches any single character),\n    [seq] (matches any character in seq), [!seq] (matches any character not in seq).\n    If you want to use a path which includes Unix shell-style wildcard characters (`*, ?, []`),\n    you can use `glob.escape(path)` before passing the argument to this function.\n\n    Note\n    ----\n    ``Batching`` (`chunked` argument) (Memory Friendly):\n\n    Used to return an Iterable of DataFrames instead of a regular DataFrame.\n\n    Two batching strategies are available:\n\n    - If **chunked=True**, depending on the size of the data, one or more data frames are returned per file in the path/dataset.\n      Unlike **chunked=INTEGER**, rows from different files are not mixed in the resulting data frames.\n\n    - If **chunked=INTEGER**, awswrangler iterates on the data by number of rows equal to the received INTEGER.\n\n    `P.S.` `chunked=True` is faster and uses less memory while `chunked=INTEGER` is more precise\n    in the number of rows.\n\n    Note\n    ----\n    If `use_threads=True`, the number of threads is obtained from os.cpu_count().\n\n    Note\n    ----\n    Filtering by `last_modified begin` and `last_modified end` is applied after listing all S3 files\n\n    Parameters\n    ----------\n    path : Union[str, List[str]]\n        S3 prefix (accepts Unix shell-style wildcards)\n        (e.g. s3://bucket/prefix) or list of S3 objects paths (e.g. [s3://bucket/key0, s3://bucket/key1]).\n    path_root : str, optional\n        Root path of the dataset. If dataset=`True`, it is used as a starting point to load partition columns.\n    dataset : bool, default False\n        If `True`, read a parquet dataset instead of individual file(s), loading all related partitions as columns.\n    path_suffix : Union[str, List[str], None]\n        Suffix or List of suffixes to be read (e.g. [".gz.parquet", ".snappy.parquet"]).\n        If None, reads all files. (default)\n    path_ignore_suffix : Union[str, List[str], None]\n        Suffix or List of suffixes to be ignored.(e.g. [".csv", "_SUCCESS"]).\n        If None, reads all files. (default)\n    ignore_empty : bool, default True\n        Ignore files with 0 bytes.\n    partition_filter : Callable[[Dict[str, str]], bool], optional\n        Callback Function filters to apply on PARTITION columns (PUSH-DOWN filter).\n        This function must receive a single argument (Dict[str, str]) where keys are partitions\n        names and values are partitions values. Partitions values must be strings and the function\n        must return a bool, True to read the partition or False to ignore it.\n        Ignored if `dataset=False`.\n        E.g ``lambda x: True if x["year"] == "2020" and x["month"] == "1" else False``\n        https://aws-data-wrangler.readthedocs.io/en/3.4.2/tutorials/023%20-%20Flexible%20Partitions%20Filter.html\n    columns : List[str], optional\n        List of columns to read from the file(s).\n    validate_schema : bool, default False\n        Check that the schema is consistent across individual files.\n    coerce_int96_timestamp_unit : str, optional\n        Cast timestamps that are stored in INT96 format to a particular resolution (e.g. "ms").\n        Setting to None is equivalent to "ns" and therefore INT96 timestamps are inferred as in nanoseconds.\n    schema : pyarrow.Schema, optional\n        Schema to use whem reading the file.\n    last_modified_begin : datetime, optional\n        Filter S3 objects by Last modified date.\n        Filter is only applied after listing all objects.\n    last_modified_end : datetime, optional\n        Filter S3 objects by Last modified date.\n        Filter is only applied after listing all objects.\n    version_id: Optional[Union[str, Dict[str, str]]]\n        Version id of the object or mapping of object path to version id.\n        (e.g. {\'s3://bucket/key0\': \'121212\', \'s3://bucket/key1\': \'343434\'})\n    dtype_backend: str, optional\n        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,\n        nullable dtypes are used for all dtypes that have a nullable implementation when\n        “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.\n\n        The dtype_backends are still experimential. The "pyarrow" backend is only supported with Pandas 2.0 or above.\n    chunked : Union[int, bool]\n        If passed, the data is split into an iterable of DataFrames (Memory friendly).\n        If `True` an iterable of DataFrames is returned without guarantee of chunksize.\n        If an `INTEGER` is passed, an iterable of DataFrames is returned with maximum rows\n        equal to the received INTEGER.\n    use_threads : Union[bool, int], default True\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled, os.cpu_count() is used as the max number of threads.\n        If integer is provided, specified number is used.\n    ray_args: typing.RayReadParquetSettings, optional\n        Parameters of the Ray Modin settings. Only used when distributed computing is used with Ray and Modin installed.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session is used if None is received.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forward to S3 botocore requests.\n    pyarrow_additional_kwargs : Dict[str, Any], optional\n        Forwarded to `to_pandas` method converting from PyArrow tables to Pandas DataFrame.\n        Valid values include "split_blocks", "self_destruct", "ignore_metadata".\n        e.g. pyarrow_additional_kwargs={\'split_blocks\': True}.\n\n    Returns\n    -------\n    Union[pandas.DataFrame, Generator[pandas.DataFrame, None, None]]\n        Pandas DataFrame or a Generator in case of `chunked=True`.\n\n    Examples\n    --------\n    Reading all Parquet files under a prefix\n\n    >>> import awswrangler as wr\n    >>> df = wr.s3.read_parquet(path=\'s3://bucket/prefix/\')\n\n    Reading all Parquet files from a list\n\n    >>> import awswrangler as wr\n    >>> df = wr.s3.read_parquet(path=[\'s3://bucket/filename0.parquet\', \'s3://bucket/filename1.parquet\'])\n\n    Reading in chunks (Chunk by file)\n\n    >>> import awswrangler as wr\n    >>> dfs = wr.s3.read_parquet(path=[\'s3://bucket/filename0.parquet\', \'s3://bucket/filename1.parquet\'], chunked=True)\n    >>> for df in dfs:\n    >>>     print(df)  # Smaller Pandas DataFrame\n\n    Reading in chunks (Chunk by 1MM rows)\n\n    >>> import awswrangler as wr\n    >>> dfs = wr.s3.read_parquet(\n    ...     path=[\'s3://bucket/filename0.parquet\', \'s3://bucket/filename1.parquet\'],\n    ...     chunked=1_000_000\n    ... )\n    >>> for df in dfs:\n    >>>     print(df)  # 1MM Pandas DataFrame\n\n    Reading Parquet Dataset with PUSH-DOWN filter over partitions\n\n    >>> import awswrangler as wr\n    >>> my_filter = lambda x: True if x["city"].startswith("new") else False\n    >>> df = wr.s3.read_parquet(path, dataset=True, partition_filter=my_filter)\n\n    '
    ray_args = ray_args if ray_args else {}
    bulk_read = ray_args.get('bulk_read', False)
    if bulk_read and validate_schema:
        exceptions.InvalidArgumentCombination('Cannot validate schema when bulk reading data files.')
    s3_client = _utils.client(service_name='s3', session=boto3_session)
    paths: List[str] = _path2list(path=path, s3_client=s3_client, suffix=path_suffix, ignore_suffix=_get_path_ignore_suffix(path_ignore_suffix=path_ignore_suffix), last_modified_begin=last_modified_begin, last_modified_end=last_modified_end, ignore_empty=ignore_empty, s3_additional_kwargs=s3_additional_kwargs)
    if not path_root:
        path_root = _get_path_root(path=path, dataset=dataset)
    if path_root and partition_filter:
        paths = _apply_partition_filter(path_root=path_root, paths=paths, filter_func=partition_filter)
    if len(paths) < 1:
        raise exceptions.NoFilesFound(f'No files Found on: {path}.')
    version_ids = _check_version_id(paths=paths, version_id=version_id)
    if validate_schema and (not bulk_read):
        metadata_reader = _ParquetTableMetadataReader()
        schema = metadata_reader.validate_schemas(paths=paths, path_root=path_root, columns=columns, validate_schema=validate_schema, s3_client=s3_client, version_ids=version_ids, use_threads=use_threads, s3_additional_kwargs=s3_additional_kwargs, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)
    arrow_kwargs = _data_types.pyarrow2pandas_defaults(use_threads=use_threads, kwargs=pyarrow_additional_kwargs, dtype_backend=dtype_backend)
    if chunked:
        return _read_parquet_chunked(s3_client=s3_client, paths=paths, path_root=path_root, columns=columns, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit, chunked=chunked, use_threads=use_threads, s3_additional_kwargs=s3_additional_kwargs, arrow_kwargs=arrow_kwargs, version_ids=version_ids)
    return _read_parquet(paths, path_root=path_root, schema=schema, columns=columns, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit, use_threads=use_threads, parallelism=ray_args.get('parallelism', -1), s3_client=s3_client, s3_additional_kwargs=s3_additional_kwargs, arrow_kwargs=arrow_kwargs, version_ids=version_ids, bulk_read=bulk_read)

@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session', 's3_additional_kwargs', 'dtype_backend'])
@apply_configs
def read_parquet_table(table: str, database: str, filename_suffix: Union[str, List[str], None]=None, filename_ignore_suffix: Union[str, List[str], None]=None, catalog_id: Optional[str]=None, partition_filter: Optional[Callable[[Dict[str, str]], bool]]=None, columns: Optional[List[str]]=None, validate_schema: bool=True, coerce_int96_timestamp_unit: Optional[str]=None, dtype_backend: Literal['numpy_nullable', 'pyarrow']='numpy_nullable', chunked: Union[bool, int]=False, use_threads: Union[bool, int]=True, ray_args: Optional[RayReadParquetSettings]=None, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None, pyarrow_additional_kwargs: Optional[Dict[str, Any]]=None) -> Union[pd.DataFrame, Iterator[pd.DataFrame]]:
    if False:
        while True:
            i = 10
    'Read Apache Parquet table registered in the AWS Glue Catalog.\n\n    Note\n    ----\n    ``Batching`` (`chunked` argument) (Memory Friendly):\n\n    Used to return an Iterable of DataFrames instead of a regular DataFrame.\n\n    Two batching strategies are available:\n\n    - If **chunked=True**, depending on the size of the data, one or more data frames are returned per file in the path/dataset.\n      Unlike **chunked=INTEGER**, rows from different files will not be mixed in the resulting data frames.\n\n    - If **chunked=INTEGER**, awswrangler will iterate on the data by number of rows equal the received INTEGER.\n\n    `P.S.` `chunked=True` is faster and uses less memory while `chunked=INTEGER` is more precise\n    in the number of rows.\n\n    Note\n    ----\n    If `use_threads=True`, the number of threads is obtained from os.cpu_count().\n\n    Parameters\n    ----------\n    table : str\n        AWS Glue Catalog table name.\n    database : str\n        AWS Glue Catalog database name.\n    filename_suffix : Union[str, List[str], None]\n        Suffix or List of suffixes to be read (e.g. [".gz.parquet", ".snappy.parquet"]).\n        If None, read all files. (default)\n    filename_ignore_suffix : Union[str, List[str], None]\n        Suffix or List of suffixes for S3 keys to be ignored.(e.g. [".csv", "_SUCCESS"]).\n        If None, read all files. (default)\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    partition_filter: Optional[Callable[[Dict[str, str]], bool]]\n        Callback Function filters to apply on PARTITION columns (PUSH-DOWN filter).\n        This function must receive a single argument (Dict[str, str]) where keys are partitions\n        names and values are partitions values. Partitions values must be strings and the function\n        must return a bool, True to read the partition or False to ignore it.\n        Ignored if `dataset=False`.\n        E.g ``lambda x: True if x["year"] == "2020" and x["month"] == "1" else False``\n        https://aws-sdk-pandas.readthedocs.io/en/3.4.2/tutorials/023%20-%20Flexible%20Partitions%20Filter.html\n    columns : List[str], optional\n        List of columns to read from the file(s).\n    validate_schema : bool, default False\n        Check that the schema is consistent across individual files.\n    coerce_int96_timestamp_unit : str, optional\n        Cast timestamps that are stored in INT96 format to a particular resolution (e.g. "ms").\n        Setting to None is equivalent to "ns" and therefore INT96 timestamps are inferred as in nanoseconds.\n    dtype_backend: str, optional\n        Which dtype_backend to use, e.g. whether a DataFrame should have NumPy arrays,\n        nullable dtypes are used for all dtypes that have a nullable implementation when\n        “numpy_nullable” is set, pyarrow is used for all dtypes if “pyarrow” is set.\n\n        The dtype_backends are still experimential. The "pyarrow" backend is only supported with Pandas 2.0 or above.\n    chunked : Union[int, bool]\n        If passed, the data is split into an iterable of DataFrames (Memory friendly).\n        If `True` an iterable of DataFrames is returned without guarantee of chunksize.\n        If an `INTEGER` is passed, an iterable of DataFrames is returned with maximum rows\n        equal to the received INTEGER.\n    use_threads : Union[bool, int], default True\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled, os.cpu_count() is used as the max number of threads.\n        If integer is provided, specified number is used.\n    ray_args: typing.RayReadParquetSettings, optional\n        Parameters of the Ray Modin settings. Only used when distributed computing is used with Ray and Modin installed.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session is used if None is received.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forward to S3 botocore requests.\n    pyarrow_additional_kwargs : Dict[str, Any], optional\n        Forwarded to `to_pandas` method converting from PyArrow tables to Pandas DataFrame.\n        Valid values include "split_blocks", "self_destruct", "ignore_metadata".\n        e.g. pyarrow_additional_kwargs={\'split_blocks\': True}.\n\n    Returns\n    -------\n    Union[pandas.DataFrame, Generator[pandas.DataFrame, None, None]]\n        Pandas DataFrame or a Generator in case of `chunked=True`.\n\n    Examples\n    --------\n    Reading Parquet Table\n\n    >>> import awswrangler as wr\n    >>> df = wr.s3.read_parquet_table(database=\'...\', table=\'...\')\n\n    Reading Parquet Table in chunks (Chunk by file)\n\n    >>> import awswrangler as wr\n    >>> dfs = wr.s3.read_parquet_table(database=\'...\', table=\'...\', chunked=True)\n    >>> for df in dfs:\n    >>>     print(df)  # Smaller Pandas DataFrame\n\n    Reading Parquet Dataset with PUSH-DOWN filter over partitions\n\n    >>> import awswrangler as wr\n    >>> my_filter = lambda x: True if x["city"].startswith("new") else False\n    >>> df = wr.s3.read_parquet_table(path, dataset=True, partition_filter=my_filter)\n\n    '
    paths: Union[str, List[str]]
    path_root: Optional[str]
    (paths, path_root, res) = _get_paths_for_glue_table(table=table, database=database, filename_suffix=filename_suffix, filename_ignore_suffix=filename_ignore_suffix, catalog_id=catalog_id, partition_filter=partition_filter, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs)
    df = read_parquet(path=paths, path_root=path_root, dataset=True, path_suffix=filename_suffix if path_root is None else None, path_ignore_suffix=filename_ignore_suffix if path_root is None else None, columns=columns, validate_schema=validate_schema, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit, dtype_backend=dtype_backend, chunked=chunked, use_threads=use_threads, ray_args=ray_args, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs, pyarrow_additional_kwargs=pyarrow_additional_kwargs)
    partial_cast_function = functools.partial(_data_types.cast_pandas_with_athena_types, dtype=_extract_partitions_dtypes_from_table_details(response=res), dtype_backend=dtype_backend)
    if _utils.is_pandas_frame(df):
        return partial_cast_function(df)
    return map(partial_cast_function, df)

@apply_configs
@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session'])
def read_parquet_metadata(path: Union[str, List[str]], dataset: bool=False, version_id: Optional[Union[str, Dict[str, str]]]=None, path_suffix: Optional[str]=None, path_ignore_suffix: Union[str, List[str], None]=None, ignore_empty: bool=True, ignore_null: bool=False, dtype: Optional[Dict[str, str]]=None, sampling: float=1.0, coerce_int96_timestamp_unit: Optional[str]=None, use_threads: Union[bool, int]=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None) -> _ReadTableMetadataReturnValue:
    if False:
        i = 10
        return i + 15
    'Read Apache Parquet file(s) metadata from an S3 prefix or list of S3 objects paths.\n\n    The concept of `dataset` enables more complex features like partitioning\n    and catalog integration (AWS Glue Catalog).\n\n    This function accepts Unix shell-style wildcards in the path argument.\n    * (matches everything), ? (matches any single character),\n    [seq] (matches any character in seq), [!seq] (matches any character not in seq).\n    If you want to use a path which includes Unix shell-style wildcard characters (`*, ?, []`),\n    you can use `glob.escape(path)` before passing the argument to this function.\n\n    Note\n    ----\n    If `use_threads=True`, the number of threads is obtained from os.cpu_count().\n\n    Parameters\n    ----------\n    path : Union[str, List[str]]\n        S3 prefix (accepts Unix shell-style wildcards)\n        (e.g. s3://bucket/prefix) or list of S3 objects paths (e.g. [s3://bucket/key0, s3://bucket/key1]).\n    dataset : bool, default False\n        If `True`, read a parquet dataset instead of individual file(s), loading all related partitions as columns.\n    version_id : Union[str, Dict[str, str]], optional\n        Version id of the object or mapping of object path to version id.\n        (e.g. {\'s3://bucket/key0\': \'121212\', \'s3://bucket/key1\': \'343434\'})\n    path_suffix : Union[str, List[str], None]\n        Suffix or List of suffixes to be read (e.g. [".gz.parquet", ".snappy.parquet"]).\n        If None, reads all files. (default)\n    path_ignore_suffix : Union[str, List[str], None]\n        Suffix or List of suffixes to be ignored.(e.g. [".csv", "_SUCCESS"]).\n        If None, reads all files. (default)\n    ignore_empty : bool, default True\n        Ignore files with 0 bytes.\n    ignore_null : bool, default False\n        Ignore columns with null type.\n    dtype : Dict[str, str], optional\n        Dictionary of columns names and Athena/Glue types to cast.\n        Use when you have columns with undetermined data types as partitions columns.\n        (e.g. {\'col name\': \'bigint\', \'col2 name\': \'int\'})\n    sampling : float\n        Ratio of files metadata to inspect.\n        Must be `0.0 < sampling <= 1.0`.\n        The higher, the more accurate.\n        The lower, the faster.\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forward to S3 botocore requests.\n\n    Returns\n    -------\n    Tuple[Dict[str, str], Optional[Dict[str, str]]]\n        columns_types: Dictionary with keys as column names and values as\n        data types (e.g. {\'col0\': \'bigint\', \'col1\': \'double\'}). /\n        partitions_types: Dictionary with keys as partition names\n        and values as data types (e.g. {\'col2\': \'date\'}).\n\n    Examples\n    --------\n    Reading all Parquet files (with partitions) metadata under a prefix\n\n    >>> import awswrangler as wr\n    >>> columns_types, partitions_types = wr.s3.read_parquet_metadata(path=\'s3://bucket/prefix/\', dataset=True)\n\n    Reading all Parquet files metadata from a list\n\n    >>> import awswrangler as wr\n    >>> columns_types, partitions_types = wr.s3.read_parquet_metadata(path=[\n    ...     \'s3://bucket/filename0.parquet\',\n    ...     \'s3://bucket/filename1.parquet\'\n    ... ])\n\n    '
    (columns_types, partitions_types, _) = _read_parquet_metadata(path=path, version_id=version_id, path_suffix=path_suffix, path_ignore_suffix=path_ignore_suffix, ignore_empty=ignore_empty, ignore_null=ignore_null, dtype=dtype, sampling=sampling, dataset=dataset, use_threads=use_threads, s3_additional_kwargs=s3_additional_kwargs, boto3_session=boto3_session, coerce_int96_timestamp_unit=coerce_int96_timestamp_unit)
    return _ReadTableMetadataReturnValue(columns_types, partitions_types)