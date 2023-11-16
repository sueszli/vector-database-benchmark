"""Amazon S3 Text Write Module (PRIVATE)."""
import csv
import logging
import uuid
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union, cast
import boto3
import pandas as pd
from pandas.io.common import infer_compression
from awswrangler import _data_types, _utils, catalog, exceptions, lakeformation, typing
from awswrangler._config import apply_configs
from awswrangler._distributed import engine
from awswrangler._utils import copy_df_shallow
from awswrangler.s3._delete import delete_objects
from awswrangler.s3._fs import open_s3_object
from awswrangler.s3._write import _COMPRESSION_2_EXT, _apply_dtype, _sanitize, _validate_args
from awswrangler.s3._write_dataset import _to_dataset
from awswrangler.typing import BucketingInfoTuple, GlueTableSettings, _S3WriteDataReturnValue
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
_logger: logging.Logger = logging.getLogger(__name__)

def _get_write_details(path: str, pandas_kwargs: Dict[str, Any]) -> Tuple[str, Optional[str], Optional[str]]:
    if False:
        while True:
            i = 10
    if pandas_kwargs.get('compression', 'infer') == 'infer':
        pandas_kwargs['compression'] = infer_compression(path, compression='infer')
    mode: str = 'w' if pandas_kwargs.get('compression') is None else 'wb'
    encoding: Optional[str] = pandas_kwargs.get('encoding', 'utf-8')
    newline: Optional[str] = pandas_kwargs.get('lineterminator', '')
    return (mode, encoding, newline)

@engine.dispatch_on_engine
def _to_text(df: pd.DataFrame, file_format: str, use_threads: Union[bool, int], s3_client: Optional['S3Client'], s3_additional_kwargs: Optional[Dict[str, str]], path: Optional[str]=None, path_root: Optional[str]=None, filename_prefix: Optional[str]=None, bucketing: bool=False, **pandas_kwargs: Any) -> List[str]:
    if False:
        while True:
            i = 10
    s3_client = s3_client if s3_client else _utils.client(service_name='s3')
    if df.empty is True:
        _logger.warning('Empty DataFrame will be written.')
    if path is None and path_root is not None:
        file_path: str = f"{path_root}{filename_prefix}.{file_format}{_COMPRESSION_2_EXT.get(pandas_kwargs.get('compression'))}"
    elif path is not None and path_root is None:
        file_path = path
    else:
        raise RuntimeError('path and path_root received at the same time.')
    (mode, encoding, newline) = _get_write_details(path=file_path, pandas_kwargs=pandas_kwargs)
    with open_s3_object(path=file_path, mode=mode, use_threads=use_threads, s3_client=s3_client, s3_additional_kwargs=s3_additional_kwargs, encoding=encoding, newline=newline) as f:
        _logger.debug('pandas_kwargs: %s', pandas_kwargs)
        if file_format == 'csv':
            df.to_csv(f, mode=mode, **pandas_kwargs)
        elif file_format == 'json':
            df.to_json(f, **pandas_kwargs)
    return [file_path]

@apply_configs
@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session'])
def to_csv(df: pd.DataFrame, path: Optional[str]=None, sep: str=',', index: bool=True, columns: Optional[List[str]]=None, use_threads: Union[bool, int]=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None, sanitize_columns: bool=False, dataset: bool=False, filename_prefix: Optional[str]=None, partition_cols: Optional[List[str]]=None, bucketing_info: Optional[BucketingInfoTuple]=None, concurrent_partitioning: bool=False, mode: Optional[Literal['append', 'overwrite', 'overwrite_partitions']]=None, catalog_versioning: bool=False, schema_evolution: bool=False, dtype: Optional[Dict[str, str]]=None, database: Optional[str]=None, table: Optional[str]=None, glue_table_settings: Optional[GlueTableSettings]=None, athena_partition_projection_settings: Optional[typing.AthenaPartitionProjectionSettings]=None, catalog_id: Optional[str]=None, **pandas_kwargs: Any) -> _S3WriteDataReturnValue:
    if False:
        while True:
            i = 10
    'Write CSV file or dataset on Amazon S3.\n\n    The concept of Dataset goes beyond the simple idea of ordinary files and enable more\n    complex features like partitioning and catalog integration (Amazon Athena/AWS Glue Catalog).\n\n    Note\n    ----\n    If database` and `table` arguments are passed, the table name and all column names\n    will be automatically sanitized using `wr.catalog.sanitize_table_name` and `wr.catalog.sanitize_column_name`.\n    Please, pass `sanitize_columns=True` to enforce this behaviour always.\n\n    Note\n    ----\n    If `table` and `database` arguments are passed, `pandas_kwargs` will be ignored due\n    restrictive quoting, date_format, escapechar and encoding required by Athena/Glue Catalog.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    df: pandas.DataFrame\n        Pandas DataFrame https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html\n    path : str, optional\n        Amazon S3 path (e.g. s3://bucket/prefix/filename.csv) (for dataset e.g. ``s3://bucket/prefix``).\n        Required if dataset=False or when creating a new dataset\n    sep : str\n        String of length 1. Field delimiter for the output file.\n    index : bool\n        Write row names (index).\n    columns : Optional[List[str]]\n        Columns to write.\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 Session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forwarded to botocore requests.\n        e.g. s3_additional_kwargs={\'ServerSideEncryption\': \'aws:kms\', \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'}\n    sanitize_columns : bool\n        True to sanitize columns names or False to keep it as is.\n        True value is forced if `dataset=True`.\n    dataset : bool\n        If True store as a dataset instead of ordinary file(s)\n        If True, enable all follow arguments:\n        partition_cols, mode, database, table, description, parameters, columns_comments, concurrent_partitioning,\n        catalog_versioning, projection_params, catalog_id, schema_evolution.\n    filename_prefix: str, optional\n        If dataset=True, add a filename prefix to the output files.\n    partition_cols: List[str], optional\n        List of column names that will be used to create partitions. Only takes effect if dataset=True.\n    bucketing_info: Tuple[List[str], int], optional\n        Tuple consisting of the column names used for bucketing as the first element and the number of buckets as the\n        second element.\n        Only `str`, `int` and `bool` are supported as column data types for bucketing.\n    concurrent_partitioning: bool\n        If True will increase the parallelism level during the partitions writing. It will decrease the\n        writing time and increase the memory usage.\n        https://aws-sdk-pandas.readthedocs.io/en/3.4.2/tutorials/022%20-%20Writing%20Partitions%20Concurrently.html\n    mode : str, optional\n        ``append`` (Default), ``overwrite``, ``overwrite_partitions``. Only takes effect if dataset=True.\n        For details check the related tutorial:\n        https://aws-sdk-pandas.readthedocs.io/en/3.4.2/stubs/awswrangler.s3.to_parquet.html#awswrangler.s3.to_parquet\n    catalog_versioning : bool\n        If True and `mode="overwrite"`, creates an archived version of the table catalog before updating it.\n    schema_evolution : bool\n        If True allows schema evolution (new or missing columns), otherwise a exception will be raised.\n        (Only considered if dataset=True and mode in ("append", "overwrite_partitions")). False by default.\n        Related tutorial:\n        https://aws-sdk-pandas.readthedocs.io/en/3.4.2/tutorials/014%20-%20Schema%20Evolution.html\n    database : str, optional\n        Glue/Athena catalog: Database name.\n    table : str, optional\n        Glue/Athena catalog: Table name.\n    glue_table_settings: dict (GlueTableSettings), optional\n        Settings for writing to the Glue table.\n    dtype : Dict[str, str], optional\n        Dictionary of columns names and Athena/Glue types to be casted.\n        Useful when you have columns with undetermined or mixed data types.\n        (e.g. {\'col name\': \'bigint\', \'col2 name\': \'int\'})\n    athena_partition_projection_settings: typing.AthenaPartitionProjectionSettings, optional\n        Parameters of the Athena Partition Projection (https://docs.aws.amazon.com/athena/latest/ug/partition-projection.html).\n        AthenaPartitionProjectionSettings is a `TypedDict`, meaning the passed parameter can be instantiated either as an\n        instance of AthenaPartitionProjectionSettings or as a regular Python dict.\n\n        Following projection parameters are supported:\n\n        .. list-table:: Projection Parameters\n           :header-rows: 1\n\n           * - Name\n             - Type\n             - Description\n           * - projection_types\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections types.\n               Valid types: "enum", "integer", "date", "injected"\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'enum\', \'col2_name\': \'integer\'})\n           * - projection_ranges\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections ranges.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'0,10\', \'col2_name\': \'-1,8675309\'})\n           * - projection_values\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections values.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'A,B,Unknown\', \'col2_name\': \'foo,boo,bar\'})\n           * - projection_intervals\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections intervals.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'1\', \'col2_name\': \'5\'})\n           * - projection_digits\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections digits.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'1\', \'col2_name\': \'2\'})\n           * - projection_formats\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections formats.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_date\': \'yyyy-MM-dd\', \'col2_timestamp\': \'yyyy-MM-dd HH:mm:ss\'})\n           * - projection_storage_location_template\n             - Optional[str]\n             - Value which is allows Athena to properly map partition values if the S3 file locations do not follow\n               a typical `.../column=value/...` pattern.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-setting-up.html\n               (e.g. s3://bucket/table_root/a=${a}/${b}/some_static_subdirectory/${c}/)\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    pandas_kwargs :\n        KEYWORD arguments forwarded to pandas.DataFrame.to_csv(). You can NOT pass `pandas_kwargs` explicit, just add\n        valid Pandas arguments in the function call and awswrangler will accept it.\n        e.g. wr.s3.to_csv(df, path, sep=\'|\', na_rep=\'NULL\', decimal=\',\')\n        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_csv.html\n\n    Returns\n    -------\n    wr.typing._S3WriteDataReturnValue\n        Dictionary with:\n        \'paths\': List of all stored files paths on S3.\n        \'partitions_values\': Dictionary of partitions added with keys as S3 path locations\n        and values as a list of partitions values as str.\n\n    Examples\n    --------\n    Writing single file\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\'col\': [1, 2, 3]}),\n    ...     path=\'s3://bucket/prefix/my_file.csv\',\n    ... )\n    {\n        \'paths\': [\'s3://bucket/prefix/my_file.csv\'],\n        \'partitions_values\': {}\n    }\n\n    Writing single file with pandas_kwargs\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\'col\': [1, 2, 3]}),\n    ...     path=\'s3://bucket/prefix/my_file.csv\',\n    ...     sep=\'|\',\n    ...     na_rep=\'NULL\',\n    ...     decimal=\',\'\n    ... )\n    {\n        \'paths\': [\'s3://bucket/prefix/my_file.csv\'],\n        \'partitions_values\': {}\n    }\n\n    Writing single file encrypted with a KMS key\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\'col\': [1, 2, 3]}),\n    ...     path=\'s3://bucket/prefix/my_file.csv\',\n    ...     s3_additional_kwargs={\n    ...         \'ServerSideEncryption\': \'aws:kms\',\n    ...         \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'\n    ...     }\n    ... )\n    {\n        \'paths\': [\'s3://bucket/prefix/my_file.csv\'],\n        \'partitions_values\': {}\n    }\n\n    Writing partitioned dataset\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\n    ...         \'col\': [1, 2, 3],\n    ...         \'col2\': [\'A\', \'A\', \'B\']\n    ...     }),\n    ...     path=\'s3://bucket/prefix\',\n    ...     dataset=True,\n    ...     partition_cols=[\'col2\']\n    ... )\n    {\n        \'paths\': [\'s3://.../col2=A/x.csv\', \'s3://.../col2=B/y.csv\'],\n        \'partitions_values: {\n            \'s3://.../col2=A/\': [\'A\'],\n            \'s3://.../col2=B/\': [\'B\']\n        }\n    }\n\n    Writing partitioned dataset with partition projection\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> from datetime import datetime\n    >>> dt = lambda x: datetime.strptime(x, "%Y-%m-%d").date()\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\n    ...         "id": [1, 2, 3],\n    ...         "value": [1000, 1001, 1002],\n    ...         "category": [\'A\', \'B\', \'C\'],\n    ...     }),\n    ...     path=\'s3://bucket/prefix\',\n    ...     dataset=True,\n    ...     partition_cols=[\'value\', \'category\'],\n    ...     athena_partition_projection_settings={\n    ...        "projection_types": {\n    ...             "value": "integer",\n    ...             "category": "enum",\n    ...         },\n    ...         "projection_ranges": {\n    ...             "value": "1000,2000",\n    ...             "category": "A,B,C",\n    ...         },\n    ...     },\n    ... )\n    {\n        \'paths\': [\n            \'s3://.../value=1000/category=A/x.json\', ...\n        ],\n        \'partitions_values\': {\n            \'s3://.../value=1000/category=A/\': [\n                \'1000\',\n                \'A\',\n            ], ...\n        }\n    }\n\n    Writing bucketed dataset\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\n    ...         \'col\': [1, 2, 3],\n    ...         \'col2\': [\'A\', \'A\', \'B\']\n    ...     }),\n    ...     path=\'s3://bucket/prefix\',\n    ...     dataset=True,\n    ...     bucketing_info=(["col2"], 2)\n    ... )\n    {\n        \'paths\': [\'s3://.../x_bucket-00000.csv\', \'s3://.../col2=B/x_bucket-00001.csv\'],\n        \'partitions_values: {}\n    }\n\n    Writing dataset to S3 with metadata on Athena/Glue Catalog.\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\n    ...         \'col\': [1, 2, 3],\n    ...         \'col2\': [\'A\', \'A\', \'B\']\n    ...     }),\n    ...     path=\'s3://bucket/prefix\',\n    ...     dataset=True,\n    ...     partition_cols=[\'col2\'],\n    ...     database=\'default\',  # Athena/Glue database\n    ...     table=\'my_table\'  # Athena/Glue table\n    ... )\n    {\n        \'paths\': [\'s3://.../col2=A/x.csv\', \'s3://.../col2=B/y.csv\'],\n        \'partitions_values: {\n            \'s3://.../col2=A/\': [\'A\'],\n            \'s3://.../col2=B/\': [\'B\']\n        }\n    }\n\n    Writing dataset to Glue governed table\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\n    ...         \'col\': [1, 2, 3],\n    ...         \'col2\': [\'A\', \'A\', \'B\'],\n    ...         \'col3\': [None, None, None]\n    ...     }),\n    ...     dataset=True,\n    ...     mode=\'append\',\n    ...     database=\'default\',  # Athena/Glue database\n    ...     table=\'my_table\',  # Athena/Glue table\n    ...     glue_table_settings=wr.typing.GlueTableSettings(\n    ...         table_type="GOVERNED",\n    ...         transaction_id="xxx",\n    ...     ),\n    ... )\n    {\n        \'paths\': [\'s3://.../x.csv\'],\n        \'partitions_values: {}\n    }\n\n    Writing dataset casting empty column data type\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_csv(\n    ...     df=pd.DataFrame({\n    ...         \'col\': [1, 2, 3],\n    ...         \'col2\': [\'A\', \'A\', \'B\'],\n    ...         \'col3\': [None, None, None]\n    ...     }),\n    ...     path=\'s3://bucket/prefix\',\n    ...     dataset=True,\n    ...     database=\'default\',  # Athena/Glue database\n    ...     table=\'my_table\'  # Athena/Glue table\n    ...     dtype={\'col3\': \'date\'}\n    ... )\n    {\n        \'paths\': [\'s3://.../x.csv\'],\n        \'partitions_values: {}\n    }\n\n    '
    if 'pandas_kwargs' in pandas_kwargs:
        raise exceptions.InvalidArgument("You can NOT pass `pandas_kwargs` explicit, just add valid Pandas arguments in the function call and awswrangler will accept it.e.g. wr.s3.to_csv(df, path, sep='|', na_rep='NULL', decimal=',', compression='gzip')")
    glue_table_settings = cast(GlueTableSettings, glue_table_settings if glue_table_settings else {})
    table_type = glue_table_settings.get('table_type')
    transaction_id = glue_table_settings.get('transaction_id')
    description = glue_table_settings.get('description')
    parameters = glue_table_settings.get('parameters')
    columns_comments = glue_table_settings.get('columns_comments')
    regular_partitions = glue_table_settings.get('regular_partitions', True)
    _validate_args(df=df, table=table, database=database, dataset=dataset, path=path, partition_cols=partition_cols, bucketing_info=bucketing_info, mode=mode, description=description, parameters=parameters, columns_comments=columns_comments, execution_engine=engine.get())
    partition_cols = partition_cols if partition_cols else []
    dtype = dtype if dtype else {}
    partitions_values: Dict[str, List[str]] = {}
    mode = 'append' if mode is None else mode
    commit_trans: bool = False
    if transaction_id:
        table_type = 'GOVERNED'
    filename_prefix = filename_prefix + uuid.uuid4().hex if filename_prefix else uuid.uuid4().hex
    s3_client = _utils.client(service_name='s3', session=boto3_session)
    if sanitize_columns is True or (database is not None and table is not None):
        (df, dtype, partition_cols, bucketing_info) = _sanitize(df=copy_df_shallow(df), dtype=dtype, partition_cols=partition_cols, bucketing_info=bucketing_info)
    catalog_table_input: Optional[Dict[str, Any]] = None
    if database and table:
        catalog_table_input = catalog._get_table_input(database=database, table=table, boto3_session=boto3_session, transaction_id=transaction_id, catalog_id=catalog_id)
        catalog_path: Optional[str] = None
        if catalog_table_input:
            table_type = catalog_table_input['TableType']
            catalog_path = catalog_table_input.get('StorageDescriptor', {}).get('Location')
        if path is None:
            if catalog_path:
                path = catalog_path
            else:
                raise exceptions.InvalidArgumentValue('Glue table does not exist in the catalog. Please pass the `path` argument to create it.')
        elif path and catalog_path:
            if path.rstrip('/') != catalog_path.rstrip('/'):
                raise exceptions.InvalidArgumentValue(f'The specified path: {path}, does not match the existing Glue catalog table path: {catalog_path}')
        if pandas_kwargs.get('compression') not in ('gzip', 'bz2', None):
            raise exceptions.InvalidArgumentCombination('If database and table are given, you must use one of these compressions: gzip, bz2 or None.')
        if table_type == 'GOVERNED' and (not transaction_id):
            _logger.debug('`transaction_id` not specified for GOVERNED table, starting transaction')
            transaction_id = lakeformation.start_transaction(read_only=False, boto3_session=boto3_session)
            commit_trans = True
    df = _apply_dtype(df=df, dtype=dtype, catalog_table_input=catalog_table_input, mode=mode)
    paths: List[str] = []
    if dataset is False:
        pandas_kwargs['sep'] = sep
        pandas_kwargs['index'] = index
        pandas_kwargs['columns'] = columns
        _to_text(df, file_format='csv', use_threads=use_threads, path=path, s3_client=s3_client, s3_additional_kwargs=s3_additional_kwargs, **pandas_kwargs)
        paths = [path]
    else:
        compression: Optional[str] = pandas_kwargs.get('compression', None)
        if database and table:
            quoting: Optional[int] = csv.QUOTE_NONE
            escapechar: Optional[str] = '\\'
            header: Union[bool, List[str]] = pandas_kwargs.get('header', False)
            date_format: Optional[str] = '%Y-%m-%d %H:%M:%S.%f'
            pd_kwargs: Dict[str, Any] = {}
        else:
            quoting = pandas_kwargs.get('quoting', None)
            escapechar = pandas_kwargs.get('escapechar', None)
            header = pandas_kwargs.get('header', True)
            date_format = pandas_kwargs.get('date_format', None)
            pd_kwargs = pandas_kwargs.copy()
            pd_kwargs.pop('quoting', None)
            pd_kwargs.pop('escapechar', None)
            pd_kwargs.pop('header', None)
            pd_kwargs.pop('date_format', None)
            pd_kwargs.pop('compression', None)
        df = df[columns] if columns else df
        columns_types: Dict[str, str] = {}
        partitions_types: Dict[str, str] = {}
        if database and table:
            (columns_types, partitions_types) = _data_types.athena_types_from_pandas_partitioned(df=df, index=index, partition_cols=partition_cols, dtype=dtype, index_left=True)
            if schema_evolution is False:
                _utils.check_schema_changes(columns_types=columns_types, table_input=catalog_table_input, mode=mode)
            create_table_args: Dict[str, Any] = {'database': database, 'table': table, 'path': path, 'columns_types': columns_types, 'table_type': table_type, 'partitions_types': partitions_types, 'bucketing_info': bucketing_info, 'description': description, 'parameters': parameters, 'columns_comments': columns_comments, 'boto3_session': boto3_session, 'mode': mode, 'transaction_id': transaction_id, 'schema_evolution': schema_evolution, 'catalog_versioning': catalog_versioning, 'sep': sep, 'athena_partition_projection_settings': athena_partition_projection_settings, 'catalog_table_input': catalog_table_input, 'catalog_id': catalog_id, 'compression': pandas_kwargs.get('compression'), 'skip_header_line_count': True if header else None, 'serde_library': None, 'serde_parameters': None}
            if catalog_table_input is None and table_type == 'GOVERNED':
                catalog._create_csv_table(**create_table_args)
                catalog_table_input = catalog._get_table_input(database=database, table=table, boto3_session=boto3_session, transaction_id=transaction_id, catalog_id=catalog_id)
                create_table_args['catalog_table_input'] = catalog_table_input
        (paths, partitions_values) = _to_dataset(func=_to_text, concurrent_partitioning=concurrent_partitioning, df=df, path_root=path, index=index, sep=sep, compression=compression, catalog_id=catalog_id, database=database, table=table, table_type=table_type, transaction_id=transaction_id, filename_prefix=filename_prefix, use_threads=use_threads, partition_cols=partition_cols, partitions_types=partitions_types, bucketing_info=bucketing_info, mode=mode, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs, file_format='csv', quoting=quoting, escapechar=escapechar, header=header, date_format=date_format, **pd_kwargs)
        if database and table:
            try:
                serde_info: Dict[str, Any] = {}
                if catalog_table_input:
                    serde_info = catalog_table_input['StorageDescriptor']['SerdeInfo']
                create_table_args['serde_library'] = serde_info.get('SerializationLibrary', None)
                create_table_args['serde_parameters'] = serde_info.get('Parameters', None)
                catalog._create_csv_table(**create_table_args)
                if partitions_values and regular_partitions is True and (table_type != 'GOVERNED'):
                    catalog.add_csv_partitions(database=database, table=table, partitions_values=partitions_values, bucketing_info=bucketing_info, boto3_session=boto3_session, sep=sep, serde_library=create_table_args['serde_library'], serde_parameters=create_table_args['serde_parameters'], catalog_id=catalog_id, columns_types=columns_types, compression=pandas_kwargs.get('compression'))
                if commit_trans:
                    lakeformation.commit_transaction(transaction_id=transaction_id, boto3_session=boto3_session)
            except Exception:
                _logger.debug('Catalog write failed, cleaning up S3 objects (len(paths): %s).', len(paths))
                delete_objects(path=paths, use_threads=use_threads, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs)
                raise
    return {'paths': paths, 'partitions_values': partitions_values}

@apply_configs
@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session'])
def to_json(df: pd.DataFrame, path: Optional[str]=None, index: bool=True, columns: Optional[List[str]]=None, use_threads: Union[bool, int]=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None, sanitize_columns: bool=False, dataset: bool=False, filename_prefix: Optional[str]=None, partition_cols: Optional[List[str]]=None, bucketing_info: Optional[BucketingInfoTuple]=None, concurrent_partitioning: bool=False, mode: Optional[Literal['append', 'overwrite', 'overwrite_partitions']]=None, catalog_versioning: bool=False, schema_evolution: bool=True, dtype: Optional[Dict[str, str]]=None, database: Optional[str]=None, table: Optional[str]=None, glue_table_settings: Optional[GlueTableSettings]=None, athena_partition_projection_settings: Optional[typing.AthenaPartitionProjectionSettings]=None, catalog_id: Optional[str]=None, **pandas_kwargs: Any) -> _S3WriteDataReturnValue:
    if False:
        while True:
            i = 10
    'Write JSON file on Amazon S3.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    df: pandas.DataFrame\n        Pandas DataFrame https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html\n    path : str\n        Amazon S3 path (e.g. s3://bucket/filename.json).\n    index : bool\n        Write row names (index).\n    columns : Optional[List[str]]\n        Columns to write.\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 Session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forwarded to botocore requests.\n        e.g. s3_additional_kwargs={\'ServerSideEncryption\': \'aws:kms\', \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'}\n    sanitize_columns : bool\n        True to sanitize columns names or False to keep it as is.\n        True value is forced if `dataset=True`.\n    dataset : bool\n        If True store as a dataset instead of ordinary file(s)\n        If True, enable all follow arguments:\n        partition_cols, mode, database, table, description, parameters, columns_comments, concurrent_partitioning,\n        catalog_versioning, projection_params, catalog_id, schema_evolution.\n    filename_prefix: str, optional\n        If dataset=True, add a filename prefix to the output files.\n    partition_cols: List[str], optional\n        List of column names that will be used to create partitions. Only takes effect if dataset=True.\n    bucketing_info: Tuple[List[str], int], optional\n        Tuple consisting of the column names used for bucketing as the first element and the number of buckets as the\n        second element.\n        Only `str`, `int` and `bool` are supported as column data types for bucketing.\n    concurrent_partitioning: bool\n        If True will increase the parallelism level during the partitions writing. It will decrease the\n        writing time and increase the memory usage.\n        https://aws-sdk-pandas.readthedocs.io/en/3.4.2/tutorials/022%20-%20Writing%20Partitions%20Concurrently.html\n    mode : str, optional\n        ``append`` (Default), ``overwrite``, ``overwrite_partitions``. Only takes effect if dataset=True.\n        For details check the related tutorial:\n        https://aws-sdk-pandas.readthedocs.io/en/3.4.2/stubs/awswrangler.s3.to_parquet.html#awswrangler.s3.to_parquet\n    catalog_versioning : bool\n        If True and `mode="overwrite"`, creates an archived version of the table catalog before updating it.\n    schema_evolution : bool\n        If True allows schema evolution (new or missing columns), otherwise a exception will be raised.\n        (Only considered if dataset=True and mode in ("append", "overwrite_partitions"))\n        Related tutorial:\n        https://aws-sdk-pandas.readthedocs.io/en/3.4.2/tutorials/014%20-%20Schema%20Evolution.html\n    database : str, optional\n        Glue/Athena catalog: Database name.\n    table : str, optional\n        Glue/Athena catalog: Table name.\n    glue_table_settings: dict (GlueTableSettings), optional\n        Settings for writing to the Glue table.\n    dtype : Dict[str, str], optional\n        Dictionary of columns names and Athena/Glue types to be casted.\n        Useful when you have columns with undetermined or mixed data types.\n        (e.g. {\'col name\': \'bigint\', \'col2 name\': \'int\'})\n    athena_partition_projection_settings: typing.AthenaPartitionProjectionSettings, optional\n        Parameters of the Athena Partition Projection (https://docs.aws.amazon.com/athena/latest/ug/partition-projection.html).\n        AthenaPartitionProjectionSettings is a `TypedDict`, meaning the passed parameter can be instantiated either as an\n        instance of AthenaPartitionProjectionSettings or as a regular Python dict.\n\n        Following projection parameters are supported:\n\n        .. list-table:: Projection Parameters\n           :header-rows: 1\n\n           * - Name\n             - Type\n             - Description\n           * - projection_types\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections types.\n               Valid types: "enum", "integer", "date", "injected"\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'enum\', \'col2_name\': \'integer\'})\n           * - projection_ranges\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections ranges.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'0,10\', \'col2_name\': \'-1,8675309\'})\n           * - projection_values\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections values.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'A,B,Unknown\', \'col2_name\': \'foo,boo,bar\'})\n           * - projection_intervals\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections intervals.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'1\', \'col2_name\': \'5\'})\n           * - projection_digits\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections digits.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_name\': \'1\', \'col2_name\': \'2\'})\n           * - projection_formats\n             - Optional[Dict[str, str]]\n             - Dictionary of partitions names and Athena projections formats.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-supported-types.html\n               (e.g. {\'col_date\': \'yyyy-MM-dd\', \'col2_timestamp\': \'yyyy-MM-dd HH:mm:ss\'})\n           * - projection_storage_location_template\n             - Optional[str]\n             - Value which is allows Athena to properly map partition values if the S3 file locations do not follow\n               a typical `.../column=value/...` pattern.\n               https://docs.aws.amazon.com/athena/latest/ug/partition-projection-setting-up.html\n               (e.g. s3://bucket/table_root/a=${a}/${b}/some_static_subdirectory/${c}/)\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    pandas_kwargs:\n        KEYWORD arguments forwarded to pandas.DataFrame.to_json(). You can NOT pass `pandas_kwargs` explicit, just add\n        valid Pandas arguments in the function call and awswrangler will accept it.\n        e.g. wr.s3.to_json(df, path, lines=True, date_format=\'iso\')\n        https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_json.html\n\n    Returns\n    -------\n    wr.typing._S3WriteDataReturnValue\n        Dictionary with:\n        \'paths\': List of all stored files paths on S3.\n        \'partitions_values\': Dictionary of partitions added with keys as S3 path locations\n        and values as a list of partitions values as str.\n\n    Examples\n    --------\n    Writing JSON file\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_json(\n    ...     df=pd.DataFrame({\'col\': [1, 2, 3]}),\n    ...     path=\'s3://bucket/filename.json\',\n    ... )\n\n    Writing JSON file using pandas_kwargs\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_json(\n    ...     df=pd.DataFrame({\'col\': [1, 2, 3]}),\n    ...     path=\'s3://bucket/filename.json\',\n    ...     lines=True,\n    ...     date_format=\'iso\'\n    ... )\n\n    Writing CSV file encrypted with a KMS key\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> wr.s3.to_json(\n    ...     df=pd.DataFrame({\'col\': [1, 2, 3]}),\n    ...     path=\'s3://bucket/filename.json\',\n    ...     s3_additional_kwargs={\n    ...         \'ServerSideEncryption\': \'aws:kms\',\n    ...         \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'\n    ...     }\n    ... )\n\n    Writing partitioned dataset with partition projection\n\n    >>> import awswrangler as wr\n    >>> import pandas as pd\n    >>> from datetime import datetime\n    >>> dt = lambda x: datetime.strptime(x, "%Y-%m-%d").date()\n    >>> wr.s3.to_json(\n    ...     df=pd.DataFrame({\n    ...         "id": [1, 2, 3],\n    ...         "value": [1000, 1001, 1002],\n    ...         "category": [\'A\', \'B\', \'C\'],\n    ...     }),\n    ...     path=\'s3://bucket/prefix\',\n    ...     dataset=True,\n    ...     partition_cols=[\'value\', \'category\'],\n    ...     athena_partition_projection_settings={\n    ...        "projection_types": {\n    ...             "value": "integer",\n    ...             "category": "enum",\n    ...         },\n    ...         "projection_ranges": {\n    ...             "value": "1000,2000",\n    ...             "category": "A,B,C",\n    ...         },\n    ...     },\n    ... )\n    {\n        \'paths\': [\n            \'s3://.../value=1000/category=A/x.json\', ...\n        ],\n        \'partitions_values\': {\n            \'s3://.../value=1000/category=A/\': [\n                \'1000\',\n                \'A\',\n            ], ...\n        }\n    }\n\n    '
    if 'pandas_kwargs' in pandas_kwargs:
        raise exceptions.InvalidArgument("You can NOT pass `pandas_kwargs` explicit, just add valid Pandas arguments in the function call and awswrangler will accept it.e.g. wr.s3.to_json(df, path, lines=True, date_format='iso')")
    glue_table_settings = cast(GlueTableSettings, glue_table_settings if glue_table_settings else {})
    table_type = glue_table_settings.get('table_type')
    transaction_id = glue_table_settings.get('transaction_id')
    description = glue_table_settings.get('description')
    parameters = glue_table_settings.get('parameters')
    columns_comments = glue_table_settings.get('columns_comments')
    regular_partitions = glue_table_settings.get('regular_partitions', True)
    _validate_args(df=df, table=table, database=database, dataset=dataset, path=path, partition_cols=partition_cols, bucketing_info=bucketing_info, mode=mode, description=description, parameters=parameters, columns_comments=columns_comments, execution_engine=engine.get())
    partition_cols = partition_cols if partition_cols else []
    dtype = dtype if dtype else {}
    partitions_values: Dict[str, List[str]] = {}
    mode = 'append' if mode is None else mode
    commit_trans: bool = False
    if transaction_id:
        table_type = 'GOVERNED'
    filename_prefix = filename_prefix + uuid.uuid4().hex if filename_prefix else uuid.uuid4().hex
    s3_client = _utils.client(service_name='s3', session=boto3_session)
    if sanitize_columns is True or (database is not None and table is not None):
        (df, dtype, partition_cols, bucketing_info) = _sanitize(df=copy_df_shallow(df), dtype=dtype, partition_cols=partition_cols, bucketing_info=bucketing_info)
    catalog_table_input: Optional[Dict[str, Any]] = None
    if database and table:
        catalog_table_input = catalog._get_table_input(database=database, table=table, boto3_session=boto3_session, transaction_id=transaction_id, catalog_id=catalog_id)
        catalog_path: Optional[str] = None
        if catalog_table_input:
            table_type = catalog_table_input['TableType']
            catalog_path = catalog_table_input.get('StorageDescriptor', {}).get('Location')
        if path is None:
            if catalog_path:
                path = catalog_path
            else:
                raise exceptions.InvalidArgumentValue('Glue table does not exist in the catalog. Please pass the `path` argument to create it.')
        elif path and catalog_path:
            if path.rstrip('/') != catalog_path.rstrip('/'):
                raise exceptions.InvalidArgumentValue(f'The specified path: {path}, does not match the existing Glue catalog table path: {catalog_path}')
        if pandas_kwargs.get('compression') not in ('gzip', 'bz2', None):
            raise exceptions.InvalidArgumentCombination('If database and table are given, you must use one of these compressions: gzip, bz2 or None.')
        if table_type == 'GOVERNED' and (not transaction_id):
            _logger.debug('`transaction_id` not specified for GOVERNED table, starting transaction')
            transaction_id = lakeformation.start_transaction(read_only=False, boto3_session=boto3_session)
            commit_trans = True
    df = _apply_dtype(df=df, dtype=dtype, catalog_table_input=catalog_table_input, mode=mode)
    if dataset is False:
        output_paths = _to_text(df, file_format='json', path=path, use_threads=use_threads, s3_client=s3_client, s3_additional_kwargs=s3_additional_kwargs, **pandas_kwargs)
        return {'paths': output_paths, 'partitions_values': {}}
    compression: Optional[str] = pandas_kwargs.pop('compression', None)
    df = df[columns] if columns else df
    columns_types: Dict[str, str] = {}
    partitions_types: Dict[str, str] = {}
    if database and table:
        (columns_types, partitions_types) = _data_types.athena_types_from_pandas_partitioned(df=df, index=index, partition_cols=partition_cols, dtype=dtype)
        if schema_evolution is False:
            _utils.check_schema_changes(columns_types=columns_types, table_input=catalog_table_input, mode=mode)
        create_table_args: Dict[str, Any] = {'database': database, 'table': table, 'path': path, 'columns_types': columns_types, 'table_type': table_type, 'partitions_types': partitions_types, 'bucketing_info': bucketing_info, 'description': description, 'parameters': parameters, 'columns_comments': columns_comments, 'boto3_session': boto3_session, 'mode': mode, 'transaction_id': transaction_id, 'catalog_versioning': catalog_versioning, 'schema_evolution': schema_evolution, 'athena_partition_projection_settings': athena_partition_projection_settings, 'catalog_table_input': catalog_table_input, 'catalog_id': catalog_id, 'compression': compression, 'serde_library': None, 'serde_parameters': None}
        if catalog_table_input is None and table_type == 'GOVERNED':
            catalog._create_json_table(**create_table_args)
            catalog_table_input = catalog._get_table_input(database=database, table=table, boto3_session=boto3_session, transaction_id=transaction_id, catalog_id=catalog_id)
            create_table_args['catalog_table_input'] = catalog_table_input
    (paths, partitions_values) = _to_dataset(func=_to_text, concurrent_partitioning=concurrent_partitioning, df=df, path_root=path, filename_prefix=filename_prefix, index=index, compression=compression, catalog_id=catalog_id, database=database, table=table, table_type=table_type, transaction_id=transaction_id, use_threads=use_threads, partition_cols=partition_cols, partitions_types=partitions_types, bucketing_info=bucketing_info, mode=mode, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs, file_format='json', **pandas_kwargs)
    if database and table:
        try:
            serde_info: Dict[str, Any] = {}
            if catalog_table_input:
                serde_info = catalog_table_input['StorageDescriptor']['SerdeInfo']
            create_table_args['serde_library'] = serde_info.get('SerializationLibrary', None)
            create_table_args['serde_parameters'] = serde_info.get('Parameters', None)
            catalog._create_json_table(**create_table_args)
            if partitions_values and regular_partitions is True and (table_type != 'GOVERNED'):
                catalog.add_json_partitions(database=database, table=table, partitions_values=partitions_values, bucketing_info=bucketing_info, boto3_session=boto3_session, serde_library=create_table_args['serde_library'], serde_parameters=create_table_args['serde_parameters'], catalog_id=catalog_id, columns_types=columns_types, compression=compression)
                if commit_trans:
                    lakeformation.commit_transaction(transaction_id=transaction_id, boto3_session=boto3_session)
        except Exception:
            _logger.debug('Catalog write failed, cleaning up S3 objects (len(paths): %s).', len(paths))
            delete_objects(path=paths, use_threads=use_threads, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs)
            raise
    return {'paths': paths, 'partitions_values': partitions_values}