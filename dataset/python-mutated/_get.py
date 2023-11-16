"""AWS Glue Catalog Get Module."""
import base64
import itertools
import logging
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union, cast
import boto3
import botocore.exceptions
import awswrangler.pandas as pd
from awswrangler import _utils, exceptions
from awswrangler._config import apply_configs
from awswrangler.catalog._utils import _catalog_id, _extract_dtypes_from_table_details, _transaction_id
if TYPE_CHECKING:
    from mypy_boto3_glue.type_defs import GetPartitionsResponseTypeDef
_logger: logging.Logger = logging.getLogger(__name__)

def _get_table_input(database: str, table: str, boto3_session: Optional[boto3.Session], transaction_id: Optional[str]=None, catalog_id: Optional[str]=None) -> Optional[Dict[str, Any]]:
    if False:
        while True:
            i = 10
    client_glue = _utils.client('glue', session=boto3_session)
    args: Dict[str, Any] = _catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, DatabaseName=database, Name=table))
    try:
        response = client_glue.get_table(**args)
    except client_glue.exceptions.EntityNotFoundException:
        return None
    table_input: Dict[str, Any] = {}
    for (k, v) in response['Table'].items():
        if k in ['Name', 'Description', 'Owner', 'LastAccessTime', 'LastAnalyzedTime', 'Retention', 'StorageDescriptor', 'PartitionKeys', 'ViewOriginalText', 'ViewExpandedText', 'TableType', 'Parameters', 'TargetTable']:
            table_input[k] = v
    return table_input

def _append_partitions(partitions_values: Dict[str, List[str]], response: 'GetPartitionsResponseTypeDef') -> Optional[str]:
    if False:
        while True:
            i = 10
    _logger.debug('response: %s', response)
    token: Optional[str] = response.get('NextToken', None)
    if response is not None and 'Partitions' in response:
        for partition in response['Partitions']:
            location: Optional[str] = partition['StorageDescriptor'].get('Location')
            if location is not None:
                values: List[str] = partition['Values']
                partitions_values[location] = values
    else:
        token = None
    return token

def _get_partitions(database: str, table: str, expression: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, List[str]]:
    if False:
        i = 10
        return i + 15
    client_glue = _utils.client('glue', session=boto3_session)
    args: Dict[str, Any] = _catalog_id(catalog_id=catalog_id, DatabaseName=database, TableName=table, MaxResults=1000, Segment={'SegmentNumber': 0, 'TotalSegments': 1}, ExcludeColumnSchema=True)
    if expression is not None:
        args['Expression'] = expression
    partitions_values: Dict[str, List[str]] = {}
    _logger.debug('Starting pagination...')
    response = client_glue.get_partitions(**args)
    token: Optional[str] = _append_partitions(partitions_values=partitions_values, response=response)
    while token is not None:
        args['NextToken'] = response['NextToken']
        response = client_glue.get_partitions(**args)
        token = _append_partitions(partitions_values=partitions_values, response=response)
    _logger.debug('Pagination done.')
    return partitions_values

@apply_configs
def get_table_types(database: str, table: str, transaction_id: Optional[str]=None, query_as_of_time: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Optional[Dict[str, str]]:
    if False:
        return 10
    "Get all columns and types from a table.\n\n    Note\n    ----\n    If reading from a governed table, pass only one of `transaction_id` or `query_as_of_time`.\n\n    Parameters\n    ----------\n    database: str\n        Database name.\n    table: str\n        Table name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    query_as_of_time: str, optional\n        The time as of when to read the table contents. Must be a valid Unix epoch timestamp.\n        Cannot be specified alongside transaction_id.\n    catalog_id: str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session: boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Optional[Dict[str, str]]\n        If table exists, a dictionary like {'col name': 'col data type'}. Otherwise None.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_table_types(database='default', table='my_table')\n    {'col0': 'int', 'col1': double}\n\n    "
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    try:
        response = client_glue.get_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, query_as_of_time=query_as_of_time, DatabaseName=database, Name=table)))
    except client_glue.exceptions.EntityNotFoundException:
        return None
    return _extract_dtypes_from_table_details(response=response)

def get_databases(catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Iterator[Dict[str, Any]]:
    if False:
        print('Hello World!')
    'Get an iterator of databases.\n\n    Parameters\n    ----------\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Iterator[Dict[str, Any]]\n        Iterator of Databases.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> dbs = wr.catalog.get_databases()\n\n    '
    client_glue = _utils.client('glue', session=boto3_session)
    paginator = client_glue.get_paginator('get_databases')
    response_iterator = paginator.paginate(**_catalog_id(catalog_id=catalog_id))
    for page in response_iterator:
        for db in page['DatabaseList']:
            yield cast(Dict[str, Any], db)

@apply_configs
def databases(limit: int=100, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> pd.DataFrame:
    if False:
        return 10
    'Get a Pandas DataFrame with all listed databases.\n\n    Parameters\n    ----------\n    limit : int, optional\n        Max number of tables to be returned.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    pandas.DataFrame\n        Pandas DataFrame filled by formatted table information.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> df_dbs = wr.catalog.databases()\n\n    '
    database_iter = get_databases(catalog_id=catalog_id, boto3_session=boto3_session)
    dbs = itertools.islice(database_iter, limit)
    df_dict: Dict[str, List[str]] = {'Database': [], 'Description': []}
    for db in dbs:
        df_dict['Database'].append(db['Name'])
        df_dict['Description'].append(db.get('Description', ''))
    return pd.DataFrame(data=df_dict)

@apply_configs
def get_tables(catalog_id: Optional[str]=None, database: Optional[str]=None, transaction_id: Optional[str]=None, name_contains: Optional[str]=None, name_prefix: Optional[str]=None, name_suffix: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Iterator[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    'Get an iterator of tables.\n\n    Note\n    ----\n    Please, do not filter using name_contains and name_prefix/name_suffix at the same time.\n    Only name_prefix and name_suffix can be combined together.\n\n    Parameters\n    ----------\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    database : str, optional\n        Database name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    name_contains : str, optional\n        Select by a specific string on table name\n    name_prefix : str, optional\n        Select by a specific prefix on table name\n    name_suffix : str, optional\n        Select by a specific suffix on table name\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Iterator[Dict[str, Any]]\n        Iterator of tables.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> tables = wr.catalog.get_tables()\n\n    '
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    paginator = client_glue.get_paginator('get_tables')
    args: Dict[str, str] = {}
    if name_prefix is not None and name_suffix is not None and (name_contains is not None):
        raise exceptions.InvalidArgumentCombination('Please, do not filter using name_contains and name_prefix/name_suffix at the same time. Only name_prefix and name_suffix can be combined together.')
    if name_prefix is not None and name_suffix is not None:
        args['Expression'] = f'{name_prefix}*{name_suffix}'
    elif name_contains is not None:
        args['Expression'] = f'*{name_contains}*'
    elif name_prefix is not None:
        args['Expression'] = f'{name_prefix}*'
    elif name_suffix is not None:
        args['Expression'] = f'*{name_suffix}'
    if database is not None:
        dbs: List[str] = [database]
    else:
        dbs = [x['Name'] for x in get_databases(catalog_id=catalog_id)]
    for db in dbs:
        args['DatabaseName'] = db
        response_iterator = paginator.paginate(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, **args)))
        try:
            for page in response_iterator:
                for tbl in page['TableList']:
                    yield cast(Dict[str, Any], tbl)
        except client_glue.exceptions.EntityNotFoundException:
            continue

@apply_configs
def tables(limit: int=100, catalog_id: Optional[str]=None, database: Optional[str]=None, transaction_id: Optional[str]=None, search_text: Optional[str]=None, name_contains: Optional[str]=None, name_prefix: Optional[str]=None, name_suffix: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    "Get a DataFrame with tables filtered by a search term, prefix, suffix.\n\n    Note\n    ----\n    Search feature is not supported for Governed tables.\n\n    Parameters\n    ----------\n    limit : int, optional\n        Max number of tables to be returned.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    database : str, optional\n        Database name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    search_text : str, optional\n        Select only tables with the given string in table's properties.\n    name_contains : str, optional\n        Select by a specific string on table name\n    name_prefix : str, optional\n        Select by a specific prefix on table name\n    name_suffix : str, optional\n        Select by a specific suffix on table name\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    pandas.DataFrame\n        Pandas DataFrame filled by formatted table information.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> df_tables = wr.catalog.tables()\n\n    "
    if search_text is None:
        table_iter = get_tables(catalog_id=catalog_id, database=database, transaction_id=transaction_id, name_contains=name_contains, name_prefix=name_prefix, name_suffix=name_suffix, boto3_session=boto3_session)
        tbls: List[Dict[str, Any]] = list(itertools.islice(table_iter, limit))
    else:
        tbls = list(search_tables(text=search_text, catalog_id=catalog_id, boto3_session=boto3_session))
        if database is not None:
            tbls = [x for x in tbls if x['DatabaseName'] == database]
        if name_contains is not None:
            tbls = [x for x in tbls if name_contains in x['Name']]
        if name_prefix is not None:
            tbls = [x for x in tbls if x['Name'].startswith(name_prefix)]
        if name_suffix is not None:
            tbls = [x for x in tbls if x['Name'].endswith(name_suffix)]
        tbls = tbls[:limit]
    df_dict: Dict[str, List[str]] = {'Database': [], 'Table': [], 'Description': [], 'TableType': [], 'Columns': [], 'Partitions': []}
    for tbl in tbls:
        df_dict['Database'].append(tbl['DatabaseName'])
        df_dict['Table'].append(tbl['Name'])
        df_dict['Description'].append(tbl.get('Description', ''))
        df_dict['TableType'].append(tbl.get('TableType', ''))
        try:
            columns = tbl['StorageDescriptor']['Columns']
            df_dict['Columns'].append(', '.join([x['Name'] for x in columns]))
        except KeyError:
            df_dict['Columns'].append('')
        if 'PartitionKeys' in tbl:
            df_dict['Partitions'].append(', '.join([x['Name'] for x in tbl['PartitionKeys']]))
        else:
            df_dict['Partitions'].append('')
    return pd.DataFrame(data=df_dict)

def search_tables(text: str, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Iterator[Dict[str, Any]]:
    if False:
        return 10
    "Get Pandas DataFrame of tables filtered by a search string.\n\n    Note\n    ----\n    Search feature is not supported for Governed tables.\n\n    Parameters\n    ----------\n    text : str, optional\n        Select only tables with the given string in table's properties.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Iterator[Dict[str, Any]]\n        Iterator of tables.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> df_tables = wr.catalog.search_tables(text='my_property')\n\n    "
    client_glue = _utils.client('glue', session=boto3_session)
    args: Dict[str, Any] = _catalog_id(catalog_id=catalog_id, SearchText=text)
    response = client_glue.search_tables(**args)
    for tbl in response['TableList']:
        yield cast(Dict[str, Any], tbl)
    while 'NextToken' in response:
        args['NextToken'] = response['NextToken']
        response = client_glue.search_tables(**args)
        for tbl in response['TableList']:
            yield cast(Dict[str, Any], tbl)

@apply_configs
def table(database: str, table: str, transaction_id: Optional[str]=None, query_as_of_time: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    "Get table details as Pandas DataFrame.\n\n    Note\n    ----\n    If reading from a governed table, pass only one of `transaction_id` or `query_as_of_time`.\n\n    Parameters\n    ----------\n    database: str\n        Database name.\n    table: str\n        Table name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    query_as_of_time: str, optional\n        The time as of when to read the table contents. Must be a valid Unix epoch timestamp.\n        Cannot be specified alongside transaction_id.\n    catalog_id: str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session: boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    pandas.DataFrame\n        Pandas DataFrame filled by formatted table information.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> df_table = wr.catalog.table(database='default', table='my_table')\n\n    "
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    tbl = client_glue.get_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, query_as_of_time=query_as_of_time, DatabaseName=database, Name=table)))['Table']
    df_dict: Dict[str, List[Union[str, bool]]] = {'Column Name': [], 'Type': [], 'Partition': [], 'Comment': []}
    if 'StorageDescriptor' in tbl:
        for col in tbl['StorageDescriptor'].get('Columns', {}):
            df_dict['Column Name'].append(col['Name'])
            df_dict['Type'].append(col['Type'])
            df_dict['Partition'].append(False)
            if 'Comment' in col:
                df_dict['Comment'].append(col['Comment'])
            else:
                df_dict['Comment'].append('')
    if 'PartitionKeys' in tbl:
        for col in tbl['PartitionKeys']:
            df_dict['Column Name'].append(col['Name'])
            df_dict['Type'].append(col['Type'])
            df_dict['Partition'].append(True)
            if 'Comment' in col:
                df_dict['Comment'].append(col['Comment'])
            else:
                df_dict['Comment'].append('')
    return pd.DataFrame(data=df_dict)

@apply_configs
def get_table_location(database: str, table: str, transaction_id: Optional[str]=None, query_as_of_time: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        for i in range(10):
            print('nop')
    "Get table's location on Glue catalog.\n\n    Note\n    ----\n    If reading from a governed table, pass only one of `transaction_id` or `query_as_of_time`.\n\n    Parameters\n    ----------\n    database: str\n        Database name.\n    table: str\n        Table name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    query_as_of_time: str, optional\n        The time as of when to read the table contents. Must be a valid Unix epoch timestamp.\n        Cannot be specified alongside transaction_id.\n    catalog_id: str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session: boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Table's location.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_table_location(database='default', table='my_table')\n    's3://bucket/prefix/'\n\n    "
    client_glue = _utils.client('glue', session=boto3_session)
    res = client_glue.get_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, query_as_of_time=query_as_of_time, DatabaseName=database, Name=table)))
    try:
        return res['Table']['StorageDescriptor']['Location']
    except KeyError as ex:
        raise exceptions.InvalidTable(f'{database}.{table}') from ex

def get_connection(name: str, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    "Get Glue connection details.\n\n    Parameters\n    ----------\n    name : str\n        Connection name.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, Any]\n        API Response for:\n        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/glue.html#Glue.Client.get_connection\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> res = wr.catalog.get_connection(name='my_connection')\n\n    "
    client_glue = _utils.client('glue', session=boto3_session)
    res = _utils.try_it(f=client_glue.get_connection, ex=botocore.exceptions.ClientError, ex_code='ThrottlingException', max_num_tries=3, **_catalog_id(catalog_id=catalog_id, Name=name, HidePassword=False))['Connection']
    if 'ENCRYPTED_PASSWORD' in res['ConnectionProperties']:
        client_kms = _utils.client(service_name='kms', session=boto3_session)
        pwd = client_kms.decrypt(CiphertextBlob=base64.b64decode(res['ConnectionProperties']['ENCRYPTED_PASSWORD']))['Plaintext'].decode('utf-8')
        res['ConnectionProperties']['PASSWORD'] = pwd
    return cast(Dict[str, Any], res)

@apply_configs
def get_parquet_partitions(database: str, table: str, expression: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, List[str]]:
    if False:
        i = 10
        return i + 15
    "Get all partitions from a Table in the AWS Glue Catalog.\n\n    Expression argument instructions:\n    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/glue.html#Glue.Client.get_partitions\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    expression : str, optional\n        An expression that filters the partitions to be returned.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, List[str]]\n        partitions_values: Dictionary with keys as S3 path locations and values as a\n        list of partitions values as str (e.g. {'s3://bucket/prefix/y=2020/m=10/': ['2020', '10']}).\n\n    Examples\n    --------\n    Fetch all partitions\n\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_parquet_partitions(\n    ...     database='default',\n    ...     table='my_table',\n    ... )\n    {\n        's3://bucket/prefix/y=2020/m=10/': ['2020', '10'],\n        's3://bucket/prefix/y=2020/m=11/': ['2020', '11'],\n        's3://bucket/prefix/y=2020/m=12/': ['2020', '12']\n    }\n\n    Filtering partitions\n\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_parquet_partitions(\n    ...     database='default',\n    ...     table='my_table',\n    ...     expression='m=10'\n    ... )\n    {\n        's3://bucket/prefix/y=2020/m=10/': ['2020', '10']\n    }\n\n    "
    return _get_partitions(database=database, table=table, expression=expression, catalog_id=catalog_id, boto3_session=boto3_session)

@apply_configs
def get_csv_partitions(database: str, table: str, expression: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, List[str]]:
    if False:
        print('Hello World!')
    "Get all partitions from a Table in the AWS Glue Catalog.\n\n    Expression argument instructions:\n    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/glue.html#Glue.Client.get_partitions\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    expression : str, optional\n        An expression that filters the partitions to be returned.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, List[str]]\n        partitions_values: Dictionary with keys as S3 path locations and values as a\n        list of partitions values as str (e.g. {'s3://bucket/prefix/y=2020/m=10/': ['2020', '10']}).\n\n    Examples\n    --------\n    Fetch all partitions\n\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_csv_partitions(\n    ...     database='default',\n    ...     table='my_table',\n    ... )\n    {\n        's3://bucket/prefix/y=2020/m=10/': ['2020', '10'],\n        's3://bucket/prefix/y=2020/m=11/': ['2020', '11'],\n        's3://bucket/prefix/y=2020/m=12/': ['2020', '12']\n    }\n\n    Filtering partitions\n\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_csv_partitions(\n    ...     database='default',\n    ...     table='my_table',\n    ...     expression='m=10'\n    ... )\n    {\n        's3://bucket/prefix/y=2020/m=10/': ['2020', '10']\n    }\n\n    "
    return _get_partitions(database=database, table=table, expression=expression, catalog_id=catalog_id, boto3_session=boto3_session)

@apply_configs
def get_partitions(database: str, table: str, expression: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, List[str]]:
    if False:
        return 10
    "Get all partitions from a Table in the AWS Glue Catalog.\n\n    Expression argument instructions:\n    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/glue.html#Glue.Client.get_partitions\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    expression : str, optional\n        An expression that filters the partitions to be returned.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, List[str]]\n        partitions_values: Dictionary with keys as S3 path locations and values as a\n        list of partitions values as str (e.g. {'s3://bucket/prefix/y=2020/m=10/': ['2020', '10']}).\n\n    Examples\n    --------\n    Fetch all partitions\n\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_partitions(\n    ...     database='default',\n    ...     table='my_table',\n    ... )\n    {\n        's3://bucket/prefix/y=2020/m=10/': ['2020', '10'],\n        's3://bucket/prefix/y=2020/m=11/': ['2020', '11'],\n        's3://bucket/prefix/y=2020/m=12/': ['2020', '12']\n    }\n\n    Filtering partitions\n\n    >>> import awswrangler as wr\n    >>> wr.catalog.get_partitions(\n    ...     database='default',\n    ...     table='my_table',\n    ...     expression='m=10'\n    ... )\n    {\n        's3://bucket/prefix/y=2020/m=10/': ['2020', '10']\n    }\n\n    "
    return _get_partitions(database=database, table=table, expression=expression, catalog_id=catalog_id, boto3_session=boto3_session)

def get_table_parameters(database: str, table: str, transaction_id: Optional[str]=None, query_as_of_time: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, str]:
    if False:
        print('Hello World!')
    'Get all parameters.\n\n    Note\n    ----\n    If reading from a governed table, pass only one of `transaction_id` or `query_as_of_time`.\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    query_as_of_time : str, optional\n        The time as of when to read the table contents. Must be a valid Unix epoch timestamp.\n        Cannot be specified alongside transaction_id.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, str]\n        Dictionary of parameters.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> pars = wr.catalog.get_table_parameters(database="...", table="...")\n\n    '
    client_glue = _utils.client('glue', session=boto3_session)
    response = client_glue.get_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, query_as_of_time=query_as_of_time, DatabaseName=database, Name=table)))
    parameters: Dict[str, str] = response['Table']['Parameters']
    return parameters

def get_table_description(database: str, table: str, transaction_id: Optional[str]=None, query_as_of_time: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Optional[str]:
    if False:
        print('Hello World!')
    'Get table description.\n\n    Note\n    ----\n    If reading from a governed table, pass only one of `transaction_id` or `query_as_of_time`.\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    query_as_of_time: str, optional\n        The time as of when to read the table contents. Must be a valid Unix epoch timestamp.\n        Cannot be specified alongside transaction_id.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Optional[str]\n        Description if exists.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> desc = wr.catalog.get_table_description(database="...", table="...")\n\n    '
    client_glue = _utils.client('glue', session=boto3_session)
    response = client_glue.get_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, query_as_of_time=query_as_of_time, DatabaseName=database, Name=table)))
    desc: Optional[str] = response['Table'].get('Description', None)
    return desc

@apply_configs
def get_columns_comments(database: str, table: str, transaction_id: Optional[str]=None, query_as_of_time: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, Optional[str]]:
    if False:
        for i in range(10):
            print('nop')
    'Get all columns comments.\n\n    Note\n    ----\n    If reading from a governed table, pass only one of `transaction_id` or `query_as_of_time`.\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    query_as_of_time: str, optional\n        The time as of when to read the table contents. Must be a valid Unix epoch timestamp.\n        Cannot be specified alongside transaction_id.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, Optional[str]]\n        Columns comments. e.g. {"col1": "foo boo bar", "col2": None}.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> pars = wr.catalog.get_columns_comments(database="...", table="...")\n\n    '
    client_glue = _utils.client('glue', session=boto3_session)
    response = client_glue.get_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, query_as_of_time=query_as_of_time, DatabaseName=database, Name=table)))
    comments: Dict[str, Optional[str]] = {}
    for c in response['Table']['StorageDescriptor']['Columns']:
        comments[c['Name']] = c.get('Comment')
    if 'PartitionKeys' in response['Table']:
        for p in response['Table']['PartitionKeys']:
            comments[p['Name']] = p.get('Comment')
    return comments

@apply_configs
def get_table_versions(database: str, table: str, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[Dict[str, Any]]:
    if False:
        i = 10
        return i + 15
    'Get all versions.\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[Dict[str, Any]\n        List of table inputs:\n        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/glue.html#Glue.Client.get_table_versions\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> tables_versions = wr.catalog.get_table_versions(database="...", table="...")\n\n    '
    client_glue = _utils.client('glue', session=boto3_session)
    paginator = client_glue.get_paginator('get_table_versions')
    versions: List[Dict[str, Any]] = []
    response_iterator = paginator.paginate(**_catalog_id(DatabaseName=database, TableName=table, catalog_id=catalog_id))
    for page in response_iterator:
        for tbl in page['TableVersions']:
            versions.append(cast(Dict[str, Any], tbl))
    return versions

@apply_configs
def get_table_number_of_versions(database: str, table: str, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> int:
    if False:
        for i in range(10):
            print('nop')
    'Get total number of versions.\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    int\n        Total number of versions.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> num = wr.catalog.get_table_number_of_versions(database="...", table="...")\n\n    '
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    paginator = client_glue.get_paginator('get_table_versions')
    count: int = 0
    response_iterator = paginator.paginate(**_catalog_id(DatabaseName=database, TableName=table, catalog_id=catalog_id))
    for page in response_iterator:
        count += len(page['TableVersions'])
    return count