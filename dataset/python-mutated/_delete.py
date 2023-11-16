"""AWS Glue Catalog Delete Module."""
import logging
from typing import Any, Dict, List, Optional
import boto3
from awswrangler import _utils, exceptions
from awswrangler._config import apply_configs
from awswrangler.catalog._definitions import _update_table_definition
from awswrangler.catalog._get import _get_partitions
from awswrangler.catalog._utils import _catalog_id, _transaction_id
_logger: logging.Logger = logging.getLogger(__name__)

@apply_configs
def delete_database(name: str, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> None:
    if False:
        return 10
    "Delete a database in AWS Glue Catalog.\n\n    Parameters\n    ----------\n    name : str\n        Database name.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    None\n        None.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> wr.catalog.delete_database(\n    ...     name='awswrangler_test'\n    ... )\n    "
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    client_glue.delete_database(**_catalog_id(Name=name, catalog_id=catalog_id))

@apply_configs
def delete_table_if_exists(database: str, table: str, transaction_id: Optional[str]=None, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> bool:
    if False:
        for i in range(10):
            print('nop')
    "Delete Glue table if exists.\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    bool\n        True if deleted, otherwise False.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> wr.catalog.delete_table_if_exists(database='default', table='my_table')  # deleted\n    True\n    >>> wr.catalog.delete_table_if_exists(database='default', table='my_table')  # Nothing to be deleted\n    False\n\n    "
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    try:
        client_glue.delete_table(**_catalog_id(**_transaction_id(transaction_id=transaction_id, DatabaseName=database, Name=table, catalog_id=catalog_id)))
        _logger.debug('Deleted catalog table: %s', table)
        return True
    except client_glue.exceptions.EntityNotFoundException:
        return False

@apply_configs
def delete_partitions(table: str, database: str, partitions_values: List[List[str]], catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> None:
    if False:
        print('Hello World!')
    "Delete specified partitions in a AWS Glue Catalog table.\n\n    Parameters\n    ----------\n    table : str\n        Table name.\n    database : str\n        Table name.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    partitions_values : List[List[str]]\n        List of lists of partitions values as strings.\n        (e.g. [['2020', '10', '25'], ['2020', '11', '16'], ['2020', '12', '19']]).\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    None\n        None.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> wr.catalog.delete_partitions(\n    ...     table='my_table',\n    ...     database='awswrangler_test',\n    ...     partitions_values=[['2020', '10', '25'], ['2020', '11', '16'], ['2020', '12', '19']]\n    ... )\n    "
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    chunks: List[List[List[str]]] = _utils.chunkify(lst=partitions_values, max_length=25)
    for chunk in chunks:
        client_glue.batch_delete_partition(**_catalog_id(catalog_id=catalog_id, DatabaseName=database, TableName=table, PartitionsToDelete=[{'Values': v} for v in chunk]))

@apply_configs
def delete_all_partitions(table: str, database: str, catalog_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None) -> List[List[str]]:
    if False:
        while True:
            i = 10
    "Delete all partitions in a AWS Glue Catalog table.\n\n    Parameters\n    ----------\n    table : str\n        Table name.\n    database : str\n        Table name.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    List[List[str]]\n        Partitions values.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> partitions = wr.catalog.delete_all_partitions(\n    ...     table='my_table',\n    ...     database='awswrangler_test',\n    ... )\n    "
    _logger.debug('Fetching existing partitions...')
    partitions_values: List[List[str]] = list(_get_partitions(database=database, table=table, boto3_session=boto3_session, catalog_id=catalog_id).values())
    _logger.debug('Number of old partitions: %s', len(partitions_values))
    _logger.debug('Deleting existing partitions...')
    delete_partitions(table=table, database=database, catalog_id=catalog_id, partitions_values=partitions_values, boto3_session=boto3_session)
    return partitions_values

@apply_configs
def delete_column(database: str, table: str, column_name: str, transaction_id: Optional[str]=None, boto3_session: Optional[boto3.Session]=None, catalog_id: Optional[str]=None) -> None:
    if False:
        i = 10
        return i + 15
    "Delete a column in a AWS Glue Catalog table.\n\n    Parameters\n    ----------\n    database : str\n        Database name.\n    table : str\n        Table name.\n    column_name : str\n        Column name\n    transaction_id: str, optional\n        The ID of the transaction (i.e. used with GOVERNED tables).\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    catalog_id : str, optional\n        The ID of the Data Catalog from which to retrieve Databases.\n        If none is provided, the AWS account ID is used by default.\n\n    Returns\n    -------\n    None\n        None\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> wr.catalog.delete_column(\n    ...     database='my_db',\n    ...     table='my_table',\n    ...     column_name='my_col',\n    ... )\n    "
    client_glue = _utils.client(service_name='glue', session=boto3_session)
    table_res = client_glue.get_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, DatabaseName=database, Name=table)))
    table_input: Dict[str, Any] = _update_table_definition(table_res)
    table_input['StorageDescriptor']['Columns'] = [i for i in table_input['StorageDescriptor']['Columns'] if i['Name'] != column_name]
    res: Dict[str, Any] = client_glue.update_table(**_catalog_id(catalog_id=catalog_id, **_transaction_id(transaction_id=transaction_id, DatabaseName=database, TableInput=table_input)))
    if 'Errors' in res and res['Errors']:
        for error in res['Errors']:
            if 'ErrorDetail' in error:
                if 'ErrorCode' in error['ErrorDetail']:
                    raise exceptions.ServiceApiError(str(res['Errors']))