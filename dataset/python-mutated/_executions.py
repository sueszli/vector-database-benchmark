"""Query executions Module for Amazon Athena."""
import logging
import time
from typing import Any, Dict, List, Optional, Union, cast
import boto3
import botocore
from typing_extensions import Literal
from awswrangler import _utils, exceptions, typing
from awswrangler._config import apply_configs
from ._cache import _CacheInfo, _check_for_cached_results
from ._utils import _QUERY_FINAL_STATES, _QUERY_WAIT_POLLING_DELAY, _apply_formatter, _get_workgroup_config, _start_query_execution, _WorkGroupConfig
_logger: logging.Logger = logging.getLogger(__name__)

@apply_configs
def start_query_execution(sql: str, database: Optional[str]=None, s3_output: Optional[str]=None, workgroup: Optional[str]=None, encryption: Optional[str]=None, kms_key: Optional[str]=None, params: Union[Dict[str, Any], List[str], None]=None, paramstyle: Literal['qmark', 'named']='named', boto3_session: Optional[boto3.Session]=None, client_request_token: Optional[str]=None, athena_cache_settings: Optional[typing.AthenaCacheSettings]=None, athena_query_wait_polling_delay: float=_QUERY_WAIT_POLLING_DELAY, data_source: Optional[str]=None, wait: bool=False) -> Union[str, Dict[str, Any]]:
    if False:
        while True:
            i = 10
    'Start a SQL Query against AWS Athena.\n\n    Note\n    ----\n    Create the default Athena bucket if it doesn\'t exist and s3_output is None.\n    (E.g. s3://aws-athena-query-results-ACCOUNT-REGION/)\n\n    Parameters\n    ----------\n    sql : str\n        SQL query.\n    database : str, optional\n        AWS Glue/Athena database name.\n    s3_output : str, optional\n        AWS S3 path.\n    workgroup : str, optional\n        Athena workgroup.\n    encryption : str, optional\n        None, \'SSE_S3\', \'SSE_KMS\', \'CSE_KMS\'.\n    kms_key : str, optional\n        For SSE-KMS and CSE-KMS , this is the KMS key ARN or ID.\n    params: Dict[str, any] | List[str], optional\n        Parameters that will be used for constructing the SQL query.\n        Only named or question mark parameters are supported.\n        The parameter style needs to be specified in the ``paramstyle`` parameter.\n\n        For ``paramstyle="named"``, this value needs to be a dictionary.\n        The dict needs to contain the information in the form ``{\'name\': \'value\'}`` and the SQL query needs to contain\n        ``:name``.\n        The formatter will be applied client-side in this scenario.\n\n        For ``paramstyle="qmark"``, this value needs to be a list of strings.\n        The formatter will be applied server-side.\n        The values are applied sequentially to the parameters in the query in the order in which the parameters occur.\n    paramstyle: str, optional\n        Determines the style of ``params``.\n        Possible values are:\n\n        - ``named``\n        - ``qmark``\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    client_request_token : str, optional\n        A unique case-sensitive string used to ensure the request to create the query is idempotent (executes only once).\n        If another StartQueryExecution request is received, the same response is returned and another query is not created.\n        If a parameter has changed, for example, the QueryString , an error is returned.\n        If you pass the same client_request_token value with different parameters the query fails with error\n        message "Idempotent parameters do not match". Use this only with ctas_approach=False and unload_approach=False\n        and disabled cache.\n    athena_cache_settings: typing.AthenaCacheSettings, optional\n        Parameters of the Athena cache settings such as max_cache_seconds, max_cache_query_inspections,\n        max_remote_cache_entries, and max_local_cache_entries.\n        AthenaCacheSettings is a `TypedDict`, meaning the passed parameter can be instantiated either as an\n        instance of AthenaCacheSettings or as a regular Python dict.\n        If cached results are valid, awswrangler ignores the `ctas_approach`, `s3_output`, `encryption`, `kms_key`,\n        `keep_files` and `ctas_temp_table_name` params.\n        If reading cached data fails for any reason, execution falls back to the usual query run path.\n    athena_query_wait_polling_delay: float, default: 0.25 seconds\n        Interval in seconds for how often the function will check if the Athena query has completed.\n    data_source : str, optional\n        Data Source / Catalog name. If None, \'AwsDataCatalog\' will be used by default.\n    wait : bool, default False\n        Indicates whether to wait for the query to finish and return a dictionary with the query execution response.\n\n    Returns\n    -------\n    Union[str, Dict[str, Any]]\n        Query execution ID if `wait` is set to `False`, dictionary with the get_query_execution response otherwise.\n\n    Examples\n    --------\n    Querying into the default data source (Amazon s3 - \'AwsDataCatalog\')\n\n    >>> import awswrangler as wr\n    >>> query_exec_id = wr.athena.start_query_execution(sql=\'...\', database=\'...\')\n\n    Querying into another data source (PostgreSQL, Redshift, etc)\n\n    >>> import awswrangler as wr\n    >>> query_exec_id = wr.athena.start_query_execution(sql=\'...\', database=\'...\', data_source=\'...\')\n\n    '
    (sql, execution_params) = _apply_formatter(sql, params, paramstyle)
    _logger.debug('Executing query:\n%s', sql)
    if not client_request_token:
        cache_info: _CacheInfo = _check_for_cached_results(sql=sql, boto3_session=boto3_session, workgroup=workgroup, athena_cache_settings=athena_cache_settings)
        _logger.debug('Cache info:\n%s', cache_info)
    if not client_request_token and cache_info.has_valid_cache and (cache_info.query_execution_id is not None):
        query_execution_id = cache_info.query_execution_id
        _logger.debug('Valid cache found. Retrieving...')
    else:
        wg_config: _WorkGroupConfig = _get_workgroup_config(session=boto3_session, workgroup=workgroup)
        query_execution_id = _start_query_execution(sql=sql, wg_config=wg_config, database=database, data_source=data_source, s3_output=s3_output, workgroup=workgroup, encryption=encryption, kms_key=kms_key, execution_params=execution_params, client_request_token=client_request_token, boto3_session=boto3_session)
    if wait:
        return wait_query(query_execution_id=query_execution_id, boto3_session=boto3_session, athena_query_wait_polling_delay=athena_query_wait_polling_delay)
    return query_execution_id

def stop_query_execution(query_execution_id: str, boto3_session: Optional[boto3.Session]=None) -> None:
    if False:
        i = 10
        return i + 15
    "Stop a query execution.\n\n    Requires you to have access to the workgroup in which the query ran.\n\n    Parameters\n    ----------\n    query_execution_id : str\n        Athena query execution ID.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    None\n        None.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> wr.athena.stop_query_execution(query_execution_id='query-execution-id')\n\n    "
    client_athena = _utils.client(service_name='athena', session=boto3_session)
    client_athena.stop_query_execution(QueryExecutionId=query_execution_id)

@apply_configs
def wait_query(query_execution_id: str, boto3_session: Optional[boto3.Session]=None, athena_query_wait_polling_delay: float=_QUERY_WAIT_POLLING_DELAY) -> Dict[str, Any]:
    if False:
        print('Hello World!')
    "Wait for the query end.\n\n    Parameters\n    ----------\n    query_execution_id : str\n        Athena query execution ID.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    athena_query_wait_polling_delay: float, default: 0.25 seconds\n        Interval in seconds for how often the function will check if the Athena query has completed.\n\n    Returns\n    -------\n    Dict[str, Any]\n        Dictionary with the get_query_execution response.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> res = wr.athena.wait_query(query_execution_id='query-execution-id')\n\n    "
    response: Dict[str, Any] = get_query_execution(query_execution_id=query_execution_id, boto3_session=boto3_session)
    state: str = response['Status']['State']
    while state not in _QUERY_FINAL_STATES:
        time.sleep(athena_query_wait_polling_delay)
        response = get_query_execution(query_execution_id=query_execution_id, boto3_session=boto3_session)
        state = response['Status']['State']
    _logger.debug('Query state: %s', state)
    _logger.debug('Query state change reason: %s', response['Status'].get('StateChangeReason'))
    if state == 'FAILED':
        raise exceptions.QueryFailed(response['Status'].get('StateChangeReason'))
    if state == 'CANCELLED':
        raise exceptions.QueryCancelled(response['Status'].get('StateChangeReason'))
    return response

def get_query_execution(query_execution_id: str, boto3_session: Optional[boto3.Session]=None) -> Dict[str, Any]:
    if False:
        i = 10
        return i + 15
    "Fetch query execution details.\n\n    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/athena.html#Athena.Client.get_query_execution\n\n    Parameters\n    ----------\n    query_execution_id : str\n        Athena query execution ID.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, Any]\n        Dictionary with the get_query_execution response.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> res = wr.athena.get_query_execution(query_execution_id='query-execution-id')\n\n    "
    client_athena = _utils.client(service_name='athena', session=boto3_session)
    response = _utils.try_it(f=client_athena.get_query_execution, ex=botocore.exceptions.ClientError, ex_code='ThrottlingException', max_num_tries=5, QueryExecutionId=query_execution_id)
    _logger.debug('Get query execution response:\n%s', response)
    return cast(Dict[str, Any], response['QueryExecution'])