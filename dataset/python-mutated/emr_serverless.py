"""EMR Serverless module."""
import logging
import pprint
import time
from typing import Any, Dict, List, Literal, Optional, TypedDict, Union
import boto3
from typing_extensions import NotRequired, Required
from awswrangler import _utils, exceptions
from awswrangler._config import apply_configs
from awswrangler.annotations import Experimental
_logger: logging.Logger = logging.getLogger(__name__)
_EMR_SERVERLESS_JOB_WAIT_POLLING_DELAY: float = 5
_EMR_SERVERLESS_JOB_FINAL_STATES: List[str] = ['SUCCESS', 'FAILED', 'CANCELLED']

class SparkSubmitJobArgs(TypedDict):
    """Typed dictionary defining the Spark submit job arguments."""
    entryPoint: Required[str]
    'The entry point for the Spark submit job run.'
    entryPointArguments: NotRequired[List[str]]
    'The arguments for the Spark submit job run.'
    sparkSubmitParameters: NotRequired[str]
    'The parameters for the Spark submit job run.'

class HiveRunJobArgs(TypedDict):
    """Typed dictionary defining the Hive job run arguments."""
    query: Required[str]
    'The S3 location of the query file for the Hive job run.'
    initQueryFile: NotRequired[str]
    'The S3 location of the query file for the Hive job run.'
    parameters: NotRequired[str]
    'The parameters for the Hive job run.'

@Experimental
def create_application(name: str, release_label: str, application_type: Literal['Spark', 'Hive']='Spark', initial_capacity: Optional[Dict[str, str]]=None, maximum_capacity: Optional[Dict[str, str]]=None, tags: Optional[Dict[str, str]]=None, autostart: bool=True, autostop: bool=True, idle_timeout: int=15, network_configuration: Optional[Dict[str, str]]=None, architecture: Literal['ARM64', 'X86_64']='X86_64', image_uri: Optional[str]=None, worker_type_specifications: Optional[Dict[str, str]]=None, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        return 10
    '\n    Create an EMR Serverless application.\n\n    https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html\n\n    Parameters\n    ----------\n    name : str\n        Name of EMR Serverless appliation\n    release_label : str\n        Release label e.g. `emr-6.10.0`\n    application_type : str, optional\n        Application type: "Spark" or "Hive". Defaults to "Spark".\n    initial_capacity : Dict[str, str], optional\n        The capacity to initialize when the application is created.\n    maximum_capacity : Dict[str, str], optional\n        The maximum capacity to allocate when the application is created.\n        This is cumulative across all workers at any given point in time,\n        not just when an application is created. No new resources will\n        be created once any one of the defined limits is hit.\n    tags : Dict[str, str], optional\n        Key/Value collection to put tags on the application.\n        e.g. {"foo": "boo", "bar": "xoo"})\n    autostart : bool, optional\n        Enables the application to automatically start on job submission. Defaults to true.\n    autostop : bool, optional\n        Enables the application to automatically stop after a certain amount of time being idle. Defaults to true.\n    idle_timeout : int, optional\n        The amount of idle time in minutes after which your application will automatically stop. Defaults to 15 minutes.\n    network_configuration : Dict[str, str], optional\n        The network configuration for customer VPC connectivity.\n    architecture : str, optional\n        The CPU architecture of an application: "ARM64" or "X86_64". Defaults to "X86_64".\n    image_uri : str, optional\n        The URI of an image in the Amazon ECR registry.\n    worker_type_specifications : Dict[str, str], optional\n        The key-value pairs that specify worker type.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Application Id.\n    '
    emr_serverless = _utils.client(service_name='emr-serverless', session=boto3_session)
    application_args: Dict[str, Any] = {'name': name, 'releaseLabel': release_label, 'type': application_type, 'autoStartConfiguration': {'enabled': autostart}, 'autoStopConfiguration': {'enabled': autostop, 'idleTimeoutMinutes': idle_timeout}, 'architecture': architecture}
    if initial_capacity:
        application_args['initialCapacity'] = initial_capacity
    if maximum_capacity:
        application_args['maximumCapacity'] = maximum_capacity
    if tags:
        application_args['tags'] = tags
    if network_configuration:
        application_args['networkConfiguration'] = network_configuration
    if worker_type_specifications:
        application_args['workerTypeSpecifications'] = worker_type_specifications
    if image_uri:
        application_args['imageConfiguration'] = {'imageUri': image_uri}
    response: Dict[str, str] = emr_serverless.create_application(**application_args)
    _logger.debug('response: \n%s', pprint.pformat(response))
    return response['applicationId']

@Experimental
@apply_configs
def run_job(application_id: str, execution_role_arn: str, job_driver_args: Union[Dict[str, Any], SparkSubmitJobArgs, HiveRunJobArgs], job_type: Literal['Spark', 'Hive']='Spark', wait: bool=True, configuration_overrides: Optional[Dict[str, Any]]=None, tags: Optional[Dict[str, str]]=None, execution_timeout: Optional[int]=None, name: Optional[str]=None, emr_serverless_job_wait_polling_delay: float=_EMR_SERVERLESS_JOB_WAIT_POLLING_DELAY, boto3_session: Optional[boto3.Session]=None) -> Union[str, Dict[str, Any]]:
    if False:
        print('Hello World!')
    '\n    Run an EMR serverless job.\n\n    https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html\n\n    Parameters\n    ----------\n    application_id : str\n        The id of the application on which to run the job.\n    execution_role_arn : str\n        The execution role ARN for the job run.\n    job_driver_args : Union[Dict[str, str], SparkSubmitJobArgs, HiveRunJobArgs]\n        The job driver arguments for the job run.\n    job_type : str, optional\n        Type of the job: "Spark" or "Hive". Defaults to "Spark".\n    wait : bool, optional\n        Whether to wait for the job completion or not. Defaults to true.\n    configuration_overrides : Dict[str, str], optional\n        The configuration overrides for the job run.\n    tags : Dict[str, str], optional\n        Key/Value collection to put tags on the application.\n        e.g. {"foo": "boo", "bar": "xoo"})\n    execution_timeout : int, optional\n        The maximum duration for the job run to run. If the job run runs beyond this duration,\n        it will be automatically cancelled.\n    name : str, optional\n        Name of the job.\n    emr_serverless_job_wait_polling_delay : int, optional\n        Time to wait between polling attempts.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Union[str, Dict[str, Any]]\n        Job Id if wait=False, or job run details.\n    '
    emr_serverless = _utils.client(service_name='emr-serverless', session=boto3_session)
    job_args: Dict[str, Any] = {'applicationId': application_id, 'executionRoleArn': execution_role_arn}
    if job_type == 'Spark':
        job_args['jobDriver'] = {'sparkSubmit': job_driver_args}
    elif job_type == 'Hive':
        job_args['jobDriver'] = {'hive': job_driver_args}
    else:
        raise exceptions.InvalidArgumentValue(f'Unsupported job type `{job_type}`')
    if configuration_overrides:
        job_args['configurationOverrides'] = configuration_overrides
    if tags:
        job_args['tags'] = tags
    if execution_timeout:
        job_args['executionTimeoutMinutes'] = execution_timeout
    if name:
        job_args['name'] = name
    response = emr_serverless.start_job_run(**job_args)
    _logger.debug('Job run response: %s', response)
    job_run_id: str = response['jobRunId']
    if wait:
        return wait_job(application_id=application_id, job_run_id=job_run_id, emr_serverless_job_wait_polling_delay=emr_serverless_job_wait_polling_delay)
    return job_run_id

@Experimental
@apply_configs
def wait_job(application_id: str, job_run_id: str, emr_serverless_job_wait_polling_delay: float=_EMR_SERVERLESS_JOB_WAIT_POLLING_DELAY, boto3_session: Optional[boto3.Session]=None) -> Dict[str, Any]:
    if False:
        return 10
    '\n    Wait for the EMR Serverless job to finish.\n\n    https://docs.aws.amazon.com/emr/latest/EMR-Serverless-UserGuide/emr-serverless.html\n\n    Parameters\n    ----------\n    application_id : str\n        The id of the application on which the job is running.\n    job_run_id : str\n        The id of the job.\n    emr_serverless_job_wait_polling_delay : int, optional\n        Time to wait between polling attempts.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, Any]\n        Job run details.\n    '
    emr_serverless = _utils.client(service_name='emr-serverless', session=boto3_session)
    response = emr_serverless.get_job_run(applicationId=application_id, jobRunId=job_run_id)
    state = response['jobRun']['state']
    while state not in _EMR_SERVERLESS_JOB_FINAL_STATES:
        time.sleep(emr_serverless_job_wait_polling_delay)
        response = emr_serverless.get_job_run(applicationId=application_id, jobRunId=job_run_id)
        state = response['jobRun']['state']
    _logger.debug('Job state: %s', state)
    if state != 'SUCCESS':
        _logger.debug('Job run response: %s', response)
        raise exceptions.EMRServerlessJobError(response.get('jobRun', {}).get('stateDetails'))
    return response