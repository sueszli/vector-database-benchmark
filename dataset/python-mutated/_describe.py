"""Amazon S3 Describe Module (INTERNAL)."""
import datetime
import itertools
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, cast
import boto3
from awswrangler import _utils
from awswrangler._distributed import engine
from awswrangler._executor import _BaseExecutor, _get_executor
from awswrangler.distributed.ray import ray_get
from awswrangler.s3 import _fs
from awswrangler.s3._list import _path2list
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
_logger: logging.Logger = logging.getLogger(__name__)

@engine.dispatch_on_engine
def _describe_object(s3_client: 'S3Client', path: str, s3_additional_kwargs: Optional[Dict[str, Any]], version_id: Optional[str]=None) -> Tuple[str, Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    s3_client = s3_client if s3_client else _utils.client(service_name='s3')
    (bucket, key) = _utils.parse_path(path=path)
    if s3_additional_kwargs:
        extra_kwargs: Dict[str, Any] = _fs.get_botocore_valid_kwargs(function_name='head_object', s3_additional_kwargs=s3_additional_kwargs)
    else:
        extra_kwargs = {}
    if version_id:
        extra_kwargs['VersionId'] = version_id
    desc = _utils.try_it(f=s3_client.head_object, ex=s3_client.exceptions.NoSuchKey, Bucket=bucket, Key=key, **extra_kwargs)
    return (path, cast(Dict[str, Any], desc))

@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session', 's3_additional_kwargs'])
def describe_objects(path: Union[str, List[str]], version_id: Optional[Union[str, Dict[str, str]]]=None, use_threads: Union[bool, int]=True, last_modified_begin: Optional[datetime.datetime]=None, last_modified_end: Optional[datetime.datetime]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, Dict[str, Any]]:
    if False:
        for i in range(10):
            print('nop')
    "Describe Amazon S3 objects from a received S3 prefix or list of S3 objects paths.\n\n    Fetch attributes like ContentLength, DeleteMarker, last_modified, ContentType, etc\n    The full list of attributes can be explored under the boto3 head_object documentation:\n    https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.head_object\n\n    This function accepts Unix shell-style wildcards in the path argument.\n    * (matches everything), ? (matches any single character),\n    [seq] (matches any character in seq), [!seq] (matches any character not in seq).\n    If you want to use a path which includes Unix shell-style wildcard characters (`*, ?, []`),\n    you can use `glob.escape(path)` before passing the path to this function.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Note\n    ----\n    The filter by last_modified begin last_modified end is applied after list all S3 files\n\n    Parameters\n    ----------\n    path : Union[str, List[str]]\n        S3 prefix (accepts Unix shell-style wildcards)\n        (e.g. s3://bucket/prefix) or list of S3 objects paths (e.g. [s3://bucket/key0, s3://bucket/key1]).\n    version_id: Optional[Union[str, Dict[str, str]]]\n        Version id of the object or mapping of object path to version id.\n        (e.g. {'s3://bucket/key0': '121212', 's3://bucket/key1': '343434'})\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    last_modified_begin\n        Filter the s3 files by the Last modified date of the object.\n        The filter is applied only after list all s3 files.\n    last_modified_end: datetime, optional\n        Filter the s3 files by the Last modified date of the object.\n        The filter is applied only after list all s3 files.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forwarded to botocore requests.\n        e.g. s3_additional_kwargs={'RequestPayer': 'requester'}\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, Dict[str, Any]]\n        Return a dictionary of objects returned from head_objects where the key is the object path.\n        The response object can be explored here:\n        https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html#S3.Client.head_object\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> descs0 = wr.s3.describe_objects(['s3://bucket/key0', 's3://bucket/key1'])  # Describe both objects\n    >>> descs1 = wr.s3.describe_objects('s3://bucket/prefix')  # Describe all objects under the prefix\n\n    "
    s3_client = _utils.client(service_name='s3', session=boto3_session)
    paths = _path2list(path=path, s3_client=s3_client, last_modified_begin=last_modified_begin, last_modified_end=last_modified_end, s3_additional_kwargs=s3_additional_kwargs)
    if len(paths) < 1:
        return {}
    executor: _BaseExecutor = _get_executor(use_threads=use_threads)
    resp_list = ray_get(executor.map(_describe_object, s3_client, paths, itertools.repeat(s3_additional_kwargs), [version_id.get(p) if isinstance(version_id, dict) else version_id for p in paths]))
    return dict(resp_list)

@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session', 's3_additional_kwargs'])
def size_objects(path: Union[str, List[str]], version_id: Optional[Union[str, Dict[str, str]]]=None, use_threads: Union[bool, int]=True, s3_additional_kwargs: Optional[Dict[str, Any]]=None, boto3_session: Optional[boto3.Session]=None) -> Dict[str, Optional[int]]:
    if False:
        return 10
    "Get the size (ContentLength) in bytes of Amazon S3 objects from a received S3 prefix or list of S3 objects paths.\n\n    This function accepts Unix shell-style wildcards in the path argument.\n    * (matches everything), ? (matches any single character),\n    [seq] (matches any character in seq), [!seq] (matches any character not in seq).\n    If you want to use a path which includes Unix shell-style wildcard characters (`*, ?, []`),\n    you can use `glob.escape(path)` before passing the path to this function.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    path : Union[str, List[str]]\n        S3 prefix (accepts Unix shell-style wildcards)\n        (e.g. s3://bucket/prefix) or list of S3 objects paths (e.g. [s3://bucket/key0, s3://bucket/key1]).\n    version_id: Optional[Union[str, Dict[str, str]]]\n        Version id of the object or mapping of object path to version id.\n        (e.g. {'s3://bucket/key0': '121212', 's3://bucket/key1': '343434'})\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forwarded to botocore requests.\n        e.g. s3_additional_kwargs={'RequestPayer': 'requester'}\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    Dict[str, Optional[int]]\n        Dictionary where the key is the object path and the value is the object size.\n\n    Examples\n    --------\n    >>> import awswrangler as wr\n    >>> sizes0 = wr.s3.size_objects(['s3://bucket/key0', 's3://bucket/key1'])  # Get the sizes of both objects\n    >>> sizes1 = wr.s3.size_objects('s3://bucket/prefix')  # Get the sizes of all objects under the received prefix\n\n    "
    desc_list = describe_objects(path=path, version_id=version_id, use_threads=use_threads, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs)
    return {k: d.get('ContentLength', None) for (k, d) in desc_list.items()}

def get_bucket_region(bucket: str, boto3_session: Optional[boto3.Session]=None) -> str:
    if False:
        return 10
    "Get bucket region name.\n\n    Parameters\n    ----------\n    bucket : str\n        Bucket name.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n\n    Returns\n    -------\n    str\n        Region code (e.g. 'us-east-1').\n\n    Examples\n    --------\n    Using the default boto3 session\n\n    >>> import awswrangler as wr\n    >>> region = wr.s3.get_bucket_region('bucket-name')\n\n    Using a custom boto3 session\n\n    >>> import boto3\n    >>> import awswrangler as wr\n    >>> region = wr.s3.get_bucket_region('bucket-name', boto3_session=boto3.Session())\n\n    "
    client_s3 = _utils.client(service_name='s3', session=boto3_session)
    _logger.debug('bucket: %s', bucket)
    region: str = client_s3.get_bucket_location(Bucket=bucket)['LocationConstraint']
    region = 'us-east-1' if region is None else region
    _logger.debug('region: %s', region)
    return region