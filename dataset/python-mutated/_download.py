"""Amazon S3 Download Module (PRIVATE)."""
import logging
from typing import Any, Dict, Optional, Union, cast
import boto3
from awswrangler.s3._fs import open_s3_object
_logger: logging.Logger = logging.getLogger(__name__)

def download(path: str, local_file: Union[str, Any], version_id: Optional[str]=None, use_threads: Union[bool, int]=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None) -> None:
    if False:
        while True:
            i = 10
    'Download file from a received S3 path to local file.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    path : str\n        S3 path (e.g. ``s3://bucket/key0``).\n    local_file : Union[str, Any]\n        A file-like object in binary mode or a path to local file (e.g. ``./local/path/to/key0``).\n    version_id: Optional[str]\n        Version id of the object.\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forward to botocore requests, only "SSECustomerAlgorithm", "SSECustomerKey" and "RequestPayer"\n        arguments will be considered.\n\n    Returns\n    -------\n    None\n\n    Examples\n    --------\n    Downloading a file using a path to local file\n\n    >>> import awswrangler as wr\n    >>> wr.s3.download(path=\'s3://bucket/key\', local_file=\'./key\')\n\n    Downloading a file using a file-like object\n\n    >>> import awswrangler as wr\n    >>> with open(file=\'./key\', mode=\'wb\') as local_f:\n    >>>     wr.s3.download(path=\'s3://bucket/key\', local_file=local_f)\n\n    '
    _logger.debug('path: %s', path)
    with open_s3_object(path=path, mode='rb', use_threads=use_threads, version_id=version_id, s3_block_size=-1, s3_additional_kwargs=s3_additional_kwargs, boto3_session=boto3_session) as s3_f:
        if isinstance(local_file, str):
            _logger.debug('Downloading local_file: %s', local_file)
            with open(file=local_file, mode='wb') as local_f:
                local_f.write(cast(bytes, s3_f.read()))
        else:
            _logger.debug('Downloading file-like object.')
            local_file.write(s3_f.read())