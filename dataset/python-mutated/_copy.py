"""Amazon S3 Copy Module (PRIVATE)."""
import itertools
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, Tuple, Union
import boto3
from boto3.s3.transfer import TransferConfig
from awswrangler import _utils, exceptions
from awswrangler._distributed import engine
from awswrangler._executor import _BaseExecutor, _get_executor
from awswrangler.distributed.ray import ray_get
from awswrangler.s3._delete import delete_objects
from awswrangler.s3._fs import get_botocore_valid_kwargs
from awswrangler.s3._list import list_objects
if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from mypy_boto3_s3.type_defs import CopySourceTypeDef
_logger: logging.Logger = logging.getLogger(__name__)

@engine.dispatch_on_engine
def _copy_objects(s3_client: Optional['S3Client'], batch: List[Tuple[str, str]], use_threads: Union[bool, int], s3_additional_kwargs: Optional[Dict[str, Any]]) -> None:
    if False:
        i = 10
        return i + 15
    _logger.debug('Copying %s objects', len(batch))
    s3_client = s3_client if s3_client else _utils.client(service_name='s3')
    for (source, target) in batch:
        (source_bucket, source_key) = _utils.parse_path(path=source)
        copy_source: CopySourceTypeDef = {'Bucket': source_bucket, 'Key': source_key}
        (target_bucket, target_key) = _utils.parse_path(path=target)
        s3_client.copy(CopySource=copy_source, Bucket=target_bucket, Key=target_key, ExtraArgs=s3_additional_kwargs, Config=TransferConfig(num_download_attempts=10, use_threads=use_threads))

def _copy(batches: List[List[Tuple[str, str]]], use_threads: Union[bool, int], boto3_session: Optional[boto3.Session], s3_additional_kwargs: Optional[Dict[str, Any]]) -> None:
    if False:
        i = 10
        return i + 15
    s3_client = _utils.client(service_name='s3', session=boto3_session)
    if s3_additional_kwargs is None:
        boto3_kwargs: Optional[Dict[str, Any]] = None
    else:
        boto3_kwargs = get_botocore_valid_kwargs(function_name='copy_object', s3_additional_kwargs=s3_additional_kwargs)
    executor: _BaseExecutor = _get_executor(use_threads=use_threads)
    ray_get(executor.map(_copy_objects, s3_client, batches, itertools.repeat(use_threads), itertools.repeat(boto3_kwargs)))

@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session'])
def merge_datasets(source_path: str, target_path: str, mode: Literal['append', 'overwrite', 'overwrite_partitions']='append', ignore_empty: bool=False, use_threads: Union[bool, int]=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Merge a source dataset into a target dataset.\n\n    This function accepts Unix shell-style wildcards in the source_path argument.\n    * (matches everything), ? (matches any single character),\n    [seq] (matches any character in seq), [!seq] (matches any character not in seq).\n    If you want to use a path which includes Unix shell-style wildcard characters (`*, ?, []`),\n    you can use `glob.escape(source_path)` before passing the path to this function.\n\n    Note\n    ----\n    If you are merging tables (S3 datasets + Glue Catalog metadata),\n    remember that you will also need to update your partitions metadata in some cases.\n    (e.g. wr.athena.repair_table(table=\'...\', database=\'...\'))\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    source_path : str,\n        S3 Path for the source directory.\n    target_path : str,\n        S3 Path for the target directory.\n    mode: str, optional\n        ``append`` (Default), ``overwrite``, ``overwrite_partitions``.\n    ignore_empty: bool\n        Ignore files with 0 bytes.\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forwarded to botocore requests.\n        e.g. s3_additional_kwargs={\'ServerSideEncryption\': \'aws:kms\', \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'}\n\n    Returns\n    -------\n    List[str]\n        List of new objects paths.\n\n    Examples\n    --------\n    Merging\n\n    >>> import awswrangler as wr\n    >>> wr.s3.merge_datasets(\n    ...     source_path="s3://bucket0/dir0/",\n    ...     target_path="s3://bucket1/dir1/",\n    ...     mode="append"\n    ... )\n    ["s3://bucket1/dir1/key0", "s3://bucket1/dir1/key1"]\n\n    Merging with a KMS key\n\n    >>> import awswrangler as wr\n    >>> wr.s3.merge_datasets(\n    ...     source_path="s3://bucket0/dir0/",\n    ...     target_path="s3://bucket1/dir1/",\n    ...     mode="append",\n    ...     s3_additional_kwargs={\n    ...         \'ServerSideEncryption\': \'aws:kms\',\n    ...         \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'\n    ...     }\n    ... )\n    ["s3://bucket1/dir1/key0", "s3://bucket1/dir1/key1"]\n\n    '
    source_path = source_path[:-1] if source_path[-1] == '/' else source_path
    target_path = target_path[:-1] if target_path[-1] == '/' else target_path
    paths: List[str] = list_objects(path=f'{source_path}/', ignore_empty=ignore_empty, boto3_session=boto3_session)
    if len(paths) < 1:
        return []
    if mode == 'overwrite':
        _logger.debug('Deleting to overwrite: %s/', target_path)
        delete_objects(path=f'{target_path}/', use_threads=use_threads, boto3_session=boto3_session)
    elif mode == 'overwrite_partitions':
        paths_wo_prefix: List[str] = [x.replace(f'{source_path}/', '') for x in paths]
        paths_wo_filename: List[str] = [f"{x.rpartition('/')[0]}/" for x in paths_wo_prefix]
        partitions_paths: List[str] = list(set(paths_wo_filename))
        target_partitions_paths = [f'{target_path}/{x}' for x in partitions_paths]
        for path in target_partitions_paths:
            _logger.debug('Deleting to overwrite_partitions: %s', path)
            delete_objects(path=path, use_threads=use_threads, boto3_session=boto3_session)
    elif mode != 'append':
        raise exceptions.InvalidArgumentValue(f'{mode} is a invalid mode option.')
    new_objects: List[str] = copy_objects(paths=paths, source_path=source_path, target_path=target_path, use_threads=use_threads, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs)
    return new_objects

@_utils.validate_distributed_kwargs(unsupported_kwargs=['boto3_session'])
def copy_objects(paths: List[str], source_path: str, target_path: str, replace_filenames: Optional[Dict[str, str]]=None, use_threads: Union[bool, int]=True, boto3_session: Optional[boto3.Session]=None, s3_additional_kwargs: Optional[Dict[str, Any]]=None) -> List[str]:
    if False:
        return 10
    'Copy a list of S3 objects to another S3 directory.\n\n    Note\n    ----\n    In case of `use_threads=True` the number of threads\n    that will be spawned will be gotten from os.cpu_count().\n\n    Parameters\n    ----------\n    paths : List[str]\n        List of S3 objects paths (e.g. [s3://bucket/dir0/key0, s3://bucket/dir0/key1]).\n    source_path : str,\n        S3 Path for the source directory.\n    target_path : str,\n        S3 Path for the target directory.\n    replace_filenames : Dict[str, str], optional\n        e.g. {"old_name.csv": "new_name.csv", "old_name2.csv": "new_name2.csv"}\n    use_threads : bool, int\n        True to enable concurrent requests, False to disable multiple threads.\n        If enabled os.cpu_count() will be used as the max number of threads.\n        If integer is provided, specified number is used.\n    boto3_session : boto3.Session(), optional\n        Boto3 Session. The default boto3 session will be used if boto3_session receive None.\n    s3_additional_kwargs : Optional[Dict[str, Any]]\n        Forwarded to botocore requests.\n        e.g. s3_additional_kwargs={\'ServerSideEncryption\': \'aws:kms\', \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'}\n\n    Returns\n    -------\n    List[str]\n        List of new objects paths.\n\n    Examples\n    --------\n    Copying\n\n    >>> import awswrangler as wr\n    >>> wr.s3.copy_objects(\n    ...     paths=["s3://bucket0/dir0/key0", "s3://bucket0/dir0/key1"],\n    ...     source_path="s3://bucket0/dir0/",\n    ...     target_path="s3://bucket1/dir1/"\n    ... )\n    ["s3://bucket1/dir1/key0", "s3://bucket1/dir1/key1"]\n\n    Copying with a KMS key\n\n    >>> import awswrangler as wr\n    >>> wr.s3.copy_objects(\n    ...     paths=["s3://bucket0/dir0/key0", "s3://bucket0/dir0/key1"],\n    ...     source_path="s3://bucket0/dir0/",\n    ...     target_path="s3://bucket1/dir1/",\n    ...     s3_additional_kwargs={\n    ...         \'ServerSideEncryption\': \'aws:kms\',\n    ...         \'SSEKMSKeyId\': \'YOUR_KMS_KEY_ARN\'\n    ...     }\n    ... )\n    ["s3://bucket1/dir1/key0", "s3://bucket1/dir1/key1"]\n\n    '
    if len(paths) < 1:
        return []
    source_path = source_path[:-1] if source_path[-1] == '/' else source_path
    target_path = target_path[:-1] if target_path[-1] == '/' else target_path
    batch: List[Tuple[str, str]] = []
    new_objects: List[str] = []
    for path in paths:
        path_wo_prefix: str = path.replace(f'{source_path}/', '')
        path_final: str = f'{target_path}/{path_wo_prefix}'
        if replace_filenames is not None:
            parts: List[str] = path_final.rsplit(sep='/', maxsplit=1)
            if len(parts) == 2:
                path_wo_filename: str = parts[0]
                filename: str = parts[1]
                if filename in replace_filenames:
                    new_filename: str = replace_filenames[filename]
                    _logger.debug('Replacing filename: %s -> %s', filename, new_filename)
                    path_final = f'{path_wo_filename}/{new_filename}'
        new_objects.append(path_final)
        batch.append((path, path_final))
    _logger.debug('Creating %s new objects', len(new_objects))
    _copy(batches=_utils.chunkify(lst=batch, max_length=1000), use_threads=use_threads, boto3_session=boto3_session, s3_additional_kwargs=s3_additional_kwargs)
    return new_objects