import logging
from datetime import datetime
from io import IOBase
from typing import Iterable, List, Optional, Set
import boto3.session
import pytz
import smart_open
from airbyte_cdk.models import FailureType
from airbyte_cdk.sources.file_based.exceptions import CustomFileBasedException, ErrorListingFiles, FileBasedSourceError
from airbyte_cdk.sources.file_based.file_based_stream_reader import AbstractFileBasedStreamReader, FileReadMode
from airbyte_cdk.sources.file_based.remote_file import RemoteFile
from botocore.client import BaseClient
from botocore.client import Config as ClientConfig
from botocore.exceptions import ClientError
from source_s3.v4.config import Config
from source_s3.v4.zip_reader import DecompressedStream, RemoteFileInsideArchive, ZipContentReader, ZipFileHandler

class SourceS3StreamReader(AbstractFileBasedStreamReader):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self._s3_client = None

    @property
    def config(self) -> Config:
        if False:
            i = 10
            return i + 15
        return self._config

    @config.setter
    def config(self, value: Config):
        if False:
            print('Hello World!')
        "\n        FileBasedSource reads the config from disk and parses it, and once parsed, the source sets the config on its StreamReader.\n\n        Note: FileBasedSource only requires the keys defined in the abstract config, whereas concrete implementations of StreamReader\n        will require keys that (for example) allow it to authenticate with the 3rd party.\n\n        Therefore, concrete implementations of AbstractFileBasedStreamReader's config setter should assert that `value` is of the correct\n        config type for that type of StreamReader.\n        "
        assert isinstance(value, Config)
        self._config = value

    @property
    def s3_client(self) -> BaseClient:
        if False:
            while True:
                i = 10
        if self.config is None:
            raise ValueError('Source config is missing; cannot create the S3 client.')
        if self._s3_client is None:
            client_kv_args = _get_s3_compatible_client_args(self.config) if self.config.endpoint else {}
            self._s3_client = boto3.client('s3', aws_access_key_id=self.config.aws_access_key_id, aws_secret_access_key=self.config.aws_secret_access_key, **client_kv_args)
        return self._s3_client

    def get_matching_files(self, globs: List[str], prefix: Optional[str], logger: logging.Logger) -> Iterable[RemoteFile]:
        if False:
            i = 10
            return i + 15
        '\n        Get all files matching the specified glob patterns.\n        '
        s3 = self.s3_client
        prefixes = [prefix] if prefix else self.get_prefixes_from_globs(globs)
        seen = set()
        total_n_keys = 0
        try:
            for current_prefix in prefixes if prefixes else [None]:
                for remote_file in self._page(s3, globs, self.config.bucket, current_prefix, seen, logger):
                    total_n_keys += 1
                    yield remote_file
            logger.info(f'Finished listing objects from S3. Found {total_n_keys} objects total ({len(seen)} unique objects).')
        except ClientError as exc:
            if exc.response['Error']['Code'] == 'NoSuchBucket':
                raise CustomFileBasedException(f'The bucket {self.config.bucket} does not exist.', failure_type=FailureType.config_error, exception=exc)
            self._raise_error_listing_files(globs, exc)
        except Exception as exc:
            self._raise_error_listing_files(globs, exc)

    def _raise_error_listing_files(self, globs: List[str], exc: Optional[Exception]=None):
        if False:
            while True:
                i = 10
        'Helper method to raise the ErrorListingFiles exception.'
        raise ErrorListingFiles(FileBasedSourceError.ERROR_LISTING_FILES, source='s3', bucket=self.config.bucket, globs=globs, endpoint=self.config.endpoint) from exc

    def open_file(self, file: RemoteFile, mode: FileReadMode, encoding: Optional[str], logger: logging.Logger) -> IOBase:
        if False:
            return 10
        try:
            params = {'client': self.s3_client}
        except Exception as exc:
            raise exc
        logger.debug(f'try to open {file.uri}')
        try:
            if isinstance(file, RemoteFileInsideArchive):
                s3_file_object = smart_open.open(f"s3://{self.config.bucket}/{file.uri.split('#')[0]}", transport_params=params, mode='rb')
                decompressed_stream = DecompressedStream(s3_file_object, file)
                result = ZipContentReader(decompressed_stream, encoding)
            else:
                result = smart_open.open(f's3://{self.config.bucket}/{file.uri}', transport_params=params, mode=mode.value, encoding=encoding)
        except OSError:
            logger.warning(f"We don't have access to {file.uri}. The file appears to have become unreachable during sync.Check whether key {file.uri} exists in `{self.config.bucket}` bucket and/or has proper ACL permissions")
        return result

    @staticmethod
    def _is_folder(file) -> bool:
        if False:
            return 10
        return file['Key'].endswith('/')

    def _page(self, s3: BaseClient, globs: List[str], bucket: str, prefix: Optional[str], seen: Set[str], logger: logging.Logger) -> Iterable[RemoteFile]:
        if False:
            i = 10
            return i + 15
        '\n        Page through lists of S3 objects.\n        '
        total_n_keys_for_prefix = 0
        kwargs = {'Bucket': bucket}
        while True:
            response = s3.list_objects_v2(Prefix=prefix, **kwargs) if prefix else s3.list_objects_v2(**kwargs)
            key_count = response.get('KeyCount')
            total_n_keys_for_prefix += key_count
            logger.info(f"Received {key_count} objects from S3 for prefix '{prefix}'.")
            if 'Contents' in response:
                for file in response['Contents']:
                    if self._is_folder(file):
                        continue
                    for remote_file in self._handle_file(file):
                        if self.file_matches_globs(remote_file, globs) and remote_file.uri not in seen:
                            seen.add(remote_file.uri)
                            yield remote_file
            else:
                logger.warning(f"Invalid response from S3; missing 'Contents' key. kwargs={kwargs}.")
            if (next_token := response.get('NextContinuationToken')):
                kwargs['ContinuationToken'] = next_token
            else:
                logger.info(f'Finished listing objects from S3 for prefix={prefix}. Found {total_n_keys_for_prefix} objects.')
                break

    def _handle_file(self, file):
        if False:
            return 10
        if file['Key'].endswith('zip'):
            yield from self._handle_zip_file(file)
        else:
            yield self._handle_regular_file(file)

    def _handle_zip_file(self, file):
        if False:
            return 10
        zip_handler = ZipFileHandler(self.s3_client, self.config)
        (zip_members, cd_start) = zip_handler.get_zip_files(file['Key'])
        for zip_member in zip_members:
            remote_file = RemoteFileInsideArchive(uri=file['Key'] + '#' + zip_member.filename, last_modified=datetime(*zip_member.date_time).astimezone(pytz.utc).replace(tzinfo=None), start_offset=zip_member.header_offset + cd_start, compressed_size=zip_member.compress_size, uncompressed_size=zip_member.file_size, compression_method=zip_member.compress_type)
            yield remote_file

    def _handle_regular_file(self, file):
        if False:
            return 10
        remote_file = RemoteFile(uri=file['Key'], last_modified=file['LastModified'].astimezone(pytz.utc).replace(tzinfo=None))
        return remote_file

def _get_s3_compatible_client_args(config: Config) -> dict:
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns map of args used for creating s3 boto3 client.\n    '
    client_kv_args = {'config': ClientConfig(s3={'addressing_style': 'auto'}), 'endpoint_url': config.endpoint, 'use_ssl': True, 'verify': True}
    return client_kv_args