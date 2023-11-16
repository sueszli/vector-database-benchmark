import json
import re
import uuid
from typing import Optional
import dagster._check as check
from dagster import InitResourceContext, Noneable, StringSource, resource
from dagster._core.storage.file_manager import check_file_like_obj
from dagster_gcp.gcs.file_manager import GCSFileHandle, GCSFileManager
from google.cloud import storage
from google.oauth2 import service_account
from orchestrator.config import get_public_url_for_gcs_file

class PublicGCSFileHandle(GCSFileHandle):

    @property
    def public_url(self):
        if False:
            return 10
        return get_public_url_for_gcs_file(self.gcs_bucket, self.gcs_key)

class ContentTypeAwareGCSFileManager(GCSFileManager):
    """
    Slighlty modified dagster_gcp.gcs.file_manager.GCSFileManager
    to allow setting the content type of the file
    """

    def get_content_type(self, ext):
        if False:
            i = 10
            return i + 15
        if ext == 'csv':
            return 'text/csv'
        elif ext == 'json':
            return 'application/json'
        elif ext == 'html':
            return 'text/html'
        elif ext == 'md':
            return 'text/markdown'
        else:
            return 'text/plain'

    def get_full_key(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        full_key = super().get_full_key(*args, **kwargs)
        if full_key.startswith('/'):
            full_key = full_key[1:]
        return full_key

    def write(self, file_obj, mode='wb', ext=None, key: Optional[str]=None) -> PublicGCSFileHandle:
        if False:
            for i in range(10):
                print('nop')
        '\n        Reworked from dagster_gcp.gcs.file_manager.GCSFileManager.write\n\n        As the original method does not allow to set the content type of the file\n        '
        key = check.opt_str_param(key, 'key', default=str(uuid.uuid4()))
        check_file_like_obj(file_obj)
        gcs_key = self.get_full_key(key + ('.' + ext if ext is not None else ''))
        bucket_obj = self._client.bucket(self._gcs_bucket)
        blob = bucket_obj.blob(gcs_key)
        blob.cache_control = 'no-cache'
        blob.content_type = self.get_content_type(ext)
        blob.upload_from_file(file_obj)
        return PublicGCSFileHandle(self._gcs_bucket, gcs_key)

    def delete_by_key(self, key: str, ext: Optional[str]=None) -> Optional[PublicGCSFileHandle]:
        if False:
            for i in range(10):
                print('nop')
        gcs_key = self.get_full_key(key + ('.' + ext if ext is not None else ''))
        bucket_obj = self._client.bucket(self._gcs_bucket)
        blob = bucket_obj.blob(gcs_key)
        if not blob.exists():
            return None
        blob.delete()
        return PublicGCSFileHandle(self._gcs_bucket, gcs_key)

@resource(config_schema={'gcp_gcs_cred_string': StringSource})
def gcp_gcs_client(resource_context: InitResourceContext) -> storage.Client:
    if False:
        print('Hello World!')
    'Create a connection to gcs.'
    resource_context.log.info('retrieving gcp_gcs_client')
    gcp_gcs_cred_string = resource_context.resource_config['gcp_gcs_cred_string']
    gcp_gsm_cred_json = json.loads(gcp_gcs_cred_string)
    credentials = service_account.Credentials.from_service_account_info(gcp_gsm_cred_json)
    return storage.Client(credentials=credentials, project=credentials.project_id)

@resource(required_resource_keys={'gcp_gcs_client'}, config_schema={'gcs_bucket': StringSource, 'prefix': StringSource})
def gcs_file_manager(resource_context) -> GCSFileManager:
    if False:
        i = 10
        return i + 15
    'FileManager that provides abstract access to GCS.\n\n    Implements the :py:class:`~dagster._core.storage.file_manager.FileManager` API.\n    '
    storage_client = resource_context.resources.gcp_gcs_client
    return ContentTypeAwareGCSFileManager(client=storage_client, gcs_bucket=resource_context.resource_config['gcs_bucket'], gcs_base_key=resource_context.resource_config['prefix'])

@resource(required_resource_keys={'gcp_gcs_client'}, config_schema={'gcs_bucket': StringSource, 'prefix': Noneable(StringSource), 'gcs_filename': StringSource})
def gcs_file_blob(resource_context: InitResourceContext) -> storage.Blob:
    if False:
        i = 10
        return i + 15
    '\n    Create a connection to a gcs file blob.\n\n    This is implemented so we are able to retrieve the metadata of a file\n    before committing to downloading the file.\n    '
    gcs_bucket = resource_context.resource_config['gcs_bucket']
    storage_client = resource_context.resources.gcp_gcs_client
    bucket = storage_client.get_bucket(gcs_bucket)
    prefix = resource_context.resource_config['prefix']
    gcs_filename = resource_context.resource_config['gcs_filename']
    gcs_file_path = f'{prefix}/{gcs_filename}' if prefix else gcs_filename
    resource_context.log.info(f'retrieving gcs file blob {gcs_file_path} in bucket: {gcs_bucket}')
    gcs_file_blob = bucket.get_blob(gcs_file_path)
    if not gcs_file_blob or not gcs_file_blob.exists():
        raise Exception(f'File does not exist at path: {gcs_file_path}')
    return gcs_file_blob

@resource(required_resource_keys={'gcp_gcs_client'}, config_schema={'gcs_bucket': StringSource, 'prefix': StringSource, 'match_regex': StringSource})
def gcs_directory_blobs(resource_context: InitResourceContext) -> storage.Blob:
    if False:
        i = 10
        return i + 15
    '\n    List all blobs in a bucket that match the prefix.\n    '
    gcs_bucket = resource_context.resource_config['gcs_bucket']
    prefix = resource_context.resource_config['prefix']
    match_regex = resource_context.resource_config['match_regex']
    storage_client = resource_context.resources.gcp_gcs_client
    bucket = storage_client.get_bucket(gcs_bucket)
    resource_context.log.info(f'retrieving gcs file blobs for prefix: {prefix}, match_regex: {match_regex}, in bucket: {gcs_bucket}')
    gcs_file_blobs = bucket.list_blobs(prefix=prefix)
    if match_regex:
        gcs_file_blobs = [blob for blob in gcs_file_blobs if re.match(match_regex, blob.name)]
    return gcs_file_blobs