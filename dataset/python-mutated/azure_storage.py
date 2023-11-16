import json
import os
import shutil
import uuid
import time
from concurrent.futures import as_completed
from tempfile import mkdtemp
from metaflow.datastore.datastore_storage import DataStoreStorage, CloseAfterUse
from metaflow.exception import MetaflowInternalError
from metaflow.metaflow_config import DATASTORE_SYSROOT_AZURE, ARTIFACT_LOCALROOT, AZURE_STORAGE_WORKLOAD_TYPE
from metaflow.plugins.azure.azure_utils import check_azure_deps, process_exception, handle_exceptions, create_static_token_credential, parse_azure_full_path
from metaflow.plugins.azure.blob_service_client_factory import get_azure_blob_service_client
from metaflow.plugins.storage_executor import StorageExecutor, handle_executor_exceptions
AZURE_STORAGE_DOWNLOAD_MAX_CONCURRENCY = 4
AZURE_STORAGE_UPLOAD_MAX_CONCURRENCY = 16
BYTES_IN_MB = 1024 * 1024
AZURE_STORAGE_DEFAULT_SCOPE = 'https://storage.azure.com/.default'

class _AzureRootClient(object):
    """
    This exists independent of AzureBlobStorage as a wrapper around SDK clients.
    It carries around parameters needed to construct Azure SDK clients on demand.

    _AzureRootClient objects will be passed from main to worker threads or processes. They
    must be picklable. We delay constructing Azure SDK objects because they are not picklable.

    For example, azure.core.TokenCredential objects are not picklable. Therefore, we pass around an
    AccessToken (token) instead, and construct TokenCredential on demand in the target thread (or process).
    Note that we do this to amortize credential retrieval cost across threads (or processes). Depending on
    the credential methods available to DefaultAzureCredential, credential retrieval can be expensive.
    E.g. Azure CLI based credential may take 500-1000ms.

    _AzureRootClient  also carries around with it blob methods that operate relative to
    datastore_root.
    """

    def __init__(self, datastore_root=None, token=None, shared_access_signature=None):
        if False:
            print('Hello World!')
        if datastore_root is None:
            raise MetaflowInternalError('datastore_root must be set')
        if token is None and shared_access_signature is None:
            raise MetaflowInternalError('either shared_access_signature or token must be set')
        if token and shared_access_signature:
            raise MetaflowInternalError('cannot set both shared_access_signature and token')
        self._datastore_root = datastore_root
        self._token = token
        self._shared_access_signature = shared_access_signature

    def get_datastore_root(self):
        if False:
            print('Hello World!')
        return self._datastore_root

    def get_blob_container_client(self):
        if False:
            for i in range(10):
                print('nop')
        if self._shared_access_signature:
            credential = self._shared_access_signature
            credential_is_cacheable = True
        else:
            credential = create_static_token_credential(self._token)
            credential_is_cacheable = True
        service = get_azure_blob_service_client(credential=credential, credential_is_cacheable=credential_is_cacheable)
        (container_name, _) = parse_azure_full_path(self._datastore_root)
        return service.get_container_client(container_name)

    def get_blob_client(self, path):
        if False:
            return 10
        container = self.get_blob_container_client()
        blob_full_path = self.get_blob_full_path(path)
        return container.get_blob_client(blob_full_path)

    def get_blob_full_path(self, path):
        if False:
            return 10
        '\n        Full path means <blob_prefix>/<path> where:\n        datastore_root is <container_name>/<blob_prefix>\n        '
        (_, blob_prefix) = parse_azure_full_path(self._datastore_root)
        if blob_prefix is None:
            return path
        path = path.lstrip('/')
        return '/'.join([blob_prefix, path])

    def is_file_single(self, path):
        if False:
            while True:
                i = 10
        'Drives AzureStorage.is_file()'
        try:
            blob = self.get_blob_client(path)
            return blob.exists()
        except Exception as e:
            process_exception(e)

    def save_bytes_single(self, path_tmpfile_metadata_triple, overwrite=False):
        if False:
            print('Hello World!')
        'Drives AzureStorage.save_bytes()'
        try:
            (path, tmpfile, metadata) = path_tmpfile_metadata_triple
            metadata_to_upload = None
            if metadata:
                metadata_to_upload = {'metaflow_user_attributes': json.dumps(metadata)}
            blob = self.get_blob_client(path)
            from azure.core.exceptions import ResourceExistsError
            with open(tmpfile, 'rb') as byte_stream:
                try:
                    if overwrite or not blob.exists():
                        blob.upload_blob(byte_stream, overwrite=overwrite, metadata=metadata_to_upload, max_concurrency=AZURE_STORAGE_UPLOAD_MAX_CONCURRENCY)
                except ResourceExistsError:
                    if overwrite:
                        raise
                    else:
                        pass
        except Exception as e:
            process_exception(e)

    def load_bytes_single(self, tmpdir, key):
        if False:
            i = 10
            return i + 15
        'Drives AzureStorage.load_bytes()'
        from azure.core.exceptions import ResourceNotFoundError
        try:
            blob = self.get_blob_client(key)
            try:
                blob_properties = blob.get_blob_properties()
            except ResourceNotFoundError:
                return (key, None, None)
            tmp_filename = os.path.join(tmpdir, str(uuid.uuid4()))
            try:
                with open(tmp_filename, 'wb') as f:
                    blob.download_blob(max_concurrency=AZURE_STORAGE_DOWNLOAD_MAX_CONCURRENCY).readinto(f)
                metaflow_user_attributes = None
                if blob_properties.metadata and 'metaflow_user_attributes' in blob_properties.metadata:
                    metaflow_user_attributes = json.loads(blob_properties.metadata['metaflow_user_attributes'])
            except Exception:
                if os.path.exists(tmp_filename):
                    os.unlink(tmp_filename)
                raise
            return (key, tmp_filename, metaflow_user_attributes)
        except Exception as e:
            process_exception(e)

    def list_content_single(self, path):
        if False:
            i = 10
            return i + 15
        'Drives AzureStorage.list_content()'
        try:
            result = []
            path = path.rstrip('/') + '/'
            full_path = self.get_blob_full_path(path)
            container = self.get_blob_container_client()
            for blob_properties_or_prefix in container.walk_blobs(name_starts_with=full_path):
                name = blob_properties_or_prefix.name
                is_file = blob_properties_or_prefix.has_key('blob_type')
                if not is_file:
                    name = name.rstrip('/')
                (_, top_level_blob_prefix) = parse_azure_full_path(self.get_datastore_root())
                if top_level_blob_prefix is not None and name[:len(top_level_blob_prefix)] == top_level_blob_prefix:
                    name = name[len(top_level_blob_prefix) + 1:]
                result.append((name, is_file))
            return result
        except Exception as e:
            process_exception(e)

class AzureStorage(DataStoreStorage):
    TYPE = 'azure'

    def __init__(self, root=None):
        if False:
            for i in range(10):
                print('nop')
        check_azure_deps(lambda : 0)
        super(AzureStorage, self).__init__(root)
        self._tmproot = ARTIFACT_LOCALROOT
        self._default_scope_token = None
        self._root_client = None
        self._use_processes = AZURE_STORAGE_WORKLOAD_TYPE == 'high_throughput'
        self._executor = StorageExecutor(use_processes=self._use_processes)
        self._executor.warm_up()

    @handle_exceptions
    def _get_default_token(self):
        if False:
            print('Hello World!')
        if not self._default_scope_token or self._default_scope_token.expires_on - time.time() < 300:
            from azure.identity import DefaultAzureCredential
            with DefaultAzureCredential() as credential:
                self._default_scope_token = credential.get_token(AZURE_STORAGE_DEFAULT_SCOPE)
        return self._default_scope_token

    @property
    def root_client(self):
        if False:
            return 10
        'Note this is for optimization only - it allows slow-initialization credentials to be\n        reused across multiple threads and processes.\n\n        Speed up applies mainly to the "no access key" path.\n        '
        if self._root_client is None:
            self._root_client = _AzureRootClient(datastore_root=self.datastore_root, token=self._get_default_token())
        return self._root_client

    @classmethod
    def get_datastore_root_from_config(cls, echo, create_on_absent=True):
        if False:
            while True:
                i = 10
        return DATASTORE_SYSROOT_AZURE

    @handle_executor_exceptions
    def is_file(self, paths):
        if False:
            print('Hello World!')
        futures = [self._executor.submit(self.root_client.is_file_single, path) for path in paths]
        return [future.result() for future in futures]

    def info_file(self, path):
        if False:
            return 10
        raise NotImplementedError()

    def size_file(self, path):
        if False:
            i = 10
            return i + 15
        from azure.core.exceptions import ResourceNotFoundError
        try:
            return self.root_client.get_blob_client(path).get_blob_properties().size
        except ResourceNotFoundError:
            return None
        except Exception as e:
            process_exception(e)

    @handle_executor_exceptions
    def list_content(self, paths):
        if False:
            i = 10
            return i + 15
        futures = [self._executor.submit(self.root_client.list_content_single, path) for path in paths]
        result = []
        for future in as_completed(futures):
            result.extend((self.list_content_result(*x) for x in future.result()))
        return result

    @handle_executor_exceptions
    def save_bytes(self, path_and_bytes_iter, overwrite=False, len_hint=0):
        if False:
            print('Hello World!')
        tmpdir = None
        try:
            tmpdir = mkdtemp(dir=ARTIFACT_LOCALROOT, prefix='metaflow.azure.save_bytes.')
            futures = []
            for (path, byte_stream) in path_and_bytes_iter:
                metadata = None
                if isinstance(byte_stream, tuple):
                    (byte_stream, metadata) = byte_stream
                tmp_filename = os.path.join(tmpdir, str(uuid.uuid4()))
                with open(tmp_filename, 'wb') as f:
                    f.write(byte_stream.read())
                futures.append(self._executor.submit(self.root_client.save_bytes_single, (path, tmp_filename, metadata), overwrite=overwrite))
            for future in as_completed(futures):
                future.result()
        finally:
            if tmpdir and os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)

    @handle_executor_exceptions
    def load_bytes(self, keys):
        if False:
            for i in range(10):
                print('nop')
        tmpdir = mkdtemp(dir=self._tmproot, prefix='metaflow.azure.load_bytes.')
        try:
            futures = [self._executor.submit(self.root_client.load_bytes_single, tmpdir, key) for key in keys]
            items = [future.result() for future in as_completed(futures)]
        except Exception:
            if os.path.exists(tmpdir):
                shutil.rmtree(tmpdir)
            raise

        class _Closer(object):

            @staticmethod
            def close():
                if False:
                    while True:
                        i = 10
                if os.path.isdir(tmpdir):
                    shutil.rmtree(tmpdir)
        return CloseAfterUse(iter(items), closer=_Closer)