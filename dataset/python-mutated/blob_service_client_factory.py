from metaflow.exception import MetaflowException
from metaflow.metaflow_config import AZURE_STORAGE_BLOB_SERVICE_ENDPOINT
from metaflow.plugins.azure.azure_utils import create_cacheable_default_azure_credentials, check_azure_deps
import os
import threading

class _ClientCache(object):
    """
    azure.storage.blob.BlobServiceClient objects internally cache HTTPS connections.
    In order to reuse HTTPS connections, we need to reuse BlobServiceClient objects.

    The effect we are going for is for each process or thread in the executor to
    EACH REUSE ITS OWN long-lived BlobServiceClient object.
    """
    _cache = dict()

    def __init__(self):
        if False:
            while True:
                i = 10
        raise RuntimeError('_ClientCache may not be instantiated!')

    @staticmethod
    def get(blob_service_endpoint, credential=None, max_single_put_size=None, max_single_get_size=None, max_chunk_get_size=None, connection_data_block_size=None):
        if False:
            print('Hello World!')
        "\n        Each spawned process will get its own fresh cache.\n\n        With each process's cache, each thread gets its own distinct service object.\n\n        The cache key includes all BlobServiceClient creation params PLUS process ID and thread ID.\n\n        This means that no more than one thread of control will ever access a cache key. This, and\n        the fact that dict() operations are thread-safe means that no additional synchronization\n        required here.\n        "
        cache_key = (os.getpid(), threading.get_ident(), credential, blob_service_endpoint, max_single_put_size, max_single_get_size, max_chunk_get_size, connection_data_block_size)
        service_just_created = None
        if cache_key not in _ClientCache._cache:
            service_just_created = _create_blob_service_client(blob_service_endpoint, credential=credential, max_single_put_size=max_single_put_size, max_single_get_size=max_single_get_size, max_chunk_get_size=max_chunk_get_size, connection_data_block_size=connection_data_block_size)
            _ClientCache._cache[cache_key] = service_just_created
        service_to_return = _ClientCache._cache[cache_key]
        if service_just_created is not None and service_just_created != service_to_return:
            print('METAFLOW WARNING: Azure _ClientCache had the same cache key updated more than once')
        return service_to_return
AZURE_CLIENT_CONNECTION_DATA_BLOCK_SIZE = 262144
AZURE_CLIENT_MAX_SINGLE_GET_SIZE_MB = 32
AZURE_CLIENT_MAX_SINGLE_PUT_SIZE_MB = 64
AZURE_CLIENT_MAX_CHUNK_GET_SIZE_MB = 16
BYTES_IN_MB = 1024 * 1024

def get_azure_blob_service_client(credential=None, credential_is_cacheable=False, max_single_get_size=AZURE_CLIENT_MAX_SINGLE_GET_SIZE_MB * BYTES_IN_MB, max_single_put_size=AZURE_CLIENT_MAX_SINGLE_PUT_SIZE_MB * BYTES_IN_MB, max_chunk_get_size=AZURE_CLIENT_MAX_CHUNK_GET_SIZE_MB * BYTES_IN_MB, connection_data_block_size=AZURE_CLIENT_CONNECTION_DATA_BLOCK_SIZE):
    if False:
        i = 10
        return i + 15
    'Returns a azure.storage.blob.BlobServiceClient.\n\n    The value adds are:\n    - connection caching (see _ClientCache)\n    - auto storage account URL detection\n    - auto credential handling (pull SAS token from environment, OR DefaultAzureCredential)\n    - sensible default values for Azure SDK tunables\n    '
    if not AZURE_STORAGE_BLOB_SERVICE_ENDPOINT:
        raise MetaflowException(msg='Must configure METAFLOW_AZURE_STORAGE_BLOB_SERVICE_ENDPOINT')
    blob_service_endpoint = AZURE_STORAGE_BLOB_SERVICE_ENDPOINT
    if not credential:
        credential = create_cacheable_default_azure_credentials()
        credential_is_cacheable = True
    if not credential_is_cacheable:
        return _create_blob_service_client(blob_service_endpoint, credential=credential, max_single_put_size=max_single_put_size, max_single_get_size=max_single_get_size, max_chunk_get_size=max_chunk_get_size, connection_data_block_size=connection_data_block_size)
    return _ClientCache.get(blob_service_endpoint, credential=credential, max_single_put_size=max_single_put_size, max_single_get_size=max_single_get_size, max_chunk_get_size=max_chunk_get_size, connection_data_block_size=connection_data_block_size)

@check_azure_deps
def _create_blob_service_client(blob_service_endpoint, credential=None, max_single_put_size=None, max_single_get_size=None, max_chunk_get_size=None, connection_data_block_size=None):
    if False:
        print('Hello World!')
    from azure.storage.blob import BlobServiceClient
    return BlobServiceClient(blob_service_endpoint, credential=credential, max_single_put_size=max_single_put_size, max_single_get_size=max_single_get_size, max_chunk_get_size=max_chunk_get_size, connection_data_block_size=connection_data_block_size)