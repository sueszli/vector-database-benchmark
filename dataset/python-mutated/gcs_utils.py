import logging
from typing import Optional
from ray._private import ray_constants
import ray._private.gcs_aio_client
from ray.core.generated.common_pb2 import ErrorType, JobConfig
from ray.core.generated.gcs_pb2 import ActorTableData, AvailableResources, ErrorTableData, GcsEntry, GcsNodeInfo, JobTableData, ObjectTableData, PlacementGroupTableData, PubSubMessage, ResourceDemand, ResourceLoad, ResourcesData, ResourceUsageBatchData, TablePrefix, TablePubsub, TaskEvents, WorkerTableData
logger = logging.getLogger(__name__)
__all__ = ['ActorTableData', 'GcsNodeInfo', 'AvailableResources', 'JobTableData', 'JobConfig', 'ErrorTableData', 'ErrorType', 'GcsEntry', 'ResourceUsageBatchData', 'ResourcesData', 'ObjectTableData', 'TablePrefix', 'TablePubsub', 'TaskEvents', 'ResourceDemand', 'ResourceLoad', 'PubSubMessage', 'WorkerTableData', 'PlacementGroupTableData']
WORKER = 0
DRIVER = 1
_MAX_MESSAGE_LENGTH = 512 * 1024 * 1024
_GRPC_KEEPALIVE_TIME_MS = 60 * 1000
_GRPC_KEEPALIVE_TIMEOUT_MS = 60 * 1000
_GRPC_OPTIONS = [*ray_constants.GLOBAL_GRPC_OPTIONS, ('grpc.max_send_message_length', _MAX_MESSAGE_LENGTH), ('grpc.max_receive_message_length', _MAX_MESSAGE_LENGTH), ('grpc.keepalive_time_ms', _GRPC_KEEPALIVE_TIME_MS), ('grpc.keepalive_timeout_ms', _GRPC_KEEPALIVE_TIMEOUT_MS)]

def create_gcs_channel(address: str, aio=False):
    if False:
        i = 10
        return i + 15
    'Returns a GRPC channel to GCS.\n\n    Args:\n        address: GCS address string, e.g. ip:port\n        aio: Whether using grpc.aio\n    Returns:\n        grpc.Channel or grpc.aio.Channel to GCS\n    '
    from ray._private.utils import init_grpc_channel
    return init_grpc_channel(address, options=_GRPC_OPTIONS, asynchronous=aio)

class GcsChannel:

    def __init__(self, gcs_address: Optional[str]=None, aio: bool=False):
        if False:
            i = 10
            return i + 15
        self._gcs_address = gcs_address
        self._aio = aio

    @property
    def address(self):
        if False:
            for i in range(10):
                print('nop')
        return self._gcs_address

    def connect(self):
        if False:
            for i in range(10):
                print('nop')
        self._channel = create_gcs_channel(self._gcs_address, self._aio)

    def channel(self):
        if False:
            print('Hello World!')
        return self._channel
GcsAioClient = ray._private.gcs_aio_client.GcsAioClient

def cleanup_redis_storage(host: str, port: int, password: str, use_ssl: bool, storage_namespace: str):
    if False:
        print('Hello World!')
    'This function is used to cleanup the storage. Before we having\n    a good design for storage backend, it can be used to delete the old\n    data. It support redis cluster and non cluster mode.\n\n    Args:\n       host: The host address of the Redis.\n       port: The port of the Redis.\n       password: The password of the Redis.\n       use_ssl: Whether to encrypt the connection.\n       storage_namespace: The namespace of the storage to be deleted.\n    '
    from ray._raylet import del_key_from_storage
    if not isinstance(host, str):
        raise ValueError('Host must be a string')
    if not isinstance(password, str):
        raise ValueError('Password must be a string')
    if port < 0:
        raise ValueError(f'Invalid port: {port}')
    if not isinstance(use_ssl, bool):
        raise TypeError('use_ssl must be a boolean')
    if not isinstance(storage_namespace, str):
        raise ValueError('storage namespace must be a string')
    return del_key_from_storage(host, port, password, use_ssl, storage_namespace)