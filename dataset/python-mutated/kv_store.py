import logging
from typing import Optional
import ray
from ray._private import ray_constants
from ray._raylet import GcsClient
from ray.serve._private.constants import RAY_SERVE_KV_TIMEOUT_S, SERVE_LOGGER_NAME
from ray.serve._private.storage.kv_store_base import KVStoreBase
logger = logging.getLogger(SERVE_LOGGER_NAME)

def get_storage_key(namespace: str, storage_key: str) -> str:
    if False:
        i = 10
        return i + 15
    'In case we need to access kvstore'
    return '{ns}-{key}'.format(ns=namespace, key=storage_key)

class KVStoreError(Exception):

    def __init__(self, rpc_code):
        if False:
            while True:
                i = 10
        self.rpc_code = rpc_code

class RayInternalKVStore(KVStoreBase):
    """Wraps ray's internal_kv with a namespace to avoid collisions.

    Supports string keys and bytes values, caller must handle serialization.
    """

    def __init__(self, namespace: Optional[str]=None, gcs_client: Optional[GcsClient]=None):
        if False:
            while True:
                i = 10
        if namespace is not None and (not isinstance(namespace, str)):
            raise TypeError('namespace must a string, got: {}.'.format(type(namespace)))
        if gcs_client is not None:
            self.gcs_client = gcs_client
        else:
            self.gcs_client = GcsClient(address=ray.get_runtime_context().gcs_address)
        self.timeout = RAY_SERVE_KV_TIMEOUT_S
        self.namespace = namespace or ''

    def get_storage_key(self, key: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        return '{ns}-{key}'.format(ns=self.namespace, key=key)

    def put(self, key: str, val: bytes) -> bool:
        if False:
            i = 10
            return i + 15
        'Put the key-value pair into the store.\n\n        Args:\n            key (str)\n            val (bytes)\n        '
        if not isinstance(key, str):
            raise TypeError('key must be a string, got: {}.'.format(type(key)))
        if not isinstance(val, bytes):
            raise TypeError('val must be bytes, got: {}.'.format(type(val)))
        try:
            return self.gcs_client.internal_kv_put(self.get_storage_key(key).encode(), val, overwrite=True, namespace=ray_constants.KV_NAMESPACE_SERVE, timeout=self.timeout)
        except ray.exceptions.RpcError as e:
            raise KVStoreError(e.rpc_code)

    def get(self, key: str) -> Optional[bytes]:
        if False:
            print('Hello World!')
        "Get the value associated with the given key from the store.\n\n        Args:\n            key (str)\n\n        Returns:\n            The bytes value. If the key wasn't found, returns None.\n        "
        if not isinstance(key, str):
            raise TypeError('key must be a string, got: {}.'.format(type(key)))
        try:
            return self.gcs_client.internal_kv_get(self.get_storage_key(key).encode(), namespace=ray_constants.KV_NAMESPACE_SERVE, timeout=self.timeout)
        except ray.exceptions.RpcError as e:
            raise KVStoreError(e.rpc_code)

    def delete(self, key: str):
        if False:
            while True:
                i = 10
        'Delete the value associated with the given key from the store.\n\n        Args:\n            key (str)\n        '
        if not isinstance(key, str):
            raise TypeError('key must be a string, got: {}.'.format(type(key)))
        try:
            return self.gcs_client.internal_kv_del(self.get_storage_key(key).encode(), False, namespace=ray_constants.KV_NAMESPACE_SERVE, timeout=self.timeout)
        except ray.exceptions.RpcError as e:
            raise KVStoreError(e.rpc_code)