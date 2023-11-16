from typing import List, Optional, Union
from ray._private.client_mode_hook import client_mode_hook
from ray._raylet import GcsClient
_initialized = False
global_gcs_client = None

def _internal_kv_reset():
    if False:
        i = 10
        return i + 15
    global global_gcs_client, _initialized
    global_gcs_client = None
    _initialized = False

def internal_kv_get_gcs_client():
    if False:
        for i in range(10):
            print('nop')
    return global_gcs_client

def _initialize_internal_kv(gcs_client: GcsClient):
    if False:
        print('Hello World!')
    'Initialize the internal KV for use in other function calls.'
    global global_gcs_client, _initialized
    assert gcs_client is not None
    global_gcs_client = gcs_client
    _initialized = True

@client_mode_hook
def _internal_kv_initialized():
    if False:
        print('Hello World!')
    return global_gcs_client is not None

@client_mode_hook
def _internal_kv_get(key: Union[str, bytes], *, namespace: Optional[Union[str, bytes]]=None) -> bytes:
    if False:
        i = 10
        return i + 15
    'Fetch the value of a binary key.'
    if isinstance(key, str):
        key = key.encode()
    if isinstance(namespace, str):
        namespace = namespace.encode()
    assert isinstance(key, bytes)
    return global_gcs_client.internal_kv_get(key, namespace)

@client_mode_hook
def _internal_kv_exists(key: Union[str, bytes], *, namespace: Optional[Union[str, bytes]]=None) -> bool:
    if False:
        while True:
            i = 10
    'Check key exists or not.'
    if isinstance(key, str):
        key = key.encode()
    if isinstance(namespace, str):
        namespace = namespace.encode()
    assert isinstance(key, bytes)
    return global_gcs_client.internal_kv_exists(key, namespace)

@client_mode_hook
def _pin_runtime_env_uri(uri: str, *, expiration_s: int) -> None:
    if False:
        return 10
    'Pin a runtime_env URI for expiration_s.'
    return global_gcs_client.pin_runtime_env_uri(uri, expiration_s)

@client_mode_hook
def _internal_kv_put(key: Union[str, bytes], value: Union[str, bytes], overwrite: bool=True, *, namespace: Optional[Union[str, bytes]]=None) -> bool:
    if False:
        while True:
            i = 10
    'Globally associates a value with a given binary key.\n\n    This only has an effect if the key does not already have a value.\n\n    Returns:\n        already_exists: whether the value already exists.\n    '
    if isinstance(key, str):
        key = key.encode()
    if isinstance(value, str):
        value = value.encode()
    if isinstance(namespace, str):
        namespace = namespace.encode()
    assert isinstance(key, bytes) and isinstance(value, bytes) and isinstance(overwrite, bool)
    return global_gcs_client.internal_kv_put(key, value, overwrite, namespace) == 0

@client_mode_hook
def _internal_kv_del(key: Union[str, bytes], *, del_by_prefix: bool=False, namespace: Optional[Union[str, bytes]]=None) -> int:
    if False:
        while True:
            i = 10
    if isinstance(key, str):
        key = key.encode()
    if isinstance(namespace, str):
        namespace = namespace.encode()
    assert isinstance(key, bytes)
    return global_gcs_client.internal_kv_del(key, del_by_prefix, namespace)

@client_mode_hook
def _internal_kv_list(prefix: Union[str, bytes], *, namespace: Optional[Union[str, bytes]]=None) -> List[bytes]:
    if False:
        i = 10
        return i + 15
    'List all keys in the internal KV store that start with the prefix.'
    if isinstance(prefix, str):
        prefix = prefix.encode()
    if isinstance(namespace, str):
        namespace = namespace.encode()
    return global_gcs_client.internal_kv_keys(prefix, namespace)