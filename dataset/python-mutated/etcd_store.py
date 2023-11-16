import datetime
import random
import time
from base64 import b64decode, b64encode
from typing import Optional
import etcd
from torch.distributed import Store

def cas_delay():
    if False:
        while True:
            i = 10
    time.sleep(random.uniform(0, 0.1))

class EtcdStore(Store):
    """
    Implement a c10 Store interface by piggybacking on the rendezvous etcd instance.

    This is the store object returned by ``EtcdRendezvous``.
    """

    def __init__(self, etcd_client, etcd_store_prefix, timeout: Optional[datetime.timedelta]=None):
        if False:
            print('Hello World!')
        super().__init__()
        self.client = etcd_client
        self.prefix = etcd_store_prefix
        if timeout is not None:
            self.set_timeout(timeout)
        if not self.prefix.endswith('/'):
            self.prefix += '/'

    def set(self, key, value):
        if False:
            print('Hello World!')
        '\n        Write a key/value pair into ``EtcdStore``.\n\n        Both key and value may be either Python ``str`` or ``bytes``.\n        '
        self.client.set(key=self.prefix + self._encode(key), value=self._encode(value))

    def get(self, key) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        '\n        Get a value by key, possibly doing a blocking wait.\n\n        If key is not immediately present, will do a blocking wait\n        for at most ``timeout`` duration or until the key is published.\n\n\n        Returns:\n            value ``(bytes)``\n\n        Raises:\n            LookupError - If key still not published after timeout\n        '
        b64_key = self.prefix + self._encode(key)
        kvs = self._try_wait_get([b64_key])
        if kvs is None:
            raise LookupError(f'Key {key} not found in EtcdStore')
        return self._decode(kvs[b64_key])

    def add(self, key, num: int) -> int:
        if False:
            return 10
        '\n        Atomically increment a value by an integer amount.\n\n        The integer is represented as a string using base 10. If key is not present,\n        a default value of ``0`` will be assumed.\n\n        Returns:\n             the new (incremented) value\n\n\n        '
        b64_key = self._encode(key)
        try:
            node = self.client.write(key=self.prefix + b64_key, value=self._encode(str(num)), prevExist=False)
            return int(self._decode(node.value))
        except etcd.EtcdAlreadyExist:
            pass
        while True:
            node = self.client.get(key=self.prefix + b64_key)
            new_value = self._encode(str(int(self._decode(node.value)) + num))
            try:
                node = self.client.test_and_set(key=node.key, value=new_value, prev_value=node.value)
                return int(self._decode(node.value))
            except etcd.EtcdCompareFailed:
                cas_delay()

    def wait(self, keys, override_timeout: Optional[datetime.timedelta]=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait until all of the keys are published, or until timeout.\n\n        Raises:\n            LookupError - if timeout occurs\n        '
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout)
        if kvs is None:
            raise LookupError('Timeout while waiting for keys in EtcdStore')

    def check(self, keys) -> bool:
        if False:
            for i in range(10):
                print('nop')
        'Check if all of the keys are immediately present (without waiting).'
        b64_keys = [self.prefix + self._encode(key) for key in keys]
        kvs = self._try_wait_get(b64_keys, override_timeout=datetime.timedelta(microseconds=1))
        return kvs is not None

    def _encode(self, value) -> str:
        if False:
            for i in range(10):
                print('nop')
        if type(value) == bytes:
            return b64encode(value).decode()
        elif type(value) == str:
            return b64encode(value.encode()).decode()
        raise ValueError('Value must be of type str or bytes')

    def _decode(self, value) -> bytes:
        if False:
            for i in range(10):
                print('nop')
        if type(value) == bytes:
            return b64decode(value)
        elif type(value) == str:
            return b64decode(value.encode())
        raise ValueError('Value must be of type str or bytes')

    def _try_wait_get(self, b64_keys, override_timeout=None):
        if False:
            return 10
        timeout = self.timeout if override_timeout is None else override_timeout
        deadline = time.time() + timeout.total_seconds()
        while True:
            all_nodes = self.client.get(key=self.prefix)
            req_nodes = {node.key: node.value for node in all_nodes.children if node.key in b64_keys}
            if len(req_nodes) == len(b64_keys):
                return req_nodes
            watch_timeout = deadline - time.time()
            if watch_timeout <= 0:
                return None
            try:
                self.client.watch(key=self.prefix, recursive=True, timeout=watch_timeout, index=all_nodes.etcd_index + 1)
            except etcd.EtcdWatchTimedOut:
                if time.time() >= deadline:
                    return None
                else:
                    continue
            except etcd.EtcdEventIndexCleared:
                continue