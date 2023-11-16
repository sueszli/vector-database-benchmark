import logging
import time
import etcd3

class ETCDClient:

    def __init__(self, host, port, retry_times=20):
        if False:
            for i in range(10):
                print('nop')
        self.retry_times = retry_times
        times = 0
        while times < self.retry_times:
            try:
                self.client = etcd3.client(host=host, port=port)
                break
            except Exception as e:
                times += 1
                logging.info(f'Initialize etcd client failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Initialize etcd client failed failed after {self.retry_times} times.')

    def put(self, key, value, lease=None, prev_kv=False):
        if False:
            i = 10
            return i + 15
        times = 0
        while times < self.retry_times:
            try:
                return self.client.put(key, value, lease, prev_kv)
            except Exception as e:
                times += 1
                logging.info(f'Put failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Put failed after {self.retry_times} times.')

    def get(self, key):
        if False:
            for i in range(10):
                print('nop')
        times = 0
        while times < self.retry_times:
            try:
                return self.client.get(key)
                break
            except Exception as e:
                times += 1
                logging.info(f'Get {key} failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Get {key} failed after {self.retry_times} times.')

    def delete(self, key, prev_kv=False, return_response=False):
        if False:
            i = 10
            return i + 15
        times = 0
        while times < self.retry_times:
            try:
                return self.client.delete(key, prev_kv, return_response)
                break
            except Exception as e:
                times += 1
                logging.info(f'Delete {key} failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Delete {key} failed after {self.retry_times} times.')

    def get_prefix(self, key_prefix, sort_order=None, sort_target='key'):
        if False:
            i = 10
            return i + 15
        times = 0
        while times < self.retry_times:
            try:
                return self.client.get_prefix(key_prefix)
                break
            except Exception as e:
                times += 1
                logging.info(f'Get prefix {key_prefix} failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Get prefix {key_prefix} failed after {self.retry_times} times.')

    def delete_prefix(self, prefix):
        if False:
            while True:
                i = 10
        times = 0
        while times < self.retry_times:
            try:
                return self.client.delete_prefix(prefix)
                break
            except Exception as e:
                times += 1
                logging.info(f'Delete prefix {prefix} failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Delete prefix {prefix} failed after {self.retry_times} times.')

    def lease(self, ttl, lease_id=None):
        if False:
            i = 10
            return i + 15
        times = 0
        while times < self.retry_times:
            try:
                return self.client.lease(ttl, lease_id)
                break
            except Exception as e:
                times += 1
                logging.info(f'Lease failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Lease failed after {self.retry_times} times.')

    def add_watch_prefix_callback(self, key_prefix, callback, **kwargs):
        if False:
            return 10
        times = 0
        while times < self.retry_times:
            try:
                return self.client.add_watch_prefix_callback(key_prefix, callback, **kwargs)
                break
            except Exception as e:
                times += 1
                logging.info(f'Add watch prefix callback failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Add watch prefix callback failed after {self.retry_times} times.')

    def cancel_watch(self, watch_id):
        if False:
            for i in range(10):
                print('nop')
        times = 0
        while times < self.retry_times:
            try:
                return self.client.cancel_watch(watch_id)
                break
            except Exception as e:
                times += 1
                logging.info(f'Cancel watch failed with exception {e}, retry after 1 second.')
                time.sleep(1)
        if times >= self.retry_times:
            raise ValueError(f'Cancel watch failed after {self.retry_times} times.')