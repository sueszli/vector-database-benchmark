from unittest.mock import Mock
import pytest
from celery.backends.consul import ConsulBackend
pytest.importorskip('consul')

class test_ConsulBackend:

    def setup_method(self):
        if False:
            return 10
        self.backend = ConsulBackend(app=self.app, url='consul://localhost:800')

    def test_supports_autoexpire(self):
        if False:
            i = 10
            return i + 15
        assert self.backend.supports_autoexpire

    def test_consul_consistency(self):
        if False:
            while True:
                i = 10
        assert self.backend.consistency == 'consistent'

    def test_get(self):
        if False:
            while True:
                i = 10
        index = 100
        data = {'Key': 'test-consul-1', 'Value': 'mypayload'}
        self.backend.one_client = Mock(name='c.client')
        self.backend.one_client.kv.get.return_value = (index, data)
        assert self.backend.get(data['Key']) == 'mypayload'

    def test_set(self):
        if False:
            return 10
        self.backend.one_client = Mock(name='c.client')
        self.backend.one_client.session.create.return_value = 'c8dfa770-4ea3-2ee9-d141-98cf0bfe9c59'
        self.backend.one_client.kv.put.return_value = True
        assert self.backend.set('Key', 'Value') is True

    def test_delete(self):
        if False:
            return 10
        self.backend.one_client = Mock(name='c.client')
        self.backend.one_client.kv.delete.return_value = True
        assert self.backend.delete('Key') is True

    def test_index_bytes_key(self):
        if False:
            for i in range(10):
                print('nop')
        key = 'test-consul-2'
        assert self.backend._key_to_consul_key(key) == key
        assert self.backend._key_to_consul_key(key.encode('utf-8')) == key