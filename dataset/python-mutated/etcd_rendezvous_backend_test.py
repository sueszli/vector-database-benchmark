import subprocess
from base64 import b64encode
from typing import ClassVar, cast
from unittest import TestCase
from etcd import EtcdKeyNotFound
from torch.distributed.elastic.rendezvous import RendezvousConnectionError, RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous_backend import EtcdRendezvousBackend, create_backend
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
from torch.distributed.elastic.rendezvous.etcd_store import EtcdStore
from rendezvous_backend_test import RendezvousBackendTestMixin

class EtcdRendezvousBackendTest(TestCase, RendezvousBackendTestMixin):
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            i = 10
            return i + 15
        cls._server.stop()

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._client = self._server.get_client()
        try:
            self._client.delete('/dummy_prefix', recursive=True, dir=True)
        except EtcdKeyNotFound:
            pass
        self._backend = EtcdRendezvousBackend(self._client, 'dummy_run_id', '/dummy_prefix')

    def _corrupt_state(self) -> None:
        if False:
            print('Hello World!')
        self._client.write('/dummy_prefix/dummy_run_id', 'non_base64')

class CreateBackendTest(TestCase):
    _server: ClassVar[EtcdServer]

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        cls._server = EtcdServer()
        cls._server.start(stderr=subprocess.DEVNULL)

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            for i in range(10):
                print('nop')
        cls._server.stop()

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self._params = RendezvousParameters(backend='dummy_backend', endpoint=self._server.get_endpoint(), run_id='dummy_run_id', min_nodes=1, max_nodes=1, protocol='hTTp', read_timeout='10')
        self._expected_read_timeout = 10

    def test_create_backend_returns_backend(self) -> None:
        if False:
            i = 10
            return i + 15
        (backend, store) = create_backend(self._params)
        self.assertEqual(backend.name, 'etcd-v2')
        self.assertIsInstance(store, EtcdStore)
        etcd_store = cast(EtcdStore, store)
        self.assertEqual(etcd_store.client.read_timeout, self._expected_read_timeout)
        client = self._server.get_client()
        backend.set_state(b'dummy_state')
        result = client.get('/torch/elastic/rendezvous/' + self._params.run_id)
        self.assertEqual(result.value, b64encode(b'dummy_state').decode())
        self.assertLessEqual(result.ttl, 7200)
        store.set('dummy_key', 'dummy_value')
        result = client.get('/torch/elastic/store/' + b64encode(b'dummy_key').decode())
        self.assertEqual(result.value, b64encode(b'dummy_value').decode())

    def test_create_backend_returns_backend_if_protocol_is_not_specified(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self._params.config['protocol']
        self.test_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(self) -> None:
        if False:
            return 10
        del self._params.config['read_timeout']
        self._expected_read_timeout = 60
        self.test_create_backend_returns_backend()

    def test_create_backend_raises_error_if_etcd_is_unreachable(self) -> None:
        if False:
            while True:
                i = 10
        self._params.endpoint = 'dummy:1234'
        with self.assertRaisesRegex(RendezvousConnectionError, '^The connection to etcd has failed. See inner exception for details.$'):
            create_backend(self._params)

    def test_create_backend_raises_error_if_protocol_is_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._params.config['protocol'] = 'dummy'
        with self.assertRaisesRegex(ValueError, '^The protocol must be HTTP or HTTPS.$'):
            create_backend(self._params)

    def test_create_backend_raises_error_if_read_timeout_is_invalid(self) -> None:
        if False:
            while True:
                i = 10
        for read_timeout in ['0', '-10']:
            with self.subTest(read_timeout=read_timeout):
                self._params.config['read_timeout'] = read_timeout
                with self.assertRaisesRegex(ValueError, '^The read timeout must be a positive integer.$'):
                    create_backend(self._params)