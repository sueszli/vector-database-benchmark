import os
import tempfile
from base64 import b64encode
from datetime import timedelta
from typing import ClassVar, cast, Callable
from unittest import TestCase, mock
from torch.distributed import TCPStore, FileStore
from torch.distributed.elastic.rendezvous import RendezvousConnectionError, RendezvousParameters, RendezvousError
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import C10dRendezvousBackend, create_backend
from rendezvous_backend_test import RendezvousBackendTestMixin

class TCPStoreBackendTest(TestCase, RendezvousBackendTestMixin):
    _store: ClassVar[TCPStore]

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            return 10
        cls._store = TCPStore('localhost', 0, is_master=True)

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self._store.delete_key('torch.rendezvous.dummy_run_id')
        self._backend = C10dRendezvousBackend(self._store, 'dummy_run_id')

    def _corrupt_state(self) -> None:
        if False:
            return 10
        self._store.set('torch.rendezvous.dummy_run_id', 'non_base64')

class FileStoreBackendTest(TestCase, RendezvousBackendTestMixin):
    _store: ClassVar[FileStore]

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (_, path) = tempfile.mkstemp()
        self._path = path
        self._store = FileStore(path)
        self._backend = C10dRendezvousBackend(self._store, 'dummy_run_id')

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        os.remove(self._path)

    def _corrupt_state(self) -> None:
        if False:
            while True:
                i = 10
        self._store.set('torch.rendezvous.dummy_run_id', 'non_base64')

class CreateBackendTest(TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        self._params = RendezvousParameters(backend='dummy_backend', endpoint='localhost:29300', run_id='dummy_run_id', min_nodes=1, max_nodes=1, is_host='true', store_type='tCp', read_timeout='10')
        (_, tmp_path) = tempfile.mkstemp()
        self._params_filestore = RendezvousParameters(backend='dummy_backend', endpoint=tmp_path, run_id='dummy_run_id', min_nodes=1, max_nodes=1, store_type='fIlE')
        self._expected_endpoint_file = tmp_path
        self._expected_temp_dir = tempfile.gettempdir()
        self._expected_endpoint_host = 'localhost'
        self._expected_endpoint_port = 29300
        self._expected_store_type = TCPStore
        self._expected_read_timeout = timedelta(seconds=10)

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        os.remove(self._expected_endpoint_file)

    def _run_test_with_store(self, store_type: str, test_to_run: Callable):
        if False:
            for i in range(10):
                print('nop')
        '\n        Use this function to specify the store type to use in a test. If\n        not used, the test will default to TCPStore.\n        '
        if store_type == 'file':
            self._params = self._params_filestore
            self._expected_store_type = FileStore
            self._expected_read_timeout = timedelta(seconds=300)
        test_to_run()

    def _assert_create_backend_returns_backend(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        (backend, store) = create_backend(self._params)
        self.assertEqual(backend.name, 'c10d')
        self.assertIsInstance(store, self._expected_store_type)
        typecast_store = cast(self._expected_store_type, store)
        self.assertEqual(typecast_store.timeout, self._expected_read_timeout)
        if self._expected_store_type == TCPStore:
            self.assertEqual(typecast_store.host, self._expected_endpoint_host)
            self.assertEqual(typecast_store.port, self._expected_endpoint_port)
        if self._expected_store_type == FileStore:
            if self._params.endpoint:
                self.assertEqual(typecast_store.path, self._expected_endpoint_file)
            else:
                self.assertTrue(typecast_store.path.startswith(self._expected_temp_dir))
        backend.set_state(b'dummy_state')
        state = store.get('torch.rendezvous.' + self._params.run_id)
        self.assertEqual(state, b64encode(b'dummy_state'))

    def test_create_backend_returns_backend(self) -> None:
        if False:
            i = 10
            return i + 15
        for store_type in ['tcp', 'file']:
            with self.subTest(store_type=store_type):
                self._run_test_with_store(store_type, self._assert_create_backend_returns_backend)

    def test_create_backend_returns_backend_if_is_host_is_false(self) -> None:
        if False:
            print('Hello World!')
        store = TCPStore(self._expected_endpoint_host, self._expected_endpoint_port, is_master=True)
        self._params.config['is_host'] = 'false'
        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self._params.config['is_host']
        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_is_host_is_not_specified_and_store_already_exists(self) -> None:
        if False:
            print('Hello World!')
        store = TCPStore(self._expected_endpoint_host, self._expected_endpoint_port, is_master=True)
        del self._params.config['is_host']
        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_endpoint_port_is_not_specified(self) -> None:
        if False:
            return 10
        self._params.endpoint = self._expected_endpoint_host
        self._expected_endpoint_port = 29400
        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_endpoint_file_is_not_specified(self) -> None:
        if False:
            i = 10
            return i + 15
        self._params_filestore.endpoint = ''
        self._run_test_with_store('file', self._assert_create_backend_returns_backend)

    def test_create_backend_returns_backend_if_store_type_is_not_specified(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        del self._params.config['store_type']
        self._expected_store_type = TCPStore
        if not self._params.get('read_timeout'):
            self._expected_read_timeout = timedelta(seconds=60)
        self._assert_create_backend_returns_backend()

    def test_create_backend_returns_backend_if_read_timeout_is_not_specified(self) -> None:
        if False:
            i = 10
            return i + 15
        del self._params.config['read_timeout']
        self._expected_read_timeout = timedelta(seconds=60)
        self._assert_create_backend_returns_backend()

    def test_create_backend_raises_error_if_store_is_unreachable(self) -> None:
        if False:
            i = 10
            return i + 15
        self._params.config['is_host'] = 'false'
        self._params.config['read_timeout'] = '2'
        with self.assertRaisesRegex(RendezvousConnectionError, '^The connection to the C10d store has failed. See inner exception for details.$'):
            create_backend(self._params)

    def test_create_backend_raises_error_if_endpoint_is_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        for is_host in [True, False]:
            with self.subTest(is_host=is_host):
                self._params.config['is_host'] = str(is_host)
                self._params.endpoint = 'dummy_endpoint'
                with self.assertRaisesRegex(RendezvousConnectionError, '^The connection to the C10d store has failed. See inner exception for details.$'):
                    create_backend(self._params)

    def test_create_backend_raises_error_if_store_type_is_invalid(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._params.config['store_type'] = 'dummy_store_type'
        with self.assertRaisesRegex(ValueError, '^Invalid store type given. Currently only supports file and tcp.$'):
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

    @mock.patch('tempfile.mkstemp')
    def test_create_backend_raises_error_if_tempfile_creation_fails(self, tempfile_mock) -> None:
        if False:
            i = 10
            return i + 15
        tempfile_mock.side_effect = OSError('test error')
        self._params_filestore.endpoint = ''
        with self.assertRaisesRegex(RendezvousError, 'The file creation for C10d store has failed. See inner exception for details.'):
            create_backend(self._params_filestore)

    @mock.patch('torch.distributed.elastic.rendezvous.c10d_rendezvous_backend.FileStore')
    def test_create_backend_raises_error_if_file_path_is_invalid(self, filestore_mock) -> None:
        if False:
            print('Hello World!')
        filestore_mock.side_effect = RuntimeError('test error')
        self._params_filestore.endpoint = 'bad file path'
        with self.assertRaisesRegex(RendezvousConnectionError, '^The connection to the C10d store has failed. See inner exception for details.$'):
            create_backend(self._params_filestore)