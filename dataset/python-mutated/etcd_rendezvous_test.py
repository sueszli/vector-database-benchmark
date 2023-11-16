import os
import unittest
import uuid
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.etcd_rendezvous import create_rdzv_handler
from torch.distributed.elastic.rendezvous.etcd_server import EtcdServer
if os.getenv('CIRCLECI'):
    print('T85992919 temporarily disabling in circle ci', file=sys.stderr)
    sys.exit(0)

class EtcdRendezvousTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls._etcd_server = EtcdServer()
        cls._etcd_server.start()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls._etcd_server.stop()

    def test_etcd_rdzv_basic_params(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that we can create the handler with a minimum set of\n        params\n        '
        rdzv_params = RendezvousParameters(backend='etcd', endpoint=f'{self._etcd_server.get_endpoint()}', run_id=f'{uuid.uuid4()}', min_nodes=1, max_nodes=1)
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        self.assertIsNotNone(etcd_rdzv)

    def test_etcd_rdzv_additional_params(self):
        if False:
            i = 10
            return i + 15
        run_id = str(uuid.uuid4())
        rdzv_params = RendezvousParameters(backend='etcd', endpoint=f'{self._etcd_server.get_endpoint()}', run_id=run_id, min_nodes=1, max_nodes=1, timeout=60, last_call_timeout=30, protocol='http')
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        self.assertIsNotNone(etcd_rdzv)
        self.assertEqual(run_id, etcd_rdzv.get_run_id())

    def test_get_backend(self):
        if False:
            print('Hello World!')
        run_id = str(uuid.uuid4())
        rdzv_params = RendezvousParameters(backend='etcd', endpoint=f'{self._etcd_server.get_endpoint()}', run_id=run_id, min_nodes=1, max_nodes=1, timeout=60, last_call_timeout=30, protocol='http')
        etcd_rdzv = create_rdzv_handler(rdzv_params)
        self.assertEqual('etcd', etcd_rdzv.get_backend())