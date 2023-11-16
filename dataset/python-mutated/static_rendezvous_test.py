import unittest
from contextlib import closing
from torch.distributed.elastic.rendezvous import RendezvousParameters
from torch.distributed.elastic.rendezvous.static_tcp_rendezvous import create_rdzv_handler
from torch.distributed.elastic.utils import get_socket_with_port

class StaticTCPRendezvousTest(unittest.TestCase):

    def test_missing_port(self):
        if False:
            i = 10
            return i + 15
        rdzv_params = RendezvousParameters(backend='static', endpoint='localhost', run_id='test_id', min_nodes=1, max_nodes=1)
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_empty_endpoint(self):
        if False:
            return 10
        rdzv_params = RendezvousParameters(backend='static', endpoint='', run_id='test_id', min_nodes=1, max_nodes=1)
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_ipv6_addr(self):
        if False:
            while True:
                i = 10
        rdzv_params = RendezvousParameters(backend='static', endpoint='[2001:0db8:85a3:0000:0000:8a2e:0370:7334]:90', run_id='test_id', min_nodes=1, max_nodes=1)
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_ipv6_addr_localhost(self):
        if False:
            for i in range(10):
                print('nop')
        rdzv_params = RendezvousParameters(backend='static', endpoint='[::1]:90', run_id='test_id', min_nodes=1, max_nodes=1)
        with self.assertRaises(ValueError):
            create_rdzv_handler(rdzv_params)

    def test_get_backend(self):
        if False:
            print('Hello World!')
        rdzv_params = RendezvousParameters(backend='static', endpoint='localhost:123', run_id='test', min_nodes=1, max_nodes=1, timeout=60, rank=0)
        static_rdzv = create_rdzv_handler(rdzv_params)
        self.assertEqual('static', static_rdzv.get_backend())

    def test_static_rdzv_multiple_calls(self):
        if False:
            for i in range(10):
                print('nop')
        sock = get_socket_with_port()
        with closing(sock):
            master_port = sock.getsockname()[1]
        master_addr = 'localhost'
        rdzv_params = RendezvousParameters(backend='static', endpoint=f'{master_addr}:{master_port}', run_id='test_id', min_nodes=1, max_nodes=1, rank=0)
        rdzv_handler = create_rdzv_handler(rdzv_params)
        (store, rank, world_size) = rdzv_handler.next_rendezvous()
        self.assertIsNotNone(store)
        self.assertEqual(0, rank)
        self.assertEqual(1, world_size)
        (store, rank, world_size) = rdzv_handler.next_rendezvous()
        self.assertIsNotNone(store)
        self.assertEqual(0, rank)
        self.assertEqual(1, world_size)