from time import sleep
import zmq
from locust.rpc import zmqrpc, Message
from locust.test.testcases import LocustTestCase
from locust.exception import RPCError, RPCSendError, RPCReceiveError

class ZMQRPC_tests(LocustTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.server = zmqrpc.Server('127.0.0.1', 0)
        self.client = zmqrpc.Client('localhost', self.server.port, 'identity')

    def tearDown(self):
        if False:
            return 10
        self.server.close()
        self.client.close()
        super().tearDown()

    def test_constructor(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.server.socket.getsockopt(zmq.TCP_KEEPALIVE), 1)
        self.assertEqual(self.server.socket.getsockopt(zmq.TCP_KEEPALIVE_IDLE), 30)
        self.assertEqual(self.client.socket.getsockopt(zmq.TCP_KEEPALIVE), 1)
        self.assertEqual(self.client.socket.getsockopt(zmq.TCP_KEEPALIVE_IDLE), 30)

    def test_client_send(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.send(Message('test', 'message', 'identity'))
        (addr, msg) = self.server.recv_from_client()
        self.assertEqual(addr, 'identity')
        self.assertEqual(msg.type, 'test')
        self.assertEqual(msg.data, 'message')

    def test_client_recv(self):
        if False:
            while True:
                i = 10
        sleep(0.1)
        self.server.send_to_client(Message('test', 'message', 'identity'))
        msg = self.client.recv()
        self.assertEqual(msg.type, 'test')
        self.assertEqual(msg.data, 'message')
        self.assertEqual(msg.node_id, 'identity')

    def test_client_retry(self):
        if False:
            while True:
                i = 10
        server = zmqrpc.Server('127.0.0.1', 0)
        server.socket.close()
        with self.assertRaises(RPCError):
            server.recv_from_client()

    def test_rpc_error(self):
        if False:
            i = 10
            return i + 15
        server = zmqrpc.Server('127.0.0.1', 0)
        with self.assertRaises(RPCError):
            server = zmqrpc.Server('127.0.0.1', server.port)
        server.close()
        with self.assertRaises(RPCSendError):
            server.send_to_client(Message('test', 'message', 'identity'))