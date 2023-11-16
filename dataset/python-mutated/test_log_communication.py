from __future__ import absolute_import, division, print_function
import contextlib
from io import StringIO
import sys
import time
from typing import Any, Callable
import unittest
import pyspark.ml.torch.log_communication
from pyspark.ml.torch.log_communication import LogStreamingServer, LogStreamingClient, LogStreamingClientBase, _SERVER_POLL_INTERVAL

@contextlib.contextmanager
def patch_stderr() -> StringIO:
    if False:
        print('Hello World!')
    'patch stdout and give an output'
    sys_stderr = sys.stderr
    io_out = StringIO()
    sys.stderr = io_out
    try:
        yield io_out
    finally:
        sys.stderr = sys_stderr

class LogStreamingServiceTestCase(unittest.TestCase):

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.default_truncate_msg_len = pyspark.ml.torch.log_communication._TRUNCATE_MSG_LEN
        pyspark.ml.torch.log_communication._TRUNCATE_MSG_LEN = 10

    def tearDown(self) -> None:
        if False:
            print('Hello World!')
        pyspark.ml.torch.log_communication._TRUNCATE_MSG_LEN = self.default_truncate_msg_len

    def basic_test(self) -> None:
        if False:
            i = 10
            return i + 15
        server = LogStreamingServer()
        server.start()
        time.sleep(1)
        client = LogStreamingClient('localhost', server.port)
        with patch_stderr() as output:
            client.send('msg 001')
            client.send('msg 002')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
            output = output.getvalue()
            self.assertIn('msg 001\nmsg 002\n', output)
        client.close()
        server.shutdown()

    def test_truncate_message(self) -> None:
        if False:
            while True:
                i = 10
        msg1 = 'abc'
        assert LogStreamingClientBase._maybe_truncate_msg(msg1) == msg1
        msg2 = 'abcdefghijkl'
        assert LogStreamingClientBase._maybe_truncate_msg(msg2) == 'abcdefghij...(truncated)'

    def test_multiple_clients(self) -> None:
        if False:
            while True:
                i = 10
        server = LogStreamingServer()
        server.start()
        time.sleep(1)
        client1 = LogStreamingClient('localhost', server.port)
        client2 = LogStreamingClient('localhost', server.port)
        with patch_stderr() as output:
            client1.send('c1 msg1')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
            client2.send('c2 msg1')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
            client1.send('c1 msg2')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
            client2.send('c2 msg2')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
            output = output.getvalue()
            self.assertIn('c1 msg1\nc2 msg1\nc1 msg2\nc2 msg2\n', output)
        client1.close()
        client2.close()
        server.shutdown()

    def test_client_should_fail_gracefully(self) -> None:
        if False:
            return 10
        server = LogStreamingServer()
        server.start()
        time.sleep(1)
        client = LogStreamingClient('localhost', server.port)
        client.send('msg 001')
        server.shutdown()
        for i in range(5):
            client.send('msg 002')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
        self.assertTrue(client.failed)
        client.close()

    def test_client_send_intermittently(self) -> None:
        if False:
            print('Hello World!')
        server = LogStreamingServer()
        server.start()
        time.sleep(1)
        client = LogStreamingClient('localhost', server.port)
        with patch_stderr() as output:
            client._connect()
            client.send('msg part1')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
            client.send(' msg part2')
            time.sleep(_SERVER_POLL_INTERVAL + 1)
            output = output.getvalue()
            self.assertIn('msg part1\n msg part2\n', output)
        client.close()
        server.shutdown()

    @staticmethod
    def test_server_shutdown() -> None:
        if False:
            while True:
                i = 10

        def run_test(client_ops: Callable) -> None:
            if False:
                while True:
                    i = 10
            server = LogStreamingServer()
            server.start()
            time.sleep(1)
            client = LogStreamingClient('localhost', server.port)
            client_ops(client)
            server.shutdown()
            client.close()

        def client_ops_close(client: Any) -> None:
            if False:
                for i in range(10):
                    print('nop')
            client.close()

        def client_ops_send_half_msg(client: Any) -> None:
            if False:
                return 10
            client._connect()
            client.sock.sendall(b'msg part1 ')
            time.sleep(_SERVER_POLL_INTERVAL + 1)

        def client_ops_send_a_msg(client: Any) -> None:
            if False:
                while True:
                    i = 10
            client.send('msg1')
            time.sleep(_SERVER_POLL_INTERVAL + 1)

        def client_ops_send_a_msg_and_close(client: Any) -> None:
            if False:
                print('Hello World!')
            client.send('msg1')
            client.close()
            time.sleep(_SERVER_POLL_INTERVAL + 1)
        run_test(client_ops_close)
        run_test(client_ops_send_half_msg)
        run_test(client_ops_send_a_msg)
        run_test(client_ops_send_a_msg_and_close)
if __name__ == '__main__':
    from pyspark.ml.torch.tests.test_log_communication import *
    try:
        import xmlrunner
        testRunner = xmlrunner.XMLTestRunner(output='target/test-reports', verbosity=2)
    except ImportError:
        testRunner = None
    unittest.main(testRunner=testRunner, verbosity=2)