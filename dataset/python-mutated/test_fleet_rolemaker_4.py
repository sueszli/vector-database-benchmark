"""Test cloud role maker."""
import os
import unittest

class TestCloudRoleMaker(unittest.TestCase):
    """
    Test cases for PaddleCloudRoleMaker.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        'Set up, set envs.'
        os.environ['PADDLE_TRAINERS_NUM'] = '2'
        os.environ['PADDLE_PSERVERS_IP_PORT_LIST'] = '127.0.0.1:36001,127.0.0.2:36001'

    def test_pslib_1(self):
        if False:
            while True:
                i = 10
        'Test cases for pslib.'
        import threading
        try:
            from paddle.distributed.fleet.utils.http_server import KVHandler, KVHTTPServer, KVServer
        except:
            print('warning: no fleet, skip test_pslib_4')
            return

        class FakeStream:
            """
            it is a fake stream only for test.
            """

            def write(self, a):
                if False:
                    print('Hello World!')
                '\n                write a to stream, do nothing\n\n                Args:\n                    a(str): the string to write\n                '
                pass

            def read(self, b):
                if False:
                    i = 10
                    return i + 15
                '\n                read data of len b from stream, do nothing\n\n                Args:\n                    b(str): the len to read\n\n                Returns:\n                    c(str): the result\n                '
                if b == 0:
                    raise ValueError('this is only for test')
                return 'fake'
        try:

            class TmpKVHander(KVHandler):
                """
                it is a fake handler only for this test case.
                """

                def __init__(self, server):
                    if False:
                        while True:
                            i = 10
                    'Init.'
                    self.path = 'a/b/c'
                    self.server = server
                    self.wfile = FakeStream()
                    self.rfile = FakeStream()
                    self.headers = {}
                    self.headers['Content-Length'] = 0

                def address_string(self):
                    if False:
                        i = 10
                        return i + 15
                    '\n                    fake address string, it will do nothing.\n                    '
                    return '123'

                def send_response(self, code):
                    if False:
                        for i in range(10):
                            print('nop')
                    '\n                    fake send response, it will do nothing.\n\n                    Args:\n                        code(int): error code\n                    '
                    pass

                def send_header(self, a, b):
                    if False:
                        for i in range(10):
                            print('nop')
                    '\n                    fake send header, it will do nothing.\n\n                    Args:\n                        a(str): some header\n                        b(str): some header\n                    '
                    pass

                def end_headers(self):
                    if False:
                        return 10
                    '\n                    fake end header, it will do nothing.\n                    '
                    pass
        except:
            print('warning: no KVHandler, skip test_pslib_4')
            return
        try:

            class TmpServer(KVHTTPServer):
                """
                it is a fake server only for this test case.
                """

                def __init__(self):
                    if False:
                        return 10
                    'Init.'
                    self.delete_kv_lock = threading.Lock()
                    self.delete_kv = {}
                    self.kv_lock = threading.Lock()
                    self.kv = {}
        except:
            print('warning: no KVHTTPServer, skip test_pslib_4')
            return
        try:

            class TmpS(KVServer):
                """
                it is a fake server only for this test case.
                """

                def __init__(self):
                    if False:
                        for i in range(10):
                            print('nop')
                    'Init.'
                    self.http_server = TmpServer()
                    self.listen_thread = None
                    self.size = {}
                    self.size['a'] = 999
        except:
            print('warning: no KVServer, skip test_pslib_4')
            return
        s = TmpServer()
        h = TmpKVHander(s)
        h.do_GET()
        h.path = 'a/b'
        h.do_GET()
        h.do_PUT()
        h.do_DELETE()
        h.path = 'a/b/c'
        s.kv['b'] = {}
        s.kv['b']['c'] = '456'
        h.do_GET()
        h.path = 'a/d/e'
        h.do_PUT()
        h.headers['Content-Length'] = 1
        h.do_PUT()
        h.do_DELETE()
        h.log_message('666')
        s.get_deleted_size('haha')
        s1 = TmpS()
        s1.should_stop()
if __name__ == '__main__':
    unittest.main()