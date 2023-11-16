import sys
import gevent
from gevent import socket
from gevent import testing as greentest
from gevent.testing import TestCase, tcp_listener
from gevent.testing import gc_collect_if_needed
from gevent.testing import skipOnPyPy
from gevent.testing import params
PY3 = sys.version_info[0] >= 3

def _write_to_closed(f, s):
    if False:
        print('Hello World!')
    try:
        r = f.write(s)
    except ValueError:
        assert PY3
    else:
        assert r is None, r

class TestGreenIo(TestCase):

    def test_close_with_makefile(self):
        if False:
            return 10

        def accept_close_early(listener):
            if False:
                for i in range(10):
                    print('nop')
            try:
                (conn, _) = listener.accept()
                fd = conn.makefile(mode='wb')
                conn.close()
                fd.write(b'hello\n')
                fd.close()
                _write_to_closed(fd, b'a')
                self.assertRaises(socket.error, conn.send, b'b')
            finally:
                listener.close()

        def accept_close_late(listener):
            if False:
                print('Hello World!')
            try:
                (conn, _) = listener.accept()
                fd = conn.makefile(mode='wb')
                fd.write(b'hello')
                fd.close()
                conn.send(b'\n')
                conn.close()
                _write_to_closed(fd, b'a')
                self.assertRaises(socket.error, conn.send, b'b')
            finally:
                listener.close()

        def did_it_work(server):
            if False:
                for i in range(10):
                    print('nop')
            client = socket.create_connection((params.DEFAULT_CONNECT, server.getsockname()[1]))
            fd = client.makefile(mode='rb')
            client.close()
            self.assertEqual(fd.readline(), b'hello\n')
            self.assertFalse(fd.read())
            fd.close()
        server = tcp_listener()
        server_greenlet = gevent.spawn(accept_close_early, server)
        did_it_work(server)
        server_greenlet.kill()
        server = tcp_listener()
        server_greenlet = gevent.spawn(accept_close_late, server)
        did_it_work(server)
        server_greenlet.kill()

    @skipOnPyPy("Takes multiple GCs and issues a warning we can't catch")
    def test_del_closes_socket(self):
        if False:
            print('Hello World!')
        import warnings

        def accept_once(listener):
            if False:
                for i in range(10):
                    print('nop')
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    conn = listener.accept()[0]
                    conn = conn.makefile(mode='wb')
                    conn.write(b'hello\n')
                    conn.close()
                    _write_to_closed(conn, b'a')
                finally:
                    listener.close()
                    del listener
                    del conn
                    gc_collect_if_needed()
                    gc_collect_if_needed()
        server = tcp_listener()
        gevent.spawn(accept_once, server)
        client = socket.create_connection((params.DEFAULT_CONNECT, server.getsockname()[1]))
        with gevent.Timeout.start_new(0.5):
            fd = client.makefile()
            client.close()
            self.assertEqual(fd.read(), 'hello\n')
            self.assertEqual(fd.read(), '')
            fd.close()
        del client
        del fd
if __name__ == '__main__':
    greentest.main()