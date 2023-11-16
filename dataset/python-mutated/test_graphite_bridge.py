import socketserver as SocketServer
import threading
import unittest
from prometheus_client import CollectorRegistry, Gauge
from prometheus_client.bridge.graphite import GraphiteBridge

def fake_timer():
    if False:
        for i in range(10):
            print('nop')
    return 1434898897.5

class TestGraphiteBridge(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.registry = CollectorRegistry()
        self.data = ''

        class TCPHandler(SocketServer.BaseRequestHandler):

            def handle(s):
                if False:
                    i = 10
                    return i + 15
                self.data = s.request.recv(1024)
        server = SocketServer.TCPServer(('', 0), TCPHandler)

        class ServingThread(threading.Thread):

            def run(self):
                if False:
                    return 10
                server.handle_request()
                server.socket.close()
        self.t = ServingThread()
        self.t.start()
        self.address = ('localhost', server.server_address[1])
        self.gb = GraphiteBridge(self.address, self.registry, _timer=fake_timer)

    def _use_tags(self):
        if False:
            i = 10
            return i + 15
        self.gb = GraphiteBridge(self.address, self.registry, tags=True, _timer=fake_timer)

    def test_nolabels(self):
        if False:
            for i in range(10):
                print('nop')
        gauge = Gauge('g', 'help', registry=self.registry)
        gauge.inc()
        self.gb.push()
        self.t.join()
        self.assertEqual(b'g 1.0 1434898897\n', self.data)

    def test_labels(self):
        if False:
            while True:
                i = 10
        labels = Gauge('labels', 'help', ['a', 'b'], registry=self.registry)
        labels.labels('c', 'd').inc()
        self.gb.push()
        self.t.join()
        self.assertEqual(b'labels.a.c.b.d 1.0 1434898897\n', self.data)

    def test_labels_tags(self):
        if False:
            for i in range(10):
                print('nop')
        self._use_tags()
        labels = Gauge('labels', 'help', ['a', 'b'], registry=self.registry)
        labels.labels('c', 'd').inc()
        self.gb.push()
        self.t.join()
        self.assertEqual(b'labels;a=c;b=d 1.0 1434898897\n', self.data)

    def test_prefix(self):
        if False:
            i = 10
            return i + 15
        labels = Gauge('labels', 'help', ['a', 'b'], registry=self.registry)
        labels.labels('c', 'd').inc()
        self.gb.push(prefix='pre.fix')
        self.t.join()
        self.assertEqual(b'pre.fix.labels.a.c.b.d 1.0 1434898897\n', self.data)

    def test_prefix_tags(self):
        if False:
            print('Hello World!')
        self._use_tags()
        labels = Gauge('labels', 'help', ['a', 'b'], registry=self.registry)
        labels.labels('c', 'd').inc()
        self.gb.push(prefix='pre.fix')
        self.t.join()
        self.assertEqual(b'pre.fix.labels;a=c;b=d 1.0 1434898897\n', self.data)

    def test_sanitizing(self):
        if False:
            i = 10
            return i + 15
        labels = Gauge('labels', 'help', ['a'], registry=self.registry)
        labels.labels('c.:8').inc()
        self.gb.push()
        self.t.join()
        self.assertEqual(b'labels.a.c__8 1.0 1434898897\n', self.data)

    def test_sanitizing_tags(self):
        if False:
            i = 10
            return i + 15
        self._use_tags()
        labels = Gauge('labels', 'help', ['a'], registry=self.registry)
        labels.labels('c.:8').inc()
        self.gb.push()
        self.t.join()
        self.assertEqual(b'labels;a=c__8 1.0 1434898897\n', self.data)