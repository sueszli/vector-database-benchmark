from calculus.client_2 import RemoteCalculationClient
from calculus.remote_1 import RemoteCalculationFactory
from twisted.internet import protocol, reactor
from twisted.trial import unittest

class RemoteRunCalculationTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        factory = RemoteCalculationFactory()
        self.port = reactor.listenTCP(0, factory, interface='127.0.0.1')
        self.client = None

    def tearDown(self):
        if False:
            print('Hello World!')
        if self.client is not None:
            self.client.transport.loseConnection()
        return self.port.stopListening()

    def _test(self, op, a, b, expected):
        if False:
            while True:
                i = 10
        creator = protocol.ClientCreator(reactor, RemoteCalculationClient)

        def cb(client):
            if False:
                i = 10
                return i + 15
            self.client = client
            return getattr(self.client, op)(a, b).addCallback(self.assertEqual, expected)
        return creator.connectTCP('127.0.0.1', self.port.getHost().port).addCallback(cb)

    def test_add(self):
        if False:
            return 10
        return self._test('add', 5, 9, 14)

    def test_subtract(self):
        if False:
            print('Hello World!')
        return self._test('subtract', 47, 13, 34)

    def test_multiply(self):
        if False:
            for i in range(10):
                print('nop')
        return self._test('multiply', 7, 3, 21)

    def test_divide(self):
        if False:
            i = 10
            return i + 15
        return self._test('divide', 84, 10, 8)