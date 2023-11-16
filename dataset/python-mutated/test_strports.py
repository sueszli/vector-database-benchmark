"""
Tests for L{twisted.application.strports}.
"""
from twisted.application import internet, strports
from twisted.internet.endpoints import TCP4ServerEndpoint
from twisted.internet.protocol import Factory
from twisted.trial.unittest import TestCase

class ServiceTests(TestCase):
    """
    Tests for L{strports.service}.
    """

    def test_service(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{strports.service} returns a L{StreamServerEndpointService}\n        constructed with an endpoint produced from\n        L{endpoint.serverFromString}, using the same syntax.\n        '
        reactor = object()
        aFactory = Factory()
        aGoodPort = 1337
        svc = strports.service('tcp:' + str(aGoodPort), aFactory, reactor=reactor)
        self.assertIsInstance(svc, internet.StreamServerEndpointService)
        self.assertTrue(svc._raiseSynchronously)
        self.assertIsInstance(svc.endpoint, TCP4ServerEndpoint)
        self.assertEqual(svc.endpoint._port, aGoodPort)
        self.assertIs(svc.factory, aFactory)
        self.assertIs(svc.endpoint._reactor, reactor)

    def test_serviceDefaultReactor(self):
        if False:
            while True:
                i = 10
        '\n        L{strports.service} will use the default reactor when none is provided\n        as an argument.\n        '
        from twisted.internet import reactor as globalReactor
        aService = strports.service('tcp:80', None)
        self.assertIs(aService.endpoint._reactor, globalReactor)