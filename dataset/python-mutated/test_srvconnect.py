"""
Test cases for L{twisted.names.srvconnect}.
"""
import random
from zope.interface.verify import verifyObject
from twisted.internet import defer, protocol
from twisted.internet.error import DNSLookupError, ServiceNameUnknownError
from twisted.internet.interfaces import IConnector
from twisted.internet.testing import MemoryReactor
from twisted.names import client, dns, srvconnect
from twisted.names.common import ResolverBase
from twisted.names.error import DNSNameError
from twisted.trial import unittest

class FakeResolver(ResolverBase):
    """
    Resolver that only gives out one given result.

    Either L{results} or L{failure} must be set and will be used for
    the return value of L{_lookup}

    @ivar results: List of L{dns.RRHeader} for the desired result.
    @type results: C{list}
    @ivar failure: Failure with an exception from L{twisted.names.error}.
    @type failure: L{Failure<twisted.python.failure.Failure>}
    """

    def __init__(self, results=None, failure=None):
        if False:
            while True:
                i = 10
        self.results = results
        self.failure = failure
        self.lookups = []

    def _lookup(self, name, cls, qtype, timeout):
        if False:
            print('Hello World!')
        '\n        Return the result or failure on lookup.\n        '
        self.lookups.append((name, cls, qtype, timeout))
        if self.results is not None:
            return defer.succeed((self.results, [], []))
        else:
            return defer.fail(self.failure)

class DummyFactory(protocol.ClientFactory):
    """
    Dummy client factory that stores the reason of connection failure.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self.reason = None

    def clientConnectionFailed(self, connector, reason):
        if False:
            return 10
        self.reason = reason

class SRVConnectorTests(unittest.TestCase):
    """
    Tests for L{srvconnect.SRVConnector}.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        self.patch(client, 'theResolver', FakeResolver())
        self.reactor = MemoryReactor()
        self.factory = DummyFactory()
        self.connector = srvconnect.SRVConnector(self.reactor, 'xmpp-server', 'example.org', self.factory)
        self.randIntArgs = []
        self.randIntResults = []

    def _randint(self, min, max):
        if False:
            i = 10
            return i + 15
        '\n        Fake randint.\n\n        Returns the first element of L{randIntResults} and records the\n        arguments passed to it in L{randIntArgs}.\n\n        @param min: Lower bound of the random number.\n        @type min: L{int}\n\n        @param max: Higher bound of the random number.\n        @type max: L{int}\n\n        @return: Fake random number from L{randIntResults}.\n        @rtype: L{int}\n        '
        self.randIntArgs.append((min, max))
        return self.randIntResults.pop(0)

    def test_interface(self):
        if False:
            return 10
        '\n        L{srvconnect.SRVConnector} implements L{IConnector}.\n        '
        verifyObject(IConnector, self.connector)

    def test_SRVPresent(self):
        if False:
            print('Hello World!')
        '\n        Test connectTCP gets called with the address from the SRV record.\n        '
        payload = dns.Record_SRV(port=6269, target='host.example.org', ttl=60)
        client.theResolver.results = [dns.RRHeader(name='example.org', type=dns.SRV, cls=dns.IN, ttl=60, payload=payload)]
        self.connector.connect()
        self.assertIsNone(self.factory.reason)
        self.assertEqual(self.reactor.tcpClients.pop()[:2], ('host.example.org', 6269))

    def test_SRVNotPresent(self):
        if False:
            i = 10
            return i + 15
        '\n        Test connectTCP gets called with fallback parameters on NXDOMAIN.\n        '
        client.theResolver.failure = DNSNameError(b'example.org')
        self.connector.connect()
        self.assertIsNone(self.factory.reason)
        self.assertEqual(self.reactor.tcpClients.pop()[:2], ('example.org', 'xmpp-server'))

    def test_SRVNoResult(self):
        if False:
            print('Hello World!')
        '\n        Test connectTCP gets called with fallback parameters on empty result.\n        '
        client.theResolver.results = []
        self.connector.connect()
        self.assertIsNone(self.factory.reason)
        self.assertEqual(self.reactor.tcpClients.pop()[:2], ('example.org', 'xmpp-server'))

    def test_SRVNoResultUnknownServiceDefaultPort(self):
        if False:
            print('Hello World!')
        '\n        connectTCP gets called with default port if the service is not defined.\n        '
        self.connector = srvconnect.SRVConnector(self.reactor, 'thisbetternotexist', 'example.org', self.factory, defaultPort=5222)
        client.theResolver.failure = ServiceNameUnknownError()
        self.connector.connect()
        self.assertIsNone(self.factory.reason)
        self.assertEqual(self.reactor.tcpClients.pop()[:2], ('example.org', 5222))

    def test_SRVNoResultUnknownServiceNoDefaultPort(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Connect fails on no result, unknown service and no default port.\n        '
        self.connector = srvconnect.SRVConnector(self.reactor, 'thisbetternotexist', 'example.org', self.factory)
        client.theResolver.failure = ServiceNameUnknownError()
        self.connector.connect()
        self.assertTrue(self.factory.reason.check(ServiceNameUnknownError))

    def test_SRVBadResult(self):
        if False:
            i = 10
            return i + 15
        '\n        Test connectTCP gets called with fallback parameters on bad result.\n        '
        client.theResolver.results = [dns.RRHeader(name='example.org', type=dns.CNAME, cls=dns.IN, ttl=60, payload=None)]
        self.connector.connect()
        self.assertIsNone(self.factory.reason)
        self.assertEqual(self.reactor.tcpClients.pop()[:2], ('example.org', 'xmpp-server'))

    def test_SRVNoService(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that connecting fails when no service is present.\n        '
        payload = dns.Record_SRV(port=5269, target=b'.', ttl=60)
        client.theResolver.results = [dns.RRHeader(name='example.org', type=dns.SRV, cls=dns.IN, ttl=60, payload=payload)]
        self.connector.connect()
        self.assertIsNotNone(self.factory.reason)
        self.factory.reason.trap(DNSLookupError)
        self.assertEqual(self.reactor.tcpClients, [])

    def test_SRVLookupName(self):
        if False:
            while True:
                i = 10
        '\n        The lookup name is a native string from service, protocol and domain.\n        '
        client.theResolver.results = []
        self.connector.connect()
        name = client.theResolver.lookups[-1][0]
        self.assertEqual(b'_xmpp-server._tcp.example.org', name)

    def test_unicodeDomain(self):
        if False:
            return 10
        '\n        L{srvconnect.SRVConnector} automatically encodes unicode domain using\n        C{idna} encoding.\n        '
        self.connector = srvconnect.SRVConnector(self.reactor, 'xmpp-client', 'échec.example.org', self.factory)
        self.assertEqual(b'xn--chec-9oa.example.org', self.connector.domain)

    def test_pickServerWeights(self):
        if False:
            while True:
                i = 10
        '\n        pickServer calculates running sum of weights and calls randint.\n\n        This exercises the server selection algorithm specified in RFC 2782 by\n        preparing fake L{random.randint} results and checking the values it was\n        called with.\n        '
        record1 = dns.Record_SRV(10, 10, 5222, 'host1.example.org')
        record2 = dns.Record_SRV(10, 20, 5222, 'host2.example.org')
        self.connector.orderedServers = [record1, record2]
        self.connector.servers = []
        self.patch(random, 'randint', self._randint)
        self.randIntResults = [11, 0]
        self.connector.pickServer()
        self.assertEqual(self.randIntArgs[0], (0, 30))
        self.connector.pickServer()
        self.assertEqual(self.randIntArgs[1], (0, 10))
        self.randIntResults = [10, 0]
        self.connector.pickServer()
        self.assertEqual(self.randIntArgs[2], (0, 30))
        self.connector.pickServer()
        self.assertEqual(self.randIntArgs[3], (0, 20))

    def test_pickServerSamePriorities(self):
        if False:
            while True:
                i = 10
        '\n        Two records with equal priorities compare on weight (ascending).\n        '
        record1 = dns.Record_SRV(10, 10, 5222, 'host1.example.org')
        record2 = dns.Record_SRV(10, 20, 5222, 'host2.example.org')
        self.connector.orderedServers = [record2, record1]
        self.connector.servers = []
        self.patch(random, 'randint', self._randint)
        self.randIntResults = [0, 0]
        self.assertEqual(('host1.example.org', 5222), self.connector.pickServer())
        self.assertEqual(('host2.example.org', 5222), self.connector.pickServer())

    def test_srvDifferentPriorities(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Two records with differing priorities compare on priority (ascending).\n        '
        record1 = dns.Record_SRV(10, 0, 5222, 'host1.example.org')
        record2 = dns.Record_SRV(20, 0, 5222, 'host2.example.org')
        self.connector.orderedServers = [record2, record1]
        self.connector.servers = []
        self.patch(random, 'randint', self._randint)
        self.randIntResults = [0, 0]
        self.assertEqual(('host1.example.org', 5222), self.connector.pickServer())
        self.assertEqual(('host2.example.org', 5222), self.connector.pickServer())