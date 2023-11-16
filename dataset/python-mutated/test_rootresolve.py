"""
Test cases for Twisted.names' root resolver.
"""
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.internet.defer import Deferred, TimeoutError, gatherResults, succeed
from twisted.internet.interfaces import IResolverSimple
from twisted.names import client, root
from twisted.names.dns import CNAME, ENAME, HS, IN, NS, OK, A, Message, Name, Query, Record_A, Record_CNAME, Record_NS, RRHeader
from twisted.names.error import DNSNameError, ResolverError
from twisted.names.root import Resolver
from twisted.names.test.test_util import MemoryReactor
from twisted.python.log import msg
from twisted.trial import util
from twisted.trial.unittest import SynchronousTestCase, TestCase

def getOnePayload(results):
    if False:
        i = 10
        return i + 15
    '\n    From the result of a L{Deferred} returned by L{IResolver.lookupAddress},\n    return the payload of the first record in the answer section.\n    '
    (ans, auth, add) = results
    return ans[0].payload

def getOneAddress(results):
    if False:
        i = 10
        return i + 15
    '\n    From the result of a L{Deferred} returned by L{IResolver.lookupAddress},\n    return the first IPv4 address from the answer section.\n    '
    return getOnePayload(results).dottedQuad()

class RootResolverTests(TestCase):
    """
    Tests for L{twisted.names.root.Resolver}.
    """

    def _queryTest(self, filter):
        if False:
            while True:
                i = 10
        '\n        Invoke L{Resolver._query} and verify that it sends the correct DNS\n        query.  Deliver a canned response to the query and return whatever the\n        L{Deferred} returned by L{Resolver._query} fires with.\n\n        @param filter: The value to pass for the C{filter} parameter to\n            L{Resolver._query}.\n        '
        reactor = MemoryReactor()
        resolver = Resolver([], reactor=reactor)
        d = resolver._query(Query(b'foo.example.com', A, IN), [('1.1.2.3', 1053)], (30,), filter)
        (portNumber, transport) = reactor.udpPorts.popitem()
        [(packet, address)] = transport._sentPackets
        message = Message()
        message.fromStr(packet)
        self.assertEqual(message.queries, [Query(b'foo.example.com', A, IN)])
        self.assertEqual(message.answers, [])
        self.assertEqual(message.authority, [])
        self.assertEqual(message.additional, [])
        response = []
        d.addCallback(response.append)
        self.assertEqual(response, [])
        del message.queries[:]
        message.answer = 1
        message.answers.append(RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21')))
        transport._protocol.datagramReceived(message.toStr(), ('1.1.2.3', 1053))
        return response[0]

    def test_filteredQuery(self):
        if False:
            i = 10
            return i + 15
        '\n        L{Resolver._query} accepts a L{Query} instance and an address, issues\n        the query, and returns a L{Deferred} which fires with the response to\n        the query.  If a true value is passed for the C{filter} parameter, the\n        result is a three-tuple of lists of records.\n        '
        (answer, authority, additional) = self._queryTest(True)
        self.assertEqual(answer, [RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21', ttl=0))])
        self.assertEqual(authority, [])
        self.assertEqual(additional, [])

    def test_unfilteredQuery(self):
        if False:
            i = 10
            return i + 15
        '\n        Similar to L{test_filteredQuery}, but for the case where a false value\n        is passed for the C{filter} parameter.  In this case, the result is a\n        L{Message} instance.\n        '
        message = self._queryTest(False)
        self.assertIsInstance(message, Message)
        self.assertEqual(message.queries, [])
        self.assertEqual(message.answers, [RRHeader(b'foo.example.com', payload=Record_A('5.8.13.21', ttl=0))])
        self.assertEqual(message.authority, [])
        self.assertEqual(message.additional, [])

    def _respond(self, answers=[], authority=[], additional=[], rCode=OK):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a L{Message} suitable for use as a response to a query.\n\n        @param answers: A C{list} of two-tuples giving data for the answers\n            section of the message.  The first element of each tuple is a name\n            for the L{RRHeader}.  The second element is the payload.\n        @param authority: A C{list} like C{answers}, but for the authority\n            section of the response.\n        @param additional: A C{list} like C{answers}, but for the\n            additional section of the response.\n        @param rCode: The response code the message will be created with.\n\n        @return: A new L{Message} initialized with the given values.\n        '
        response = Message(rCode=rCode)
        for (section, data) in [(response.answers, answers), (response.authority, authority), (response.additional, additional)]:
            section.extend([RRHeader(name, record.TYPE, getattr(record, 'CLASS', IN), payload=record) for (name, record) in data])
        return response

    def _getResolver(self, serverResponses, maximumQueries=10):
        if False:
            print('Hello World!')
        '\n        Create and return a new L{root.Resolver} modified to resolve queries\n        against the record data represented by C{servers}.\n\n        @param serverResponses: A mapping from dns server addresses to\n            mappings.  The inner mappings are from query two-tuples (name,\n            type) to dictionaries suitable for use as **arguments to\n            L{_respond}.  See that method for details.\n        '
        roots = ['1.1.2.3']
        resolver = Resolver(roots, maximumQueries)

        def query(query, serverAddresses, timeout, filter):
            if False:
                while True:
                    i = 10
            msg(f'Query for QNAME {query.name} at {serverAddresses!r}')
            for addr in serverAddresses:
                try:
                    server = serverResponses[addr]
                except KeyError:
                    continue
                records = server[query.name.name, query.type]
                return succeed(self._respond(**records))
        resolver._query = query
        return resolver

    def test_lookupAddress(self):
        if False:
            return 10
        '\n        L{root.Resolver.lookupAddress} looks up the I{A} records for the\n        specified hostname by first querying one of the root servers the\n        resolver was created with and then following the authority delegations\n        until a result is received.\n        '
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'authority': [(b'foo.example.com', Record_NS(b'ns1.example.com'))], 'additional': [(b'ns1.example.com', Record_A('34.55.89.144'))]}}, ('34.55.89.144', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.1'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        d.addCallback(getOneAddress)
        d.addCallback(self.assertEqual, '10.0.0.1')
        return d

    def test_lookupChecksClass(self):
        if False:
            i = 10
            return i + 15
        '\n        If a response includes a record with a class different from the one\n        in the query, it is ignored and lookup continues until a record with\n        the right class is found.\n        '
        badClass = Record_A('10.0.0.1')
        badClass.CLASS = HS
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', badClass)], 'authority': [(b'foo.example.com', Record_NS(b'ns1.example.com'))], 'additional': [(b'ns1.example.com', Record_A('10.0.0.2'))]}}, ('10.0.0.2', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.3'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        d.addCallback(getOnePayload)
        d.addCallback(self.assertEqual, Record_A('10.0.0.3'))
        return d

    def test_missingGlue(self):
        if False:
            i = 10
            return i + 15
        '\n        If an intermediate response includes no glue records for the\n        authorities, separate queries are made to find those addresses.\n        '
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'authority': [(b'foo.example.com', Record_NS(b'ns1.example.org'))]}, (b'ns1.example.org', A): {'answers': [(b'ns1.example.org', Record_A('10.0.0.1'))]}}, ('10.0.0.1', 53): {(b'foo.example.com', A): {'answers': [(b'foo.example.com', Record_A('10.0.0.2'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        d.addCallback(getOneAddress)
        d.addCallback(self.assertEqual, '10.0.0.2')
        return d

    def test_missingName(self):
        if False:
            while True:
                i = 10
        '\n        If a name is missing, L{Resolver.lookupAddress} returns a L{Deferred}\n        which fails with L{DNSNameError}.\n        '
        servers = {('1.1.2.3', 53): {(b'foo.example.com', A): {'rCode': ENAME}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'foo.example.com')
        return self.assertFailure(d, DNSNameError)

    def test_answerless(self):
        if False:
            while True:
                i = 10
        '\n        If a query is responded to with no answers or nameserver records, the\n        L{Deferred} returned by L{Resolver.lookupAddress} fires with\n        L{ResolverError}.\n        '
        servers = {('1.1.2.3', 53): {(b'example.com', A): {}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, ResolverError)

    def test_delegationLookupError(self):
        if False:
            while True:
                i = 10
        '\n        If there is an error resolving the nameserver in a delegation response,\n        the L{Deferred} returned by L{Resolver.lookupAddress} fires with that\n        error.\n        '
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {'rCode': ENAME}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, DNSNameError)

    def test_delegationLookupEmpty(self):
        if False:
            print('Hello World!')
        '\n        If there are no records in the response to a lookup of a delegation\n        nameserver, the L{Deferred} returned by L{Resolver.lookupAddress} fires\n        with L{ResolverError}.\n        '
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, ResolverError)

    def test_lookupNameservers(self):
        if False:
            print('Hello World!')
        '\n        L{Resolver.lookupNameservers} is like L{Resolver.lookupAddress}, except\n        it queries for I{NS} records instead of I{A} records.\n        '
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'rCode': ENAME}, (b'example.com', NS): {'answers': [(b'example.com', Record_NS(b'ns1.example.com'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupNameservers(b'example.com')

        def getOneName(results):
            if False:
                return 10
            (ans, auth, add) = results
            return ans[0].payload.name
        d.addCallback(getOneName)
        d.addCallback(self.assertEqual, Name(b'ns1.example.com'))
        return d

    def test_returnCanonicalName(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If a I{CNAME} record is encountered as the answer to a query for\n        another record type, that record is returned as the answer.\n        '
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_CNAME(b'example.net')), (b'example.net', Record_A('10.0.0.7'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        d.addCallback(lambda results: results[0])
        d.addCallback(self.assertEqual, [RRHeader(b'example.com', CNAME, payload=Record_CNAME(b'example.net')), RRHeader(b'example.net', A, payload=Record_A('10.0.0.7'))])
        return d

    def test_followCanonicalName(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If no record of the requested type is included in a response, but a\n        I{CNAME} record for the query name is included, queries are made to\n        resolve the value of the I{CNAME}.\n        '
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_CNAME(b'example.net'))]}, (b'example.net', A): {'answers': [(b'example.net', Record_A('10.0.0.5'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        d.addCallback(lambda results: results[0])
        d.addCallback(self.assertEqual, [RRHeader(b'example.com', CNAME, payload=Record_CNAME(b'example.net')), RRHeader(b'example.net', A, payload=Record_A('10.0.0.5'))])
        return d

    def test_detectCanonicalNameLoop(self):
        if False:
            i = 10
            return i + 15
        '\n        If there is a cycle between I{CNAME} records in a response, this is\n        detected and the L{Deferred} returned by the lookup method fails\n        with L{ResolverError}.\n        '
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_CNAME(b'example.net')), (b'example.net', Record_CNAME(b'example.com'))]}}}
        resolver = self._getResolver(servers)
        d = resolver.lookupAddress(b'example.com')
        return self.assertFailure(d, ResolverError)

    def test_boundedQueries(self):
        if False:
            while True:
                i = 10
        "\n        L{Resolver.lookupAddress} won't issue more queries following\n        delegations than the limit passed to its initializer.\n        "
        servers = {('1.1.2.3', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns1.example.com'))]}, (b'ns1.example.com', A): {'answers': [(b'ns1.example.com', Record_A('10.0.0.2'))]}}, ('10.0.0.2', 53): {(b'example.com', A): {'authority': [(b'example.com', Record_NS(b'ns2.example.com'))], 'additional': [(b'ns2.example.com', Record_A('10.0.0.3'))]}}, ('10.0.0.3', 53): {(b'example.com', A): {'answers': [(b'example.com', Record_A('10.0.0.4'))]}}}
        failer = self._getResolver(servers, 3)
        failD = self.assertFailure(failer.lookupAddress(b'example.com'), ResolverError)
        succeeder = self._getResolver(servers, 4)
        succeedD = succeeder.lookupAddress(b'example.com')
        succeedD.addCallback(getOnePayload)
        succeedD.addCallback(self.assertEqual, Record_A('10.0.0.4'))
        return gatherResults([failD, succeedD])

class ResolverFactoryArguments(Exception):
    """
    Raised by L{raisingResolverFactory} with the *args and **kwargs passed to
    that function.
    """

    def __init__(self, args, kwargs):
        if False:
            return 10
        '\n        Store the supplied args and kwargs as attributes.\n\n        @param args: Positional arguments.\n        @param kwargs: Keyword arguments.\n        '
        self.args = args
        self.kwargs = kwargs

def raisingResolverFactory(*args, **kwargs):
    if False:
        return 10
    '\n    Raise a L{ResolverFactoryArguments} exception containing the\n    positional and keyword arguments passed to resolverFactory.\n\n    @param args: A L{list} of all the positional arguments supplied by\n        the caller.\n\n    @param kwargs: A L{list} of all the keyword arguments supplied by\n        the caller.\n    '
    raise ResolverFactoryArguments(args, kwargs)

class RootResolverResolverFactoryTests(TestCase):
    """
    Tests for L{root.Resolver._resolverFactory}.
    """

    def test_resolverFactoryArgumentPresent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{root.Resolver.__init__} accepts a C{resolverFactory}\n        argument and assigns it to C{self._resolverFactory}.\n        '
        r = Resolver(hints=[None], resolverFactory=raisingResolverFactory)
        self.assertIs(r._resolverFactory, raisingResolverFactory)

    def test_resolverFactoryArgumentAbsent(self):
        if False:
            while True:
                i = 10
        '\n        L{root.Resolver.__init__} sets L{client.Resolver} as the\n        C{_resolverFactory} if a C{resolverFactory} argument is not\n        supplied.\n        '
        r = Resolver(hints=[None])
        self.assertIs(r._resolverFactory, client.Resolver)

    def test_resolverFactoryOnlyExpectedArguments(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{root.Resolver._resolverFactory} is supplied with C{reactor} and\n        C{servers} keyword arguments.\n        '
        dummyReactor = object()
        r = Resolver(hints=['192.0.2.101'], resolverFactory=raisingResolverFactory, reactor=dummyReactor)
        e = self.assertRaises(ResolverFactoryArguments, r.lookupAddress, 'example.com')
        self.assertEqual(((), {'reactor': dummyReactor, 'servers': [('192.0.2.101', 53)]}), (e.args, e.kwargs))
ROOT_SERVERS = ['a.root-servers.net', 'b.root-servers.net', 'c.root-servers.net', 'd.root-servers.net', 'e.root-servers.net', 'f.root-servers.net', 'g.root-servers.net', 'h.root-servers.net', 'i.root-servers.net', 'j.root-servers.net', 'k.root-servers.net', 'l.root-servers.net', 'm.root-servers.net']

@implementer(IResolverSimple)
class StubResolver:
    """
    An L{IResolverSimple} implementer which traces all getHostByName
    calls and their deferred results. The deferred results can be
    accessed and fired synchronously.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        @type calls: L{list} of L{tuple} containing C{args} and\n            C{kwargs} supplied to C{getHostByName} calls.\n        @type pendingResults: L{list} of L{Deferred} returned by\n            C{getHostByName}.\n        '
        self.calls = []
        self.pendingResults = []

    def getHostByName(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        A fake implementation of L{IResolverSimple.getHostByName}\n\n        @param args: A L{list} of all the positional arguments supplied by\n           the caller.\n\n        @param kwargs: A L{list} of all the keyword arguments supplied by\n           the caller.\n\n        @return: A L{Deferred} which may be fired later from the test\n            fixture.\n        '
        self.calls.append((args, kwargs))
        d = Deferred()
        self.pendingResults.append(d)
        return d
verifyClass(IResolverSimple, StubResolver)

class BootstrapTests(SynchronousTestCase):
    """
    Tests for L{root.bootstrap}
    """

    def test_returnsDeferredResolver(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{root.bootstrap} returns an object which is initially a\n        L{root.DeferredResolver}.\n        '
        deferredResolver = root.bootstrap(StubResolver())
        self.assertIsInstance(deferredResolver, root.DeferredResolver)

    def test_resolves13RootServers(self):
        if False:
            print('Hello World!')
        '\n        The L{IResolverSimple} supplied to L{root.bootstrap} is used to lookup\n        the IP addresses of the 13 root name servers.\n        '
        stubResolver = StubResolver()
        root.bootstrap(stubResolver)
        self.assertEqual(stubResolver.calls, [((s,), {}) for s in ROOT_SERVERS])

    def test_becomesResolver(self):
        if False:
            return 10
        '\n        The L{root.DeferredResolver} initially returned by L{root.bootstrap}\n        becomes a L{root.Resolver} when the supplied resolver has successfully\n        looked up all root hints.\n        '
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        for d in stubResolver.pendingResults:
            d.callback('192.0.2.101')
        self.assertIsInstance(deferredResolver, Resolver)

    def test_resolverReceivesRootHints(self):
        if False:
            i = 10
            return i + 15
        '\n        The L{root.Resolver} which eventually replaces L{root.DeferredResolver}\n        is supplied with the IP addresses of the 13 root servers.\n        '
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        for d in stubResolver.pendingResults:
            d.callback('192.0.2.101')
        self.assertEqual(deferredResolver.hints, ['192.0.2.101'] * 13)

    def test_continuesWhenSomeRootHintsFail(self):
        if False:
            while True:
                i = 10
        '\n        The L{root.Resolver} is eventually created, even if some of the root\n        hint lookups fail. Only the working root hint IP addresses are supplied\n        to the L{root.Resolver}.\n        '
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        results = iter(stubResolver.pendingResults)
        d1 = next(results)
        for d in results:
            d.callback('192.0.2.101')
        d1.errback(TimeoutError())

        def checkHints(res):
            if False:
                return 10
            self.assertEqual(deferredResolver.hints, ['192.0.2.101'] * 12)
        d1.addBoth(checkHints)

    def test_continuesWhenAllRootHintsFail(self):
        if False:
            print('Hello World!')
        '\n        The L{root.Resolver} is eventually created, even if all of the root hint\n        lookups fail. Pending and new lookups will then fail with\n        AttributeError.\n        '
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver)
        results = iter(stubResolver.pendingResults)
        d1 = next(results)
        for d in results:
            d.errback(TimeoutError())
        d1.errback(TimeoutError())

        def checkHints(res):
            if False:
                i = 10
                return i + 15
            self.assertEqual(deferredResolver.hints, [])
        d1.addBoth(checkHints)
        self.addCleanup(self.flushLoggedErrors, TimeoutError)

    def test_passesResolverFactory(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{root.bootstrap} accepts a C{resolverFactory} argument which is passed\n        as an argument to L{root.Resolver} when it has successfully looked up\n        root hints.\n        '
        stubResolver = StubResolver()
        deferredResolver = root.bootstrap(stubResolver, resolverFactory=raisingResolverFactory)
        for d in stubResolver.pendingResults:
            d.callback('192.0.2.101')
        self.assertIs(deferredResolver._resolverFactory, raisingResolverFactory)

class StubDNSDatagramProtocol:
    """
    A do-nothing stand-in for L{DNSDatagramProtocol} which can be used to avoid
    network traffic in tests where that kind of thing doesn't matter.
    """

    def query(self, *a, **kw):
        if False:
            while True:
                i = 10
        return Deferred()
_retrySuppression = util.suppress(category=DeprecationWarning, message='twisted.names.root.retry is deprecated since Twisted 10.0.  Use a Resolver object for retry logic.')