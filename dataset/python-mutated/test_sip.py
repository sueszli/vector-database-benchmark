"""
Session Initialization Protocol tests.
"""
from twisted.cred import checkers, portal
from twisted.internet import defer, reactor
from twisted.protocols import sip
from twisted.trial import unittest
try:
    from twisted.internet.asyncioreactor import AsyncioSelectorReactor
except BaseException:
    AsyncioSelectorReactor = None
from zope.interface import implementer
request1 = '\n\r\n\n\r' + 'INVITE sip:foo SIP/2.0\nFrom: mo\nTo: joe\nContent-Length: 4\n\nabcd'.replace('\n', '\r\n')
request2 = 'INVITE sip:foo SIP/2.0\nFrom: mo\nTo: joe\n\n1234'.replace('\n', '\r\n')
request3 = 'INVITE sip:foo SIP/2.0\nFrom: mo\nTo: joe\nContent-Length: 4\n\n1234\n\nlalalal'.replace('\n', '\r\n')
request4 = 'INVITE sip:foo SIP/2.0\nFrom: mo\nTo: joe\nContent-Length: 0\n\nINVITE sip:loop SIP/2.0\nFrom: foo\nTo: bar\nContent-Length: 4\n\nabcdINVITE sip:loop SIP/2.0\nFrom: foo\nTo: bar\nContent-Length: 4\n\n1234'.replace('\n', '\r\n')
response1 = 'SIP/2.0 200 OK\nFrom:  foo\nTo:bar\nContent-Length: 0\n\n'.replace('\n', '\r\n')
request_short = 'INVITE sip:foo SIP/2.0\nf: mo\nt: joe\nl: 4\n\nabcd'.replace('\n', '\r\n')
request_natted = 'INVITE sip:foo SIP/2.0\nVia: SIP/2.0/UDP 10.0.0.1:5060;rport\n\n'.replace('\n', '\r\n')
response_multiline = 'SIP/2.0 200 OK\nVia: SIP/2.0/UDP server10.biloxi.com\n    ;branch=z9hG4bKnashds8;received=192.0.2.3\nVia: SIP/2.0/UDP bigbox3.site3.atlanta.com\n    ;branch=z9hG4bK77ef4c2312983.1;received=192.0.2.2\nVia: SIP/2.0/UDP pc33.atlanta.com\n    ;branch=z9hG4bK776asdhds ;received=192.0.2.1\nTo: Bob <sip:bob@biloxi.com>;tag=a6c85cf\nFrom: Alice <sip:alice@atlanta.com>;tag=1928301774\nCall-ID: a84b4c76e66710@pc33.atlanta.com\nCSeq: 314159 INVITE\nContact: <sip:bob@192.0.2.4>\nContent-Type: application/sdp\nContent-Length: 0\n\n'.replace('\n', '\r\n')

class TestRealm:

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            print('Hello World!')
        return (sip.IContact, None, lambda : None)

class MessageParsingTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.l = []
        self.parser = sip.MessagesParser(self.l.append)

    def feedMessage(self, message):
        if False:
            i = 10
            return i + 15
        self.parser.dataReceived(message)
        self.parser.dataDone()

    def validateMessage(self, m, method, uri, headers, body):
        if False:
            while True:
                i = 10
        '\n        Validate Requests.\n        '
        self.assertEqual(m.method, method)
        self.assertEqual(m.uri.toString(), uri)
        self.assertEqual(m.headers, headers)
        self.assertEqual(m.body, body)
        self.assertEqual(m.finished, 1)

    def testSimple(self):
        if False:
            while True:
                i = 10
        l = self.l
        self.feedMessage(request1)
        self.assertEqual(len(l), 1)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, 'abcd')

    def testTwoMessages(self):
        if False:
            i = 10
            return i + 15
        l = self.l
        self.feedMessage(request1)
        self.feedMessage(request2)
        self.assertEqual(len(l), 2)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, 'abcd')
        self.validateMessage(l[1], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe']}, '1234')

    def testGarbage(self):
        if False:
            i = 10
            return i + 15
        l = self.l
        self.feedMessage(request3)
        self.assertEqual(len(l), 1)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, '1234')

    def testThreeInOne(self):
        if False:
            return 10
        l = self.l
        self.feedMessage(request4)
        self.assertEqual(len(l), 3)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['0']}, '')
        self.validateMessage(l[1], 'INVITE', 'sip:loop', {'from': ['foo'], 'to': ['bar'], 'content-length': ['4']}, 'abcd')
        self.validateMessage(l[2], 'INVITE', 'sip:loop', {'from': ['foo'], 'to': ['bar'], 'content-length': ['4']}, '1234')

    def testShort(self):
        if False:
            while True:
                i = 10
        l = self.l
        self.feedMessage(request_short)
        self.assertEqual(len(l), 1)
        self.validateMessage(l[0], 'INVITE', 'sip:foo', {'from': ['mo'], 'to': ['joe'], 'content-length': ['4']}, 'abcd')

    def testSimpleResponse(self):
        if False:
            print('Hello World!')
        l = self.l
        self.feedMessage(response1)
        self.assertEqual(len(l), 1)
        m = l[0]
        self.assertEqual(m.code, 200)
        self.assertEqual(m.phrase, 'OK')
        self.assertEqual(m.headers, {'from': ['foo'], 'to': ['bar'], 'content-length': ['0']})
        self.assertEqual(m.body, '')
        self.assertEqual(m.finished, 1)

    def test_multiLine(self):
        if False:
            return 10
        '\n        A header may be split across multiple lines.  Subsequent lines begin\n        with C{" "} or C{"\\t"}.\n        '
        l = self.l
        self.feedMessage(response_multiline)
        self.assertEqual(len(l), 1)
        m = l[0]
        self.assertEqual(m.headers['via'][0], 'SIP/2.0/UDP server10.biloxi.com;branch=z9hG4bKnashds8;received=192.0.2.3')
        self.assertEqual(m.headers['via'][1], 'SIP/2.0/UDP bigbox3.site3.atlanta.com;branch=z9hG4bK77ef4c2312983.1;received=192.0.2.2')
        self.assertEqual(m.headers['via'][2], 'SIP/2.0/UDP pc33.atlanta.com;branch=z9hG4bK776asdhds ;received=192.0.2.1')

class MessageParsingFeedDataCharByCharTests(MessageParsingTests):
    """
    Same as base class, but feed data char by char.
    """

    def feedMessage(self, message):
        if False:
            return 10
        for c in message:
            self.parser.dataReceived(c)
        self.parser.dataDone()

class MakeMessageTests(unittest.TestCase):

    def testRequest(self):
        if False:
            return 10
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('foo', 'bar')
        self.assertEqual(r.toString(), 'INVITE sip:foo SIP/2.0\r\nFoo: bar\r\n\r\n')

    def testResponse(self):
        if False:
            for i in range(10):
                print('nop')
        r = sip.Response(200, 'OK')
        r.addHeader('foo', 'bar')
        r.addHeader('Content-Length', '4')
        r.bodyDataReceived('1234')
        self.assertEqual(r.toString(), 'SIP/2.0 200 OK\r\nFoo: bar\r\nContent-Length: 4\r\n\r\n1234')

    def testStatusCode(self):
        if False:
            i = 10
            return i + 15
        r = sip.Response(200)
        self.assertEqual(r.toString(), 'SIP/2.0 200 OK\r\n\r\n')

class ViaTests(unittest.TestCase):

    def checkRoundtrip(self, v):
        if False:
            for i in range(10):
                print('nop')
        s = v.toString()
        self.assertEqual(s, sip.parseViaHeader(s).toString())

    def testExtraWhitespace(self):
        if False:
            return 10
        v1 = sip.parseViaHeader('SIP/2.0/UDP 192.168.1.1:5060')
        v2 = sip.parseViaHeader('SIP/2.0/UDP     192.168.1.1:5060')
        self.assertEqual(v1.transport, v2.transport)
        self.assertEqual(v1.host, v2.host)
        self.assertEqual(v1.port, v2.port)

    def test_complex(self):
        if False:
            print('Hello World!')
        '\n        Test parsing a Via header with one of everything.\n        '
        s = 'SIP/2.0/UDP first.example.com:4000;ttl=16;maddr=224.2.0.1 ;branch=a7c6a8dlze (Example)'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.transport, 'UDP')
        self.assertEqual(v.host, 'first.example.com')
        self.assertEqual(v.port, 4000)
        self.assertIsNone(v.rport)
        self.assertIsNone(v.rportValue)
        self.assertFalse(v.rportRequested)
        self.assertEqual(v.ttl, 16)
        self.assertEqual(v.maddr, '224.2.0.1')
        self.assertEqual(v.branch, 'a7c6a8dlze')
        self.assertEqual(v.hidden, 0)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP first.example.com:4000;ttl=16;branch=a7c6a8dlze;maddr=224.2.0.1')
        self.checkRoundtrip(v)

    def test_simple(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test parsing a simple Via header.\n        '
        s = 'SIP/2.0/UDP example.com;hidden'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.transport, 'UDP')
        self.assertEqual(v.host, 'example.com')
        self.assertEqual(v.port, 5060)
        self.assertIsNone(v.rport)
        self.assertIsNone(v.rportValue)
        self.assertFalse(v.rportRequested)
        self.assertIsNone(v.ttl)
        self.assertIsNone(v.maddr)
        self.assertIsNone(v.branch)
        self.assertTrue(v.hidden)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP example.com:5060;hidden')
        self.checkRoundtrip(v)

    def testSimpler(self):
        if False:
            for i in range(10):
                print('nop')
        v = sip.Via('example.com')
        self.checkRoundtrip(v)

    def test_deprecatedRPort(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setting rport to True is deprecated, but still produces a Via header\n        with the expected properties.\n        '
        v = sip.Via('foo.bar', rport=True)
        warnings = self.flushWarnings(offendingFunctions=[self.test_deprecatedRPort])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['message'], 'rport=True is deprecated since Twisted 9.0.')
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport')
        self.assertTrue(v.rport)
        self.assertTrue(v.rportRequested)
        self.assertIsNone(v.rportValue)

    def test_rport(self):
        if False:
            return 10
        '\n        An rport setting of None should insert the parameter with no value.\n        '
        v = sip.Via('foo.bar', rport=None)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport')
        self.assertTrue(v.rportRequested)
        self.assertIsNone(v.rportValue)

    def test_rportValue(self):
        if False:
            print('Hello World!')
        '\n        An rport numeric setting should insert the parameter with the number\n        value given.\n        '
        v = sip.Via('foo.bar', rport=1)
        self.assertEqual(v.toString(), 'SIP/2.0/UDP foo.bar:5060;rport=1')
        self.assertFalse(v.rportRequested)
        self.assertEqual(v.rportValue, 1)
        self.assertEqual(v.rport, 1)

    def testNAT(self):
        if False:
            i = 10
            return i + 15
        s = 'SIP/2.0/UDP 10.0.0.1:5060;received=22.13.1.5;rport=12345'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.transport, 'UDP')
        self.assertEqual(v.host, '10.0.0.1')
        self.assertEqual(v.port, 5060)
        self.assertEqual(v.received, '22.13.1.5')
        self.assertEqual(v.rport, 12345)
        self.assertNotEqual(v.toString().find('rport=12345'), -1)

    def test_unknownParams(self):
        if False:
            while True:
                i = 10
        '\n        Parsing and serializing Via headers with unknown parameters should work.\n        '
        s = 'SIP/2.0/UDP example.com:5060;branch=a12345b;bogus;pie=delicious'
        v = sip.parseViaHeader(s)
        self.assertEqual(v.toString(), s)

class URLTests(unittest.TestCase):

    def testRoundtrip(self):
        if False:
            while True:
                i = 10
        for url in ['sip:j.doe@big.com', 'sip:j.doe:secret@big.com;transport=tcp', 'sip:j.doe@big.com?subject=project', 'sip:example.com']:
            self.assertEqual(sip.parseURL(url).toString(), url)

    def testComplex(self):
        if False:
            while True:
                i = 10
        s = 'sip:user:pass@hosta:123;transport=udp;user=phone;method=foo;ttl=12;maddr=1.2.3.4;blah;goo=bar?a=b&c=d'
        url = sip.parseURL(s)
        for (k, v) in [('username', 'user'), ('password', 'pass'), ('host', 'hosta'), ('port', 123), ('transport', 'udp'), ('usertype', 'phone'), ('method', 'foo'), ('ttl', 12), ('maddr', '1.2.3.4'), ('other', ['blah', 'goo=bar']), ('headers', {'a': 'b', 'c': 'd'})]:
            self.assertEqual(getattr(url, k), v)

class ParseTests(unittest.TestCase):

    def testParseAddress(self):
        if False:
            print('Hello World!')
        for (address, name, urls, params) in [('"A. G. Bell" <sip:foo@example.com>', 'A. G. Bell', 'sip:foo@example.com', {}), ('Anon <sip:foo@example.com>', 'Anon', 'sip:foo@example.com', {}), ('sip:foo@example.com', '', 'sip:foo@example.com', {}), ('<sip:foo@example.com>', '', 'sip:foo@example.com', {}), ('foo <sip:foo@example.com>;tag=bar;foo=baz', 'foo', 'sip:foo@example.com', {'tag': 'bar', 'foo': 'baz'})]:
            (gname, gurl, gparams) = sip.parseAddress(address)
            self.assertEqual(name, gname)
            self.assertEqual(gurl.toString(), urls)
            self.assertEqual(gparams, params)

@implementer(sip.ILocator)
class DummyLocator:

    def getAddress(self, logicalURL):
        if False:
            i = 10
            return i + 15
        return defer.succeed(sip.URL('server.com', port=5060))

@implementer(sip.ILocator)
class FailingLocator:

    def getAddress(self, logicalURL):
        if False:
            return 10
        return defer.fail(LookupError())

class ProxyTests(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.proxy = sip.Proxy('127.0.0.1')
        self.proxy.locator = DummyLocator()
        self.sent = []
        self.proxy.sendMessage = lambda dest, msg: self.sent.append((dest, msg))

    def testRequestForward(self):
        if False:
            for i in range(10):
                print('nop')
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('via', sip.Via('1.2.3.5').toString())
        r.addHeader('foo', 'bar')
        r.addHeader('to', '<sip:joe@server.com>')
        r.addHeader('contact', '<sip:joe@1.2.3.5>')
        self.proxy.datagramReceived(r.toString(), ('1.2.3.4', 5060))
        self.assertEqual(len(self.sent), 1)
        (dest, m) = self.sent[0]
        self.assertEqual(dest.port, 5060)
        self.assertEqual(dest.host, 'server.com')
        self.assertEqual(m.uri.toString(), 'sip:foo')
        self.assertEqual(m.method, 'INVITE')
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 127.0.0.1:5060', 'SIP/2.0/UDP 1.2.3.4:5060', 'SIP/2.0/UDP 1.2.3.5:5060'])

    def testReceivedRequestForward(self):
        if False:
            for i in range(10):
                print('nop')
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('foo', 'bar')
        r.addHeader('to', '<sip:joe@server.com>')
        r.addHeader('contact', '<sip:joe@1.2.3.4>')
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        (dest, m) = self.sent[0]
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 127.0.0.1:5060', 'SIP/2.0/UDP 1.2.3.4:5060;received=1.1.1.1'])

    def testResponseWrongVia(self):
        if False:
            for i in range(10):
                print('nop')
        r = sip.Response(200)
        r.addHeader('via', sip.Via('foo.com').toString())
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(self.sent), 0)

    def testResponseForward(self):
        if False:
            i = 10
            return i + 15
        r = sip.Response(200)
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        r.addHeader('via', sip.Via('client.com', port=1234).toString())
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(self.sent), 1)
        (dest, m) = self.sent[0]
        self.assertEqual((dest.host, dest.port), ('client.com', 1234))
        self.assertEqual(m.code, 200)
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP client.com:1234'])

    def testReceivedResponseForward(self):
        if False:
            i = 10
            return i + 15
        r = sip.Response(200)
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        r.addHeader('via', sip.Via('10.0.0.1', received='client.com').toString())
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(self.sent), 1)
        (dest, m) = self.sent[0]
        self.assertEqual((dest.host, dest.port), ('client.com', 5060))

    def testResponseToUs(self):
        if False:
            return 10
        r = sip.Response(200)
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        l = []
        self.proxy.gotResponse = lambda *a: l.append(a)
        self.proxy.datagramReceived(r.toString(), ('1.1.1.1', 5060))
        self.assertEqual(len(l), 1)
        (m, addr) = l[0]
        self.assertEqual(len(m.headers.get('via', [])), 0)
        self.assertEqual(m.code, 200)

    def testLoop(self):
        if False:
            while True:
                i = 10
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('via', sip.Via('127.0.0.1').toString())
        self.proxy.datagramReceived(r.toString(), ('client.com', 5060))
        self.assertEqual(self.sent, [])

    def testCantForwardRequest(self):
        if False:
            while True:
                i = 10
        r = sip.Request('INVITE', 'sip:foo')
        r.addHeader('via', sip.Via('1.2.3.4').toString())
        r.addHeader('to', '<sip:joe@server.com>')
        self.proxy.locator = FailingLocator()
        self.proxy.datagramReceived(r.toString(), ('1.2.3.4', 5060))
        self.assertEqual(len(self.sent), 1)
        (dest, m) = self.sent[0]
        self.assertEqual((dest.host, dest.port), ('1.2.3.4', 5060))
        self.assertEqual(m.code, 404)
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP 1.2.3.4:5060'])

class RegistrationTests(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.proxy = sip.RegisterProxy(host='127.0.0.1')
        self.registry = sip.InMemoryRegistry('bell.example.com')
        self.proxy.registry = self.proxy.locator = self.registry
        self.sent = []
        self.proxy.sendMessage = lambda dest, msg: self.sent.append((dest, msg))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        for (d, uri) in self.registry.users.values():
            d.cancel()
        del self.proxy

    def register(self):
        if False:
            for i in range(10):
                print('nop')
        r = sip.Request('REGISTER', 'sip:bell.example.com')
        r.addHeader('to', 'sip:joe@bell.example.com')
        r.addHeader('contact', 'sip:joe@client.com:1234')
        r.addHeader('via', sip.Via('client.com').toString())
        self.proxy.datagramReceived(r.toString(), ('client.com', 5060))

    def unregister(self):
        if False:
            return 10
        r = sip.Request('REGISTER', 'sip:bell.example.com')
        r.addHeader('to', 'sip:joe@bell.example.com')
        r.addHeader('contact', '*')
        r.addHeader('via', sip.Via('client.com').toString())
        r.addHeader('expires', '0')
        self.proxy.datagramReceived(r.toString(), ('client.com', 5060))

    def testRegister(self):
        if False:
            while True:
                i = 10
        self.register()
        (dest, m) = self.sent[0]
        self.assertEqual((dest.host, dest.port), ('client.com', 5060))
        self.assertEqual(m.code, 200)
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP client.com:5060'])
        self.assertEqual(m.headers['to'], ['sip:joe@bell.example.com'])
        self.assertEqual(m.headers['contact'], ['sip:joe@client.com:5060'])
        if type(reactor) != AsyncioSelectorReactor:
            self.assertTrue(int(m.headers['expires'][0]) in (3600, 3601, 3599, 3598))
        self.assertEqual(len(self.registry.users), 1)
        (dc, uri) = self.registry.users['joe']
        self.assertEqual(uri.toString(), 'sip:joe@client.com:5060')
        d = self.proxy.locator.getAddress(sip.URL(username='joe', host='bell.example.com'))
        d.addCallback(lambda desturl: (desturl.host, desturl.port))
        d.addCallback(self.assertEqual, ('client.com', 5060))
        return d

    def testUnregister(self):
        if False:
            return 10
        self.register()
        self.unregister()
        (dest, m) = self.sent[1]
        self.assertEqual((dest.host, dest.port), ('client.com', 5060))
        self.assertEqual(m.code, 200)
        self.assertEqual(m.headers['via'], ['SIP/2.0/UDP client.com:5060'])
        self.assertEqual(m.headers['to'], ['sip:joe@bell.example.com'])
        self.assertEqual(m.headers['contact'], ['sip:joe@client.com:5060'])
        self.assertEqual(m.headers['expires'], ['0'])
        self.assertEqual(self.registry.users, {})

    def addPortal(self):
        if False:
            for i in range(10):
                print('nop')
        r = TestRealm()
        p = portal.Portal(r)
        c = checkers.InMemoryUsernamePasswordDatabaseDontUse()
        c.addUser('userXname@127.0.0.1', 'passXword')
        p.registerChecker(c)
        self.proxy.portal = p

    def testFailedAuthentication(self):
        if False:
            i = 10
            return i + 15
        self.addPortal()
        self.register()
        self.assertEqual(len(self.registry.users), 0)
        self.assertEqual(len(self.sent), 1)
        (dest, m) = self.sent[0]
        self.assertEqual(m.code, 401)

    def testWrongDomainRegister(self):
        if False:
            print('Hello World!')
        r = sip.Request('REGISTER', 'sip:wrong.com')
        r.addHeader('to', 'sip:joe@bell.example.com')
        r.addHeader('contact', 'sip:joe@client.com:1234')
        r.addHeader('via', sip.Via('client.com').toString())
        self.proxy.datagramReceived(r.toString(), ('client.com', 5060))
        self.assertEqual(len(self.sent), 0)

    def testWrongToDomainRegister(self):
        if False:
            while True:
                i = 10
        r = sip.Request('REGISTER', 'sip:bell.example.com')
        r.addHeader('to', 'sip:joe@foo.com')
        r.addHeader('contact', 'sip:joe@client.com:1234')
        r.addHeader('via', sip.Via('client.com').toString())
        self.proxy.datagramReceived(r.toString(), ('client.com', 5060))
        self.assertEqual(len(self.sent), 0)

    def testWrongDomainLookup(self):
        if False:
            print('Hello World!')
        self.register()
        url = sip.URL(username='joe', host='foo.com')
        d = self.proxy.locator.getAddress(url)
        self.assertFailure(d, LookupError)
        return d

    def testNoContactLookup(self):
        if False:
            while True:
                i = 10
        self.register()
        url = sip.URL(username='jane', host='bell.example.com')
        d = self.proxy.locator.getAddress(url)
        self.assertFailure(d, LookupError)
        return d

class Client(sip.Base):

    def __init__(self):
        if False:
            while True:
                i = 10
        sip.Base.__init__(self)
        self.received = []
        self.deferred = defer.Deferred()

    def handle_response(self, response, addr):
        if False:
            i = 10
            return i + 15
        self.received.append(response)
        self.deferred.callback(self.received)

class LiveTests(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.proxy = sip.RegisterProxy(host='127.0.0.1')
        self.registry = sip.InMemoryRegistry('bell.example.com')
        self.proxy.registry = self.proxy.locator = self.registry
        self.serverPort = reactor.listenUDP(0, self.proxy, interface='127.0.0.1')
        self.client = Client()
        self.clientPort = reactor.listenUDP(0, self.client, interface='127.0.0.1')
        self.serverAddress = (self.serverPort.getHost().host, self.serverPort.getHost().port)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        for (d, uri) in self.registry.users.values():
            d.cancel()
        d1 = defer.maybeDeferred(self.clientPort.stopListening)
        d2 = defer.maybeDeferred(self.serverPort.stopListening)
        return defer.gatherResults([d1, d2])

    def testRegister(self):
        if False:
            print('Hello World!')
        p = self.clientPort.getHost().port
        r = sip.Request('REGISTER', 'sip:bell.example.com')
        r.addHeader('to', 'sip:joe@bell.example.com')
        r.addHeader('contact', 'sip:joe@127.0.0.1:%d' % p)
        r.addHeader('via', sip.Via('127.0.0.1', port=p).toString())
        self.client.sendMessage(sip.URL(host='127.0.0.1', port=self.serverAddress[1]), r)
        d = self.client.deferred

        def check(received):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(len(received), 1)
            r = received[0]
            self.assertEqual(r.code, 200)
        d.addCallback(check)
        return d

    def test_amoralRPort(self):
        if False:
            i = 10
            return i + 15
        "\n        rport is allowed without a value, apparently because server\n        implementors might be too stupid to check the received port\n        against 5060 and see if they're equal, and because client\n        implementors might be too stupid to bind to port 5060, or set a\n        value on the rport parameter they send if they bind to another\n        port.\n        "
        p = self.clientPort.getHost().port
        r = sip.Request('REGISTER', 'sip:bell.example.com')
        r.addHeader('to', 'sip:joe@bell.example.com')
        r.addHeader('contact', 'sip:joe@127.0.0.1:%d' % p)
        r.addHeader('via', sip.Via('127.0.0.1', port=p, rport=True).toString())
        warnings = self.flushWarnings(offendingFunctions=[self.test_amoralRPort])
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['message'], 'rport=True is deprecated since Twisted 9.0.')
        self.assertEqual(warnings[0]['category'], DeprecationWarning)
        self.client.sendMessage(sip.URL(host='127.0.0.1', port=self.serverAddress[1]), r)
        d = self.client.deferred

        def check(received):
            if False:
                i = 10
                return i + 15
            self.assertEqual(len(received), 1)
            r = received[0]
            self.assertEqual(r.code, 200)
        d.addCallback(check)
        return d