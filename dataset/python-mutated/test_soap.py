"""Test SOAP support."""
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.trial.unittest import TestCase
from twisted.web import error, server
try:
    import SOAPpy
    from twisted.web import soap
    from twisted.web.soap import SOAPPublisher
except ImportError:
    SOAPpy = None
    SOAPPublisher = object

class Test(SOAPPublisher):

    def soap_add(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        return a + b

    def soap_kwargs(self, a=1, b=2):
        if False:
            while True:
                i = 10
        return a + b
    soap_kwargs.useKeywords = True

    def soap_triple(self, string, num):
        if False:
            i = 10
            return i + 15
        return [string, num, None]

    def soap_struct(self):
        if False:
            for i in range(10):
                print('nop')
        return SOAPpy.structType({'a': 'c'})

    def soap_defer(self, x):
        if False:
            i = 10
            return i + 15
        return defer.succeed(x)

    def soap_deferFail(self):
        if False:
            while True:
                i = 10
        return defer.fail(ValueError())

    def soap_fail(self):
        if False:
            return 10
        raise RuntimeError

    def soap_deferFault(self):
        if False:
            return 10
        return defer.fail(ValueError())

    def soap_complex(self):
        if False:
            i = 10
            return i + 15
        return {'a': ['b', 'c', 12, []], 'D': 'foo'}

    def soap_dict(self, map, key):
        if False:
            print('Hello World!')
        return map[key]

@skipIf(not SOAPpy, 'SOAPpy not installed')
class SOAPTests(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.publisher = Test()
        self.p = reactor.listenTCP(0, server.Site(self.publisher), interface='127.0.0.1')
        self.port = self.p.getHost().port

    def tearDown(self):
        if False:
            while True:
                i = 10
        return self.p.stopListening()

    def proxy(self):
        if False:
            print('Hello World!')
        return soap.Proxy('http://127.0.0.1:%d/' % self.port)

    def testResults(self):
        if False:
            i = 10
            return i + 15
        inputOutput = [('add', (2, 3), 5), ('defer', ('a',), 'a'), ('dict', ({'a': 1}, 'a'), 1), ('triple', ('a', 1), ['a', 1, None])]
        dl = []
        for (meth, args, outp) in inputOutput:
            d = self.proxy().callRemote(meth, *args)
            d.addCallback(self.assertEqual, outp)
            dl.append(d)
        d = self.proxy().callRemote('complex')
        d.addCallback(lambda result: result._asdict())
        d.addCallback(self.assertEqual, {'a': ['b', 'c', 12, []], 'D': 'foo'})
        dl.append(d)
        return defer.DeferredList(dl, fireOnOneErrback=True)

    def testMethodNotFound(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that a non existing method return error 500.\n        '
        d = self.proxy().callRemote('doesntexist')
        self.assertFailure(d, error.Error)

        def cb(err):
            if False:
                while True:
                    i = 10
            self.assertEqual(int(err.status), 500)
        d.addCallback(cb)
        return d

    def testLookupFunction(self):
        if False:
            print('Hello World!')
        '\n        Test lookupFunction method on publisher, to see available remote\n        methods.\n        '
        self.assertTrue(self.publisher.lookupFunction('add'))
        self.assertTrue(self.publisher.lookupFunction('fail'))
        self.assertFalse(self.publisher.lookupFunction('foobar'))