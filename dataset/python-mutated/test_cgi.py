"""
Tests for L{twisted.web.twcgi}.
"""
import json
import os
import sys
from io import BytesIO
from twisted.internet import address, error, interfaces, reactor
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, util
from twisted.trial import unittest
from twisted.web import client, http, http_headers, resource, server, twcgi
from twisted.web.http import INTERNAL_SERVER_ERROR, NOT_FOUND
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
DUMMY_CGI = 'print("Header: OK")\nprint("")\nprint("cgi output")\n'
DUAL_HEADER_CGI = 'print("Header: spam")\nprint("Header: eggs")\nprint("")\nprint("cgi output")\n'
BROKEN_HEADER_CGI = 'print("XYZ")\nprint("")\nprint("cgi output")\n'
SPECIAL_HEADER_CGI = 'print("Server: monkeys")\nprint("Date: last year")\nprint("")\nprint("cgi output")\n'
READINPUT_CGI = '# This is an example of a correctly-written CGI script which reads a body\n# from stdin, which only reads env[\'CONTENT_LENGTH\'] bytes.\n\nimport os, sys\n\nbody_length = int(os.environ.get(\'CONTENT_LENGTH\',0))\nindata = sys.stdin.read(body_length)\nprint("Header: OK")\nprint("")\nprint("readinput ok")\n'
READALLINPUT_CGI = '# This is an example of the typical (incorrect) CGI script which expects\n# the server to close stdin when the body of the request is complete.\n# A correct CGI should only read env[\'CONTENT_LENGTH\'] bytes.\n\nimport sys\n\nindata = sys.stdin.read()\nprint("Header: OK")\nprint("")\nprint("readallinput ok")\n'
NO_DUPLICATE_CONTENT_TYPE_HEADER_CGI = 'print("content-type: text/cgi-duplicate-test")\nprint("")\nprint("cgi output")\n'
HEADER_OUTPUT_CGI = 'import json\nimport os\nprint("")\nprint("")\nvals = {x:y for x,y in os.environ.items() if x.startswith("HTTP_")}\nprint(json.dumps(vals))\n'
URL_PARAMETER_CGI = 'import os\nparam = str(os.environ[\'QUERY_STRING\'])\nprint("Header: OK")\nprint("")\nprint(param)\n'

class PythonScript(twcgi.FilteredScript):
    filter = sys.executable

class _StartServerAndTearDownMixin:

    def startServer(self, cgi):
        if False:
            for i in range(10):
                print('nop')
        root = resource.Resource()
        cgipath = util.sibpath(__file__, cgi)
        root.putChild(b'cgi', PythonScript(cgipath))
        site = server.Site(root)
        self.p = reactor.listenTCP(0, site)
        return self.p.getHost().port

    def tearDown(self):
        if False:
            print('Hello World!')
        if getattr(self, 'p', None):
            return self.p.stopListening()

    def writeCGI(self, source):
        if False:
            while True:
                i = 10
        cgiFilename = os.path.abspath(self.mktemp())
        with open(cgiFilename, 'wt') as cgiFile:
            cgiFile.write(source)
        return cgiFilename

class CGITests(_StartServerAndTearDownMixin, unittest.TestCase):
    """
    Tests for L{twcgi.FilteredScript}.
    """
    if not interfaces.IReactorProcess.providedBy(reactor):
        skip = 'CGI tests require a functional reactor.spawnProcess()'

    def test_CGI(self):
        if False:
            while True:
                i = 10
        cgiFilename = self.writeCGI(DUMMY_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = client.Agent(reactor).request(b'GET', url)
        d.addCallback(client.readBody)
        d.addCallback(self._testCGI_1)
        return d

    def _testCGI_1(self, res):
        if False:
            print('Hello World!')
        self.assertEqual(res, b'cgi output' + os.linesep.encode('ascii'))

    def test_protectedServerAndDate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the CGI script emits a I{Server} or I{Date} header, these are\n        ignored.\n        '
        cgiFilename = self.writeCGI(SPECIAL_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)

        def checkResponse(response):
            if False:
                for i in range(10):
                    print('nop')
            self.assertNotIn('monkeys', response.headers.getRawHeaders('server'))
            self.assertNotIn('last year', response.headers.getRawHeaders('date'))
        d.addCallback(checkResponse)
        return d

    def test_noDuplicateContentTypeHeaders(self):
        if False:
            while True:
                i = 10
        "\n        If the CGI script emits a I{content-type} header, make sure that the\n        server doesn't add an additional (duplicate) one, as per ticket 4786.\n        "
        cgiFilename = self.writeCGI(NO_DUPLICATE_CONTENT_TYPE_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)

        def checkResponse(response):
            if False:
                print('Hello World!')
            self.assertEqual(response.headers.getRawHeaders('content-type'), ['text/cgi-duplicate-test'])
            return response
        d.addCallback(checkResponse)
        return d

    def test_noProxyPassthrough(self):
        if False:
            return 10
        '\n        The CGI script is never called with the Proxy header passed through.\n        '
        cgiFilename = self.writeCGI(HEADER_OUTPUT_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        headers = http_headers.Headers({b'Proxy': [b'foo'], b'X-Innocent-Header': [b'bar']})
        d = agent.request(b'GET', url, headers=headers)

        def checkResponse(response):
            if False:
                for i in range(10):
                    print('nop')
            headers = json.loads(response.decode('ascii'))
            self.assertEqual(set(headers.keys()), {'HTTP_HOST', 'HTTP_CONNECTION', 'HTTP_X_INNOCENT_HEADER'})
        d.addCallback(client.readBody)
        d.addCallback(checkResponse)
        return d

    def test_duplicateHeaderCGI(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If a CGI script emits two instances of the same header, both are sent\n        in the response.\n        '
        cgiFilename = self.writeCGI(DUAL_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)

        def checkResponse(response):
            if False:
                print('Hello World!')
            self.assertEqual(response.headers.getRawHeaders('header'), ['spam', 'eggs'])
        d.addCallback(checkResponse)
        return d

    def test_malformedHeaderCGI(self):
        if False:
            i = 10
            return i + 15
        '\n        Check for the error message in the duplicated header\n        '
        cgiFilename = self.writeCGI(BROKEN_HEADER_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(discardBody)
        loggedMessages = []

        def addMessage(eventDict):
            if False:
                for i in range(10):
                    print('nop')
            loggedMessages.append(log.textFromEventDict(eventDict))
        log.addObserver(addMessage)
        self.addCleanup(log.removeObserver, addMessage)

        def checkResponse(ignored):
            if False:
                for i in range(10):
                    print('nop')
            self.assertIn('ignoring malformed CGI header: ' + repr(b'XYZ'), loggedMessages)
        d.addCallback(checkResponse)
        return d

    def test_ReadEmptyInput(self):
        if False:
            return 10
        cgiFilename = os.path.abspath(self.mktemp())
        with open(cgiFilename, 'wt') as cgiFile:
            cgiFile.write(READINPUT_CGI)
        portnum = self.startServer(cgiFilename)
        agent = client.Agent(reactor)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = agent.request(b'GET', url)
        d.addCallback(client.readBody)
        d.addCallback(self._test_ReadEmptyInput_1)
        return d
    test_ReadEmptyInput.timeout = 5

    def _test_ReadEmptyInput_1(self, res):
        if False:
            i = 10
            return i + 15
        expected = f'readinput ok{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_ReadInput(self):
        if False:
            print('Hello World!')
        cgiFilename = os.path.abspath(self.mktemp())
        with open(cgiFilename, 'wt') as cgiFile:
            cgiFile.write(READINPUT_CGI)
        portnum = self.startServer(cgiFilename)
        agent = client.Agent(reactor)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = agent.request(uri=url, method=b'POST', bodyProducer=client.FileBodyProducer(BytesIO(b'Here is your stdin')))
        d.addCallback(client.readBody)
        d.addCallback(self._test_ReadInput_1)
        return d
    test_ReadInput.timeout = 5

    def _test_ReadInput_1(self, res):
        if False:
            i = 10
            return i + 15
        expected = f'readinput ok{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_ReadAllInput(self):
        if False:
            i = 10
            return i + 15
        cgiFilename = os.path.abspath(self.mktemp())
        with open(cgiFilename, 'wt') as cgiFile:
            cgiFile.write(READALLINPUT_CGI)
        portnum = self.startServer(cgiFilename)
        url = 'http://localhost:%d/cgi' % (portnum,)
        url = url.encode('ascii')
        d = client.Agent(reactor).request(uri=url, method=b'POST', bodyProducer=client.FileBodyProducer(BytesIO(b'Here is your stdin')))
        d.addCallback(client.readBody)
        d.addCallback(self._test_ReadAllInput_1)
        return d
    test_ReadAllInput.timeout = 5

    def _test_ReadAllInput_1(self, res):
        if False:
            return 10
        expected = f'readallinput ok{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_useReactorArgument(self):
        if False:
            return 10
        '\n        L{twcgi.FilteredScript.runProcess} uses the reactor passed as an\n        argument to the constructor.\n        '

        class FakeReactor:
            """
            A fake reactor recording whether spawnProcess is called.
            """
            called = False

            def spawnProcess(self, *args, **kwargs):
                if False:
                    while True:
                        i = 10
                '\n                Set the C{called} flag to C{True} if C{spawnProcess} is called.\n\n                @param args: Positional arguments.\n                @param kwargs: Keyword arguments.\n                '
                self.called = True
        fakeReactor = FakeReactor()
        request = DummyRequest(['a', 'b'])
        request.client = address.IPv4Address('TCP', '127.0.0.1', 12345)
        resource = twcgi.FilteredScript('dummy-file', reactor=fakeReactor)
        _render(resource, request)
        self.assertTrue(fakeReactor.called)

class CGIScriptTests(_StartServerAndTearDownMixin, unittest.TestCase):
    """
    Tests for L{twcgi.CGIScript}.
    """

    def test_urlParameters(self):
        if False:
            print('Hello World!')
        '\n        If the CGI script is passed URL parameters, do not fall over,\n        as per ticket 9887.\n        '
        cgiFilename = self.writeCGI(URL_PARAMETER_CGI)
        portnum = self.startServer(cgiFilename)
        url = b'http://localhost:%d/cgi?param=1234' % (portnum,)
        agent = client.Agent(reactor)
        d = agent.request(b'GET', url)
        d.addCallback(client.readBody)
        d.addCallback(self._test_urlParameters_1)
        return d

    def _test_urlParameters_1(self, res):
        if False:
            for i in range(10):
                print('nop')
        expected = f'param=1234{os.linesep}'
        expected = expected.encode('ascii')
        self.assertEqual(res, expected)

    def test_pathInfo(self):
        if False:
            return 10
        '\n        L{twcgi.CGIScript.render} sets the process environment\n        I{PATH_INFO} from the request path.\n        '

        class FakeReactor:
            """
            A fake reactor recording the environment passed to spawnProcess.
            """

            def spawnProcess(self, process, filename, args, env, wdir):
                if False:
                    return 10
                '\n                Store the C{env} L{dict} to an instance attribute.\n\n                @param process: Ignored\n                @param filename: Ignored\n                @param args: Ignored\n                @param env: The environment L{dict} which will be stored\n                @param wdir: Ignored\n                '
                self.process_env = env
        _reactor = FakeReactor()
        resource = twcgi.CGIScript(self.mktemp(), reactor=_reactor)
        request = DummyRequest(['a', 'b'])
        request.client = address.IPv4Address('TCP', '127.0.0.1', 12345)
        _render(resource, request)
        self.assertEqual(_reactor.process_env['PATH_INFO'], '/a/b')

class CGIDirectoryTests(unittest.TestCase):
    """
    Tests for L{twcgi.CGIDirectory}.
    """

    def test_render(self):
        if False:
            while True:
                i = 10
        '\n        L{twcgi.CGIDirectory.render} sets the HTTP response code to I{NOT\n        FOUND}.\n        '
        resource = twcgi.CGIDirectory(self.mktemp())
        request = DummyRequest([''])
        d = _render(resource, request)

        def cbRendered(ignored):
            if False:
                print('Hello World!')
            self.assertEqual(request.responseCode, NOT_FOUND)
        d.addCallback(cbRendered)
        return d

    def test_notFoundChild(self):
        if False:
            i = 10
            return i + 15
        '\n        L{twcgi.CGIDirectory.getChild} returns a resource which renders an\n        response with the HTTP I{NOT FOUND} status code if the indicated child\n        does not exist as an entry in the directory used to initialized the\n        L{twcgi.CGIDirectory}.\n        '
        path = self.mktemp()
        os.makedirs(path)
        resource = twcgi.CGIDirectory(path)
        request = DummyRequest(['foo'])
        child = resource.getChild('foo', request)
        d = _render(child, request)

        def cbRendered(ignored):
            if False:
                while True:
                    i = 10
            self.assertEqual(request.responseCode, NOT_FOUND)
        d.addCallback(cbRendered)
        return d

class CGIProcessProtocolTests(unittest.TestCase):
    """
    Tests for L{twcgi.CGIProcessProtocol}.
    """

    def test_prematureEndOfHeaders(self):
        if False:
            print('Hello World!')
        '\n        If the process communicating with L{CGIProcessProtocol} ends before\n        finishing writing out headers, the response has I{INTERNAL SERVER\n        ERROR} as its status code.\n        '
        request = DummyRequest([''])
        protocol = twcgi.CGIProcessProtocol(request)
        protocol.processEnded(failure.Failure(error.ProcessTerminated()))
        self.assertEqual(request.responseCode, INTERNAL_SERVER_ERROR)

    def test_connectionLost(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure that the CGI process ends cleanly when the request connection\n        is lost.\n        '
        d = DummyChannel()
        request = http.Request(d, True)
        protocol = twcgi.CGIProcessProtocol(request)
        request.connectionLost(failure.Failure(ConnectionLost('Connection done')))
        protocol.processEnded(failure.Failure(error.ProcessTerminated()))

def discardBody(response):
    if False:
        print('Hello World!')
    '\n    Discard the body of a HTTP response.\n\n    @param response: The response.\n\n    @return: The response.\n    '
    return client.readBody(response).addCallback(lambda _: response)