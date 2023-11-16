from xmlrpc.server import DocXMLRPCServer
import http.client
import re
import sys
import threading
import unittest

def make_request_and_skipIf(condition, reason):
    if False:
        return 10
    if not condition:
        return lambda func: func

    def decorator(func):
        if False:
            print('Hello World!')

        def make_request_and_skip(self):
            if False:
                print('Hello World!')
            self.client.request('GET', '/')
            self.client.getresponse()
            raise unittest.SkipTest(reason)
        return make_request_and_skip
    return decorator

def make_server():
    if False:
        print('Hello World!')
    serv = DocXMLRPCServer(('localhost', 0), logRequests=False)
    try:
        serv.set_server_title('DocXMLRPCServer Test Documentation')
        serv.set_server_name('DocXMLRPCServer Test Docs')
        serv.set_server_documentation("This is an XML-RPC server's documentation, but the server can be used by POSTing to /RPC2. Try self.add, too.")

        class TestClass(object):

            def test_method(self, arg):
                if False:
                    i = 10
                    return i + 15
                "Test method's docs. This method truly does very little."
                self.arg = arg
        serv.register_introspection_functions()
        serv.register_instance(TestClass())

        def add(x, y):
            if False:
                for i in range(10):
                    print('nop')
            'Add two instances together. This follows PEP008, but has nothing\n            to do with RFC1952. Case should matter: pEp008 and rFC1952.  Things\n            that start with http and ftp should be auto-linked, too:\n            http://google.com.\n            '
            return x + y

        def annotation(x: int):
            if False:
                i = 10
                return i + 15
            ' Use function annotations. '
            return x

        class ClassWithAnnotation:

            def method_annotation(self, x: bytes):
                if False:
                    while True:
                        i = 10
                return x.decode()
        serv.register_function(add)
        serv.register_function(lambda x, y: x - y)
        serv.register_function(annotation)
        serv.register_instance(ClassWithAnnotation())
        return serv
    except:
        serv.server_close()
        raise

class DocXMLRPCHTTPGETServer(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        DocXMLRPCServer._send_traceback_header = True
        self.serv = make_server()
        self.thread = threading.Thread(target=self.serv.serve_forever)
        self.thread.start()
        PORT = self.serv.server_address[1]
        self.client = http.client.HTTPConnection('localhost:%d' % PORT)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.close()
        DocXMLRPCServer._send_traceback_header = False
        self.serv.shutdown()
        self.thread.join()
        self.serv.server_close()

    def test_valid_get_response(self):
        if False:
            for i in range(10):
                print('nop')
        self.client.request('GET', '/')
        response = self.client.getresponse()
        self.assertEqual(response.status, 200)
        self.assertEqual(response.getheader('Content-type'), 'text/html')
        response.read()

    def test_invalid_get_response(self):
        if False:
            i = 10
            return i + 15
        self.client.request('GET', '/spam')
        response = self.client.getresponse()
        self.assertEqual(response.status, 404)
        self.assertEqual(response.getheader('Content-type'), 'text/plain')
        response.read()

    def test_lambda(self):
        if False:
            i = 10
            return i + 15
        'Test that lambda functionality stays the same.  The output produced\n        currently is, I suspect invalid because of the unencoded brackets in the\n        HTML, "<lambda>".\n\n        The subtraction lambda method is tested.\n        '
        self.client.request('GET', '/')
        response = self.client.getresponse()
        self.assertIn(b'<dl><dt><a name="-&lt;lambda&gt;"><strong>&lt;lambda&gt;</strong></a>(x, y)</dt></dl>', response.read())

    @make_request_and_skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_autolinking(self):
        if False:
            for i in range(10):
                print('nop')
        'Test that the server correctly automatically wraps references to\n        PEPS and RFCs with links, and that it linkifies text starting with\n        http or ftp protocol prefixes.\n\n        The documentation for the "add" method contains the test material.\n        '
        self.client.request('GET', '/')
        response = self.client.getresponse().read()
        self.assertIn(b'<dl><dt><a name="-add"><strong>add</strong></a>(x, y)</dt><dd><tt>Add&nbsp;two&nbsp;instances&nbsp;together.&nbsp;This&nbsp;follows&nbsp;<a href="https://www.python.org/dev/peps/pep-0008/">PEP008</a>,&nbsp;but&nbsp;has&nbsp;nothing<br>\nto&nbsp;do&nbsp;with&nbsp;<a href="http://www.rfc-editor.org/rfc/rfc1952.txt">RFC1952</a>.&nbsp;Case&nbsp;should&nbsp;matter:&nbsp;pEp008&nbsp;and&nbsp;rFC1952.&nbsp;&nbsp;Things<br>\nthat&nbsp;start&nbsp;with&nbsp;http&nbsp;and&nbsp;ftp&nbsp;should&nbsp;be&nbsp;auto-linked,&nbsp;too:<br>\n<a href="http://google.com">http://google.com</a>.</tt></dd></dl>', response)

    @make_request_and_skipIf(sys.flags.optimize >= 2, 'Docstrings are omitted with -O2 and above')
    def test_system_methods(self):
        if False:
            return 10
        'Test the presence of three consecutive system.* methods.\n\n        This also tests their use of parameter type recognition and the\n        systems related to that process.\n        '
        self.client.request('GET', '/')
        response = self.client.getresponse().read()
        self.assertIn(b'<dl><dt><a name="-system.methodHelp"><strong>system.methodHelp</strong></a>(method_name)</dt><dd><tt><a href="#-system.methodHelp">system.methodHelp</a>(\'add\')&nbsp;=&gt;&nbsp;"Adds&nbsp;two&nbsp;integers&nbsp;together"<br>\n&nbsp;<br>\nReturns&nbsp;a&nbsp;string&nbsp;containing&nbsp;documentation&nbsp;for&nbsp;the&nbsp;specified&nbsp;method.</tt></dd></dl>\n<dl><dt><a name="-system.methodSignature"><strong>system.methodSignature</strong></a>(method_name)</dt><dd><tt><a href="#-system.methodSignature">system.methodSignature</a>(\'add\')&nbsp;=&gt;&nbsp;[double,&nbsp;int,&nbsp;int]<br>\n&nbsp;<br>\nReturns&nbsp;a&nbsp;list&nbsp;describing&nbsp;the&nbsp;signature&nbsp;of&nbsp;the&nbsp;method.&nbsp;In&nbsp;the<br>\nabove&nbsp;example,&nbsp;the&nbsp;add&nbsp;method&nbsp;takes&nbsp;two&nbsp;integers&nbsp;as&nbsp;arguments<br>\nand&nbsp;returns&nbsp;a&nbsp;double&nbsp;result.<br>\n&nbsp;<br>\nThis&nbsp;server&nbsp;does&nbsp;NOT&nbsp;support&nbsp;system.methodSignature.</tt></dd></dl>', response)

    def test_autolink_dotted_methods(self):
        if False:
            while True:
                i = 10
        'Test that selfdot values are made strong automatically in the\n        documentation.'
        self.client.request('GET', '/')
        response = self.client.getresponse()
        self.assertIn(b'Try&nbsp;self.<strong>add</strong>,&nbsp;too.', response.read())

    def test_annotations(self):
        if False:
            print('Hello World!')
        ' Test that annotations works as expected '
        self.client.request('GET', '/')
        response = self.client.getresponse()
        docstring = b'' if sys.flags.optimize >= 2 else b'<dd><tt>Use&nbsp;function&nbsp;annotations.</tt></dd>'
        self.assertIn(b'<dl><dt><a name="-annotation"><strong>annotation</strong></a>(x: int)</dt>' + docstring + b'</dl>\n<dl><dt><a name="-method_annotation"><strong>method_annotation</strong></a>(x: bytes)</dt></dl>', response.read())

    def test_server_title_escape(self):
        if False:
            i = 10
            return i + 15
        self.serv.set_server_title('test_title<script>')
        self.serv.set_server_documentation('test_documentation<script>')
        self.assertEqual('test_title<script>', self.serv.server_title)
        self.assertEqual('test_documentation<script>', self.serv.server_documentation)
        generated = self.serv.generate_html_documentation()
        title = re.search('<title>(.+?)</title>', generated).group()
        documentation = re.search('<p><tt>(.+?)</tt></p>', generated).group()
        self.assertEqual('<title>Python: test_title&lt;script&gt;</title>', title)
        self.assertEqual('<p><tt>test_documentation&lt;script&gt;</tt></p>', documentation)
if __name__ == '__main__':
    unittest.main()