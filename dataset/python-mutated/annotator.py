"""
Defines a base class that can be used to annotate.
"""
import io
from multiprocessing import Process
from http.server import BaseHTTPRequestHandler, HTTPServer
from http import client as HTTPStatus
from stanza.protobuf import Document, parseFromDelimitedString, writeToDelimitedString

class Annotator(Process):
    """
    This annotator base class hosts a lightweight server that accepts
    annotation requests from CoreNLP.
    Each annotator simply defines 3 functions: requires, provides and annotate.

    This class takes care of defining appropriate endpoints to interface
    with CoreNLP.
    """

    @property
    def name(self):
        if False:
            return 10
        '\n        Name of the annotator (used by CoreNLP)\n        '
        raise NotImplementedError()

    @property
    def requires(self):
        if False:
            while True:
                i = 10
        '\n        Requires has to specify all the annotations required before we\n        are called.\n        '
        raise NotImplementedError()

    @property
    def provides(self):
        if False:
            i = 10
            return i + 15
        '\n        The set of annotations guaranteed to be provided when we are done.\n        NOTE: that these annotations are either fully qualified Java\n        class names or refer to nested classes of\n        edu.stanford.nlp.ling.CoreAnnotations (as is the case below).\n        '
        raise NotImplementedError()

    def annotate(self, ann):
        if False:
            print('Hello World!')
        '\n        @ann: is a protobuf annotation object.\n        Actually populate @ann with tokens.\n        '
        raise NotImplementedError()

    @property
    def properties(self):
        if False:
            return 10
        '\n        Defines a Java property to define this annotator to CoreNLP.\n        '
        return {'customAnnotatorClass.{}'.format(self.name): 'edu.stanford.nlp.pipeline.GenericWebServiceAnnotator', 'generic.endpoint': 'http://{}:{}'.format(self.host, self.port), 'generic.requires': ','.join(self.requires), 'generic.provides': ','.join(self.provides)}

    class _Handler(BaseHTTPRequestHandler):
        annotator = None

        def __init__(self, request, client_address, server):
            if False:
                for i in range(10):
                    print('nop')
            BaseHTTPRequestHandler.__init__(self, request, client_address, server)

        def do_GET(self):
            if False:
                while True:
                    i = 10
            '\n            Handle a ping request\n            '
            if not self.path.endswith('/'):
                self.path += '/'
            if self.path == '/ping/':
                msg = 'pong'.encode('UTF-8')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/application')
                self.send_header('Content-Length', len(msg))
                self.end_headers()
                self.wfile.write(msg)
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()

        def do_POST(self):
            if False:
                print('Hello World!')
            '\n            Handle an annotate request\n            '
            if not self.path.endswith('/'):
                self.path += '/'
            if self.path == '/annotate/':
                length = int(self.headers.get('content-length'))
                msg = self.rfile.read(length)
                doc = Document()
                parseFromDelimitedString(doc, msg)
                self.annotator.annotate(doc)
                with io.BytesIO() as stream:
                    writeToDelimitedString(doc, stream)
                    msg = stream.getvalue()
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'application/x-protobuf')
                self.send_header('Content-Length', len(msg))
                self.end_headers()
                self.wfile.write(msg)
            else:
                self.send_response(HTTPStatus.BAD_REQUEST)
                self.end_headers()

    def __init__(self, host='', port=8432):
        if False:
            print('Hello World!')
        '\n        Launches a server endpoint to communicate with CoreNLP\n        '
        Process.__init__(self)
        (self.host, self.port) = (host, port)
        self._Handler.annotator = self

    def run(self):
        if False:
            return 10
        "\n        Runs the server using Python's simple HTTPServer.\n        TODO: make this multithreaded.\n        "
        httpd = HTTPServer((self.host, self.port), self._Handler)
        sa = httpd.socket.getsockname()
        serve_message = 'Serving HTTP on {host} port {port} (http://{host}:{port}/) ...'
        print(serve_message.format(host=sa[0], port=sa[1]))
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print('\nKeyboard interrupt received, exiting.')
            httpd.shutdown()