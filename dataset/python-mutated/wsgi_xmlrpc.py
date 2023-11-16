from six.moves.xmlrpc_server import SimpleXMLRPCDispatcher
import logging
logger = logging.getLogger(__name__)

class WSGIXMLRPCApplication(object):
    """Application to handle requests to the XMLRPC service"""

    def __init__(self, instance=None, methods=None):
        if False:
            return 10
        'Create windmill xmlrpc dispatcher'
        if methods is None:
            methods = []
        try:
            self.dispatcher = SimpleXMLRPCDispatcher(allow_none=True, encoding=None)
        except TypeError:
            self.dispatcher = SimpleXMLRPCDispatcher()
        if instance is not None:
            self.dispatcher.register_instance(instance)
        for method in methods:
            self.dispatcher.register_function(method)
        self.dispatcher.register_introspection_functions()

    def register_instance(self, instance):
        if False:
            for i in range(10):
                print('nop')
        return self.dispatcher.register_instance(instance)

    def register_function(self, function, name=None):
        if False:
            while True:
                i = 10
        return self.dispatcher.register_function(function, name)

    def handler(self, environ, start_response):
        if False:
            while True:
                i = 10
        'XMLRPC service for windmill browser core to communicate with'
        if environ['REQUEST_METHOD'] == 'POST':
            return self.handle_POST(environ, start_response)
        else:
            start_response('400 Bad request', [('Content-Type', 'text/plain')])
            return ['']

    def handle_POST(self, environ, start_response):
        if False:
            while True:
                i = 10
        "Handles the HTTP POST request.\n\n        Attempts to interpret all HTTP POST requests as XML-RPC calls,\n        which are forwarded to the server's _dispatch method for handling.\n\n        Most code taken from SimpleXMLRPCServer with modifications for wsgi and my custom dispatcher.\n        "
        try:
            length = int(environ['CONTENT_LENGTH'])
            data = environ['wsgi.input'].read(length)
            response = self.dispatcher._marshaled_dispatch(data, getattr(self.dispatcher, '_dispatch', None))
            response += b'\n'
        except Exception as e:
            logger.exception(e)
            start_response('500 Server error', [('Content-Type', 'text/plain')])
            return []
        else:
            start_response('200 OK', [('Content-Type', 'text/xml'), ('Content-Length', str(len(response)))])
            return [response]

    def __call__(self, environ, start_response):
        if False:
            i = 10
            return i + 15
        return self.handler(environ, start_response)