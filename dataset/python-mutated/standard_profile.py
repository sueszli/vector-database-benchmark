import socketserver
import sys
import traceback
import warnings
import xmlrpc.client as xmlrpc
from xmlrpc.server import SimpleXMLRPCRequestHandler, SimpleXMLRPCServer
from .constants import SAMP_ICON
from .errors import SAMPWarning
__all__ = []

class SAMPSimpleXMLRPCRequestHandler(SimpleXMLRPCRequestHandler):
    """
    XMLRPC handler of Standard Profile requests.
    """

    def do_GET(self):
        if False:
            print('Hello World!')
        if self.path == '/samp/icon':
            self.send_response(200, 'OK')
            self.send_header('Content-Type', 'image/png')
            self.end_headers()
            self.wfile.write(SAMP_ICON)

    def do_POST(self):
        if False:
            print('Hello World!')
        "\n        Handles the HTTP POST request.\n\n        Attempts to interpret all HTTP POST requests as XML-RPC calls,\n        which are forwarded to the server's ``_dispatch`` method for\n        handling.\n        "
        if not self.is_rpc_path_valid():
            self.report_404()
            return
        try:
            max_chunk_size = 10 * 1024 * 1024
            size_remaining = int(self.headers['content-length'])
            L = []
            while size_remaining:
                chunk_size = min(size_remaining, max_chunk_size)
                L.append(self.rfile.read(chunk_size))
                size_remaining -= len(L[-1])
            data = b''.join(L)
            (params, method) = xmlrpc.loads(data)
            if method == 'samp.webhub.register':
                params = list(params)
                params.append(self.client_address)
                if 'Origin' in self.headers:
                    params.append(self.headers.get('Origin'))
                else:
                    params.append('unknown')
                params = tuple(params)
                data = xmlrpc.dumps(params, methodname=method)
            elif method in ('samp.hub.notify', 'samp.hub.notifyAll', 'samp.hub.call', 'samp.hub.callAll', 'samp.hub.callAndWait'):
                user = 'unknown'
                if method == 'samp.hub.callAndWait':
                    params[2]['host'] = self.address_string()
                    params[2]['user'] = user
                else:
                    params[-1]['host'] = self.address_string()
                    params[-1]['user'] = user
                data = xmlrpc.dumps(params, methodname=method)
            data = self.decode_request_content(data)
            if data is None:
                return
            response = self.server._marshaled_dispatch(data, getattr(self, '_dispatch', None), self.path)
        except Exception as e:
            self.send_response(500)
            if hasattr(self.server, '_send_traceback_header') and self.server._send_traceback_header:
                self.send_header('X-exception', str(e))
                trace = traceback.format_exc()
                trace = str(trace.encode('ASCII', 'backslashreplace'), 'ASCII')
                self.send_header('X-traceback', trace)
            self.send_header('Content-length', '0')
            self.end_headers()
        else:
            self.send_response(200)
            self.send_header('Content-type', 'text/xml')
            if self.encode_threshold is not None:
                if len(response) > self.encode_threshold:
                    q = self.accept_encodings().get('gzip', 0)
                    if q:
                        try:
                            response = xmlrpc.gzip_encode(response)
                            self.send_header('Content-Encoding', 'gzip')
                        except NotImplementedError:
                            pass
            self.send_header('Content-length', str(len(response)))
            self.end_headers()
            self.wfile.write(response)

class ThreadingXMLRPCServer(socketserver.ThreadingMixIn, SimpleXMLRPCServer):
    """
    Asynchronous multithreaded XMLRPC server.
    """

    def __init__(self, addr, log=None, requestHandler=SAMPSimpleXMLRPCRequestHandler, logRequests=True, allow_none=True, encoding=None):
        if False:
            print('Hello World!')
        self.log = log
        SimpleXMLRPCServer.__init__(self, addr, requestHandler, logRequests, allow_none, encoding)

    def handle_error(self, request, client_address):
        if False:
            while True:
                i = 10
        if self.log is None:
            socketserver.BaseServer.handle_error(self, request, client_address)
        else:
            warnings.warn('Exception happened during processing of request from {}: {}'.format(client_address, sys.exc_info()[1]), SAMPWarning)