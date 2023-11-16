from urllib.parse import parse_qs
from urllib.request import urlopen
from astropy.utils.data import get_pkg_data_contents
from .standard_profile import SAMPSimpleXMLRPCRequestHandler, ThreadingXMLRPCServer
__all__ = []
CROSS_DOMAIN = get_pkg_data_contents('data/crossdomain.xml')
CLIENT_ACCESS_POLICY = get_pkg_data_contents('data/clientaccesspolicy.xml')

class WebProfileRequestHandler(SAMPSimpleXMLRPCRequestHandler):
    """
    Handler of XMLRPC requests performed through the Web Profile.
    """

    def _send_CORS_header(self):
        if False:
            while True:
                i = 10
        if self.headers.get('Origin') is not None:
            method = self.headers.get('Access-Control-Request-Method')
            if method and self.command == 'OPTIONS':
                self.send_header('Content-Length', '0')
                self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin'))
                self.send_header('Access-Control-Allow-Methods', method)
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Access-Control-Allow-Credentials', 'true')
            else:
                self.send_header('Access-Control-Allow-Origin', self.headers.get('Origin'))
                self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                self.send_header('Access-Control-Allow-Credentials', 'true')

    def end_headers(self):
        if False:
            print('Hello World!')
        self._send_CORS_header()
        SAMPSimpleXMLRPCRequestHandler.end_headers(self)

    def _serve_cross_domain_xml(self):
        if False:
            i = 10
            return i + 15
        cross_domain = False
        if self.path == '/crossdomain.xml':
            response = CROSS_DOMAIN
            self.send_response(200, 'OK')
            self.send_header('Content-Type', 'text/x-cross-domain-policy')
            self.send_header('Content-Length', f'{len(response)}')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
            self.wfile.flush()
            cross_domain = True
        elif self.path == '/clientaccesspolicy.xml':
            response = CLIENT_ACCESS_POLICY
            self.send_response(200, 'OK')
            self.send_header('Content-Type', 'text/xml')
            self.send_header('Content-Length', f'{len(response)}')
            self.end_headers()
            self.wfile.write(response.encode('utf-8'))
            self.wfile.flush()
            cross_domain = True
        return cross_domain

    def do_POST(self):
        if False:
            i = 10
            return i + 15
        if self._serve_cross_domain_xml():
            return
        return SAMPSimpleXMLRPCRequestHandler.do_POST(self)

    def do_HEAD(self):
        if False:
            return 10
        if not self.is_http_path_valid():
            self.report_404()
            return
        if self._serve_cross_domain_xml():
            return

    def do_OPTIONS(self):
        if False:
            while True:
                i = 10
        self.send_response(200, 'OK')
        self.end_headers()

    def do_GET(self):
        if False:
            i = 10
            return i + 15
        if not self.is_http_path_valid():
            self.report_404()
            return
        split_path = self.path.split('?')
        if split_path[0] in [f'/translator/{clid}' for clid in self.server.clients]:
            urlpath = parse_qs(split_path[1])
            try:
                proxyfile = urlopen(urlpath['ref'][0])
                self.send_response(200, 'OK')
                self.end_headers()
                self.wfile.write(proxyfile.read())
                proxyfile.close()
            except OSError:
                self.report_404()
                return
        if self._serve_cross_domain_xml():
            return

    def is_http_path_valid(self):
        if False:
            for i in range(10):
                print('nop')
        valid_paths = ['/clientaccesspolicy.xml', '/crossdomain.xml'] + [f'/translator/{clid}' for clid in self.server.clients]
        return self.path.split('?')[0] in valid_paths

class WebProfileXMLRPCServer(ThreadingXMLRPCServer):
    """
    XMLRPC server supporting the SAMP Web Profile.
    """

    def __init__(self, addr, log=None, requestHandler=WebProfileRequestHandler, logRequests=True, allow_none=True, encoding=None):
        if False:
            i = 10
            return i + 15
        self.clients = []
        ThreadingXMLRPCServer.__init__(self, addr, log, requestHandler, logRequests, allow_none, encoding)

    def add_client(self, client_id):
        if False:
            return 10
        self.clients.append(client_id)

    def remove_client(self, client_id):
        if False:
            i = 10
            return i + 15
        try:
            self.clients.remove(client_id)
        except ValueError:
            pass

def web_profile_text_dialog(request, queue):
    if False:
        while True:
            i = 10
    samp_name = 'unknown'
    if isinstance(request[0], str):
        samp_name = request[0]
    else:
        samp_name = request[0]['samp.name']
    text = f'A Web application which declares to be\n\nName: {samp_name}\nOrigin: {request[2]}\n\nis requesting to be registered with the SAMP Hub.\nPay attention that if you permit its registration, such\napplication will acquire all current user privileges, like\nfile read/write.\n\nDo you give your consent? [yes|no]'
    print(text)
    answer = input('>>> ')
    queue.put(answer.lower() in ['yes', 'y'])