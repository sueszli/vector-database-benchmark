from urllib.parse import urlparse
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.serving import WSGIRequestHandler
from searx import settings

class ReverseProxyPathFix:
    """Wrap the application in this middleware and configure the
    front-end server to add these headers, to let you quietly bind
    this to a URL other than / and to an HTTP scheme that is
    different than what is used locally.

    http://flask.pocoo.org/snippets/35/

    In nginx:
    location /myprefix {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Scheme $scheme;
        proxy_set_header X-Script-Name /myprefix;
        }

    :param wsgi_app: the WSGI application
    """

    def __init__(self, wsgi_app):
        if False:
            while True:
                i = 10
        self.wsgi_app = wsgi_app
        self.script_name = None
        self.scheme = None
        self.server = None
        if settings['server']['base_url']:
            base_url = urlparse(settings['server']['base_url'])
            self.script_name = base_url.path
            if self.script_name.endswith('/'):
                self.script_name = self.script_name[:-1]
            self.scheme = base_url.scheme
            self.server = base_url.netloc

    def __call__(self, environ, start_response):
        if False:
            while True:
                i = 10
        script_name = self.script_name or environ.get('HTTP_X_SCRIPT_NAME', '')
        if script_name:
            environ['SCRIPT_NAME'] = script_name
            path_info = environ['PATH_INFO']
            if path_info.startswith(script_name):
                environ['PATH_INFO'] = path_info[len(script_name):]
        scheme = self.scheme or environ.get('HTTP_X_SCHEME', '')
        if scheme:
            environ['wsgi.url_scheme'] = scheme
        server = self.server or environ.get('HTTP_X_FORWARDED_HOST', '')
        if server:
            environ['HTTP_HOST'] = server
        return self.wsgi_app(environ, start_response)

def patch_application(app):
    if False:
        i = 10
        return i + 15
    WSGIRequestHandler.protocol_version = 'HTTP/{}'.format(settings['server']['http_protocol_version'])
    app.wsgi_app = ReverseProxyPathFix(ProxyFix(app.wsgi_app))