import argparse
import base64
import sys
import textwrap
from urllib.parse import unquote
from pyramid.request import Request
from pyramid.scripts.common import get_config_loader, parse_vars

def main(argv=sys.argv, quiet=False):
    if False:
        return 10
    command = PRequestCommand(argv, quiet)
    return command.run()

class PRequestCommand:
    description = '    Submit a HTTP request to a web application.\n\n    This command makes an artificial request to a web application that uses a\n    PasteDeploy (.ini) configuration file for the server and application.\n\n    Use "prequest config.ini /path" to request "/path".\n\n    Use "prequest --method=POST config.ini /path < data" to do a POST with\n    the given request body.\n\n    Use "prequest --method=PUT config.ini /path < data" to do a\n    PUT with the given request body.\n\n    Use "prequest --method=PATCH config.ini /path < data" to do a\n    PATCH with the given request body.\n\n    Use "prequest --method=OPTIONS config.ini /path" to do an\n    OPTIONS request.\n\n    Use "prequest --method=PROPFIND config.ini /path" to do a\n    PROPFIND request.\n\n    If the path is relative (doesn\'t begin with "/") it is interpreted as\n    relative to "/".  The path passed to this script should be URL-quoted.\n    The path can be succeeded with a query string (e.g. \'/path?a=1&=b2\').\n\n    The variable "environ[\'paste.command_request\']" will be set to "True" in\n    the request\'s WSGI environment, so your application can distinguish these\n    calls from normal requests.\n    '
    parser = argparse.ArgumentParser(description=textwrap.dedent(description), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--app-name', dest='app_name', metavar='NAME', help="Load the named application from the config file (default 'main')")
    parser.add_argument('--header', dest='headers', metavar='NAME:VALUE', action='append', help='Header to add to request (you can use this option multiple times)')
    parser.add_argument('-d', '--display-headers', dest='display_headers', action='store_true', help='Display status and headers before the response body')
    parser.add_argument('-m', '--method', dest='method', choices=['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'PROPFIND', 'OPTIONS'], help='Request method type (GET, POST, PUT, PATCH, DELETE, PROPFIND, OPTIONS)')
    parser.add_argument('-l', '--login', dest='login', help='HTTP basic auth username:password pair')
    parser.add_argument('config_uri', nargs='?', default=None, help='The URI to the configuration file.')
    parser.add_argument('path_info', nargs='?', default=None, help='The path of the request.')
    parser.add_argument('config_vars', nargs='*', default=(), help='Variables required by the config file. For example, `http_port=%%(http_port)s` would expect `http_port=8080` to be passed here.')
    _get_config_loader = staticmethod(get_config_loader)
    stdin = sys.stdin

    def __init__(self, argv, quiet=False):
        if False:
            while True:
                i = 10
        self.quiet = quiet
        self.args = self.parser.parse_args(argv[1:])

    def out(self, msg):
        if False:
            while True:
                i = 10
        if not self.quiet:
            print(msg)

    def run(self):
        if False:
            i = 10
            return i + 15
        if not self.args.config_uri or not self.args.path_info:
            self.out('You must provide at least two arguments')
            return 2
        config_uri = self.args.config_uri
        config_vars = parse_vars(self.args.config_vars)
        path = self.args.path_info
        loader = self._get_config_loader(config_uri)
        loader.setup_logging(config_vars)
        app = loader.get_wsgi_app(self.args.app_name, config_vars)
        if not path.startswith('/'):
            path = '/' + path
        try:
            (path, qs) = path.split('?', 1)
        except ValueError:
            qs = ''
        path = unquote(path)
        headers = {}
        if self.args.login:
            enc = base64.b64encode(self.args.login.encode('ascii'))
            headers['Authorization'] = 'Basic ' + enc.decode('ascii')
        if self.args.headers:
            for item in self.args.headers:
                if ':' not in item:
                    self.out("Bad --header=%s option, value must be in the form 'name:value'" % item)
                    return 2
                (name, value) = item.split(':', 1)
                headers[name] = value.strip()
        request_method = (self.args.method or 'GET').upper()
        environ = {'REQUEST_METHOD': request_method, 'SCRIPT_NAME': '', 'PATH_INFO': path, 'SERVER_NAME': 'localhost', 'SERVER_PORT': '80', 'SERVER_PROTOCOL': 'HTTP/1.0', 'CONTENT_TYPE': 'text/plain', 'REMOTE_ADDR': '127.0.0.1', 'wsgi.run_once': True, 'wsgi.multithread': False, 'wsgi.multiprocess': False, 'wsgi.errors': sys.stderr, 'wsgi.url_scheme': 'http', 'wsgi.version': (1, 0), 'QUERY_STRING': qs, 'HTTP_ACCEPT': 'text/plain;q=1.0, */*;q=0.1', 'paste.command_request': True}
        if request_method in ('POST', 'PUT', 'PATCH'):
            environ['wsgi.input'] = self.stdin
            environ['CONTENT_LENGTH'] = '-1'
        for (name, value) in headers.items():
            if name.lower() == 'content-type':
                name = 'CONTENT_TYPE'
            else:
                name = 'HTTP_' + name.upper().replace('-', '_')
            environ[name] = value
        request = Request.blank(path, environ=environ)
        response = request.get_response(app)
        if self.args.display_headers:
            self.out(response.status)
            for (name, value) in response.headerlist:
                self.out(f'{name}: {value}')
        if response.charset:
            self.out(response.ubody)
        else:
            self.out(response.body)
        return 0
if __name__ == '__main__':
    sys.exit(main() or 0)