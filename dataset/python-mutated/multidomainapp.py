import re

class SubDomainApp:
    """WSGI application to delegate requests based on domain name.
"""

    def __init__(self, mapping):
        if False:
            i = 10
            return i + 15
        self.mapping = mapping

    def __call__(self, environ, start_response):
        if False:
            print('Hello World!')
        host = environ.get('HTTP_HOST', '')
        host = host.split(':')[0]
        for (pattern, app) in self.mapping:
            if re.match('^' + pattern + '$', host):
                return app(environ, start_response)
        else:
            start_response('404 Not Found', [])
            return [b'']

def hello(environ, start_response):
    if False:
        print('Hello World!')
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'Hello, world\n']

def bye(environ, start_response):
    if False:
        while True:
            i = 10
    start_response('200 OK', [('Content-Type', 'text/plain')])
    return [b'Goodbye!\n']
app = SubDomainApp([('localhost', hello), ('.*', bye)])