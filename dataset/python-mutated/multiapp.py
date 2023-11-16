try:
    from routes import Mapper
except ImportError:
    print('This example requires Routes to be installed')
from test import app as app1
from test import app as app2

class Application(object):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.map = Mapper()
        self.map.connect('app1', '/app1url', app=app1)
        self.map.connect('app2', '/app2url', app=app2)

    def __call__(self, environ, start_response):
        if False:
            for i in range(10):
                print('nop')
        match = self.map.routematch(environ=environ)
        if not match:
            return self.error404(environ, start_response)
        return match[0]['app'](environ, start_response)

    def error404(self, environ, start_response):
        if False:
            while True:
                i = 10
        html = b'        <html>\n          <head>\n            <title>404 - Not Found</title>\n          </head>\n          <body>\n            <h1>404 - Not Found</h1>\n          </body>\n        </html>\n        '
        headers = [('Content-Type', 'text/html'), ('Content-Length', str(len(html)))]
        start_response('404 Not Found', headers)
        return [html]
app = Application()