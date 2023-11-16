from gevent import pywsgi, spawn

def http_to_https(target_url):
    if False:
        return 10

    def app(environ, start_response):
        if False:
            print('Hello World!')
        if environ['PATH_INFO'] == '/':
            start_response('301 Move Permanently', [('Location', target_url), ('Connection', 'close'), ('Cache-control', 'private')])
            return [b"\n    <html>\n    <body>\n        Migrating to HTTPS protocol.\n        Please go to <a href='{self.target}'>{self.target}</a> to use Ajenti.\n    </body>\n    </html>"]
        start_response('404 Not Found', [('Content-Type', 'text/html')])
        return [b'<h1>Not Found</h1>']
    pywsgi.WSGIServer(('', 80), application=app).serve_forever()