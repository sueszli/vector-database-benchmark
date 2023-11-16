"""Secure WSGI server example based on gevent.pywsgi"""
from __future__ import print_function
from gevent import pywsgi

def hello_world(env, start_response):
    if False:
        for i in range(10):
            print('nop')
    if env['PATH_INFO'] == '/':
        start_response('200 OK', [('Content-Type', 'text/html')])
        return [b'<b>hello world</b>']
    start_response('404 Not Found', [('Content-Type', 'text/html')])
    return [b'<h1>Not Found</h1>']
print('Serving on https://:8443')
server = pywsgi.WSGIServer(('127.0.0.1', 8443), hello_world, keyfile='server.key', certfile='server.crt')
server.serve_forever()