"""
This script simply prints all received HTTP Trailers.

HTTP requests and responses can contain trailing headers which are sent after
the body is fully transmitted. Such trailers need to be announced in the initial
headers by name, so the receiving endpoint can wait and read them after the
body.
"""
from mitmproxy import http
from mitmproxy.http import Headers

def request(flow: http.HTTPFlow):
    if False:
        for i in range(10):
            print('nop')
    if flow.request.trailers:
        print('HTTP Trailers detected! Request contains:', flow.request.trailers)
    if flow.request.path == '/inject_trailers':
        if flow.request.is_http10:
            return
        elif flow.request.is_http11:
            if not flow.request.content:
                return
            flow.request.headers['transfer-encoding'] = 'chunked'
        flow.request.headers['trailer'] = 'x-my-injected-trailer-header'
        flow.request.trailers = Headers([(b'x-my-injected-trailer-header', b'foobar')])
        print('Injected a new request trailer...', flow.request.headers['trailer'])

def response(flow: http.HTTPFlow):
    if False:
        for i in range(10):
            print('nop')
    assert flow.response
    if flow.response.trailers:
        print('HTTP Trailers detected! Response contains:', flow.response.trailers)
    if flow.request.path == '/inject_trailers':
        if flow.request.is_http10:
            return
        elif flow.request.is_http11:
            if not flow.response.content:
                return
            flow.response.headers['transfer-encoding'] = 'chunked'
        flow.response.headers['trailer'] = 'x-my-injected-trailer-header'
        flow.response.trailers = Headers([(b'x-my-injected-trailer-header', b'foobar')])
        print('Injected a new response trailer...', flow.response.headers['trailer'])