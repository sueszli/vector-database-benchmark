"""Send a reply from the proxy without sending the request to the remote server."""
from mitmproxy import http

def request(flow: http.HTTPFlow) -> None:
    if False:
        print('Hello World!')
    if flow.request.pretty_url == 'http://example.com/path':
        flow.response = http.Response.make(200, b'Hello World', {'Content-Type': 'text/html'})