"""Modify HTTP query parameters."""
from mitmproxy import http

def request(flow: http.HTTPFlow) -> None:
    if False:
        for i in range(10):
            print('nop')
    flow.request.query['mitmproxy'] = 'rocks'