"""Redirect HTTP requests to another server."""
from mitmproxy import http

def request(flow: http.HTTPFlow) -> None:
    if False:
        i = 10
        return i + 15
    if flow.request.pretty_host == 'example.org':
        flow.request.host = 'mitmproxy.org'