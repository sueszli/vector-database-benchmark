"""Modify an HTTP form submission."""
from mitmproxy import http

def request(flow: http.HTTPFlow) -> None:
    if False:
        i = 10
        return i + 15
    if flow.request.urlencoded_form:
        flow.request.urlencoded_form['mitmproxy'] = 'rocks'
    else:
        flow.request.urlencoded_form = [('foo', 'bar')]