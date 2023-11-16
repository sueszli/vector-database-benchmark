"""Streaming HTTP uploads module.

This module extends the standard httplib and urllib2 objects so that
iterable objects can be used in the body of HTTP requests.

In most cases all one should have to do is call :func:`register_openers()`
to register the new streaming http handlers which will take priority over
the default handlers, and then you can use iterable objects in the body
of HTTP requests.

**N.B.** You must specify a Content-Length header if using an iterable object
since there is no way to determine in advance the total size that will be
yielded, and there is no way to reset an interator.

Example usage:

>>> from StringIO import StringIO
>>> import urllib2, poster.streaminghttp

>>> opener = poster.streaminghttp.register_openers()

>>> s = "Test file data"
>>> f = StringIO(s)

>>> req = urllib2.Request("http://localhost:5000", f,
...                       {'Content-Length': str(len(s))})
"""
import socket
import sys
from cloudinary.compat import NotConnected, httplib, urllib2
__all__ = ['StreamingHTTPConnection', 'StreamingHTTPRedirectHandler', 'StreamingHTTPHandler', 'register_openers']
if hasattr(httplib, 'HTTPS'):
    __all__.extend(['StreamingHTTPSHandler', 'StreamingHTTPSConnection'])

class _StreamingHTTPMixin:
    """Mixin class for HTTP and HTTPS connections that implements a streaming
    send method."""

    def send(self, value):
        if False:
            print('Hello World!')
        'Send ``value`` to the server.\n\n        ``value`` can be a string object, a file-like object that supports\n        a .read() method, or an iterable object that supports a .next()\n        method.\n        '
        if self.sock is None:
            if self.auto_open:
                self.connect()
            else:
                raise NotConnected()
        if self.debuglevel > 0:
            print('send:', repr(value))
        try:
            blocksize = 8192
            if hasattr(value, 'read'):
                if hasattr(value, 'seek'):
                    value.seek(0)
                if self.debuglevel > 0:
                    print('sendIng a read()able')
                data = value.read(blocksize)
                while data:
                    self.sock.sendall(data)
                    data = value.read(blocksize)
            elif hasattr(value, 'next'):
                if hasattr(value, 'reset'):
                    value.reset()
                if self.debuglevel > 0:
                    print('sendIng an iterable')
                for data in value:
                    self.sock.sendall(data)
            else:
                self.sock.sendall(value)
        except socket.error:
            e = sys.exc_info()[1]
            if e[0] == 32:
                self.close()
            raise

class StreamingHTTPConnection(_StreamingHTTPMixin, httplib.HTTPConnection):
    """Subclass of `httplib.HTTPConnection` that overrides the `send()` method
    to support iterable body objects"""

class StreamingHTTPRedirectHandler(urllib2.HTTPRedirectHandler):
    """Subclass of `urllib2.HTTPRedirectHandler` that overrides the
    `redirect_request` method to properly handle redirected POST requests

    This class is required because python 2.5's HTTPRedirectHandler does
    not remove the Content-Type or Content-Length headers when requesting
    the new resource, but the body of the original request is not preserved.
    """
    handler_order = urllib2.HTTPRedirectHandler.handler_order - 1

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if False:
            while True:
                i = 10
        "Return a Request or None in response to a redirect.\n\n        This is called by the http_error_30x methods when a\n        redirection response is received.  If a redirection should\n        take place, return a new Request to allow http_error_30x to\n        perform the redirect.  Otherwise, raise HTTPError if no-one\n        else should try to handle this url.  Return None if you can't\n        but another Handler might.\n        "
        m = req.get_method()
        if code in (301, 302, 303, 307) and m in ('GET', 'HEAD') or (code in (301, 302, 303) and m == 'POST'):
            newurl = newurl.replace(' ', '%20')
            newheaders = dict(((k, v) for (k, v) in req.headers.items() if k.lower() not in ('content-length', 'content-type')))
            return urllib2.Request(newurl, headers=newheaders, origin_req_host=req.get_origin_req_host(), unverifiable=True)
        else:
            raise urllib2.HTTPError(req.get_full_url(), code, msg, headers, fp)

class StreamingHTTPHandler(urllib2.HTTPHandler):
    """Subclass of `urllib2.HTTPHandler` that uses
    StreamingHTTPConnection as its http connection class."""
    handler_order = urllib2.HTTPHandler.handler_order - 1

    def http_open(self, req):
        if False:
            return 10
        'Open a StreamingHTTPConnection for the given request'
        return self.do_open(StreamingHTTPConnection, req)

    def http_request(self, req):
        if False:
            print('Hello World!')
        "Handle a HTTP request.  Make sure that Content-Length is specified\n        if we're using an interable value"
        if req.has_data():
            data = req.get_data()
            if hasattr(data, 'read') or hasattr(data, 'next'):
                if not req.has_header('Content-length'):
                    raise ValueError('No Content-Length specified for iterable body')
        return urllib2.HTTPHandler.do_request_(self, req)
if hasattr(httplib, 'HTTPS'):

    class StreamingHTTPSConnection(_StreamingHTTPMixin, httplib.HTTPSConnection):
        """Subclass of `httplib.HTTSConnection` that overrides the `send()`
        method to support iterable body objects"""

    class StreamingHTTPSHandler(urllib2.HTTPSHandler):
        """Subclass of `urllib2.HTTPSHandler` that uses
        StreamingHTTPSConnection as its http connection class."""
        handler_order = urllib2.HTTPSHandler.handler_order - 1

        def https_open(self, req):
            if False:
                i = 10
                return i + 15
            return self.do_open(StreamingHTTPSConnection, req)

        def https_request(self, req):
            if False:
                return 10
            if req.has_data():
                data = req.get_data()
                if hasattr(data, 'read') or hasattr(data, 'next'):
                    if not req.has_header('Content-length'):
                        raise ValueError('No Content-Length specified for iterable body')
            return urllib2.HTTPSHandler.do_request_(self, req)

def get_handlers():
    if False:
        for i in range(10):
            print('nop')
    handlers = [StreamingHTTPHandler, StreamingHTTPRedirectHandler]
    if hasattr(httplib, 'HTTPS'):
        handlers.append(StreamingHTTPSHandler)
    return handlers

def register_openers():
    if False:
        for i in range(10):
            print('nop')
    'Register the streaming http handlers in the global urllib2 default\n    opener object.\n\n    Returns the created OpenerDirector object.'
    opener = urllib2.build_opener(*get_handlers())
    urllib2.install_opener(opener)
    return opener