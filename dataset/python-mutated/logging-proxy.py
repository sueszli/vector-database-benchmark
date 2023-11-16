"""
An example of a proxy which logs all requests processed through it.

Usage:
    $ python logging-proxy.py

Then configure your web browser to use localhost:8080 as a proxy, and visit a
URL (This is not a SOCKS proxy). When browsing in this configuration, this
example will proxy connections from the browser to the server indicated by URLs
which are visited.  The client IP and the request hostname will be logged for
each request.

HTTP is supported.  HTTPS is not supported.

See also proxy.py for a simpler proxy example.
"""
from twisted.internet import reactor
from twisted.web import http, proxy

class LoggingProxyRequest(proxy.ProxyRequest):

    def process(self):
        if False:
            i = 10
            return i + 15
        "\n        It's normal to see a blank HTTPS page. As the proxy only works\n        with the HTTP protocol.\n        "
        print('Request from %s for %s' % (self.getClientIP(), self.getAllHeaders()['host']))
        try:
            proxy.ProxyRequest.process(self)
        except KeyError:
            print('HTTPS is not supported at the moment!')

class LoggingProxy(proxy.Proxy):
    requestFactory = LoggingProxyRequest

class LoggingProxyFactory(http.HTTPFactory):

    def buildProtocol(self, addr):
        if False:
            while True:
                i = 10
        return LoggingProxy()
reactor.listenTCP(8080, LoggingProxyFactory())
reactor.run()