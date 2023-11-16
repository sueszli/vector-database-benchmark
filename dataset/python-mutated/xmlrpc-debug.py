"""
This example prints raw XML-RPC traffic for a client.

Usage:
    $ python xmlrpc-debug.py

The example will make a simple XML-RPC request to bugzilla.redhat.com and print
the raw XML response string from the server.
"""
from twisted.internet import reactor
from twisted.web.xmlrpc import Proxy, QueryFactory

class DebuggingQueryFactory(QueryFactory):
    """Print the server's raw responses before continuing with parsing."""

    def parseResponse(self, contents):
        if False:
            return 10
        print(contents)
        return QueryFactory.parseResponse(self, contents)

def printValue(value):
    if False:
        for i in range(10):
            print('nop')
    print(repr(value))
    reactor.stop()

def printError(error):
    if False:
        while True:
            i = 10
    print('error', error)
    reactor.stop()
proxy = Proxy(b'https://bugzilla.redhat.com/xmlrpc.cgi')
proxy.queryFactory = DebuggingQueryFactory
proxy.callRemote('Bugzilla.version').addCallbacks(printValue, printError)
reactor.run()