"""
This example makes remote XML-RPC calls.

Usage:
    $ python xmlrpcclient.py

The example will make an XML-RPC request to advogato.org and display the result.
"""
from twisted.internet import reactor
from twisted.web.xmlrpc import Proxy

def printValue(value):
    if False:
        while True:
            i = 10
    print(repr(value))
    reactor.stop()

def printError(error):
    if False:
        print('Hello World!')
    print('error', error)
    reactor.stop()

def capitalize(value):
    if False:
        return 10
    print(repr(value))
    proxy.callRemote('test.capitalize', 'moshe zadka').addCallbacks(printValue, printError)
proxy = Proxy(b'http://advogato.org/XMLRPC')
proxy.callRemote('test.sumprod', 2, 5).addCallbacks(capitalize, printError)
reactor.run()