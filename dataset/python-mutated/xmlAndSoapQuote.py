import os
from twisted.internet import endpoints
from twisted.web import resource, server, soap, xmlrpc

def getQuote():
    if False:
        for i in range(10):
            print('nop')
    return 'Victory to the burgeois, you capitalist swine!'

class XMLRPCQuoter(xmlrpc.XMLRPC):

    def xmlrpc_quote(self):
        if False:
            while True:
                i = 10
        return getQuote()

class SOAPQuoter(soap.SOAPPublisher):

    def soap_quote(self):
        if False:
            while True:
                i = 10
        return getQuote()

def main():
    if False:
        for i in range(10):
            print('nop')
    from twisted.internet import reactor
    root = resource.Resource()
    root.putChild('RPC2', XMLRPCQuoter())
    root.putChild('SOAP', SOAPQuoter())
    endpoint = endpoints.TCP4ServerEndpoint(reactor, 7080)
    endpoint.listen(server.Site(root))
    reactor.run()
if __name__ == '__main__':
    main()