from twisted.internet import endpoints
from twisted.web import server, xmlrpc

class EchoHandler:

    def echo(self, x):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all passed args\n        '
        return x

class AddHandler:

    def add(self, a, b):
        if False:
            return 10
        '\n        Return sum of arguments.\n        '
        return a + b

class Example(xmlrpc.XMLRPC):
    """
    An example of using you own policy to fetch the handler
    """

    def __init__(self):
        if False:
            return 10
        xmlrpc.XMLRPC.__init__(self)
        self._addHandler = AddHandler()
        self._echoHandler = EchoHandler()
        self._procedureToCallable = {'add': self._addHandler.add, 'echo': self._echoHandler.echo}

    def lookupProcedure(self, procedurePath):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._procedureToCallable[procedurePath]
        except KeyError as e:
            raise xmlrpc.NoSuchFunction(self.NOT_FOUND, 'procedure %s not found' % procedurePath)

    def listProcedures(self):
        if False:
            return 10
        '\n        Since we override lookupProcedure, its suggested to override\n        listProcedures too.\n        '
        return ['add', 'echo']
if __name__ == '__main__':
    from twisted.internet import reactor
    r = Example()
    endpoint = endpoints.TCP4ServerEndpoint(reactor, 7080)
    endpoint.listen(server.Site(r))
    reactor.run()