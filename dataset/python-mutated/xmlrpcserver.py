"""
An example of an XML-RPC server in Twisted.

Usage:
    $ python xmlrpc.py

An example session (assuming the server is running):

    >>> import xmlrpclib
    >>> s = xmlrpclib.Server('http://localhost:7080/')
    >>> s.echo("lala")
    ['lala']
    >>> s.echo("lala", 1)
    ['lala', 1]
    >>> s.echo("lala", 4)
    ['lala', 4]
    >>> s.echo("lala", 4, 3.4)
    ['lala', 4, 3.3999999999999999]
    >>> s.echo("lala", 4, [1, 2])
    ['lala', 4, [1, 2]]

"""
from xmlrpc.client import Fault
from twisted.internet import defer
from twisted.web import xmlrpc

class Echoer(xmlrpc.XMLRPC):
    """
    An example object to be published.

    Has five methods accessible by XML-RPC, 'echo', 'hello', 'defer',
    'defer_fail' and 'fail.
    """

    def xmlrpc_echo(self, *args):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return all passed args.\n        '
        return args

    def xmlrpc_hello(self):
        if False:
            print('Hello World!')
        "\n        Return 'hello, world'.\n        "
        return 'hello, world!'

    def xmlrpc_defer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show how xmlrpc methods can return Deferred.\n        '
        return defer.succeed('hello')

    def xmlrpc_defer_fail(self):
        if False:
            i = 10
            return i + 15
        '\n        Show how xmlrpc methods can return failed Deferred.\n        '
        return defer.fail(12)

    def xmlrpc_fail(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show how we can return a failure code.\n        '
        return Fault(7, 'Out of cheese.')

def main():
    if False:
        while True:
            i = 10
    from twisted.internet import reactor
    from twisted.web import server
    r = Echoer()
    reactor.listenTCP(7080, server.Site(r))
    reactor.run()
if __name__ == '__main__':
    main()