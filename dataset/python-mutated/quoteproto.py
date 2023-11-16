from zope.interface import Interface
from twisted.internet.protocol import Factory, Protocol

class IQuoter(Interface):
    """
    An object that returns quotes.
    """

    def getQuote():
        if False:
            return 10
        '\n        Return a quote.\n        '

class QOTD(Protocol):

    def connectionMade(self):
        if False:
            return 10
        self.transport.write(self.factory.quoter.getQuote() + '\r\n')
        self.transport.loseConnection()

class QOTDFactory(Factory):
    """
    A factory for the Quote of the Day protocol.

    @type quoter: L{IQuoter} provider
    @ivar quoter: An object which provides L{IQuoter} which will be used by
        the L{QOTD} protocol to get quotes to emit.
    """
    protocol = QOTD

    def __init__(self, quoter):
        if False:
            while True:
                i = 10
        self.quoter = quoter