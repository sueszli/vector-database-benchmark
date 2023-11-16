from twisted.internet.interfaces import IAddress
from twisted.internet.interfaces import ITransport
from twisted.protocols import basic
from zope.interface import implementer
from buildbot.util import unicode2bytes

@implementer(IAddress)
class NullAddress:
    """an address for NullTransport"""

@implementer(ITransport)
class NullTransport:
    """a do-nothing transport to make NetstringReceiver happy"""

    def write(self, data):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def writeSequence(self, data):
        if False:
            while True:
                i = 10
        raise NotImplementedError

    def loseConnection(self):
        if False:
            print('Hello World!')
        pass

    def getPeer(self):
        if False:
            return 10
        return NullAddress

    def getHost(self):
        if False:
            print('Hello World!')
        return NullAddress

class NetstringParser(basic.NetstringReceiver):
    """
    Adapts the Twisted netstring support (which assumes it is on a socket) to
    work on simple strings, too.  Call the C{feed} method with arbitrary blocks
    of data, and override the C{stringReceived} method to get called for each
    embedded netstring.  The default implementation collects the netstrings in
    the list C{self.strings}.
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.makeConnection(NullTransport())
        self.strings = []

    def feed(self, data):
        if False:
            i = 10
            return i + 15
        data = unicode2bytes(data)
        self.dataReceived(data)
        if self.brokenPeer:
            raise basic.NetstringParseError

    def stringReceived(self, string):
        if False:
            while True:
                i = 10
        self.strings.append(string)