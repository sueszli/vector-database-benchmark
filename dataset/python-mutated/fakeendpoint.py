"""
Fake client and server endpoint string parser plugins for testing purposes.
"""
from zope.interface.declarations import implementer
from twisted.internet.interfaces import IStreamClientEndpoint, IStreamClientEndpointStringParserWithReactor, IStreamServerEndpoint, IStreamServerEndpointStringParser
from twisted.plugin import IPlugin

@implementer(IPlugin)
class PluginBase:

    def __init__(self, pfx):
        if False:
            while True:
                i = 10
        self.prefix = pfx

@implementer(IStreamClientEndpointStringParserWithReactor)
class FakeClientParserWithReactor(PluginBase):

    def parseStreamClient(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        return StreamClient(self, a, kw)

@implementer(IStreamServerEndpointStringParser)
class FakeParser(PluginBase):

    def parseStreamServer(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        return StreamServer(self, a, kw)

class EndpointBase:

    def __init__(self, parser, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.parser = parser
        self.args = args
        self.kwargs = kwargs

@implementer(IStreamClientEndpoint)
class StreamClient(EndpointBase):

    def connect(self, protocolFactory=None):
        if False:
            return 10
        pass

@implementer(IStreamServerEndpoint)
class StreamServer(EndpointBase):

    def listen(self, protocolFactory=None):
        if False:
            for i in range(10):
                print('nop')
        pass
fake = FakeParser('fake')
fakeClientWithReactor = FakeClientParserWithReactor('crfake')
fakeClientWithReactorAndPreference = FakeClientParserWithReactor('cpfake')