"""
Loopback helper used in test_ssh and test_recvline
"""
from twisted.protocols import loopback

class LoopbackRelay(loopback.LoopbackRelay):
    clearCall = None

    def logPrefix(self):
        if False:
            return 10
        return f'LoopbackRelay({self.target.__class__.__name__!r})'

    def write(self, data):
        if False:
            return 10
        loopback.LoopbackRelay.write(self, data)
        if self.clearCall is not None:
            self.clearCall.cancel()
        from twisted.internet import reactor
        self.clearCall = reactor.callLater(0, self._clearBuffer)

    def _clearBuffer(self):
        if False:
            return 10
        self.clearCall = None
        loopback.LoopbackRelay.clearBuffer(self)