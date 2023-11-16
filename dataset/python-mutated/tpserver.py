"""Throughput server."""
import sys
from twisted.internet import protocol, reactor
from twisted.protocols.wire import Discard
from twisted.python import log

def main():
    if False:
        return 10
    f = protocol.ServerFactory()
    f.protocol = Discard
    reactor.listenTCP(8000, f)
    reactor.run()
if __name__ == '__main__':
    main()