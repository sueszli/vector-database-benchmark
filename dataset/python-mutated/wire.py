"""Implement standard (and unused) TCP protocols.

These protocols are either provided by inetd, or are not provided at all.
"""
import struct
import time
from zope.interface import implementer
from twisted.internet import interfaces, protocol

class Echo(protocol.Protocol):
    """
    As soon as any data is received, write it back (RFC 862).
    """

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        self.transport.write(data)

class Discard(protocol.Protocol):
    """
    Discard any received data (RFC 863).
    """

    def dataReceived(self, data):
        if False:
            return 10
        pass

@implementer(interfaces.IProducer)
class Chargen(protocol.Protocol):
    """
    Generate repeating noise (RFC 864).
    """
    noise = b'@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~ !"#$%&?'

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport.registerProducer(self, 0)

    def resumeProducing(self):
        if False:
            i = 10
            return i + 15
        self.transport.write(self.noise)

    def pauseProducing(self):
        if False:
            print('Hello World!')
        pass

    def stopProducing(self):
        if False:
            i = 10
            return i + 15
        pass

class QOTD(protocol.Protocol):
    """
    Return a quote of the day (RFC 865).
    """

    def connectionMade(self):
        if False:
            return 10
        self.transport.write(self.getQuote())
        self.transport.loseConnection()

    def getQuote(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a quote. May be overrriden in subclasses.\n        '
        return b'An apple a day keeps the doctor away.\r\n'

class Who(protocol.Protocol):
    """
    Return list of active users (RFC 866)
    """

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.transport.write(self.getUsers())
        self.transport.loseConnection()

    def getUsers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return active users. Override in subclasses.\n        '
        return b'root\r\n'

class Daytime(protocol.Protocol):
    """
    Send back the daytime in ASCII form (RFC 867).
    """

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        self.transport.write(time.asctime(time.gmtime(time.time())) + b'\r\n')
        self.transport.loseConnection()

class Time(protocol.Protocol):
    """
    Send back the time in machine readable form (RFC 868).
    """

    def connectionMade(self):
        if False:
            return 10
        result = struct.pack('!i', int(time.time()))
        self.transport.write(result)
        self.transport.loseConnection()
__all__ = ['Echo', 'Discard', 'Chargen', 'QOTD', 'Who', 'Daytime', 'Time']