"""
Tests for L{twisted.internet.serialport}.
"""
from twisted.internet.error import ConnectionDone
from twisted.internet.protocol import Protocol
from twisted.python.failure import Failure
from twisted.trial import unittest
try:
    from twisted.internet import serialport as _serialport
except ImportError:
    serialport = None
else:
    serialport = _serialport

class DoNothing:
    """
    Object with methods that do nothing.
    """

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        pass

    def __getattr__(self, attr):
        if False:
            return 10
        return lambda *args, **kwargs: None

class SerialPortTests(unittest.TestCase):
    """
    Minimal testing for Twisted's serial port support.

    See ticket #2462 for the eventual full test suite.
    """
    if serialport is None:
        skip = 'Serial port support is not available.'

    def test_connectionMadeLost(self):
        if False:
            print('Hello World!')
        '\n        C{connectionMade} and C{connectionLost} are called on the protocol by\n        the C{SerialPort}.\n        '

        class DummySerialPort(serialport.SerialPort):
            _serialFactory = DoNothing

            def _finishPortSetup(self):
                if False:
                    while True:
                        i = 10
                pass
        events = []

        class SerialProtocol(Protocol):

            def connectionMade(self):
                if False:
                    return 10
                events.append('connectionMade')

            def connectionLost(self, reason):
                if False:
                    print('Hello World!')
                events.append(('connectionLost', reason))
        port = DummySerialPort(SerialProtocol(), '', reactor=DoNothing())
        self.assertEqual(events, ['connectionMade'])
        f = Failure(ConnectionDone())
        port.connectionLost(f)
        self.assertEqual(events, ['connectionMade', ('connectionLost', f)])