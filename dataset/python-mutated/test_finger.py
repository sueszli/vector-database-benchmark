"""
Tests for L{twisted.protocols.finger}.
"""
from twisted.internet.testing import StringTransport
from twisted.protocols import finger
from twisted.trial import unittest

class FingerTests(unittest.TestCase):
    """
    Tests for L{finger.Finger}.
    """

    def setUp(self) -> None:
        if False:
            return 10
        '\n        Create and connect a L{finger.Finger} instance.\n        '
        self.transport = StringTransport()
        self.protocol = finger.Finger()
        self.protocol.makeConnection(self.transport)

    def test_simple(self) -> None:
        if False:
            while True:
                i = 10
        '\n        When L{finger.Finger} receives a CR LF terminated line, it responds\n        with the default user status message - that no such user exists.\n        '
        self.protocol.dataReceived(b'moshez\r\n')
        self.assertEqual(self.transport.value(), b'Login: moshez\nNo such user\n')

    def test_simpleW(self) -> None:
        if False:
            print('Hello World!')
        '\n        The behavior for a query which begins with C{"/w"} is the same as the\n        behavior for one which does not.  The user is reported as not existing.\n        '
        self.protocol.dataReceived(b'/w moshez\r\n')
        self.assertEqual(self.transport.value(), b'Login: moshez\nNo such user\n')

    def test_forwarding(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        When L{finger.Finger} receives a request for a remote user, it responds\n        with a message rejecting the request.\n        '
        self.protocol.dataReceived(b'moshez@example.com\r\n')
        self.assertEqual(self.transport.value(), b'Finger forwarding service denied\n')

    def test_list(self) -> None:
        if False:
            return 10
        '\n        When L{finger.Finger} receives a blank line, it responds with a message\n        rejecting the request for all online users.\n        '
        self.protocol.dataReceived(b'\r\n')
        self.assertEqual(self.transport.value(), b'Finger online list denied\n')