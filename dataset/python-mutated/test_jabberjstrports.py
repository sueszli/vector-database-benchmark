"""
Tests for L{twisted.words.protocols.jabber.jstrports}.
"""
from twisted.application.internet import TCPClient
from twisted.trial import unittest
from twisted.words.protocols.jabber import jstrports

class JabberStrPortsPlaceHolderTests(unittest.TestCase):
    """
    Tests for L{jstrports}
    """

    def test_parse(self):
        if False:
            print('Hello World!')
        '\n        L{jstrports.parse} accepts an endpoint description string and returns a\n        tuple and dict of parsed endpoint arguments.\n        '
        expected = ('TCP', ('DOMAIN', 65535, 'Factory'), {})
        got = jstrports.parse('tcp:DOMAIN:65535', 'Factory')
        self.assertEqual(expected, got)

    def test_client(self):
        if False:
            print('Hello World!')
        '\n        L{jstrports.client} returns a L{TCPClient} service.\n        '
        got = jstrports.client('tcp:DOMAIN:65535', 'Factory')
        self.assertIsInstance(got, TCPClient)