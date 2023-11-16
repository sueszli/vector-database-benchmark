from twisted.protocols import basic
from twisted.trial import unittest
from buildbot.util import netstrings

class NetstringParser(unittest.TestCase):

    def test_valid_netstrings(self):
        if False:
            i = 10
            return i + 15
        p = netstrings.NetstringParser()
        p.feed('5:hello,5:world,')
        self.assertEqual(p.strings, [b'hello', b'world'])

    def test_valid_netstrings_byte_by_byte(self):
        if False:
            return 10
        p = netstrings.NetstringParser()
        for c in '5:hello,5:world,':
            p.feed(c)
        self.assertEqual(p.strings, [b'hello', b'world'])

    def test_invalid_netstring(self):
        if False:
            print('Hello World!')
        p = netstrings.NetstringParser()
        with self.assertRaises(basic.NetstringParseError):
            p.feed('5-hello!')

    def test_incomplete_netstring(self):
        if False:
            return 10
        p = netstrings.NetstringParser()
        p.feed('11:hello world,6:foob')
        self.assertEqual(p.strings, [b'hello world'])