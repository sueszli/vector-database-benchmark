"""
Tests for L{twisted.words.xmpproutertap}.
"""
from twisted.application import internet
from twisted.trial import unittest
from twisted.words import xmpproutertap as tap
from twisted.words.protocols.jabber import component

class XMPPRouterTapTests(unittest.TestCase):

    def test_port(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The port option is recognised as a parameter.\n        '
        opt = tap.Options()
        opt.parseOptions(['--port', '7001'])
        self.assertEqual(opt['port'], '7001')

    def test_portDefault(self) -> None:
        if False:
            while True:
                i = 10
        "\n        The port option has '5347' as default value\n        "
        opt = tap.Options()
        opt.parseOptions([])
        self.assertEqual(opt['port'], 'tcp:5347:interface=127.0.0.1')

    def test_secret(self) -> None:
        if False:
            print('Hello World!')
        '\n        The secret option is recognised as a parameter.\n        '
        opt = tap.Options()
        opt.parseOptions(['--secret', 'hushhush'])
        self.assertEqual(opt['secret'], 'hushhush')

    def test_secretDefault(self) -> None:
        if False:
            print('Hello World!')
        "\n        The secret option has 'secret' as default value\n        "
        opt = tap.Options()
        opt.parseOptions([])
        self.assertEqual(opt['secret'], 'secret')

    def test_verbose(self) -> None:
        if False:
            return 10
        '\n        The verbose option is recognised as a flag.\n        '
        opt = tap.Options()
        opt.parseOptions(['--verbose'])
        self.assertTrue(opt['verbose'])

    def test_makeService(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The service gets set up with a router and factory.\n        '
        opt = tap.Options()
        opt.parseOptions([])
        s = tap.makeService(opt)
        self.assertIsInstance(s, internet.StreamServerEndpointService)
        self.assertEqual('127.0.0.1', s.endpoint._interface)
        self.assertEqual(5347, s.endpoint._port)
        factory = s.factory
        self.assertIsInstance(factory, component.XMPPComponentServerFactory)
        self.assertIsInstance(factory.router, component.Router)
        self.assertEqual('secret', factory.secret)
        self.assertFalse(factory.logTraffic)

    def test_makeServiceVerbose(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The verbose flag enables traffic logging.\n        '
        opt = tap.Options()
        opt.parseOptions(['--verbose'])
        s = tap.makeService(opt)
        self.assertTrue(s.factory.logTraffic)