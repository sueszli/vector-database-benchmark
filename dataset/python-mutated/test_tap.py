"""
Tests for L{twisted.names.tap}.
"""
from twisted.internet.base import ThreadedResolver
from twisted.names.client import Resolver
from twisted.names.dns import PORT
from twisted.names.resolve import ResolverChain
from twisted.names.secondary import SecondaryAuthorityService
from twisted.names.tap import Options, _buildResolvers
from twisted.python.runtime import platform
from twisted.python.usage import UsageError
from twisted.trial.unittest import SynchronousTestCase

class OptionsTests(SynchronousTestCase):
    """
    Tests for L{Options}, defining how command line arguments for the DNS server
    are parsed.
    """

    def test_malformedSecondary(self) -> None:
        if False:
            return 10
        '\n        If the value supplied for an I{--secondary} option does not provide a\n        server IP address, optional port number, and domain name,\n        L{Options.parseOptions} raises L{UsageError}.\n        '
        options = Options()
        self.assertRaises(UsageError, options.parseOptions, ['--secondary', ''])
        self.assertRaises(UsageError, options.parseOptions, ['--secondary', '1.2.3.4'])
        self.assertRaises(UsageError, options.parseOptions, ['--secondary', '1.2.3.4:hello'])
        self.assertRaises(UsageError, options.parseOptions, ['--secondary', '1.2.3.4:hello/example.com'])

    def test_secondary(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        An argument of the form C{"ip/domain"} is parsed by L{Options} for the\n        I{--secondary} option and added to its list of secondaries, using the\n        default DNS port number.\n        '
        options = Options()
        options.parseOptions(['--secondary', '1.2.3.4/example.com'])
        self.assertEqual([(('1.2.3.4', PORT), ['example.com'])], options.secondaries)

    def test_secondaryExplicitPort(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        An argument of the form C{"ip:port/domain"} can be used to specify an\n        alternate port number for which to act as a secondary.\n        '
        options = Options()
        options.parseOptions(['--secondary', '1.2.3.4:5353/example.com'])
        self.assertEqual([(('1.2.3.4', 5353), ['example.com'])], options.secondaries)

    def test_secondaryAuthorityServices(self) -> None:
        if False:
            print('Hello World!')
        '\n        After parsing I{--secondary} options, L{Options} constructs a\n        L{SecondaryAuthorityService} instance for each configured secondary.\n        '
        options = Options()
        options.parseOptions(['--secondary', '1.2.3.4:5353/example.com', '--secondary', '1.2.3.5:5354/example.com'])
        self.assertEqual(len(options.svcs), 2)
        secondary = options.svcs[0]
        self.assertIsInstance(options.svcs[0], SecondaryAuthorityService)
        self.assertEqual(secondary.primary, '1.2.3.4')
        self.assertEqual(secondary._port, 5353)
        secondary = options.svcs[1]
        self.assertIsInstance(options.svcs[1], SecondaryAuthorityService)
        self.assertEqual(secondary.primary, '1.2.3.5')
        self.assertEqual(secondary._port, 5354)

    def test_recursiveConfiguration(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Recursive DNS lookups, if enabled, should be a last-resort option.\n        Any other lookup method (cache, local lookup, etc.) should take\n        precedence over recursive lookups\n        '
        options = Options()
        options.parseOptions(['--hosts-file', 'hosts.txt', '--recursive'])
        (ca, cl) = _buildResolvers(options)
        for x in cl:
            if isinstance(x, ResolverChain):
                recurser = x.resolvers[-1]
                if isinstance(recurser, Resolver):
                    recurser._parseCall.cancel()
        if platform.getType() != 'posix':
            from twisted.internet import reactor
            for x in reactor._newTimedCalls:
                self.assertEqual(x.func.__func__, ThreadedResolver._cleanup)
                x.cancel()
        self.assertIsInstance(cl[-1], ResolverChain)