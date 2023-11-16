"""
Tests for L{twisted.conch.manhole_tap}.
"""
from twisted.application.internet import StreamServerEndpointService
from twisted.application.service import MultiService
from twisted.conch import telnet
from twisted.cred import error
from twisted.cred.credentials import UsernamePassword
from twisted.python import usage
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
cryptography = requireModule('cryptography')
if cryptography:
    from twisted.conch import manhole_ssh, manhole_tap

class MakeServiceTests(TestCase):
    """
    Tests for L{manhole_tap.makeService}.
    """
    if not cryptography:
        skip = "can't run without cryptography"
    usernamePassword = (b'iamuser', b'thisispassword')

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        '\n        Create a passwd-like file with a user.\n        '
        self.filename = self.mktemp()
        with open(self.filename, 'wb') as f:
            f.write(b':'.join(self.usernamePassword))
        self.options = manhole_tap.Options()

    def test_requiresPort(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{manhole_tap.makeService} requires either 'telnetPort' or 'sshPort' to\n        be given.\n        "
        with self.assertRaises(usage.UsageError) as e:
            manhole_tap.Options().parseOptions([])
        self.assertEqual(e.exception.args[0], 'At least one of --telnetPort and --sshPort must be specified')

    def test_telnetPort(self) -> None:
        if False:
            return 10
        '\n        L{manhole_tap.makeService} will make a telnet service on the port\n        defined by C{--telnetPort}. It will not make a SSH service.\n        '
        self.options.parseOptions(['--telnetPort', 'tcp:222'])
        service = manhole_tap.makeService(self.options)
        self.assertIsInstance(service, MultiService)
        self.assertEqual(len(service.services), 1)
        self.assertIsInstance(service.services[0], StreamServerEndpointService)
        self.assertIsInstance(service.services[0].factory.protocol, manhole_tap.makeTelnetProtocol)
        self.assertEqual(service.services[0].endpoint._port, 222)

    def test_sshPort(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{manhole_tap.makeService} will make a SSH service on the port\n        defined by C{--sshPort}. It will not make a telnet service.\n        '
        self.options.parseOptions(['--sshKeyDir', self.mktemp(), '--sshKeySize', '512', '--sshPort', 'tcp:223'])
        service = manhole_tap.makeService(self.options)
        self.assertIsInstance(service, MultiService)
        self.assertEqual(len(service.services), 1)
        self.assertIsInstance(service.services[0], StreamServerEndpointService)
        self.assertIsInstance(service.services[0].factory, manhole_ssh.ConchFactory)
        self.assertEqual(service.services[0].endpoint._port, 223)

    def test_passwd(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The C{--passwd} command-line option will load a passwd-like file.\n        '
        self.options.parseOptions(['--telnetPort', 'tcp:22', '--passwd', self.filename])
        service = manhole_tap.makeService(self.options)
        portal = service.services[0].factory.protocol.portal
        self.assertEqual(len(portal.checkers.keys()), 2)
        self.assertTrue(self.successResultOf(portal.login(UsernamePassword(*self.usernamePassword), None, telnet.ITelnetProtocol)))
        self.assertIsInstance(self.failureResultOf(portal.login(UsernamePassword(b'wrong', b'user'), None, telnet.ITelnetProtocol)).value, error.UnauthorizedLogin)