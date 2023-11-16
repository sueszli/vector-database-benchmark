import os
import socket
import subprocess
import sys
from itertools import count
from unittest import skipIf
from zope.interface import implementer
from twisted.conch.error import ConchError
from twisted.conch.test.keydata import privateDSA_openssh, privateRSA_openssh, publicDSA_openssh, publicRSA_openssh
from twisted.conch.test.test_ssh import ConchTestRealm
from twisted.cred import portal
from twisted.internet import defer, protocol, reactor
from twisted.internet.error import ProcessExitedAlready
from twisted.internet.task import LoopingCall
from twisted.internet.utils import getProcessValue
from twisted.python import filepath, log, runtime
from twisted.python.filepath import FilePath
from twisted.python.procutils import which
from twisted.python.reflect import requireModule
from twisted.trial.unittest import SkipTest, TestCase
try:
    from twisted.conch.test.test_ssh import ConchTestServerFactory, conchTestPublicKeyChecker
except ImportError:
    pass
cryptography = requireModule('cryptography')
if cryptography:
    from twisted.conch.avatar import ConchUser
    from twisted.conch.ssh.session import ISession, SSHSession, wrapProtocol
else:
    from twisted.conch.interfaces import ISession

    class ConchUser:
        pass
try:
    from twisted.conch.scripts.conch import SSHSession as _StdioInteractingSession
except ImportError as e:
    StdioInteractingSession = None
    _reason = str(e)
    del e
else:
    StdioInteractingSession = _StdioInteractingSession

def _has_ipv6():
    if False:
        for i in range(10):
            print('nop')
    'Returns True if the system can bind an IPv6 address.'
    sock = None
    has_ipv6 = False
    try:
        sock = socket.socket(socket.AF_INET6)
        sock.bind(('::1', 0))
        has_ipv6 = True
    except OSError:
        pass
    if sock:
        sock.close()
    return has_ipv6
HAS_IPV6 = _has_ipv6()

class FakeStdio:
    """
    A fake for testing L{twisted.conch.scripts.conch.SSHSession.eofReceived} and
    L{twisted.conch.scripts.cftp.SSHSession.eofReceived}.

    @ivar writeConnLost: A flag which records whether L{loserWriteConnection}
        has been called.
    """
    writeConnLost = False

    def loseWriteConnection(self):
        if False:
            return 10
        '\n        Record the call to loseWriteConnection.\n        '
        self.writeConnLost = True

class StdioInteractingSessionTests(TestCase):
    """
    Tests for L{twisted.conch.scripts.conch.SSHSession}.
    """
    if StdioInteractingSession is None:
        skip = _reason

    def test_eofReceived(self):
        if False:
            i = 10
            return i + 15
        '\n        L{twisted.conch.scripts.conch.SSHSession.eofReceived} loses the\n        write half of its stdio connection.\n        '
        stdio = FakeStdio()
        channel = StdioInteractingSession()
        channel.stdio = stdio
        channel.eofReceived()
        self.assertTrue(stdio.writeConnLost)

class Echo(protocol.Protocol):

    def connectionMade(self):
        if False:
            for i in range(10):
                print('nop')
        log.msg('ECHO CONNECTION MADE')

    def connectionLost(self, reason):
        if False:
            for i in range(10):
                print('nop')
        log.msg('ECHO CONNECTION DONE')

    def dataReceived(self, data):
        if False:
            while True:
                i = 10
        self.transport.write(data)
        if b'\n' in data:
            self.transport.loseConnection()

class EchoFactory(protocol.Factory):
    protocol = Echo

class ConchTestOpenSSHProcess(protocol.ProcessProtocol):
    """
    Test protocol for launching an OpenSSH client process.

    @ivar deferred: Set by whatever uses this object. Accessed using
    L{_getDeferred}, which destroys the value so the Deferred is not
    fired twice. Fires when the process is terminated.
    """
    deferred = None
    buf = b''
    problems = b''

    def _getDeferred(self):
        if False:
            return 10
        (d, self.deferred) = (self.deferred, None)
        return d

    def outReceived(self, data):
        if False:
            print('Hello World!')
        self.buf += data

    def errReceived(self, data):
        if False:
            while True:
                i = 10
        self.problems += data

    def processEnded(self, reason):
        if False:
            return 10
        "\n        Called when the process has ended.\n\n        @param reason: a Failure giving the reason for the process' end.\n        "
        if reason.value.exitCode != 0:
            self._getDeferred().errback(ConchError('exit code was not 0: {} ({})'.format(reason.value.exitCode, self.problems.decode('charmap'))))
        else:
            buf = self.buf.replace(b'\r\n', b'\n')
            self._getDeferred().callback(buf)

class ConchTestForwardingProcess(protocol.ProcessProtocol):
    """
    Manages a third-party process which launches a server.

    Uses L{ConchTestForwardingPort} to connect to the third-party server.
    Once L{ConchTestForwardingPort} has disconnected, kill the process and fire
    a Deferred with the data received by the L{ConchTestForwardingPort}.

    @ivar deferred: Set by whatever uses this object. Accessed using
    L{_getDeferred}, which destroys the value so the Deferred is not
    fired twice. Fires when the process is terminated.
    """
    deferred = None

    def __init__(self, port, data):
        if False:
            while True:
                i = 10
        "\n        @type port: L{int}\n        @param port: The port on which the third-party server is listening.\n        (it is assumed that the server is running on localhost).\n\n        @type data: L{str}\n        @param data: This is sent to the third-party server. Must end with '\n'\n        in order to trigger a disconnect.\n        "
        self.port = port
        self.buffer = None
        self.data = data

    def _getDeferred(self):
        if False:
            while True:
                i = 10
        (d, self.deferred) = (self.deferred, None)
        return d

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self._connect()

    def _connect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Connect to the server, which is often a third-party process.\n        Tries to reconnect if it fails because we have no way of determining\n        exactly when the port becomes available for listening -- we can only\n        know when the process starts.\n        '
        cc = protocol.ClientCreator(reactor, ConchTestForwardingPort, self, self.data)
        d = cc.connectTCP('127.0.0.1', self.port)
        d.addErrback(self._ebConnect)
        return d

    def _ebConnect(self, f):
        if False:
            i = 10
            return i + 15
        reactor.callLater(0.1, self._connect)

    def forwardingPortDisconnected(self, buffer):
        if False:
            i = 10
            return i + 15
        '\n        The network connection has died; save the buffer of output\n        from the network and attempt to quit the process gracefully,\n        and then (after the reactor has spun) send it a KILL signal.\n        '
        self.buffer = buffer
        self.transport.write(b'\x03')
        self.transport.loseConnection()
        reactor.callLater(0, self._reallyDie)

    def _reallyDie(self):
        if False:
            i = 10
            return i + 15
        try:
            self.transport.signalProcess('KILL')
        except ProcessExitedAlready:
            pass

    def processEnded(self, reason):
        if False:
            for i in range(10):
                print('nop')
        '\n        Fire the Deferred at self.deferred with the data collected\n        from the L{ConchTestForwardingPort} connection, if any.\n        '
        self._getDeferred().callback(self.buffer)

class ConchTestForwardingPort(protocol.Protocol):
    """
    Connects to server launched by a third-party process (managed by
    L{ConchTestForwardingProcess}) sends data, then reports whatever it
    received back to the L{ConchTestForwardingProcess} once the connection
    is ended.
    """

    def __init__(self, protocol, data):
        if False:
            print('Hello World!')
        '\n        @type protocol: L{ConchTestForwardingProcess}\n        @param protocol: The L{ProcessProtocol} which made this connection.\n\n        @type data: str\n        @param data: The data to be sent to the third-party server.\n        '
        self.protocol = protocol
        self.data = data

    def connectionMade(self):
        if False:
            i = 10
            return i + 15
        self.buffer = b''
        self.transport.write(self.data)

    def dataReceived(self, data):
        if False:
            print('Hello World!')
        self.buffer += data

    def connectionLost(self, reason):
        if False:
            i = 10
            return i + 15
        self.protocol.forwardingPortDisconnected(self.buffer)

def _makeArgs(args, mod='conch'):
    if False:
        while True:
            i = 10
    start = [sys.executable, "-c\n### Twisted Preamble\nimport sys, os\npath = os.path.abspath(sys.argv[0])\nwhile os.path.dirname(path) != path:\n    if os.path.basename(path).startswith('Twisted'):\n        sys.path.insert(0, path)\n        break\n    path = os.path.dirname(path)\n\nfrom twisted.conch.scripts.%s import run\nrun()" % mod]
    madeArgs = []
    for arg in start + list(args):
        if isinstance(arg, str):
            arg = arg.encode('utf-8')
        madeArgs.append(arg)
    return madeArgs

class ConchServerSetupMixin:
    if not cryptography:
        skip = "can't run without cryptography"

    @staticmethod
    def realmFactory():
        if False:
            return 10
        return ConchTestRealm(b'testuser')

    def _createFiles(self):
        if False:
            i = 10
            return i + 15
        for f in ['rsa_test', 'rsa_test.pub', 'dsa_test', 'dsa_test.pub', 'kh_test']:
            if os.path.exists(f):
                os.remove(f)
        with open('rsa_test', 'wb') as f:
            f.write(privateRSA_openssh)
        with open('rsa_test.pub', 'wb') as f:
            f.write(publicRSA_openssh)
        with open('dsa_test.pub', 'wb') as f:
            f.write(publicDSA_openssh)
        with open('dsa_test', 'wb') as f:
            f.write(privateDSA_openssh)
        os.chmod('dsa_test', 384)
        os.chmod('rsa_test', 384)
        permissions = FilePath('dsa_test').getPermissions()
        if permissions.group.read or permissions.other.read:
            raise SkipTest('private key readable by others despite chmod; possible windows permission issue? see https://tm.tl/9767')
        with open('kh_test', 'wb') as f:
            f.write(b'127.0.0.1 ' + publicRSA_openssh)

    def _getFreePort(self):
        if False:
            i = 10
            return i + 15
        s = socket.socket()
        s.bind(('', 0))
        port = s.getsockname()[1]
        s.close()
        return port

    def _makeConchFactory(self):
        if False:
            return 10
        '\n        Make a L{ConchTestServerFactory}, which allows us to start a\n        L{ConchTestServer} -- i.e. an actually listening conch.\n        '
        realm = self.realmFactory()
        p = portal.Portal(realm)
        p.registerChecker(conchTestPublicKeyChecker())
        factory = ConchTestServerFactory()
        factory.portal = p
        return factory

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self._createFiles()
        self.conchFactory = self._makeConchFactory()
        self.conchFactory.expectedLoseConnection = 1
        self.conchServer = reactor.listenTCP(0, self.conchFactory, interface='127.0.0.1')
        self.echoServer = reactor.listenTCP(0, EchoFactory())
        self.echoPort = self.echoServer.getHost().port
        if HAS_IPV6:
            self.echoServerV6 = reactor.listenTCP(0, EchoFactory(), interface='::1')
            self.echoPortV6 = self.echoServerV6.getHost().port

    def tearDown(self):
        if False:
            print('Hello World!')
        try:
            self.conchFactory.proto.done = 1
        except AttributeError:
            pass
        else:
            self.conchFactory.proto.transport.loseConnection()
        deferreds = [defer.maybeDeferred(self.conchServer.stopListening), defer.maybeDeferred(self.echoServer.stopListening)]
        if HAS_IPV6:
            deferreds.append(defer.maybeDeferred(self.echoServerV6.stopListening))
        return defer.gatherResults(deferreds)

class ForwardingMixin(ConchServerSetupMixin):
    """
    Template class for tests of the Conch server's ability to forward arbitrary
    protocols over SSH.

    These tests are integration tests, not unit tests. They launch a Conch
    server, a custom TCP server (just an L{EchoProtocol}) and then call
    L{execute}.

    L{execute} is implemented by subclasses of L{ForwardingMixin}. It should
    cause an SSH client to connect to the Conch server, asking it to forward
    data to the custom TCP server.
    """

    def test_exec(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that we can use whatever client to send the command "echo goodbye"\n        to the Conch server. Make sure we receive "goodbye" back from the\n        server.\n        '
        d = self.execute('echo goodbye', ConchTestOpenSSHProcess())
        return d.addCallback(self.assertEqual, b'goodbye\n')

    def test_localToRemoteForwarding(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that we can use whatever client to forward a local port to a\n        specified port on the server.\n        '
        localPort = self._getFreePort()
        process = ConchTestForwardingProcess(localPort, b'test\n')
        d = self.execute('', process, sshArgs='-N -L%i:127.0.0.1:%i' % (localPort, self.echoPort))
        d.addCallback(self.assertEqual, b'test\n')
        return d

    def test_remoteToLocalForwarding(self):
        if False:
            print('Hello World!')
        '\n        Test that we can use whatever client to forward a port from the server\n        to a port locally.\n        '
        localPort = self._getFreePort()
        process = ConchTestForwardingProcess(localPort, b'test\n')
        d = self.execute('', process, sshArgs='-N -R %i:127.0.0.1:%i' % (localPort, self.echoPort))
        d.addCallback(self.assertEqual, b'test\n')
        return d

@implementer(ISession)
class RekeyAvatar(ConchUser):
    """
    This avatar implements a shell which sends 60 numbered lines to whatever
    connects to it, then closes the session with a 0 exit status.

    60 lines is selected as being enough to send more than 2kB of traffic, the
    amount the client is configured to initiate a rekey after.
    """

    def __init__(self):
        if False:
            return 10
        ConchUser.__init__(self)
        self.channelLookup[b'session'] = SSHSession

    def openShell(self, transport):
        if False:
            while True:
                i = 10
        '\n        Write 60 lines of data to the transport, then exit.\n        '
        proto = protocol.Protocol()
        proto.makeConnection(transport)
        transport.makeConnection(wrapProtocol(proto))

        def write(counter):
            if False:
                while True:
                    i = 10
            i = next(counter)
            if i == 60:
                call.stop()
                transport.session.conn.sendRequest(transport.session, b'exit-status', b'\x00\x00\x00\x00')
                transport.loseConnection()
            else:
                line = 'line #%02d\n' % (i,)
                line = line.encode('utf-8')
                transport.write(line)
        call = LoopingCall(write, count())
        call.start(0.01)

    def closed(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Ignore the close of the session.\n        '

    def eofReceived(self):
        if False:
            while True:
                i = 10
        pass

    def execCommand(self, proto, command):
        if False:
            while True:
                i = 10
        pass

    def getPty(self, term, windowSize, modes):
        if False:
            i = 10
            return i + 15
        pass

    def windowChanged(self, newWindowSize):
        if False:
            print('Hello World!')
        pass

class RekeyRealm:
    """
    This realm gives out new L{RekeyAvatar} instances for any avatar request.
    """

    def requestAvatar(self, avatarID, mind, *interfaces):
        if False:
            while True:
                i = 10
        return (interfaces[0], RekeyAvatar(), lambda : None)

class RekeyTestsMixin(ConchServerSetupMixin):
    """
    TestCase mixin which defines tests exercising L{SSHTransportBase}'s handling
    of rekeying messages.
    """
    realmFactory = RekeyRealm

    def test_clientRekey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        After a client-initiated rekey is completed, application data continues\n        to be passed over the SSH connection.\n        '
        process = ConchTestOpenSSHProcess()
        d = self.execute('', process, '-o RekeyLimit=2K')

        def finished(result):
            if False:
                for i in range(10):
                    print('nop')
            expectedResult = '\n'.join(['line #%02d' % (i,) for i in range(60)]) + '\n'
            expectedResult = expectedResult.encode('utf-8')
            self.assertEqual(result, expectedResult)
        d.addCallback(finished)
        return d

class OpenSSHClientMixin:
    if not which('ssh'):
        skip = 'no ssh command-line client available'

    def execute(self, remoteCommand, process, sshArgs=''):
        if False:
            print('Hello World!')
        "\n        Connects to the SSH server started in L{ConchServerSetupMixin.setUp} by\n        running the 'ssh' command line tool.\n\n        @type remoteCommand: str\n        @param remoteCommand: The command (with arguments) to run on the\n        remote end.\n\n        @type process: L{ConchTestOpenSSHProcess}\n\n        @type sshArgs: str\n        @param sshArgs: Arguments to pass to the 'ssh' process.\n\n        @return: L{defer.Deferred}\n        "
        d = getProcessValue(which('ssh')[0], ('-o', 'PubkeyAcceptedKeyTypes=ssh-dss', '-V'))

        def hasPAKT(status):
            if False:
                print('Hello World!')
            if status == 0:
                opts = '-oPubkeyAcceptedKeyTypes=ssh-dss '
            else:
                opts = ''
            process.deferred = defer.Deferred()
            cmdline = 'ssh -2 -l testuser -p %i -F /dev/null -oUserKnownHostsFile=kh_test -oPasswordAuthentication=no -oHostKeyAlgorithms=ssh-rsa -a -i dsa_test ' + opts + sshArgs + ' 127.0.0.1 ' + remoteCommand
            port = self.conchServer.getHost().port
            cmds = (cmdline % port).split()
            encodedCmds = []
            for cmd in cmds:
                if isinstance(cmd, str):
                    cmd = cmd.encode('utf-8')
                encodedCmds.append(cmd)
            reactor.spawnProcess(process, which('ssh')[0], encodedCmds)
            return process.deferred
        return d.addCallback(hasPAKT)

class OpenSSHKeyExchangeTests(ConchServerSetupMixin, OpenSSHClientMixin, TestCase):
    """
    Tests L{SSHTransportBase}'s key exchange algorithm compatibility with
    OpenSSH.
    """

    def assertExecuteWithKexAlgorithm(self, keyExchangeAlgo):
        if False:
            while True:
                i = 10
        '\n        Call execute() method of L{OpenSSHClientMixin} with an ssh option that\n        forces the exclusive use of the key exchange algorithm specified by\n        keyExchangeAlgo\n\n        @type keyExchangeAlgo: L{str}\n        @param keyExchangeAlgo: The key exchange algorithm to use\n\n        @return: L{defer.Deferred}\n        '
        kexAlgorithms = []
        try:
            output = subprocess.check_output([which('ssh')[0], '-Q', 'kex'], stderr=subprocess.STDOUT)
            if not isinstance(output, str):
                output = output.decode('utf-8')
            kexAlgorithms = output.split()
        except BaseException:
            pass
        if keyExchangeAlgo not in kexAlgorithms:
            raise SkipTest(f'{keyExchangeAlgo} not supported by ssh client')
        d = self.execute('echo hello', ConchTestOpenSSHProcess(), '-oKexAlgorithms=' + keyExchangeAlgo)
        return d.addCallback(self.assertEqual, b'hello\n')

    def test_ECDHSHA256(self):
        if False:
            while True:
                i = 10
        '\n        The ecdh-sha2-nistp256 key exchange algorithm is compatible with\n        OpenSSH\n        '
        return self.assertExecuteWithKexAlgorithm('ecdh-sha2-nistp256')

    def test_ECDHSHA384(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        The ecdh-sha2-nistp384 key exchange algorithm is compatible with\n        OpenSSH\n        '
        return self.assertExecuteWithKexAlgorithm('ecdh-sha2-nistp384')

    def test_ECDHSHA521(self):
        if False:
            i = 10
            return i + 15
        '\n        The ecdh-sha2-nistp521 key exchange algorithm is compatible with\n        OpenSSH\n        '
        return self.assertExecuteWithKexAlgorithm('ecdh-sha2-nistp521')

    def test_DH_GROUP14(self):
        if False:
            print('Hello World!')
        '\n        The diffie-hellman-group14-sha1 key exchange algorithm is compatible\n        with OpenSSH.\n        '
        return self.assertExecuteWithKexAlgorithm('diffie-hellman-group14-sha1')

    def test_DH_GROUP_EXCHANGE_SHA1(self):
        if False:
            i = 10
            return i + 15
        '\n        The diffie-hellman-group-exchange-sha1 key exchange algorithm is\n        compatible with OpenSSH.\n        '
        return self.assertExecuteWithKexAlgorithm('diffie-hellman-group-exchange-sha1')

    def test_DH_GROUP_EXCHANGE_SHA256(self):
        if False:
            return 10
        '\n        The diffie-hellman-group-exchange-sha256 key exchange algorithm is\n        compatible with OpenSSH.\n        '
        return self.assertExecuteWithKexAlgorithm('diffie-hellman-group-exchange-sha256')

    def test_unsupported_algorithm(self):
        if False:
            print('Hello World!')
        '\n        The list of key exchange algorithms supported\n        by OpenSSH client is obtained with C{ssh -Q kex}.\n        '
        self.assertRaises(SkipTest, self.assertExecuteWithKexAlgorithm, 'unsupported-algorithm')

class OpenSSHClientForwardingTests(ForwardingMixin, OpenSSHClientMixin, TestCase):
    """
    Connection forwarding tests run against the OpenSSL command line client.
    """

    @skipIf(not HAS_IPV6, 'Requires IPv6 support')
    def test_localToRemoteForwardingV6(self):
        if False:
            return 10
        '\n        Forwarding of arbitrary IPv6 TCP connections via SSH.\n        '
        localPort = self._getFreePort()
        process = ConchTestForwardingProcess(localPort, b'test\n')
        d = self.execute('', process, sshArgs='-N -L%i:[::1]:%i' % (localPort, self.echoPortV6))
        d.addCallback(self.assertEqual, b'test\n')
        return d

class OpenSSHClientRekeyTests(RekeyTestsMixin, OpenSSHClientMixin, TestCase):
    """
    Rekeying tests run against the OpenSSL command line client.
    """

class CmdLineClientTests(ForwardingMixin, TestCase):
    """
    Connection forwarding tests run against the Conch command line client.
    """
    if runtime.platformType == 'win32':
        skip = "can't run cmdline client on win32"

    def execute(self, remoteCommand, process, sshArgs='', conchArgs=None):
        if False:
            while True:
                i = 10
        "\n        As for L{OpenSSHClientTestCase.execute}, except it runs the 'conch'\n        command line tool, not 'ssh'.\n        "
        if conchArgs is None:
            conchArgs = []
        process.deferred = defer.Deferred()
        port = self.conchServer.getHost().port
        cmd = '-p {} -l testuser --known-hosts kh_test --user-authentications publickey -a -i dsa_test -v '.format(port) + sshArgs + ' 127.0.0.1 ' + remoteCommand
        cmds = _makeArgs(conchArgs + cmd.split())
        env = os.environ.copy()
        env['PYTHONPATH'] = os.pathsep.join(sys.path)
        encodedCmds = []
        encodedEnv = {}
        for cmd in cmds:
            if isinstance(cmd, str):
                cmd = cmd.encode('utf-8')
            encodedCmds.append(cmd)
        for var in env:
            val = env[var]
            if isinstance(var, str):
                var = var.encode('utf-8')
            if isinstance(val, str):
                val = val.encode('utf-8')
            encodedEnv[var] = val
        reactor.spawnProcess(process, sys.executable, encodedCmds, env=encodedEnv)
        return process.deferred

    def test_runWithLogFile(self):
        if False:
            return 10
        '\n        It can store logs to a local file.\n        '

        def cb_check_log(result):
            if False:
                print('Hello World!')
            logContent = logPath.getContent()
            self.assertIn(b'Log opened.', logContent)
        logPath = filepath.FilePath(self.mktemp())
        d = self.execute(remoteCommand='echo goodbye', process=ConchTestOpenSSHProcess(), conchArgs=['--log', '--logfile', logPath.path, '--host-key-algorithms', 'ssh-rsa'])
        d.addCallback(self.assertEqual, b'goodbye\n')
        d.addCallback(cb_check_log)
        return d

    def test_runWithNoHostAlgorithmsSpecified(self):
        if False:
            print('Hello World!')
        '\n        Do not use --host-key-algorithms flag on command line.\n        '
        d = self.execute(remoteCommand='echo goodbye', process=ConchTestOpenSSHProcess())
        d.addCallback(self.assertEqual, b'goodbye\n')
        return d