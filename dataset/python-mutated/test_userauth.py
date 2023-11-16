"""
Tests for the implementation of the ssh-userauth service.

Maintainer: Paul Swartz
"""
from types import ModuleType
from typing import Optional
from zope.interface import implementer
from twisted.conch.error import ConchError, ValidPublicKey
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IAnonymous, ISSHPrivateKey, IUsernamePassword
from twisted.cred.error import UnauthorizedLogin
from twisted.cred.portal import IRealm, Portal
from twisted.internet import defer, task
from twisted.protocols import loopback
from twisted.python.reflect import requireModule
from twisted.trial import unittest
keys: Optional[ModuleType] = None
if requireModule('cryptography'):
    from twisted.conch.checkers import SSHProtocolChecker
    from twisted.conch.ssh import keys, transport, userauth
    from twisted.conch.ssh.common import NS
    from twisted.conch.test import keydata
else:

    class transport:

        class SSHTransportBase:
            """
            A stub class so that later class definitions won't die.
            """

    class userauth:

        class SSHUserAuthClient:
            """
            A stub class so that later class definitions won't die.
            """

class ClientUserAuth(userauth.SSHUserAuthClient):
    """
    A mock user auth client.
    """

    def getPublicKey(self):
        if False:
            i = 10
            return i + 15
        "\n        If this is the first time we've been called, return a blob for\n        the DSA key.  Otherwise, return a blob\n        for the RSA key.\n        "
        if self.lastPublicKey:
            return keys.Key.fromString(keydata.publicRSA_openssh)
        else:
            return defer.succeed(keys.Key.fromString(keydata.publicDSA_openssh))

    def getPrivateKey(self):
        if False:
            while True:
                i = 10
        '\n        Return the private key object for the RSA key.\n        '
        return defer.succeed(keys.Key.fromString(keydata.privateRSA_openssh))

    def getPassword(self, prompt=None):
        if False:
            print('Hello World!')
        "\n        Return 'foo' as the password.\n        "
        return defer.succeed(b'foo')

    def getGenericAnswers(self, name, information, answers):
        if False:
            print('Hello World!')
        "\n        Return 'foo' as the answer to two questions.\n        "
        return defer.succeed(('foo', 'foo'))

class OldClientAuth(userauth.SSHUserAuthClient):
    """
    The old SSHUserAuthClient returned a cryptography key object from
    getPrivateKey() and a string from getPublicKey
    """

    def getPrivateKey(self):
        if False:
            i = 10
            return i + 15
        return defer.succeed(keys.Key.fromString(keydata.privateRSA_openssh).keyObject)

    def getPublicKey(self):
        if False:
            return 10
        return keys.Key.fromString(keydata.publicRSA_openssh).blob()

class ClientAuthWithoutPrivateKey(userauth.SSHUserAuthClient):
    """
    This client doesn't have a private key, but it does have a public key.
    """

    def getPrivateKey(self):
        if False:
            for i in range(10):
                print('nop')
        return

    def getPublicKey(self):
        if False:
            while True:
                i = 10
        return keys.Key.fromString(keydata.publicRSA_openssh)

class FakeTransport(transport.SSHTransportBase):
    """
    L{userauth.SSHUserAuthServer} expects an SSH transport which has a factory
    attribute which has a portal attribute. Because the portal is important for
    testing authentication, we need to be able to provide an interesting portal
    object to the L{SSHUserAuthServer}.

    In addition, we want to be able to capture any packets sent over the
    transport.

    @ivar packets: a list of 2-tuples: (messageType, data).  Each 2-tuple is
        a sent packet.
    @type packets: C{list}
    @param lostConnecion: True if loseConnection has been called on us.
    @type lostConnection: L{bool}
    """

    class Service:
        """
        A mock service, representing the other service offered by the server.
        """
        name = b'nancy'

        def serviceStarted(self):
            if False:
                while True:
                    i = 10
            pass

    class Factory:
        """
        A mock factory, representing the factory that spawned this user auth
        service.
        """

        def getService(self, transport, service):
            if False:
                for i in range(10):
                    print('nop')
            '\n            Return our fake service.\n            '
            if service == b'none':
                return FakeTransport.Service

    def __init__(self, portal):
        if False:
            while True:
                i = 10
        self.factory = self.Factory()
        self.factory.portal = portal
        self.lostConnection = False
        self.transport = self
        self.packets = []

    def sendPacket(self, messageType, message):
        if False:
            i = 10
            return i + 15
        '\n        Record the packet sent by the service.\n        '
        self.packets.append((messageType, message))

    def isEncrypted(self, direction):
        if False:
            for i in range(10):
                print('nop')
        "\n        Pretend that this transport encrypts traffic in both directions. The\n        SSHUserAuthServer disables password authentication if the transport\n        isn't encrypted.\n        "
        return True

    def loseConnection(self):
        if False:
            print('Hello World!')
        self.lostConnection = True

@implementer(IRealm)
class Realm:
    """
    A mock realm for testing L{userauth.SSHUserAuthServer}.

    This realm is not actually used in the course of testing, so it returns the
    simplest thing that could possibly work.
    """

    def requestAvatar(self, avatarId, mind, *interfaces):
        if False:
            for i in range(10):
                print('nop')
        return defer.succeed((interfaces[0], None, lambda : None))

@implementer(ICredentialsChecker)
class PasswordChecker:
    """
    A very simple username/password checker which authenticates anyone whose
    password matches their username and rejects all others.
    """
    credentialInterfaces = (IUsernamePassword,)

    def requestAvatarId(self, creds):
        if False:
            for i in range(10):
                print('nop')
        if creds.username == creds.password:
            return defer.succeed(creds.username)
        return defer.fail(UnauthorizedLogin('Invalid username/password pair'))

@implementer(ICredentialsChecker)
class PrivateKeyChecker:
    """
    A very simple public key checker which authenticates anyone whose
    public/private keypair is the same keydata.public/privateRSA_openssh.
    """
    credentialInterfaces = (ISSHPrivateKey,)

    def requestAvatarId(self, creds):
        if False:
            print('Hello World!')
        if creds.blob == keys.Key.fromString(keydata.publicRSA_openssh).blob():
            if creds.signature is not None:
                obj = keys.Key.fromString(creds.blob)
                if obj.verify(creds.signature, creds.sigData):
                    return creds.username
            else:
                raise ValidPublicKey()
        raise UnauthorizedLogin()

@implementer(ICredentialsChecker)
class AnonymousChecker:
    """
    A simple checker which isn't supported by L{SSHUserAuthServer}.
    """
    credentialInterfaces = (IAnonymous,)

    def requestAvatarId(self, credentials):
        if False:
            print('Hello World!')
        pass

class SSHUserAuthServerTests(unittest.TestCase):
    """
    Tests for SSHUserAuthServer.
    """
    if keys is None:
        skip = 'cannot run without cryptography'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.realm = Realm()
        self.portal = Portal(self.realm)
        self.portal.registerChecker(PasswordChecker())
        self.portal.registerChecker(PrivateKeyChecker())
        self.authServer = userauth.SSHUserAuthServer()
        self.authServer.transport = FakeTransport(self.portal)
        self.authServer.serviceStarted()
        self.authServer.supportedAuthentications.sort()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        self.authServer.serviceStopped()
        self.authServer = None

    def _checkFailed(self, ignored):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the authentication has failed.\n        '
        self.assertEqual(self.authServer.transport.packets[-1], (userauth.MSG_USERAUTH_FAILURE, NS(b'password,publickey') + b'\x00'))

    def test_noneAuthentication(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        A client may request a list of authentication \'method name\' values\n        that may continue by using the "none" authentication \'method name\'.\n\n        See RFC 4252 Section 5.2.\n        '
        d = self.authServer.ssh_USERAUTH_REQUEST(NS(b'foo') + NS(b'service') + NS(b'none'))
        return d.addCallback(self._checkFailed)

    def test_successfulPasswordAuthentication(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When provided with correct password authentication information, the\n        server should respond by sending a MSG_USERAUTH_SUCCESS message with\n        no other data.\n\n        See RFC 4252, Section 5.1.\n        '
        packet = b''.join([NS(b'foo'), NS(b'none'), NS(b'password'), b'\x00', NS(b'foo')])
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)

        def check(ignored):
            if False:
                while True:
                    i = 10
            self.assertEqual(self.authServer.transport.packets, [(userauth.MSG_USERAUTH_SUCCESS, b'')])
        return d.addCallback(check)

    def test_failedPasswordAuthentication(self):
        if False:
            i = 10
            return i + 15
        '\n        When provided with invalid authentication details, the server should\n        respond by sending a MSG_USERAUTH_FAILURE message which states whether\n        the authentication was partially successful, and provides other, open\n        options for authentication.\n\n        See RFC 4252, Section 5.1.\n        '
        packet = b''.join([NS(b'foo'), NS(b'none'), NS(b'password'), b'\x00', NS(b'bar')])
        self.authServer.clock = task.Clock()
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)
        self.assertEqual(self.authServer.transport.packets, [])
        self.authServer.clock.advance(2)
        return d.addCallback(self._checkFailed)

    def test_successfulPrivateKeyAuthentication(self):
        if False:
            print('Hello World!')
        '\n        Test that private key authentication completes successfully,\n        '
        blob = keys.Key.fromString(keydata.publicRSA_openssh).blob()
        obj = keys.Key.fromString(keydata.privateRSA_openssh)
        packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\xff' + NS(obj.sshType()) + NS(blob)
        self.authServer.transport.sessionID = b'test'
        signature = obj.sign(NS(b'test') + bytes((userauth.MSG_USERAUTH_REQUEST,)) + packet)
        packet += NS(signature)
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)

        def check(ignored):
            if False:
                print('Hello World!')
            self.assertEqual(self.authServer.transport.packets, [(userauth.MSG_USERAUTH_SUCCESS, b'')])
        return d.addCallback(check)

    def test_requestRaisesConchError(self):
        if False:
            return 10
        '\n        ssh_USERAUTH_REQUEST should raise a ConchError if tryAuth returns\n        None. Added to catch a bug noticed by pyflakes.\n        '
        d = defer.Deferred()

        def mockCbFinishedAuth(self, ignored):
            if False:
                print('Hello World!')
            self.fail('request should have raised ConochError')

        def mockTryAuth(kind, user, data):
            if False:
                print('Hello World!')
            return None

        def mockEbBadAuth(reason):
            if False:
                while True:
                    i = 10
            d.errback(reason.value)
        self.patch(self.authServer, 'tryAuth', mockTryAuth)
        self.patch(self.authServer, '_cbFinishedAuth', mockCbFinishedAuth)
        self.patch(self.authServer, '_ebBadAuth', mockEbBadAuth)
        packet = NS(b'user') + NS(b'none') + NS(b'public-key') + NS(b'data')
        self.authServer.ssh_USERAUTH_REQUEST(packet)
        return self.assertFailure(d, ConchError)

    def test_verifyValidPrivateKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that verifying a valid private key works.\n        '
        blob = keys.Key.fromString(keydata.publicRSA_openssh).blob()
        packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\x00' + NS(b'ssh-rsa') + NS(blob)
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)

        def check(ignored):
            if False:
                return 10
            self.assertEqual(self.authServer.transport.packets, [(userauth.MSG_USERAUTH_PK_OK, NS(b'ssh-rsa') + NS(blob))])
        return d.addCallback(check)

    def test_failedPrivateKeyAuthenticationWithoutSignature(self):
        if False:
            return 10
        '\n        Test that private key authentication fails when the public key\n        is invalid.\n        '
        blob = keys.Key.fromString(keydata.publicDSA_openssh).blob()
        packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\x00' + NS(b'ssh-dsa') + NS(blob)
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)
        return d.addCallback(self._checkFailed)

    def test_failedPrivateKeyAuthenticationWithSignature(self):
        if False:
            print('Hello World!')
        '\n        Test that private key authentication fails when the public key\n        is invalid.\n        '
        blob = keys.Key.fromString(keydata.publicRSA_openssh).blob()
        obj = keys.Key.fromString(keydata.privateRSA_openssh)
        packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\xff' + NS(b'ssh-rsa') + NS(blob) + NS(obj.sign(blob))
        self.authServer.transport.sessionID = b'test'
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)
        return d.addCallback(self._checkFailed)

    def test_unsupported_publickey(self):
        if False:
            return 10
        '\n        Private key authentication fails when the public key type is\n        unsupported or the public key is corrupt.\n        '
        blob = keys.Key.fromString(keydata.publicDSA_openssh).blob()
        blob = NS(b'ssh-bad-type') + blob[11:]
        packet = NS(b'foo') + NS(b'none') + NS(b'publickey') + b'\x00' + NS(b'ssh-rsa') + NS(blob)
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)
        return d.addCallback(self._checkFailed)

    def test_ignoreUnknownCredInterfaces(self):
        if False:
            while True:
                i = 10
        "\n        L{SSHUserAuthServer} sets up\n        C{SSHUserAuthServer.supportedAuthentications} by checking the portal's\n        credentials interfaces and mapping them to SSH authentication method\n        strings.  If the Portal advertises an interface that\n        L{SSHUserAuthServer} can't map, it should be ignored.  This is a white\n        box test.\n        "
        server = userauth.SSHUserAuthServer()
        server.transport = FakeTransport(self.portal)
        self.portal.registerChecker(AnonymousChecker())
        server.serviceStarted()
        server.serviceStopped()
        server.supportedAuthentications.sort()
        self.assertEqual(server.supportedAuthentications, [b'password', b'publickey'])

    def test_removePasswordIfUnencrypted(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that the userauth service does not advertise password\n        authentication if the password would be send in cleartext.\n        '
        self.assertIn(b'password', self.authServer.supportedAuthentications)
        clearAuthServer = userauth.SSHUserAuthServer()
        clearAuthServer.transport = FakeTransport(self.portal)
        clearAuthServer.transport.isEncrypted = lambda x: False
        clearAuthServer.serviceStarted()
        clearAuthServer.serviceStopped()
        self.assertNotIn(b'password', clearAuthServer.supportedAuthentications)
        halfAuthServer = userauth.SSHUserAuthServer()
        halfAuthServer.transport = FakeTransport(self.portal)
        halfAuthServer.transport.isEncrypted = lambda x: x == 'in'
        halfAuthServer.serviceStarted()
        halfAuthServer.serviceStopped()
        self.assertIn(b'password', halfAuthServer.supportedAuthentications)

    def test_unencryptedConnectionWithoutPasswords(self):
        if False:
            print('Hello World!')
        '\n        If the L{SSHUserAuthServer} is not advertising passwords, then an\n        unencrypted connection should not cause any warnings or exceptions.\n        This is a white box test.\n        '
        portal = Portal(self.realm)
        portal.registerChecker(PrivateKeyChecker())
        clearAuthServer = userauth.SSHUserAuthServer()
        clearAuthServer.transport = FakeTransport(portal)
        clearAuthServer.transport.isEncrypted = lambda x: False
        clearAuthServer.serviceStarted()
        clearAuthServer.serviceStopped()
        self.assertEqual(clearAuthServer.supportedAuthentications, [b'publickey'])
        halfAuthServer = userauth.SSHUserAuthServer()
        halfAuthServer.transport = FakeTransport(portal)
        halfAuthServer.transport.isEncrypted = lambda x: x == 'in'
        halfAuthServer.serviceStarted()
        halfAuthServer.serviceStopped()
        self.assertEqual(clearAuthServer.supportedAuthentications, [b'publickey'])

    def test_loginTimeout(self):
        if False:
            while True:
                i = 10
        '\n        Test that the login times out.\n        '
        timeoutAuthServer = userauth.SSHUserAuthServer()
        timeoutAuthServer.clock = task.Clock()
        timeoutAuthServer.transport = FakeTransport(self.portal)
        timeoutAuthServer.serviceStarted()
        timeoutAuthServer.clock.advance(11 * 60 * 60)
        timeoutAuthServer.serviceStopped()
        self.assertEqual(timeoutAuthServer.transport.packets, [(transport.MSG_DISCONNECT, b'\x00' * 3 + bytes((transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE,)) + NS(b'you took too long') + NS(b''))])
        self.assertTrue(timeoutAuthServer.transport.lostConnection)

    def test_cancelLoginTimeout(self):
        if False:
            while True:
                i = 10
        '\n        Test that stopping the service also stops the login timeout.\n        '
        timeoutAuthServer = userauth.SSHUserAuthServer()
        timeoutAuthServer.clock = task.Clock()
        timeoutAuthServer.transport = FakeTransport(self.portal)
        timeoutAuthServer.serviceStarted()
        timeoutAuthServer.serviceStopped()
        timeoutAuthServer.clock.advance(11 * 60 * 60)
        self.assertEqual(timeoutAuthServer.transport.packets, [])
        self.assertFalse(timeoutAuthServer.transport.lostConnection)

    def test_tooManyAttempts(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that the server disconnects if the client fails authentication\n        too many times.\n        '
        packet = b''.join([NS(b'foo'), NS(b'none'), NS(b'password'), b'\x00', NS(b'bar')])
        self.authServer.clock = task.Clock()
        for i in range(21):
            d = self.authServer.ssh_USERAUTH_REQUEST(packet)
            self.authServer.clock.advance(2)

        def check(ignored):
            if False:
                while True:
                    i = 10
            self.assertEqual(self.authServer.transport.packets[-1], (transport.MSG_DISCONNECT, b'\x00' * 3 + bytes((transport.DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE,)) + NS(b'too many bad auths') + NS(b'')))
        return d.addCallback(check)

    def test_failIfUnknownService(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If the user requests a service that we don't support, the\n        authentication should fail.\n        "
        packet = NS(b'foo') + NS(b'') + NS(b'password') + b'\x00' + NS(b'foo')
        self.authServer.clock = task.Clock()
        d = self.authServer.ssh_USERAUTH_REQUEST(packet)
        return d.addCallback(self._checkFailed)

    def test_tryAuthEdgeCases(self):
        if False:
            i = 10
            return i + 15
        '\n        tryAuth() has two edge cases that are difficult to reach.\n\n        1) an authentication method auth_* returns None instead of a Deferred.\n        2) an authentication type that is defined does not have a matching\n           auth_* method.\n\n        Both these cases should return a Deferred which fails with a\n        ConchError.\n        '

        def mockAuth(packet):
            if False:
                while True:
                    i = 10
            return None
        self.patch(self.authServer, 'auth_publickey', mockAuth)
        self.patch(self.authServer, 'auth_password', None)

        def secondTest(ignored):
            if False:
                print('Hello World!')
            d2 = self.authServer.tryAuth(b'password', None, None)
            return self.assertFailure(d2, ConchError)
        d1 = self.authServer.tryAuth(b'publickey', None, None)
        return self.assertFailure(d1, ConchError).addCallback(secondTest)

class SSHUserAuthClientTests(unittest.TestCase):
    """
    Tests for SSHUserAuthClient.
    """
    if keys is None:
        skip = 'cannot run without cryptography'

    def setUp(self):
        if False:
            while True:
                i = 10
        self.authClient = ClientUserAuth(b'foo', FakeTransport.Service())
        self.authClient.transport = FakeTransport(None)
        self.authClient.transport.sessionID = b'test'
        self.authClient.serviceStarted()

    def tearDown(self):
        if False:
            return 10
        self.authClient.serviceStopped()
        self.authClient = None

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that client is initialized properly.\n        '
        self.assertEqual(self.authClient.user, b'foo')
        self.assertEqual(self.authClient.instance.name, b'nancy')
        self.assertEqual(self.authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_USERAUTH_SUCCESS(self):
        if False:
            print('Hello World!')
        '\n        Test that the client succeeds properly.\n        '
        instance = [None]

        def stubSetService(service):
            if False:
                i = 10
                return i + 15
            instance[0] = service
        self.authClient.transport.setService = stubSetService
        self.authClient.ssh_USERAUTH_SUCCESS(b'')
        self.assertEqual(instance[0], self.authClient.instance)

    def test_publickey(self):
        if False:
            return 10
        '\n        Test that the client can authenticate with a public key.\n        '
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'publickey') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x00' + NS(b'ssh-dss') + NS(keys.Key.fromString(keydata.publicDSA_openssh).blob())))
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'publickey') + b'\x00')
        blob = NS(keys.Key.fromString(keydata.publicRSA_openssh).blob())
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x00' + NS(b'ssh-rsa') + blob))
        self.authClient.ssh_USERAUTH_PK_OK(NS(b'ssh-rsa') + NS(keys.Key.fromString(keydata.publicRSA_openssh).blob()))
        sigData = NS(self.authClient.transport.sessionID) + bytes((userauth.MSG_USERAUTH_REQUEST,)) + NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x01' + NS(b'ssh-rsa') + blob
        obj = keys.Key.fromString(keydata.privateRSA_openssh)
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'publickey') + b'\x01' + NS(b'ssh-rsa') + blob + NS(obj.sign(sigData))))

    def test_publickey_without_privatekey(self):
        if False:
            i = 10
            return i + 15
        "\n        If the SSHUserAuthClient doesn't return anything from signData,\n        the client should start the authentication over again by requesting\n        'none' authentication.\n        "
        authClient = ClientAuthWithoutPrivateKey(b'foo', FakeTransport.Service())
        authClient.transport = FakeTransport(None)
        authClient.transport.sessionID = b'test'
        authClient.serviceStarted()
        authClient.tryAuth(b'publickey')
        authClient.transport.packets = []
        self.assertIsNone(authClient.ssh_USERAUTH_PK_OK(b''))
        self.assertEqual(authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_no_publickey(self):
        if False:
            return 10
        "\n        If there's no public key, auth_publickey should return a Deferred\n        called back with a False value.\n        "
        self.authClient.getPublicKey = lambda x: None
        d = self.authClient.tryAuth(b'publickey')

        def check(result):
            if False:
                i = 10
                return i + 15
            self.assertFalse(result)
        return d.addCallback(check)

    def test_password(self):
        if False:
            while True:
                i = 10
        '\n        Test that the client can authentication with a password.  This\n        includes changing the password.\n        '
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\x00' + NS(b'foo')))
        self.authClient.ssh_USERAUTH_PK_OK(NS(b'') + NS(b''))
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\xff' + NS(b'foo') * 2))

    def test_no_password(self):
        if False:
            return 10
        '\n        If getPassword returns None, tryAuth should return False.\n        '
        self.authClient.getPassword = lambda : None
        self.assertFalse(self.authClient.tryAuth(b'password'))

    def test_keyboardInteractive(self):
        if False:
            print('Hello World!')
        '\n        Make sure that the client can authenticate with the keyboard\n        interactive method.\n        '
        self.authClient.ssh_USERAUTH_PK_OK_keyboard_interactive(NS(b'') + NS(b'') + NS(b'') + b'\x00\x00\x00\x01' + NS(b'Password: ') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_INFO_RESPONSE, b'\x00\x00\x00\x02' + NS(b'foo') + NS(b'foo')))

    def test_USERAUTH_PK_OK_unknown_method(self):
        if False:
            i = 10
            return i + 15
        "\n        If C{SSHUserAuthClient} gets a MSG_USERAUTH_PK_OK packet when it's not\n        expecting it, it should fail the current authentication and move on to\n        the next type.\n        "
        self.authClient.lastAuth = b'unknown'
        self.authClient.transport.packets = []
        self.authClient.ssh_USERAUTH_PK_OK(b'')
        self.assertEqual(self.authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_USERAUTH_FAILURE_sorting(self):
        if False:
            i = 10
            return i + 15
        '\n        ssh_USERAUTH_FAILURE should sort the methods by their position\n        in SSHUserAuthClient.preferredOrder.  Methods that are not in\n        preferredOrder should be sorted at the end of that list.\n        '

        def auth_firstmethod():
            if False:
                i = 10
                return i + 15
            self.authClient.transport.sendPacket(255, b'here is data')

        def auth_anothermethod():
            if False:
                i = 10
                return i + 15
            self.authClient.transport.sendPacket(254, b'other data')
            return True
        self.authClient.auth_firstmethod = auth_firstmethod
        self.authClient.auth_anothermethod = auth_anothermethod
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'anothermethod,password') + b'\x00')
        self.assertEqual(self.authClient.transport.packets[-1], (userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'password') + b'\x00' + NS(b'foo')))
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'firstmethod,anothermethod,password') + b'\xff')
        self.assertEqual(self.authClient.transport.packets[-2:], [(255, b'here is data'), (254, b'other data')])

    def test_disconnectIfNoMoreAuthentication(self):
        if False:
            return 10
        '\n        If there are no more available user authentication messages,\n        the SSHUserAuthClient should disconnect with code\n        DISCONNECT_NO_MORE_AUTH_METHODS_AVAILABLE.\n        '
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\x00')
        self.authClient.ssh_USERAUTH_FAILURE(NS(b'password') + b'\xff')
        self.assertEqual(self.authClient.transport.packets[-1], (transport.MSG_DISCONNECT, b'\x00\x00\x00\x0e' + NS(b'no more authentication methods available') + b'\x00\x00\x00\x00'))

    def test_ebAuth(self):
        if False:
            return 10
        "\n        _ebAuth (the generic authentication error handler) should send\n        a request for the 'none' authentication method.\n        "
        self.authClient.transport.packets = []
        self.authClient._ebAuth(None)
        self.assertEqual(self.authClient.transport.packets, [(userauth.MSG_USERAUTH_REQUEST, NS(b'foo') + NS(b'nancy') + NS(b'none'))])

    def test_defaults(self):
        if False:
            while True:
                i = 10
        '\n        getPublicKey() should return None.  getPrivateKey() should return a\n        failed Deferred.  getPassword() should return a failed Deferred.\n        getGenericAnswers() should return a failed Deferred.\n        '
        authClient = userauth.SSHUserAuthClient(b'foo', FakeTransport.Service())
        self.assertIsNone(authClient.getPublicKey())

        def check(result):
            if False:
                print('Hello World!')
            result.trap(NotImplementedError)
            d = authClient.getPassword()
            return d.addCallback(self.fail).addErrback(check2)

        def check2(result):
            if False:
                return 10
            result.trap(NotImplementedError)
            d = authClient.getGenericAnswers(None, None, None)
            return d.addCallback(self.fail).addErrback(check3)

        def check3(result):
            if False:
                for i in range(10):
                    print('nop')
            result.trap(NotImplementedError)
        d = authClient.getPrivateKey()
        return d.addCallback(self.fail).addErrback(check)

class LoopbackTests(unittest.TestCase):
    if keys is None:
        skip = 'cannot run without cryptography'

    class Factory:

        class Service:
            name = b'TestService'

            def serviceStarted(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.transport.loseConnection()

            def serviceStopped(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

        def getService(self, avatar, name):
            if False:
                return 10
            return self.Service

    def test_loopback(self):
        if False:
            return 10
        '\n        Test that the userauth server and client play nicely with each other.\n        '
        server = userauth.SSHUserAuthServer()
        client = ClientUserAuth(b'foo', self.Factory.Service())
        server.transport = transport.SSHTransportBase()
        server.transport.service = server
        server.transport.isEncrypted = lambda x: True
        client.transport = transport.SSHTransportBase()
        client.transport.service = client
        server.transport.sessionID = client.transport.sessionID = b''
        server.transport.sendKexInit = client.transport.sendKexInit = lambda : None
        server.transport.factory = self.Factory()
        server.passwordDelay = 0
        realm = Realm()
        portal = Portal(realm)
        checker = SSHProtocolChecker()
        checker.registerChecker(PasswordChecker())
        checker.registerChecker(PrivateKeyChecker())
        checker.areDone = lambda aId: len(checker.successfulCredentials[aId]) == 2
        portal.registerChecker(checker)
        server.transport.factory.portal = portal
        d = loopback.loopbackAsync(server.transport, client.transport)
        server.transport.transport.logPrefix = lambda : '_ServerLoopback'
        client.transport.transport.logPrefix = lambda : '_ClientLoopback'
        server.serviceStarted()
        client.serviceStarted()

        def check(ignored):
            if False:
                print('Hello World!')
            self.assertEqual(server.transport.service.name, b'TestService')
        return d.addCallback(check)

class ModuleInitializationTests(unittest.TestCase):
    if keys is None:
        skip = 'cannot run without cryptography'

    def test_messages(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(userauth.SSHUserAuthServer.protocolMessages[60], 'MSG_USERAUTH_PK_OK')
        self.assertEqual(userauth.SSHUserAuthClient.protocolMessages[60], 'MSG_USERAUTH_PK_OK')