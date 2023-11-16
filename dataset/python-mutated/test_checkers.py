"""
Tests for L{twisted.conch.checkers}.
"""
import os
from base64 import encodebytes
from collections import namedtuple
from io import BytesIO
from typing import Optional
cryptSkip: Optional[str]
try:
    import crypt
except ImportError:
    cryptSkip = 'cannot run without crypt module'
else:
    cryptSkip = None
from zope.interface.verify import verifyObject
from twisted.cred.checkers import InMemoryUsernamePasswordDatabaseDontUse
from twisted.cred.credentials import ISSHPrivateKey, IUsernamePassword, SSHPrivateKey, UsernamePassword
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet.defer import Deferred
from twisted.python import util
from twisted.python.fakepwd import ShadowDatabase, UserDatabase
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
if requireModule('cryptography'):
    dependencySkip = None
    from twisted.conch import checkers
    from twisted.conch.error import NotEnoughAuthentication, ValidPublicKey
    from twisted.conch.ssh import keys
    from twisted.conch.test import keydata
else:
    dependencySkip = "can't run without cryptography"
if getattr(os, 'geteuid', None) is not None:
    euidSkip = None
else:
    euidSkip = 'Cannot run without effective UIDs (questionable)'

class HelperTests(TestCase):
    """
    Tests for helper functions L{verifyCryptedPassword}, L{_pwdGetByName} and
    L{_shadowGetByName}.
    """
    skip = cryptSkip or dependencySkip

    def setUp(self):
        if False:
            return 10
        self.mockos = MockOS()

    def test_verifyCryptedPassword(self):
        if False:
            print('Hello World!')
        '\n        L{verifyCryptedPassword} returns C{True} if the plaintext password\n        passed to it matches the encrypted password passed to it.\n        '
        password = 'secret string'
        salt = 'salty'
        crypted = crypt.crypt(password, salt)
        self.assertTrue(checkers.verifyCryptedPassword(crypted, password), '{!r} supposed to be valid encrypted password for {!r}'.format(crypted, password))

    def test_verifyCryptedPasswordMD5(self):
        if False:
            i = 10
            return i + 15
        '\n        L{verifyCryptedPassword} returns True if the provided cleartext password\n        matches the provided MD5 password hash.\n        '
        password = 'password'
        salt = '$1$salt'
        crypted = crypt.crypt(password, salt)
        self.assertTrue(checkers.verifyCryptedPassword(crypted, password), '{!r} supposed to be valid encrypted password for {}'.format(crypted, password))

    def test_refuteCryptedPassword(self):
        if False:
            i = 10
            return i + 15
        '\n        L{verifyCryptedPassword} returns C{False} if the plaintext password\n        passed to it does not match the encrypted password passed to it.\n        '
        password = 'string secret'
        wrong = 'secret string'
        crypted = crypt.crypt(password, password)
        self.assertFalse(checkers.verifyCryptedPassword(crypted, wrong), '{!r} not supposed to be valid encrypted password for {}'.format(crypted, wrong))

    def test_pwdGetByName(self):
        if False:
            while True:
                i = 10
        '\n        L{_pwdGetByName} returns a tuple of items from the UNIX /etc/passwd\n        database if the L{pwd} module is present.\n        '
        userdb = UserDatabase()
        userdb.addUser('alice', 'secrit', 1, 2, 'first last', '/foo', '/bin/sh')
        self.patch(checkers, 'pwd', userdb)
        self.assertEqual(checkers._pwdGetByName('alice'), userdb.getpwnam('alice'))

    def test_pwdGetByNameWithoutPwd(self):
        if False:
            return 10
        "\n        If the C{pwd} module isn't present, L{_pwdGetByName} returns L{None}.\n        "
        self.patch(checkers, 'pwd', None)
        self.assertIsNone(checkers._pwdGetByName('alice'))

    def test_shadowGetByName(self):
        if False:
            i = 10
            return i + 15
        '\n        L{_shadowGetByName} returns a tuple of items from the UNIX /etc/shadow\n        database if the L{spwd} is present.\n        '
        userdb = ShadowDatabase()
        userdb.addUser('bob', 'passphrase', 1, 2, 3, 4, 5, 6, 7)
        self.patch(checkers, 'spwd', userdb)
        self.mockos.euid = 2345
        self.mockos.egid = 1234
        self.patch(util, 'os', self.mockos)
        self.assertEqual(checkers._shadowGetByName('bob'), userdb.getspnam('bob'))
        self.assertEqual(self.mockos.seteuidCalls, [0, 2345])
        self.assertEqual(self.mockos.setegidCalls, [0, 1234])

    def test_shadowGetByNameWithoutSpwd(self):
        if False:
            i = 10
            return i + 15
        '\n        L{_shadowGetByName} returns L{None} if C{spwd} is not present.\n        '
        self.patch(checkers, 'spwd', None)
        self.assertIsNone(checkers._shadowGetByName('bob'))
        self.assertEqual(self.mockos.seteuidCalls, [])
        self.assertEqual(self.mockos.setegidCalls, [])

class SSHPublicKeyDatabaseTests(TestCase):
    """
    Tests for L{SSHPublicKeyDatabase}.
    """
    skip = euidSkip or dependencySkip

    def setUp(self) -> None:
        if False:
            i = 10
            return i + 15
        self.checker = checkers.SSHPublicKeyDatabase()
        self.key1 = encodebytes(b'foobar')
        self.key2 = encodebytes(b'eggspam')
        self.content = b't1 ' + self.key1 + b' foo\nt2 ' + self.key2 + b' egg\n'
        self.mockos = MockOS()
        self.patch(util, 'os', self.mockos)
        self.path = FilePath(self.mktemp())
        assert isinstance(self.path.path, str)
        self.sshDir = self.path.child('.ssh')
        self.sshDir.makedirs()
        userdb = UserDatabase()
        userdb.addUser('user', 'password', 1, 2, 'first last', self.path.path, '/bin/shell')
        self.checker._userdb = userdb

    def test_deprecated(self):
        if False:
            print('Hello World!')
        '\n        L{SSHPublicKeyDatabase} is deprecated as of version 15.0\n        '
        warningsShown = self.flushWarnings(offendingFunctions=[self.setUp])
        self.assertEqual(warningsShown[0]['category'], DeprecationWarning)
        self.assertEqual(warningsShown[0]['message'], 'twisted.conch.checkers.SSHPublicKeyDatabase was deprecated in Twisted 15.0.0: Please use twisted.conch.checkers.SSHPublicKeyChecker, initialized with an instance of twisted.conch.checkers.UNIXAuthorizedKeysFiles instead.')
        self.assertEqual(len(warningsShown), 1)

    def _testCheckKey(self, filename):
        if False:
            i = 10
            return i + 15
        self.sshDir.child(filename).setContent(self.content)
        user = UsernamePassword(b'user', b'password')
        user.blob = b'foobar'
        self.assertTrue(self.checker.checkKey(user))
        user.blob = b'eggspam'
        self.assertTrue(self.checker.checkKey(user))
        user.blob = b'notallowed'
        self.assertFalse(self.checker.checkKey(user))

    def test_checkKey(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SSHPublicKeyDatabase.checkKey} should retrieve the content of the\n        authorized_keys file and check the keys against that file.\n        '
        self._testCheckKey('authorized_keys')
        self.assertEqual(self.mockos.seteuidCalls, [])
        self.assertEqual(self.mockos.setegidCalls, [])

    def test_checkKey2(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{SSHPublicKeyDatabase.checkKey} should retrieve the content of the\n        authorized_keys2 file and check the keys against that file.\n        '
        self._testCheckKey('authorized_keys2')
        self.assertEqual(self.mockos.seteuidCalls, [])
        self.assertEqual(self.mockos.setegidCalls, [])

    def test_checkKeyAsRoot(self):
        if False:
            while True:
                i = 10
        '\n        If the key file is readable, L{SSHPublicKeyDatabase.checkKey} should\n        switch its uid/gid to the ones of the authenticated user.\n        '
        keyFile = self.sshDir.child('authorized_keys')
        keyFile.setContent(self.content)
        keyFile.chmod(0)
        self.addCleanup(keyFile.chmod, 511)
        savedSeteuid = self.mockos.seteuid

        def seteuid(euid):
            if False:
                i = 10
                return i + 15
            keyFile.chmod(511)
            return savedSeteuid(euid)
        self.mockos.euid = 2345
        self.mockos.egid = 1234
        self.patch(self.mockos, 'seteuid', seteuid)
        self.patch(util, 'os', self.mockos)
        user = UsernamePassword(b'user', b'password')
        user.blob = b'foobar'
        self.assertTrue(self.checker.checkKey(user))
        self.assertEqual(self.mockos.seteuidCalls, [0, 1, 0, 2345])
        self.assertEqual(self.mockos.setegidCalls, [2, 1234])

    def test_requestAvatarId(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{SSHPublicKeyDatabase.requestAvatarId} should return the avatar id\n        passed in if its C{_checkKey} method returns True.\n        '

        def _checkKey(ignored):
            if False:
                while True:
                    i = 10
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', b'ssh-rsa', keydata.publicRSA_openssh, b'foo', keys.Key.fromString(keydata.privateRSA_openssh).sign(b'foo'))
        d = self.checker.requestAvatarId(credentials)

        def _verify(avatarId):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(avatarId, b'test')
        return d.addCallback(_verify)

    def test_requestAvatarIdWithoutSignature(self):
        if False:
            while True:
                i = 10
        '\n        L{SSHPublicKeyDatabase.requestAvatarId} should raise L{ValidPublicKey}\n        if the credentials represent a valid key without a signature.  This\n        tells the user that the key is valid for login, but does not actually\n        allow that user to do so without a signature.\n        '

        def _checkKey(ignored):
            if False:
                while True:
                    i = 10
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', b'ssh-rsa', keydata.publicRSA_openssh, None, None)
        d = self.checker.requestAvatarId(credentials)
        return self.assertFailure(d, ValidPublicKey)

    def test_requestAvatarIdInvalidKey(self):
        if False:
            i = 10
            return i + 15
        '\n        If L{SSHPublicKeyDatabase.checkKey} returns False,\n        C{_cbRequestAvatarId} should raise L{UnauthorizedLogin}.\n        '

        def _checkKey(ignored):
            if False:
                for i in range(10):
                    print('nop')
            return False
        self.patch(self.checker, 'checkKey', _checkKey)
        d = self.checker.requestAvatarId(None)
        return self.assertFailure(d, UnauthorizedLogin)

    def test_requestAvatarIdInvalidSignature(self):
        if False:
            while True:
                i = 10
        '\n        Valid keys with invalid signatures should cause\n        L{SSHPublicKeyDatabase.requestAvatarId} to return a {UnauthorizedLogin}\n        failure\n        '

        def _checkKey(ignored):
            if False:
                return 10
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', b'ssh-rsa', keydata.publicRSA_openssh, b'foo', keys.Key.fromString(keydata.privateDSA_openssh).sign(b'foo'))
        d = self.checker.requestAvatarId(credentials)
        return self.assertFailure(d, UnauthorizedLogin)

    def test_requestAvatarIdNormalizeException(self):
        if False:
            i = 10
            return i + 15
        '\n        Exceptions raised while verifying the key should be normalized into an\n        C{UnauthorizedLogin} failure.\n        '

        def _checkKey(ignored):
            if False:
                while True:
                    i = 10
            return True
        self.patch(self.checker, 'checkKey', _checkKey)
        credentials = SSHPrivateKey(b'test', None, b'blob', b'sigData', b'sig')
        d = self.checker.requestAvatarId(credentials)

        def _verifyLoggedException(failure):
            if False:
                while True:
                    i = 10
            errors = self.flushLoggedErrors(keys.BadKeyError)
            self.assertEqual(len(errors), 1)
            return failure
        d.addErrback(_verifyLoggedException)
        return self.assertFailure(d, UnauthorizedLogin)

class SSHProtocolCheckerTests(TestCase):
    """
    Tests for L{SSHProtocolChecker}.
    """
    skip = dependencySkip

    def test_registerChecker(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SSHProcotolChecker.registerChecker} should add the given checker to\n        the list of registered checkers.\n        '
        checker = checkers.SSHProtocolChecker()
        self.assertEqual(checker.credentialInterfaces, [])
        checker.registerChecker(checkers.SSHPublicKeyDatabase())
        self.assertEqual(checker.credentialInterfaces, [ISSHPrivateKey])
        self.assertIsInstance(checker.checkers[ISSHPrivateKey], checkers.SSHPublicKeyDatabase)

    def test_registerCheckerWithInterface(self):
        if False:
            i = 10
            return i + 15
        '\n        If a specific interface is passed into\n        L{SSHProtocolChecker.registerChecker}, that interface should be\n        registered instead of what the checker specifies in\n        credentialIntefaces.\n        '
        checker = checkers.SSHProtocolChecker()
        self.assertEqual(checker.credentialInterfaces, [])
        checker.registerChecker(checkers.SSHPublicKeyDatabase(), IUsernamePassword)
        self.assertEqual(checker.credentialInterfaces, [IUsernamePassword])
        self.assertIsInstance(checker.checkers[IUsernamePassword], checkers.SSHPublicKeyDatabase)

    def test_requestAvatarId(self):
        if False:
            return 10
        '\n        L{SSHProtocolChecker.requestAvatarId} should defer to one if its\n        registered checkers to authenticate a user.\n        '
        checker = checkers.SSHProtocolChecker()
        passwordDatabase = InMemoryUsernamePasswordDatabaseDontUse()
        passwordDatabase.addUser(b'test', b'test')
        checker.registerChecker(passwordDatabase)
        d = checker.requestAvatarId(UsernamePassword(b'test', b'test'))

        def _callback(avatarId):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(avatarId, b'test')
        return d.addCallback(_callback)

    def test_requestAvatarIdWithNotEnoughAuthentication(self):
        if False:
            i = 10
            return i + 15
        '\n        If the client indicates that it is never satisfied, by always returning\n        False from _areDone, then L{SSHProtocolChecker} should raise\n        L{NotEnoughAuthentication}.\n        '
        checker = checkers.SSHProtocolChecker()

        def _areDone(avatarId):
            if False:
                print('Hello World!')
            return False
        self.patch(checker, 'areDone', _areDone)
        passwordDatabase = InMemoryUsernamePasswordDatabaseDontUse()
        passwordDatabase.addUser(b'test', b'test')
        checker.registerChecker(passwordDatabase)
        d = checker.requestAvatarId(UsernamePassword(b'test', b'test'))
        return self.assertFailure(d, NotEnoughAuthentication)

    def test_requestAvatarIdInvalidCredential(self):
        if False:
            i = 10
            return i + 15
        "\n        If the passed credentials aren't handled by any registered checker,\n        L{SSHProtocolChecker} should raise L{UnhandledCredentials}.\n        "
        checker = checkers.SSHProtocolChecker()
        d = checker.requestAvatarId(UsernamePassword(b'test', b'test'))
        return self.assertFailure(d, UnhandledCredentials)

    def test_areDone(self):
        if False:
            print('Hello World!')
        '\n        The default L{SSHProcotolChecker.areDone} should simply return True.\n        '
        self.assertTrue(checkers.SSHProtocolChecker().areDone(None))

class UNIXPasswordDatabaseTests(TestCase):
    """
    Tests for L{UNIXPasswordDatabase}.
    """
    skip = cryptSkip or dependencySkip

    def assertLoggedIn(self, d: Deferred[bytes], username: bytes) -> None:
        if False:
            print('Hello World!')
        "\n        Assert that the L{Deferred} passed in is called back with the value\n        'username'.  This represents a valid login for this TestCase.\n\n        @param d: a L{Deferred} from an L{IChecker.requestAvatarId} method.\n        "
        self.assertEqual(self.successResultOf(d), username)

    def test_defaultCheckers(self):
        if False:
            print('Hello World!')
        '\n        L{UNIXPasswordDatabase} with no arguments has checks the C{pwd} database\n        and then the C{spwd} database.\n        '
        checker = checkers.UNIXPasswordDatabase()

        def crypted(username, password):
            if False:
                while True:
                    i = 10
            salt = crypt.crypt(password, username)
            crypted = crypt.crypt(password, '$1$' + salt)
            return crypted
        pwd = UserDatabase()
        pwd.addUser('alice', crypted('alice', 'password'), 1, 2, 'foo', '/foo', '/bin/sh')
        pwd.addUser('bob', 'x', 1, 2, 'bar', '/bar', '/bin/sh')
        spwd = ShadowDatabase()
        spwd.addUser('alice', 'wrong', 1, 2, 3, 4, 5, 6, 7)
        spwd.addUser('bob', crypted('bob', 'password'), 8, 9, 10, 11, 12, 13, 14)
        self.patch(checkers, 'pwd', pwd)
        self.patch(checkers, 'spwd', spwd)
        mockos = MockOS()
        self.patch(util, 'os', mockos)
        mockos.euid = 2345
        mockos.egid = 1234
        cred = UsernamePassword(b'alice', b'password')
        self.assertLoggedIn(checker.requestAvatarId(cred), b'alice')
        self.assertEqual(mockos.seteuidCalls, [])
        self.assertEqual(mockos.setegidCalls, [])
        cred.username = b'bob'
        self.assertLoggedIn(checker.requestAvatarId(cred), b'bob')
        self.assertEqual(mockos.seteuidCalls, [0, 2345])
        self.assertEqual(mockos.setegidCalls, [0, 1234])

    def assertUnauthorizedLogin(self, d):
        if False:
            while True:
                i = 10
        "\n        Asserts that the L{Deferred} passed in is erred back with an\n        L{UnauthorizedLogin} L{Failure}.  This reprsents an invalid login for\n        this TestCase.\n\n        NOTE: To work, this method's return value must be returned from the\n        test method, or otherwise hooked up to the test machinery.\n\n        @param d: a L{Deferred} from an L{IChecker.requestAvatarId} method.\n        @type d: L{Deferred}\n        @rtype: L{None}\n        "
        self.failureResultOf(d, checkers.UnauthorizedLogin)

    def test_passInCheckers(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{UNIXPasswordDatabase} takes a list of functions to check for UNIX\n        user information.\n        '
        password = crypt.crypt('secret', 'secret')
        userdb = UserDatabase()
        userdb.addUser('anybody', password, 1, 2, 'foo', '/bar', '/bin/sh')
        checker = checkers.UNIXPasswordDatabase([userdb.getpwnam])
        self.assertLoggedIn(checker.requestAvatarId(UsernamePassword(b'anybody', b'secret')), b'anybody')

    def test_verifyPassword(self):
        if False:
            return 10
        '\n        If the encrypted password provided by the getpwnam function is valid\n        (verified by the L{verifyCryptedPassword} function), we callback the\n        C{requestAvatarId} L{Deferred} with the username.\n        '

        def verifyCryptedPassword(crypted, pw):
            if False:
                i = 10
                return i + 15
            return crypted == pw

        def getpwnam(username):
            if False:
                i = 10
                return i + 15
            return [username, username]
        self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
        checker = checkers.UNIXPasswordDatabase([getpwnam])
        credential = UsernamePassword(b'username', b'username')
        self.assertLoggedIn(checker.requestAvatarId(credential), b'username')

    def test_failOnKeyError(self):
        if False:
            i = 10
            return i + 15
        '\n        If the getpwnam function raises a KeyError, the login fails with an\n        L{UnauthorizedLogin} exception.\n        '

        def getpwnam(username):
            if False:
                return 10
            raise KeyError(username)
        checker = checkers.UNIXPasswordDatabase([getpwnam])
        credential = UsernamePassword(b'username', b'password')
        self.assertUnauthorizedLogin(checker.requestAvatarId(credential))

    def test_failOnBadPassword(self):
        if False:
            print('Hello World!')
        "\n        If the verifyCryptedPassword function doesn't verify the password, the\n        login fails with an L{UnauthorizedLogin} exception.\n        "

        def verifyCryptedPassword(crypted, pw):
            if False:
                for i in range(10):
                    print('nop')
            return False

        def getpwnam(username):
            if False:
                return 10
            return [username, b'password']
        self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
        checker = checkers.UNIXPasswordDatabase([getpwnam])
        credential = UsernamePassword(b'username', b'password')
        self.assertUnauthorizedLogin(checker.requestAvatarId(credential))

    def test_loopThroughFunctions(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        UNIXPasswordDatabase.requestAvatarId loops through each getpwnam\n        function associated with it and returns a L{Deferred} which fires with\n        the result of the first one which returns a value other than None.\n        ones do not verify the password.\n        '

        def verifyCryptedPassword(crypted, pw):
            if False:
                for i in range(10):
                    print('nop')
            return crypted == pw

        def getpwnam1(username):
            if False:
                for i in range(10):
                    print('nop')
            return [username, 'not the password']

        def getpwnam2(username):
            if False:
                return 10
            return [username, 'password']
        self.patch(checkers, 'verifyCryptedPassword', verifyCryptedPassword)
        checker = checkers.UNIXPasswordDatabase([getpwnam1, getpwnam2])
        credential = UsernamePassword(b'username', b'password')
        self.assertLoggedIn(checker.requestAvatarId(credential), b'username')

    def test_failOnSpecial(self):
        if False:
            return 10
        '\n        If the password returned by any function is C{""}, C{"x"}, or C{"*"} it\n        is not compared against the supplied password.  Instead it is skipped.\n        '
        pwd = UserDatabase()
        pwd.addUser('alice', '', 1, 2, '', 'foo', 'bar')
        pwd.addUser('bob', 'x', 1, 2, '', 'foo', 'bar')
        pwd.addUser('carol', '*', 1, 2, '', 'foo', 'bar')
        self.patch(checkers, 'pwd', pwd)
        checker = checkers.UNIXPasswordDatabase([checkers._pwdGetByName])
        cred = UsernamePassword(b'alice', b'')
        self.assertUnauthorizedLogin(checker.requestAvatarId(cred))
        cred = UsernamePassword(b'bob', b'x')
        self.assertUnauthorizedLogin(checker.requestAvatarId(cred))
        cred = UsernamePassword(b'carol', b'*')
        self.assertUnauthorizedLogin(checker.requestAvatarId(cred))

class AuthorizedKeyFileReaderTests(TestCase):
    """
    Tests for L{checkers.readAuthorizedKeyFile}
    """
    skip = dependencySkip

    def test_ignoresComments(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{checkers.readAuthorizedKeyFile} does not attempt to turn comments\n        into keys\n        '
        fileobj = BytesIO(b'# this comment is ignored\nthis is not\n# this is again\nand this is not')
        result = checkers.readAuthorizedKeyFile(fileobj, lambda x: x)
        self.assertEqual([b'this is not', b'and this is not'], list(result))

    def test_ignoresLeadingWhitespaceAndEmptyLines(self):
        if False:
            while True:
                i = 10
        '\n        L{checkers.readAuthorizedKeyFile} ignores leading whitespace in\n        lines, as well as empty lines\n        '
        fileobj = BytesIO(b'\n                           # ignore\n                           not ignored\n                           ')
        result = checkers.readAuthorizedKeyFile(fileobj, parseKey=lambda x: x)
        self.assertEqual([b'not ignored'], list(result))

    def test_ignoresUnparsableKeys(self):
        if False:
            while True:
                i = 10
        '\n        L{checkers.readAuthorizedKeyFile} does not raise an exception\n        when a key fails to parse (raises a\n        L{twisted.conch.ssh.keys.BadKeyError}), but rather just keeps going\n        '

        def failOnSome(line):
            if False:
                i = 10
                return i + 15
            if line.startswith(b'f'):
                raise keys.BadKeyError('failed to parse')
            return line
        fileobj = BytesIO(b'failed key\ngood key')
        result = checkers.readAuthorizedKeyFile(fileobj, parseKey=failOnSome)
        self.assertEqual([b'good key'], list(result))

class InMemorySSHKeyDBTests(TestCase):
    """
    Tests for L{checkers.InMemorySSHKeyDB}
    """
    skip = dependencySkip

    def test_implementsInterface(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{checkers.InMemorySSHKeyDB} implements\n        L{checkers.IAuthorizedKeysDB}\n        '
        keydb = checkers.InMemorySSHKeyDB({b'alice': [b'key']})
        verifyObject(checkers.IAuthorizedKeysDB, keydb)

    def test_noKeysForUnauthorizedUser(self):
        if False:
            while True:
                i = 10
        '\n        If the user is not in the mapping provided to\n        L{checkers.InMemorySSHKeyDB}, an empty iterator is returned\n        by L{checkers.InMemorySSHKeyDB.getAuthorizedKeys}\n        '
        keydb = checkers.InMemorySSHKeyDB({b'alice': [b'keys']})
        self.assertEqual([], list(keydb.getAuthorizedKeys(b'bob')))

    def test_allKeysForAuthorizedUser(self):
        if False:
            return 10
        '\n        If the user is in the mapping provided to\n        L{checkers.InMemorySSHKeyDB}, an iterator with all the keys\n        is returned by L{checkers.InMemorySSHKeyDB.getAuthorizedKeys}\n        '
        keydb = checkers.InMemorySSHKeyDB({b'alice': [b'a', b'b']})
        self.assertEqual([b'a', b'b'], list(keydb.getAuthorizedKeys(b'alice')))

class UNIXAuthorizedKeysFilesTests(TestCase):
    """
    Tests for L{checkers.UNIXAuthorizedKeysFiles}.
    """
    skip = dependencySkip

    def setUp(self) -> None:
        if False:
            print('Hello World!')
        self.path = FilePath(self.mktemp())
        assert isinstance(self.path.path, str)
        self.path.makedirs()
        self.userdb = UserDatabase()
        self.userdb.addUser('alice', 'password', 1, 2, 'alice lastname', self.path.path, '/bin/shell')
        self.sshDir = self.path.child('.ssh')
        self.sshDir.makedirs()
        authorizedKeys = self.sshDir.child('authorized_keys')
        authorizedKeys.setContent(b'key 1\nkey 2')
        self.expectedKeys = [b'key 1', b'key 2']

    def test_implementsInterface(self):
        if False:
            i = 10
            return i + 15
        '\n        L{checkers.UNIXAuthorizedKeysFiles} implements\n        L{checkers.IAuthorizedKeysDB}.\n        '
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb)
        verifyObject(checkers.IAuthorizedKeysDB, keydb)

    def test_noKeysForUnauthorizedUser(self):
        if False:
            print('Hello World!')
        '\n        If the user is not in the user database provided to\n        L{checkers.UNIXAuthorizedKeysFiles}, an empty iterator is returned\n        by L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys}.\n        '
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual([], list(keydb.getAuthorizedKeys(b'bob')))

    def test_allKeysInAllAuthorizedFilesForAuthorizedUser(self):
        if False:
            i = 10
            return i + 15
        '\n        If the user is in the user database provided to\n        L{checkers.UNIXAuthorizedKeysFiles}, an iterator with all the keys in\n        C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2} is returned\n        by L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys}.\n        '
        self.sshDir.child('authorized_keys2').setContent(b'key 3')
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual(self.expectedKeys + [b'key 3'], list(keydb.getAuthorizedKeys(b'alice')))

    def test_ignoresNonexistantFile(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys} returns only\n        the keys in C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2}\n        if they exist.\n        '
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual(self.expectedKeys, list(keydb.getAuthorizedKeys(b'alice')))

    def test_ignoresUnreadableFile(self):
        if False:
            i = 10
            return i + 15
        '\n        L{checkers.UNIXAuthorizedKeysFiles.getAuthorizedKeys} returns only\n        the keys in C{~/.ssh/authorized_keys} and C{~/.ssh/authorized_keys2}\n        if they are readable.\n        '
        self.sshDir.child('authorized_keys2').makedirs()
        keydb = checkers.UNIXAuthorizedKeysFiles(self.userdb, parseKey=lambda x: x)
        self.assertEqual(self.expectedKeys, list(keydb.getAuthorizedKeys(b'alice')))
_KeyDB = namedtuple('_KeyDB', ['getAuthorizedKeys'])

class _DummyException(Exception):
    """
    Fake exception to be used for testing.
    """
    pass

class SSHPublicKeyCheckerTests(TestCase):
    """
    Tests for L{checkers.SSHPublicKeyChecker}.
    """
    skip = dependencySkip

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.credentials = SSHPrivateKey(b'alice', b'ssh-rsa', keydata.publicRSA_openssh, b'foo', keys.Key.fromString(keydata.privateRSA_openssh).sign(b'foo'))
        self.keydb = _KeyDB(lambda _: [keys.Key.fromString(keydata.publicRSA_openssh)])
        self.checker = checkers.SSHPublicKeyChecker(self.keydb)

    def test_credentialsWithoutSignature(self):
        if False:
            print('Hello World!')
        '\n        Calling L{checkers.SSHPublicKeyChecker.requestAvatarId} with\n        credentials that do not have a signature fails with L{ValidPublicKey}.\n        '
        self.credentials.signature = None
        self.failureResultOf(self.checker.requestAvatarId(self.credentials), ValidPublicKey)

    def test_credentialsWithBadKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling L{checkers.SSHPublicKeyChecker.requestAvatarId} with\n        credentials that have a bad key fails with L{keys.BadKeyError}.\n        '
        self.credentials.blob = b''
        self.failureResultOf(self.checker.requestAvatarId(self.credentials), keys.BadKeyError)

    def test_credentialsNoMatchingKey(self):
        if False:
            while True:
                i = 10
        '\n        If L{checkers.IAuthorizedKeysDB.getAuthorizedKeys} returns no keys\n        that match the credentials,\n        L{checkers.SSHPublicKeyChecker.requestAvatarId} fails with\n        L{UnauthorizedLogin}.\n        '
        self.credentials.blob = keydata.publicDSA_openssh
        self.failureResultOf(self.checker.requestAvatarId(self.credentials), UnauthorizedLogin)

    def test_credentialsInvalidSignature(self):
        if False:
            while True:
                i = 10
        '\n        Calling L{checkers.SSHPublicKeyChecker.requestAvatarId} with\n        credentials that are incorrectly signed fails with\n        L{UnauthorizedLogin}.\n        '
        self.credentials.signature = keys.Key.fromString(keydata.privateDSA_openssh).sign(b'foo')
        self.failureResultOf(self.checker.requestAvatarId(self.credentials), UnauthorizedLogin)

    def test_failureVerifyingKey(self):
        if False:
            return 10
        '\n        If L{keys.Key.verify} raises an exception,\n        L{checkers.SSHPublicKeyChecker.requestAvatarId} fails with\n        L{UnauthorizedLogin}.\n        '

        def fail(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise _DummyException()
        self.patch(keys.Key, 'verify', fail)
        self.failureResultOf(self.checker.requestAvatarId(self.credentials), UnauthorizedLogin)
        self.flushLoggedErrors(_DummyException)

    def test_usernameReturnedOnSuccess(self):
        if False:
            while True:
                i = 10
        '\n        L{checker.SSHPublicKeyChecker.requestAvatarId}, if successful,\n        callbacks with the username.\n        '
        d = self.checker.requestAvatarId(self.credentials)
        self.assertEqual(b'alice', self.successResultOf(d))