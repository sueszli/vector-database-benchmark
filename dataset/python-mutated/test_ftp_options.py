"""
Tests for L{twisted.tap.ftp}.
"""
from twisted.cred import credentials, error
from twisted.python import versions
from twisted.python.filepath import FilePath
from twisted.tap.ftp import Options
from twisted.trial.unittest import TestCase

class FTPOptionsTests(TestCase):
    """
    Tests for the command line option parser used for C{twistd ftp}.
    """
    usernamePassword = (b'iamuser', b'thisispassword')

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Create a file with two users.\n        '
        self.filename = self.mktemp()
        f = FilePath(self.filename)
        f.setContent(b':'.join(self.usernamePassword))
        self.options = Options()

    def test_passwordfileDeprecation(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        The C{--password-file} option will emit a warning stating that\n        said option is deprecated.\n        '
        self.callDeprecated(versions.Version('Twisted', 11, 1, 0), self.options.opt_password_file, self.filename)

    def test_authAdded(self) -> None:
        if False:
            while True:
                i = 10
        '\n        The C{--auth} command-line option will add a checker to the list of\n        checkers\n        '
        numCheckers = len(self.options['credCheckers'])
        self.options.parseOptions(['--auth', 'file:' + self.filename])
        self.assertEqual(len(self.options['credCheckers']), numCheckers + 1)

    def test_authFailure(self):
        if False:
            i = 10
            return i + 15
        '\n        The checker created by the C{--auth} command-line option returns a\n        L{Deferred} that fails with L{UnauthorizedLogin} when\n        presented with credentials that are unknown to that checker.\n        '
        self.options.parseOptions(['--auth', 'file:' + self.filename])
        checker = self.options['credCheckers'][-1]
        invalid = credentials.UsernamePassword(self.usernamePassword[0], 'fake')
        return checker.requestAvatarId(invalid).addCallbacks(lambda ignore: self.fail('Wrong password should raise error'), lambda err: err.trap(error.UnauthorizedLogin))

    def test_authSuccess(self):
        if False:
            return 10
        '\n        The checker created by the C{--auth} command-line option returns a\n        L{Deferred} that returns the avatar id when presented with credentials\n        that are known to that checker.\n        '
        self.options.parseOptions(['--auth', 'file:' + self.filename])
        checker = self.options['credCheckers'][-1]
        correct = credentials.UsernamePassword(*self.usernamePassword)
        return checker.requestAvatarId(correct).addCallback(lambda username: self.assertEqual(username, correct.username))