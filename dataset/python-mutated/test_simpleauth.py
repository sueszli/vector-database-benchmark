"""
Tests for basic constructs of L{twisted.cred.credentials}.
"""
from twisted.cred.credentials import IUsernameHashedPassword, IUsernamePassword, UsernamePassword
from twisted.cred.test.test_cred import _uhpVersion
from twisted.trial.unittest import TestCase

class UsernamePasswordTests(TestCase):
    """
    Tests for L{UsernamePassword}.
    """

    def test_initialisation(self) -> None:
        if False:
            return 10
        '\n        The initialisation of L{UsernamePassword} will set C{username} and\n        C{password} on it.\n        '
        creds = UsernamePassword(b'foo', b'bar')
        self.assertEqual(creds.username, b'foo')
        self.assertEqual(creds.password, b'bar')

    def test_correctPassword(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Calling C{checkPassword} on a L{UsernamePassword} will return L{True}\n        when the password given is the password on the object.\n        '
        creds = UsernamePassword(b'user', b'pass')
        self.assertTrue(creds.checkPassword(b'pass'))

    def test_wrongPassword(self) -> None:
        if False:
            return 10
        '\n        Calling C{checkPassword} on a L{UsernamePassword} will return L{False}\n        when the password given is NOT the password on the object.\n        '
        creds = UsernamePassword(b'user', b'pass')
        self.assertFalse(creds.checkPassword(b'someotherpass'))

    def test_interface(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{UsernamePassword} implements L{IUsernamePassword}.\n        '
        self.assertTrue(IUsernamePassword.implementedBy(UsernamePassword))

class UsernameHashedPasswordTests(TestCase):
    """
    Tests for L{UsernameHashedPassword}.
    """

    def test_initialisation(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The initialisation of L{UsernameHashedPassword} will set C{username}\n        and C{hashed} on it.\n        '
        UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
        creds = UsernameHashedPassword(b'foo', b'bar')
        self.assertEqual(creds.username, b'foo')
        self.assertEqual(creds.hashed, b'bar')

    def test_correctPassword(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Calling C{checkPassword} on a L{UsernameHashedPassword} will return\n        L{True} when the password given is the password on the object.\n        '
        UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
        creds = UsernameHashedPassword(b'user', b'pass')
        self.assertTrue(creds.checkPassword(b'pass'))

    def test_wrongPassword(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling C{checkPassword} on a L{UsernameHashedPassword} will return\n        L{False} when the password given is NOT the password on the object.\n        '
        UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
        creds = UsernameHashedPassword(b'user', b'pass')
        self.assertFalse(creds.checkPassword(b'someotherpass'))

    def test_interface(self) -> None:
        if False:
            return 10
        '\n        L{UsernameHashedPassword} implements L{IUsernameHashedPassword}.\n        '
        UsernameHashedPassword = self.getDeprecatedModuleAttribute('twisted.cred.credentials', 'UsernameHashedPassword', _uhpVersion)
        self.assertTrue(IUsernameHashedPassword.implementedBy(UsernameHashedPassword))