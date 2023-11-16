"""
Tests for L{twisted.cred}'s implementation of CRAM-MD5.
"""
import hashlib
from binascii import hexlify
from hmac import HMAC
from twisted.cred.credentials import CramMD5Credentials, IUsernameHashedPassword
from twisted.trial.unittest import TestCase

class CramMD5CredentialsTests(TestCase):
    """
    Tests for L{CramMD5Credentials}.
    """

    def test_idempotentChallenge(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        The same L{CramMD5Credentials} will always provide the same challenge,\n        no matter how many times it is called.\n        '
        c = CramMD5Credentials()
        chal = c.getChallenge()
        self.assertEqual(chal, c.getChallenge())

    def test_checkPassword(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        When a valid response (which is a hex digest of the challenge that has\n        been encrypted by the user's shared secret) is set on the\n        L{CramMD5Credentials} that created the challenge, and C{checkPassword}\n        is called with the user's shared secret, it will return L{True}.\n        "
        c = CramMD5Credentials()
        chal = c.getChallenge()
        c.response = hexlify(HMAC(b'secret', chal, digestmod=hashlib.md5).digest())
        self.assertTrue(c.checkPassword(b'secret'))

    def test_noResponse(self) -> None:
        if False:
            while True:
                i = 10
        '\n        When there is no response set, calling C{checkPassword} will return\n        L{False}.\n        '
        c = CramMD5Credentials()
        self.assertFalse(c.checkPassword(b'secret'))

    def test_wrongPassword(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        When an invalid response is set on the L{CramMD5Credentials} (one that\n        is not the hex digest of the challenge, encrypted with the user's shared\n        secret) and C{checkPassword} is called with the user's correct shared\n        secret, it will return L{False}.\n        "
        c = CramMD5Credentials()
        chal = c.getChallenge()
        c.response = hexlify(HMAC(b'thewrongsecret', chal, digestmod=hashlib.md5).digest())
        self.assertFalse(c.checkPassword(b'secret'))

    def test_setResponse(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        When C{setResponse} is called with a string that is the username and\n        the hashed challenge separated with a space, they will be set on the\n        L{CramMD5Credentials}.\n        '
        c = CramMD5Credentials()
        chal = c.getChallenge()
        c.setResponse(b' '.join((b'squirrel', hexlify(HMAC(b'supersecret', chal, digestmod=hashlib.md5).digest()))))
        self.assertTrue(c.checkPassword(b'supersecret'))
        self.assertEqual(c.username, b'squirrel')

    def test_interface(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{CramMD5Credentials} implements the L{IUsernameHashedPassword}\n        interface.\n        '
        self.assertTrue(IUsernameHashedPassword.implementedBy(CramMD5Credentials))