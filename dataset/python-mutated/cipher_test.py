"""Tests for the cipher module."""
from fire import testutils
from examples.cipher import cipher

class CipherTest(testutils.BaseTestCase):

    def testCipher(self):
        if False:
            while True:
                i = 10
        self.assertEqual(cipher.rot13('Hello world!'), 'Uryyb jbeyq!')
        self.assertEqual(cipher.caesar_encode(13, 'Hello world!'), 'Uryyb jbeyq!')
        self.assertEqual(cipher.caesar_decode(13, 'Uryyb jbeyq!'), 'Hello world!')
        self.assertEqual(cipher.caesar_encode(1, 'Hello world!'), 'Ifmmp xpsme!')
        self.assertEqual(cipher.caesar_decode(1, 'Ifmmp xpsme!'), 'Hello world!')
if __name__ == '__main__':
    testutils.main()