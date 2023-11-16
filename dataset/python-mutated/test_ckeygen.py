"""
Tests for L{twisted.conch.scripts.ckeygen}.
"""
from __future__ import annotations
import getpass
import os
import subprocess
import sys
from io import StringIO
from typing import Callable
from typing_extensions import NoReturn
from twisted.conch.test.keydata import privateECDSA_openssh, privateEd25519_openssh_new, privateRSA_openssh, privateRSA_openssh_encrypted, publicRSA_openssh
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.trial.unittest import TestCase
if requireModule('cryptography'):
    from twisted.conch.scripts.ckeygen import _getKeyOrDefault, _saveKey, changePassPhrase, displayPublicKey, enumrepresentation, printFingerprint
    from twisted.conch.ssh.keys import BadFingerPrintFormat, BadKeyError, FingerprintFormats, Key
else:
    skip = 'cryptography required for twisted.conch.scripts.ckeygen'

def makeGetpass(*passphrases: str) -> Callable[[object], str]:
    if False:
        i = 10
        return i + 15
    '\n    Return a callable to patch C{getpass.getpass}.  Yields a passphrase each\n    time called. Use case is to provide an old, then new passphrase(s) as if\n    requested interactively.\n\n    @param passphrases: The list of passphrases returned, one per each call.\n\n    @return: A callable to patch C{getpass.getpass}.\n    '
    passphrasesIter = iter(passphrases)

    def fakeGetpass(_: object) -> str:
        if False:
            print('Hello World!')
        return next(passphrasesIter)
    return fakeGetpass

class KeyGenTests(TestCase):
    """
    Tests for various functions used to implement the I{ckeygen} script.
    """

    def setUp(self) -> None:
        if False:
            return 10
        "\n        Patch C{sys.stdout} so tests can make assertions about what's printed.\n        "
        self.stdout = StringIO()
        self.patch(sys, 'stdout', self.stdout)

    def _testrun(self, keyType: str, keySize: str | None=None, privateKeySubtype: str | None=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        filename = self.mktemp()
        args = ['ckeygen', '-t', keyType, '-f', filename, '--no-passphrase']
        if keySize is not None:
            args.extend(['-b', keySize])
        if privateKeySubtype is not None:
            args.extend(['--private-key-subtype', privateKeySubtype])
        subprocess.call(args)
        privKey = Key.fromFile(filename)
        pubKey = Key.fromFile(filename + '.pub')
        if keyType == 'ecdsa':
            self.assertEqual(privKey.type(), 'EC')
        elif keyType == 'ed25519':
            self.assertEqual(privKey.type(), 'Ed25519')
        else:
            self.assertEqual(privKey.type(), keyType.upper())
        self.assertTrue(pubKey.isPublic())

    def test_keygeneration(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self._testrun('ecdsa', '384')
        self._testrun('ecdsa', '384', privateKeySubtype='v1')
        self._testrun('ecdsa')
        self._testrun('ecdsa', privateKeySubtype='v1')
        self._testrun('ed25519')
        self._testrun('dsa', '2048')
        self._testrun('dsa', '2048', privateKeySubtype='v1')
        self._testrun('dsa')
        self._testrun('dsa', privateKeySubtype='v1')
        self._testrun('rsa', '2048')
        self._testrun('rsa', '2048', privateKeySubtype='v1')
        self._testrun('rsa')
        self._testrun('rsa', privateKeySubtype='v1')

    def test_runBadKeytype(self) -> None:
        if False:
            return 10
        filename = self.mktemp()
        with self.assertRaises(subprocess.CalledProcessError):
            subprocess.check_call(['ckeygen', '-t', 'foo', '-f', filename])

    def test_enumrepresentation(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{enumrepresentation} takes a dictionary as input and returns a\n        dictionary with its attributes changed to enum representation.\n        '
        options = enumrepresentation({'format': 'md5-hex'})
        self.assertIs(options['format'], FingerprintFormats.MD5_HEX)

    def test_enumrepresentationsha256(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test for format L{FingerprintFormats.SHA256-BASE64}.\n        '
        options = enumrepresentation({'format': 'sha256-base64'})
        self.assertIs(options['format'], FingerprintFormats.SHA256_BASE64)

    def test_enumrepresentationBadFormat(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test for unsupported fingerprint format\n        '
        with self.assertRaises(BadFingerPrintFormat) as em:
            enumrepresentation({'format': 'sha-base64'})
        self.assertEqual('Unsupported fingerprint format: sha-base64', em.exception.args[0])

    def test_printFingerprint(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{printFingerprint} writes a line to standard out giving the number of\n        bits of the key, its fingerprint, and the basename of the file from it\n        was read.\n        '
        filename = self.mktemp()
        FilePath(filename).setContent(publicRSA_openssh)
        printFingerprint({'filename': filename, 'format': 'md5-hex'})
        self.assertEqual(self.stdout.getvalue(), '2048 85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da temp\n')

    def test_printFingerprintsha256(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{printFigerprint} will print key fingerprint in\n        L{FingerprintFormats.SHA256-BASE64} format if explicitly specified.\n        '
        filename = self.mktemp()
        FilePath(filename).setContent(publicRSA_openssh)
        printFingerprint({'filename': filename, 'format': 'sha256-base64'})
        self.assertEqual(self.stdout.getvalue(), '2048 FBTCOoknq0mHy+kpfnY9tDdcAJuWtCpuQMaV3EsvbUI= temp\n')

    def test_printFingerprintBadFingerPrintFormat(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{printFigerprint} raises C{keys.BadFingerprintFormat} when unsupported\n        formats are requested.\n        '
        filename = self.mktemp()
        FilePath(filename).setContent(publicRSA_openssh)
        with self.assertRaises(BadFingerPrintFormat) as em:
            printFingerprint({'filename': filename, 'format': 'sha-base64'})
        self.assertEqual('Unsupported fingerprint format: sha-base64', em.exception.args[0])

    def test_printFingerprintSuffixAppended(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        L{printFingerprint} checks if the filename with the  '.pub' suffix\n        exists in ~/.ssh.\n        "
        filename = self.mktemp()
        FilePath(filename + '.pub').setContent(publicRSA_openssh)
        printFingerprint({'filename': filename, 'format': 'md5-hex'})
        self.assertEqual(self.stdout.getvalue(), '2048 85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da temp.pub\n')

    def test_saveKey(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{_saveKey} writes the private and public parts of a key to two\n        different files and writes a report of this to standard out.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_rsa').path
        key = Key.fromString(privateRSA_openssh)
        _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex'})
        self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\n85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da\n' % (filename, filename))
        self.assertEqual(key.fromString(base.child('id_rsa').getContent(), None, 'passphrase'), key)
        self.assertEqual(Key.fromString(base.child('id_rsa.pub').getContent()), key.public())

    def test_saveKeyECDSA(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_saveKey} writes the private and public parts of a key to two\n        different files and writes a report of this to standard out.\n        Test with ECDSA key.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_ecdsa').path
        key = Key.fromString(privateECDSA_openssh)
        _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex'})
        self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\n1e:ab:83:a6:f2:04:22:99:7c:64:14:d2:ab:fa:f5:16\n' % (filename, filename))
        self.assertEqual(key.fromString(base.child('id_ecdsa').getContent(), None, 'passphrase'), key)
        self.assertEqual(Key.fromString(base.child('id_ecdsa.pub').getContent()), key.public())

    def test_saveKeyEd25519(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_saveKey} writes the private and public parts of a key to two\n        different files and writes a report of this to standard out.\n        Test with Ed25519 key.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_ed25519').path
        key = Key.fromString(privateEd25519_openssh_new)
        _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex'})
        self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\nab:ee:c8:ed:e5:01:1b:45:b7:8d:b2:f0:8f:61:1c:14\n' % (filename, filename))
        self.assertEqual(key.fromString(base.child('id_ed25519').getContent(), None, 'passphrase'), key)
        self.assertEqual(Key.fromString(base.child('id_ed25519.pub').getContent()), key.public())

    def test_saveKeysha256(self) -> None:
        if False:
            while True:
                i = 10
        '\n        L{_saveKey} will generate key fingerprint in\n        L{FingerprintFormats.SHA256-BASE64} format if explicitly specified.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_rsa').path
        key = Key.fromString(privateRSA_openssh)
        _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'sha256-base64'})
        self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=SHA256_BASE64> is:\nFBTCOoknq0mHy+kpfnY9tDdcAJuWtCpuQMaV3EsvbUI=\n' % (filename, filename))
        self.assertEqual(key.fromString(base.child('id_rsa').getContent(), None, 'passphrase'), key)
        self.assertEqual(Key.fromString(base.child('id_rsa.pub').getContent()), key.public())

    def test_saveKeyBadFingerPrintformat(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_saveKey} raises C{keys.BadFingerprintFormat} when unsupported\n        formats are requested.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_rsa').path
        key = Key.fromString(privateRSA_openssh)
        with self.assertRaises(BadFingerPrintFormat) as em:
            _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'sha-base64'})
        self.assertEqual('Unsupported fingerprint format: sha-base64', em.exception.args[0])

    def test_saveKeyEmptyPassphrase(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{_saveKey} will choose an empty string for the passphrase if\n        no-passphrase is C{True}.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_rsa').path
        key = Key.fromString(privateRSA_openssh)
        _saveKey(key, {'filename': filename, 'no-passphrase': True, 'format': 'md5-hex'})
        self.assertEqual(key.fromString(base.child('id_rsa').getContent(), None, b''), key)

    def test_saveKeyECDSAEmptyPassphrase(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{_saveKey} will choose an empty string for the passphrase if\n        no-passphrase is C{True}.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_ecdsa').path
        key = Key.fromString(privateECDSA_openssh)
        _saveKey(key, {'filename': filename, 'no-passphrase': True, 'format': 'md5-hex'})
        self.assertEqual(key.fromString(base.child('id_ecdsa').getContent(), None), key)

    def test_saveKeyEd25519EmptyPassphrase(self) -> None:
        if False:
            return 10
        '\n        L{_saveKey} will choose an empty string for the passphrase if\n        no-passphrase is C{True}.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_ed25519').path
        key = Key.fromString(privateEd25519_openssh_new)
        _saveKey(key, {'filename': filename, 'no-passphrase': True, 'format': 'md5-hex'})
        self.assertEqual(key.fromString(base.child('id_ed25519').getContent(), None), key)

    def test_saveKeyNoFilename(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        When no path is specified, it will ask for the path used to store the\n        key.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        keyPath = base.child('custom_key').path
        input_prompts: list[str] = []
        import twisted.conch.scripts.ckeygen

        def mock_input(*args: object) -> str:
            if False:
                i = 10
                return i + 15
            input_prompts.append('')
            return ''
        self.patch(twisted.conch.scripts.ckeygen, '_inputSaveFile', lambda _: keyPath)
        key = Key.fromString(privateRSA_openssh)
        _saveKey(key, {'filename': None, 'no-passphrase': True, 'format': 'md5-hex'}, mock_input)
        persistedKeyContent = base.child('custom_key').getContent()
        persistedKey = key.fromString(persistedKeyContent, None, b'')
        self.assertEqual(key, persistedKey)

    def test_saveKeyFileExists(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        When the specified file exists, it will ask the user for confirmation\n        before overwriting.\n        '

        def mock_input(*args: object) -> list[str]:
            if False:
                for i in range(10):
                    print('nop')
            return ['n']
        base = FilePath(self.mktemp())
        base.makedirs()
        keyPath = base.child('custom_key').path
        self.patch(os.path, 'exists', lambda _: True)
        key = Key.fromString(privateRSA_openssh)
        options = {'filename': keyPath, 'no-passphrase': True, 'format': 'md5-hex'}
        self.assertRaises(SystemExit, _saveKey, key, options, mock_input)

    def test_saveKeySubtypeV1(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{_saveKey} can be told to write the new private key file in OpenSSH\n        v1 format.\n        '
        base = FilePath(self.mktemp())
        base.makedirs()
        filename = base.child('id_rsa').path
        key = Key.fromString(privateRSA_openssh)
        _saveKey(key, {'filename': filename, 'pass': 'passphrase', 'format': 'md5-hex', 'private-key-subtype': 'v1'})
        self.assertEqual(self.stdout.getvalue(), 'Your identification has been saved in %s\nYour public key has been saved in %s.pub\nThe key fingerprint in <FingerprintFormats=MD5_HEX> is:\n85:25:04:32:58:55:96:9f:57:ee:fb:a8:1a:ea:69:da\n' % (filename, filename))
        privateKeyContent = base.child('id_rsa').getContent()
        self.assertEqual(key.fromString(privateKeyContent, None, 'passphrase'), key)
        self.assertTrue(privateKeyContent.startswith(b'-----BEGIN OPENSSH PRIVATE KEY-----\n'))
        self.assertEqual(Key.fromString(base.child('id_rsa.pub').getContent()), key.public())

    def test_displayPublicKey(self) -> None:
        if False:
            return 10
        '\n        L{displayPublicKey} prints out the public key associated with a given\n        private key.\n        '
        filename = self.mktemp()
        pubKey = Key.fromString(publicRSA_openssh)
        FilePath(filename).setContent(privateRSA_openssh)
        displayPublicKey({'filename': filename})
        displayed = self.stdout.getvalue().strip('\n').encode('ascii')
        self.assertEqual(displayed, pubKey.toString('openssh'))

    def test_displayPublicKeyEncrypted(self) -> None:
        if False:
            i = 10
            return i + 15
        "\n        L{displayPublicKey} prints out the public key associated with a given\n        private key using the given passphrase when it's encrypted.\n        "
        filename = self.mktemp()
        pubKey = Key.fromString(publicRSA_openssh)
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        displayPublicKey({'filename': filename, 'pass': 'encrypted'})
        displayed = self.stdout.getvalue().strip('\n').encode('ascii')
        self.assertEqual(displayed, pubKey.toString('openssh'))

    def test_displayPublicKeyEncryptedPassphrasePrompt(self) -> None:
        if False:
            print('Hello World!')
        "\n        L{displayPublicKey} prints out the public key associated with a given\n        private key, asking for the passphrase when it's encrypted.\n        "
        filename = self.mktemp()
        pubKey = Key.fromString(publicRSA_openssh)
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        self.patch(getpass, 'getpass', lambda x: 'encrypted')
        displayPublicKey({'filename': filename})
        displayed = self.stdout.getvalue().strip('\n').encode('ascii')
        self.assertEqual(displayed, pubKey.toString('openssh'))

    def test_displayPublicKeyWrongPassphrase(self) -> None:
        if False:
            print('Hello World!')
        '\n        L{displayPublicKey} fails with a L{BadKeyError} when trying to decrypt\n        an encrypted key with the wrong password.\n        '
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        self.assertRaises(BadKeyError, displayPublicKey, {'filename': filename, 'pass': 'wrong'})

    def test_changePassphrase(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{changePassPhrase} allows a user to change the passphrase of a\n        private key interactively.\n        '
        oldNewConfirm = makeGetpass('encrypted', 'newpass', 'newpass')
        self.patch(getpass, 'getpass', oldNewConfirm)
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        changePassPhrase({'filename': filename})
        self.assertEqual(self.stdout.getvalue().strip('\n'), 'Your identification has been saved with the new passphrase.')
        self.assertNotEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())

    def test_changePassphraseWithOld(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{changePassPhrase} allows a user to change the passphrase of a\n        private key, providing the old passphrase and prompting for new one.\n        '
        newConfirm = makeGetpass('newpass', 'newpass')
        self.patch(getpass, 'getpass', newConfirm)
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        changePassPhrase({'filename': filename, 'pass': 'encrypted'})
        self.assertEqual(self.stdout.getvalue().strip('\n'), 'Your identification has been saved with the new passphrase.')
        self.assertNotEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())

    def test_changePassphraseWithBoth(self) -> None:
        if False:
            return 10
        '\n        L{changePassPhrase} allows a user to change the passphrase of a private\n        key by providing both old and new passphrases without prompting.\n        '
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        changePassPhrase({'filename': filename, 'pass': 'encrypted', 'newpass': 'newencrypt'})
        self.assertEqual(self.stdout.getvalue().strip('\n'), 'Your identification has been saved with the new passphrase.')
        self.assertNotEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())

    def test_changePassphraseWrongPassphrase(self) -> None:
        if False:
            return 10
        '\n        L{changePassPhrase} exits if passed an invalid old passphrase when\n        trying to change the passphrase of a private key.\n        '
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'pass': 'wrong'})
        self.assertEqual('Could not change passphrase: old passphrase error', str(error))
        self.assertEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())

    def test_changePassphraseEmptyGetPass(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{changePassPhrase} exits if no passphrase is specified for the\n        C{getpass} call and the key is encrypted.\n        '
        self.patch(getpass, 'getpass', makeGetpass(''))
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename})
        self.assertEqual('Could not change passphrase: Passphrase must be provided for an encrypted key', str(error))
        self.assertEqual(privateRSA_openssh_encrypted, FilePath(filename).getContent())

    def test_changePassphraseBadKey(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{changePassPhrase} exits if the file specified points to an invalid\n        key.\n        '
        filename = self.mktemp()
        FilePath(filename).setContent(b'foobar')
        error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename})
        expected = "Could not change passphrase: cannot guess the type of b'foobar'"
        self.assertEqual(expected, str(error))
        self.assertEqual(b'foobar', FilePath(filename).getContent())

    def test_changePassphraseCreateError(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        "\n        L{changePassPhrase} doesn't modify the key file if an unexpected error\n        happens when trying to create the key with the new passphrase.\n        "
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh)

        def toString(*args: object, **kwargs: object) -> NoReturn:
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('oops')
        self.patch(Key, 'toString', toString)
        error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'newpass': 'newencrypt'})
        self.assertEqual('Could not change passphrase: oops', str(error))
        self.assertEqual(privateRSA_openssh, FilePath(filename).getContent())

    def test_changePassphraseEmptyStringError(self) -> None:
        if False:
            while True:
                i = 10
        "\n        L{changePassPhrase} doesn't modify the key file if C{toString} returns\n        an empty string.\n        "
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh)

        def toString(*args: object, **kwargs: object) -> str:
            if False:
                return 10
            return ''
        self.patch(Key, 'toString', toString)
        error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'newpass': 'newencrypt'})
        expected = "Could not change passphrase: cannot guess the type of b''"
        self.assertEqual(expected, str(error))
        self.assertEqual(privateRSA_openssh, FilePath(filename).getContent())

    def test_changePassphrasePublicKey(self) -> None:
        if False:
            print('Hello World!')
        "\n        L{changePassPhrase} exits when trying to change the passphrase on a\n        public key, and doesn't change the file.\n        "
        filename = self.mktemp()
        FilePath(filename).setContent(publicRSA_openssh)
        error = self.assertRaises(SystemExit, changePassPhrase, {'filename': filename, 'newpass': 'pass'})
        self.assertEqual('Could not change passphrase: key not encrypted', str(error))
        self.assertEqual(publicRSA_openssh, FilePath(filename).getContent())

    def test_changePassphraseSubtypeV1(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{changePassPhrase} can be told to write the new private key file in\n        OpenSSH v1 format.\n        '
        oldNewConfirm = makeGetpass('encrypted', 'newpass', 'newpass')
        self.patch(getpass, 'getpass', oldNewConfirm)
        filename = self.mktemp()
        FilePath(filename).setContent(privateRSA_openssh_encrypted)
        changePassPhrase({'filename': filename, 'private-key-subtype': 'v1'})
        self.assertEqual(self.stdout.getvalue().strip('\n'), 'Your identification has been saved with the new passphrase.')
        privateKeyContent = FilePath(filename).getContent()
        self.assertNotEqual(privateRSA_openssh_encrypted, privateKeyContent)
        self.assertTrue(privateKeyContent.startswith(b'-----BEGIN OPENSSH PRIVATE KEY-----\n'))

    def test_useDefaultForKey(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{options} will default to "~/.ssh/id_rsa" if the user doesn\'t\n        specify a key.\n        '
        input_prompts: list[str] = []

        def mock_input(*args: object) -> str:
            if False:
                while True:
                    i = 10
            input_prompts.append('')
            return ''
        options = {'filename': ''}
        filename = _getKeyOrDefault(options, mock_input)
        self.assertEqual(options['filename'], '')
        self.assertTrue(filename.endswith(os.path.join('.ssh', 'id_rsa')))
        self.assertEqual(1, len(input_prompts))
        self.assertEqual([''], input_prompts)

    def test_displayPublicKeyHandleFileNotFound(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Ensure FileNotFoundError is handled, whether the user has supplied\n        a bad path, or has no key at the default path.\n        '
        options = {'filename': '/foo/bar'}
        exc = self.assertRaises(SystemExit, displayPublicKey, options)
        self.assertIn('could not be opened, please specify a file.', exc.args[0])

    def test_changePassPhraseHandleFileNotFound(self) -> None:
        if False:
            return 10
        '\n        Ensure FileNotFoundError is handled for an invalid filename.\n        '
        options = {'filename': '/foo/bar'}
        exc = self.assertRaises(SystemExit, changePassPhrase, options)
        self.assertIn('could not be opened, please specify a file.', exc.args[0])

    def test_printFingerprintHandleFileNotFound(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Ensure FileNotFoundError is handled for an invalid filename.\n        '
        options = {'filename': '/foo/bar', 'format': 'md5-hex'}
        exc = self.assertRaises(SystemExit, printFingerprint, options)
        self.assertIn('could not be opened, please specify a file.', exc.args[0])