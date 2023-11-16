"""
Tests for L{twisted.conch.openssh_compat}.
"""
import os
from unittest import skipIf
from twisted.conch.ssh._kex import getDHGeneratorAndPrime
from twisted.conch.test import keydata
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.test_process import MockOS
from twisted.trial.unittest import TestCase
doSkip = False
skipReason = ''
if requireModule('cryptography'):
    from twisted.conch.openssh_compat.factory import OpenSSHFactory
else:
    doSkip = True
    skipReason = 'Cannot run without cryptography'
if not hasattr(os, 'geteuid'):
    doSkip = True
    skipReason = 'geteuid/seteuid not available'

@skipIf(doSkip, skipReason)
class OpenSSHFactoryTests(TestCase):
    """
    Tests for L{OpenSSHFactory}.
    """

    def setUp(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.factory = OpenSSHFactory()
        self.keysDir = FilePath(self.mktemp())
        self.keysDir.makedirs()
        self.factory.dataRoot = self.keysDir.path
        self.moduliDir = FilePath(self.mktemp())
        self.moduliDir.makedirs()
        self.factory.moduliRoot = self.moduliDir.path
        self.keysDir.child('ssh_host_foo').setContent(b'foo')
        self.keysDir.child('bar_key').setContent(b'foo')
        self.keysDir.child('ssh_host_one_key').setContent(keydata.privateRSA_openssh)
        self.keysDir.child('ssh_host_two_key').setContent(keydata.privateDSA_openssh)
        self.keysDir.child('ssh_host_three_key').setContent(b'not a key content')
        self.keysDir.child('ssh_host_one_key.pub').setContent(keydata.publicRSA_openssh)
        self.moduliDir.child('moduli').setContent(b'\n#    $OpenBSD: moduli,v 1.xx 2016/07/26 12:34:56 jhacker Exp $i\n# Time Type Tests Tries Size Generator Modulus\n20030501000000 2 6 100 2047 2 FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E088A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE649286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF6955817183995497CEA956AE515D2261898FA051015728E5A8AACAA68FFFFFFFFFFFFFFFF\n')
        self.mockos = MockOS()
        self.patch(os, 'seteuid', self.mockos.seteuid)
        self.patch(os, 'setegid', self.mockos.setegid)

    def test_getPublicKeys(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        L{OpenSSHFactory.getPublicKeys} should return the available public keys\n        in the data directory\n        '
        keys = self.factory.getPublicKeys()
        self.assertEqual(len(keys), 1)
        keyTypes = keys.keys()
        self.assertEqual(list(keyTypes), [b'ssh-rsa'])

    def test_getPrivateKeys(self) -> None:
        if False:
            while True:
                i = 10
        '\n        Will return the available private keys in the data directory, ignoring\n        key files which failed to be loaded.\n        '
        keys = self.factory.getPrivateKeys()
        self.assertEqual(len(keys), 2)
        keyTypes = keys.keys()
        self.assertEqual(set(keyTypes), {b'ssh-rsa', b'ssh-dss'})
        self.assertEqual(self.mockos.seteuidCalls, [])
        self.assertEqual(self.mockos.setegidCalls, [])

    def test_getPrivateKeysAsRoot(self) -> None:
        if False:
            while True:
                i = 10
        "\n        L{OpenSSHFactory.getPrivateKeys} should switch to root if the keys\n        aren't readable by the current user.\n        "
        keyFile = self.keysDir.child('ssh_host_two_key')
        keyFile.chmod(0)
        self.addCleanup(keyFile.chmod, 511)
        savedSeteuid = os.seteuid

        def seteuid(euid: int) -> None:
            if False:
                return 10
            keyFile.chmod(511)
            return savedSeteuid(euid)
        self.patch(os, 'seteuid', seteuid)
        keys = self.factory.getPrivateKeys()
        self.assertEqual(len(keys), 2)
        keyTypes = keys.keys()
        self.assertEqual(set(keyTypes), {b'ssh-rsa', b'ssh-dss'})
        self.assertEqual(self.mockos.seteuidCalls, [0, os.geteuid()])
        self.assertEqual(self.mockos.setegidCalls, [0, os.getegid()])

    def test_getPrimes(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        L{OpenSSHFactory.getPrimes} should return the available primes\n        in the moduli directory.\n        '
        primes = self.factory.getPrimes()
        self.assertEqual(primes, {2048: [getDHGeneratorAndPrime(b'diffie-hellman-group14-sha1')]})