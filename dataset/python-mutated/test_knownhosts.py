"""
Tests for L{twisted.conch.client.knownhosts}.
"""
import os
from binascii import Error as BinasciiError, a2b_base64, b2a_base64
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.internet.defer import Deferred
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.reflect import requireModule
from twisted.test.testutils import ComparisonTestsMixin
from twisted.trial.unittest import TestCase
if requireModule('cryptography'):
    from twisted.conch.client import default
    from twisted.conch.client.knownhosts import ConsoleUI, HashedEntry, KnownHostsFile, PlainEntry, UnparsedEntry
    from twisted.conch.ssh.keys import BadKeyError, Key
    from twisted.conch.test import keydata
else:
    skip = 'cryptography required for twisted.conch.knownhosts.'
sampleEncodedKey = b'AAAAB3NzaC1yc2EAAAABIwAAAQEAsV0VMRbGmzhqxxayLRHmvnFvtyNqgbNKV46dU1bVFB+3ytNvue4Riqv/SVkPRNwMb7eWH29SviXaBxUhYyzKkDoNUq3rTNnH1Vnif6d6X4JCrUb5d3W+DmYClyJrZ5HgD/hUpdSkTRqdbQ2TrvSAxRacj+vHHT4F4dm1bJSewm3B2D8HVOoi/CbVh3dsIiCdp8VltdZx4qYVfYe2LwVINCbAa3d3tj9ma7RVfw3OH2Mfb+toLd1N5tBQFb7oqTt2nC6I/6Bd4JwPUld+IEitw/suElq/AIJVQXXujeyiZlea90HE65U2mF1ytr17HTAIT2ySokJWyuBANGACk6iIaw=='
otherSampleEncodedKey = b'AAAAB3NzaC1yc2EAAAABIwAAAIEAwaeCZd3UCuPXhX39+/p9qO028jTF76DMVd9mPvYVDVXufWckKZauF7+0b7qm+ChT7kan6BzRVo4++gCVNfAlMzLysSt3ylmOR48tFpAfygg9UCX3DjHz0ElOOUKh3iifc9aUShD0OPaK3pR5JJ8jfiBfzSYWt/hDi/iZ4igsSs8='
thirdSampleEncodedKey = b'AAAAB3NzaC1yc2EAAAABIwAAAQEAl/TQakPkePlnwCBRPitIVUTg6Z8VzN1en+DGkyo/evkmLw7o4NWR5qbysk9A9jXW332nxnEuAnbcCam9SHe1su1liVfyIK0+3bdn0YRB0sXIbNEtMs2LtCho/aV3cXPS+Cf1yut3wvIpaRnAzXxuKPCTXQ7/y0IXa8TwkRBH58OJa3RqfQ/NsSp5SAfdsrHyH2aitiVKm2jfbTKzSEqOQG/zq4J9GXTkq61gZugory/Tvl5/yPgSnOR6C9jVOMHf27ZPoRtyj9SY343Hd2QHiIE0KPZJEgCynKeWoKz8v6eTSK8n4rBnaqWdp8MnGZK1WGy05MguXbyCDuTC8AmJXQ=='
ecdsaSampleEncodedKey = b'AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBIFwh3/zBANyPPIE60SMMfdKMYo3OvfvzGLZphzuKrzSt0q4uF+/iYqtYiHhryAwU/fDWlUQ9kck9f+IlpsNtY4='
sampleKey = a2b_base64(sampleEncodedKey)
otherSampleKey = a2b_base64(otherSampleEncodedKey)
thirdSampleKey = a2b_base64(thirdSampleEncodedKey)
ecdsaSampleKey = a2b_base64(ecdsaSampleEncodedKey)
samplePlaintextLine = b'www.twistedmatrix.com ssh-rsa ' + sampleEncodedKey + b'\n'
otherSamplePlaintextLine = b'divmod.com ssh-rsa ' + otherSampleEncodedKey + b'\n'
sampleHostIPLine = b'www.twistedmatrix.com,198.49.126.131 ssh-rsa ' + sampleEncodedKey + b'\n'
sampleHashedLine = b'|1|gJbSEPBG9ZSBoZpHNtZBD1bHKBA=|bQv+0Xa0dByrwkA1EB0E7Xop/Fo= ssh-rsa ' + sampleEncodedKey + b'\n'

class EntryTestsMixin:
    """
    Tests for implementations of L{IKnownHostEntry}.  Subclasses must set the
    'entry' attribute to a provider of that interface, the implementation of
    that interface under test.

    @ivar entry: a provider of L{IKnownHostEntry} with a hostname of
    www.twistedmatrix.com and an RSA key of sampleKey.
    """

    def test_providesInterface(self):
        if False:
            print('Hello World!')
        '\n        The given entry should provide IKnownHostEntry.\n        '
        verifyObject(IKnownHostEntry, self.entry)

    def test_fromString(self):
        if False:
            print('Hello World!')
        "\n        Constructing a plain text entry from an unhashed known_hosts entry will\n        result in an L{IKnownHostEntry} provider with 'keyString', 'hostname',\n        and 'keyType' attributes.  While outside the interface in question,\n        these attributes are held in common by L{PlainEntry} and L{HashedEntry}\n        implementations; other implementations should override this method in\n        subclasses.\n        "
        entry = self.entry
        self.assertEqual(entry.publicKey, Key.fromString(sampleKey))
        self.assertEqual(entry.keyType, b'ssh-rsa')

    def test_matchesKey(self):
        if False:
            while True:
                i = 10
        '\n        L{IKnownHostEntry.matchesKey} checks to see if an entry matches a given\n        SSH key.\n        '
        twistedmatrixDotCom = Key.fromString(sampleKey)
        divmodDotCom = Key.fromString(otherSampleKey)
        self.assertEqual(True, self.entry.matchesKey(twistedmatrixDotCom))
        self.assertEqual(False, self.entry.matchesKey(divmodDotCom))

    def test_matchesHost(self):
        if False:
            while True:
                i = 10
        '\n        L{IKnownHostEntry.matchesHost} checks to see if an entry matches a\n        given hostname.\n        '
        self.assertTrue(self.entry.matchesHost(b'www.twistedmatrix.com'))
        self.assertFalse(self.entry.matchesHost(b'www.divmod.com'))

class PlainEntryTests(EntryTestsMixin, TestCase):
    """
    Test cases for L{PlainEntry}.
    """
    plaintextLine = samplePlaintextLine
    hostIPLine = sampleHostIPLine

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set 'entry' to a sample plain-text entry with sampleKey as its key.\n        "
        self.entry = PlainEntry.fromString(self.plaintextLine)

    def test_matchesHostIP(self):
        if False:
            return 10
        '\n        A "hostname,ip" formatted line will match both the host and the IP.\n        '
        self.entry = PlainEntry.fromString(self.hostIPLine)
        self.assertTrue(self.entry.matchesHost(b'198.49.126.131'))
        self.test_matchesHost()

    def test_toString(self):
        if False:
            print('Hello World!')
        '\n        L{PlainEntry.toString} generates the serialized OpenSSL format string\n        for the entry, sans newline.\n        '
        self.assertEqual(self.entry.toString(), self.plaintextLine.rstrip(b'\n'))
        multiHostEntry = PlainEntry.fromString(self.hostIPLine)
        self.assertEqual(multiHostEntry.toString(), self.hostIPLine.rstrip(b'\n'))

class PlainTextWithCommentTests(PlainEntryTests):
    """
    Test cases for L{PlainEntry} when parsed from a line with a comment.
    """
    plaintextLine = samplePlaintextLine[:-1] + b' plain text comment.\n'
    hostIPLine = sampleHostIPLine[:-1] + b' text following host/IP line\n'

class HashedEntryTests(EntryTestsMixin, ComparisonTestsMixin, TestCase):
    """
    Tests for L{HashedEntry}.

    This suite doesn't include any tests for host/IP pairs because hashed
    entries store IP addresses the same way as hostnames and does not support
    comma-separated lists.  (If you hash the IP and host together you can't
    tell if you've got the key already for one or the other.)
    """
    hashedLine = sampleHashedLine

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set 'entry' to a sample hashed entry for twistedmatrix.com with\n        sampleKey as its key.\n        "
        self.entry = HashedEntry.fromString(self.hashedLine)

    def test_toString(self):
        if False:
            i = 10
            return i + 15
        '\n        L{HashedEntry.toString} generates the serialized OpenSSL format string\n        for the entry, sans the newline.\n        '
        self.assertEqual(self.entry.toString(), self.hashedLine.rstrip(b'\n'))

    def test_equality(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Two L{HashedEntry} instances compare equal if and only if they represent\n        the same host and key in exactly the same way: the host salt, host hash,\n        public key type, public key, and comment fields must all be equal.\n        '
        hostSalt = b'gJbSEPBG9ZSBoZpHNtZBD1bHKBA'
        hostHash = b'bQv+0Xa0dByrwkA1EB0E7Xop/Fo'
        publicKey = Key.fromString(sampleKey)
        keyType = networkString(publicKey.type())
        comment = b'hello, world'
        entry = HashedEntry(hostSalt, hostHash, keyType, publicKey, comment)
        duplicate = HashedEntry(hostSalt, hostHash, keyType, publicKey, comment)
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt[::-1], hostHash, keyType, publicKey, comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash[::-1], keyType, publicKey, comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash, keyType[::-1], publicKey, comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash, keyType, Key.fromString(otherSampleKey), comment))
        self.assertNormalEqualityImplementation(entry, duplicate, HashedEntry(hostSalt, hostHash, keyType, publicKey, comment[::-1]))

class HashedEntryWithCommentTests(HashedEntryTests):
    """
    Test cases for L{PlainEntry} when parsed from a line with a comment.
    """
    hashedLine = sampleHashedLine[:-1] + b' plain text comment.\n'

class UnparsedEntryTests(TestCase, EntryTestsMixin):
    """
    Tests for L{UnparsedEntry}
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        "\n        Set up the 'entry' to be an unparsed entry for some random text.\n        "
        self.entry = UnparsedEntry(b'    This is a bogus entry.  \n')

    def test_fromString(self):
        if False:
            while True:
                i = 10
        '\n        Creating an L{UnparsedEntry} should simply record the string it was\n        passed.\n        '
        self.assertEqual(b'    This is a bogus entry.  \n', self.entry._string)

    def test_matchesHost(self):
        if False:
            while True:
                i = 10
        "\n        An unparsed entry can't match any hosts.\n        "
        self.assertFalse(self.entry.matchesHost(b'www.twistedmatrix.com'))

    def test_matchesKey(self):
        if False:
            return 10
        "\n        An unparsed entry can't match any keys.\n        "
        self.assertFalse(self.entry.matchesKey(Key.fromString(sampleKey)))

    def test_toString(self):
        if False:
            return 10
        '\n        L{UnparsedEntry.toString} returns its input string, sans trailing\n        newline.\n        '
        self.assertEqual(b'    This is a bogus entry.  ', self.entry.toString())

class ParseErrorTests(TestCase):
    """
    L{HashedEntry.fromString} and L{PlainEntry.fromString} can raise a variety
    of errors depending on misformattings of certain strings.  These tests make
    sure those errors are caught.  Since many of the ways that this can go
    wrong are in the lower-level APIs being invoked by the parsing logic,
    several of these are integration tests with the C{base64} and
    L{twisted.conch.ssh.keys} modules.
    """

    def invalidEntryTest(self, cls):
        if False:
            return 10
        '\n        If there are fewer than three elements, C{fromString} should raise\n        L{InvalidEntry}.\n        '
        self.assertRaises(InvalidEntry, cls.fromString, b'invalid')

    def notBase64Test(self, cls):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the key is not base64, C{fromString} should raise L{BinasciiError}.\n        '
        self.assertRaises(BinasciiError, cls.fromString, b'x x x')

    def badKeyTest(self, cls, prefix):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the key portion of the entry is valid base64, but is not actually an\n        SSH key, C{fromString} should raise L{BadKeyError}.\n        '
        self.assertRaises(BadKeyError, cls.fromString, b' '.join([prefix, b'ssh-rsa', b2a_base64(b"Hey, this isn't an SSH key!").strip()]))

    def test_invalidPlainEntry(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If there are fewer than three whitespace-separated elements in an\n        entry, L{PlainEntry.fromString} should raise L{InvalidEntry}.\n        '
        self.invalidEntryTest(PlainEntry)

    def test_invalidHashedEntry(self):
        if False:
            while True:
                i = 10
        '\n        If there are fewer than three whitespace-separated elements in an\n        entry, or the hostname salt/hash portion has more than two elements,\n        L{HashedEntry.fromString} should raise L{InvalidEntry}.\n        '
        self.invalidEntryTest(HashedEntry)
        (a, b, c) = sampleHashedLine.split()
        self.assertRaises(InvalidEntry, HashedEntry.fromString, b' '.join([a + b'||', b, c]))

    def test_plainNotBase64(self):
        if False:
            print('Hello World!')
        '\n        If the key portion of a plain entry is not decodable as base64,\n        C{fromString} should raise L{BinasciiError}.\n        '
        self.notBase64Test(PlainEntry)

    def test_hashedNotBase64(self):
        if False:
            print('Hello World!')
        '\n        If the key, host salt, or host hash portion of a hashed entry is not\n        encoded, it will raise L{BinasciiError}.\n        '
        self.notBase64Test(HashedEntry)
        (a, b, c) = sampleHashedLine.split()
        self.assertRaises(BinasciiError, HashedEntry.fromString, b' '.join([b'|1|x|' + b2a_base64(b'stuff').strip(), b, c]))
        self.assertRaises(BinasciiError, HashedEntry.fromString, b' '.join([HashedEntry.MAGIC + b2a_base64(b'stuff').strip() + b'|x', b, c]))
        self.assertRaises(BinasciiError, HashedEntry.fromString, b' '.join([b'|1|x|x', b, c]))

    def test_hashedBadKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the key portion of the entry is valid base64, but is not actually an\n        SSH key, C{HashedEntry.fromString} should raise L{BadKeyError}.\n        '
        (a, b, c) = sampleHashedLine.split()
        self.badKeyTest(HashedEntry, a)

    def test_plainBadKey(self):
        if False:
            while True:
                i = 10
        '\n        If the key portion of the entry is valid base64, but is not actually an\n        SSH key, C{PlainEntry.fromString} should raise L{BadKeyError}.\n        '
        self.badKeyTest(PlainEntry, b'hostname')

class KnownHostsDatabaseTests(TestCase):
    """
    Tests for L{KnownHostsFile}.
    """

    def pathWithContent(self, content):
        if False:
            while True:
                i = 10
        '\n        Return a FilePath with the given initial content.\n        '
        fp = FilePath(self.mktemp())
        fp.setContent(content)
        return fp

    def loadSampleHostsFile(self, content=sampleHashedLine + otherSamplePlaintextLine + b'\n# That was a blank line.\nThis is just unparseable.\n|1|This also unparseable.\n'):
        if False:
            print('Hello World!')
        '\n        Return a sample hosts file, with keys for www.twistedmatrix.com and\n        divmod.com present.\n        '
        return KnownHostsFile.fromPath(self.pathWithContent(content))

    def test_readOnlySavePath(self):
        if False:
            return 10
        '\n        L{KnownHostsFile.savePath} is read-only; if an assignment is made to\n        it, L{AttributeError} is raised and the value is unchanged.\n        '
        path = FilePath(self.mktemp())
        new = FilePath(self.mktemp())
        hostsFile = KnownHostsFile(path)
        self.assertRaises(AttributeError, setattr, hostsFile, 'savePath', new)
        self.assertEqual(path, hostsFile.savePath)

    def test_defaultInitializerIgnoresExisting(self):
        if False:
            i = 10
            return i + 15
        '\n        The default initializer for L{KnownHostsFile} disregards any existing\n        contents in the save path.\n        '
        hostsFile = KnownHostsFile(self.pathWithContent(sampleHashedLine))
        self.assertEqual([], list(hostsFile.iterentries()))

    def test_defaultInitializerClobbersExisting(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        After using the default initializer for L{KnownHostsFile}, the first use\n        of L{KnownHostsFile.save} overwrites any existing contents in the save\n        path.\n        '
        path = self.pathWithContent(sampleHashedLine)
        hostsFile = KnownHostsFile(path)
        entry = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        hostsFile.save()
        self.assertEqual([entry], list(hostsFile.iterentries()))
        self.assertEqual(entry.toString() + b'\n', path.getContent())

    def test_saveResetsClobberState(self):
        if False:
            while True:
                i = 10
        '\n        After L{KnownHostsFile.save} is used once with an instance initialized\n        by the default initializer, contents of the save path are respected and\n        preserved.\n        '
        hostsFile = KnownHostsFile(self.pathWithContent(sampleHashedLine))
        preSave = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        hostsFile.save()
        postSave = hostsFile.addHostKey(b'another.example.com', Key.fromString(thirdSampleKey))
        hostsFile.save()
        self.assertEqual([preSave, postSave], list(hostsFile.iterentries()))

    def test_loadFromPath(self):
        if False:
            i = 10
            return i + 15
        '\n        Loading a L{KnownHostsFile} from a path with six entries in it will\n        result in a L{KnownHostsFile} object with six L{IKnownHostEntry}\n        providers in it.\n        '
        hostsFile = self.loadSampleHostsFile()
        self.assertEqual(6, len(list(hostsFile.iterentries())))

    def test_iterentriesUnsaved(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the save path for a L{KnownHostsFile} does not exist,\n        L{KnownHostsFile.iterentries} still returns added but unsaved entries.\n        '
        hostsFile = KnownHostsFile(FilePath(self.mktemp()))
        hostsFile.addHostKey(b'www.example.com', Key.fromString(sampleKey))
        self.assertEqual(1, len(list(hostsFile.iterentries())))

    def test_verifyHashedEntry(self):
        if False:
            while True:
                i = 10
        '\n        Loading a L{KnownHostsFile} from a path containing a single valid\n        L{HashedEntry} entry will result in a L{KnownHostsFile} object\n        with one L{IKnownHostEntry} provider.\n        '
        hostsFile = self.loadSampleHostsFile(sampleHashedLine)
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], HashedEntry)
        self.assertTrue(entries[0].matchesHost(b'www.twistedmatrix.com'))
        self.assertEqual(1, len(entries))

    def test_verifyPlainEntry(self):
        if False:
            while True:
                i = 10
        '\n        Loading a L{KnownHostsFile} from a path containing a single valid\n        L{PlainEntry} entry will result in a L{KnownHostsFile} object\n        with one L{IKnownHostEntry} provider.\n        '
        hostsFile = self.loadSampleHostsFile(otherSamplePlaintextLine)
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], PlainEntry)
        self.assertTrue(entries[0].matchesHost(b'divmod.com'))
        self.assertEqual(1, len(entries))

    def test_verifyUnparsedEntry(self):
        if False:
            print('Hello World!')
        "\n        Loading a L{KnownHostsFile} from a path that only contains '\n' will\n        result in a L{KnownHostsFile} object containing a L{UnparsedEntry}\n        object.\n        "
        hostsFile = self.loadSampleHostsFile(b'\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'')
        self.assertEqual(1, len(entries))

    def test_verifyUnparsedComment(self):
        if False:
            while True:
                i = 10
        '\n        Loading a L{KnownHostsFile} from a path that contains a comment will\n        result in a L{KnownHostsFile} object containing a L{UnparsedEntry}\n        object.\n        '
        hostsFile = self.loadSampleHostsFile(b'# That was a blank line.\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'# That was a blank line.')

    def test_verifyUnparsableLine(self):
        if False:
            return 10
        '\n        Loading a L{KnownHostsFile} from a path that contains an unparseable\n        line will be represented as an L{UnparsedEntry} instance.\n        '
        hostsFile = self.loadSampleHostsFile(b'This is just unparseable.\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'This is just unparseable.')
        self.assertEqual(1, len(entries))

    def test_verifyUnparsableEncryptionMarker(self):
        if False:
            i = 10
            return i + 15
        '\n        Loading a L{KnownHostsFile} from a path containing an unparseable line\n        that starts with an encryption marker will be represented as an\n        L{UnparsedEntry} instance.\n        '
        hostsFile = self.loadSampleHostsFile(b'|1|This is unparseable.\n')
        entries = list(hostsFile.iterentries())
        self.assertIsInstance(entries[0], UnparsedEntry)
        self.assertEqual(entries[0].toString(), b'|1|This is unparseable.')
        self.assertEqual(1, len(entries))

    def test_loadNonExistent(self):
        if False:
            while True:
                i = 10
        '\n        Loading a L{KnownHostsFile} from a path that does not exist should\n        result in an empty L{KnownHostsFile} that will save back to that path.\n        '
        pn = self.mktemp()
        knownHostsFile = KnownHostsFile.fromPath(FilePath(pn))
        entries = list(knownHostsFile.iterentries())
        self.assertEqual([], entries)
        self.assertFalse(FilePath(pn).exists())
        knownHostsFile.save()
        self.assertTrue(FilePath(pn).exists())

    def test_loadNonExistentParent(self):
        if False:
            i = 10
            return i + 15
        '\n        Loading a L{KnownHostsFile} from a path whose parent directory does not\n        exist should result in an empty L{KnownHostsFile} that will save back\n        to that path, creating its parent directory(ies) in the process.\n        '
        thePath = FilePath(self.mktemp())
        knownHostsPath = thePath.child('foo').child(b'known_hosts')
        knownHostsFile = KnownHostsFile.fromPath(knownHostsPath)
        knownHostsFile.save()
        knownHostsPath.restat(False)
        self.assertTrue(knownHostsPath.exists())

    def test_savingAddsEntry(self):
        if False:
            i = 10
            return i + 15
        '\n        L{KnownHostsFile.save} will write out a new file with any entries\n        that have been added.\n        '
        path = self.pathWithContent(sampleHashedLine + otherSamplePlaintextLine)
        knownHostsFile = KnownHostsFile.fromPath(path)
        newEntry = knownHostsFile.addHostKey(b'some.example.com', Key.fromString(thirdSampleKey))
        expectedContent = sampleHashedLine + otherSamplePlaintextLine + HashedEntry.MAGIC + b2a_base64(newEntry._hostSalt).strip() + b'|' + b2a_base64(newEntry._hostHash).strip() + b' ssh-rsa ' + thirdSampleEncodedKey + b'\n'
        self.assertEqual(3, expectedContent.count(b'\n'))
        knownHostsFile.save()
        self.assertEqual(expectedContent, path.getContent())

    def test_savingAvoidsDuplication(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{KnownHostsFile.save} only writes new entries to the save path, not\n        entries which were added and already written by a previous call to\n        C{save}.\n        '
        path = FilePath(self.mktemp())
        knownHosts = KnownHostsFile(path)
        entry = knownHosts.addHostKey(b'some.example.com', Key.fromString(sampleKey))
        knownHosts.save()
        knownHosts.save()
        knownHosts = KnownHostsFile.fromPath(path)
        self.assertEqual([entry], list(knownHosts.iterentries()))

    def test_savingsPreservesExisting(self):
        if False:
            print('Hello World!')
        '\n        L{KnownHostsFile.save} will not overwrite existing entries in its save\n        path, even if they were only added after the L{KnownHostsFile} instance\n        was initialized.\n        '
        path = self.pathWithContent(sampleHashedLine)
        knownHosts = KnownHostsFile.fromPath(path)
        with path.open('a') as hostsFileObj:
            hostsFileObj.write(otherSamplePlaintextLine)
        key = Key.fromString(thirdSampleKey)
        knownHosts.addHostKey(b'brandnew.example.com', key)
        knownHosts.save()
        knownHosts = KnownHostsFile.fromPath(path)
        self.assertEqual([True, True, True], [knownHosts.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)), knownHosts.hasHostKey(b'divmod.com', Key.fromString(otherSampleKey)), knownHosts.hasHostKey(b'brandnew.example.com', key)])

    def test_hasPresentKey(self):
        if False:
            return 10
        '\n        L{KnownHostsFile.hasHostKey} returns C{True} when a key for the given\n        hostname is present and matches the expected key.\n        '
        hostsFile = self.loadSampleHostsFile()
        self.assertTrue(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)))

    def test_notPresentKey(self):
        if False:
            i = 10
            return i + 15
        '\n        L{KnownHostsFile.hasHostKey} returns C{False} when a key for the given\n        hostname is not present.\n        '
        hostsFile = self.loadSampleHostsFile()
        self.assertFalse(hostsFile.hasHostKey(b'non-existent.example.com', Key.fromString(sampleKey)))
        self.assertTrue(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(sampleKey)))
        self.assertFalse(hostsFile.hasHostKey(b'www.twistedmatrix.com', Key.fromString(ecdsaSampleKey)))

    def test_hasLaterAddedKey(self):
        if False:
            print('Hello World!')
        '\n        L{KnownHostsFile.hasHostKey} returns C{True} when a key for the given\n        hostname is present in the file, even if it is only added to the file\n        after the L{KnownHostsFile} instance is initialized.\n        '
        key = Key.fromString(sampleKey)
        entry = PlainEntry([b'brandnew.example.com'], key.sshType(), key, b'')
        hostsFile = self.loadSampleHostsFile()
        with hostsFile.savePath.open('a') as hostsFileObj:
            hostsFileObj.write(entry.toString() + b'\n')
        self.assertEqual(True, hostsFile.hasHostKey(b'brandnew.example.com', key))

    def test_savedEntryHasKeyMismatch(self):
        if False:
            i = 10
            return i + 15
        '\n        L{KnownHostsFile.hasHostKey} raises L{HostKeyChanged} if the host key is\n        present in the underlying file, but different from the expected one.\n        The resulting exception should have an C{offendingEntry} indicating the\n        given entry.\n        '
        hostsFile = self.loadSampleHostsFile()
        entries = list(hostsFile.iterentries())
        exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
        self.assertEqual(exception.offendingEntry, entries[0])
        self.assertEqual(exception.lineno, 1)
        self.assertEqual(exception.path, hostsFile.savePath)

    def test_savedEntryAfterAddHasKeyMismatch(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Even after a new entry has been added in memory but not yet saved, the\n        L{HostKeyChanged} exception raised by L{KnownHostsFile.hasHostKey} has a\n        C{lineno} attribute which indicates the 1-based line number of the\n        offending entry in the underlying file when the given host key does not\n        match the expected host key.\n        '
        hostsFile = self.loadSampleHostsFile()
        hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
        self.assertEqual(exception.lineno, 1)
        self.assertEqual(exception.path, hostsFile.savePath)

    def test_unsavedEntryHasKeyMismatch(self):
        if False:
            return 10
        '\n        L{KnownHostsFile.hasHostKey} raises L{HostKeyChanged} if the host key is\n        present in memory (but not yet saved), but different from the expected\n        one.  The resulting exception has a C{offendingEntry} indicating the\n        given entry, but no filename or line number information (reflecting the\n        fact that the entry exists only in memory).\n        '
        hostsFile = KnownHostsFile(FilePath(self.mktemp()))
        entry = hostsFile.addHostKey(b'www.example.com', Key.fromString(otherSampleKey))
        exception = self.assertRaises(HostKeyChanged, hostsFile.hasHostKey, b'www.example.com', Key.fromString(thirdSampleKey))
        self.assertEqual(exception.offendingEntry, entry)
        self.assertIsNone(exception.lineno)
        self.assertIsNone(exception.path)

    def test_addHostKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{KnownHostsFile.addHostKey} adds a new L{HashedEntry} to the host\n        file, and returns it.\n        '
        hostsFile = self.loadSampleHostsFile()
        aKey = Key.fromString(thirdSampleKey)
        self.assertEqual(False, hostsFile.hasHostKey(b'somewhere.example.com', aKey))
        newEntry = hostsFile.addHostKey(b'somewhere.example.com', aKey)
        self.assertEqual(20, len(newEntry._hostSalt))
        self.assertEqual(True, newEntry.matchesHost(b'somewhere.example.com'))
        self.assertEqual(newEntry.keyType, b'ssh-rsa')
        self.assertEqual(aKey, newEntry.publicKey)
        self.assertEqual(True, hostsFile.hasHostKey(b'somewhere.example.com', aKey))

    def test_randomSalts(self):
        if False:
            while True:
                i = 10
        '\n        L{KnownHostsFile.addHostKey} generates a random salt for each new key,\n        so subsequent salts will be different.\n        '
        hostsFile = self.loadSampleHostsFile()
        aKey = Key.fromString(thirdSampleKey)
        self.assertNotEqual(hostsFile.addHostKey(b'somewhere.example.com', aKey)._hostSalt, hostsFile.addHostKey(b'somewhere-else.example.com', aKey)._hostSalt)

    def test_verifyValidKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verifying a valid key should return a L{Deferred} which fires with\n        True.\n        '
        hostsFile = self.loadSampleHostsFile()
        hostsFile.addHostKey(b'1.2.3.4', Key.fromString(sampleKey))
        ui = FakeUI()
        d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'1.2.3.4', Key.fromString(sampleKey))
        l = []
        d.addCallback(l.append)
        self.assertEqual(l, [True])

    def test_verifyInvalidKey(self):
        if False:
            print('Hello World!')
        '\n        Verifying an invalid key should return a L{Deferred} which fires with a\n        L{HostKeyChanged} failure.\n        '
        hostsFile = self.loadSampleHostsFile()
        wrongKey = Key.fromString(thirdSampleKey)
        ui = FakeUI()
        hostsFile.addHostKey(b'1.2.3.4', Key.fromString(sampleKey))
        d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'1.2.3.4', wrongKey)
        return self.assertFailure(d, HostKeyChanged)

    def verifyNonPresentKey(self):
        if False:
            while True:
                i = 10
        "\n        Set up a test to verify a key that isn't present.  Return a 3-tuple of\n        the UI, a list set up to collect the result of the verifyHostKey call,\n        and the sample L{KnownHostsFile} being used.\n\n        This utility method avoids returning a L{Deferred}, and records results\n        in the returned list instead, because the events which get generated\n        here are pre-recorded in the 'ui' object.  If the L{Deferred} in\n        question does not fire, the it will fail quickly with an empty list.\n        "
        hostsFile = self.loadSampleHostsFile()
        absentKey = Key.fromString(thirdSampleKey)
        ui = FakeUI()
        l = []
        d = hostsFile.verifyHostKey(ui, b'sample-host.example.com', b'4.3.2.1', absentKey)
        d.addBoth(l.append)
        self.assertEqual([], l)
        self.assertEqual(ui.promptText, b"The authenticity of host 'sample-host.example.com (4.3.2.1)' can't be established.\nRSA key fingerprint is SHA256:mS7mDBGhewdzJkaKRkx+wMjUdZb/GzvgcdoYjX5Js9I=.\nAre you sure you want to continue connecting (yes/no)? ")
        return (ui, l, hostsFile)

    def test_verifyNonPresentKey_Yes(self):
        if False:
            i = 10
            return i + 15
        '\n        Verifying a key where neither the hostname nor the IP are present\n        should result in the UI being prompted with a message explaining as\n        much.  If the UI says yes, the Deferred should fire with True.\n        '
        (ui, l, knownHostsFile) = self.verifyNonPresentKey()
        ui.promptDeferred.callback(True)
        self.assertEqual([True], l)
        reloaded = KnownHostsFile.fromPath(knownHostsFile.savePath)
        self.assertEqual(True, reloaded.hasHostKey(b'4.3.2.1', Key.fromString(thirdSampleKey)))
        self.assertEqual(True, reloaded.hasHostKey(b'sample-host.example.com', Key.fromString(thirdSampleKey)))

    def test_verifyNonPresentKey_No(self):
        if False:
            i = 10
            return i + 15
        '\n        Verifying a key where neither the hostname nor the IP are present\n        should result in the UI being prompted with a message explaining as\n        much.  If the UI says no, the Deferred should fail with\n        UserRejectedKey.\n        '
        (ui, l, knownHostsFile) = self.verifyNonPresentKey()
        ui.promptDeferred.callback(False)
        l[0].trap(UserRejectedKey)

    def test_verifyNonPresentECKey(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set up a test to verify an ECDSA key that isn't present.\n        Return a 3-tuple of the UI, a list set up to collect the result\n        of the verifyHostKey call, and the sample L{KnownHostsFile} being used.\n        "
        ecObj = Key._fromECComponents(x=keydata.ECDatanistp256['x'], y=keydata.ECDatanistp256['y'], privateValue=keydata.ECDatanistp256['privateValue'], curve=keydata.ECDatanistp256['curve'])
        hostsFile = self.loadSampleHostsFile()
        ui = FakeUI()
        l = []
        d = hostsFile.verifyHostKey(ui, b'sample-host.example.com', b'4.3.2.1', ecObj)
        d.addBoth(l.append)
        self.assertEqual([], l)
        self.assertEqual(ui.promptText, b"The authenticity of host 'sample-host.example.com (4.3.2.1)' can't be established.\nECDSA key fingerprint is SHA256:fJnSpgCcYoYYsaBbnWj1YBghGh/QTDgfe4w4U5M5tEo=.\nAre you sure you want to continue connecting (yes/no)? ")

    def test_verifyHostIPMismatch(self):
        if False:
            print('Hello World!')
        '\n        Verifying a key where the host is present (and correct), but the IP is\n        present and different, should result the deferred firing in a\n        HostKeyChanged failure.\n        '
        hostsFile = self.loadSampleHostsFile()
        wrongKey = Key.fromString(thirdSampleKey)
        ui = FakeUI()
        d = hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'4.3.2.1', wrongKey)
        return self.assertFailure(d, HostKeyChanged)

    def test_verifyKeyForHostAndIP(self):
        if False:
            i = 10
            return i + 15
        '\n        Verifying a key where the hostname is present but the IP is not should\n        result in the key being added for the IP and the user being warned\n        about the change.\n        '
        ui = FakeUI()
        hostsFile = self.loadSampleHostsFile()
        expectedKey = Key.fromString(sampleKey)
        hostsFile.verifyHostKey(ui, b'www.twistedmatrix.com', b'5.4.3.2', expectedKey)
        self.assertEqual(True, KnownHostsFile.fromPath(hostsFile.savePath).hasHostKey(b'5.4.3.2', expectedKey))
        self.assertEqual(["Warning: Permanently added the RSA host key for IP address '5.4.3.2' to the list of known hosts."], ui.userWarnings)

    def test_getHostKeyAlgorithms(self):
        if False:
            i = 10
            return i + 15
        '\n        For a given host, get the host key algorithms for that\n        host in the known_hosts file.\n        '
        hostsFile = self.loadSampleHostsFile()
        hostsFile.addHostKey(b'www.twistedmatrix.com', Key.fromString(otherSampleKey))
        hostsFile.addHostKey(b'www.twistedmatrix.com', Key.fromString(ecdsaSampleKey))
        hostsFile.save()
        options = {}
        options['known-hosts'] = hostsFile.savePath.path
        algorithms = default.getHostKeyAlgorithms(b'www.twistedmatrix.com', options)
        expectedAlgorithms = [b'ssh-rsa', b'ecdsa-sha2-nistp256']
        self.assertEqual(algorithms, expectedAlgorithms)

class FakeFile:
    """
    A fake file-like object that acts enough like a file for
    L{ConsoleUI.prompt}.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.inlines = []
        self.outchunks = []
        self.closed = False

    def readline(self):
        if False:
            return 10
        "\n        Return a line from the 'inlines' list.\n        "
        return self.inlines.pop(0)

    def write(self, chunk):
        if False:
            for i in range(10):
                print('nop')
        "\n        Append the given item to the 'outchunks' list.\n        "
        if self.closed:
            raise OSError('the file was closed')
        self.outchunks.append(chunk)

    def close(self):
        if False:
            i = 10
            return i + 15
        "\n        Set the 'closed' flag to True, explicitly marking that it has been\n        closed.\n        "
        self.closed = True

class ConsoleUITests(TestCase):
    """
    Test cases for L{ConsoleUI}.
    """

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create a L{ConsoleUI} pointed at a L{FakeFile}.\n        '
        self.fakeFile = FakeFile()
        self.ui = ConsoleUI(self.openFile)

    def openFile(self):
        if False:
            print('Hello World!')
        '\n        Return the current fake file.\n        '
        return self.fakeFile

    def newFile(self, lines):
        if False:
            print('Hello World!')
        '\n        Create a new fake file (the next file that self.ui will open) with the\n        given list of lines to be returned from readline().\n        '
        self.fakeFile = FakeFile()
        self.fakeFile.inlines = lines

    def test_promptYes(self):
        if False:
            while True:
                i = 10
        "\n        L{ConsoleUI.prompt} writes a message to the console, then reads a line.\n        If that line is 'yes', then it returns a L{Deferred} that fires with\n        True.\n        "
        for okYes in [b'yes', b'Yes', b'yes\n']:
            self.newFile([okYes])
            l = []
            self.ui.prompt('Hello, world!').addCallback(l.append)
            self.assertEqual(['Hello, world!'], self.fakeFile.outchunks)
            self.assertEqual([True], l)
            self.assertTrue(self.fakeFile.closed)

    def test_promptNo(self):
        if False:
            i = 10
            return i + 15
        "\n        L{ConsoleUI.prompt} writes a message to the console, then reads a line.\n        If that line is 'no', then it returns a L{Deferred} that fires with\n        False.\n        "
        for okNo in [b'no', b'No', b'no\n']:
            self.newFile([okNo])
            l = []
            self.ui.prompt('Goodbye, world!').addCallback(l.append)
            self.assertEqual(['Goodbye, world!'], self.fakeFile.outchunks)
            self.assertEqual([False], l)
            self.assertTrue(self.fakeFile.closed)

    def test_promptRepeatedly(self):
        if False:
            print('Hello World!')
        '\n        L{ConsoleUI.prompt} writes a message to the console, then reads a line.\n        If that line is neither \'yes\' nor \'no\', then it says "Please enter\n        \'yes\' or \'no\'" until it gets a \'yes\' or a \'no\', at which point it\n        returns a Deferred that answers either True or False.\n        '
        self.newFile([b'what', b'uh', b'okay', b'yes'])
        l = []
        self.ui.prompt(b'Please say something useful.').addCallback(l.append)
        self.assertEqual([True], l)
        self.assertEqual(self.fakeFile.outchunks, [b'Please say something useful.'] + [b"Please type 'yes' or 'no': "] * 3)
        self.assertTrue(self.fakeFile.closed)
        self.newFile([b'blah', b'stuff', b'feh', b'no'])
        l = []
        self.ui.prompt(b'Please say something negative.').addCallback(l.append)
        self.assertEqual([False], l)
        self.assertEqual(self.fakeFile.outchunks, [b'Please say something negative.'] + [b"Please type 'yes' or 'no': "] * 3)
        self.assertTrue(self.fakeFile.closed)

    def test_promptOpenFailed(self):
        if False:
            print('Hello World!')
        '\n        If the C{opener} passed to L{ConsoleUI} raises an exception, that\n        exception will fail the L{Deferred} returned from L{ConsoleUI.prompt}.\n        '

        def raiseIt():
            if False:
                i = 10
                return i + 15
            raise OSError()
        ui = ConsoleUI(raiseIt)
        d = ui.prompt('This is a test.')
        return self.assertFailure(d, IOError)

    def test_warn(self):
        if False:
            while True:
                i = 10
        '\n        L{ConsoleUI.warn} should output a message to the console object.\n        '
        self.ui.warn('Test message.')
        self.assertEqual(['Test message.'], self.fakeFile.outchunks)
        self.assertTrue(self.fakeFile.closed)

    def test_warnOpenFailed(self):
        if False:
            i = 10
            return i + 15
        "\n        L{ConsoleUI.warn} should log a traceback if the output can't be opened.\n        "

        def raiseIt():
            if False:
                print('Hello World!')
            1 / 0
        ui = ConsoleUI(raiseIt)
        ui.warn('This message never makes it.')
        self.assertEqual(len(self.flushLoggedErrors(ZeroDivisionError)), 1)

class FakeUI:
    """
    A fake UI object, adhering to the interface expected by
    L{KnownHostsFile.verifyHostKey}

    @ivar userWarnings: inputs provided to 'warn'.

    @ivar promptDeferred: last result returned from 'prompt'.

    @ivar promptText: the last input provided to 'prompt'.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.userWarnings = []
        self.promptDeferred = None
        self.promptText = None

    def prompt(self, text):
        if False:
            for i in range(10):
                print('nop')
        '\n        Issue the user an interactive prompt, which they can accept or deny.\n        '
        self.promptText = text
        self.promptDeferred = Deferred()
        return self.promptDeferred

    def warn(self, text):
        if False:
            i = 10
            return i + 15
        '\n        Issue a non-interactive warning to the user.\n        '
        self.userWarnings.append(text)

class FakeObject:
    """
    A fake object that can have some attributes.  Used to fake
    L{SSHClientTransport} and L{SSHClientFactory}.
    """

@skipIf(not FilePath('/dev/tty').exists(), 'Platform lacks /dev/tty')
class DefaultAPITests(TestCase):
    """
    The API in L{twisted.conch.client.default.verifyHostKey} is the integration
    point between the code in the rest of conch and L{KnownHostsFile}.
    """

    def patchedOpen(self, fname, mode, **kwargs):
        if False:
            print('Hello World!')
        "\n        The patched version of 'open'; this returns a L{FakeFile} that the\n        instantiated L{ConsoleUI} can use.\n        "
        self.assertEqual(fname, '/dev/tty')
        self.assertEqual(mode, 'r+b')
        self.assertEqual(kwargs['buffering'], 0)
        return self.fakeFile

    def setUp(self):
        if False:
            i = 10
            return i + 15
        "\n        Patch 'open' in verifyHostKey.\n        "
        self.fakeFile = FakeFile()
        self.patch(default, '_open', self.patchedOpen)
        self.hostsOption = self.mktemp()
        self.hashedEntries = {}
        knownHostsFile = KnownHostsFile(FilePath(self.hostsOption))
        for host in (b'exists.example.com', b'4.3.2.1'):
            entry = knownHostsFile.addHostKey(host, Key.fromString(sampleKey))
            self.hashedEntries[host] = entry
        knownHostsFile.save()
        self.fakeTransport = FakeObject()
        self.fakeTransport.factory = FakeObject()
        self.options = self.fakeTransport.factory.options = {'host': b'exists.example.com', 'known-hosts': self.hostsOption}

    def test_verifyOKKey(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{default.verifyHostKey} should return a L{Deferred} which fires with\n        C{1} when passed a host, IP, and key which already match the\n        known_hosts file it is supposed to check.\n        '
        l = []
        default.verifyHostKey(self.fakeTransport, b'4.3.2.1', sampleKey, b"I don't care.").addCallback(l.append)
        self.assertEqual([1], l)

    def replaceHome(self, tempHome):
        if False:
            print('Hello World!')
        '\n        Replace the HOME environment variable until the end of the current\n        test, with the given new home-directory, so that L{os.path.expanduser}\n        will yield controllable, predictable results.\n\n        @param tempHome: the pathname to replace the HOME variable with.\n\n        @type tempHome: L{str}\n        '
        oldHome = os.environ.get('HOME')

        def cleanupHome():
            if False:
                return 10
            if oldHome is None:
                del os.environ['HOME']
            else:
                os.environ['HOME'] = oldHome
        self.addCleanup(cleanupHome)
        os.environ['HOME'] = tempHome

    def test_noKnownHostsOption(self):
        if False:
            print('Hello World!')
        "\n        L{default.verifyHostKey} should find your known_hosts file in\n        ~/.ssh/known_hosts if you don't specify one explicitly on the command\n        line.\n        "
        l = []
        tmpdir = self.mktemp()
        oldHostsOption = self.hostsOption
        hostsNonOption = FilePath(tmpdir).child('.ssh').child('known_hosts')
        hostsNonOption.parent().makedirs()
        FilePath(oldHostsOption).moveTo(hostsNonOption)
        self.replaceHome(tmpdir)
        self.options['known-hosts'] = None
        default.verifyHostKey(self.fakeTransport, b'4.3.2.1', sampleKey, b"I don't care.").addCallback(l.append)
        self.assertEqual([1], l)

    def test_verifyHostButNotIP(self):
        if False:
            while True:
                i = 10
        '\n        L{default.verifyHostKey} should return a L{Deferred} which fires with\n        C{1} when passed a host which matches with an IP is not present in its\n        known_hosts file, and should also warn the user that it has added the\n        IP address.\n        '
        l = []
        default.verifyHostKey(self.fakeTransport, b'8.7.6.5', sampleKey, b'Fingerprint not required.').addCallback(l.append)
        self.assertEqual(["Warning: Permanently added the RSA host key for IP address '8.7.6.5' to the list of known hosts."], self.fakeFile.outchunks)
        self.assertEqual([1], l)
        knownHostsFile = KnownHostsFile.fromPath(FilePath(self.hostsOption))
        self.assertTrue(knownHostsFile.hasHostKey(b'8.7.6.5', Key.fromString(sampleKey)))

    def test_verifyQuestion(self):
        if False:
            print('Hello World!')
        '\n        L{default.verifyHostKey} should return a L{Default} which fires with\n        C{0} when passed an unknown host that the user refuses to acknowledge.\n        '
        self.fakeTransport.factory.options['host'] = b'fake.example.com'
        self.fakeFile.inlines.append(b'no')
        d = default.verifyHostKey(self.fakeTransport, b'9.8.7.6', otherSampleKey, b'No fingerprint!')
        self.assertEqual([b"The authenticity of host 'fake.example.com (9.8.7.6)' can't be established.\nRSA key fingerprint is SHA256:vD0YydsNIUYJa7yLZl3tIL8h0vZvQ8G+HPG7JLmQV0s=.\nAre you sure you want to continue connecting (yes/no)? "], self.fakeFile.outchunks)
        return self.assertFailure(d, UserRejectedKey)

    def test_verifyBadKey(self):
        if False:
            print('Hello World!')
        '\n        L{default.verifyHostKey} should return a L{Deferred} which fails with\n        L{HostKeyChanged} if the host key is incorrect.\n        '
        d = default.verifyHostKey(self.fakeTransport, b'4.3.2.1', otherSampleKey, 'Again, not required.')
        return self.assertFailure(d, HostKeyChanged)

    def test_inKnownHosts(self):
        if False:
            return 10
        '\n        L{default.isInKnownHosts} should return C{1} when a host with a key\n        is in the known hosts file.\n        '
        host = self.hashedEntries[b'4.3.2.1'].toString().split()[0]
        r = default.isInKnownHosts(host, Key.fromString(sampleKey).blob(), {'known-hosts': FilePath(self.hostsOption).path})
        self.assertEqual(1, r)

    def test_notInKnownHosts(self):
        if False:
            print('Hello World!')
        '\n        L{default.isInKnownHosts} should return C{0} when a host with a key\n        is not in the known hosts file.\n        '
        r = default.isInKnownHosts('not.there', b'irrelevant', {'known-hosts': FilePath(self.hostsOption).path})
        self.assertEqual(0, r)

    def test_inKnownHostsKeyChanged(self):
        if False:
            return 10
        '\n        L{default.isInKnownHosts} should return C{2} when a host with a key\n        other than the given one is in the known hosts file.\n        '
        host = self.hashedEntries[b'4.3.2.1'].toString().split()[0]
        r = default.isInKnownHosts(host, Key.fromString(otherSampleKey).blob(), {'known-hosts': FilePath(self.hostsOption).path})
        self.assertEqual(2, r)