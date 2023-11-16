"""
An implementation of the OpenSSH known_hosts database.

@since: 8.2
"""
import hmac
import sys
from binascii import Error as DecodeError, a2b_base64, b2a_base64
from contextlib import closing
from hashlib import sha1
from zope.interface import implementer
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.conch.ssh.keys import BadKeyError, FingerprintFormats, Key
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString
from twisted.python.randbytes import secureRandom
from twisted.python.util import FancyEqMixin
log = Logger()

def _b64encode(s):
    if False:
        print('Hello World!')
    '\n    Encode a binary string as base64 with no trailing newline.\n\n    @param s: The string to encode.\n    @type s: L{bytes}\n\n    @return: The base64-encoded string.\n    @rtype: L{bytes}\n    '
    return b2a_base64(s).strip()

def _extractCommon(string):
    if False:
        print('Hello World!')
    '\n    Extract common elements of base64 keys from an entry in a hosts file.\n\n    @param string: A known hosts file entry (a single line).\n    @type string: L{bytes}\n\n    @return: a 4-tuple of hostname data (L{bytes}), ssh key type (L{bytes}), key\n        (L{Key}), and comment (L{bytes} or L{None}).  The hostname data is\n        simply the beginning of the line up to the first occurrence of\n        whitespace.\n    @rtype: L{tuple}\n    '
    elements = string.split(None, 2)
    if len(elements) != 3:
        raise InvalidEntry()
    (hostnames, keyType, keyAndComment) = elements
    splitkey = keyAndComment.split(None, 1)
    if len(splitkey) == 2:
        (keyString, comment) = splitkey
        comment = comment.rstrip(b'\n')
    else:
        keyString = splitkey[0]
        comment = None
    key = Key.fromString(a2b_base64(keyString))
    return (hostnames, keyType, key, comment)

class _BaseEntry:
    """
    Abstract base of both hashed and non-hashed entry objects, since they
    represent keys and key types the same way.

    @ivar keyType: The type of the key; either ssh-dss or ssh-rsa.
    @type keyType: L{bytes}

    @ivar publicKey: The server public key indicated by this line.
    @type publicKey: L{twisted.conch.ssh.keys.Key}

    @ivar comment: Trailing garbage after the key line.
    @type comment: L{bytes}
    """

    def __init__(self, keyType, publicKey, comment):
        if False:
            print('Hello World!')
        self.keyType = keyType
        self.publicKey = publicKey
        self.comment = comment

    def matchesKey(self, keyObject):
        if False:
            for i in range(10):
                print('nop')
        "\n        Check to see if this entry matches a given key object.\n\n        @param keyObject: A public key object to check.\n        @type keyObject: L{Key}\n\n        @return: C{True} if this entry's key matches C{keyObject}, C{False}\n            otherwise.\n        @rtype: L{bool}\n        "
        return self.publicKey == keyObject

@implementer(IKnownHostEntry)
class PlainEntry(_BaseEntry):
    """
    A L{PlainEntry} is a representation of a plain-text entry in a known_hosts
    file.

    @ivar _hostnames: the list of all host-names associated with this entry.
    @type _hostnames: L{list} of L{bytes}
    """

    def __init__(self, hostnames, keyType, publicKey, comment):
        if False:
            while True:
                i = 10
        self._hostnames = hostnames
        super().__init__(keyType, publicKey, comment)

    @classmethod
    def fromString(cls, string):
        if False:
            i = 10
            return i + 15
        '\n        Parse a plain-text entry in a known_hosts file, and return a\n        corresponding L{PlainEntry}.\n\n        @param string: a space-separated string formatted like "hostname\n        key-type base64-key-data comment".\n\n        @type string: L{bytes}\n\n        @raise DecodeError: if the key is not valid encoded as valid base64.\n\n        @raise InvalidEntry: if the entry does not have the right number of\n        elements and is therefore invalid.\n\n        @raise BadKeyError: if the key, once decoded from base64, is not\n        actually an SSH key.\n\n        @return: an IKnownHostEntry representing the hostname and key in the\n        input line.\n\n        @rtype: L{PlainEntry}\n        '
        (hostnames, keyType, key, comment) = _extractCommon(string)
        self = cls(hostnames.split(b','), keyType, key, comment)
        return self

    def matchesHost(self, hostname):
        if False:
            i = 10
            return i + 15
        '\n        Check to see if this entry matches a given hostname.\n\n        @param hostname: A hostname or IP address literal to check against this\n            entry.\n        @type hostname: L{bytes}\n\n        @return: C{True} if this entry is for the given hostname or IP address,\n            C{False} otherwise.\n        @rtype: L{bool}\n        '
        if isinstance(hostname, str):
            hostname = hostname.encode('utf-8')
        return hostname in self._hostnames

    def toString(self):
        if False:
            while True:
                i = 10
        '\n        Implement L{IKnownHostEntry.toString} by recording the comma-separated\n        hostnames, key type, and base-64 encoded key.\n\n        @return: The string representation of this entry, with unhashed hostname\n            information.\n        @rtype: L{bytes}\n        '
        fields = [b','.join(self._hostnames), self.keyType, _b64encode(self.publicKey.blob())]
        if self.comment is not None:
            fields.append(self.comment)
        return b' '.join(fields)

@implementer(IKnownHostEntry)
class UnparsedEntry:
    """
    L{UnparsedEntry} is an entry in a L{KnownHostsFile} which can't actually be
    parsed; therefore it matches no keys and no hosts.
    """

    def __init__(self, string):
        if False:
            print('Hello World!')
        '\n        Create an unparsed entry from a line in a known_hosts file which cannot\n        otherwise be parsed.\n        '
        self._string = string

    def matchesHost(self, hostname):
        if False:
            i = 10
            return i + 15
        '\n        Always returns False.\n        '
        return False

    def matchesKey(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Always returns False.\n        '
        return False

    def toString(self):
        if False:
            print('Hello World!')
        '\n        Returns the input line, without its newline if one was given.\n\n        @return: The string representation of this entry, almost exactly as was\n            used to initialize this entry but without a trailing newline.\n        @rtype: L{bytes}\n        '
        return self._string.rstrip(b'\n')

def _hmacedString(key, string):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return the SHA-1 HMAC hash of the given key and string.\n\n    @param key: The HMAC key.\n    @type key: L{bytes}\n\n    @param string: The string to be hashed.\n    @type string: L{bytes}\n\n    @return: The keyed hash value.\n    @rtype: L{bytes}\n    '
    hash = hmac.HMAC(key, digestmod=sha1)
    if isinstance(string, str):
        string = string.encode('utf-8')
    hash.update(string)
    return hash.digest()

@implementer(IKnownHostEntry)
class HashedEntry(_BaseEntry, FancyEqMixin):
    """
    A L{HashedEntry} is a representation of an entry in a known_hosts file
    where the hostname has been hashed and salted.

    @ivar _hostSalt: the salt to combine with a hostname for hashing.

    @ivar _hostHash: the hashed representation of the hostname.

    @cvar MAGIC: the 'hash magic' string used to identify a hashed line in a
    known_hosts file as opposed to a plaintext one.
    """
    MAGIC = b'|1|'
    compareAttributes = ('_hostSalt', '_hostHash', 'keyType', 'publicKey', 'comment')

    def __init__(self, hostSalt, hostHash, keyType, publicKey, comment):
        if False:
            i = 10
            return i + 15
        self._hostSalt = hostSalt
        self._hostHash = hostHash
        super().__init__(keyType, publicKey, comment)

    @classmethod
    def fromString(cls, string):
        if False:
            i = 10
            return i + 15
        '\n        Load a hashed entry from a string representing a line in a known_hosts\n        file.\n\n        @param string: A complete single line from a I{known_hosts} file,\n            formatted as defined by OpenSSH.\n        @type string: L{bytes}\n\n        @raise DecodeError: if the key, the hostname, or the is not valid\n            encoded as valid base64\n\n        @raise InvalidEntry: if the entry does not have the right number of\n            elements and is therefore invalid, or the host/hash portion contains\n            more items than just the host and hash.\n\n        @raise BadKeyError: if the key, once decoded from base64, is not\n            actually an SSH key.\n\n        @return: The newly created L{HashedEntry} instance, initialized with the\n            information from C{string}.\n        '
        (stuff, keyType, key, comment) = _extractCommon(string)
        saltAndHash = stuff[len(cls.MAGIC):].split(b'|')
        if len(saltAndHash) != 2:
            raise InvalidEntry()
        (hostSalt, hostHash) = saltAndHash
        self = cls(a2b_base64(hostSalt), a2b_base64(hostHash), keyType, key, comment)
        return self

    def matchesHost(self, hostname):
        if False:
            print('Hello World!')
        '\n        Implement L{IKnownHostEntry.matchesHost} to compare the hash of the\n        input to the stored hash.\n\n        @param hostname: A hostname or IP address literal to check against this\n            entry.\n        @type hostname: L{bytes}\n\n        @return: C{True} if this entry is for the given hostname or IP address,\n            C{False} otherwise.\n        @rtype: L{bool}\n        '
        return hmac.compare_digest(_hmacedString(self._hostSalt, hostname), self._hostHash)

    def toString(self):
        if False:
            while True:
                i = 10
        '\n        Implement L{IKnownHostEntry.toString} by base64-encoding the salt, host\n        hash, and key.\n\n        @return: The string representation of this entry, with the hostname part\n            hashed.\n        @rtype: L{bytes}\n        '
        fields = [self.MAGIC + b'|'.join([_b64encode(self._hostSalt), _b64encode(self._hostHash)]), self.keyType, _b64encode(self.publicKey.blob())]
        if self.comment is not None:
            fields.append(self.comment)
        return b' '.join(fields)

class KnownHostsFile:
    """
    A structured representation of an OpenSSH-format ~/.ssh/known_hosts file.

    @ivar _added: A list of L{IKnownHostEntry} providers which have been added
        to this instance in memory but not yet saved.

    @ivar _clobber: A flag indicating whether the current contents of the save
        path will be disregarded and potentially overwritten or not.  If
        C{True}, this will be done.  If C{False}, entries in the save path will
        be read and new entries will be saved by appending rather than
        overwriting.
    @type _clobber: L{bool}

    @ivar _savePath: See C{savePath} parameter of L{__init__}.
    """

    def __init__(self, savePath):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new, empty KnownHostsFile.\n\n        Unless you want to erase the current contents of C{savePath}, you want\n        to use L{KnownHostsFile.fromPath} instead.\n\n        @param savePath: The L{FilePath} to which to save new entries.\n        @type savePath: L{FilePath}\n        '
        self._added = []
        self._savePath = savePath
        self._clobber = True

    @property
    def savePath(self):
        if False:
            while True:
                i = 10
        '\n        @see: C{savePath} parameter of L{__init__}\n        '
        return self._savePath

    def iterentries(self):
        if False:
            while True:
                i = 10
        '\n        Iterate over the host entries in this file.\n\n        @return: An iterable the elements of which provide L{IKnownHostEntry}.\n            There is an element for each entry in the file as well as an element\n            for each added but not yet saved entry.\n        @rtype: iterable of L{IKnownHostEntry} providers\n        '
        for entry in self._added:
            yield entry
        if self._clobber:
            return
        try:
            fp = self._savePath.open()
        except OSError:
            return
        with fp:
            for line in fp:
                try:
                    if line.startswith(HashedEntry.MAGIC):
                        entry = HashedEntry.fromString(line)
                    else:
                        entry = PlainEntry.fromString(line)
                except (DecodeError, InvalidEntry, BadKeyError):
                    entry = UnparsedEntry(line)
                yield entry

    def hasHostKey(self, hostname, key):
        if False:
            print('Hello World!')
        '\n        Check for an entry with matching hostname and key.\n\n        @param hostname: A hostname or IP address literal to check for.\n        @type hostname: L{bytes}\n\n        @param key: The public key to check for.\n        @type key: L{Key}\n\n        @return: C{True} if the given hostname and key are present in this file,\n            C{False} if they are not.\n        @rtype: L{bool}\n\n        @raise HostKeyChanged: if the host key found for the given hostname\n            does not match the given key.\n        '
        for (lineidx, entry) in enumerate(self.iterentries(), -len(self._added)):
            if entry.matchesHost(hostname) and entry.keyType == key.sshType():
                if entry.matchesKey(key):
                    return True
                else:
                    if lineidx < 0:
                        line = None
                        path = None
                    else:
                        line = lineidx + 1
                        path = self._savePath
                    raise HostKeyChanged(entry, path, line)
        return False

    def verifyHostKey(self, ui, hostname, ip, key):
        if False:
            i = 10
            return i + 15
        '\n        Verify the given host key for the given IP and host, asking for\n        confirmation from, and notifying, the given UI about changes to this\n        file.\n\n        @param ui: The user interface to request an IP address from.\n\n        @param hostname: The hostname that the user requested to connect to.\n\n        @param ip: The string representation of the IP address that is actually\n        being connected to.\n\n        @param key: The public key of the server.\n\n        @return: a L{Deferred} that fires with True when the key has been\n            verified, or fires with an errback when the key either cannot be\n            verified or has changed.\n        @rtype: L{Deferred}\n        '
        hhk = defer.execute(self.hasHostKey, hostname, key)

        def gotHasKey(result):
            if False:
                print('Hello World!')
            if result:
                if not self.hasHostKey(ip, key):
                    ui.warn("Warning: Permanently added the %s host key for IP address '%s' to the list of known hosts." % (key.type(), nativeString(ip)))
                    self.addHostKey(ip, key)
                    self.save()
                return result
            else:

                def promptResponse(response):
                    if False:
                        return 10
                    if response:
                        self.addHostKey(hostname, key)
                        self.addHostKey(ip, key)
                        self.save()
                        return response
                    else:
                        raise UserRejectedKey()
                keytype = key.type()
                if keytype == 'EC':
                    keytype = 'ECDSA'
                prompt = "The authenticity of host '%s (%s)' can't be established.\n%s key fingerprint is SHA256:%s.\nAre you sure you want to continue connecting (yes/no)? " % (nativeString(hostname), nativeString(ip), keytype, key.fingerprint(format=FingerprintFormats.SHA256_BASE64))
                proceed = ui.prompt(prompt.encode(sys.getdefaultencoding()))
                return proceed.addCallback(promptResponse)
        return hhk.addCallback(gotHasKey)

    def addHostKey(self, hostname, key):
        if False:
            i = 10
            return i + 15
        '\n        Add a new L{HashedEntry} to the key database.\n\n        Note that you still need to call L{KnownHostsFile.save} if you wish\n        these changes to be persisted.\n\n        @param hostname: A hostname or IP address literal to associate with the\n            new entry.\n        @type hostname: L{bytes}\n\n        @param key: The public key to associate with the new entry.\n        @type key: L{Key}\n\n        @return: The L{HashedEntry} that was added.\n        @rtype: L{HashedEntry}\n        '
        salt = secureRandom(20)
        keyType = key.sshType()
        entry = HashedEntry(salt, _hmacedString(salt, hostname), keyType, key, None)
        self._added.append(entry)
        return entry

    def save(self):
        if False:
            return 10
        '\n        Save this L{KnownHostsFile} to the path it was loaded from.\n        '
        p = self._savePath.parent()
        if not p.isdir():
            p.makedirs()
        if self._clobber:
            mode = 'wb'
        else:
            mode = 'ab'
        with self._savePath.open(mode) as hostsFileObj:
            if self._added:
                hostsFileObj.write(b'\n'.join([entry.toString() for entry in self._added]) + b'\n')
                self._added = []
        self._clobber = False

    @classmethod
    def fromPath(cls, path):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a new L{KnownHostsFile}, potentially reading existing known\n        hosts information from the given file.\n\n        @param path: A path object to use for both reading contents from and\n            later saving to.  If no file exists at this path, it is not an\n            error; a L{KnownHostsFile} with no entries is returned.\n        @type path: L{FilePath}\n\n        @return: A L{KnownHostsFile} initialized with entries from C{path}.\n        @rtype: L{KnownHostsFile}\n        '
        knownHosts = cls(path)
        knownHosts._clobber = False
        return knownHosts

class ConsoleUI:
    """
    A UI object that can ask true/false questions and post notifications on the
    console, to be used during key verification.
    """

    def __init__(self, opener):
        if False:
            while True:
                i = 10
        '\n        @param opener: A no-argument callable which should open a console\n            binary-mode file-like object to be used for reading and writing.\n            This initializes the C{opener} attribute.\n        @type opener: callable taking no arguments and returning a read/write\n            file-like object\n        '
        self.opener = opener

    def prompt(self, text):
        if False:
            i = 10
            return i + 15
        "\n        Write the given text as a prompt to the console output, then read a\n        result from the console input.\n\n        @param text: Something to present to a user to solicit a yes or no\n            response.\n        @type text: L{bytes}\n\n        @return: a L{Deferred} which fires with L{True} when the user answers\n            'yes' and L{False} when the user answers 'no'.  It may errback if\n            there were any I/O errors.\n        "
        d = defer.succeed(None)

        def body(ignored):
            if False:
                while True:
                    i = 10
            with closing(self.opener()) as f:
                f.write(text)
                while True:
                    answer = f.readline().strip().lower()
                    if answer == b'yes':
                        return True
                    elif answer == b'no':
                        return False
                    else:
                        f.write(b"Please type 'yes' or 'no': ")
        return d.addCallback(body)

    def warn(self, text):
        if False:
            print('Hello World!')
        '\n        Notify the user (non-interactively) of the provided text, by writing it\n        to the console.\n\n        @param text: Some information the user is to be made aware of.\n        @type text: L{bytes}\n        '
        try:
            with closing(self.opener()) as f:
                f.write(text)
        except Exception:
            log.failure('Failed to write to console')