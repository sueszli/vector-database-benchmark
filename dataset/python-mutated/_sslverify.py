from __future__ import annotations
import warnings
from binascii import hexlify
from functools import lru_cache
from hashlib import md5
from typing import Dict
from zope.interface import Interface, implementer
from OpenSSL import SSL, crypto
from OpenSSL._util import lib as pyOpenSSLlib
import attr
from constantly import FlagConstant, Flags, NamedConstant, Names
from incremental import Version
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.defer import Deferred
from twisted.internet.error import CertificateError, VerifyError
from twisted.internet.interfaces import IAcceptableCiphers, ICipher, IOpenSSLClientConnectionCreator, IOpenSSLContextFactory
from twisted.python import log, util
from twisted.python.compat import nativeString
from twisted.python.deprecate import _mutuallyExclusiveArguments, deprecated
from twisted.python.failure import Failure
from twisted.python.randbytes import secureRandom
from ._idna import _idnaBytes

class TLSVersion(Names):
    """
    TLS versions that we can negotiate with the client/server.
    """
    SSLv3 = NamedConstant()
    TLSv1_0 = NamedConstant()
    TLSv1_1 = NamedConstant()
    TLSv1_2 = NamedConstant()
    TLSv1_3 = NamedConstant()
_tlsDisableFlags = {TLSVersion.SSLv3: SSL.OP_NO_SSLv3, TLSVersion.TLSv1_0: SSL.OP_NO_TLSv1, TLSVersion.TLSv1_1: SSL.OP_NO_TLSv1_1, TLSVersion.TLSv1_2: SSL.OP_NO_TLSv1_2, TLSVersion.TLSv1_3: getattr(SSL, 'OP_NO_TLSv1_3', 0)}

def _getExcludedTLSProtocols(oldest, newest):
    if False:
        return 10
    '\n    Given a pair of L{TLSVersion} constants, figure out what versions we want\n    to disable (as OpenSSL is an exclusion based API).\n\n    @param oldest: The oldest L{TLSVersion} we want to allow.\n    @type oldest: L{TLSVersion} constant\n\n    @param newest: The newest L{TLSVersion} we want to allow, or L{None} for no\n        upper limit.\n    @type newest: L{TLSVersion} constant or L{None}\n\n    @return: The versions we want to disable.\n    @rtype: L{list} of L{TLSVersion} constants.\n    '
    versions = list(TLSVersion.iterconstants())
    excludedVersions = [x for x in versions[:versions.index(oldest)]]
    if newest:
        excludedVersions.extend([x for x in versions[versions.index(newest):]])
    return excludedVersions

class SimpleVerificationError(Exception):
    """
    Not a very useful verification error.
    """

def simpleVerifyHostname(connection, hostname):
    if False:
        while True:
            i = 10
    "\n    Check only the common name in the certificate presented by the peer and\n    only for an exact match.\n\n    This is to provide I{something} in the way of hostname verification to\n    users who haven't installed C{service_identity}. This check is overly\n    strict, relies on a deprecated TLS feature (you're supposed to ignore the\n    commonName if the subjectAlternativeName extensions are present, I\n    believe), and lots of valid certificates will fail.\n\n    @param connection: the OpenSSL connection to verify.\n    @type connection: L{OpenSSL.SSL.Connection}\n\n    @param hostname: The hostname expected by the user.\n    @type hostname: L{unicode}\n\n    @raise twisted.internet.ssl.VerificationError: if the common name and\n        hostname don't match.\n    "
    commonName = connection.get_peer_certificate().get_subject().commonName
    if commonName != hostname:
        raise SimpleVerificationError(repr(commonName) + '!=' + repr(hostname))

def simpleVerifyIPAddress(connection, hostname):
    if False:
        print('Hello World!')
    '\n    Always fails validation of IP addresses\n\n    @param connection: the OpenSSL connection to verify.\n    @type connection: L{OpenSSL.SSL.Connection}\n\n    @param hostname: The hostname expected by the user.\n    @type hostname: L{unicode}\n\n    @raise twisted.internet.ssl.VerificationError: Always raised\n    '
    raise SimpleVerificationError('Cannot verify certificate IP addresses')

def _usablePyOpenSSL(version):
    if False:
        return 10
    '\n    Check pyOpenSSL version string whether we can use it for host verification.\n\n    @param version: A pyOpenSSL version string.\n    @type version: L{str}\n\n    @rtype: L{bool}\n    '
    (major, minor) = (int(part) for part in version.split('.')[:2])
    return (major, minor) >= (0, 12)

def _selectVerifyImplementation():
    if False:
        return 10
    '\n    Determine if C{service_identity} is installed. If so, use it. If not, use\n    simplistic and incorrect checking as implemented in\n    L{simpleVerifyHostname}.\n\n    @return: 2-tuple of (C{verify_hostname}, C{VerificationError})\n    @rtype: L{tuple}\n    '
    whatsWrong = 'Without the service_identity module, Twisted can perform only rudimentary TLS client hostname verification.  Many valid certificate/hostname mappings may be rejected.'
    try:
        from service_identity import VerificationError
        from service_identity.pyopenssl import verify_hostname, verify_ip_address
        return (verify_hostname, verify_ip_address, VerificationError)
    except ImportError as e:
        warnings.warn_explicit("You do not have a working installation of the service_identity module: '" + str(e) + "'.  Please install it from <https://pypi.python.org/pypi/service_identity> and make sure all of its dependencies are satisfied.  " + whatsWrong, category=UserWarning, filename='', lineno=0)
    return (simpleVerifyHostname, simpleVerifyIPAddress, SimpleVerificationError)
(verifyHostname, verifyIPAddress, VerificationError) = _selectVerifyImplementation()

class ProtocolNegotiationSupport(Flags):
    """
    L{ProtocolNegotiationSupport} defines flags which are used to indicate the
    level of NPN/ALPN support provided by the TLS backend.

    @cvar NOSUPPORT: There is no support for NPN or ALPN. This is exclusive
        with both L{NPN} and L{ALPN}.
    @cvar NPN: The implementation supports Next Protocol Negotiation.
    @cvar ALPN: The implementation supports Application Layer Protocol
        Negotiation.
    """
    NPN = FlagConstant(1)
    ALPN = FlagConstant(2)
ProtocolNegotiationSupport.NOSUPPORT = ProtocolNegotiationSupport.NPN ^ ProtocolNegotiationSupport.NPN

def protocolNegotiationMechanisms():
    if False:
        while True:
            i = 10
    '\n    Checks whether your versions of PyOpenSSL and OpenSSL are recent enough to\n    support protocol negotiation, and if they are, what kind of protocol\n    negotiation is supported.\n\n    @return: A combination of flags from L{ProtocolNegotiationSupport} that\n        indicate which mechanisms for protocol negotiation are supported.\n    @rtype: L{constantly.FlagConstant}\n    '
    support = ProtocolNegotiationSupport.NOSUPPORT
    ctx = SSL.Context(SSL.SSLv23_METHOD)
    try:
        ctx.set_npn_advertise_callback(lambda c: None)
    except (AttributeError, NotImplementedError):
        pass
    else:
        support |= ProtocolNegotiationSupport.NPN
    try:
        ctx.set_alpn_select_callback(lambda c: None)
    except (AttributeError, NotImplementedError):
        pass
    else:
        support |= ProtocolNegotiationSupport.ALPN
    return support
_x509names = {'CN': 'commonName', 'commonName': 'commonName', 'O': 'organizationName', 'organizationName': 'organizationName', 'OU': 'organizationalUnitName', 'organizationalUnitName': 'organizationalUnitName', 'L': 'localityName', 'localityName': 'localityName', 'ST': 'stateOrProvinceName', 'stateOrProvinceName': 'stateOrProvinceName', 'C': 'countryName', 'countryName': 'countryName', 'emailAddress': 'emailAddress'}

class DistinguishedName(Dict[str, bytes]):
    """
    Identify and describe an entity.

    Distinguished names are used to provide a minimal amount of identifying
    information about a certificate issuer or subject.  They are commonly
    created with one or more of the following fields::

        commonName (CN)
        organizationName (O)
        organizationalUnitName (OU)
        localityName (L)
        stateOrProvinceName (ST)
        countryName (C)
        emailAddress

    A L{DistinguishedName} should be constructed using keyword arguments whose
    keys can be any of the field names above (as a native string), and the
    values are either Unicode text which is encodable to ASCII, or L{bytes}
    limited to the ASCII subset. Any fields passed to the constructor will be
    set as attributes, accessible using both their extended name and their
    shortened acronym. The attribute values will be the ASCII-encoded
    bytes. For example::

        >>> dn = DistinguishedName(commonName=b'www.example.com',
        ...                        C='US')
        >>> dn.C
        b'US'
        >>> dn.countryName
        b'US'
        >>> hasattr(dn, "organizationName")
        False

    L{DistinguishedName} instances can also be used as dictionaries; the keys
    are extended name of the fields::

        >>> dn.keys()
        ['countryName', 'commonName']
        >>> dn['countryName']
        b'US'

    """
    __slots__ = ()

    def __init__(self, **kw):
        if False:
            return 10
        for (k, v) in kw.items():
            setattr(self, k, v)

    def _copyFrom(self, x509name):
        if False:
            print('Hello World!')
        for name in _x509names:
            value = getattr(x509name, name, None)
            if value is not None:
                setattr(self, name, value)

    def _copyInto(self, x509name):
        if False:
            print('Hello World!')
        for (k, v) in self.items():
            setattr(x509name, k, nativeString(v))

    def __repr__(self) -> str:
        if False:
            return 10
        return '<DN %s>' % dict.__repr__(self)[1:-1]

    def __getattr__(self, attr):
        if False:
            return 10
        try:
            return self[_x509names[attr]]
        except KeyError:
            raise AttributeError(attr)

    def __setattr__(self, attr, value):
        if False:
            return 10
        if attr not in _x509names:
            raise AttributeError(f'{attr} is not a valid OpenSSL X509 name field')
        realAttr = _x509names[attr]
        if not isinstance(value, bytes):
            value = value.encode('ascii')
        self[realAttr] = value

    def inspect(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a multi-line, human-readable representation of this DN.\n\n        @rtype: L{str}\n        '
        l = []
        lablen = 0

        def uniqueValues(mapping):
            if False:
                print('Hello World!')
            return set(mapping.values())
        for k in sorted(uniqueValues(_x509names)):
            label = util.nameToLabel(k)
            lablen = max(len(label), lablen)
            v = getattr(self, k, None)
            if v is not None:
                l.append((label, nativeString(v)))
        lablen += 2
        for (n, (label, attrib)) in enumerate(l):
            l[n] = label.rjust(lablen) + ': ' + attrib
        return '\n'.join(l)
DN = DistinguishedName

class CertBase:
    """
    Base class for public (certificate only) and private (certificate + key
    pair) certificates.

    @ivar original: The underlying OpenSSL certificate object.
    @type original: L{OpenSSL.crypto.X509}
    """

    def __init__(self, original):
        if False:
            return 10
        self.original = original

    def _copyName(self, suffix):
        if False:
            while True:
                i = 10
        dn = DistinguishedName()
        dn._copyFrom(getattr(self.original, 'get_' + suffix)())
        return dn

    def getSubject(self):
        if False:
            while True:
                i = 10
        '\n        Retrieve the subject of this certificate.\n\n        @return: A copy of the subject of this certificate.\n        @rtype: L{DistinguishedName}\n        '
        return self._copyName('subject')

    def __conform__(self, interface):
        if False:
            while True:
                i = 10
        '\n        Convert this L{CertBase} into a provider of the given interface.\n\n        @param interface: The interface to conform to.\n        @type interface: L{zope.interface.interfaces.IInterface}\n\n        @return: an L{IOpenSSLTrustRoot} provider or L{NotImplemented}\n        @rtype: L{IOpenSSLTrustRoot} or L{NotImplemented}\n        '
        if interface is IOpenSSLTrustRoot:
            return OpenSSLCertificateAuthorities([self.original])
        return NotImplemented

def _handleattrhelper(Class, transport, methodName):
    if False:
        while True:
            i = 10
    '\n    (private) Helper for L{Certificate.peerFromTransport} and\n    L{Certificate.hostFromTransport} which checks for incompatible handle types\n    and null certificates and raises the appropriate exception or returns the\n    appropriate certificate object.\n    '
    method = getattr(transport.getHandle(), f'get_{methodName}_certificate', None)
    if method is None:
        raise CertificateError('non-TLS transport {!r} did not have {} certificate'.format(transport, methodName))
    cert = method()
    if cert is None:
        raise CertificateError('TLS transport {!r} did not have {} certificate'.format(transport, methodName))
    return Class(cert)

class Certificate(CertBase):
    """
    An x509 certificate.
    """

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return '<{} Subject={} Issuer={}>'.format(self.__class__.__name__, self.getSubject().commonName, self.getIssuer().commonName)

    def __eq__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        if isinstance(other, Certificate):
            return self.dump() == other.dump()
        return NotImplemented

    @classmethod
    def load(Class, requestData, format=crypto.FILETYPE_ASN1, args=()):
        if False:
            i = 10
            return i + 15
        '\n        Load a certificate from an ASN.1- or PEM-format string.\n\n        @rtype: C{Class}\n        '
        return Class(crypto.load_certificate(format, requestData), *args)
    _load = load

    def dumpPEM(self):
        if False:
            i = 10
            return i + 15
        '\n        Dump this certificate to a PEM-format data string.\n\n        @rtype: L{str}\n        '
        return self.dump(crypto.FILETYPE_PEM)

    @classmethod
    def loadPEM(Class, data):
        if False:
            print('Hello World!')
        '\n        Load a certificate from a PEM-format data string.\n\n        @rtype: C{Class}\n        '
        return Class.load(data, crypto.FILETYPE_PEM)

    @classmethod
    def peerFromTransport(Class, transport):
        if False:
            while True:
                i = 10
        '\n        Get the certificate for the remote end of the given transport.\n\n        @param transport: an L{ISystemHandle} provider\n\n        @rtype: C{Class}\n\n        @raise CertificateError: if the given transport does not have a peer\n            certificate.\n        '
        return _handleattrhelper(Class, transport, 'peer')

    @classmethod
    def hostFromTransport(Class, transport):
        if False:
            print('Hello World!')
        '\n        Get the certificate for the local end of the given transport.\n\n        @param transport: an L{ISystemHandle} provider; the transport we will\n\n        @rtype: C{Class}\n\n        @raise CertificateError: if the given transport does not have a host\n            certificate.\n        '
        return _handleattrhelper(Class, transport, 'host')

    def getPublicKey(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the public key for this certificate.\n\n        @rtype: L{PublicKey}\n        '
        return PublicKey(self.original.get_pubkey())

    def dump(self, format: int=crypto.FILETYPE_ASN1) -> bytes:
        if False:
            i = 10
            return i + 15
        return crypto.dump_certificate(format, self.original)

    def serialNumber(self):
        if False:
            i = 10
            return i + 15
        '\n        Retrieve the serial number of this certificate.\n\n        @rtype: L{int}\n        '
        return self.original.get_serial_number()

    def digest(self, method='md5'):
        if False:
            return 10
        '\n        Return a digest hash of this certificate using the specified hash\n        algorithm.\n\n        @param method: One of C{\'md5\'} or C{\'sha\'}.\n\n        @return: The digest of the object, formatted as b":"-delimited hex\n            pairs\n        @rtype: L{bytes}\n        '
        return self.original.digest(method)

    def _inspect(self):
        if False:
            i = 10
            return i + 15
        return '\n'.join(['Certificate For Subject:', self.getSubject().inspect(), '\nIssuer:', self.getIssuer().inspect(), '\nSerial Number: %d' % self.serialNumber(), 'Digest: %s' % nativeString(self.digest())])

    def inspect(self):
        if False:
            return 10
        '\n        Return a multi-line, human-readable representation of this\n        Certificate, including information about the subject, issuer, and\n        public key.\n        '
        return '\n'.join((self._inspect(), self.getPublicKey().inspect()))

    def getIssuer(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve the issuer of this certificate.\n\n        @rtype: L{DistinguishedName}\n        @return: A copy of the issuer of this certificate.\n        '
        return self._copyName('issuer')

    def options(self, *authorities):
        if False:
            return 10
        raise NotImplementedError('Possible, but doubtful we need this yet')

class CertificateRequest(CertBase):
    """
    An x509 certificate request.

    Certificate requests are given to certificate authorities to be signed and
    returned resulting in an actual certificate.
    """

    @classmethod
    def load(Class, requestData, requestFormat=crypto.FILETYPE_ASN1):
        if False:
            return 10
        req = crypto.load_certificate_request(requestFormat, requestData)
        dn = DistinguishedName()
        dn._copyFrom(req.get_subject())
        if not req.verify(req.get_pubkey()):
            raise VerifyError(f"Can't verify that request for {dn!r} is self-signed.")
        return Class(req)

    def dump(self, format=crypto.FILETYPE_ASN1):
        if False:
            i = 10
            return i + 15
        return crypto.dump_certificate_request(format, self.original)

class PrivateCertificate(Certificate):
    """
    An x509 certificate and private key.
    """

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return Certificate.__repr__(self) + ' with ' + repr(self.privateKey)

    def _setPrivateKey(self, privateKey):
        if False:
            for i in range(10):
                print('nop')
        if not privateKey.matches(self.getPublicKey()):
            raise VerifyError('Certificate public and private keys do not match.')
        self.privateKey = privateKey
        return self

    def newCertificate(self, newCertData, format=crypto.FILETYPE_ASN1):
        if False:
            while True:
                i = 10
        "\n        Create a new L{PrivateCertificate} from the given certificate data and\n        this instance's private key.\n        "
        return self.load(newCertData, self.privateKey, format)

    @classmethod
    def load(Class, data, privateKey, format=crypto.FILETYPE_ASN1):
        if False:
            print('Hello World!')
        return Class._load(data, format)._setPrivateKey(privateKey)

    def inspect(self):
        if False:
            while True:
                i = 10
        return '\n'.join([Certificate._inspect(self), self.privateKey.inspect()])

    def dumpPEM(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Dump both public and private parts of a private certificate to\n        PEM-format data.\n        '
        return self.dump(crypto.FILETYPE_PEM) + self.privateKey.dump(crypto.FILETYPE_PEM)

    @classmethod
    def loadPEM(Class, data):
        if False:
            i = 10
            return i + 15
        '\n        Load both private and public parts of a private certificate from a\n        chunk of PEM-format data.\n        '
        return Class.load(data, KeyPair.load(data, crypto.FILETYPE_PEM), crypto.FILETYPE_PEM)

    @classmethod
    def fromCertificateAndKeyPair(Class, certificateInstance, privateKey):
        if False:
            print('Hello World!')
        privcert = Class(certificateInstance.original)
        return privcert._setPrivateKey(privateKey)

    def options(self, *authorities):
        if False:
            print('Hello World!')
        "\n        Create a context factory using this L{PrivateCertificate}'s certificate\n        and private key.\n\n        @param authorities: A list of L{Certificate} object\n\n        @return: A context factory.\n        @rtype: L{CertificateOptions <twisted.internet.ssl.CertificateOptions>}\n        "
        options = dict(privateKey=self.privateKey.original, certificate=self.original)
        if authorities:
            options.update(dict(trustRoot=OpenSSLCertificateAuthorities([auth.original for auth in authorities])))
        return OpenSSLCertificateOptions(**options)

    def certificateRequest(self, format=crypto.FILETYPE_ASN1, digestAlgorithm='sha256'):
        if False:
            for i in range(10):
                print('nop')
        return self.privateKey.certificateRequest(self.getSubject(), format, digestAlgorithm)

    def signCertificateRequest(self, requestData, verifyDNCallback, serialNumber, requestFormat=crypto.FILETYPE_ASN1, certificateFormat=crypto.FILETYPE_ASN1):
        if False:
            print('Hello World!')
        issuer = self.getSubject()
        return self.privateKey.signCertificateRequest(issuer, requestData, verifyDNCallback, serialNumber, requestFormat, certificateFormat)

    def signRequestObject(self, certificateRequest, serialNumber, secondsToExpiry=60 * 60 * 24 * 365, digestAlgorithm='sha256'):
        if False:
            for i in range(10):
                print('nop')
        return self.privateKey.signRequestObject(self.getSubject(), certificateRequest, serialNumber, secondsToExpiry, digestAlgorithm)

class PublicKey:
    """
    A L{PublicKey} is a representation of the public part of a key pair.

    You can't do a whole lot with it aside from comparing it to other
    L{PublicKey} objects.

    @note: If constructing a L{PublicKey} manually, be sure to pass only a
        L{OpenSSL.crypto.PKey} that does not contain a private key!

    @ivar original: The original private key.
    """

    def __init__(self, osslpkey):
        if False:
            for i in range(10):
                print('nop')
        '\n        @param osslpkey: The underlying pyOpenSSL key object.\n        @type osslpkey: L{OpenSSL.crypto.PKey}\n        '
        self.original = osslpkey

    def matches(self, otherKey):
        if False:
            i = 10
            return i + 15
        '\n        Does this L{PublicKey} contain the same value as another L{PublicKey}?\n\n        @param otherKey: The key to compare C{self} to.\n        @type otherKey: L{PublicKey}\n\n        @return: L{True} if these keys match, L{False} if not.\n        @rtype: L{bool}\n        '
        return self.keyHash() == otherKey.keyHash()

    def __repr__(self) -> str:
        if False:
            print('Hello World!')
        return f'<{self.__class__.__name__} {self.keyHash()}>'

    def keyHash(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute a hash of the underlying PKey object.\n\n        The purpose of this method is to allow you to determine if two\n        certificates share the same public key; it is not really useful for\n        anything else.\n\n        In versions of Twisted prior to 15.0, C{keyHash} used a technique\n        involving certificate requests for computing the hash that was not\n        stable in the face of changes to the underlying OpenSSL library.\n\n        @return: Return a 32-character hexadecimal string uniquely identifying\n            this public key, I{for this version of Twisted}.\n        @rtype: native L{str}\n        '
        raw = crypto.dump_publickey(crypto.FILETYPE_ASN1, self.original)
        h = md5()
        h.update(raw)
        return h.hexdigest()

    def inspect(self):
        if False:
            i = 10
            return i + 15
        return f'Public Key with Hash: {self.keyHash()}'

class KeyPair(PublicKey):

    @classmethod
    def load(Class, data, format=crypto.FILETYPE_ASN1):
        if False:
            return 10
        return Class(crypto.load_privatekey(format, data))

    def dump(self, format=crypto.FILETYPE_ASN1):
        if False:
            return 10
        return crypto.dump_privatekey(format, self.original)

    @deprecated(Version('Twisted', 15, 0, 0), 'a real persistence system')
    def __getstate__(self):
        if False:
            i = 10
            return i + 15
        return self.dump()

    @deprecated(Version('Twisted', 15, 0, 0), 'a real persistence system')
    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.__init__(crypto.load_privatekey(crypto.FILETYPE_ASN1, state))

    def inspect(self):
        if False:
            while True:
                i = 10
        t = self.original.type()
        if t == crypto.TYPE_RSA:
            ts = 'RSA'
        elif t == crypto.TYPE_DSA:
            ts = 'DSA'
        else:
            ts = '(Unknown Type!)'
        L = (self.original.bits(), ts, self.keyHash())
        return '%s-bit %s Key Pair with Hash: %s' % L

    @classmethod
    def generate(Class, kind=crypto.TYPE_RSA, size=2048):
        if False:
            i = 10
            return i + 15
        pkey = crypto.PKey()
        pkey.generate_key(kind, size)
        return Class(pkey)

    def newCertificate(self, newCertData, format=crypto.FILETYPE_ASN1):
        if False:
            for i in range(10):
                print('nop')
        return PrivateCertificate.load(newCertData, self, format)

    def requestObject(self, distinguishedName, digestAlgorithm='sha256'):
        if False:
            return 10
        req = crypto.X509Req()
        req.set_pubkey(self.original)
        distinguishedName._copyInto(req.get_subject())
        req.sign(self.original, digestAlgorithm)
        return CertificateRequest(req)

    def certificateRequest(self, distinguishedName, format=crypto.FILETYPE_ASN1, digestAlgorithm='sha256'):
        if False:
            i = 10
            return i + 15
        "\n        Create a certificate request signed with this key.\n\n        @return: a string, formatted according to the 'format' argument.\n        "
        return self.requestObject(distinguishedName, digestAlgorithm).dump(format)

    def signCertificateRequest(self, issuerDistinguishedName, requestData, verifyDNCallback, serialNumber, requestFormat=crypto.FILETYPE_ASN1, certificateFormat=crypto.FILETYPE_ASN1, secondsToExpiry=60 * 60 * 24 * 365, digestAlgorithm='sha256'):
        if False:
            i = 10
            return i + 15
        "\n        Given a blob of certificate request data and a certificate authority's\n        DistinguishedName, return a blob of signed certificate data.\n\n        If verifyDNCallback returns a Deferred, I will return a Deferred which\n        fires the data when that Deferred has completed.\n        "
        hlreq = CertificateRequest.load(requestData, requestFormat)
        dn = hlreq.getSubject()
        vval = verifyDNCallback(dn)

        def verified(value):
            if False:
                return 10
            if not value:
                raise VerifyError('DN callback {!r} rejected request DN {!r}'.format(verifyDNCallback, dn))
            return self.signRequestObject(issuerDistinguishedName, hlreq, serialNumber, secondsToExpiry, digestAlgorithm).dump(certificateFormat)
        if isinstance(vval, Deferred):
            return vval.addCallback(verified)
        else:
            return verified(vval)

    def signRequestObject(self, issuerDistinguishedName, requestObject, serialNumber, secondsToExpiry=60 * 60 * 24 * 365, digestAlgorithm='sha256'):
        if False:
            i = 10
            return i + 15
        '\n        Sign a CertificateRequest instance, returning a Certificate instance.\n        '
        req = requestObject.original
        cert = crypto.X509()
        issuerDistinguishedName._copyInto(cert.get_issuer())
        cert.set_subject(req.get_subject())
        cert.set_pubkey(req.get_pubkey())
        cert.gmtime_adj_notBefore(0)
        cert.gmtime_adj_notAfter(secondsToExpiry)
        cert.set_serial_number(serialNumber)
        cert.sign(self.original, digestAlgorithm)
        return Certificate(cert)

    def selfSignedCert(self, serialNumber, **kw):
        if False:
            i = 10
            return i + 15
        dn = DN(**kw)
        return PrivateCertificate.fromCertificateAndKeyPair(self.signRequestObject(dn, self.requestObject(dn), serialNumber), self)

class IOpenSSLTrustRoot(Interface):
    """
    Trust settings for an OpenSSL context.

    Note that this interface's methods are private, so things outside of
    Twisted shouldn't implement it.
    """

    def _addCACertsToContext(context):
        if False:
            while True:
                i = 10
        '\n        Add certificate-authority certificates to an SSL context whose\n        connections should trust those authorities.\n\n        @param context: An SSL context for a connection which should be\n            verified by some certificate authority.\n        @type context: L{OpenSSL.SSL.Context}\n\n        @return: L{None}\n        '

@implementer(IOpenSSLTrustRoot)
class OpenSSLCertificateAuthorities:
    """
    Trust an explicitly specified set of certificates, represented by a list of
    L{OpenSSL.crypto.X509} objects.
    """

    def __init__(self, caCerts):
        if False:
            print('Hello World!')
        '\n        @param caCerts: The certificate authorities to trust when using this\n            object as a C{trustRoot} for L{OpenSSLCertificateOptions}.\n        @type caCerts: L{list} of L{OpenSSL.crypto.X509}\n        '
        self._caCerts = caCerts

    def _addCACertsToContext(self, context):
        if False:
            return 10
        store = context.get_cert_store()
        for cert in self._caCerts:
            store.add_cert(cert)

def trustRootFromCertificates(certificates):
    if False:
        print('Hello World!')
    '\n    Builds an object that trusts multiple root L{Certificate}s.\n\n    When passed to L{optionsForClientTLS}, connections using those options will\n    reject any server certificate not signed by at least one of the\n    certificates in the `certificates` list.\n\n    @since: 16.0\n\n    @param certificates: All certificates which will be trusted.\n    @type certificates: C{iterable} of L{CertBase}\n\n    @rtype: L{IOpenSSLTrustRoot}\n    @return: an object suitable for use as the trustRoot= keyword argument to\n        L{optionsForClientTLS}\n    '
    certs = []
    for cert in certificates:
        if isinstance(cert, CertBase):
            cert = cert.original
        else:
            raise TypeError('certificates items must be twisted.internet.ssl.CertBase instances')
        certs.append(cert)
    return OpenSSLCertificateAuthorities(certs)

@implementer(IOpenSSLTrustRoot)
class OpenSSLDefaultPaths:
    """
    Trust the set of default verify paths that OpenSSL was built with, as
    specified by U{SSL_CTX_set_default_verify_paths
    <https://www.openssl.org/docs/man1.1.1/man3/SSL_CTX_load_verify_locations.html>}.
    """

    def _addCACertsToContext(self, context):
        if False:
            i = 10
            return i + 15
        context.set_default_verify_paths()

def platformTrust():
    if False:
        for i in range(10):
            print('nop')
    '\n    Attempt to discover a set of trusted certificate authority certificates\n    (or, in other words: trust roots, or root certificates) whose trust is\n    managed and updated by tools outside of Twisted.\n\n    If you are writing any client-side TLS code with Twisted, you should use\n    this as the C{trustRoot} argument to L{CertificateOptions\n    <twisted.internet.ssl.CertificateOptions>}.\n\n    The result of this function should be like the up-to-date list of\n    certificates in a web browser.  When developing code that uses\n    C{platformTrust}, you can think of it that way.  However, the choice of\n    which certificate authorities to trust is never Twisted\'s responsibility.\n    Unless you\'re writing a very unusual application or library, it\'s not your\n    code\'s responsibility either.  The user may use platform-specific tools for\n    defining which server certificates should be trusted by programs using TLS.\n    The purpose of using this API is to respect that decision as much as\n    possible.\n\n    This should be a set of trust settings most appropriate for I{client} TLS\n    connections; i.e. those which need to verify a server\'s authenticity.  You\n    should probably use this by default for any client TLS connection that you\n    create.  For servers, however, client certificates are typically not\n    verified; or, if they are, their verification will depend on a custom,\n    application-specific certificate authority.\n\n    @since: 14.0\n\n    @note: Currently, L{platformTrust} depends entirely upon your OpenSSL build\n        supporting a set of "L{default verify paths <OpenSSLDefaultPaths>}"\n        which correspond to certificate authority trust roots.  Unfortunately,\n        whether this is true of your system is both outside of Twisted\'s\n        control and difficult (if not impossible) for Twisted to detect\n        automatically.\n\n        Nevertheless, this ought to work as desired by default on:\n\n            - Ubuntu Linux machines with the U{ca-certificates\n              <https://launchpad.net/ubuntu/+source/ca-certificates>} package\n              installed,\n\n            - macOS when using the system-installed version of OpenSSL (i.e.\n              I{not} one installed via MacPorts or Homebrew),\n\n            - any build of OpenSSL which has had certificate authority\n              certificates installed into its default verify paths (by default,\n              C{/usr/local/ssl/certs} if you\'ve built your own OpenSSL), or\n\n            - any process where the C{SSL_CERT_FILE} environment variable is\n              set to the path of a file containing your desired CA certificates\n              bundle.\n\n        Hopefully soon, this API will be updated to use more sophisticated\n        trust-root discovery mechanisms.  Until then, you can follow tickets in\n        the Twisted tracker for progress on this implementation on U{Microsoft\n        Windows <https://twistedmatrix.com/trac/ticket/6371>}, U{macOS\n        <https://twistedmatrix.com/trac/ticket/6372>}, and U{a fallback for\n        other platforms which do not have native trust management tools\n        <https://twistedmatrix.com/trac/ticket/6934>}.\n\n    @return: an appropriate trust settings object for your platform.\n    @rtype: L{IOpenSSLTrustRoot}\n\n    @raise NotImplementedError: if this platform is not yet supported by\n        Twisted.  At present, only OpenSSL is supported.\n    '
    return OpenSSLDefaultPaths()

def _tolerateErrors(wrapped):
    if False:
        for i in range(10):
            print('nop')
    "\n    Wrap up an C{info_callback} for pyOpenSSL so that if something goes wrong\n    the error is immediately logged and the connection is dropped if possible.\n\n    This wrapper exists because some versions of pyOpenSSL don't handle errors\n    from callbacks at I{all}, and those which do write tracebacks directly to\n    stderr rather than to a supplied logging system.  This reports unexpected\n    errors to the Twisted logging system.\n\n    Also, this terminates the connection immediately if possible because if\n    you've got bugs in your verification logic it's much safer to just give up.\n\n    @param wrapped: A valid C{info_callback} for pyOpenSSL.\n    @type wrapped: L{callable}\n\n    @return: A valid C{info_callback} for pyOpenSSL that handles any errors in\n        C{wrapped}.\n    @rtype: L{callable}\n    "

    def infoCallback(connection, where, ret):
        if False:
            i = 10
            return i + 15
        try:
            return wrapped(connection, where, ret)
        except BaseException:
            f = Failure()
            log.err(f, 'Error during info_callback')
            connection.get_app_data().failVerification(f)
    return infoCallback

@implementer(IOpenSSLClientConnectionCreator)
class ClientTLSOptions:
    """
    Client creator for TLS.

    Private implementation type (not exposed to applications) for public
    L{optionsForClientTLS} API.

    @ivar _ctx: The context to use for new connections.
    @type _ctx: L{OpenSSL.SSL.Context}

    @ivar _hostname: The hostname to verify, as specified by the application,
        as some human-readable text.
    @type _hostname: L{unicode}

    @ivar _hostnameBytes: The hostname to verify, decoded into IDNA-encoded
        bytes.  This is passed to APIs which think that hostnames are bytes,
        such as OpenSSL's SNI implementation.
    @type _hostnameBytes: L{bytes}

    @ivar _hostnameASCII: The hostname, as transcoded into IDNA ASCII-range
        unicode code points.  This is pre-transcoded because the
        C{service_identity} package is rather strict about requiring the
        C{idna} package from PyPI for internationalized domain names, rather
        than working with Python's built-in (but sometimes broken) IDNA
        encoding.  ASCII values, however, will always work.
    @type _hostnameASCII: L{unicode}

    @ivar _hostnameIsDnsName: Whether or not the C{_hostname} is a DNSName.
        Will be L{False} if C{_hostname} is an IP address or L{True} if
        C{_hostname} is a DNSName
    @type _hostnameIsDnsName: L{bool}
    """

    def __init__(self, hostname, ctx):
        if False:
            for i in range(10):
                print('nop')
        '\n        Initialize L{ClientTLSOptions}.\n\n        @param hostname: The hostname to verify as input by a human.\n        @type hostname: L{unicode}\n\n        @param ctx: an L{OpenSSL.SSL.Context} to use for new connections.\n        @type ctx: L{OpenSSL.SSL.Context}.\n        '
        self._ctx = ctx
        self._hostname = hostname
        if isIPAddress(hostname) or isIPv6Address(hostname):
            self._hostnameBytes = hostname.encode('ascii')
            self._hostnameIsDnsName = False
        else:
            self._hostnameBytes = _idnaBytes(hostname)
            self._hostnameIsDnsName = True
        self._hostnameASCII = self._hostnameBytes.decode('ascii')
        ctx.set_info_callback(_tolerateErrors(self._identityVerifyingInfoCallback))

    def clientConnectionForTLS(self, tlsProtocol):
        if False:
            while True:
                i = 10
        "\n        Create a TLS connection for a client.\n\n        @note: This will call C{set_app_data} on its connection.  If you're\n            delegating to this implementation of this method, don't ever call\n            C{set_app_data} or C{set_info_callback} on the returned connection,\n            or you'll break the implementation of various features of this\n            class.\n\n        @param tlsProtocol: the TLS protocol initiating the connection.\n        @type tlsProtocol: L{twisted.protocols.tls.TLSMemoryBIOProtocol}\n\n        @return: the configured client connection.\n        @rtype: L{OpenSSL.SSL.Connection}\n        "
        context = self._ctx
        connection = SSL.Connection(context, None)
        connection.set_app_data(tlsProtocol)
        return connection

    def _identityVerifyingInfoCallback(self, connection, where, ret):
        if False:
            return 10
        '\n        U{info_callback\n        <http://pythonhosted.org/pyOpenSSL/api/ssl.html#OpenSSL.SSL.Context.set_info_callback>\n        } for pyOpenSSL that verifies the hostname in the presented certificate\n        matches the one passed to this L{ClientTLSOptions}.\n\n        @param connection: the connection which is handshaking.\n        @type connection: L{OpenSSL.SSL.Connection}\n\n        @param where: flags indicating progress through a TLS handshake.\n        @type where: L{int}\n\n        @param ret: ignored\n        @type ret: ignored\n        '
        if where & SSL.SSL_CB_HANDSHAKE_START and self._hostnameIsDnsName:
            connection.set_tlsext_host_name(self._hostnameBytes)
        elif where & SSL.SSL_CB_HANDSHAKE_DONE:
            try:
                if self._hostnameIsDnsName:
                    verifyHostname(connection, self._hostnameASCII)
                else:
                    verifyIPAddress(connection, self._hostnameASCII)
            except VerificationError:
                f = Failure()
                transport = connection.get_app_data()
                transport.failVerification(f)

def optionsForClientTLS(hostname, trustRoot=None, clientCertificate=None, acceptableProtocols=None, *, extraCertificateOptions=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Create a L{client connection creator <IOpenSSLClientConnectionCreator>} for\n    use with APIs such as L{SSL4ClientEndpoint\n    <twisted.internet.endpoints.SSL4ClientEndpoint>}, L{connectSSL\n    <twisted.internet.interfaces.IReactorSSL.connectSSL>}, and L{startTLS\n    <twisted.internet.interfaces.ITLSTransport.startTLS>}.\n\n    @since: 14.0\n\n    @param hostname: The expected name of the remote host. This serves two\n        purposes: first, and most importantly, it verifies that the certificate\n        received from the server correctly identifies the specified hostname.\n        The second purpose is to use the U{Server Name Indication extension\n        <https://en.wikipedia.org/wiki/Server_Name_Indication>} to indicate to\n        the server which certificate should be used.\n    @type hostname: L{unicode}\n\n    @param trustRoot: Specification of trust requirements of peers. This may be\n        a L{Certificate} or the result of L{platformTrust}. By default it is\n        L{platformTrust} and you probably shouldn't adjust it unless you really\n        know what you're doing. Be aware that clients using this interface\n        I{must} verify the server; you cannot explicitly pass L{None} since\n        that just means to use L{platformTrust}.\n    @type trustRoot: L{IOpenSSLTrustRoot}\n\n    @param clientCertificate: The certificate and private key that the client\n        will use to authenticate to the server. If unspecified, the client will\n        not authenticate.\n    @type clientCertificate: L{PrivateCertificate}\n\n    @param acceptableProtocols: The protocols this peer is willing to speak\n        after the TLS negotiation has completed, advertised over both ALPN and\n        NPN. If this argument is specified, and no overlap can be found with\n        the other peer, the connection will fail to be established. If the\n        remote peer does not offer NPN or ALPN, the connection will be\n        established, but no protocol wil be negotiated. Protocols earlier in\n        the list are preferred over those later in the list.\n    @type acceptableProtocols: L{list} of L{bytes}\n\n    @param extraCertificateOptions: A dictionary of additional keyword arguments\n        to be presented to L{CertificateOptions}. Please avoid using this unless\n        you absolutely need to; any time you need to pass an option here that is\n        a bug in this interface.\n    @type extraCertificateOptions: L{dict}\n\n    @return: A client connection creator.\n    @rtype: L{IOpenSSLClientConnectionCreator}\n    "
    if extraCertificateOptions is None:
        extraCertificateOptions = {}
    if trustRoot is None:
        trustRoot = platformTrust()
    if not isinstance(hostname, str):
        raise TypeError('optionsForClientTLS requires text for host names, not ' + hostname.__class__.__name__)
    if clientCertificate:
        extraCertificateOptions.update(privateKey=clientCertificate.privateKey.original, certificate=clientCertificate.original)
    certificateOptions = OpenSSLCertificateOptions(trustRoot=trustRoot, acceptableProtocols=acceptableProtocols, **extraCertificateOptions)
    return ClientTLSOptions(hostname, certificateOptions.getContext())

@implementer(IOpenSSLContextFactory)
class OpenSSLCertificateOptions:
    """
    A L{CertificateOptions <twisted.internet.ssl.CertificateOptions>} specifies
    the security properties for a client or server TLS connection used with
    OpenSSL.

    @ivar _options: Any option flags to set on the L{OpenSSL.SSL.Context}
        object that will be created.
    @type _options: L{int}

    @ivar _cipherString: An OpenSSL-specific cipher string.
    @type _cipherString: L{unicode}

    @ivar _defaultMinimumTLSVersion: The default TLS version that will be
        negotiated.  This should be a "safe default", with wide client and
        server support, vs an optimally secure one that excludes a large number
        of users.  As of May 2022, TLSv1.2 is that safe default.
    @type _defaultMinimumTLSVersion: L{TLSVersion} constant
    """
    _contextFactory = SSL.Context
    _context = None
    _OP_NO_TLSv1_3 = _tlsDisableFlags[TLSVersion.TLSv1_3]
    _defaultMinimumTLSVersion = TLSVersion.TLSv1_2

    @_mutuallyExclusiveArguments([['trustRoot', 'requireCertificate'], ['trustRoot', 'verify'], ['trustRoot', 'caCerts'], ['method', 'insecurelyLowerMinimumTo'], ['method', 'raiseMinimumTo'], ['raiseMinimumTo', 'insecurelyLowerMinimumTo'], ['method', 'lowerMaximumSecurityTo']])
    def __init__(self, privateKey=None, certificate=None, method=None, verify=False, caCerts=None, verifyDepth=9, requireCertificate=True, verifyOnce=True, enableSingleUseKeys=True, enableSessions=False, fixBrokenPeers=False, enableSessionTickets=False, extraCertChain=None, acceptableCiphers=None, dhParameters=None, trustRoot=None, acceptableProtocols=None, raiseMinimumTo=None, insecurelyLowerMinimumTo=None, lowerMaximumSecurityTo=None):
        if False:
            return 10
        "\n        Create an OpenSSL context SSL connection context factory.\n\n        @param privateKey: A PKey object holding the private key.\n\n        @param certificate: An X509 object holding the certificate.\n\n        @param method: Deprecated, use a combination of\n            C{insecurelyLowerMinimumTo}, C{raiseMinimumTo}, or\n            C{lowerMaximumSecurityTo} instead.  The SSL protocol to use, one of\n            C{TLS_METHOD}, C{TLSv1_2_METHOD}, or C{TLSv1_2_METHOD} (or any\n            future method constants provided by pyOpenSSL).  By default, a\n            setting will be used which allows TLSv1.2 and TLSv1.3.  Can not be\n            used with C{insecurelyLowerMinimumTo}, C{raiseMinimumTo}, or\n            C{lowerMaximumSecurityTo}.\n\n        @param verify: Please use a C{trustRoot} keyword argument instead,\n            since it provides the same functionality in a less error-prone way.\n            By default this is L{False}.\n\n            If L{True}, verify certificates received from the peer and fail the\n            handshake if verification fails.  Otherwise, allow anonymous\n            sessions and sessions with certificates which fail validation.\n\n        @param caCerts: Please use a C{trustRoot} keyword argument instead,\n            since it provides the same functionality in a less error-prone way.\n\n            List of certificate authority certificate objects to use to verify\n            the peer's certificate.  Only used if verify is L{True} and will be\n            ignored otherwise.  Since verify is L{False} by default, this is\n            L{None} by default.\n\n        @type caCerts: L{list} of L{OpenSSL.crypto.X509}\n\n        @param verifyDepth: Depth in certificate chain down to which to verify.\n            If unspecified, use the underlying default (9).\n\n        @param requireCertificate: Please use a C{trustRoot} keyword argument\n            instead, since it provides the same functionality in a less\n            error-prone way.\n\n            If L{True}, do not allow anonymous sessions; defaults to L{True}.\n\n        @param verifyOnce: If True, do not re-verify the certificate on session\n            resumption.\n\n        @param enableSingleUseKeys: If L{True}, generate a new key whenever\n            ephemeral DH and ECDH parameters are used to prevent small subgroup\n            attacks and to ensure perfect forward secrecy.\n\n        @param enableSessions: This allows a shortened handshake to be used\n            when a known client reconnects to the same process.  If True,\n            enable OpenSSL's session caching.  Note that session caching only\n            works on a single Twisted node at once.  Also, it is currently\n            somewhat risky due to U{a crashing bug when using OpenSSL 1.1.1\n            <https://twistedmatrix.com/trac/ticket/9764>}.\n\n        @param fixBrokenPeers: If True, enable various non-spec protocol fixes\n            for broken SSL implementations.  This should be entirely safe,\n            according to the OpenSSL documentation, but YMMV.  This option is\n            now off by default, because it causes problems with connections\n            between peers using OpenSSL 0.9.8a.\n\n        @param enableSessionTickets: If L{True}, enable session ticket\n            extension for session resumption per RFC 5077.  Note there is no\n            support for controlling session tickets.  This option is off by\n            default, as some server implementations don't correctly process\n            incoming empty session ticket extensions in the hello.\n\n        @param extraCertChain: List of certificates that I{complete} your\n            verification chain if the certificate authority that signed your\n            C{certificate} isn't widely supported.  Do I{not} add\n            C{certificate} to it.\n        @type extraCertChain: C{list} of L{OpenSSL.crypto.X509}\n\n        @param acceptableCiphers: Ciphers that are acceptable for connections.\n            Uses a secure default if left L{None}.\n        @type acceptableCiphers: L{IAcceptableCiphers}\n\n        @param dhParameters: Key generation parameters that are required for\n            Diffie-Hellman key exchange.  If this argument is left L{None},\n            C{EDH} ciphers are I{disabled} regardless of C{acceptableCiphers}.\n        @type dhParameters: L{DiffieHellmanParameters\n            <twisted.internet.ssl.DiffieHellmanParameters>}\n\n        @param trustRoot: Specification of trust requirements of peers.  If\n            this argument is specified, the peer is verified.  It requires a\n            certificate, and that certificate must be signed by one of the\n            certificate authorities specified by this object.\n\n            Note that since this option specifies the same information as\n            C{caCerts}, C{verify}, and C{requireCertificate}, specifying any of\n            those options in combination with this one will raise a\n            L{TypeError}.\n\n        @type trustRoot: L{IOpenSSLTrustRoot}\n\n        @param acceptableProtocols: The protocols this peer is willing to speak\n            after the TLS negotiation has completed, advertised over both ALPN\n            and NPN.  If this argument is specified, and no overlap can be\n            found with the other peer, the connection will fail to be\n            established.  If the remote peer does not offer NPN or ALPN, the\n            connection will be established, but no protocol wil be negotiated.\n            Protocols earlier in the list are preferred over those later in the\n            list.\n        @type acceptableProtocols: L{list} of L{bytes}\n\n        @param raiseMinimumTo: The minimum TLS version that you want to use, or\n            Twisted's default if it is higher.  Use this if you want to make\n            your client/server more secure than Twisted's default, but will\n            accept Twisted's default instead if it moves higher than this\n            value.  You probably want to use this over\n            C{insecurelyLowerMinimumTo}.\n        @type raiseMinimumTo: L{TLSVersion} constant\n\n        @param insecurelyLowerMinimumTo: The minimum TLS version to use,\n            possibly lower than Twisted's default.  If not specified, it is a\n            generally considered safe default (TLSv1.0).  If you want to raise\n            your minimum TLS version to above that of this default, use\n            C{raiseMinimumTo}.  DO NOT use this argument unless you are\n            absolutely sure this is what you want.\n        @type insecurelyLowerMinimumTo: L{TLSVersion} constant\n\n        @param lowerMaximumSecurityTo: The maximum TLS version to use.  If not\n            specified, it is the most recent your OpenSSL supports.  You only\n            want to set this if the peer that you are communicating with has\n            problems with more recent TLS versions, it lowers your security\n            when communicating with newer peers.  DO NOT use this argument\n            unless you are absolutely sure this is what you want.\n        @type lowerMaximumSecurityTo: L{TLSVersion} constant\n\n        @raise ValueError: when C{privateKey} or C{certificate} are set without\n            setting the respective other.\n        @raise ValueError: when C{verify} is L{True} but C{caCerts} doesn't\n            specify any CA certificates.\n        @raise ValueError: when C{extraCertChain} is passed without specifying\n            C{privateKey} or C{certificate}.\n        @raise ValueError: when C{acceptableCiphers} doesn't yield any usable\n            ciphers for the current platform.\n\n        @raise TypeError: if C{trustRoot} is passed in combination with\n            C{caCert}, C{verify}, or C{requireCertificate}.  Please prefer\n            C{trustRoot} in new code, as its semantics are less tricky.\n        @raise TypeError: if C{method} is passed in combination with\n            C{tlsProtocols}.  Please prefer the more explicit C{tlsProtocols}\n            in new code.\n\n        @raises NotImplementedError: If acceptableProtocols were provided but\n            no negotiation mechanism is available.\n        "
        if (privateKey is None) != (certificate is None):
            raise ValueError('Specify neither or both of privateKey and certificate')
        self.privateKey = privateKey
        self.certificate = certificate
        self._options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE
        self._mode = SSL.MODE_RELEASE_BUFFERS
        if method is None:
            self.method = SSL.TLS_METHOD
            if raiseMinimumTo:
                if lowerMaximumSecurityTo and raiseMinimumTo > lowerMaximumSecurityTo:
                    raise ValueError('raiseMinimumTo needs to be lower than lowerMaximumSecurityTo')
                if raiseMinimumTo > self._defaultMinimumTLSVersion:
                    insecurelyLowerMinimumTo = raiseMinimumTo
            if insecurelyLowerMinimumTo is None:
                insecurelyLowerMinimumTo = self._defaultMinimumTLSVersion
                if lowerMaximumSecurityTo and insecurelyLowerMinimumTo > lowerMaximumSecurityTo:
                    insecurelyLowerMinimumTo = lowerMaximumSecurityTo
            if lowerMaximumSecurityTo and insecurelyLowerMinimumTo > lowerMaximumSecurityTo:
                raise ValueError('insecurelyLowerMinimumTo needs to be lower than lowerMaximumSecurityTo')
            excludedVersions = _getExcludedTLSProtocols(insecurelyLowerMinimumTo, lowerMaximumSecurityTo)
            for version in excludedVersions:
                self._options |= _tlsDisableFlags[version]
        else:
            warnings.warn('Passing method to twisted.internet.ssl.CertificateOptions was deprecated in Twisted 17.1.0. Please use a combination of insecurelyLowerMinimumTo, raiseMinimumTo, and lowerMaximumSecurityTo instead, as Twisted will correctly configure the method.', DeprecationWarning, stacklevel=3)
            self.method = method
        if verify and (not caCerts):
            raise ValueError('Specify client CA certificate information if and only if enabling certificate verification')
        self.verify = verify
        if extraCertChain is not None and None in (privateKey, certificate):
            raise ValueError('A private key and a certificate are required when adding a supplemental certificate chain.')
        if extraCertChain is not None:
            self.extraCertChain = extraCertChain
        else:
            self.extraCertChain = []
        self.caCerts = caCerts
        self.verifyDepth = verifyDepth
        self.requireCertificate = requireCertificate
        self.verifyOnce = verifyOnce
        self.enableSingleUseKeys = enableSingleUseKeys
        if enableSingleUseKeys:
            self._options |= SSL.OP_SINGLE_DH_USE | SSL.OP_SINGLE_ECDH_USE
        self.enableSessions = enableSessions
        self.fixBrokenPeers = fixBrokenPeers
        if fixBrokenPeers:
            self._options |= SSL.OP_ALL
        self.enableSessionTickets = enableSessionTickets
        if not enableSessionTickets:
            self._options |= SSL.OP_NO_TICKET
        self.dhParameters = dhParameters
        self._ecChooser = _ChooseDiffieHellmanEllipticCurve(SSL.OPENSSL_VERSION_NUMBER, openSSLlib=pyOpenSSLlib, openSSLcrypto=crypto)
        if acceptableCiphers is None:
            acceptableCiphers = defaultCiphers
        self._cipherString = ':'.join((c.fullName for c in acceptableCiphers.selectCiphers(_expandCipherString('ALL', self.method, self._options))))
        if self._cipherString == '':
            raise ValueError('Supplied IAcceptableCiphers yielded no usable ciphers on this platform.')
        if trustRoot is None:
            if self.verify:
                trustRoot = OpenSSLCertificateAuthorities(caCerts)
        else:
            self.verify = True
            self.requireCertificate = True
            trustRoot = IOpenSSLTrustRoot(trustRoot)
        self.trustRoot = trustRoot
        if acceptableProtocols is not None and (not protocolNegotiationMechanisms()):
            raise NotImplementedError('No support for protocol negotiation on this platform.')
        self._acceptableProtocols = acceptableProtocols

    def __getstate__(self):
        if False:
            return 10
        d = self.__dict__.copy()
        try:
            del d['_context']
        except KeyError:
            pass
        return d

    def __setstate__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self.__dict__ = state

    def getContext(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return an L{OpenSSL.SSL.Context} object.\n        '
        if self._context is None:
            self._context = self._makeContext()
        return self._context

    def _makeContext(self):
        if False:
            i = 10
            return i + 15
        ctx = self._contextFactory(self.method)
        ctx.set_options(self._options)
        ctx.set_mode(self._mode)
        if self.certificate is not None and self.privateKey is not None:
            ctx.use_certificate(self.certificate)
            ctx.use_privatekey(self.privateKey)
            for extraCert in self.extraCertChain:
                ctx.add_extra_chain_cert(extraCert)
            ctx.check_privatekey()
        verifyFlags = SSL.VERIFY_NONE
        if self.verify:
            verifyFlags = SSL.VERIFY_PEER
            if self.requireCertificate:
                verifyFlags |= SSL.VERIFY_FAIL_IF_NO_PEER_CERT
            if self.verifyOnce:
                verifyFlags |= SSL.VERIFY_CLIENT_ONCE
            self.trustRoot._addCACertsToContext(ctx)
        ctx.set_verify(verifyFlags)
        if self.verifyDepth is not None:
            ctx.set_verify_depth(self.verifyDepth)
        sessionIDContext = hexlify(secureRandom(7))
        ctx.set_session_id(sessionIDContext)
        if self.enableSessions:
            ctx.set_session_cache_mode(SSL.SESS_CACHE_SERVER)
        else:
            ctx.set_session_cache_mode(SSL.SESS_CACHE_OFF)
        if self.dhParameters:
            ctx.load_tmp_dh(self.dhParameters._dhFile.path)
        ctx.set_cipher_list(self._cipherString.encode('ascii'))
        self._ecChooser.configureECDHCurve(ctx)
        if self._acceptableProtocols:
            _setAcceptableProtocols(ctx, self._acceptableProtocols)
        return ctx
OpenSSLCertificateOptions.__getstate__ = deprecated(Version('Twisted', 15, 0, 0), 'a real persistence system')(OpenSSLCertificateOptions.__getstate__)
OpenSSLCertificateOptions.__setstate__ = deprecated(Version('Twisted', 15, 0, 0), 'a real persistence system')(OpenSSLCertificateOptions.__setstate__)

@implementer(ICipher)
@attr.s(frozen=True, auto_attribs=True)
class OpenSSLCipher:
    """
    A representation of an OpenSSL cipher.

    @ivar fullName: The full name of the cipher. For example
        C{u"ECDHE-RSA-AES256-GCM-SHA384"}.
    @type fullName: L{unicode}
    """
    fullName: str

@lru_cache(maxsize=32)
def _expandCipherString(cipherString, method, options):
    if False:
        i = 10
        return i + 15
    '\n    Expand C{cipherString} according to C{method} and C{options} to a tuple of\n    explicit ciphers that are supported by the current platform.\n\n    @param cipherString: An OpenSSL cipher string to expand.\n    @type cipherString: L{unicode}\n\n    @param method: An OpenSSL method like C{SSL.TLS_METHOD} used for\n        determining the effective ciphers.\n\n    @param options: OpenSSL options like C{SSL.OP_NO_SSLv3} ORed together.\n    @type options: L{int}\n\n    @return: The effective list of explicit ciphers that results from the\n        arguments on the current platform.\n    @rtype: L{tuple} of L{ICipher}\n    '
    ctx = SSL.Context(method)
    ctx.set_options(options)
    try:
        ctx.set_cipher_list(cipherString.encode('ascii'))
    except SSL.Error as e:
        if not e.args[0]:
            return tuple()
        if e.args[0][0][2] == 'no cipher match':
            return tuple()
        else:
            raise
    conn = SSL.Connection(ctx, None)
    ciphers = conn.get_cipher_list()
    if isinstance(ciphers[0], str):
        return tuple((OpenSSLCipher(cipher) for cipher in ciphers))
    else:
        return tuple((OpenSSLCipher(cipher.decode('ascii')) for cipher in ciphers))

@lru_cache(maxsize=128)
def _selectCiphers(wantedCiphers, availableCiphers):
    if False:
        print('Hello World!')
    '\n    Caclulate the acceptable list of ciphers from the ciphers we want and the\n    ciphers we have support for.\n\n    @param wantedCiphers: The ciphers we want to use.\n    @type wantedCiphers: L{tuple} of L{OpenSSLCipher}\n\n    @param availableCiphers: The ciphers we have available to use.\n    @type availableCiphers: L{tuple} of L{OpenSSLCipher}\n\n    @rtype: L{tuple} of L{OpenSSLCipher}\n    '
    return tuple((cipher for cipher in wantedCiphers if cipher in availableCiphers))

@implementer(IAcceptableCiphers)
class OpenSSLAcceptableCiphers:
    """
    A representation of ciphers that are acceptable for TLS connections.
    """

    def __init__(self, ciphers):
        if False:
            return 10
        self._ciphers = tuple(ciphers)

    def selectCiphers(self, availableCiphers):
        if False:
            while True:
                i = 10
        return _selectCiphers(self._ciphers, tuple(availableCiphers))

    @classmethod
    def fromOpenSSLCipherString(cls, cipherString):
        if False:
            return 10
        '\n        Create a new instance using an OpenSSL cipher string.\n\n        @param cipherString: An OpenSSL cipher string that describes what\n            cipher suites are acceptable.\n            See the documentation of U{OpenSSL\n            <http://www.openssl.org/docs/apps/ciphers.html#CIPHER_STRINGS>} or\n            U{Apache\n            <http://httpd.apache.org/docs/2.4/mod/mod_ssl.html#sslciphersuite>}\n            for details.\n        @type cipherString: L{unicode}\n\n        @return: Instance representing C{cipherString}.\n        @rtype: L{twisted.internet.ssl.AcceptableCiphers}\n        '
        return cls(_expandCipherString(nativeString(cipherString), SSL.TLS_METHOD, SSL.OP_NO_SSLv2 | SSL.OP_NO_SSLv3))
defaultCiphers = OpenSSLAcceptableCiphers.fromOpenSSLCipherString('TLS13-AES-256-GCM-SHA384:TLS13-CHACHA20-POLY1305-SHA256:TLS13-AES-128-GCM-SHA256:ECDH+AESGCM:ECDH+CHACHA20:DH+AESGCM:DH+CHACHA20:ECDH+AES256:DH+AES256:ECDH+AES128:DH+AES:RSA+AESGCM:RSA+AES:!aNULL:!MD5:!DSS')
_defaultCurveName = 'prime256v1'

class _ChooseDiffieHellmanEllipticCurve:
    """
    Chooses the best elliptic curve for Elliptic Curve Diffie-Hellman
    key exchange, and provides a C{configureECDHCurve} method to set
    the curve, when appropriate, on a new L{OpenSSL.SSL.Context}.

    The C{configureECDHCurve} method will be set to one of the
    following based on the provided OpenSSL version and configuration:

        - L{_configureOpenSSL110}

        - L{_configureOpenSSL102}

        - L{_configureOpenSSL101}

        - L{_configureOpenSSL101NoCurves}.

    @param openSSLVersion: The OpenSSL version number.
    @type openSSLVersion: L{int}

    @see: L{OpenSSL.SSL.OPENSSL_VERSION_NUMBER}

    @param openSSLlib: The OpenSSL C{cffi} library module.
    @param openSSLcrypto: The OpenSSL L{crypto} module.

    @see: L{crypto}
    """

    def __init__(self, openSSLVersion, openSSLlib, openSSLcrypto):
        if False:
            return 10
        self._openSSLlib = openSSLlib
        self._openSSLcrypto = openSSLcrypto
        if openSSLVersion >= 269484032:
            self.configureECDHCurve = self._configureOpenSSL110
        elif openSSLVersion >= 268443648:
            self.configureECDHCurve = self._configureOpenSSL102
        else:
            try:
                self._ecCurve = openSSLcrypto.get_elliptic_curve(_defaultCurveName)
            except ValueError:
                self.configureECDHCurve = self._configureOpenSSL101NoCurves
            else:
                self.configureECDHCurve = self._configureOpenSSL101

    def _configureOpenSSL110(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        '\n        OpenSSL 1.1.0 Contexts are preconfigured with an optimal set\n        of ECDH curves.  This method does nothing.\n\n        @param ctx: L{OpenSSL.SSL.Context}\n        '

    def _configureOpenSSL102(self, ctx):
        if False:
            return 10
        '\n        Have the context automatically choose elliptic curves for\n        ECDH.  Run on OpenSSL 1.0.2 and OpenSSL 1.1.0+, but only has\n        an effect on OpenSSL 1.0.2.\n\n        @param ctx: The context which .\n        @type ctx: L{OpenSSL.SSL.Context}\n        '
        ctxPtr = ctx._context
        try:
            self._openSSLlib.SSL_CTX_set_ecdh_auto(ctxPtr, True)
        except BaseException:
            pass

    def _configureOpenSSL101(self, ctx):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the default elliptic curve for ECDH on the context.  Only\n        run on OpenSSL 1.0.1.\n\n        @param ctx: The context on which to set the ECDH curve.\n        @type ctx: L{OpenSSL.SSL.Context}\n        '
        try:
            ctx.set_tmp_ecdh(self._ecCurve)
        except BaseException:
            pass

    def _configureOpenSSL101NoCurves(self, ctx):
        if False:
            i = 10
            return i + 15
        "\n        No elliptic curves are available on OpenSSL 1.0.1. We can't\n        set anything, so do nothing.\n\n        @param ctx: The context on which to set the ECDH curve.\n        @type ctx: L{OpenSSL.SSL.Context}\n        "

class OpenSSLDiffieHellmanParameters:
    """
    A representation of key generation parameters that are required for
    Diffie-Hellman key exchange.
    """

    def __init__(self, parameters):
        if False:
            i = 10
            return i + 15
        self._dhFile = parameters

    @classmethod
    def fromFile(cls, filePath):
        if False:
            for i in range(10):
                print('nop')
        "\n        Load parameters from a file.\n\n        Such a file can be generated using the C{openssl} command line tool as\n        following:\n\n        C{openssl dhparam -out dh_param_2048.pem -2 2048}\n\n        Please refer to U{OpenSSL's C{dhparam} documentation\n        <http://www.openssl.org/docs/apps/dhparam.html>} for further details.\n\n        @param filePath: A file containing parameters for Diffie-Hellman key\n            exchange.\n        @type filePath: L{FilePath <twisted.python.filepath.FilePath>}\n\n        @return: An instance that loads its parameters from C{filePath}.\n        @rtype: L{DiffieHellmanParameters\n            <twisted.internet.ssl.DiffieHellmanParameters>}\n        "
        return cls(filePath)

def _setAcceptableProtocols(context, acceptableProtocols):
    if False:
        print('Hello World!')
    '\n    Called to set up the L{OpenSSL.SSL.Context} for doing NPN and/or ALPN\n    negotiation.\n\n    @param context: The context which is set up.\n    @type context: L{OpenSSL.SSL.Context}\n\n    @param acceptableProtocols: The protocols this peer is willing to speak\n        after the TLS negotiation has completed, advertised over both ALPN and\n        NPN. If this argument is specified, and no overlap can be found with\n        the other peer, the connection will fail to be established. If the\n        remote peer does not offer NPN or ALPN, the connection will be\n        established, but no protocol wil be negotiated. Protocols earlier in\n        the list are preferred over those later in the list.\n    @type acceptableProtocols: L{list} of L{bytes}\n    '

    def protoSelectCallback(conn, protocols):
        if False:
            i = 10
            return i + 15
        '\n        NPN client-side and ALPN server-side callback used to select\n        the next protocol. Prefers protocols found earlier in\n        C{_acceptableProtocols}.\n\n        @param conn: The context which is set up.\n        @type conn: L{OpenSSL.SSL.Connection}\n\n        @param conn: Protocols advertised by the other side.\n        @type conn: L{list} of L{bytes}\n        '
        overlap = set(protocols) & set(acceptableProtocols)
        for p in acceptableProtocols:
            if p in overlap:
                return p
        else:
            return b''
    if not acceptableProtocols:
        return
    supported = protocolNegotiationMechanisms()
    if supported & ProtocolNegotiationSupport.NPN:

        def npnAdvertiseCallback(conn):
            if False:
                print('Hello World!')
            return acceptableProtocols
        context.set_npn_advertise_callback(npnAdvertiseCallback)
        context.set_npn_select_callback(protoSelectCallback)
    if supported & ProtocolNegotiationSupport.ALPN:
        context.set_alpn_select_callback(protoSelectCallback)
        context.set_alpn_protos(acceptableProtocols)