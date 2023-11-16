"""
Tests for L{twisted.internet._sslverify}.
"""
import datetime
import itertools
import sys
from unittest import skipIf
from zope.interface import implementer
from incremental import Version
from twisted.internet import defer, interfaces, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.error import CertificateError, ConnectionClosed, ConnectionLost
from twisted.internet.task import Clock
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.modules import getModule
from twisted.python.reflect import requireModule
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_twisted import SetAsideModule
from twisted.trial import util
from twisted.trial.unittest import SkipTest, SynchronousTestCase, TestCase
skipSSL = ''
skipSNI = ''
skipNPN = ''
skipALPN = ''
if requireModule('OpenSSL'):
    import ipaddress
    from OpenSSL import SSL
    from OpenSSL.crypto import FILETYPE_PEM, TYPE_RSA, X509, PKey, get_elliptic_curves
    from cryptography import x509
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import Encoding, NoEncryption, PrivateFormat
    from cryptography.x509.oid import NameOID
    from twisted.internet import ssl
    try:
        ctx = SSL.Context(SSL.SSLv23_METHOD)
        ctx.set_npn_advertise_callback(lambda c: None)
    except (NotImplementedError, AttributeError):
        skipNPN = 'NPN is deprecated (and OpenSSL 1.0.1 or greater required for NPN support)'
    try:
        ctx = SSL.Context(SSL.SSLv23_METHOD)
        ctx.set_alpn_select_callback(lambda c: None)
    except NotImplementedError:
        skipALPN = 'OpenSSL 1.0.2 or greater required for ALPN support'
else:
    skipSSL = 'OpenSSL is required for SSL tests.'
    skipSNI = skipSSL
    skipNPN = skipSSL
    skipALPN = skipSSL
if not skipSSL:
    from twisted.internet import _sslverify as sslverify
    from twisted.internet.ssl import VerificationError, platformTrust
    from twisted.protocols.tls import TLSMemoryBIOFactory
A_HOST_CERTIFICATE_PEM = '\n-----BEGIN CERTIFICATE-----\n        MIIC2jCCAkMCAjA5MA0GCSqGSIb3DQEBBAUAMIG0MQswCQYDVQQGEwJVUzEiMCAG\n        A1UEAxMZZXhhbXBsZS50d2lzdGVkbWF0cml4LmNvbTEPMA0GA1UEBxMGQm9zdG9u\n        MRwwGgYDVQQKExNUd2lzdGVkIE1hdHJpeCBMYWJzMRYwFAYDVQQIEw1NYXNzYWNo\n        dXNldHRzMScwJQYJKoZIhvcNAQkBFhhub2JvZHlAdHdpc3RlZG1hdHJpeC5jb20x\n        ETAPBgNVBAsTCFNlY3VyaXR5MB4XDTA2MDgxNjAxMDEwOFoXDTA3MDgxNjAxMDEw\n        OFowgbQxCzAJBgNVBAYTAlVTMSIwIAYDVQQDExlleGFtcGxlLnR3aXN0ZWRtYXRy\n        aXguY29tMQ8wDQYDVQQHEwZCb3N0b24xHDAaBgNVBAoTE1R3aXN0ZWQgTWF0cml4\n        IExhYnMxFjAUBgNVBAgTDU1hc3NhY2h1c2V0dHMxJzAlBgkqhkiG9w0BCQEWGG5v\n        Ym9keUB0d2lzdGVkbWF0cml4LmNvbTERMA8GA1UECxMIU2VjdXJpdHkwgZ8wDQYJ\n        KoZIhvcNAQEBBQADgY0AMIGJAoGBAMzH8CDF/U91y/bdbdbJKnLgnyvQ9Ig9ZNZp\n        8hpsu4huil60zF03+Lexg2l1FIfURScjBuaJMR6HiMYTMjhzLuByRZ17KW4wYkGi\n        KXstz03VIKy4Tjc+v4aXFI4XdRw10gGMGQlGGscXF/RSoN84VoDKBfOMWdXeConJ\n        VyC4w3iJAgMBAAEwDQYJKoZIhvcNAQEEBQADgYEAviMT4lBoxOgQy32LIgZ4lVCj\n        JNOiZYg8GMQ6y0ugp86X80UjOvkGtNf/R7YgED/giKRN/q/XJiLJDEhzknkocwmO\n        S+4b2XpiaZYxRyKWwL221O7CGmtWYyZl2+92YYmmCiNzWQPfP6BOMlfax0AGLHls\n        fXzCWdG0O/3Lk2SRM0I=\n-----END CERTIFICATE-----\n'
A_PEER_CERTIFICATE_PEM = '\n-----BEGIN CERTIFICATE-----\n        MIIC3jCCAkcCAjA6MA0GCSqGSIb3DQEBBAUAMIG2MQswCQYDVQQGEwJVUzEiMCAG\n        A1UEAxMZZXhhbXBsZS50d2lzdGVkbWF0cml4LmNvbTEPMA0GA1UEBxMGQm9zdG9u\n        MRwwGgYDVQQKExNUd2lzdGVkIE1hdHJpeCBMYWJzMRYwFAYDVQQIEw1NYXNzYWNo\n        dXNldHRzMSkwJwYJKoZIhvcNAQkBFhpzb21lYm9keUB0d2lzdGVkbWF0cml4LmNv\n        bTERMA8GA1UECxMIU2VjdXJpdHkwHhcNMDYwODE2MDEwMTU2WhcNMDcwODE2MDEw\n        MTU2WjCBtjELMAkGA1UEBhMCVVMxIjAgBgNVBAMTGWV4YW1wbGUudHdpc3RlZG1h\n        dHJpeC5jb20xDzANBgNVBAcTBkJvc3RvbjEcMBoGA1UEChMTVHdpc3RlZCBNYXRy\n        aXggTGFiczEWMBQGA1UECBMNTWFzc2FjaHVzZXR0czEpMCcGCSqGSIb3DQEJARYa\n        c29tZWJvZHlAdHdpc3RlZG1hdHJpeC5jb20xETAPBgNVBAsTCFNlY3VyaXR5MIGf\n        MA0GCSqGSIb3DQEBAQUAA4GNADCBiQKBgQCnm+WBlgFNbMlHehib9ePGGDXF+Nz4\n        CjGuUmVBaXCRCiVjg3kSDecwqfb0fqTksBZ+oQ1UBjMcSh7OcvFXJZnUesBikGWE\n        JE4V8Bjh+RmbJ1ZAlUPZ40bAkww0OpyIRAGMvKG+4yLFTO4WDxKmfDcrOb6ID8WJ\n        e1u+i3XGkIf/5QIDAQABMA0GCSqGSIb3DQEBBAUAA4GBAD4Oukm3YYkhedUepBEA\n        vvXIQhVDqL7mk6OqYdXmNj6R7ZMC8WWvGZxrzDI1bZuB+4aIxxd1FXC3UOHiR/xg\n        i9cDl1y8P/qRp4aEBNF6rI0D4AxTbfnHQx4ERDAOShJdYZs/2zifPJ6va6YvrEyr\n        yqDtGhklsWW3ZwBzEh5VEOUp\n-----END CERTIFICATE-----\n'
A_KEYPAIR = getModule(__name__).filePath.sibling('server.pem').getContent()

def counter(counter=itertools.count()):
    if False:
        return 10
    "\n    Each time we're called, return the next integer in the natural numbers.\n    "
    return next(counter)

def makeCertificate(**kw):
    if False:
        print('Hello World!')
    keypair = PKey()
    keypair.generate_key(TYPE_RSA, 2048)
    certificate = X509()
    certificate.gmtime_adj_notBefore(0)
    certificate.gmtime_adj_notAfter(60 * 60 * 24 * 365)
    for xname in (certificate.get_issuer(), certificate.get_subject()):
        for (k, v) in kw.items():
            setattr(xname, k, nativeString(v))
    certificate.set_serial_number(counter())
    certificate.set_pubkey(keypair)
    certificate.sign(keypair, 'md5')
    return (keypair, certificate)

def certificatesForAuthorityAndServer(serviceIdentity='example.com'):
    if False:
        while True:
            i = 10
    '\n    Create a self-signed CA certificate and server certificate signed by the\n    CA.\n\n    @param serviceIdentity: The identity (hostname) of the server.\n    @type serviceIdentity: L{unicode}\n\n    @return: a 2-tuple of C{(certificate_authority_certificate,\n        server_certificate)}\n    @rtype: L{tuple} of (L{sslverify.Certificate},\n        L{sslverify.PrivateCertificate})\n    '
    commonNameForCA = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Testing Example CA')])
    commonNameForServer = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, 'Testing Example Server')])
    oneDay = datetime.timedelta(1, 0, 0)
    privateKeyForCA = rsa.generate_private_key(public_exponent=65537, key_size=4096, backend=default_backend())
    publicKeyForCA = privateKeyForCA.public_key()
    caCertificate = x509.CertificateBuilder().subject_name(commonNameForCA).issuer_name(commonNameForCA).not_valid_before(datetime.datetime.today() - oneDay).not_valid_after(datetime.datetime.today() + oneDay).serial_number(x509.random_serial_number()).public_key(publicKeyForCA).add_extension(x509.BasicConstraints(ca=True, path_length=9), critical=True).sign(private_key=privateKeyForCA, algorithm=hashes.SHA256(), backend=default_backend())
    privateKeyForServer = rsa.generate_private_key(public_exponent=65537, key_size=4096, backend=default_backend())
    publicKeyForServer = privateKeyForServer.public_key()
    try:
        ipAddress = ipaddress.ip_address(serviceIdentity)
    except ValueError:
        subjectAlternativeNames = [x509.DNSName(serviceIdentity.encode('idna').decode('ascii'))]
    else:
        subjectAlternativeNames = [x509.IPAddress(ipAddress)]
    serverCertificate = x509.CertificateBuilder().subject_name(commonNameForServer).issuer_name(commonNameForCA).not_valid_before(datetime.datetime.today() - oneDay).not_valid_after(datetime.datetime.today() + oneDay).serial_number(x509.random_serial_number()).public_key(publicKeyForServer).add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True).add_extension(x509.SubjectAlternativeName(subjectAlternativeNames), critical=True).sign(private_key=privateKeyForCA, algorithm=hashes.SHA256(), backend=default_backend())
    caSelfCert = sslverify.Certificate.loadPEM(caCertificate.public_bytes(Encoding.PEM))
    serverCert = sslverify.PrivateCertificate.loadPEM(b'\n'.join([privateKeyForServer.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()), serverCertificate.public_bytes(Encoding.PEM)]))
    return (caSelfCert, serverCert)

def _loopbackTLSConnection(serverOpts, clientOpts):
    if False:
        i = 10
        return i + 15
    '\n    Common implementation code for both L{loopbackTLSConnection} and\n    L{loopbackTLSConnectionInMemory}. Creates a loopback TLS connection\n    using the provided server and client context factories.\n\n    @param serverOpts: An OpenSSL context factory for the server.\n    @type serverOpts: C{OpenSSLCertificateOptions}, or any class with an\n        equivalent API.\n\n    @param clientOpts: An OpenSSL context factory for the client.\n    @type clientOpts: C{OpenSSLCertificateOptions}, or any class with an\n        equivalent API.\n\n    @return: 5-tuple of server-tls-protocol, server-inner-protocol,\n        client-tls-protocol, client-inner-protocol and L{IOPump}\n    @rtype: L{tuple}\n    '

    class GreetingServer(protocol.Protocol):
        greeting = b'greetings!'

        def connectionMade(self):
            if False:
                for i in range(10):
                    print('nop')
            self.transport.write(self.greeting)

    class ListeningClient(protocol.Protocol):
        data = b''
        lostReason = None

        def dataReceived(self, data):
            if False:
                return 10
            self.data += data

        def connectionLost(self, reason):
            if False:
                while True:
                    i = 10
            self.lostReason = reason
    clientWrappedProto = ListeningClient()
    serverWrappedProto = GreetingServer()
    plainClientFactory = protocol.Factory()
    plainClientFactory.protocol = lambda : clientWrappedProto
    plainServerFactory = protocol.Factory()
    plainServerFactory.protocol = lambda : serverWrappedProto
    clock = Clock()
    clientFactory = TLSMemoryBIOFactory(clientOpts, isClient=True, wrappedFactory=plainServerFactory, clock=clock)
    serverFactory = TLSMemoryBIOFactory(serverOpts, isClient=False, wrappedFactory=plainClientFactory, clock=clock)
    (sProto, cProto, pump) = connectedServerAndClient(lambda : serverFactory.buildProtocol(None), lambda : clientFactory.buildProtocol(None), clock=clock)
    pump.flush()
    return (sProto, cProto, serverWrappedProto, clientWrappedProto, pump)

def loopbackTLSConnection(trustRoot, privateKeyFile, chainedCertFile=None):
    if False:
        i = 10
        return i + 15
    "\n    Create a loopback TLS connection with the given trust and keys.\n\n    @param trustRoot: the C{trustRoot} argument for the client connection's\n        context.\n    @type trustRoot: L{sslverify.IOpenSSLTrustRoot}\n\n    @param privateKeyFile: The name of the file containing the private key.\n    @type privateKeyFile: L{str} (native string; file name)\n\n    @param chainedCertFile: The name of the chained certificate file.\n    @type chainedCertFile: L{str} (native string; file name)\n\n    @return: 3-tuple of server-protocol, client-protocol, and L{IOPump}\n    @rtype: L{tuple}\n    "

    class ContextFactory:

        def getContext(self):
            if False:
                i = 10
                return i + 15
            '\n            Create a context for the server side of the connection.\n\n            @return: an SSL context using a certificate and key.\n            @rtype: C{OpenSSL.SSL.Context}\n            '
            ctx = SSL.Context(SSL.SSLv23_METHOD)
            if chainedCertFile is not None:
                ctx.use_certificate_chain_file(chainedCertFile)
            ctx.use_privatekey_file(privateKeyFile)
            ctx.check_privatekey()
            return ctx
    serverOpts = ContextFactory()
    clientOpts = sslverify.OpenSSLCertificateOptions(trustRoot=trustRoot)
    return _loopbackTLSConnection(serverOpts, clientOpts)

def loopbackTLSConnectionInMemory(trustRoot, privateKey, serverCertificate, clientProtocols=None, serverProtocols=None, clientOptions=None):
    if False:
        while True:
            i = 10
    "\n    Create a loopback TLS connection with the given trust and keys. Like\n    L{loopbackTLSConnection}, but using in-memory certificates and keys rather\n    than writing them to disk.\n\n    @param trustRoot: the C{trustRoot} argument for the client connection's\n        context.\n    @type trustRoot: L{sslverify.IOpenSSLTrustRoot}\n\n    @param privateKey: The private key.\n    @type privateKey: L{str} (native string)\n\n    @param serverCertificate: The certificate used by the server.\n    @type chainedCertFile: L{str} (native string)\n\n    @param clientProtocols: The protocols the client is willing to negotiate\n        using NPN/ALPN.\n\n    @param serverProtocols: The protocols the server is willing to negotiate\n        using NPN/ALPN.\n\n    @param clientOptions: The type of C{OpenSSLCertificateOptions} class to\n        use for the client. Defaults to C{OpenSSLCertificateOptions}.\n\n    @return: 3-tuple of server-protocol, client-protocol, and L{IOPump}\n    @rtype: L{tuple}\n    "
    if clientOptions is None:
        clientOptions = sslverify.OpenSSLCertificateOptions
    clientCertOpts = clientOptions(trustRoot=trustRoot, acceptableProtocols=clientProtocols)
    serverCertOpts = sslverify.OpenSSLCertificateOptions(privateKey=privateKey, certificate=serverCertificate, acceptableProtocols=serverProtocols)
    return _loopbackTLSConnection(serverCertOpts, clientCertOpts)

def pathContainingDumpOf(testCase, *dumpables):
    if False:
        return 10
    '\n    Create a temporary file to store some serializable-as-PEM objects in, and\n    return its name.\n\n    @param testCase: a test case to use for generating a temporary directory.\n    @type testCase: L{twisted.trial.unittest.TestCase}\n\n    @param dumpables: arguments are objects from pyOpenSSL with a C{dump}\n        method, taking a pyOpenSSL file-type constant, such as\n        L{OpenSSL.crypto.FILETYPE_PEM} or L{OpenSSL.crypto.FILETYPE_ASN1}.\n    @type dumpables: L{tuple} of L{object} with C{dump} method taking L{int}\n        returning L{bytes}\n\n    @return: the path to a file where all of the dumpables were dumped in PEM\n        format.\n    @rtype: L{str}\n    '
    fname = testCase.mktemp()
    with open(fname, 'wb') as f:
        for dumpable in dumpables:
            f.write(dumpable.dump(FILETYPE_PEM))
    return fname

class DataCallbackProtocol(protocol.Protocol):

    def dataReceived(self, data):
        if False:
            for i in range(10):
                print('nop')
        (d, self.factory.onData) = (self.factory.onData, None)
        if d is not None:
            d.callback(data)

    def connectionLost(self, reason):
        if False:
            return 10
        (d, self.factory.onLost) = (self.factory.onLost, None)
        if d is not None:
            d.errback(reason)

class WritingProtocol(protocol.Protocol):
    byte = b'x'

    def connectionMade(self):
        if False:
            return 10
        self.transport.write(self.byte)

    def connectionLost(self, reason):
        if False:
            while True:
                i = 10
        self.factory.onLost.errback(reason)

class FakeContext:
    """
    Introspectable fake of an C{OpenSSL.SSL.Context}.

    Saves call arguments for later introspection.

    Necessary because C{Context} offers poor introspection.  cf. this
    U{pyOpenSSL bug<https://bugs.launchpad.net/pyopenssl/+bug/1173899>}.

    @ivar _method: See C{method} parameter of L{__init__}.

    @ivar _options: L{int} of C{OR}ed values from calls of L{set_options}.

    @ivar _certificate: Set by L{use_certificate}.

    @ivar _privateKey: Set by L{use_privatekey}.

    @ivar _verify: Set by L{set_verify}.

    @ivar _verifyDepth: Set by L{set_verify_depth}.

    @ivar _mode: Set by L{set_mode}.

    @ivar _sessionID: Set by L{set_session_id}.

    @ivar _extraCertChain: Accumulated L{list} of all extra certificates added
        by L{add_extra_chain_cert}.

    @ivar _cipherList: Set by L{set_cipher_list}.

    @ivar _dhFilename: Set by L{load_tmp_dh}.

    @ivar _defaultVerifyPathsSet: Set by L{set_default_verify_paths}

    @ivar _ecCurve: Set by L{set_tmp_ecdh}
    """
    _options = 0

    def __init__(self, method):
        if False:
            for i in range(10):
                print('nop')
        self._method = method
        self._extraCertChain = []
        self._defaultVerifyPathsSet = False
        self._ecCurve = None
        self._sessionCacheMode = SSL.SESS_CACHE_SERVER

    def set_options(self, options):
        if False:
            return 10
        self._options |= options

    def use_certificate(self, certificate):
        if False:
            i = 10
            return i + 15
        self._certificate = certificate

    def use_privatekey(self, privateKey):
        if False:
            i = 10
            return i + 15
        self._privateKey = privateKey

    def check_privatekey(self):
        if False:
            while True:
                i = 10
        return None

    def set_mode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the mode. See L{SSL.Context.set_mode}.\n\n        @param mode: See L{SSL.Context.set_mode}.\n        '
        self._mode = mode

    def set_verify(self, flags, callback=None):
        if False:
            print('Hello World!')
        self._verify = (flags, callback)

    def set_verify_depth(self, depth):
        if False:
            i = 10
            return i + 15
        self._verifyDepth = depth

    def set_session_id(self, sessionIDContext):
        if False:
            while True:
                i = 10
        self._sessionIDContext = sessionIDContext

    def set_session_cache_mode(self, cacheMode):
        if False:
            print('Hello World!')
        '\n        Set the session cache mode on the context, as per\n        L{SSL.Context.set_session_cache_mode}.\n        '
        self._sessionCacheMode = cacheMode

    def get_session_cache_mode(self):
        if False:
            while True:
                i = 10
        '\n        Retrieve the session cache mode from the context, as per\n        L{SSL.Context.get_session_cache_mode}.\n        '
        return self._sessionCacheMode

    def add_extra_chain_cert(self, cert):
        if False:
            i = 10
            return i + 15
        self._extraCertChain.append(cert)

    def set_cipher_list(self, cipherList):
        if False:
            for i in range(10):
                print('nop')
        self._cipherList = cipherList

    def load_tmp_dh(self, dhfilename):
        if False:
            return 10
        self._dhFilename = dhfilename

    def set_default_verify_paths(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the default paths for the platform.\n        '
        self._defaultVerifyPathsSet = True

    def set_tmp_ecdh(self, curve):
        if False:
            while True:
                i = 10
        '\n        Set an ECDH curve.  Should only be called by OpenSSL 1.0.1\n        code.\n\n        @param curve: See L{OpenSSL.SSL.Context.set_tmp_ecdh}\n        '
        self._ecCurve = curve

class ClientOptionsTests(SynchronousTestCase):
    """
    Tests for L{sslverify.optionsForClientTLS}.
    """
    if skipSSL:
        skip = skipSSL

    def test_extraKeywords(self):
        if False:
            i = 10
            return i + 15
        '\n        When passed a keyword parameter other than C{extraCertificateOptions},\n        L{sslverify.optionsForClientTLS} raises an exception just like a\n        normal Python function would.\n        '
        error = self.assertRaises(TypeError, sslverify.optionsForClientTLS, hostname='alpha', someRandomThing='beta')
        self.assertEqual(str(error), "optionsForClientTLS() got an unexpected keyword argument 'someRandomThing'")

    def test_bytesFailFast(self):
        if False:
            i = 10
            return i + 15
        '\n        If you pass L{bytes} as the hostname to\n        L{sslverify.optionsForClientTLS} it immediately raises a L{TypeError}.\n        '
        error = self.assertRaises(TypeError, sslverify.optionsForClientTLS, b'not-actually-a-hostname.com')
        expectedText = 'optionsForClientTLS requires text for host names, not ' + bytes.__name__
        self.assertEqual(str(error), expectedText)

    def test_dNSNameHostname(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If you pass a dNSName to L{sslverify.optionsForClientTLS}\n        L{_hostnameIsDnsName} will be True\n        '
        options = sslverify.optionsForClientTLS('example.com')
        self.assertTrue(options._hostnameIsDnsName)

    def test_IPv4AddressHostname(self):
        if False:
            return 10
        '\n        If you pass an IPv4 address to L{sslverify.optionsForClientTLS}\n        L{_hostnameIsDnsName} will be False\n        '
        options = sslverify.optionsForClientTLS('127.0.0.1')
        self.assertFalse(options._hostnameIsDnsName)

    def test_IPv6AddressHostname(self):
        if False:
            while True:
                i = 10
        '\n        If you pass an IPv6 address to L{sslverify.optionsForClientTLS}\n        L{_hostnameIsDnsName} will be False\n        '
        options = sslverify.optionsForClientTLS('::1')
        self.assertFalse(options._hostnameIsDnsName)

class FakeChooseDiffieHellmanEllipticCurve:
    """
    A fake implementation of L{_ChooseDiffieHellmanEllipticCurve}
    """

    def __init__(self, versionNumber, openSSLlib, openSSLcrypto):
        if False:
            print('Hello World!')
        '\n        A no-op constructor.\n        '

    def configureECDHCurve(self, ctx):
        if False:
            while True:
                i = 10
        '\n        A null configuration.\n\n        @param ctx: An L{OpenSSL.SSL.Context} that would be\n            configured.\n        '

class OpenSSLOptionsTestsMixin:
    """
    A mixin for L{OpenSSLOptions} test cases creates client and server
    certificates, signs them with a CA, and provides a L{loopback}
    that creates TLS a connections with them.
    """
    if skipSSL:
        skip = skipSSL
    serverPort = clientConn = None
    onServerLost = onClientLost = None

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Create class variables of client and server certificates.\n        '
        (self.sKey, self.sCert) = makeCertificate(O=b'Server Test Certificate', CN=b'server')
        (self.cKey, self.cCert) = makeCertificate(O=b'Client Test Certificate', CN=b'client')
        self.caCert1 = makeCertificate(O=b'CA Test Certificate 1', CN=b'ca1')[1]
        self.caCert2 = makeCertificate(O=b'CA Test Certificate', CN=b'ca2')[1]
        self.caCerts = [self.caCert1, self.caCert2]
        self.extraCertChain = self.caCerts

    def tearDown(self):
        if False:
            return 10
        if self.serverPort is not None:
            self.serverPort.stopListening()
        if self.clientConn is not None:
            self.clientConn.disconnect()
        L = []
        if self.onServerLost is not None:
            L.append(self.onServerLost)
        if self.onClientLost is not None:
            L.append(self.onClientLost)
        return defer.DeferredList(L, consumeErrors=True)

    def loopback(self, serverCertOpts, clientCertOpts, onServerLost=None, onClientLost=None, onData=None):
        if False:
            return 10
        if onServerLost is None:
            self.onServerLost = onServerLost = defer.Deferred()
        if onClientLost is None:
            self.onClientLost = onClientLost = defer.Deferred()
        if onData is None:
            onData = defer.Deferred()
        serverFactory = protocol.ServerFactory()
        serverFactory.protocol = DataCallbackProtocol
        serverFactory.onLost = onServerLost
        serverFactory.onData = onData
        clientFactory = protocol.ClientFactory()
        clientFactory.protocol = WritingProtocol
        clientFactory.onLost = onClientLost
        self.serverPort = reactor.listenSSL(0, serverFactory, serverCertOpts)
        self.clientConn = reactor.connectSSL('127.0.0.1', self.serverPort.getHost().port, clientFactory, clientCertOpts)

class OpenSSLOptionsTests(OpenSSLOptionsTestsMixin, TestCase):
    """
    Tests for L{sslverify.OpenSSLOptions}.
    """

    def setUp(self):
        if False:
            print('Hello World!')
        '\n        Same as L{OpenSSLOptionsTestsMixin.setUp}, but it also patches\n        L{sslverify._ChooseDiffieHellmanEllipticCurve}.\n        '
        super().setUp()
        self.patch(sslverify, '_ChooseDiffieHellmanEllipticCurve', FakeChooseDiffieHellmanEllipticCurve)

    def test_constructorWithOnlyPrivateKey(self):
        if False:
            return 10
        '\n        C{privateKey} and C{certificate} make only sense if both are set.\n        '
        self.assertRaises(ValueError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey)

    def test_constructorWithOnlyCertificate(self):
        if False:
            while True:
                i = 10
        '\n        C{privateKey} and C{certificate} make only sense if both are set.\n        '
        self.assertRaises(ValueError, sslverify.OpenSSLCertificateOptions, certificate=self.sCert)

    def test_constructorWithCertificateAndPrivateKey(self):
        if False:
            while True:
                i = 10
        '\n        Specifying C{privateKey} and C{certificate} initializes correctly.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert)
        self.assertEqual(opts.privateKey, self.sKey)
        self.assertEqual(opts.certificate, self.sCert)
        self.assertEqual(opts.extraCertChain, [])

    def test_constructorDoesNotAllowVerifyWithoutCACerts(self):
        if False:
            print('Hello World!')
        '\n        C{verify} must not be C{True} without specifying C{caCerts}.\n        '
        self.assertRaises(ValueError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey, certificate=self.sCert, verify=True)

    def test_constructorDoesNotAllowLegacyWithTrustRoot(self):
        if False:
            i = 10
            return i + 15
        '\n        C{verify}, C{requireCertificate}, and C{caCerts} must not be specified\n        by the caller (to be I{any} value, even the default!) when specifying\n        C{trustRoot}.\n        '
        self.assertRaises(TypeError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey, certificate=self.sCert, verify=True, trustRoot=None, caCerts=self.caCerts)
        self.assertRaises(TypeError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey, certificate=self.sCert, trustRoot=None, requireCertificate=True)

    def test_constructorAllowsCACertsWithoutVerify(self):
        if False:
            while True:
                i = 10
        "\n        It's currently a NOP, but valid.\n        "
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, caCerts=self.caCerts)
        self.assertFalse(opts.verify)
        self.assertEqual(self.caCerts, opts.caCerts)

    def test_constructorWithVerifyAndCACerts(self):
        if False:
            print('Hello World!')
        '\n        Specifying C{verify} and C{caCerts} initializes correctly.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=True, caCerts=self.caCerts)
        self.assertTrue(opts.verify)
        self.assertEqual(self.caCerts, opts.caCerts)

    def test_constructorSetsExtraChain(self):
        if False:
            while True:
                i = 10
        '\n        Setting C{extraCertChain} works if C{certificate} and C{privateKey} are\n        set along with it.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, extraCertChain=self.extraCertChain)
        self.assertEqual(self.extraCertChain, opts.extraCertChain)

    def test_constructorDoesNotAllowExtraChainWithoutPrivateKey(self):
        if False:
            return 10
        "\n        A C{extraCertChain} without C{privateKey} doesn't make sense and is\n        thus rejected.\n        "
        self.assertRaises(ValueError, sslverify.OpenSSLCertificateOptions, certificate=self.sCert, extraCertChain=self.extraCertChain)

    def test_constructorDoesNotAllowExtraChainWithOutPrivateKey(self):
        if False:
            i = 10
            return i + 15
        "\n        A C{extraCertChain} without C{certificate} doesn't make sense and is\n        thus rejected.\n        "
        self.assertRaises(ValueError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey, extraCertChain=self.extraCertChain)

    def test_extraChainFilesAreAddedIfSupplied(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If C{extraCertChain} is set and all prerequisites are met, the\n        specified chain certificates are added to C{Context}s that get\n        created.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, extraCertChain=self.extraCertChain)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        self.assertEqual(self.sKey, ctx._privateKey)
        self.assertEqual(self.sCert, ctx._certificate)
        self.assertEqual(self.extraCertChain, ctx._extraCertChain)

    def test_extraChainDoesNotBreakPyOpenSSL(self):
        if False:
            print('Hello World!')
        "\n        C{extraCertChain} doesn't break C{OpenSSL.SSL.Context} creation.\n        "
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, extraCertChain=self.extraCertChain)
        ctx = opts.getContext()
        self.assertIsInstance(ctx, SSL.Context)

    def test_acceptableCiphersAreAlwaysSet(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If the user doesn't supply custom acceptable ciphers, a shipped secure\n        default is used.  We can't check directly for it because the effective\n        cipher string we set varies with platforms.\n        "
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        self.assertEqual(opts._cipherString.encode('ascii'), ctx._cipherList)

    def test_givesMeaningfulErrorMessageIfNoCipherMatches(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        If there is no valid cipher that matches the user's wishes,\n        a L{ValueError} is raised.\n        "
        self.assertRaises(ValueError, sslverify.OpenSSLCertificateOptions, privateKey=self.sKey, certificate=self.sCert, acceptableCiphers=sslverify.OpenSSLAcceptableCiphers.fromOpenSSLCipherString(''))

    def test_honorsAcceptableCiphersArgument(self):
        if False:
            while True:
                i = 10
        '\n        If acceptable ciphers are passed, they are used.\n        '

        @implementer(interfaces.IAcceptableCiphers)
        class FakeAcceptableCiphers:

            def selectCiphers(self, _):
                if False:
                    return 10
                return [sslverify.OpenSSLCipher('sentinel')]
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, acceptableCiphers=FakeAcceptableCiphers())
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        self.assertEqual(b'sentinel', ctx._cipherList)

    def test_basicSecurityOptionsAreSet(self):
        if False:
            return 10
        '\n        Every context must have C{OP_NO_SSLv2}, C{OP_NO_COMPRESSION}, and\n        C{OP_CIPHER_SERVER_PREFERENCE} set.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE
        self.assertEqual(options, ctx._options & options)

    def test_modeIsSet(self):
        if False:
            return 10
        '\n        Every context must be in C{MODE_RELEASE_BUFFERS} mode.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        self.assertEqual(SSL.MODE_RELEASE_BUFFERS, ctx._mode)

    def test_singleUseKeys(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If C{singleUseKeys} is set, every context must have\n        C{OP_SINGLE_DH_USE} and C{OP_SINGLE_ECDH_USE} set.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, enableSingleUseKeys=True)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_SINGLE_DH_USE | SSL.OP_SINGLE_ECDH_USE
        self.assertEqual(options, ctx._options & options)

    def test_methodIsDeprecated(self):
        if False:
            while True:
                i = 10
        '\n        Passing C{method} to L{sslverify.OpenSSLCertificateOptions} is\n        deprecated.\n        '
        sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.SSLv23_METHOD)
        message = 'Passing method to twisted.internet.ssl.CertificateOptions was deprecated in Twisted 17.1.0. Please use a combination of insecurelyLowerMinimumTo, raiseMinimumTo, and lowerMaximumSecurityTo instead, as Twisted will correctly configure the method.'
        warnings = self.flushWarnings([self.test_methodIsDeprecated])
        self.assertEqual(1, len(warnings))
        self.assertEqual(DeprecationWarning, warnings[0]['category'])
        self.assertEqual(message, warnings[0]['message'])

    def test_tlsv12ByDefault(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{sslverify.OpenSSLCertificateOptions} will make the default minimum\n        TLS version v1.2, if no C{method}, or C{insecurelyLowerMinimumTo} is\n        given.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsAtLeastWithMinimum(self):
        if False:
            return 10
        '\n        Passing C{insecurelyLowerMinimumTo} along with C{raiseMinimumTo} to\n        L{sslverify.OpenSSLCertificateOptions} will cause it to raise an\n        exception.\n        '
        with self.assertRaises(TypeError) as e:
            sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, raiseMinimumTo=sslverify.TLSVersion.TLSv1_2, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_2)
        self.assertIn('raiseMinimumTo', e.exception.args[0])
        self.assertIn('insecurelyLowerMinimumTo', e.exception.args[0])
        self.assertIn('exclusive', e.exception.args[0])

    def test_tlsProtocolsNoMethodWithAtLeast(self):
        if False:
            print('Hello World!')
        '\n        Passing C{raiseMinimumTo} along with C{method} to\n        L{sslverify.OpenSSLCertificateOptions} will cause it to raise an\n        exception.\n        '
        with self.assertRaises(TypeError) as e:
            sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.SSLv23_METHOD, raiseMinimumTo=sslverify.TLSVersion.TLSv1_2)
        self.assertIn('method', e.exception.args[0])
        self.assertIn('raiseMinimumTo', e.exception.args[0])
        self.assertIn('exclusive', e.exception.args[0])

    def test_tlsProtocolsNoMethodWithMinimum(self):
        if False:
            i = 10
            return i + 15
        '\n        Passing C{insecurelyLowerMinimumTo} along with C{method} to\n        L{sslverify.OpenSSLCertificateOptions} will cause it to raise an\n        exception.\n        '
        with self.assertRaises(TypeError) as e:
            sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.SSLv23_METHOD, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_2)
        self.assertIn('method', e.exception.args[0])
        self.assertIn('insecurelyLowerMinimumTo', e.exception.args[0])
        self.assertIn('exclusive', e.exception.args[0])

    def test_tlsProtocolsNoMethodWithMaximum(self):
        if False:
            return 10
        '\n        Passing C{lowerMaximumSecurityTo} along with C{method} to\n        L{sslverify.OpenSSLCertificateOptions} will cause it to raise an\n        exception.\n        '
        with self.assertRaises(TypeError) as e:
            sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.TLS_METHOD, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_2)
        self.assertIn('method', e.exception.args[0])
        self.assertIn('lowerMaximumSecurityTo', e.exception.args[0])
        self.assertIn('exclusive', e.exception.args[0])

    def test_tlsVersionRangeInOrder(self):
        if False:
            return 10
        '\n        Passing out of order TLS versions to C{insecurelyLowerMinimumTo} and\n        C{lowerMaximumSecurityTo} will cause it to raise an exception.\n        '
        with self.assertRaises(ValueError) as e:
            sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_0, lowerMaximumSecurityTo=sslverify.TLSVersion.SSLv3)
        self.assertEqual(e.exception.args, ('insecurelyLowerMinimumTo needs to be lower than lowerMaximumSecurityTo',))

    def test_tlsVersionRangeInOrderAtLeast(self):
        if False:
            return 10
        '\n        Passing out of order TLS versions to C{raiseMinimumTo} and\n        C{lowerMaximumSecurityTo} will cause it to raise an exception.\n        '
        with self.assertRaises(ValueError) as e:
            sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, raiseMinimumTo=sslverify.TLSVersion.TLSv1_0, lowerMaximumSecurityTo=sslverify.TLSVersion.SSLv3)
        self.assertEqual(e.exception.args, ('raiseMinimumTo needs to be lower than lowerMaximumSecurityTo',))

    def test_tlsProtocolsreduceToMaxWithoutMin(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{lowerMaximumSecurityTo} but no C{raiseMinimumTo} or\n        C{insecurelyLowerMinimumTo} set, and C{lowerMaximumSecurityTo} is\n        below the minimum default, the minimum will be made the new maximum.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, lowerMaximumSecurityTo=sslverify.TLSVersion.SSLv3)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1 | SSL.OP_NO_TLSv1_2 | opts._OP_NO_TLSv1_3
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsSSLv3Only(self):
        if False:
            print('Hello World!')
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{insecurelyLowerMinimumTo} and C{lowerMaximumSecurityTo} set to\n        SSLv3, it will exclude all others.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.SSLv3, lowerMaximumSecurityTo=sslverify.TLSVersion.SSLv3)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1 | SSL.OP_NO_TLSv1_2 | opts._OP_NO_TLSv1_3
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsTLSv1Point0Only(self):
        if False:
            i = 10
            return i + 15
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{insecurelyLowerMinimumTo} and C{lowerMaximumSecurityTo} set to v1.0,\n        it will exclude all others.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_0, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_0)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1_1 | SSL.OP_NO_TLSv1_2 | opts._OP_NO_TLSv1_3
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsTLSv1Point1Only(self):
        if False:
            while True:
                i = 10
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{insecurelyLowerMinimumTo} and C{lowerMaximumSecurityTo} set to v1.1,\n        it will exclude all others.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_1, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_1)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_2 | opts._OP_NO_TLSv1_3
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsTLSv1Point2Only(self):
        if False:
            print('Hello World!')
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{insecurelyLowerMinimumTo} and C{lowerMaximumSecurityTo} set to v1.2,\n        it will exclude all others.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_2, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_2)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1 | opts._OP_NO_TLSv1_3
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsAllModernTLS(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{insecurelyLowerMinimumTo} set to TLSv1.0 and\n        C{lowerMaximumSecurityTo} to TLSv1.2, it will exclude both SSLs and\n        the (unreleased) TLSv1.3.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_0, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_2)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | opts._OP_NO_TLSv1_3
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsAtLeastAllSecureTLS(self):
        if False:
            while True:
                i = 10
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{raiseMinimumTo} set to TLSv1.2, it will ignore all TLSs below\n        1.2 and SSL.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, raiseMinimumTo=sslverify.TLSVersion.TLSv1_2)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1
        self.assertEqual(options, ctx._options & options)

    def test_tlsProtocolsAtLeastWillAcceptHigherDefault(self):
        if False:
            while True:
                i = 10
        "\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{raiseMinimumTo} set to a value lower than Twisted's default will\n        cause it to use the more secure default.\n        "
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, raiseMinimumTo=sslverify.TLSVersion.SSLv3)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1
        self.assertEqual(options, ctx._options & options)
        self.assertEqual(opts._defaultMinimumTLSVersion, sslverify.TLSVersion.TLSv1_2)

    def test_tlsProtocolsAllSecureTLS(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When calling L{sslverify.OpenSSLCertificateOptions} with\n        C{insecurelyLowerMinimumTo} set to TLSv1.2, it will ignore all TLSs below\n        1.2 and SSL.\n        '
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, insecurelyLowerMinimumTo=sslverify.TLSVersion.TLSv1_2)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        options = SSL.OP_NO_SSLv2 | SSL.OP_NO_COMPRESSION | SSL.OP_CIPHER_SERVER_PREFERENCE | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1
        self.assertEqual(options, ctx._options & options)

    def test_dhParams(self):
        if False:
            print('Hello World!')
        '\n        If C{dhParams} is set, they are loaded into each new context.\n        '

        class FakeDiffieHellmanParameters:
            _dhFile = FilePath(b'dh.params')
        dhParams = FakeDiffieHellmanParameters()
        opts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, dhParameters=dhParams)
        opts._contextFactory = FakeContext
        ctx = opts.getContext()
        self.assertEqual(FakeDiffieHellmanParameters._dhFile.path, ctx._dhFilename)

    def test_abbreviatingDistinguishedNames(self):
        if False:
            return 10
        '\n        Check that abbreviations used in certificates correctly map to\n        complete names.\n        '
        self.assertEqual(sslverify.DN(CN=b'a', OU=b'hello'), sslverify.DistinguishedName(commonName=b'a', organizationalUnitName=b'hello'))
        self.assertNotEqual(sslverify.DN(CN=b'a', OU=b'hello'), sslverify.DN(CN=b'a', OU=b'hello', emailAddress=b'xxx'))
        dn = sslverify.DN(CN=b'abcdefg')
        self.assertRaises(AttributeError, setattr, dn, 'Cn', b'x')
        self.assertEqual(dn.CN, dn.commonName)
        dn.CN = b'bcdefga'
        self.assertEqual(dn.CN, dn.commonName)

    def testInspectDistinguishedName(self):
        if False:
            i = 10
            return i + 15
        n = sslverify.DN(commonName=b'common name', organizationName=b'organization name', organizationalUnitName=b'organizational unit name', localityName=b'locality name', stateOrProvinceName=b'state or province name', countryName=b'country name', emailAddress=b'email address')
        s = n.inspect()
        for k in ['common name', 'organization name', 'organizational unit name', 'locality name', 'state or province name', 'country name', 'email address']:
            self.assertIn(k, s, f'{k!r} was not in inspect output.')
            self.assertIn(k.title(), s, f'{k!r} was not in inspect output.')

    def testInspectDistinguishedNameWithoutAllFields(self):
        if False:
            return 10
        n = sslverify.DN(localityName=b'locality name')
        s = n.inspect()
        for k in ['common name', 'organization name', 'organizational unit name', 'state or province name', 'country name', 'email address']:
            self.assertNotIn(k, s, f'{k!r} was in inspect output.')
            self.assertNotIn(k.title(), s, f'{k!r} was in inspect output.')
        self.assertIn('locality name', s)
        self.assertIn('Locality Name', s)

    def test_inspectCertificate(self):
        if False:
            print('Hello World!')
        '\n        Test that the C{inspect} method of L{sslverify.Certificate} returns\n        a human-readable string containing some basic information about the\n        certificate.\n        '
        c = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
        pk = c.getPublicKey()
        keyHash = pk.keyHash()
        self.assertEqual(c.inspect().split('\n'), ['Certificate For Subject:', '               Common Name: example.twistedmatrix.com', '              Country Name: US', '             Email Address: nobody@twistedmatrix.com', '             Locality Name: Boston', '         Organization Name: Twisted Matrix Labs', '  Organizational Unit Name: Security', '    State Or Province Name: Massachusetts', '', 'Issuer:', '               Common Name: example.twistedmatrix.com', '              Country Name: US', '             Email Address: nobody@twistedmatrix.com', '             Locality Name: Boston', '         Organization Name: Twisted Matrix Labs', '  Organizational Unit Name: Security', '    State Or Province Name: Massachusetts', '', 'Serial Number: 12345', 'Digest: C4:96:11:00:30:C3:EC:EE:A3:55:AA:ED:8C:84:85:18', 'Public Key with Hash: ' + keyHash])

    def test_publicKeyMatching(self):
        if False:
            return 10
        '\n        L{PublicKey.matches} returns L{True} for keys from certificates with\n        the same key, and L{False} for keys from certificates with different\n        keys.\n        '
        hostA = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
        hostB = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
        peerA = sslverify.Certificate.loadPEM(A_PEER_CERTIFICATE_PEM)
        self.assertTrue(hostA.getPublicKey().matches(hostB.getPublicKey()))
        self.assertFalse(peerA.getPublicKey().matches(hostA.getPublicKey()))

    def test_enablingAndDisablingSessions(self):
        if False:
            return 10
        '\n        The enableSessions argument sets the session cache mode; it defaults to\n        False (at least until https://twistedmatrix.com/trac/ticket/9764 can be\n        resolved).\n        '
        options = sslverify.OpenSSLCertificateOptions()
        self.assertEqual(options.enableSessions, False)
        ctx = options.getContext()
        self.assertEqual(ctx.get_session_cache_mode(), SSL.SESS_CACHE_OFF)
        options = sslverify.OpenSSLCertificateOptions(enableSessions=True)
        self.assertEqual(options.enableSessions, True)
        ctx = options.getContext()
        self.assertEqual(ctx.get_session_cache_mode(), SSL.SESS_CACHE_SERVER)

    def test_certificateOptionsSerialization(self):
        if False:
            i = 10
            return i + 15
        '\n        Test that __setstate__(__getstate__()) round-trips properly.\n        '
        firstOpts = sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, method=SSL.SSLv23_METHOD, verify=True, caCerts=[self.sCert], verifyDepth=2, requireCertificate=False, verifyOnce=False, enableSingleUseKeys=False, enableSessions=False, fixBrokenPeers=True, enableSessionTickets=True)
        context = firstOpts.getContext()
        self.assertIs(context, firstOpts._context)
        self.assertIsNotNone(context)
        state = firstOpts.__getstate__()
        self.assertNotIn('_context', state)
        opts = sslverify.OpenSSLCertificateOptions()
        opts.__setstate__(state)
        self.assertEqual(opts.privateKey, self.sKey)
        self.assertEqual(opts.certificate, self.sCert)
        self.assertEqual(opts.method, SSL.SSLv23_METHOD)
        self.assertTrue(opts.verify)
        self.assertEqual(opts.caCerts, [self.sCert])
        self.assertEqual(opts.verifyDepth, 2)
        self.assertFalse(opts.requireCertificate)
        self.assertFalse(opts.verifyOnce)
        self.assertFalse(opts.enableSingleUseKeys)
        self.assertFalse(opts.enableSessions)
        self.assertTrue(opts.fixBrokenPeers)
        self.assertTrue(opts.enableSessionTickets)
    test_certificateOptionsSerialization.suppress = [util.suppress(category=DeprecationWarning, message='twisted\\.internet\\._sslverify\\.*__[gs]etstate__')]

    def test_certificateOptionsSessionTickets(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Enabling session tickets should not set the OP_NO_TICKET option.\n        '
        opts = sslverify.OpenSSLCertificateOptions(enableSessionTickets=True)
        ctx = opts.getContext()
        self.assertEqual(0, ctx.set_options(0) & 16384)

    def test_certificateOptionsSessionTicketsDisabled(self):
        if False:
            i = 10
            return i + 15
        '\n        Enabling session tickets should set the OP_NO_TICKET option.\n        '
        opts = sslverify.OpenSSLCertificateOptions(enableSessionTickets=False)
        ctx = opts.getContext()
        self.assertEqual(16384, ctx.set_options(0) & 16384)

    def test_allowedAnonymousClientConnection(self):
        if False:
            while True:
                i = 10
        "\n        Check that anonymous connections are allowed when certificates aren't\n        required on the server.\n        "
        onData = defer.Deferred()
        self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, requireCertificate=False), sslverify.OpenSSLCertificateOptions(requireCertificate=False), onData=onData)
        return onData.addCallback(lambda result: self.assertEqual(result, WritingProtocol.byte))

    def test_refusedAnonymousClientConnection(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that anonymous connections are refused when certificates are\n        required on the server.\n        '
        onServerLost = defer.Deferred()
        onClientLost = defer.Deferred()
        self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=True, caCerts=[self.sCert], requireCertificate=True), sslverify.OpenSSLCertificateOptions(requireCertificate=False), onServerLost=onServerLost, onClientLost=onClientLost)
        d = defer.DeferredList([onClientLost, onServerLost], consumeErrors=True)

        def afterLost(result):
            if False:
                i = 10
                return i + 15
            ((cSuccess, cResult), (sSuccess, sResult)) = result
            self.assertFalse(cSuccess)
            self.assertFalse(sSuccess)
            self.assertIsInstance(cResult.value, (SSL.Error, ConnectionLost))
            self.assertIsInstance(sResult.value, SSL.Error)
        return d.addCallback(afterLost)

    def test_failedCertificateVerification(self):
        if False:
            while True:
                i = 10
        '\n        Check that connecting with a certificate not accepted by the server CA\n        fails.\n        '
        onServerLost = defer.Deferred()
        onClientLost = defer.Deferred()
        self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=False, requireCertificate=False), sslverify.OpenSSLCertificateOptions(verify=True, requireCertificate=False, caCerts=[self.cCert]), onServerLost=onServerLost, onClientLost=onClientLost)
        d = defer.DeferredList([onClientLost, onServerLost], consumeErrors=True)

        def afterLost(result):
            if False:
                return 10
            ((cSuccess, cResult), (sSuccess, sResult)) = result
            self.assertFalse(cSuccess)
            self.assertFalse(sSuccess)
        return d.addCallback(afterLost)

    def test_successfulCertificateVerification(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test a successful connection with client certificate validation on\n        server side.\n        '
        onData = defer.Deferred()
        self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=False, requireCertificate=False), sslverify.OpenSSLCertificateOptions(verify=True, requireCertificate=True, caCerts=[self.sCert]), onData=onData)
        return onData.addCallback(lambda result: self.assertEqual(result, WritingProtocol.byte))

    def test_successfulSymmetricSelfSignedCertificateVerification(self):
        if False:
            while True:
                i = 10
        '\n        Test a successful connection with validation on both server and client\n        sides.\n        '
        onData = defer.Deferred()
        self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, verify=True, requireCertificate=True, caCerts=[self.cCert]), sslverify.OpenSSLCertificateOptions(privateKey=self.cKey, certificate=self.cCert, verify=True, requireCertificate=True, caCerts=[self.sCert]), onData=onData)
        return onData.addCallback(lambda result: self.assertEqual(result, WritingProtocol.byte))

    def test_verification(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check certificates verification building custom certificates data.\n        '
        clientDN = sslverify.DistinguishedName(commonName='client')
        clientKey = sslverify.KeyPair.generate()
        clientCertReq = clientKey.certificateRequest(clientDN)
        serverDN = sslverify.DistinguishedName(commonName='server')
        serverKey = sslverify.KeyPair.generate()
        serverCertReq = serverKey.certificateRequest(serverDN)
        clientSelfCertReq = clientKey.certificateRequest(clientDN)
        clientSelfCertData = clientKey.signCertificateRequest(clientDN, clientSelfCertReq, lambda dn: True, 132)
        clientSelfCert = clientKey.newCertificate(clientSelfCertData)
        serverSelfCertReq = serverKey.certificateRequest(serverDN)
        serverSelfCertData = serverKey.signCertificateRequest(serverDN, serverSelfCertReq, lambda dn: True, 516)
        serverSelfCert = serverKey.newCertificate(serverSelfCertData)
        clientCertData = serverKey.signCertificateRequest(serverDN, clientCertReq, lambda dn: True, 7)
        clientCert = clientKey.newCertificate(clientCertData)
        serverCertData = clientKey.signCertificateRequest(clientDN, serverCertReq, lambda dn: True, 42)
        serverCert = serverKey.newCertificate(serverCertData)
        onData = defer.Deferred()
        serverOpts = serverCert.options(serverSelfCert)
        clientOpts = clientCert.options(clientSelfCert)
        self.loopback(serverOpts, clientOpts, onData=onData)
        return onData.addCallback(lambda result: self.assertEqual(result, WritingProtocol.byte))

class OpenSSLOptionsECDHIntegrationTests(OpenSSLOptionsTestsMixin, TestCase):
    """
    ECDH-related integration tests for L{OpenSSLOptions}.
    """

    def test_ellipticCurveDiffieHellman(self):
        if False:
            while True:
                i = 10
        '\n        Connections use ECDH when OpenSSL supports it.\n        '
        if not get_elliptic_curves():
            raise SkipTest('OpenSSL does not support ECDH.')
        onData = defer.Deferred()
        self.loopback(sslverify.OpenSSLCertificateOptions(privateKey=self.sKey, certificate=self.sCert, requireCertificate=False, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_3), sslverify.OpenSSLCertificateOptions(requireCertificate=False, lowerMaximumSecurityTo=sslverify.TLSVersion.TLSv1_3), onData=onData)

        @onData.addCallback
        def assertECDH(_):
            if False:
                return 10
            self.assertEqual(len(self.clientConn.factory.protocols), 1)
            [clientProtocol] = self.clientConn.factory.protocols
            cipher = clientProtocol.getHandle().get_cipher_name()
            self.assertIn('ECDH', cipher)
        return onData

class DeprecationTests(SynchronousTestCase):
    """
    Tests for deprecation of L{sslverify.OpenSSLCertificateOptions}'s support
    of the pickle protocol.
    """
    if skipSSL:
        skip = skipSSL

    def test_getstateDeprecation(self):
        if False:
            return 10
        '\n        L{sslverify.OpenSSLCertificateOptions.__getstate__} is deprecated.\n        '
        self.callDeprecated((Version('Twisted', 15, 0, 0), 'a real persistence system'), sslverify.OpenSSLCertificateOptions().__getstate__)

    def test_setstateDeprecation(self):
        if False:
            while True:
                i = 10
        '\n        L{sslverify.OpenSSLCertificateOptions.__setstate__} is deprecated.\n        '
        self.callDeprecated((Version('Twisted', 15, 0, 0), 'a real persistence system'), sslverify.OpenSSLCertificateOptions().__setstate__, {})

class TrustRootTests(TestCase):
    """
    Tests for L{sslverify.OpenSSLCertificateOptions}' C{trustRoot} argument,
    L{sslverify.platformTrust}, and their interactions.
    """
    if skipSSL:
        skip = skipSSL

    def setUp(self):
        if False:
            while True:
                i = 10
        '\n        Patch L{sslverify._ChooseDiffieHellmanEllipticCurve}.\n        '
        self.patch(sslverify, '_ChooseDiffieHellmanEllipticCurve', FakeChooseDiffieHellmanEllipticCurve)

    def test_caCertsPlatformDefaults(self):
        if False:
            i = 10
            return i + 15
        '\n        Specifying a C{trustRoot} of L{sslverify.OpenSSLDefaultPaths} when\n        initializing L{sslverify.OpenSSLCertificateOptions} loads the\n        platform-provided trusted certificates via C{set_default_verify_paths}.\n        '
        opts = sslverify.OpenSSLCertificateOptions(trustRoot=sslverify.OpenSSLDefaultPaths())
        fc = FakeContext(SSL.TLSv1_METHOD)
        opts._contextFactory = lambda method: fc
        opts.getContext()
        self.assertTrue(fc._defaultVerifyPathsSet)

    def test_trustRootPlatformRejectsUntrustedCA(self):
        if False:
            print('Hello World!')
        '\n        Specifying a C{trustRoot} of L{platformTrust} when initializing\n        L{sslverify.OpenSSLCertificateOptions} causes certificates issued by a\n        newly created CA to be rejected by an SSL connection using these\n        options.\n\n        Note that this test should I{always} pass, even on platforms where the\n        CA certificates are not installed, as long as L{platformTrust} rejects\n        completely invalid / unknown root CA certificates.  This is simply a\n        smoke test to make sure that verification is happening at all.\n        '
        (caSelfCert, serverCert) = certificatesForAuthorityAndServer()
        chainedCert = pathContainingDumpOf(self, serverCert, caSelfCert)
        privateKey = pathContainingDumpOf(self, serverCert.privateKey)
        (sProto, cProto, sWrapped, cWrapped, pump) = loopbackTLSConnection(trustRoot=platformTrust(), privateKeyFile=privateKey, chainedCertFile=chainedCert)
        self.assertEqual(cWrapped.data, b'')
        self.assertEqual(cWrapped.lostReason.type, SSL.Error)
        err = cWrapped.lostReason.value
        self.assertEqual(err.args[0][0][2], 'tlsv1 alert unknown ca')

    def test_trustRootSpecificCertificate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifying a L{Certificate} object for L{trustRoot} will result in that\n        certificate being the only trust root for a client.\n        '
        (caCert, serverCert) = certificatesForAuthorityAndServer()
        (otherCa, otherServer) = certificatesForAuthorityAndServer()
        (sProto, cProto, sWrapped, cWrapped, pump) = loopbackTLSConnection(trustRoot=caCert, privateKeyFile=pathContainingDumpOf(self, serverCert.privateKey), chainedCertFile=pathContainingDumpOf(self, serverCert))
        pump.flush()
        self.assertIsNone(cWrapped.lostReason)
        self.assertEqual(cWrapped.data, sWrapped.greeting)

class ServiceIdentityTests(SynchronousTestCase):
    """
    Tests for the verification of the peer's service's identity via the
    C{hostname} argument to L{sslverify.OpenSSLCertificateOptions}.
    """
    if skipSSL:
        skip = skipSSL

    def serviceIdentitySetup(self, clientHostname, serverHostname, serverContextSetup=lambda ctx: None, validCertificate=True, clientPresentsCertificate=False, validClientCertificate=True, serverVerifies=False, buggyInfoCallback=False, fakePlatformTrust=False, useDefaultTrust=False):
        if False:
            for i in range(10):
                print('nop')
        "\n        Connect a server and a client.\n\n        @param clientHostname: The I{client's idea} of the server's hostname;\n            passed as the C{hostname} to the\n            L{sslverify.OpenSSLCertificateOptions} instance.\n        @type clientHostname: L{unicode}\n\n        @param serverHostname: The I{server's own idea} of the server's\n            hostname; present in the certificate presented by the server.\n        @type serverHostname: L{unicode}\n\n        @param serverContextSetup: a 1-argument callable invoked with the\n            L{OpenSSL.SSL.Context} after it's produced.\n        @type serverContextSetup: L{callable} taking L{OpenSSL.SSL.Context}\n            returning L{None}.\n\n        @param validCertificate: Is the server's certificate valid?  L{True} if\n            so, L{False} otherwise.\n        @type validCertificate: L{bool}\n\n        @param clientPresentsCertificate: Should the client present a\n            certificate to the server?  Defaults to 'no'.\n        @type clientPresentsCertificate: L{bool}\n\n        @param validClientCertificate: If the client presents a certificate,\n            should it actually be a valid one, i.e. signed by the same CA that\n            the server is checking?  Defaults to 'yes'.\n        @type validClientCertificate: L{bool}\n\n        @param serverVerifies: Should the server verify the client's\n            certificate?  Defaults to 'no'.\n        @type serverVerifies: L{bool}\n\n        @param buggyInfoCallback: Should we patch the implementation so that\n            the C{info_callback} passed to OpenSSL to have a bug and raise an\n            exception (L{ZeroDivisionError})?  Defaults to 'no'.\n        @type buggyInfoCallback: L{bool}\n\n        @param fakePlatformTrust: Should we fake the platformTrust to be the\n            same as our fake server certificate authority, so that we can test\n            it's being used?  Defaults to 'no' and we just pass platform trust.\n        @type fakePlatformTrust: L{bool}\n\n        @param useDefaultTrust: Should we avoid passing the C{trustRoot} to\n            L{ssl.optionsForClientTLS}?  Defaults to 'no'.\n        @type useDefaultTrust: L{bool}\n\n        @return: the client TLS protocol, the client wrapped protocol,\n            the server TLS protocol, the server wrapped protocol and\n            an L{IOPump} which, when its C{pump} and C{flush} methods are\n            called, will move data between the created client and server\n            protocol instances\n        @rtype: 5-L{tuple} of 4 L{IProtocol}s and L{IOPump}\n        "
        (serverCA, serverCert) = certificatesForAuthorityAndServer(serverHostname)
        other = {}
        passClientCert = None
        (clientCA, clientCert) = certificatesForAuthorityAndServer('client')
        if serverVerifies:
            other.update(trustRoot=clientCA)
        if clientPresentsCertificate:
            if validClientCertificate:
                passClientCert = clientCert
            else:
                (bogusCA, bogus) = certificatesForAuthorityAndServer('client')
                passClientCert = bogus
        serverOpts = sslverify.OpenSSLCertificateOptions(privateKey=serverCert.privateKey.original, certificate=serverCert.original, **other)
        serverContextSetup(serverOpts.getContext())
        if not validCertificate:
            (serverCA, otherServer) = certificatesForAuthorityAndServer(serverHostname)
        if buggyInfoCallback:

            def broken(*a, **k):
                if False:
                    while True:
                        i = 10
                '\n                Raise an exception.\n\n                @param a: Arguments for an C{info_callback}\n\n                @param k: Keyword arguments for an C{info_callback}\n                '
                1 / 0
            self.patch(sslverify.ClientTLSOptions, '_identityVerifyingInfoCallback', broken)
        signature = {'hostname': clientHostname}
        if passClientCert:
            signature.update(clientCertificate=passClientCert)
        if not useDefaultTrust:
            signature.update(trustRoot=serverCA)
        if fakePlatformTrust:
            self.patch(sslverify, 'platformTrust', lambda : serverCA)
        clientOpts = sslverify.optionsForClientTLS(**signature)

        class GreetingServer(protocol.Protocol):
            greeting = b'greetings!'
            lostReason = None
            data = b''

            def connectionMade(self):
                if False:
                    i = 10
                    return i + 15
                self.transport.write(self.greeting)

            def dataReceived(self, data):
                if False:
                    print('Hello World!')
                self.data += data

            def connectionLost(self, reason):
                if False:
                    print('Hello World!')
                self.lostReason = reason

        class GreetingClient(protocol.Protocol):
            greeting = b'cheerio!'
            data = b''
            lostReason = None

            def connectionMade(self):
                if False:
                    print('Hello World!')
                self.transport.write(self.greeting)

            def dataReceived(self, data):
                if False:
                    return 10
                self.data += data

            def connectionLost(self, reason):
                if False:
                    while True:
                        i = 10
                self.lostReason = reason
        serverWrappedProto = GreetingServer()
        clientWrappedProto = GreetingClient()
        clientFactory = protocol.Factory()
        clientFactory.protocol = lambda : clientWrappedProto
        serverFactory = protocol.Factory()
        serverFactory.protocol = lambda : serverWrappedProto
        self.serverOpts = serverOpts
        self.clientOpts = clientOpts
        clock = Clock()
        clientTLSFactory = TLSMemoryBIOFactory(clientOpts, isClient=True, wrappedFactory=clientFactory, clock=clock)
        serverTLSFactory = TLSMemoryBIOFactory(serverOpts, isClient=False, wrappedFactory=serverFactory, clock=clock)
        (cProto, sProto, pump) = connectedServerAndClient(lambda : serverTLSFactory.buildProtocol(None), lambda : clientTLSFactory.buildProtocol(None), clock=clock)
        pump.flush()
        return (cProto, sProto, clientWrappedProto, serverWrappedProto, pump)

    def test_invalidHostname(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When a certificate containing an invalid hostname is received from the\n        server, the connection is immediately dropped.\n        '
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('wrong-host.example.com', 'correct-host.example.com')
        self.assertEqual(cWrapped.data, b'')
        self.assertEqual(sWrapped.data, b'')
        cErr = cWrapped.lostReason.value
        sErr = sWrapped.lostReason.value
        self.assertIsInstance(cErr, VerificationError)
        self.assertIsInstance(sErr, ConnectionClosed)

    def test_validHostname(self):
        if False:
            return 10
        '\n        Whenever a valid certificate containing a valid hostname is received,\n        connection proceeds normally.\n        '
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('valid.example.com', 'valid.example.com')
        self.assertEqual(cWrapped.data, b'greetings!')
        cErr = cWrapped.lostReason
        sErr = sWrapped.lostReason
        self.assertIsNone(cErr)
        self.assertIsNone(sErr)

    def test_validHostnameInvalidCertificate(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When an invalid certificate containing a perfectly valid hostname is\n        received, the connection is aborted with an OpenSSL error.\n        '
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', validCertificate=False)
        self.assertEqual(cWrapped.data, b'')
        self.assertEqual(sWrapped.data, b'')
        cErr = cWrapped.lostReason.value
        sErr = sWrapped.lostReason.value
        self.assertIsInstance(cErr, SSL.Error)
        self.assertIsInstance(sErr, SSL.Error)

    def test_realCAsBetterNotSignOurBogusTestCerts(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If we use the default trust from the platform, our dinky certificate\n        should I{really} fail.\n        '
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', validCertificate=False, useDefaultTrust=True)
        self.assertEqual(cWrapped.data, b'')
        self.assertEqual(sWrapped.data, b'')
        cErr = cWrapped.lostReason.value
        sErr = sWrapped.lostReason.value
        self.assertIsInstance(cErr, SSL.Error)
        self.assertIsInstance(sErr, SSL.Error)

    def test_butIfTheyDidItWouldWork(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{ssl.optionsForClientTLS} should be using L{ssl.platformTrust} by\n        default, so if we fake that out then it should trust ourselves again.\n        '
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', useDefaultTrust=True, fakePlatformTrust=True)
        self.assertEqual(cWrapped.data, b'greetings!')
        cErr = cWrapped.lostReason
        sErr = sWrapped.lostReason
        self.assertIsNone(cErr)
        self.assertIsNone(sErr)

    def test_clientPresentsCertificate(self):
        if False:
            i = 10
            return i + 15
        '\n        When the server verifies and the client presents a valid certificate\n        for that verification by passing it to\n        L{sslverify.optionsForClientTLS}, communication proceeds.\n        '
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', validCertificate=True, serverVerifies=True, clientPresentsCertificate=True)
        self.assertEqual(cWrapped.data, b'greetings!')
        cErr = cWrapped.lostReason
        sErr = sWrapped.lostReason
        self.assertIsNone(cErr)
        self.assertIsNone(sErr)

    def test_clientPresentsBadCertificate(self):
        if False:
            while True:
                i = 10
        '\n        When the server verifies and the client presents an invalid certificate\n        for that verification by passing it to\n        L{sslverify.optionsForClientTLS}, the connection cannot be established\n        with an SSL error.\n        '
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', validCertificate=True, serverVerifies=True, validClientCertificate=False, clientPresentsCertificate=True)
        self.assertEqual(cWrapped.data, b'')
        cErr = cWrapped.lostReason.value
        sErr = sWrapped.lostReason.value
        self.assertIsInstance(cErr, SSL.Error)
        self.assertIsInstance(sErr, SSL.Error)

    @skipIf(skipSNI, skipSNI)
    def test_hostnameIsIndicated(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Specifying the C{hostname} argument to L{CertificateOptions} also sets\n        the U{Server Name Extension\n        <https://en.wikipedia.org/wiki/Server_Name_Indication>} TLS indication\n        field to the correct value.\n        '
        names = []

        def setupServerContext(ctx):
            if False:
                for i in range(10):
                    print('nop')

            def servername_received(conn):
                if False:
                    for i in range(10):
                        print('nop')
                names.append(conn.get_servername().decode('ascii'))
            ctx.set_tlsext_servername_callback(servername_received)
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('valid.example.com', 'valid.example.com', setupServerContext)
        self.assertEqual(names, ['valid.example.com'])

    @skipIf(skipSNI, skipSNI)
    def test_hostnameEncoding(self):
        if False:
            i = 10
            return i + 15
        '\n        Hostnames are encoded as IDNA.\n        '
        names = []
        hello = 'hállo.example.com'

        def setupServerContext(ctx):
            if False:
                i = 10
                return i + 15

            def servername_received(conn):
                if False:
                    while True:
                        i = 10
                serverIDNA = _idnaText(conn.get_servername())
                names.append(serverIDNA)
            ctx.set_tlsext_servername_callback(servername_received)
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup(hello, hello, setupServerContext)
        self.assertEqual(names, [hello])
        self.assertEqual(cWrapped.data, b'greetings!')
        cErr = cWrapped.lostReason
        sErr = sWrapped.lostReason
        self.assertIsNone(cErr)
        self.assertIsNone(sErr)

    def test_fallback(self):
        if False:
            while True:
                i = 10
        "\n        L{sslverify.simpleVerifyHostname} checks string equality on the\n        commonName of a connection's certificate's subject, doing nothing if it\n        matches and raising L{VerificationError} if it doesn't.\n        "
        name = 'something.example.com'

        class Connection:

            def get_peer_certificate(self):
                if False:
                    for i in range(10):
                        print('nop')
                '\n                Fake of L{OpenSSL.SSL.Connection.get_peer_certificate}.\n\n                @return: A certificate with a known common name.\n                @rtype: L{OpenSSL.crypto.X509}\n                '
                cert = X509()
                cert.get_subject().commonName = name
                return cert
        conn = Connection()
        self.assertIs(sslverify.simpleVerifyHostname(conn, 'something.example.com'), None)
        self.assertRaises(sslverify.SimpleVerificationError, sslverify.simpleVerifyHostname, conn, 'nonsense')

    def test_surpriseFromInfoCallback(self):
        if False:
            while True:
                i = 10
        "\n        pyOpenSSL isn't always so great about reporting errors.  If one occurs\n        in the verification info callback, it should be logged and the\n        connection should be shut down (if possible, anyway; the app_data could\n        be clobbered but there's no point testing for that).\n        "
        (cProto, sProto, cWrapped, sWrapped, pump) = self.serviceIdentitySetup('correct-host.example.com', 'correct-host.example.com', buggyInfoCallback=True)
        self.assertEqual(cWrapped.data, b'')
        self.assertEqual(sWrapped.data, b'')
        cErr = cWrapped.lostReason.value
        sErr = sWrapped.lostReason.value
        self.assertIsInstance(cErr, ZeroDivisionError)
        self.assertIsInstance(sErr, (ConnectionClosed, SSL.Error))
        errors = self.flushLoggedErrors(ZeroDivisionError)
        self.assertTrue(errors)

def negotiateProtocol(serverProtocols, clientProtocols, clientOptions=None):
    if False:
        while True:
            i = 10
    '\n    Create the TLS connection and negotiate a next protocol.\n\n    @param serverProtocols: The protocols the server is willing to negotiate.\n    @param clientProtocols: The protocols the client is willing to negotiate.\n    @param clientOptions: The type of C{OpenSSLCertificateOptions} class to\n        use for the client. Defaults to C{OpenSSLCertificateOptions}.\n    @return: A L{tuple} of the negotiated protocol and the reason the\n        connection was lost.\n    '
    (caCertificate, serverCertificate) = certificatesForAuthorityAndServer()
    trustRoot = sslverify.OpenSSLCertificateAuthorities([caCertificate.original])
    (sProto, cProto, sWrapped, cWrapped, pump) = loopbackTLSConnectionInMemory(trustRoot=trustRoot, privateKey=serverCertificate.privateKey.original, serverCertificate=serverCertificate.original, clientProtocols=clientProtocols, serverProtocols=serverProtocols, clientOptions=clientOptions)
    pump.flush()
    return (cProto.negotiatedProtocol, cWrapped.lostReason)

class NPNOrALPNTests(TestCase):
    """
    NPN and ALPN protocol selection.

    These tests only run on platforms that have a PyOpenSSL version >= 0.15,
    and OpenSSL version 1.0.1 or later.
    """
    if skipSSL:
        skip = skipSSL
    elif skipNPN:
        skip = skipNPN

    def test_nextProtocolMechanismsNPNIsSupported(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When at least NPN is available on the platform, NPN is in the set of\n        supported negotiation protocols.\n        '
        supportedProtocols = sslverify.protocolNegotiationMechanisms()
        self.assertTrue(sslverify.ProtocolNegotiationSupport.NPN in supportedProtocols)

    def test_NPNAndALPNSuccess(self):
        if False:
            return 10
        '\n        When both ALPN and NPN are used, and both the client and server have\n        overlapping protocol choices, a protocol is successfully negotiated.\n        Further, the negotiated protocol is the first one in the list.\n        '
        protocols = [b'h2', b'http/1.1']
        (negotiatedProtocol, lostReason) = negotiateProtocol(clientProtocols=protocols, serverProtocols=protocols)
        self.assertEqual(negotiatedProtocol, b'h2')
        self.assertIsNone(lostReason)

    def test_NPNAndALPNDifferent(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Client and server have different protocol lists: only the common\n        element is chosen.\n        '
        serverProtocols = [b'h2', b'http/1.1', b'spdy/2']
        clientProtocols = [b'spdy/3', b'http/1.1']
        (negotiatedProtocol, lostReason) = negotiateProtocol(clientProtocols=clientProtocols, serverProtocols=serverProtocols)
        self.assertEqual(negotiatedProtocol, b'http/1.1')
        self.assertIsNone(lostReason)

    def test_NPNAndALPNNoAdvertise(self):
        if False:
            print('Hello World!')
        '\n        When one peer does not advertise any protocols, the connection is set\n        up with no next protocol.\n        '
        protocols = [b'h2', b'http/1.1']
        (negotiatedProtocol, lostReason) = negotiateProtocol(clientProtocols=protocols, serverProtocols=[])
        self.assertIsNone(negotiatedProtocol)
        self.assertIsNone(lostReason)

    def test_NPNAndALPNNoOverlap(self):
        if False:
            i = 10
            return i + 15
        '\n        When the client and server have no overlap of protocols, the connection\n        fails.\n        '
        clientProtocols = [b'h2', b'http/1.1']
        serverProtocols = [b'spdy/3']
        (negotiatedProtocol, lostReason) = negotiateProtocol(serverProtocols=clientProtocols, clientProtocols=serverProtocols)
        self.assertIsNone(negotiatedProtocol)
        self.assertEqual(lostReason.type, SSL.Error)

class ALPNTests(TestCase):
    """
    ALPN protocol selection.

    These tests only run on platforms that have a PyOpenSSL version >= 0.15,
    and OpenSSL version 1.0.2 or later.

    This covers only the ALPN specific logic, as any platform that has ALPN
    will also have NPN and so will run the NPNAndALPNTest suite as well.
    """
    if skipSSL:
        skip = skipSSL
    elif skipALPN:
        skip = skipALPN

    def test_nextProtocolMechanismsALPNIsSupported(self):
        if False:
            while True:
                i = 10
        '\n        When ALPN is available on a platform, protocolNegotiationMechanisms\n        includes ALPN in the suported protocols.\n        '
        supportedProtocols = sslverify.protocolNegotiationMechanisms()
        self.assertTrue(sslverify.ProtocolNegotiationSupport.ALPN in supportedProtocols)

class NPNAndALPNAbsentTests(TestCase):
    """
    NPN/ALPN operations fail on platforms that do not support them.

    These tests only run on platforms that have a PyOpenSSL version < 0.15,
    an OpenSSL version earlier than 1.0.1, or an OpenSSL/cryptography built
    without NPN support.
    """
    if skipSSL:
        skip = skipSSL
    elif not skipNPN or not skipALPN:
        skip = 'NPN and/or ALPN is present on this platform'

    def test_nextProtocolMechanismsNoNegotiationSupported(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        When neither NPN or ALPN are available on a platform, there are no\n        supported negotiation protocols.\n        '
        supportedProtocols = sslverify.protocolNegotiationMechanisms()
        self.assertFalse(supportedProtocols)

    def test_NPNAndALPNNotImplemented(self):
        if False:
            return 10
        '\n        A NotImplementedError is raised when using acceptableProtocols on a\n        platform that does not support either NPN or ALPN.\n        '
        protocols = [b'h2', b'http/1.1']
        self.assertRaises(NotImplementedError, negotiateProtocol, serverProtocols=protocols, clientProtocols=protocols)

    def test_NegotiatedProtocolReturnsNone(self):
        if False:
            print('Hello World!')
        "\n        negotiatedProtocol return L{None} even when NPN/ALPN aren't supported.\n        This works because, as neither are supported, negotiation isn't even\n        attempted.\n        "
        serverProtocols = None
        clientProtocols = None
        (negotiatedProtocol, lostReason) = negotiateProtocol(clientProtocols=clientProtocols, serverProtocols=serverProtocols)
        self.assertIsNone(negotiatedProtocol)
        self.assertIsNone(lostReason)

class _NotSSLTransport:

    def getHandle(self):
        if False:
            i = 10
            return i + 15
        return self

class _MaybeSSLTransport:

    def getHandle(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def get_peer_certificate(self):
        if False:
            i = 10
            return i + 15
        return None

    def get_host_certificate(self):
        if False:
            while True:
                i = 10
        return None

class _ActualSSLTransport:

    def getHandle(self):
        if False:
            print('Hello World!')
        return self

    def get_host_certificate(self):
        if False:
            while True:
                i = 10
        return sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM).original

    def get_peer_certificate(self):
        if False:
            while True:
                i = 10
        return sslverify.Certificate.loadPEM(A_PEER_CERTIFICATE_PEM).original

class ConstructorsTests(TestCase):
    if skipSSL:
        skip = skipSSL

    def test_peerFromNonSSLTransport(self):
        if False:
            return 10
        '\n        Verify that peerFromTransport raises an exception if the transport\n        passed is not actually an SSL transport.\n        '
        x = self.assertRaises(CertificateError, sslverify.Certificate.peerFromTransport, _NotSSLTransport())
        self.assertTrue(str(x).startswith('non-TLS'))

    def test_peerFromBlankSSLTransport(self):
        if False:
            return 10
        "\n        Verify that peerFromTransport raises an exception if the transport\n        passed is an SSL transport, but doesn't have a peer certificate.\n        "
        x = self.assertRaises(CertificateError, sslverify.Certificate.peerFromTransport, _MaybeSSLTransport())
        self.assertTrue(str(x).startswith('TLS'))

    def test_hostFromNonSSLTransport(self):
        if False:
            return 10
        '\n        Verify that hostFromTransport raises an exception if the transport\n        passed is not actually an SSL transport.\n        '
        x = self.assertRaises(CertificateError, sslverify.Certificate.hostFromTransport, _NotSSLTransport())
        self.assertTrue(str(x).startswith('non-TLS'))

    def test_hostFromBlankSSLTransport(self):
        if False:
            return 10
        "\n        Verify that hostFromTransport raises an exception if the transport\n        passed is an SSL transport, but doesn't have a host certificate.\n        "
        x = self.assertRaises(CertificateError, sslverify.Certificate.hostFromTransport, _MaybeSSLTransport())
        self.assertTrue(str(x).startswith('TLS'))

    def test_hostFromSSLTransport(self):
        if False:
            return 10
        '\n        Verify that hostFromTransport successfully creates the correct\n        certificate if passed a valid SSL transport.\n        '
        self.assertEqual(sslverify.Certificate.hostFromTransport(_ActualSSLTransport()).serialNumber(), 12345)

    def test_peerFromSSLTransport(self):
        if False:
            i = 10
            return i + 15
        '\n        Verify that peerFromTransport successfully creates the correct\n        certificate if passed a valid SSL transport.\n        '
        self.assertEqual(sslverify.Certificate.peerFromTransport(_ActualSSLTransport()).serialNumber(), 12346)

class MultipleCertificateTrustRootTests(TestCase):
    """
    Test the behavior of the trustRootFromCertificates() API call.
    """
    if skipSSL:
        skip = skipSSL

    def test_trustRootFromCertificatesPrivatePublic(self):
        if False:
            print('Hello World!')
        '\n        L{trustRootFromCertificates} accepts either a L{sslverify.Certificate}\n        or a L{sslverify.PrivateCertificate} instance.\n        '
        privateCert = sslverify.PrivateCertificate.loadPEM(A_KEYPAIR)
        cert = sslverify.Certificate.loadPEM(A_HOST_CERTIFICATE_PEM)
        mt = sslverify.trustRootFromCertificates([privateCert, cert])
        (sProto, cProto, sWrap, cWrap, pump) = loopbackTLSConnectionInMemory(trustRoot=mt, privateKey=privateCert.privateKey.original, serverCertificate=privateCert.original)
        self.assertEqual(cWrap.data, b'greetings!')
        self.assertIsNone(cWrap.lostReason)

    def test_trustRootSelfSignedServerCertificate(self):
        if False:
            i = 10
            return i + 15
        '\n        L{trustRootFromCertificates} called with a single self-signed\n        certificate will cause L{optionsForClientTLS} to accept client\n        connections to a server with that certificate.\n        '
        (key, cert) = makeCertificate(O=b'Server Test Certificate', CN=b'server')
        selfSigned = sslverify.PrivateCertificate.fromCertificateAndKeyPair(sslverify.Certificate(cert), sslverify.KeyPair(key))
        trust = sslverify.trustRootFromCertificates([selfSigned])
        (sProto, cProto, sWrap, cWrap, pump) = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=selfSigned.privateKey.original, serverCertificate=selfSigned.original)
        self.assertEqual(cWrap.data, b'greetings!')
        self.assertIsNone(cWrap.lostReason)

    def test_trustRootCertificateAuthorityTrustsConnection(self):
        if False:
            print('Hello World!')
        '\n        L{trustRootFromCertificates} called with certificate A will cause\n        L{optionsForClientTLS} to accept client connections to a server with\n        certificate B where B is signed by A.\n        '
        (caCert, serverCert) = certificatesForAuthorityAndServer()
        trust = sslverify.trustRootFromCertificates([caCert])
        (sProto, cProto, sWrap, cWrap, pump) = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=serverCert.privateKey.original, serverCertificate=serverCert.original)
        self.assertEqual(cWrap.data, b'greetings!')
        self.assertIsNone(cWrap.lostReason)

    def test_trustRootFromCertificatesUntrusted(self):
        if False:
            return 10
        '\n        L{trustRootFromCertificates} called with certificate A will cause\n        L{optionsForClientTLS} to disallow any connections to a server with\n        certificate B where B is not signed by A.\n        '
        (key, cert) = makeCertificate(O=b'Server Test Certificate', CN=b'server')
        serverCert = sslverify.PrivateCertificate.fromCertificateAndKeyPair(sslverify.Certificate(cert), sslverify.KeyPair(key))
        untrustedCert = sslverify.Certificate(makeCertificate(O=b'CA Test Certificate', CN=b'unknown CA')[1])
        trust = sslverify.trustRootFromCertificates([untrustedCert])
        (sProto, cProto, sWrap, cWrap, pump) = loopbackTLSConnectionInMemory(trustRoot=trust, privateKey=serverCert.privateKey.original, serverCertificate=serverCert.original)
        self.assertEqual(cWrap.data, b'')
        self.assertEqual(cWrap.lostReason.type, SSL.Error)
        err = cWrap.lostReason.value
        self.assertEqual(err.args[0][0][2], 'tlsv1 alert unknown ca')

    def test_trustRootFromCertificatesOpenSSLObjects(self):
        if False:
            i = 10
            return i + 15
        '\n        L{trustRootFromCertificates} rejects any L{OpenSSL.crypto.X509}\n        instances in the list passed to it.\n        '
        private = sslverify.PrivateCertificate.loadPEM(A_KEYPAIR)
        certX509 = private.original
        exception = self.assertRaises(TypeError, sslverify.trustRootFromCertificates, [certX509])
        self.assertEqual('certificates items must be twisted.internet.ssl.CertBase instances', exception.args[0])

class OpenSSLCipherTests(TestCase):
    """
    Tests for twisted.internet._sslverify.OpenSSLCipher.
    """
    if skipSSL:
        skip = skipSSL
    cipherName = 'CIPHER-STRING'

    def test_constructorSetsFullName(self):
        if False:
            print('Hello World!')
        '\n        The first argument passed to the constructor becomes the full name.\n        '
        self.assertEqual(self.cipherName, sslverify.OpenSSLCipher(self.cipherName).fullName)

    def test_repr(self):
        if False:
            print('Hello World!')
        '\n        C{repr(cipher)} returns a valid constructor call.\n        '
        cipher = sslverify.OpenSSLCipher(self.cipherName)
        self.assertEqual(cipher, eval(repr(cipher), {'OpenSSLCipher': sslverify.OpenSSLCipher}))

    def test_eqSameClass(self):
        if False:
            return 10
        '\n        Equal type and C{fullName} means that the objects are equal.\n        '
        cipher1 = sslverify.OpenSSLCipher(self.cipherName)
        cipher2 = sslverify.OpenSSLCipher(self.cipherName)
        self.assertEqual(cipher1, cipher2)

    def test_eqSameNameDifferentType(self):
        if False:
            i = 10
            return i + 15
        "\n        If ciphers have the same name but different types, they're still\n        different.\n        "

        class DifferentCipher:
            fullName = self.cipherName
        self.assertNotEqual(sslverify.OpenSSLCipher(self.cipherName), DifferentCipher())

class ExpandCipherStringTests(TestCase):
    """
    Tests for twisted.internet._sslverify._expandCipherString.
    """
    if skipSSL:
        skip = skipSSL

    def test_doesNotStumbleOverEmptyList(self):
        if False:
            return 10
        '\n        If the expanded cipher list is empty, an empty L{list} is returned.\n        '
        self.assertEqual(tuple(), sslverify._expandCipherString('', SSL.SSLv23_METHOD, 0))

    def test_doesNotSwallowOtherSSLErrors(self):
        if False:
            print('Hello World!')
        '\n        Only no cipher matches get swallowed, every other SSL error gets\n        propagated.\n        '

        def raiser(_):
            if False:
                i = 10
                return i + 15
            raise SSL.Error([['', '', '']])
        ctx = FakeContext(SSL.SSLv23_METHOD)
        ctx.set_cipher_list = raiser
        self.patch(sslverify.SSL, 'Context', lambda _: ctx)
        self.assertRaises(SSL.Error, sslverify._expandCipherString, 'ALL', SSL.SSLv23_METHOD, 0)

    def test_returnsTupleOfICiphers(self):
        if False:
            i = 10
            return i + 15
        '\n        L{sslverify._expandCipherString} always returns a L{tuple} of\n        L{interfaces.ICipher}.\n        '
        ciphers = sslverify._expandCipherString('ALL', SSL.SSLv23_METHOD, 0)
        self.assertIsInstance(ciphers, tuple)
        bogus = []
        for c in ciphers:
            if not interfaces.ICipher.providedBy(c):
                bogus.append(c)
        self.assertEqual([], bogus)

class AcceptableCiphersTests(TestCase):
    """
    Tests for twisted.internet._sslverify.OpenSSLAcceptableCiphers.
    """
    if skipSSL:
        skip = skipSSL

    def test_selectOnEmptyListReturnsEmptyList(self):
        if False:
            return 10
        '\n        If no ciphers are available, nothing can be selected.\n        '
        ac = sslverify.OpenSSLAcceptableCiphers(tuple())
        self.assertEqual(tuple(), ac.selectCiphers(tuple()))

    def test_selectReturnsOnlyFromAvailable(self):
        if False:
            i = 10
            return i + 15
        '\n        Select only returns a cross section of what is available and what is\n        desirable.\n        '
        ac = sslverify.OpenSSLAcceptableCiphers([sslverify.OpenSSLCipher('A'), sslverify.OpenSSLCipher('B')])
        self.assertEqual((sslverify.OpenSSLCipher('B'),), ac.selectCiphers([sslverify.OpenSSLCipher('B'), sslverify.OpenSSLCipher('C')]))

    def test_fromOpenSSLCipherStringExpandsToTupleOfCiphers(self):
        if False:
            i = 10
            return i + 15
        '\n        If L{sslverify.OpenSSLAcceptableCiphers.fromOpenSSLCipherString} is\n        called it expands the string to a tuple of ciphers.\n        '
        ac = sslverify.OpenSSLAcceptableCiphers.fromOpenSSLCipherString('ALL')
        self.assertIsInstance(ac._ciphers, tuple)
        self.assertTrue(all((sslverify.ICipher.providedBy(c) for c in ac._ciphers)))

class DiffieHellmanParametersTests(TestCase):
    """
    Tests for twisted.internet._sslverify.OpenSSLDHParameters.
    """
    if skipSSL:
        skip = skipSSL
    filePath = FilePath(b'dh.params')

    def test_fromFile(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Calling C{fromFile} with a filename returns an instance with that file\n        name saved.\n        '
        params = sslverify.OpenSSLDiffieHellmanParameters.fromFile(self.filePath)
        self.assertEqual(self.filePath, params._dhFile)

class FakeLibState:
    """
    State for L{FakeLib}

    @param setECDHAutoRaises: An exception
        L{FakeLib.SSL_CTX_set_ecdh_auto} should raise; if L{None},
        nothing is raised.

    @ivar ecdhContexts: A list of SSL contexts with which
        L{FakeLib.SSL_CTX_set_ecdh_auto} was called
    @type ecdhContexts: L{list} of L{OpenSSL.SSL.Context}s

    @ivar ecdhValues: A list of boolean values with which
        L{FakeLib.SSL_CTX_set_ecdh_auto} was called
    @type ecdhValues: L{list} of L{boolean}s
    """
    __slots__ = ('setECDHAutoRaises', 'ecdhContexts', 'ecdhValues')

    def __init__(self, setECDHAutoRaises):
        if False:
            for i in range(10):
                print('nop')
        self.setECDHAutoRaises = setECDHAutoRaises
        self.ecdhContexts = []
        self.ecdhValues = []

class FakeLib:
    """
    An introspectable fake of cryptography's lib object.

    @param state: A L{FakeLibState} instance that contains this fake's
        state.
    """

    def __init__(self, state):
        if False:
            for i in range(10):
                print('nop')
        self._state = state

    def SSL_CTX_set_ecdh_auto(self, ctx, value):
        if False:
            while True:
                i = 10
        '\n        Record the context and value under in the C{_state} instance\n        variable.\n\n        @see: L{FakeLibState}\n\n        @param ctx: An SSL context.\n        @type ctx: L{OpenSSL.SSL.Context}\n\n        @param value: A boolean value\n        @type value: L{bool}\n        '
        self._state.ecdhContexts.append(ctx)
        self._state.ecdhValues.append(value)
        if self._state.setECDHAutoRaises is not None:
            raise self._state.setECDHAutoRaises

class FakeLibTests(TestCase):
    """
    Tests for L{FakeLib}.
    """

    def test_SSL_CTX_set_ecdh_auto(self):
        if False:
            print('Hello World!')
        '\n        L{FakeLib.SSL_CTX_set_ecdh_auto} records context and value it\n        was called with.\n        '
        state = FakeLibState(setECDHAutoRaises=None)
        lib = FakeLib(state)
        self.assertNot(state.ecdhContexts)
        self.assertNot(state.ecdhValues)
        (context, value) = ('CONTEXT', True)
        lib.SSL_CTX_set_ecdh_auto(context, value)
        self.assertEqual(state.ecdhContexts, [context])
        self.assertEqual(state.ecdhValues, [True])

    def test_SSL_CTX_set_ecdh_autoRaises(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{FakeLib.SSL_CTX_set_ecdh_auto} raises the exception provided\n        by its state, while still recording its arguments.\n        '
        state = FakeLibState(setECDHAutoRaises=ValueError)
        lib = FakeLib(state)
        self.assertNot(state.ecdhContexts)
        self.assertNot(state.ecdhValues)
        (context, value) = ('CONTEXT', True)
        self.assertRaises(ValueError, lib.SSL_CTX_set_ecdh_auto, context, value)
        self.assertEqual(state.ecdhContexts, [context])
        self.assertEqual(state.ecdhValues, [True])

class FakeCryptoState:
    """
    State for L{FakeCrypto}

    @param getEllipticCurveRaises: What
        L{FakeCrypto.get_elliptic_curve} should raise; L{None} and it
        won't raise anything

    @param getEllipticCurveReturns: What
        L{FakeCrypto.get_elliptic_curve} should return.

    @ivar getEllipticCurveCalls: The arguments with which
        L{FakeCrypto.get_elliptic_curve} has been called.
    @type getEllipticCurveCalls: L{list}
    """
    __slots__ = ('getEllipticCurveRaises', 'getEllipticCurveReturns', 'getEllipticCurveCalls')

    def __init__(self, getEllipticCurveRaises, getEllipticCurveReturns):
        if False:
            return 10
        self.getEllipticCurveRaises = getEllipticCurveRaises
        self.getEllipticCurveReturns = getEllipticCurveReturns
        self.getEllipticCurveCalls = []

class FakeCrypto:
    """
    An introspectable fake of pyOpenSSL's L{OpenSSL.crypto} module.

    @ivar state: A L{FakeCryptoState} instance
    """

    def __init__(self, state):
        if False:
            while True:
                i = 10
        self._state = state

    def get_elliptic_curve(self, curve):
        if False:
            i = 10
            return i + 15
        '\n        A fake that records the curve with which it was called.\n\n        @param curve: see L{crypto.get_elliptic_curve}\n\n        @return: see L{FakeCryptoState.getEllipticCurveReturns}\n        @raises: see L{FakeCryptoState.getEllipticCurveRaises}\n        '
        self._state.getEllipticCurveCalls.append(curve)
        if self._state.getEllipticCurveRaises is not None:
            raise self._state.getEllipticCurveRaises
        return self._state.getEllipticCurveReturns

class FakeCryptoTests(SynchronousTestCase):
    """
    Tests for L{FakeCrypto}.
    """

    def test_get_elliptic_curveRecordsArgument(self):
        if False:
            while True:
                i = 10
        '\n        L{FakeCrypto.test_get_elliptic_curve} records the curve with\n        which it was called.\n        '
        state = FakeCryptoState(getEllipticCurveRaises=None, getEllipticCurveReturns=None)
        crypto = FakeCrypto(state)
        crypto.get_elliptic_curve('a curve name')
        self.assertEqual(state.getEllipticCurveCalls, ['a curve name'])

    def test_get_elliptic_curveReturns(self):
        if False:
            print('Hello World!')
        '\n        L{FakeCrypto.test_get_elliptic_curve} returns the value\n        specified by its state object and records what it was called\n        with.\n        '
        returnValue = 'object'
        state = FakeCryptoState(getEllipticCurveRaises=None, getEllipticCurveReturns=returnValue)
        crypto = FakeCrypto(state)
        self.assertIs(crypto.get_elliptic_curve('another curve name'), returnValue)
        self.assertEqual(state.getEllipticCurveCalls, ['another curve name'])

    def test_get_elliptic_curveRaises(self):
        if False:
            while True:
                i = 10
        '\n        L{FakeCrypto.test_get_elliptic_curve} raises the exception\n        specified by its state object.\n        '
        state = FakeCryptoState(getEllipticCurveRaises=ValueError, getEllipticCurveReturns=None)
        crypto = FakeCrypto(state)
        self.assertRaises(ValueError, crypto.get_elliptic_curve, 'yet another curve name')
        self.assertEqual(state.getEllipticCurveCalls, ['yet another curve name'])

class ChooseDiffieHellmanEllipticCurveTests(SynchronousTestCase):
    """
    Tests for L{sslverify._ChooseDiffieHellmanEllipticCurve}.

    @cvar OPENSSL_110: A version number for OpenSSL 1.1.0

    @cvar OPENSSL_102: A version number for OpenSSL 1.0.2

    @cvar OPENSSL_101: A version number for OpenSSL 1.0.1

    @see:
        U{https://wiki.openssl.org/index.php/Manual:OPENSSL_VERSION_NUMBER(3)}
    """
    if skipSSL:
        skip = skipSSL
    OPENSSL_110 = 269484159
    OPENSSL_102 = 268443887
    OPENSSL_101 = 268439887

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.libState = FakeLibState(setECDHAutoRaises=False)
        self.lib = FakeLib(self.libState)
        self.cryptoState = FakeCryptoState(getEllipticCurveReturns=None, getEllipticCurveRaises=None)
        self.crypto = FakeCrypto(self.cryptoState)
        self.context = FakeContext(SSL.SSLv23_METHOD)

    def test_openSSL110(self):
        if False:
            while True:
                i = 10
        '\n        No configuration of contexts occurs under OpenSSL 1.1.0 and\n        later, because they create contexts with secure ECDH curves.\n\n        @see: U{http://twistedmatrix.com/trac/ticket/9210}\n        '
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_110, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(self.context)
        self.assertFalse(self.libState.ecdhContexts)
        self.assertFalse(self.libState.ecdhValues)
        self.assertFalse(self.cryptoState.getEllipticCurveCalls)
        self.assertIsNone(self.context._ecCurve)

    def test_openSSL102(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        OpenSSL 1.0.2 does not set ECDH curves by default, but\n        C{SSL_CTX_set_ecdh_auto} requests that a context choose a\n        secure set curves automatically.\n        '
        context = SSL.Context(SSL.SSLv23_METHOD)
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_102, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(context)
        self.assertEqual(self.libState.ecdhContexts, [context._context])
        self.assertEqual(self.libState.ecdhValues, [True])
        self.assertFalse(self.cryptoState.getEllipticCurveCalls)
        self.assertIsNone(self.context._ecCurve)

    def test_openSSL102SetECDHAutoRaises(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        An exception raised by C{SSL_CTX_set_ecdh_auto} under OpenSSL\n        1.0.2 is suppressed because ECDH is best-effort.\n        '
        self.libState.setECDHAutoRaises = BaseException
        context = SSL.Context(SSL.SSLv23_METHOD)
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_102, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(context)
        self.assertEqual(self.libState.ecdhContexts, [context._context])
        self.assertEqual(self.libState.ecdhValues, [True])
        self.assertFalse(self.cryptoState.getEllipticCurveCalls)

    def test_openSSL101(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        OpenSSL 1.0.1 does not set ECDH curves by default, nor does\n        it expose L{SSL_CTX_set_ecdh_auto}.  Instead, a single ECDH\n        curve can be set with L{OpenSSL.SSL.Context.set_tmp_ecdh}.\n        '
        self.cryptoState.getEllipticCurveReturns = curve = 'curve object'
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_101, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(self.context)
        self.assertFalse(self.libState.ecdhContexts)
        self.assertFalse(self.libState.ecdhValues)
        self.assertEqual(self.cryptoState.getEllipticCurveCalls, [sslverify._defaultCurveName])
        self.assertIs(self.context._ecCurve, curve)

    def test_openSSL101SetECDHRaises(self):
        if False:
            i = 10
            return i + 15
        '\n        An exception raised by L{OpenSSL.SSL.Context.set_tmp_ecdh}\n        under OpenSSL 1.0.1 is suppressed because ECHDE is best-effort.\n        '

        def set_tmp_ecdh(ctx):
            if False:
                i = 10
                return i + 15
            raise BaseException
        self.context.set_tmp_ecdh = set_tmp_ecdh
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_101, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(self.context)
        self.assertFalse(self.libState.ecdhContexts)
        self.assertFalse(self.libState.ecdhValues)
        self.assertEqual(self.cryptoState.getEllipticCurveCalls, [sslverify._defaultCurveName])

    def test_openSSL101NoECC(self):
        if False:
            print('Hello World!')
        "\n        Contexts created under an OpenSSL 1.0.1 that doesn't support\n        ECC have no configuration applied.\n        "
        self.cryptoState.getEllipticCurveRaises = ValueError
        chooser = sslverify._ChooseDiffieHellmanEllipticCurve(self.OPENSSL_101, openSSLlib=self.lib, openSSLcrypto=self.crypto)
        chooser.configureECDHCurve(self.context)
        self.assertFalse(self.libState.ecdhContexts)
        self.assertFalse(self.libState.ecdhValues)
        self.assertIsNone(self.context._ecCurve)

class KeyPairTests(TestCase):
    """
    Tests for L{sslverify.KeyPair}.
    """
    if skipSSL:
        skip = skipSSL

    def setUp(self):
        if False:
            i = 10
            return i + 15
        '\n        Create test certificate.\n        '
        self.sKey = makeCertificate(O=b'Server Test Certificate', CN=b'server')[0]

    def test_getstateDeprecation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{sslverify.KeyPair.__getstate__} is deprecated.\n        '
        self.callDeprecated((Version('Twisted', 15, 0, 0), 'a real persistence system'), sslverify.KeyPair(self.sKey).__getstate__)

    def test_setstateDeprecation(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        {sslverify.KeyPair.__setstate__} is deprecated.\n        '
        state = sslverify.KeyPair(self.sKey).dump()
        self.callDeprecated((Version('Twisted', 15, 0, 0), 'a real persistence system'), sslverify.KeyPair(self.sKey).__setstate__, state)

    def test_noTrailingNewlinePemCert(self):
        if False:
            i = 10
            return i + 15
        noTrailingNewlineKeyPemPath = getModule('twisted.test').filePath.sibling('cert.pem.no_trailing_newline')
        certPEM = noTrailingNewlineKeyPemPath.getContent()
        ssl.Certificate.loadPEM(certPEM)

class SelectVerifyImplementationTests(SynchronousTestCase):
    """
    Tests for L{_selectVerifyImplementation}.
    """
    if skipSSL:
        skip = skipSSL

    def test_dependencyMissing(self):
        if False:
            while True:
                i = 10
        '\n        If I{service_identity} cannot be imported then\n        L{_selectVerifyImplementation} returns L{simpleVerifyHostname} and\n        L{SimpleVerificationError}.\n        '
        with SetAsideModule('service_identity'):
            sys.modules['service_identity'] = None
            result = sslverify._selectVerifyImplementation()
            expected = (sslverify.simpleVerifyHostname, sslverify.simpleVerifyIPAddress, sslverify.SimpleVerificationError)
            self.assertEqual(expected, result)
    test_dependencyMissing.suppress = [util.suppress(message='You do not have a working installation of the service_identity module')]

    def test_dependencyMissingWarning(self):
        if False:
            i = 10
            return i + 15
        '\n        If I{service_identity} cannot be imported then\n        L{_selectVerifyImplementation} emits a L{UserWarning} advising the user\n        of the exact error.\n        '
        with SetAsideModule('service_identity'):
            sys.modules['service_identity'] = None
            sslverify._selectVerifyImplementation()
        [warning] = list((warning for warning in self.flushWarnings() if warning['category'] == UserWarning))
        expectedMessage = "You do not have a working installation of the service_identity module: 'import of service_identity halted; None in sys.modules'.  Please install it from <https://pypi.python.org/pypi/service_identity> and make sure all of its dependencies are satisfied.  Without the service_identity module, Twisted can perform only rudimentary TLS client hostname verification.  Many valid certificate/hostname mappings may be rejected."
        self.assertEqual(warning['message'], expectedMessage)
        self.assertEqual(warning['filename'], '')
        self.assertEqual(warning['lineno'], 0)