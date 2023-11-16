import logging
from service_identity import VerificationError
from service_identity.pyopenssl import verify_hostname, verify_ip_address
from zope.interface import implementer
from OpenSSL import SSL, crypto
from twisted.internet._sslverify import _defaultCurveName
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.interfaces import IOpenSSLClientConnectionCreator
from twisted.internet.ssl import CertificateOptions, ContextFactory, TLSVersion, platformTrust
from twisted.protocols.tls import TLSMemoryBIOProtocol
from twisted.python.failure import Failure
from twisted.web.iweb import IPolicyForHTTPS
from synapse.config.homeserver import HomeServerConfig
logger = logging.getLogger(__name__)
_TLS_VERSION_MAP = {'1': TLSVersion.TLSv1_0, '1.1': TLSVersion.TLSv1_1, '1.2': TLSVersion.TLSv1_2, '1.3': TLSVersion.TLSv1_3}

class ServerContextFactory(ContextFactory):
    """Factory for PyOpenSSL SSL contexts that are used to handle incoming
    connections.

    TODO: replace this with an implementation of IOpenSSLServerConnectionCreator,
    per https://github.com/matrix-org/synapse/issues/1691
    """

    def __init__(self, config: HomeServerConfig):
        if False:
            for i in range(10):
                print('nop')
        self._context = SSL.Context(SSL.SSLv23_METHOD)
        self.configure_context(self._context, config)

    @staticmethod
    def configure_context(context: SSL.Context, config: HomeServerConfig) -> None:
        if False:
            print('Hello World!')
        try:
            _ecCurve = crypto.get_elliptic_curve(_defaultCurveName)
            context.set_tmp_ecdh(_ecCurve)
        except Exception:
            logger.exception('Failed to enable elliptic curve for TLS')
        context.set_options(SSL.OP_NO_SSLv2 | SSL.OP_NO_SSLv3 | SSL.OP_NO_TLSv1 | SSL.OP_NO_TLSv1_1)
        context.use_certificate_chain_file(config.tls.tls_certificate_file)
        assert config.tls.tls_private_key is not None
        context.use_privatekey(config.tls.tls_private_key)
        context.set_cipher_list(b'ECDH+AESGCM:ECDH+CHACHA20:ECDH+AES256:ECDH+AES128:!aNULL:!SHA1:!AESCCM')

    def getContext(self) -> SSL.Context:
        if False:
            i = 10
            return i + 15
        return self._context

@implementer(IPolicyForHTTPS)
class FederationPolicyForHTTPS:
    """Factory for Twisted SSLClientConnectionCreators that are used to make connections
    to remote servers for federation.

    Uses one of two OpenSSL context objects for all connections, depending on whether
    we should do SSL certificate verification.

    get_options decides whether we should do SSL certificate verification and
    constructs an SSLClientConnectionCreator factory accordingly.
    """

    def __init__(self, config: HomeServerConfig):
        if False:
            while True:
                i = 10
        self._config = config
        trust_root = config.tls.federation_ca_trust_root
        if trust_root is None:
            trust_root = platformTrust()
        minTLS = _TLS_VERSION_MAP[config.tls.federation_client_minimum_tls_version]
        _verify_ssl = CertificateOptions(trustRoot=trust_root, insecurelyLowerMinimumTo=minTLS)
        self._verify_ssl_context = _verify_ssl.getContext()
        self._verify_ssl_context.set_info_callback(_context_info_cb)
        _no_verify_ssl = CertificateOptions(insecurelyLowerMinimumTo=minTLS)
        self._no_verify_ssl_context = _no_verify_ssl.getContext()
        self._no_verify_ssl_context.set_info_callback(_context_info_cb)
        self._should_verify = self._config.tls.federation_verify_certificates
        self._federation_certificate_verification_whitelist = self._config.tls.federation_certificate_verification_whitelist

    def get_options(self, host: bytes) -> IOpenSSLClientConnectionCreator:
        if False:
            for i in range(10):
                print('nop')
        ascii_host = host.decode('ascii')
        should_verify = self._should_verify
        if self._should_verify:
            for regex in self._federation_certificate_verification_whitelist:
                if regex.match(ascii_host):
                    should_verify = False
                    break
        ssl_context = self._verify_ssl_context if should_verify else self._no_verify_ssl_context
        return SSLClientConnectionCreator(host, ssl_context, should_verify)

    def creatorForNetloc(self, hostname: bytes, port: int) -> IOpenSSLClientConnectionCreator:
        if False:
            i = 10
            return i + 15
        'Implements the IPolicyForHTTPS interface so that this can be passed\n        directly to agents.\n        '
        return self.get_options(hostname)

@implementer(IPolicyForHTTPS)
class RegularPolicyForHTTPS:
    """Factory for Twisted SSLClientConnectionCreators that are used to make connections
    to remote servers, for other than federation.

    Always uses the same OpenSSL context object, which uses the default OpenSSL CA
    trust root.
    """

    def __init__(self) -> None:
        if False:
            return 10
        trust_root = platformTrust()
        self._ssl_context = CertificateOptions(trustRoot=trust_root).getContext()
        self._ssl_context.set_info_callback(_context_info_cb)

    def creatorForNetloc(self, hostname: bytes, port: int) -> IOpenSSLClientConnectionCreator:
        if False:
            while True:
                i = 10
        return SSLClientConnectionCreator(hostname, self._ssl_context, True)

def _context_info_cb(ssl_connection: SSL.Connection, where: int, ret: int) -> None:
    if False:
        print('Hello World!')
    "The 'information callback' for our openssl context objects.\n\n    Note: Once this is set as the info callback on a Context object, the Context should\n    only be used with the SSLClientConnectionCreator.\n    "
    tls_protocol = ssl_connection.get_app_data()
    try:
        tls_protocol._synapse_tls_verifier.verify_context_info_cb(ssl_connection, where)
    except BaseException:
        logger.exception('Error during info_callback')
        f = Failure()
        tls_protocol.failVerification(f)

@implementer(IOpenSSLClientConnectionCreator)
class SSLClientConnectionCreator:
    """Creates openssl connection objects for client connections.

    Replaces twisted.internet.ssl.ClientTLSOptions
    """

    def __init__(self, hostname: bytes, ctx: SSL.Context, verify_certs: bool):
        if False:
            for i in range(10):
                print('nop')
        self._ctx = ctx
        self._verifier = ConnectionVerifier(hostname, verify_certs)

    def clientConnectionForTLS(self, tls_protocol: TLSMemoryBIOProtocol) -> SSL.Connection:
        if False:
            print('Hello World!')
        context = self._ctx
        connection = SSL.Connection(context, None)
        connection.set_app_data(tls_protocol)
        tls_protocol._synapse_tls_verifier = self._verifier
        return connection

class ConnectionVerifier:
    """Set the SNI, and do cert verification

    This is a thing which is attached to the TLSMemoryBIOProtocol, and is called by
    the ssl context's info callback.
    """

    def __init__(self, hostname: bytes, verify_certs: bool):
        if False:
            i = 10
            return i + 15
        self._verify_certs = verify_certs
        _decoded = hostname.decode('ascii')
        if isIPAddress(_decoded) or isIPv6Address(_decoded):
            self._is_ip_address = True
        else:
            self._is_ip_address = False
        self._hostnameBytes = hostname
        self._hostnameASCII = self._hostnameBytes.decode('ascii')

    def verify_context_info_cb(self, ssl_connection: SSL.Connection, where: int) -> None:
        if False:
            return 10
        if where & SSL.SSL_CB_HANDSHAKE_START and (not self._is_ip_address):
            ssl_connection.set_tlsext_host_name(self._hostnameBytes)
        if where & SSL.SSL_CB_HANDSHAKE_DONE and self._verify_certs:
            try:
                if self._is_ip_address:
                    verify_ip_address(ssl_connection, self._hostnameASCII)
                else:
                    verify_hostname(ssl_connection, self._hostnameASCII)
            except VerificationError:
                f = Failure()
                tls_protocol = ssl_connection.get_app_data()
                tls_protocol.failVerification(f)