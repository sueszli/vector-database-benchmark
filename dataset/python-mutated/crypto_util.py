"""Crypto utilities."""
import binascii
import contextlib
import ipaddress
import logging
import os
import re
import socket
from typing import Any
from typing import Callable
from typing import List
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import Union
import josepy as jose
from OpenSSL import crypto
from OpenSSL import SSL
from acme import errors
logger = logging.getLogger(__name__)
_DEFAULT_SSL_METHOD = SSL.SSLv23_METHOD

class _DefaultCertSelection:

    def __init__(self, certs: Mapping[bytes, Tuple[crypto.PKey, crypto.X509]]):
        if False:
            print('Hello World!')
        self.certs = certs

    def __call__(self, connection: SSL.Connection) -> Optional[Tuple[crypto.PKey, crypto.X509]]:
        if False:
            i = 10
            return i + 15
        server_name = connection.get_servername()
        if server_name:
            return self.certs.get(server_name, None)
        return None

class SSLSocket:
    """SSL wrapper for sockets.

    :ivar socket sock: Original wrapped socket.
    :ivar dict certs: Mapping from domain names (`bytes`) to
        `OpenSSL.crypto.X509`.
    :ivar method: See `OpenSSL.SSL.Context` for allowed values.
    :ivar alpn_selection: Hook to select negotiated ALPN protocol for
        connection.
    :ivar cert_selection: Hook to select certificate for connection. If given,
        `certs` parameter would be ignored, and therefore must be empty.

    """

    def __init__(self, sock: socket.socket, certs: Optional[Mapping[bytes, Tuple[crypto.PKey, crypto.X509]]]=None, method: int=_DEFAULT_SSL_METHOD, alpn_selection: Optional[Callable[[SSL.Connection, List[bytes]], bytes]]=None, cert_selection: Optional[Callable[[SSL.Connection], Optional[Tuple[crypto.PKey, crypto.X509]]]]=None) -> None:
        if False:
            while True:
                i = 10
        self.sock = sock
        self.alpn_selection = alpn_selection
        self.method = method
        if not cert_selection and (not certs):
            raise ValueError('Neither cert_selection or certs specified.')
        if cert_selection and certs:
            raise ValueError('Both cert_selection and certs specified.')
        actual_cert_selection: Union[_DefaultCertSelection, Optional[Callable[[SSL.Connection], Optional[Tuple[crypto.PKey, crypto.X509]]]]] = cert_selection
        if actual_cert_selection is None:
            actual_cert_selection = _DefaultCertSelection(certs if certs else {})
        self.cert_selection = actual_cert_selection

    def __getattr__(self, name: str) -> Any:
        if False:
            for i in range(10):
                print('nop')
        return getattr(self.sock, name)

    def _pick_certificate_cb(self, connection: SSL.Connection) -> None:
        if False:
            while True:
                i = 10
        'SNI certificate callback.\n\n        This method will set a new OpenSSL context object for this\n        connection when an incoming connection provides an SNI name\n        (in order to serve the appropriate certificate, if any).\n\n        :param connection: The TLS connection object on which the SNI\n            extension was received.\n        :type connection: :class:`OpenSSL.Connection`\n\n        '
        pair = self.cert_selection(connection)
        if pair is None:
            logger.debug('Certificate selection for server name %s failed, dropping SSL', connection.get_servername())
            return
        (key, cert) = pair
        new_context = SSL.Context(self.method)
        new_context.set_options(SSL.OP_NO_SSLv2)
        new_context.set_options(SSL.OP_NO_SSLv3)
        new_context.use_privatekey(key)
        new_context.use_certificate(cert)
        if self.alpn_selection is not None:
            new_context.set_alpn_select_callback(self.alpn_selection)
        connection.set_context(new_context)

    class FakeConnection:
        """Fake OpenSSL.SSL.Connection."""

        def __init__(self, connection: SSL.Connection) -> None:
            if False:
                return 10
            self._wrapped = connection

        def __getattr__(self, name: str) -> Any:
            if False:
                for i in range(10):
                    print('nop')
            return getattr(self._wrapped, name)

        def shutdown(self, *unused_args: Any) -> bool:
            if False:
                return 10
            try:
                return self._wrapped.shutdown()
            except SSL.Error as error:
                raise socket.error(error)

    def accept(self) -> Tuple[FakeConnection, Any]:
        if False:
            i = 10
            return i + 15
        (sock, addr) = self.sock.accept()
        try:
            context = SSL.Context(self.method)
            context.set_options(SSL.OP_NO_SSLv2)
            context.set_options(SSL.OP_NO_SSLv3)
            context.set_tlsext_servername_callback(self._pick_certificate_cb)
            if self.alpn_selection is not None:
                context.set_alpn_select_callback(self.alpn_selection)
            ssl_sock = self.FakeConnection(SSL.Connection(context, sock))
            ssl_sock.set_accept_state()
            logger.debug('Performing handshake with %s', addr)
            try:
                ssl_sock.do_handshake()
            except SSL.Error as error:
                raise socket.error(error)
            return (ssl_sock, addr)
        except:
            sock.close()
            raise

def probe_sni(name: bytes, host: bytes, port: int=443, timeout: int=300, method: int=_DEFAULT_SSL_METHOD, source_address: Tuple[str, int]=('', 0), alpn_protocols: Optional[Sequence[bytes]]=None) -> crypto.X509:
    if False:
        for i in range(10):
            print('nop')
    'Probe SNI server for SSL certificate.\n\n    :param bytes name: Byte string to send as the server name in the\n        client hello message.\n    :param bytes host: Host to connect to.\n    :param int port: Port to connect to.\n    :param int timeout: Timeout in seconds.\n    :param method: See `OpenSSL.SSL.Context` for allowed values.\n    :param tuple source_address: Enables multi-path probing (selection\n        of source interface). See `socket.creation_connection` for more\n        info. Available only in Python 2.7+.\n    :param alpn_protocols: Protocols to request using ALPN.\n    :type alpn_protocols: `Sequence` of `bytes`\n\n    :raises acme.errors.Error: In case of any problems.\n\n    :returns: SSL certificate presented by the server.\n    :rtype: OpenSSL.crypto.X509\n\n    '
    context = SSL.Context(method)
    context.set_timeout(timeout)
    socket_kwargs = {'source_address': source_address}
    try:
        logger.debug('Attempting to connect to %s:%d%s.', host, port, ' from {0}:{1}'.format(source_address[0], source_address[1]) if any(source_address) else '')
        socket_tuple: Tuple[bytes, int] = (host, port)
        sock = socket.create_connection(socket_tuple, **socket_kwargs)
    except socket.error as error:
        raise errors.Error(error)
    with contextlib.closing(sock) as client:
        client_ssl = SSL.Connection(context, client)
        client_ssl.set_connect_state()
        client_ssl.set_tlsext_host_name(name)
        if alpn_protocols is not None:
            client_ssl.set_alpn_protos(alpn_protocols)
        try:
            client_ssl.do_handshake()
            client_ssl.shutdown()
        except SSL.Error as error:
            raise errors.Error(error)
    cert = client_ssl.get_peer_certificate()
    assert cert
    return cert

def make_csr(private_key_pem: bytes, domains: Optional[Union[Set[str], List[str]]]=None, must_staple: bool=False, ipaddrs: Optional[List[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]]=None) -> bytes:
    if False:
        return 10
    'Generate a CSR containing domains or IPs as subjectAltNames.\n\n    :param buffer private_key_pem: Private key, in PEM PKCS#8 format.\n    :param list domains: List of DNS names to include in subjectAltNames of CSR.\n    :param bool must_staple: Whether to include the TLS Feature extension (aka\n        OCSP Must Staple: https://tools.ietf.org/html/rfc7633).\n    :param list ipaddrs: List of IPaddress(type ipaddress.IPv4Address or ipaddress.IPv6Address)\n    names to include in subbjectAltNames of CSR.\n    params ordered this way for backward competablity when called by positional argument.\n    :returns: buffer PEM-encoded Certificate Signing Request.\n    '
    private_key = crypto.load_privatekey(crypto.FILETYPE_PEM, private_key_pem)
    csr = crypto.X509Req()
    sanlist = []
    if domains is None:
        domains = []
    if ipaddrs is None:
        ipaddrs = []
    if len(domains) + len(ipaddrs) == 0:
        raise ValueError('At least one of domains or ipaddrs parameter need to be not empty')
    for address in domains:
        sanlist.append('DNS:' + address)
    for ips in ipaddrs:
        sanlist.append('IP:' + ips.exploded)
    san_string = ', '.join(sanlist).encode('ascii')
    extensions = [crypto.X509Extension(b'subjectAltName', critical=False, value=san_string)]
    if must_staple:
        extensions.append(crypto.X509Extension(b'1.3.6.1.5.5.7.1.24', critical=False, value=b'DER:30:03:02:01:05'))
    csr.add_extensions(extensions)
    csr.set_pubkey(private_key)
    csr.set_version(0)
    csr.sign(private_key, 'sha256')
    return crypto.dump_certificate_request(crypto.FILETYPE_PEM, csr)

def _pyopenssl_cert_or_req_all_names(loaded_cert_or_req: Union[crypto.X509, crypto.X509Req]) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    common_name = loaded_cert_or_req.get_subject().CN
    sans = _pyopenssl_cert_or_req_san(loaded_cert_or_req)
    if common_name is None:
        return sans
    return [common_name] + [d for d in sans if d != common_name]

def _pyopenssl_cert_or_req_san(cert_or_req: Union[crypto.X509, crypto.X509Req]) -> List[str]:
    if False:
        print('Hello World!')
    'Get Subject Alternative Names from certificate or CSR using pyOpenSSL.\n\n    .. todo:: Implement directly in PyOpenSSL!\n\n    .. note:: Although this is `acme` internal API, it is used by\n        `letsencrypt`.\n\n    :param cert_or_req: Certificate or CSR.\n    :type cert_or_req: `OpenSSL.crypto.X509` or `OpenSSL.crypto.X509Req`.\n\n    :returns: A list of Subject Alternative Names that is DNS.\n    :rtype: `list` of `str`\n\n    '
    part_separator = ':'
    prefix = 'DNS' + part_separator
    sans_parts = _pyopenssl_extract_san_list_raw(cert_or_req)
    return [part.split(part_separator)[1] for part in sans_parts if part.startswith(prefix)]

def _pyopenssl_cert_or_req_san_ip(cert_or_req: Union[crypto.X509, crypto.X509Req]) -> List[str]:
    if False:
        i = 10
        return i + 15
    'Get Subject Alternative Names IPs from certificate or CSR using pyOpenSSL.\n\n    :param cert_or_req: Certificate or CSR.\n    :type cert_or_req: `OpenSSL.crypto.X509` or `OpenSSL.crypto.X509Req`.\n\n    :returns: A list of Subject Alternative Names that are IP Addresses.\n    :rtype: `list` of `str`. note that this returns as string, not IPaddress object\n\n    '
    part_separator = ':'
    prefix = 'IP Address' + part_separator
    sans_parts = _pyopenssl_extract_san_list_raw(cert_or_req)
    return [part[len(prefix):] for part in sans_parts if part.startswith(prefix)]

def _pyopenssl_extract_san_list_raw(cert_or_req: Union[crypto.X509, crypto.X509Req]) -> List[str]:
    if False:
        while True:
            i = 10
    'Get raw SAN string from cert or csr, parse it as UTF-8 and return.\n\n    :param cert_or_req: Certificate or CSR.\n    :type cert_or_req: `OpenSSL.crypto.X509` or `OpenSSL.crypto.X509Req`.\n\n    :returns: raw san strings, parsed byte as utf-8\n    :rtype: `list` of `str`\n\n    '
    if isinstance(cert_or_req, crypto.X509):
        text = crypto.dump_certificate(crypto.FILETYPE_TEXT, cert_or_req).decode('utf-8')
    else:
        text = crypto.dump_certificate_request(crypto.FILETYPE_TEXT, cert_or_req).decode('utf-8')
    raw_san = re.search('X509v3 Subject Alternative Name:(?: critical)?\\s*(.*)', text)
    parts_separator = ', '
    sans_parts = [] if raw_san is None else raw_san.group(1).split(parts_separator)
    return sans_parts

def gen_ss_cert(key: crypto.PKey, domains: Optional[List[str]]=None, not_before: Optional[int]=None, validity: int=7 * 24 * 60 * 60, force_san: bool=True, extensions: Optional[List[crypto.X509Extension]]=None, ips: Optional[List[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]]=None) -> crypto.X509:
    if False:
        for i in range(10):
            print('nop')
    'Generate new self-signed certificate.\n\n    :type domains: `list` of `str`\n    :param OpenSSL.crypto.PKey key:\n    :param bool force_san:\n    :param extensions: List of additional extensions to include in the cert.\n    :type extensions: `list` of `OpenSSL.crypto.X509Extension`\n    :type ips: `list` of (`ipaddress.IPv4Address` or `ipaddress.IPv6Address`)\n\n    If more than one domain is provided, all of the domains are put into\n    ``subjectAltName`` X.509 extension and first domain is set as the\n    subject CN. If only one domain is provided no ``subjectAltName``\n    extension is used, unless `force_san` is ``True``.\n\n    '
    assert domains or ips, 'Must provide one or more hostnames or IPs for the cert.'
    cert = crypto.X509()
    cert.set_serial_number(int(binascii.hexlify(os.urandom(16)), 16))
    cert.set_version(2)
    if extensions is None:
        extensions = []
    if domains is None:
        domains = []
    if ips is None:
        ips = []
    extensions.append(crypto.X509Extension(b'basicConstraints', True, b'CA:TRUE, pathlen:0'))
    if len(domains) > 0:
        cert.get_subject().CN = domains[0]
    cert.set_issuer(cert.get_subject())
    sanlist = []
    for address in domains:
        sanlist.append('DNS:' + address)
    for ip in ips:
        sanlist.append('IP:' + ip.exploded)
    san_string = ', '.join(sanlist).encode('ascii')
    if force_san or len(domains) > 1 or len(ips) > 0:
        extensions.append(crypto.X509Extension(b'subjectAltName', critical=False, value=san_string))
    cert.add_extensions(extensions)
    cert.gmtime_adj_notBefore(0 if not_before is None else not_before)
    cert.gmtime_adj_notAfter(validity)
    cert.set_pubkey(key)
    cert.sign(key, 'sha256')
    return cert

def dump_pyopenssl_chain(chain: Union[List[jose.ComparableX509], List[crypto.X509]], filetype: int=crypto.FILETYPE_PEM) -> bytes:
    if False:
        while True:
            i = 10
    'Dump certificate chain into a bundle.\n\n    :param list chain: List of `OpenSSL.crypto.X509` (or wrapped in\n        :class:`josepy.util.ComparableX509`).\n\n    :returns: certificate chain bundle\n    :rtype: bytes\n\n    '

    def _dump_cert(cert: Union[jose.ComparableX509, crypto.X509]) -> bytes:
        if False:
            i = 10
            return i + 15
        if isinstance(cert, jose.ComparableX509):
            if isinstance(cert.wrapped, crypto.X509Req):
                raise errors.Error('Unexpected CSR provided.')
            cert = cert.wrapped
        return crypto.dump_certificate(filetype, cert)
    return b''.join((_dump_cert(cert) for cert in chain))