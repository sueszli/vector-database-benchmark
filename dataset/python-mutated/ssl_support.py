"""Support for SSL in PyMongo."""
from __future__ import annotations
from typing import Optional
from pymongo.errors import ConfigurationError
HAVE_SSL = True
try:
    import pymongo.pyopenssl_context as _ssl
except ImportError:
    try:
        import pymongo.ssl_context as _ssl
    except ImportError:
        HAVE_SSL = False
if HAVE_SSL:
    import ssl as _stdlibssl
    from ssl import CERT_NONE, CERT_REQUIRED
    HAS_SNI = _ssl.HAS_SNI
    IPADDR_SAFE = True
    SSLError = _ssl.SSLError
    BLOCKING_IO_ERRORS = _ssl.BLOCKING_IO_ERRORS

    def get_ssl_context(certfile: Optional[str], passphrase: Optional[str], ca_certs: Optional[str], crlfile: Optional[str], allow_invalid_certificates: bool, allow_invalid_hostnames: bool, disable_ocsp_endpoint_check: bool) -> _ssl.SSLContext:
        if False:
            return 10
        'Create and return an SSLContext object.'
        verify_mode = CERT_NONE if allow_invalid_certificates else CERT_REQUIRED
        ctx = _ssl.SSLContext(_ssl.PROTOCOL_SSLv23)
        if verify_mode != CERT_NONE:
            ctx.check_hostname = not allow_invalid_hostnames
        else:
            ctx.check_hostname = False
        if hasattr(ctx, 'check_ocsp_endpoint'):
            ctx.check_ocsp_endpoint = not disable_ocsp_endpoint_check
        if hasattr(ctx, 'options'):
            ctx.options |= _ssl.OP_NO_SSLv2
            ctx.options |= _ssl.OP_NO_SSLv3
            ctx.options |= _ssl.OP_NO_COMPRESSION
            ctx.options |= _ssl.OP_NO_RENEGOTIATION
        if certfile is not None:
            try:
                ctx.load_cert_chain(certfile, None, passphrase)
            except _ssl.SSLError as exc:
                raise ConfigurationError(f"Private key doesn't match certificate: {exc}") from None
        if crlfile is not None:
            if _ssl.IS_PYOPENSSL:
                raise ConfigurationError('tlsCRLFile cannot be used with PyOpenSSL')
            ctx.verify_flags = getattr(_ssl, 'VERIFY_CRL_CHECK_LEAF', 0)
            ctx.load_verify_locations(crlfile)
        if ca_certs is not None:
            ctx.load_verify_locations(ca_certs)
        elif verify_mode != CERT_NONE:
            ctx.load_default_certs()
        ctx.verify_mode = verify_mode
        return ctx
else:

    class SSLError(Exception):
        pass
    HAS_SNI = False
    IPADDR_SAFE = False
    BLOCKING_IO_ERRORS = ()

    def get_ssl_context(*dummy):
        if False:
            while True:
                i = 10
        'No ssl module, raise ConfigurationError.'
        raise ConfigurationError('The ssl module is not available.')