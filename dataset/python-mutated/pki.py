"""
Authenticate via a PKI certificate.

.. note::

    This module is Experimental and should be used with caution

Provides an authenticate function that will allow the caller to authenticate
a user via their public cert against a pre-defined Certificate Authority.

TODO: Add a 'ca_dir' option to configure a directory of CA files, a la Apache.

:depends:    - pyOpenSSL module
"""
import logging
import salt.utils.files
try:
    try:
        from M2Crypto import X509
        HAS_M2 = True
    except ImportError:
        HAS_M2 = False
        try:
            from Cryptodome.Util import asn1
        except ImportError:
            from Crypto.Util import asn1
        import OpenSSL
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
log = logging.getLogger(__name__)

def __virtual__():
    if False:
        while True:
            i = 10
    '\n    Requires newer pycrypto and pyOpenSSL\n    '
    if HAS_DEPS:
        return True
    return False

def auth(username, password, **kwargs):
    if False:
        print('Hello World!')
    '\n    Returns True if the given user cert (password is the cert contents)\n    was issued by the CA and if cert\'s Common Name is equal to username.\n\n    Returns False otherwise.\n\n    ``username``: we need it to run the auth function from CLI/API;\n                  it should be in master config auth/acl\n    ``password``: contents of user certificate (pem-encoded user public key);\n                  why "password"? For CLI, it\'s the only available name\n\n    Configure the CA cert in the master config file:\n\n    .. code-block:: yaml\n\n        external_auth:\n          pki:\n            ca_file: /etc/pki/tls/ca_certs/trusted-ca.crt\n            your_user:\n              - .*\n    '
    pem = password
    cacert_file = __salt__['config.get']('external_auth:pki:ca_file')
    log.debug('Attempting to authenticate via pki.')
    log.debug('Using CA file: %s', cacert_file)
    log.debug('Certificate contents: %s', pem)
    if HAS_M2:
        cert = X509.load_cert_string(pem, X509.FORMAT_PEM)
        cacert = X509.load_cert(cacert_file, X509.FORMAT_PEM)
        if cert.verify(cacert.get_pubkey()):
            log.info('Successfully authenticated certificate: %s', pem)
            return True
        else:
            log.info('Failed to authenticate certificate: %s', pem)
            return False
    c = OpenSSL.crypto
    cert = c.load_certificate(c.FILETYPE_PEM, pem)
    with salt.utils.files.fopen(cacert_file) as f:
        cacert = c.load_certificate(c.FILETYPE_PEM, f.read())
    algo = cert.get_signature_algorithm()
    cert_asn1 = c.dump_certificate(c.FILETYPE_ASN1, cert)
    der = asn1.DerSequence()
    der.decode(cert_asn1)
    der_cert = der[0]
    der_sig = der[2]
    der_sig_in = asn1.DerObject()
    der_sig_in.decode(der_sig)
    sig0 = der_sig_in.payload
    if sig0[0] != '\x00':
        raise Exception('Number of unused bits is strange')
    sig = sig0[1:]
    try:
        c.verify(cacert, sig, der_cert, algo)
        assert dict(cert.get_subject().get_components())['CN'] == username, "Certificate's CN should match the username"
        log.info('Successfully authenticated certificate: %s', pem)
        return True
    except (OpenSSL.crypto.Error, AssertionError):
        log.info('Failed to authenticate certificate: %s', pem)
    return False