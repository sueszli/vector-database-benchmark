"""
Key Exchange algorithms as listed in appendix C of RFC 4346.

XXX No support yet for PSK (also, no static DH, DSS, SRP or KRB).
"""
from scapy.layers.tls.keyexchange import ServerDHParams, ServerRSAParams, ClientDiffieHellmanPublic, ClientECDiffieHellmanPublic, _tls_server_ecdh_cls_guess, EncryptedPreMasterSecret
_tls_kx_algs = {}

class _GenericKXMetaclass(type):
    """
    We could try to set server_kx_msg and client_kx_msg while parsing
    the class name... :)
    """

    def __new__(cls, kx_name, bases, dct):
        if False:
            for i in range(10):
                print('nop')
        if kx_name != '_GenericKX':
            dct['name'] = kx_name[3:]
        the_class = super(_GenericKXMetaclass, cls).__new__(cls, kx_name, bases, dct)
        if kx_name != '_GenericKX':
            the_class.export = kx_name.endswith('_EXPORT')
            the_class.anonymous = '_anon' in kx_name
            the_class.no_ske = not ('DHE' in kx_name or '_anon' in kx_name)
            the_class.no_ske &= not the_class.export
            _tls_kx_algs[kx_name[3:]] = the_class
        return the_class

class _GenericKX(metaclass=_GenericKXMetaclass):
    pass

class KX_NULL(_GenericKX):
    descr = 'No key exchange'
    server_kx_msg_cls = lambda _, m: None
    client_kx_msg_cls = None

class KX_SSLv2(_GenericKX):
    descr = 'SSLv2 dummy key exchange class'
    server_kx_msg_cls = lambda _, m: None
    client_kx_msg_cls = None

class KX_TLS13(_GenericKX):
    descr = 'TLS 1.3 dummy key exchange class'
    server_kx_msg_cls = lambda _, m: None
    client_kx_msg_cls = None

class KX_RSA(_GenericKX):
    descr = 'RSA encryption'
    server_kx_msg_cls = lambda _, m: None
    client_kx_msg_cls = EncryptedPreMasterSecret

class KX_DHE_RSA(_GenericKX):
    descr = 'Ephemeral DH with RSA signature'
    server_kx_msg_cls = lambda _, m: ServerDHParams
    client_kx_msg_cls = ClientDiffieHellmanPublic

class KX_ECDHE_RSA(_GenericKX):
    descr = 'Ephemeral ECDH with RSA signature'
    server_kx_msg_cls = lambda _, m: _tls_server_ecdh_cls_guess(m)
    client_kx_msg_cls = ClientECDiffieHellmanPublic

class KX_RSA_EXPORT(KX_RSA):
    descr = 'RSA encryption, export version'
    server_kx_msg_cls = lambda _, m: ServerRSAParams

class KX_DHE_RSA_EXPORT(KX_DHE_RSA):
    descr = 'Ephemeral DH with RSA signature, export version'

class KX_ECDHE_ECDSA(_GenericKX):
    descr = 'Ephemeral ECDH with ECDSA signature'
    server_kx_msg_cls = lambda _, m: _tls_server_ecdh_cls_guess(m)
    client_kx_msg_cls = ClientECDiffieHellmanPublic

class KX_DH_anon(_GenericKX):
    descr = 'Anonymous DH, no signatures'
    server_kx_msg_cls = lambda _, m: ServerDHParams
    client_kx_msg_cls = ClientDiffieHellmanPublic

class KX_ECDH_anon(_GenericKX):
    descr = 'ECDH anonymous key exchange'
    server_kx_msg_cls = lambda _, m: _tls_server_ecdh_cls_guess(m)
    client_kx_msg_cls = ClientECDiffieHellmanPublic

class KX_DH_anon_EXPORT(KX_DH_anon):
    descr = 'Anonymous DH, no signatures - Export version'