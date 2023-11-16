"""
TLS key exchange logic.
"""
import math
import struct
from scapy.config import conf, crypto_validator
from scapy.error import warning
from scapy.fields import ByteEnumField, ByteField, EnumField, FieldLenField, FieldListField, PacketField, ShortEnumField, ShortField, StrFixedLenField, StrLenField
from scapy.compat import orb
from scapy.packet import Packet, Raw, Padding
from scapy.layers.tls.cert import PubKeyRSA, PrivKeyRSA
from scapy.layers.tls.session import _GenericTLSSessionInheritance
from scapy.layers.tls.basefields import _tls_version, _TLSClientVersionField
from scapy.layers.tls.crypto.pkcs1 import pkcs_i2osp, pkcs_os2ip
from scapy.layers.tls.crypto.groups import _ffdh_groups, _tls_named_curves, _tls_named_groups_generate, _tls_named_groups_import, _tls_named_groups_pubbytes
if conf.crypto_valid:
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import dh, ec
    from cryptography.hazmat.primitives import serialization
if conf.crypto_valid_advanced:
    from cryptography.hazmat.primitives.asymmetric import x25519
    from cryptography.hazmat.primitives.asymmetric import x448
_tls_hash_sig = {0: 'none+anon', 1: 'none+rsa', 2: 'none+dsa', 3: 'none+ecdsa', 256: 'md5+anon', 257: 'md5+rsa', 258: 'md5+dsa', 259: 'md5+ecdsa', 512: 'sha1+anon', 513: 'sha1+rsa', 514: 'sha1+dsa', 515: 'sha1+ecdsa', 768: 'sha224+anon', 769: 'sha224+rsa', 770: 'sha224+dsa', 771: 'sha224+ecdsa', 1024: 'sha256+anon', 1025: 'sha256+rsa', 1026: 'sha256+dsa', 1027: 'sha256+ecdsa', 1280: 'sha384+anon', 1281: 'sha384+rsa', 1282: 'sha384+dsa', 1283: 'sha384+ecdsa', 1536: 'sha512+anon', 1537: 'sha512+rsa', 1538: 'sha512+dsa', 1539: 'sha512+ecdsa', 2052: 'sha256+rsaepss', 2053: 'sha384+rsaepss', 2054: 'sha512+rsaepss', 2055: 'ed25519', 2056: 'ed448', 2057: 'sha256+rsapss', 2058: 'sha384+rsapss', 2059: 'sha512+rsapss'}

def phantom_mode(pkt):
    if False:
        return 10
    "\n    We expect this. If tls_version is not set, this means we did not process\n    any complete ClientHello, so we're most probably reading/building a\n    signature_algorithms extension, hence we cannot be in phantom_mode.\n    However, if the tls_version has been set, we test for TLS 1.2.\n    "
    if not pkt.tls_session:
        return False
    if not pkt.tls_session.tls_version:
        return False
    return pkt.tls_session.tls_version < 771

def phantom_decorate(f, get_or_add):
    if False:
        print('Hello World!')
    '\n    Decorator for version-dependent fields.\n    If get_or_add is True (means get), we return s, self.phantom_value.\n    If it is False (means add), we return s.\n    '

    def wrapper(*args):
        if False:
            for i in range(10):
                print('nop')
        (self, pkt, s) = args[:3]
        if phantom_mode(pkt):
            if get_or_add:
                return (s, self.phantom_value)
            return s
        return f(*args)
    return wrapper

class SigAndHashAlgField(EnumField):
    """Used in _TLSSignature."""
    phantom_value = None
    getfield = phantom_decorate(EnumField.getfield, True)
    addfield = phantom_decorate(EnumField.addfield, False)

class SigAndHashAlgsLenField(FieldLenField):
    """Used in TLS_Ext_SignatureAlgorithms and TLSCertificateResquest."""
    phantom_value = 0
    getfield = phantom_decorate(FieldLenField.getfield, True)
    addfield = phantom_decorate(FieldLenField.addfield, False)

class SigAndHashAlgsField(FieldListField):
    """Used in TLS_Ext_SignatureAlgorithms and TLSCertificateResquest."""
    phantom_value = []
    getfield = phantom_decorate(FieldListField.getfield, True)
    addfield = phantom_decorate(FieldListField.addfield, False)

class SigLenField(FieldLenField):
    """There is a trick for SSLv2, which uses implicit lengths..."""

    def getfield(self, pkt, s):
        if False:
            for i in range(10):
                print('nop')
        v = pkt.tls_session.tls_version
        if v and v < 768:
            return (s, None)
        return super(SigLenField, self).getfield(pkt, s)

    def addfield(self, pkt, s, val):
        if False:
            for i in range(10):
                print('nop')
        'With SSLv2 you will never be able to add a sig_len.'
        v = pkt.tls_session.tls_version
        if v and v < 768:
            return s
        return super(SigLenField, self).addfield(pkt, s, val)

class SigValField(StrLenField):
    """There is a trick for SSLv2, which uses implicit lengths..."""

    def getfield(self, pkt, m):
        if False:
            return 10
        s = pkt.tls_session
        if s.tls_version and s.tls_version < 768:
            if len(s.client_certs) > 0:
                sig_len = s.client_certs[0].pubKey.pubkey.key_size // 8
            else:
                warning("No client certificate provided. We're making a wild guess about the signature size.")
                sig_len = 256
            return (m[sig_len:], self.m2i(pkt, m[:sig_len]))
        return super(SigValField, self).getfield(pkt, m)

class _TLSSignature(_GenericTLSSessionInheritance):
    """
    Prior to TLS 1.2, digitally-signed structure implicitly used the
    concatenation of a MD5 hash and a SHA-1 hash.
    Then TLS 1.2 introduced explicit SignatureAndHashAlgorithms,
    i.e. couples of (hash_alg, sig_alg). See RFC 5246, section 7.4.1.4.1.

    By default, the _TLSSignature implements the TLS 1.2 scheme,
    but if it is provided a TLS context with a tls_version < 0x0303
    at initialization, it will fall back to the implicit signature.
    Even more, the 'sig_len' field won't be used with SSLv2.

    #XXX 'sig_alg' should be set in __init__ depending on the context.
    """
    name = 'TLS Digital Signature'
    fields_desc = [SigAndHashAlgField('sig_alg', 2052, _tls_hash_sig), SigLenField('sig_len', None, fmt='!H', length_of='sig_val'), SigValField('sig_val', None, length_from=lambda pkt: pkt.sig_len)]

    def __init__(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        super(_TLSSignature, self).__init__(*args, **kargs)
        if self.tls_session and self.tls_session.tls_version:
            if self.tls_session.tls_version < 771:
                self.sig_alg = None
            elif self.tls_session.tls_version == 772:
                self.sig_alg = 2052

    def _update_sig(self, m, key):
        if False:
            return 10
        "\n        Sign 'm' with the PrivKey 'key' and update our own 'sig_val'.\n        Note that, even when 'sig_alg' is not None, we use the signature scheme\n        of the PrivKey (neither do we care to compare the both of them).\n        "
        if self.sig_alg is None:
            if self.tls_session.tls_version >= 768:
                self.sig_val = key.sign(m, t='pkcs', h='md5-sha1')
            else:
                self.sig_val = key.sign(m, t='pkcs', h='md5')
        else:
            (h, sig) = _tls_hash_sig[self.sig_alg].split('+')
            if sig.endswith('pss'):
                t = 'pss'
            else:
                t = 'pkcs'
            self.sig_val = key.sign(m, t=t, h=h)

    def _verify_sig(self, m, cert):
        if False:
            return 10
        "\n        Verify that our own 'sig_val' carries the signature of 'm' by the\n        key associated to the Cert 'cert'.\n        "
        if self.sig_val:
            if self.sig_alg:
                (h, sig) = _tls_hash_sig[self.sig_alg].split('+')
                if sig.endswith('pss'):
                    t = 'pss'
                else:
                    t = 'pkcs'
                return cert.verify(m, self.sig_val, t=t, h=h)
            elif self.tls_session.tls_version >= 768:
                return cert.verify(m, self.sig_val, t='pkcs', h='md5-sha1')
            else:
                return cert.verify(m, self.sig_val, t='pkcs', h='md5')
        return False

    def guess_payload_class(self, p):
        if False:
            print('Hello World!')
        return Padding

class _TLSSignatureField(PacketField):
    """
    Used for 'digitally-signed struct' in several ServerKeyExchange,
    and also in CertificateVerify. We can handle the anonymous case.
    """
    __slots__ = ['length_from']

    def __init__(self, name, default, length_from=None):
        if False:
            while True:
                i = 10
        self.length_from = length_from
        PacketField.__init__(self, name, default, _TLSSignature)

    def m2i(self, pkt, m):
        if False:
            while True:
                i = 10
        tmp_len = self.length_from(pkt)
        if tmp_len == 0:
            return None
        return _TLSSignature(m, tls_session=pkt.tls_session)

    def getfield(self, pkt, s):
        if False:
            while True:
                i = 10
        i = self.m2i(pkt, s)
        if i is None:
            return (s, None)
        remain = b''
        if conf.padding_layer in i:
            r = i[conf.padding_layer]
            del r.underlayer.payload
            remain = r.load
        return (remain, i)

class _TLSServerParamsField(PacketField):
    """
    This is a dispatcher for the Server*DHParams below, used in
    TLSServerKeyExchange and based on the key_exchange.server_kx_msg_cls.
    When this cls is None, it means that we should not see a ServerKeyExchange,
    so we grab everything within length_from and make it available using Raw.

    When the context has not been set (e.g. when no ServerHello was parsed or
    dissected beforehand), we (kinda) clumsily set the cls by trial and error.
    XXX We could use Serv*DHParams.check_params() once it has been implemented.
    """
    __slots__ = ['length_from']

    def __init__(self, name, default, length_from=None):
        if False:
            return 10
        self.length_from = length_from
        PacketField.__init__(self, name, default, None)

    def m2i(self, pkt, m):
        if False:
            i = 10
            return i + 15
        s = pkt.tls_session
        tmp_len = self.length_from(pkt)
        if s.prcs:
            cls = s.prcs.key_exchange.server_kx_msg_cls(m)
            if cls is None:
                return Raw(m[:tmp_len]) / Padding(m[tmp_len:])
            return cls(m, tls_session=s)
        else:
            try:
                p = ServerDHParams(m, tls_session=s)
                if pkcs_os2ip(p.load[:2]) not in _tls_hash_sig:
                    raise Exception
                return p
            except Exception:
                cls = _tls_server_ecdh_cls_guess(m)
                p = cls(m, tls_session=s)
                if pkcs_os2ip(p.load[:2]) not in _tls_hash_sig:
                    return Raw(m[:tmp_len]) / Padding(m[tmp_len:])
                return p

class ServerDHParams(_GenericTLSSessionInheritance):
    """
    ServerDHParams for FFDH-based key exchanges, as defined in RFC 5246/7.4.3.

    Either with .fill_missing() or .post_dissection(), the server_kx_privkey or
    server_kx_pubkey of the TLS context are updated according to the
    parsed/assembled values. It is the user's responsibility to store and
    restore the original values if he wants to keep them. For instance, this
    could be done between the writing of a ServerKeyExchange and the receiving
    of a ClientKeyExchange (which includes secret generation).
    """
    name = 'Server FFDH parameters'
    fields_desc = [FieldLenField('dh_plen', None, length_of='dh_p'), StrLenField('dh_p', '', length_from=lambda pkt: pkt.dh_plen), FieldLenField('dh_glen', None, length_of='dh_g'), StrLenField('dh_g', '', length_from=lambda pkt: pkt.dh_glen), FieldLenField('dh_Yslen', None, length_of='dh_Ys'), StrLenField('dh_Ys', '', length_from=lambda pkt: pkt.dh_Yslen)]

    @crypto_validator
    def fill_missing(self):
        if False:
            while True:
                i = 10
        '\n        We do not want TLSServerKeyExchange.build() to overload and recompute\n        things every time it is called. This method can be called specifically\n        to have things filled in a smart fashion.\n\n        Note that we do not expect default_params.g to be more than 0xff.\n        '
        s = self.tls_session
        default_params = _ffdh_groups['modp2048'][0].parameter_numbers()
        default_mLen = _ffdh_groups['modp2048'][1]
        if not self.dh_p:
            self.dh_p = pkcs_i2osp(default_params.p, default_mLen // 8)
        if self.dh_plen is None:
            self.dh_plen = len(self.dh_p)
        if not self.dh_g:
            self.dh_g = pkcs_i2osp(default_params.g, 1)
        if self.dh_glen is None:
            self.dh_glen = 1
        p = pkcs_os2ip(self.dh_p)
        g = pkcs_os2ip(self.dh_g)
        real_params = dh.DHParameterNumbers(p, g).parameters(default_backend())
        if not self.dh_Ys:
            s.server_kx_privkey = real_params.generate_private_key()
            pubkey = s.server_kx_privkey.public_key()
            y = pubkey.public_numbers().y
            self.dh_Ys = pkcs_i2osp(y, pubkey.key_size // 8)
        if self.dh_Yslen is None:
            self.dh_Yslen = len(self.dh_Ys)
        if not s.client_kx_ffdh_params:
            s.client_kx_ffdh_params = real_params

    @crypto_validator
    def register_pubkey(self):
        if False:
            print('Hello World!')
        '\n        XXX Check that the pubkey received is in the group.\n        '
        p = pkcs_os2ip(self.dh_p)
        g = pkcs_os2ip(self.dh_g)
        pn = dh.DHParameterNumbers(p, g)
        y = pkcs_os2ip(self.dh_Ys)
        public_numbers = dh.DHPublicNumbers(y, pn)
        s = self.tls_session
        s.server_kx_pubkey = public_numbers.public_key(default_backend())
        if not s.client_kx_ffdh_params:
            s.client_kx_ffdh_params = pn.parameters(default_backend())

    def post_dissection(self, r):
        if False:
            while True:
                i = 10
        try:
            self.register_pubkey()
        except ImportError:
            pass

    def guess_payload_class(self, p):
        if False:
            return 10
        '\n        The signature after the params gets saved as Padding.\n        This way, the .getfield() which _TLSServerParamsField inherits\n        from PacketField will return the signature remain as expected.\n        '
        return Padding
_tls_ec_curve_types = {1: 'explicit_prime', 2: 'explicit_char2', 3: 'named_curve'}
_tls_ec_basis_types = {0: 'ec_basis_trinomial', 1: 'ec_basis_pentanomial'}

class ECCurvePkt(Packet):
    name = 'Elliptic Curve'
    fields_desc = [FieldLenField('alen', None, length_of='a', fmt='B'), StrLenField('a', '', length_from=lambda pkt: pkt.alen), FieldLenField('blen', None, length_of='b', fmt='B'), StrLenField('b', '', length_from=lambda pkt: pkt.blen)]

class ECTrinomialBasis(Packet):
    name = 'EC Trinomial Basis'
    val = 0
    fields_desc = [FieldLenField('klen', None, length_of='k', fmt='B'), StrLenField('k', '', length_from=lambda pkt: pkt.klen)]

    def guess_payload_class(self, p):
        if False:
            while True:
                i = 10
        return Padding

class ECPentanomialBasis(Packet):
    name = 'EC Pentanomial Basis'
    val = 1
    fields_desc = [FieldLenField('k1len', None, length_of='k1', fmt='B'), StrLenField('k1', '', length_from=lambda pkt: pkt.k1len), FieldLenField('k2len', None, length_of='k2', fmt='B'), StrLenField('k2', '', length_from=lambda pkt: pkt.k2len), FieldLenField('k3len', None, length_of='k3', fmt='B'), StrLenField('k3', '', length_from=lambda pkt: pkt.k3len)]

    def guess_payload_class(self, p):
        if False:
            return 10
        return Padding
_tls_ec_basis_cls = {0: ECTrinomialBasis, 1: ECPentanomialBasis}

class _ECBasisTypeField(ByteEnumField):
    __slots__ = ['basis_type_of']

    def __init__(self, name, default, enum, basis_type_of, remain=0):
        if False:
            for i in range(10):
                print('nop')
        self.basis_type_of = basis_type_of
        EnumField.__init__(self, name, default, enum, 'B')

    def i2m(self, pkt, x):
        if False:
            while True:
                i = 10
        if x is None:
            (fld, fval) = pkt.getfield_and_val(self.basis_type_of)
            x = fld.i2basis_type(pkt, fval)
        return x

class _ECBasisField(PacketField):
    __slots__ = ['clsdict', 'basis_type_from']

    def __init__(self, name, default, basis_type_from, clsdict):
        if False:
            print('Hello World!')
        self.clsdict = clsdict
        self.basis_type_from = basis_type_from
        PacketField.__init__(self, name, default, None)

    def m2i(self, pkt, m):
        if False:
            for i in range(10):
                print('nop')
        basis = self.basis_type_from(pkt)
        cls = self.clsdict[basis]
        return cls(m)

    def i2basis_type(self, pkt, x):
        if False:
            while True:
                i = 10
        val = 0
        try:
            val = x.val
        except Exception:
            pass
        return val

class ServerECDHExplicitPrimeParams(_GenericTLSSessionInheritance):
    """
    We provide parsing abilities for ExplicitPrimeParams, but there is no
    support from the cryptography library, hence no context operations.
    """
    name = 'Server ECDH parameters - Explicit Prime'
    fields_desc = [ByteEnumField('curve_type', 1, _tls_ec_curve_types), FieldLenField('plen', None, length_of='p', fmt='B'), StrLenField('p', '', length_from=lambda pkt: pkt.plen), PacketField('curve', None, ECCurvePkt), FieldLenField('baselen', None, length_of='base', fmt='B'), StrLenField('base', '', length_from=lambda pkt: pkt.baselen), FieldLenField('orderlen', None, length_of='order', fmt='B'), StrLenField('order', '', length_from=lambda pkt: pkt.orderlen), FieldLenField('cofactorlen', None, length_of='cofactor', fmt='B'), StrLenField('cofactor', '', length_from=lambda pkt: pkt.cofactorlen), FieldLenField('pointlen', None, length_of='point', fmt='B'), StrLenField('point', '', length_from=lambda pkt: pkt.pointlen)]

    def fill_missing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Note that if it is not set by the user, the cofactor will always\n        be 1. It is true for most, but not all, TLS elliptic curves.\n        '
        if self.curve_type is None:
            self.curve_type = _tls_ec_curve_types['explicit_prime']

    def guess_payload_class(self, p):
        if False:
            print('Hello World!')
        return Padding

class ServerECDHExplicitChar2Params(_GenericTLSSessionInheritance):
    """
    We provide parsing abilities for Char2Params, but there is no
    support from the cryptography library, hence no context operations.
    """
    name = 'Server ECDH parameters - Explicit Char2'
    fields_desc = [ByteEnumField('curve_type', 2, _tls_ec_curve_types), ShortField('m', None), _ECBasisTypeField('basis_type', None, _tls_ec_basis_types, 'basis'), _ECBasisField('basis', ECTrinomialBasis(), lambda pkt: pkt.basis_type, _tls_ec_basis_cls), PacketField('curve', ECCurvePkt(), ECCurvePkt), FieldLenField('baselen', None, length_of='base', fmt='B'), StrLenField('base', '', length_from=lambda pkt: pkt.baselen), ByteField('order', None), ByteField('cofactor', None), FieldLenField('pointlen', None, length_of='point', fmt='B'), StrLenField('point', '', length_from=lambda pkt: pkt.pointlen)]

    def fill_missing(self):
        if False:
            i = 10
            return i + 15
        if self.curve_type is None:
            self.curve_type = _tls_ec_curve_types['explicit_char2']

    def guess_payload_class(self, p):
        if False:
            for i in range(10):
                print('nop')
        return Padding

class ServerECDHNamedCurveParams(_GenericTLSSessionInheritance):
    name = 'Server ECDH parameters - Named Curve'
    fields_desc = [ByteEnumField('curve_type', 3, _tls_ec_curve_types), ShortEnumField('named_curve', None, _tls_named_curves), FieldLenField('pointlen', None, length_of='point', fmt='B'), StrLenField('point', None, length_from=lambda pkt: pkt.pointlen)]

    @crypto_validator
    def fill_missing(self):
        if False:
            print('Hello World!')
        "\n        We do not want TLSServerKeyExchange.build() to overload and recompute\n        things every time it is called. This method can be called specifically\n        to have things filled in a smart fashion.\n\n        XXX We should account for the point_format (before 'point' filling).\n        "
        s = self.tls_session
        if self.curve_type is None:
            self.curve_type = _tls_ec_curve_types['named_curve']
        if self.named_curve is None:
            self.named_curve = 23
        curve_group = self.named_curve
        if curve_group not in _tls_named_curves:
            curve_group = 23
        s.server_kx_privkey = _tls_named_groups_generate(curve_group)
        if self.point is None:
            self.point = _tls_named_groups_pubbytes(s.server_kx_privkey)
        if self.pointlen is None:
            self.pointlen = len(self.point)
        if not s.client_kx_ecdh_params:
            s.client_kx_ecdh_params = curve_group

    @crypto_validator
    def register_pubkey(self):
        if False:
            return 10
        '\n        XXX Support compressed point format.\n        XXX Check that the pubkey received is on the curve.\n        '
        s = self.tls_session
        s.server_kx_pubkey = _tls_named_groups_import(self.named_curve, self.point)
        if not s.client_kx_ecdh_params:
            s.client_kx_ecdh_params = self.named_curve

    def post_dissection(self, r):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.register_pubkey()
        except ImportError:
            pass

    def guess_payload_class(self, p):
        if False:
            while True:
                i = 10
        return Padding
_tls_server_ecdh_cls = {1: ServerECDHExplicitPrimeParams, 2: ServerECDHExplicitChar2Params, 3: ServerECDHNamedCurveParams}

def _tls_server_ecdh_cls_guess(m):
    if False:
        print('Hello World!')
    if not m:
        return None
    curve_type = orb(m[0])
    return _tls_server_ecdh_cls.get(curve_type, None)

class ServerRSAParams(_GenericTLSSessionInheritance):
    """
    Defined for RSA_EXPORT kx : it enables servers to share RSA keys shorter
    than their principal {>512}-bit key, when it is not allowed for kx.

    This should not appear in standard RSA kx negotiation, as the key
    has already been advertised in the Certificate message.
    """
    name = 'Server RSA_EXPORT parameters'
    fields_desc = [FieldLenField('rsamodlen', None, length_of='rsamod'), StrLenField('rsamod', '', length_from=lambda pkt: pkt.rsamodlen), FieldLenField('rsaexplen', None, length_of='rsaexp'), StrLenField('rsaexp', '', length_from=lambda pkt: pkt.rsaexplen)]

    @crypto_validator
    def fill_missing(self):
        if False:
            for i in range(10):
                print('nop')
        k = PrivKeyRSA()
        k.fill_and_store(modulusLen=512)
        self.tls_session.server_tmp_rsa_key = k
        pubNum = k.pubkey.public_numbers()
        if not self.rsamod:
            self.rsamod = pkcs_i2osp(pubNum.n, k.pubkey.key_size // 8)
        if self.rsamodlen is None:
            self.rsamodlen = len(self.rsamod)
        rsaexplen = math.ceil(math.log(pubNum.e) / math.log(2) / 8.0)
        if not self.rsaexp:
            self.rsaexp = pkcs_i2osp(pubNum.e, rsaexplen)
        if self.rsaexplen is None:
            self.rsaexplen = len(self.rsaexp)

    @crypto_validator
    def register_pubkey(self):
        if False:
            print('Hello World!')
        mLen = self.rsamodlen
        m = self.rsamod
        e = self.rsaexp
        self.tls_session.server_tmp_rsa_key = PubKeyRSA((e, m, mLen))

    def post_dissection(self, pkt):
        if False:
            print('Hello World!')
        try:
            self.register_pubkey()
        except ImportError:
            pass

    def guess_payload_class(self, p):
        if False:
            for i in range(10):
                print('nop')
        return Padding

class ServerPSKParams(Packet):
    """
    XXX We provide some parsing abilities for ServerPSKParams, but the
    context operations have not been implemented yet. See RFC 4279.
    Note that we do not cover the (EC)DHE_PSK key exchange,
    which should contain a Server*DHParams after 'psk_identity_hint'.
    """
    name = 'Server PSK parameters'
    fields_desc = [FieldLenField('psk_identity_hint_len', None, length_of='psk_identity_hint', fmt='!H'), StrLenField('psk_identity_hint', '', length_from=lambda pkt: pkt.psk_identity_hint_len)]

    def fill_missing(self):
        if False:
            while True:
                i = 10
        pass

    def post_dissection(self, pkt):
        if False:
            while True:
                i = 10
        pass

    def guess_payload_class(self, p):
        if False:
            i = 10
            return i + 15
        return Padding

class ClientDiffieHellmanPublic(_GenericTLSSessionInheritance):
    """
    If the user provides a value for dh_Yc attribute, we assume he will set
    the pms and ms accordingly and trigger the key derivation on his own.

    XXX As specified in 7.4.7.2. of RFC 4346, we should distinguish the needs
    for implicit or explicit value depending on availability of DH parameters
    in *client* certificate. For now we can only do ephemeral/explicit DH.
    """
    name = 'Client DH Public Value'
    fields_desc = [FieldLenField('dh_Yclen', None, length_of='dh_Yc'), StrLenField('dh_Yc', '', length_from=lambda pkt: pkt.dh_Yclen)]

    @crypto_validator
    def fill_missing(self):
        if False:
            print('Hello World!')
        s = self.tls_session
        s.client_kx_privkey = s.client_kx_ffdh_params.generate_private_key()
        pubkey = s.client_kx_privkey.public_key()
        y = pubkey.public_numbers().y
        self.dh_Yc = pkcs_i2osp(y, pubkey.key_size // 8)
        if s.client_kx_privkey and s.server_kx_pubkey:
            pms = s.client_kx_privkey.exchange(s.server_kx_pubkey)
            s.pre_master_secret = pms.lstrip(b'\x00')
            if not s.extms:
                s.compute_ms_and_derive_keys()

    def post_build(self, pkt, pay):
        if False:
            return 10
        if not self.dh_Yc:
            try:
                self.fill_missing()
            except ImportError:
                pass
        if self.dh_Yclen is None:
            self.dh_Yclen = len(self.dh_Yc)
        return pkcs_i2osp(self.dh_Yclen, 2) + self.dh_Yc + pay

    def post_dissection(self, m):
        if False:
            i = 10
            return i + 15
        '\n        First we update the client DHParams. Then, we try to update the server\n        DHParams generated during Server*DHParams building, with the shared\n        secret. Finally, we derive the session keys and update the context.\n        '
        s = self.tls_session
        if s.client_kx_ffdh_params:
            y = pkcs_os2ip(self.dh_Yc)
            param_numbers = s.client_kx_ffdh_params.parameter_numbers()
            public_numbers = dh.DHPublicNumbers(y, param_numbers)
            s.client_kx_pubkey = public_numbers.public_key(default_backend())
        if s.server_kx_privkey and s.client_kx_pubkey:
            ZZ = s.server_kx_privkey.exchange(s.client_kx_pubkey)
            s.pre_master_secret = ZZ.lstrip(b'\x00')
            if not s.extms:
                s.compute_ms_and_derive_keys()

    def guess_payload_class(self, p):
        if False:
            return 10
        return Padding

class ClientECDiffieHellmanPublic(_GenericTLSSessionInheritance):
    """
    Note that the 'len' field is 1 byte longer than with the previous class.
    """
    name = 'Client ECDH Public Value'
    fields_desc = [FieldLenField('ecdh_Yclen', None, length_of='ecdh_Yc', fmt='B'), StrLenField('ecdh_Yc', '', length_from=lambda pkt: pkt.ecdh_Yclen)]

    @crypto_validator
    def fill_missing(self):
        if False:
            i = 10
            return i + 15
        s = self.tls_session
        s.client_kx_privkey = _tls_named_groups_generate(s.client_kx_ecdh_params)
        pubkey = s.client_kx_privkey.public_key()
        if isinstance(pubkey, (x25519.X25519PublicKey, x448.X448PublicKey)):
            self.ecdh_Yc = pubkey.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
            if s.client_kx_privkey and s.server_kx_pubkey:
                pms = s.client_kx_privkey.exchange(s.server_kx_pubkey)
        else:
            x = pubkey.public_numbers().x
            y = pubkey.public_numbers().y
            self.ecdh_Yc = b'\x04' + pkcs_i2osp(x, pubkey.key_size // 8) + pkcs_i2osp(y, pubkey.key_size // 8)
            if s.client_kx_privkey and s.server_kx_pubkey:
                pms = s.client_kx_privkey.exchange(ec.ECDH(), s.server_kx_pubkey)
        if s.client_kx_privkey and s.server_kx_pubkey:
            s.pre_master_secret = pms
            if not s.extms:
                s.compute_ms_and_derive_keys()

    def post_build(self, pkt, pay):
        if False:
            return 10
        if not self.ecdh_Yc:
            try:
                self.fill_missing()
            except ImportError:
                pass
        if self.ecdh_Yclen is None:
            self.ecdh_Yclen = len(self.ecdh_Yc)
        return pkcs_i2osp(self.ecdh_Yclen, 1) + self.ecdh_Yc + pay

    def post_dissection(self, m):
        if False:
            i = 10
            return i + 15
        s = self.tls_session
        if s.client_kx_ecdh_params:
            s.client_kx_pubkey = _tls_named_groups_import(s.client_kx_ecdh_params, self.ecdh_Yc)
        if s.server_kx_privkey and s.client_kx_pubkey:
            ZZ = s.server_kx_privkey.exchange(ec.ECDH(), s.client_kx_pubkey)
            s.pre_master_secret = ZZ
            if not s.extms:
                s.compute_ms_and_derive_keys()

class _UnEncryptedPreMasterSecret(Raw):
    """
    When the content of an EncryptedPreMasterSecret could not be deciphered,
    we use this class to represent the encrypted data.
    """
    name = 'RSA Encrypted PreMaster Secret (protected)'

    def __init__(self, *args, **kargs):
        if False:
            print('Hello World!')
        kargs.pop('tls_session', None)
        return super(_UnEncryptedPreMasterSecret, self).__init__(*args, **kargs)

class EncryptedPreMasterSecret(_GenericTLSSessionInheritance):
    """
    Pay attention to implementation notes in section 7.4.7.1 of RFC 5246.
    """
    name = 'RSA Encrypted PreMaster Secret'
    fields_desc = [_TLSClientVersionField('client_version', None, _tls_version), StrFixedLenField('random', None, 46)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            while True:
                i = 10
        if _pkt and 'tls_session' in kargs:
            s = kargs['tls_session']
            if s.server_tmp_rsa_key is None and s.server_rsa_key is None:
                return _UnEncryptedPreMasterSecret
        return EncryptedPreMasterSecret

    def pre_dissect(self, m):
        if False:
            i = 10
            return i + 15
        s = self.tls_session
        tbd = m
        tls_version = s.tls_version
        if tls_version is None:
            tls_version = s.advertised_tls_version
        if tls_version >= 769:
            if len(m) < 2:
                return m
            tmp_len = struct.unpack('!H', m[:2])[0]
            if len(m) != tmp_len + 2:
                err = 'TLS 1.0+, but RSA Encrypted PMS with no explicit length'
                warning(err)
            else:
                tbd = m[2:]
        if s.server_tmp_rsa_key is not None:
            decrypted = s.server_tmp_rsa_key.decrypt(tbd)
            pms = decrypted[-48:]
        elif s.server_rsa_key is not None:
            decrypted = s.server_rsa_key.decrypt(tbd)
            pms = decrypted[-48:]
        else:
            pms = b'\x00' * 48
            err = 'No server RSA key to decrypt Pre Master Secret. Skipping.'
            warning(err)
        s.pre_master_secret = pms
        if not s.extms:
            s.compute_ms_and_derive_keys()
        return pms

    def post_build(self, pkt, pay):
        if False:
            while True:
                i = 10
        '\n        We encrypt the premaster secret (the 48 bytes) with either the server\n        certificate or the temporary RSA key provided in a server key exchange\n        message. After that step, we add the 2 bytes to provide the length, as\n        described in implementation notes at the end of section 7.4.7.1.\n        '
        enc = pkt
        s = self.tls_session
        s.pre_master_secret = enc
        if not s.extms:
            s.compute_ms_and_derive_keys()
        if s.server_tmp_rsa_key is not None:
            enc = s.server_tmp_rsa_key.encrypt(pkt, t='pkcs')
        elif s.server_certs is not None and len(s.server_certs) > 0:
            enc = s.server_certs[0].encrypt(pkt, t='pkcs')
        else:
            warning('No material to encrypt Pre Master Secret')
        tmp_len = b''
        tls_version = s.tls_version
        if tls_version is None:
            tls_version = s.advertised_tls_version
        if tls_version >= 769:
            tmp_len = struct.pack('!H', len(enc))
        return tmp_len + enc + pay

    def guess_payload_class(self, p):
        if False:
            print('Hello World!')
        return Padding

class ClientPSKIdentity(Packet):
    """
    XXX We provide parsing abilities for ServerPSKParams, but the context
    operations have not been implemented yet. See RFC 4279.
    Note that we do not cover the (EC)DHE_PSK nor the RSA_PSK key exchange,
    which should contain either an EncryptedPMS or a ClientDiffieHellmanPublic.
    """
    name = 'Server PSK parameters'
    fields_desc = [FieldLenField('psk_identity_len', None, length_of='psk_identity', fmt='!H'), StrLenField('psk_identity', '', length_from=lambda pkt: pkt.psk_identity_len)]