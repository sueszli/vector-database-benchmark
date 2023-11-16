"""
IPsec layer
===========

Example of use:

>>> sa = SecurityAssociation(ESP, spi=0xdeadbeef, crypt_algo='AES-CBC',
...                          crypt_key=b'sixteenbytes key')
>>> p = IP(src='1.1.1.1', dst='2.2.2.2')
>>> p /= TCP(sport=45012, dport=80)
>>> p /= Raw('testdata')
>>> p = IP(raw(p))
>>> p
<IP  version=4L ihl=5L tos=0x0 len=48 id=1 flags= frag=0L ttl=64 proto=tcp chksum=0x74c2 src=1.1.1.1 dst=2.2.2.2 options=[] |<TCP  sport=45012 dport=http seq=0 ack=0 dataofs=5L reserved=0L flags=S window=8192 chksum=0x1914 urgptr=0 options=[] |<Raw  load='testdata' |>>>  # noqa: E501
>>>
>>> e = sa.encrypt(p)
>>> e
<IP  version=4L ihl=5L tos=0x0 len=76 id=1 flags= frag=0L ttl=64 proto=esp chksum=0x747a src=1.1.1.1 dst=2.2.2.2 |<ESP  spi=0xdeadbeef seq=1 data=b'\\xf8\\xdb\\x1e\\x83[T\\xab\\\\\\xd2\\x1b\\xed\\xd1\\xe5\\xc8Y\\xc2\\xa5d\\x92\\xc1\\x05\\x17\\xa6\\x92\\x831\\xe6\\xc1]\\x9a\\xd6K}W\\x8bFfd\\xa5B*+\\xde\\xc8\\x89\\xbf{\\xa9' |>>  # noqa: E501
>>>
>>> d = sa.decrypt(e)
>>> d
<IP  version=4L ihl=5L tos=0x0 len=48 id=1 flags= frag=0L ttl=64 proto=tcp chksum=0x74c2 src=1.1.1.1 dst=2.2.2.2 |<TCP  sport=45012 dport=http seq=0 ack=0 dataofs=5L reserved=0L flags=S window=8192 chksum=0x1914 urgptr=0 options=[] |<Raw  load='testdata' |>>>  # noqa: E501
>>>
>>> d == p
True
"""
try:
    from math import gcd
except ImportError:
    from fractions import gcd
import os
import socket
import struct
import warnings
from scapy.config import conf, crypto_validator
from scapy.compat import orb, raw
from scapy.data import IP_PROTOS
from scapy.error import log_loading
from scapy.fields import ByteEnumField, ByteField, IntField, PacketField, ShortField, StrField, XByteField, XIntField, XStrField, XStrLenField
from scapy.packet import Packet, Raw, bind_bottom_up, bind_layers, bind_top_down
from scapy.layers.inet import IP, UDP
from scapy.layers.inet6 import IPv6, IPv6ExtHdrHopByHop, IPv6ExtHdrDestOpt, IPv6ExtHdrRouting

class AH(Packet):
    """
    Authentication Header

    See https://tools.ietf.org/rfc/rfc4302.txt
    """
    name = 'AH'

    def __get_icv_len(self):
        if False:
            return 10
        '\n        Compute the size of the ICV based on the payloadlen field.\n        Padding size is included as it can only be known from the authentication  # noqa: E501\n        algorithm provided by the Security Association.\n        '
        return (self.payloadlen - 1) * 4
    fields_desc = [ByteEnumField('nh', None, IP_PROTOS), ByteField('payloadlen', None), ShortField('reserved', None), XIntField('spi', 1), IntField('seq', 0), XStrLenField('icv', None, length_from=__get_icv_len), XStrLenField('padding', None, length_from=lambda x: 0)]
    overload_fields = {IP: {'proto': socket.IPPROTO_AH}, IPv6: {'nh': socket.IPPROTO_AH}, IPv6ExtHdrHopByHop: {'nh': socket.IPPROTO_AH}, IPv6ExtHdrDestOpt: {'nh': socket.IPPROTO_AH}, IPv6ExtHdrRouting: {'nh': socket.IPPROTO_AH}}
bind_layers(IP, AH, proto=socket.IPPROTO_AH)
bind_layers(IPv6, AH, nh=socket.IPPROTO_AH)
bind_layers(AH, IP, nh=socket.IPPROTO_IP)
bind_layers(AH, IPv6, nh=socket.IPPROTO_IPV6)

class ESP(Packet):
    """
    Encapsulated Security Payload

    See https://tools.ietf.org/rfc/rfc4303.txt
    """
    name = 'ESP'
    fields_desc = [XIntField('spi', 1), IntField('seq', 0), XStrField('data', None)]

    @classmethod
    def dispatch_hook(cls, _pkt=None, *args, **kargs):
        if False:
            while True:
                i = 10
        if _pkt:
            if len(_pkt) >= 4 and struct.unpack('!I', _pkt[0:4])[0] == 0:
                return NON_ESP
            elif len(_pkt) == 1 and struct.unpack('!B', _pkt)[0] == 255:
                return NAT_KEEPALIVE
            else:
                return ESP
        return cls
    overload_fields = {IP: {'proto': socket.IPPROTO_ESP}, IPv6: {'nh': socket.IPPROTO_ESP}, IPv6ExtHdrHopByHop: {'nh': socket.IPPROTO_ESP}, IPv6ExtHdrDestOpt: {'nh': socket.IPPROTO_ESP}, IPv6ExtHdrRouting: {'nh': socket.IPPROTO_ESP}}

class NON_ESP(Packet):
    fields_desc = [XIntField('non_esp', 0)]

class NAT_KEEPALIVE(Packet):
    fields_desc = [XByteField('nat_keepalive', 255)]
bind_layers(IP, ESP, proto=socket.IPPROTO_ESP)
bind_layers(IPv6, ESP, nh=socket.IPPROTO_ESP)
bind_bottom_up(UDP, ESP, dport=4500)
bind_bottom_up(UDP, ESP, sport=4500)
bind_top_down(UDP, ESP, dport=4500, sport=4500)
bind_top_down(UDP, NON_ESP, dport=4500, sport=4500)
bind_top_down(UDP, NAT_KEEPALIVE, dport=4500, sport=4500)

class _ESPPlain(Packet):
    """
    Internal class to represent unencrypted ESP packets.
    """
    name = 'ESP'
    fields_desc = [XIntField('spi', 0), IntField('seq', 0), StrField('iv', ''), PacketField('data', '', Raw), StrField('padding', ''), ByteField('padlen', 0), ByteEnumField('nh', 0, IP_PROTOS), StrField('icv', '')]

    def data_for_encryption(self):
        if False:
            i = 10
            return i + 15
        return raw(self.data) + self.padding + struct.pack('BB', self.padlen, self.nh)
if conf.crypto_valid:
    from cryptography.exceptions import InvalidTag
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import aead, Cipher, algorithms, modes
else:
    log_loading.info("Can't import python-cryptography v1.7+. Disabled IPsec encryption/authentication.")
    default_backend = None
    InvalidTag = Exception
    Cipher = algorithms = modes = None

def _lcm(a, b):
    if False:
        for i in range(10):
            print('nop')
    '\n    Least Common Multiple between 2 integers.\n    '
    if a == 0 or b == 0:
        return 0
    else:
        return abs(a * b) // gcd(a, b)

class CryptAlgo(object):
    """
    IPsec encryption algorithm
    """

    def __init__(self, name, cipher, mode, block_size=None, iv_size=None, key_size=None, icv_size=None, salt_size=None, format_mode_iv=None):
        if False:
            while True:
                i = 10
        '\n        :param name: the name of this encryption algorithm\n        :param cipher: a Cipher module\n        :param mode: the mode used with the cipher module\n        :param block_size: the length a block for this algo. Defaults to the\n                           `block_size` of the cipher.\n        :param iv_size: the length of the initialization vector of this algo.\n                        Defaults to the `block_size` of the cipher.\n        :param key_size: an integer or list/tuple of integers. If specified,\n                         force the secret keys length to one of the values.\n                         Defaults to the `key_size` of the cipher.\n        :param icv_size: the length of the Integrity Check Value of this algo.\n                         Used by Combined Mode Algorithms e.g. GCM\n        :param salt_size: the length of the salt to use as the IV prefix.\n                          Usually used by Counter modes e.g. CTR\n        :param format_mode_iv: function to format the Initialization Vector\n                               e.g. handle the salt value\n                               Default is the random buffer from `generate_iv`\n        '
        self.name = name
        self.cipher = cipher
        self.mode = mode
        self.icv_size = icv_size
        self.is_aead = False
        self.ciphers_aead_api = False
        if modes:
            if self.mode is not None:
                self.is_aead = issubclass(self.mode, modes.ModeWithAuthenticationTag)
            elif self.cipher in (aead.AESGCM, aead.AESCCM, aead.ChaCha20Poly1305):
                self.is_aead = True
                self.ciphers_aead_api = True
        if block_size is not None:
            self.block_size = block_size
        elif cipher is not None:
            self.block_size = cipher.block_size // 8
        else:
            self.block_size = 1
        if iv_size is None:
            self.iv_size = self.block_size
        else:
            self.iv_size = iv_size
        if key_size is not None:
            self.key_size = key_size
        elif cipher is not None:
            self.key_size = tuple((i // 8 for i in cipher.key_sizes))
        else:
            self.key_size = None
        if salt_size is None:
            self.salt_size = 0
        else:
            self.salt_size = salt_size
        if format_mode_iv is None:
            self._format_mode_iv = lambda iv, **kw: iv
        else:
            self._format_mode_iv = format_mode_iv

    def check_key(self, key):
        if False:
            print('Hello World!')
        '\n        Check that the key length is valid.\n\n        :param key:    a byte string\n        '
        if self.key_size and (not (len(key) == self.key_size or len(key) in self.key_size)):
            raise TypeError('invalid key size %s, must be %s' % (len(key), self.key_size))

    def generate_iv(self):
        if False:
            return 10
        '\n        Generate a random initialization vector.\n        '
        return os.urandom(self.iv_size)

    @crypto_validator
    def new_cipher(self, key, mode_iv, digest=None):
        if False:
            print('Hello World!')
        '\n        :param key:     the secret key, a byte string\n        :param mode_iv: the initialization vector or nonce, a byte string.\n                        Formatted by `format_mode_iv`.\n        :param digest:  also known as tag or icv. A byte string containing the\n                        digest of the encrypted data. Only use this during\n                        decryption!\n\n        :returns:    an initialized cipher object for this algo\n        '
        if self.is_aead and digest is not None:
            return Cipher(self.cipher(key), self.mode(mode_iv, digest, len(digest)), default_backend())
        else:
            return Cipher(self.cipher(key), self.mode(mode_iv), default_backend())

    def pad(self, esp):
        if False:
            while True:
                i = 10
        "\n        Add the correct amount of padding so that the data to encrypt is\n        exactly a multiple of the algorithm's block size.\n\n        Also, make sure that the total ESP packet length is a multiple of 4\n        bytes.\n\n        :param esp:    an unencrypted _ESPPlain packet\n\n        :returns:    an unencrypted _ESPPlain packet with valid padding\n        "
        data_len = len(esp.data) + 2
        align = _lcm(self.block_size, 4)
        esp.padlen = -data_len % align
        esp.padding = struct.pack('B' * esp.padlen, *range(1, esp.padlen + 1))
        payload_len = len(esp.iv) + len(esp.data) + len(esp.padding) + 2
        if payload_len % 4 != 0:
            raise ValueError('The size of the ESP data is not aligned to 32 bits after padding.')
        return esp

    def encrypt(self, sa, esp, key, icv_size=None, esn_en=False, esn=0):
        if False:
            print('Hello World!')
        '\n        Encrypt an ESP packet\n\n        :param sa:   the SecurityAssociation associated with the ESP packet.\n        :param esp:  an unencrypted _ESPPlain packet with valid padding\n        :param key:  the secret key used for encryption\n        :param icv_size: the length of the icv used for integrity check\n        :esn_en:     extended sequence number enable which allows to use 64-bit\n                     sequence number instead of 32-bit when using an AEAD\n                     algorithm\n        :esn:        extended sequence number (32 MSB)\n        :return:    a valid ESP packet encrypted with this algorithm\n        '
        if icv_size is None:
            icv_size = self.icv_size if self.is_aead else 0
        data = esp.data_for_encryption()
        if self.cipher:
            mode_iv = self._format_mode_iv(algo=self, sa=sa, iv=esp.iv)
            aad = None
            if self.is_aead:
                if esn_en:
                    aad = struct.pack('!LLL', esp.spi, esn, esp.seq)
                else:
                    aad = struct.pack('!LL', esp.spi, esp.seq)
            if self.ciphers_aead_api:
                if self.cipher == aead.AESCCM:
                    cipher = self.cipher(key, tag_length=icv_size)
                else:
                    cipher = self.cipher(key)
                if self.name == 'AES-NULL-GMAC':
                    data = data + cipher.encrypt(mode_iv, b'', aad + esp.iv + data)
                else:
                    data = cipher.encrypt(mode_iv, data, aad)
            else:
                cipher = self.new_cipher(key, mode_iv)
                encryptor = cipher.encryptor()
                if self.is_aead:
                    encryptor.authenticate_additional_data(aad)
                    data = encryptor.update(data) + encryptor.finalize()
                    data += encryptor.tag[:icv_size]
                else:
                    data = encryptor.update(data) + encryptor.finalize()
        return ESP(spi=esp.spi, seq=esp.seq, data=esp.iv + data)

    def decrypt(self, sa, esp, key, icv_size=None, esn_en=False, esn=0):
        if False:
            return 10
        '\n        Decrypt an ESP packet\n\n        :param sa: the SecurityAssociation associated with the ESP packet.\n        :param esp: an encrypted ESP packet\n        :param key: the secret key used for encryption\n        :param icv_size: the length of the icv used for integrity check\n        :param esn_en: extended sequence number enable which allows to use\n                       64-bit sequence number instead of 32-bit when using an\n                       AEAD algorithm\n        :param esn: extended sequence number (32 MSB)\n        :returns: a valid ESP packet encrypted with this algorithm\n        :raise scapy.layers.ipsec.IPSecIntegrityError: if the integrity check\n            fails with an AEAD algorithm\n        '
        if icv_size is None:
            icv_size = self.icv_size if self.is_aead else 0
        iv = esp.data[:self.iv_size]
        data = esp.data[self.iv_size:len(esp.data) - icv_size]
        icv = esp.data[len(esp.data) - icv_size:]
        if self.cipher:
            mode_iv = self._format_mode_iv(sa=sa, iv=iv)
            aad = None
            if self.is_aead:
                if esn_en:
                    aad = struct.pack('!LLL', esp.spi, esn, esp.seq)
                else:
                    aad = struct.pack('!LL', esp.spi, esp.seq)
            if self.ciphers_aead_api:
                if self.cipher == aead.AESCCM:
                    cipher = self.cipher(key, tag_length=icv_size)
                else:
                    cipher = self.cipher(key)
                try:
                    if self.name == 'AES-NULL-GMAC':
                        data = data + cipher.decrypt(mode_iv, icv, aad + iv + data)
                    else:
                        data = cipher.decrypt(mode_iv, data + icv, aad)
                except InvalidTag as err:
                    raise IPSecIntegrityError(err)
            else:
                cipher = self.new_cipher(key, mode_iv, icv)
                decryptor = cipher.decryptor()
                if self.is_aead:
                    decryptor.authenticate_additional_data(aad)
                try:
                    data = decryptor.update(data) + decryptor.finalize()
                except InvalidTag as err:
                    raise IPSecIntegrityError(err)
        padlen = orb(data[-2])
        nh = orb(data[-1])
        data = data[:len(data) - padlen - 2]
        padding = data[len(data) - padlen - 2:len(data) - 2]
        return _ESPPlain(spi=esp.spi, seq=esp.seq, iv=iv, data=data, padding=padding, padlen=padlen, nh=nh, icv=icv)
CRYPT_ALGOS = {'NULL': CryptAlgo('NULL', cipher=None, mode=None, iv_size=0)}
if algorithms:
    CRYPT_ALGOS['AES-CBC'] = CryptAlgo('AES-CBC', cipher=algorithms.AES, mode=modes.CBC)
    _aes_ctr_format_mode_iv = lambda sa, iv, **kw: sa.crypt_salt + iv + b'\x00\x00\x00\x01'
    CRYPT_ALGOS['AES-CTR'] = CryptAlgo('AES-CTR', cipher=algorithms.AES, mode=modes.CTR, block_size=1, iv_size=8, salt_size=4, format_mode_iv=_aes_ctr_format_mode_iv)
    _salt_format_mode_iv = lambda sa, iv, **kw: sa.crypt_salt + iv
    CRYPT_ALGOS['AES-GCM'] = CryptAlgo('AES-GCM', cipher=aead.AESGCM, key_size=(16, 24, 32), mode=None, salt_size=4, block_size=1, iv_size=8, icv_size=16, format_mode_iv=_salt_format_mode_iv)
    CRYPT_ALGOS['AES-NULL-GMAC'] = CryptAlgo('AES-NULL-GMAC', cipher=aead.AESGCM, key_size=(16, 24, 32), mode=None, salt_size=4, block_size=1, iv_size=8, icv_size=16, format_mode_iv=_salt_format_mode_iv)
    CRYPT_ALGOS['AES-CCM'] = CryptAlgo('AES-CCM', cipher=aead.AESCCM, mode=None, key_size=(16, 24, 32), block_size=1, iv_size=8, salt_size=3, icv_size=16, format_mode_iv=_salt_format_mode_iv)
    CRYPT_ALGOS['CHACHA20-POLY1305'] = CryptAlgo('CHACHA20-POLY1305', cipher=aead.ChaCha20Poly1305, mode=None, key_size=32, block_size=1, iv_size=8, salt_size=4, icv_size=16, format_mode_iv=_salt_format_mode_iv)
    CRYPT_ALGOS['DES'] = CryptAlgo('DES', cipher=algorithms.TripleDES, mode=modes.CBC, key_size=(8,))
    CRYPT_ALGOS['3DES'] = CryptAlgo('3DES', cipher=algorithms.TripleDES, mode=modes.CBC)
    try:
        from cryptography.utils import CryptographyDeprecationWarning
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=CryptographyDeprecationWarning)
            CRYPT_ALGOS['CAST'] = CryptAlgo('CAST', cipher=algorithms.CAST5, mode=modes.CBC)
            CRYPT_ALGOS['Blowfish'] = CryptAlgo('Blowfish', cipher=algorithms.Blowfish, mode=modes.CBC)
    except AttributeError:
        pass
if conf.crypto_valid:
    from cryptography.hazmat.primitives.hmac import HMAC
    from cryptography.hazmat.primitives.cmac import CMAC
    from cryptography.hazmat.primitives import hashes
else:
    HMAC = CMAC = hashes = None

class IPSecIntegrityError(Exception):
    """
    Error risen when the integrity check fails.
    """
    pass

class AuthAlgo(object):
    """
    IPsec integrity algorithm
    """

    def __init__(self, name, mac, digestmod, icv_size, key_size=None):
        if False:
            print('Hello World!')
        '\n        :param name: the name of this integrity algorithm\n        :param mac: a Message Authentication Code module\n        :param digestmod: a Hash or Cipher module\n        :param icv_size: the length of the integrity check value of this algo\n        :param key_size: an integer or list/tuple of integers. If specified,\n                         force the secret keys length to one of the values.\n                         Defaults to the `key_size` of the cipher.\n        '
        self.name = name
        self.mac = mac
        self.digestmod = digestmod
        self.icv_size = icv_size
        self.key_size = key_size

    def check_key(self, key):
        if False:
            return 10
        '\n        Check that the key length is valid.\n\n        :param key:    a byte string\n        '
        if self.key_size and len(key) not in self.key_size:
            raise TypeError('invalid key size %s, must be one of %s' % (len(key), self.key_size))

    @crypto_validator
    def new_mac(self, key):
        if False:
            return 10
        '\n        :param key:    a byte string\n        :returns:       an initialized mac object for this algo\n        '
        if self.mac is CMAC:
            return self.mac(self.digestmod(key), default_backend())
        else:
            return self.mac(key, self.digestmod(), default_backend())

    def sign(self, pkt, key, esn_en=False, esn=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sign an IPsec (ESP or AH) packet with this algo.\n\n        :param pkt:    a packet that contains a valid encrypted ESP or AH layer\n        :param key:    the authentication key, a byte string\n        :param esn_en: extended sequence number enable which allows to use\n                       64-bit sequence number instead of 32-bit\n        :param esn: extended sequence number (32 MSB)\n\n        :returns: the signed packet\n        '
        if not self.mac:
            return pkt
        mac = self.new_mac(key)
        if pkt.haslayer(ESP):
            mac.update(raw(pkt[ESP]))
            pkt[ESP].data += mac.finalize()[:self.icv_size]
        elif pkt.haslayer(AH):
            clone = zero_mutable_fields(pkt.copy(), sending=True)
            if esn_en:
                temp = raw(clone) + struct.pack('!L', esn)
            else:
                temp = raw(clone)
            mac.update(temp)
            pkt[AH].icv = mac.finalize()[:self.icv_size]
        return pkt

    def verify(self, pkt, key, esn_en=False, esn=0):
        if False:
            for i in range(10):
                print('nop')
        '\n        Check that the integrity check value (icv) of a packet is valid.\n\n        :param pkt:    a packet that contains a valid encrypted ESP or AH layer\n        :param key:    the authentication key, a byte string\n        :param esn_en: extended sequence number enable which allows to use\n                       64-bit sequence number instead of 32-bit\n        :param esn: extended sequence number (32 MSB)\n\n        :raise scapy.layers.ipsec.IPSecIntegrityError: if the integrity check\n            fails\n        '
        if not self.mac or self.icv_size == 0:
            return
        mac = self.new_mac(key)
        pkt_icv = 'not found'
        if isinstance(pkt, ESP):
            pkt_icv = pkt.data[len(pkt.data) - self.icv_size:]
            clone = pkt.copy()
            clone.data = clone.data[:len(clone.data) - self.icv_size]
            temp = raw(clone)
        elif pkt.haslayer(AH):
            if len(pkt[AH].icv) != self.icv_size:
                pkt[AH].padding = pkt[AH].icv[self.icv_size:]
                pkt[AH].icv = pkt[AH].icv[:self.icv_size]
            pkt_icv = pkt[AH].icv
            clone = zero_mutable_fields(pkt.copy(), sending=False)
            if esn_en:
                temp = raw(clone) + struct.pack('!L', esn)
            else:
                temp = raw(clone)
        mac.update(temp)
        computed_icv = mac.finalize()[:self.icv_size]
        if pkt_icv != computed_icv:
            raise IPSecIntegrityError('pkt_icv=%r, computed_icv=%r' % (pkt_icv, computed_icv))
AUTH_ALGOS = {'NULL': AuthAlgo('NULL', mac=None, digestmod=None, icv_size=0)}
if HMAC and hashes:
    AUTH_ALGOS['HMAC-SHA1-96'] = AuthAlgo('HMAC-SHA1-96', mac=HMAC, digestmod=hashes.SHA1, icv_size=12)
    AUTH_ALGOS['SHA2-256-128'] = AuthAlgo('SHA2-256-128', mac=HMAC, digestmod=hashes.SHA256, icv_size=16)
    AUTH_ALGOS['SHA2-384-192'] = AuthAlgo('SHA2-384-192', mac=HMAC, digestmod=hashes.SHA384, icv_size=24)
    AUTH_ALGOS['SHA2-512-256'] = AuthAlgo('SHA2-512-256', mac=HMAC, digestmod=hashes.SHA512, icv_size=32)
    AUTH_ALGOS['HMAC-MD5-96'] = AuthAlgo('HMAC-MD5-96', mac=HMAC, digestmod=hashes.MD5, icv_size=12)
if CMAC and algorithms:
    AUTH_ALGOS['AES-CMAC-96'] = AuthAlgo('AES-CMAC-96', mac=CMAC, digestmod=algorithms.AES, icv_size=12, key_size=(16,))

def split_for_transport(orig_pkt, transport_proto):
    if False:
        print('Hello World!')
    '\n    Split an IP(v6) packet in the correct location to insert an ESP or AH\n    header.\n\n    :param orig_pkt: the packet to split. Must be an IP or IPv6 packet\n    :param transport_proto: the IPsec protocol number that will be inserted\n                            at the split position.\n    :returns: a tuple (header, nh, payload) where nh is the protocol number of\n             payload.\n    '
    header = orig_pkt.__class__(raw(orig_pkt))
    next_hdr = header.payload
    nh = None
    if header.version == 4:
        nh = header.proto
        header.proto = transport_proto
        header.remove_payload()
        del header.chksum
        del header.len
        return (header, nh, next_hdr)
    else:
        found_rt_hdr = False
        prev = header
        while isinstance(next_hdr, (IPv6ExtHdrHopByHop, IPv6ExtHdrRouting, IPv6ExtHdrDestOpt)):
            if isinstance(next_hdr, IPv6ExtHdrHopByHop):
                pass
            if isinstance(next_hdr, IPv6ExtHdrRouting):
                found_rt_hdr = True
            elif isinstance(next_hdr, IPv6ExtHdrDestOpt) and found_rt_hdr:
                break
            prev = next_hdr
            next_hdr = next_hdr.payload
        nh = prev.nh
        prev.nh = transport_proto
        prev.remove_payload()
        del header.plen
        return (header, nh, next_hdr)
IMMUTABLE_IPV4_OPTIONS = (0, 1, 2, 5, 6, 20, 21)

def zero_mutable_fields(pkt, sending=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    When using AH, all "mutable" fields must be "zeroed" before calculating\n    the ICV. See RFC 4302, Section 3.3.3.1. Handling Mutable Fields.\n\n    :param pkt: an IP(v6) packet containing an AH layer.\n                NOTE: The packet will be modified\n    :param sending: if true, ipv6 routing headers will not be reordered\n    '
    if pkt.haslayer(AH):
        pkt[AH].icv = b'\x00' * len(pkt[AH].icv)
    else:
        raise TypeError('no AH layer found')
    if pkt.version == 4:
        pkt.tos = 0
        pkt.flags = 0
        pkt.ttl = 0
        pkt.chksum = 0
        immutable_opts = []
        for opt in pkt.options:
            if opt.option in IMMUTABLE_IPV4_OPTIONS:
                immutable_opts.append(opt)
            else:
                immutable_opts.append(Raw(b'\x00' * len(opt)))
        pkt.options = immutable_opts
    else:
        pkt.tc = 0
        pkt.fl = 0
        pkt.hlim = 0
        next_hdr = pkt.payload
        while isinstance(next_hdr, (IPv6ExtHdrHopByHop, IPv6ExtHdrRouting, IPv6ExtHdrDestOpt)):
            if isinstance(next_hdr, (IPv6ExtHdrHopByHop, IPv6ExtHdrDestOpt)):
                for opt in next_hdr.options:
                    if opt.otype & 32:
                        opt.optdata = b'\x00' * opt.optlen
            elif isinstance(next_hdr, IPv6ExtHdrRouting) and sending:
                next_hdr.segleft = 0
                if next_hdr.addresses:
                    final = next_hdr.addresses.pop()
                    next_hdr.addresses.insert(0, pkt.dst)
                    pkt.dst = final
            else:
                break
            next_hdr = next_hdr.payload
    return pkt

class SecurityAssociation(object):
    """
    This class is responsible of "encryption" and "decryption" of IPsec packets.  # noqa: E501
    """
    SUPPORTED_PROTOS = (IP, IPv6)

    def __init__(self, proto, spi, seq_num=1, crypt_algo=None, crypt_key=None, crypt_icv_size=None, auth_algo=None, auth_key=None, tunnel_header=None, nat_t_header=None, esn_en=False, esn=0):
        if False:
            return 10
        '\n        :param proto: the IPsec proto to use (ESP or AH)\n        :param spi: the Security Parameters Index of this SA\n        :param seq_num: the initial value for the sequence number on encrypted\n                        packets\n        :param crypt_algo: the encryption algorithm name (only used with ESP)\n        :param crypt_key: the encryption key (only used with ESP)\n        :param crypt_icv_size: change the default size of the crypt_algo\n                               (only used with ESP)\n        :param auth_algo: the integrity algorithm name\n        :param auth_key: the integrity key\n        :param tunnel_header: an instance of a IP(v6) header that will be used\n                              to encapsulate the encrypted packets.\n        :param nat_t_header: an instance of a UDP header that will be used\n                             for NAT-Traversal.\n        :param esn_en: extended sequence number enable which allows to use\n                       64-bit sequence number instead of 32-bit when using an\n                       AEAD algorithm\n        :param esn: extended sequence number (32 MSB)\n        '
        if proto not in {ESP, AH, ESP.name, AH.name}:
            raise ValueError('proto must be either ESP or AH')
        if isinstance(proto, str):
            self.proto = {ESP.name: ESP, AH.name: AH}[proto]
        else:
            self.proto = proto
        self.spi = spi
        self.seq_num = seq_num
        self.esn_en = esn_en
        self.esn = esn
        if crypt_algo:
            if crypt_algo not in CRYPT_ALGOS:
                raise TypeError('unsupported encryption algo %r, try %r' % (crypt_algo, list(CRYPT_ALGOS)))
            self.crypt_algo = CRYPT_ALGOS[crypt_algo]
            if crypt_key:
                salt_size = self.crypt_algo.salt_size
                self.crypt_key = crypt_key[:len(crypt_key) - salt_size]
                self.crypt_salt = crypt_key[len(crypt_key) - salt_size:]
            else:
                self.crypt_key = None
                self.crypt_salt = None
        else:
            self.crypt_algo = CRYPT_ALGOS['NULL']
            self.crypt_key = None
            self.crypt_salt = None
        self.crypt_icv_size = crypt_icv_size
        if auth_algo:
            if auth_algo not in AUTH_ALGOS:
                raise TypeError('unsupported integrity algo %r, try %r' % (auth_algo, list(AUTH_ALGOS)))
            self.auth_algo = AUTH_ALGOS[auth_algo]
            self.auth_key = auth_key
        else:
            self.auth_algo = AUTH_ALGOS['NULL']
            self.auth_key = None
        if tunnel_header and (not isinstance(tunnel_header, (IP, IPv6))):
            raise TypeError('tunnel_header must be %s or %s' % (IP.name, IPv6.name))
        self.tunnel_header = tunnel_header
        if nat_t_header:
            if proto is not ESP:
                raise TypeError('nat_t_header is only allowed with ESP')
            if not isinstance(nat_t_header, UDP):
                raise TypeError('nat_t_header must be %s' % UDP.name)
        self.nat_t_header = nat_t_header

    def check_spi(self, pkt):
        if False:
            return 10
        if pkt.spi != self.spi:
            raise TypeError('packet spi=0x%x does not match the SA spi=0x%x' % (pkt.spi, self.spi))

    def _encrypt_esp(self, pkt, seq_num=None, iv=None, esn_en=None, esn=None):
        if False:
            i = 10
            return i + 15
        if iv is None:
            iv = self.crypt_algo.generate_iv()
        elif len(iv) != self.crypt_algo.iv_size:
            raise TypeError('iv length must be %s' % self.crypt_algo.iv_size)
        esp = _ESPPlain(spi=self.spi, seq=seq_num or self.seq_num, iv=iv)
        if self.tunnel_header:
            tunnel = self.tunnel_header.copy()
            if tunnel.version == 4:
                del tunnel.proto
                del tunnel.len
                del tunnel.chksum
            else:
                del tunnel.nh
                del tunnel.plen
            pkt = tunnel.__class__(raw(tunnel / pkt))
        (ip_header, nh, payload) = split_for_transport(pkt, socket.IPPROTO_ESP)
        esp.data = payload
        esp.nh = nh
        esp = self.crypt_algo.pad(esp)
        esp = self.crypt_algo.encrypt(self, esp, self.crypt_key, self.crypt_icv_size, esn_en=esn_en or self.esn_en, esn=esn or self.esn)
        self.auth_algo.sign(esp, self.auth_key)
        if self.nat_t_header:
            nat_t_header = self.nat_t_header.copy()
            nat_t_header.chksum = 0
            del nat_t_header.len
            if ip_header.version == 4:
                del ip_header.proto
            else:
                del ip_header.nh
            ip_header /= nat_t_header
        if ip_header.version == 4:
            del ip_header.len
            del ip_header.chksum
        else:
            del ip_header.plen
        if seq_num is None:
            self.seq_num += 1
        return ip_header.__class__(raw(ip_header / esp))

    def _encrypt_ah(self, pkt, seq_num=None, esn_en=False, esn=0):
        if False:
            while True:
                i = 10
        ah = AH(spi=self.spi, seq=seq_num or self.seq_num, icv=b'\x00' * self.auth_algo.icv_size)
        if self.tunnel_header:
            tunnel = self.tunnel_header.copy()
            if tunnel.version == 4:
                del tunnel.proto
                del tunnel.len
                del tunnel.chksum
            else:
                del tunnel.nh
                del tunnel.plen
            pkt = tunnel.__class__(raw(tunnel / pkt))
        (ip_header, nh, payload) = split_for_transport(pkt, socket.IPPROTO_AH)
        ah.nh = nh
        if ip_header.version == 6 and len(ah) % 8 != 0:
            ah.padding = b'\x00' * (-len(ah) % 8)
        elif len(ah) % 4 != 0:
            ah.padding = b'\x00' * (-len(ah) % 4)
        ah.payloadlen = len(ah) // 4 - 2
        if ip_header.version == 4:
            ip_header.len = len(ip_header) + len(ah) + len(payload)
            del ip_header.chksum
            ip_header = ip_header.__class__(raw(ip_header))
        else:
            ip_header.plen = len(ip_header.payload) + len(ah) + len(payload)
        signed_pkt = self.auth_algo.sign(ip_header / ah / payload, self.auth_key, esn_en=esn_en or self.esn_en, esn=esn or self.esn)
        if seq_num is None:
            self.seq_num += 1
        return signed_pkt

    def encrypt(self, pkt, seq_num=None, iv=None, esn_en=None, esn=None):
        if False:
            return 10
        '\n        Encrypt (and encapsulate) an IP(v6) packet with ESP or AH according\n        to this SecurityAssociation.\n\n        :param pkt:     the packet to encrypt\n        :param seq_num: if specified, use this sequence number instead of the\n                        generated one\n        :param esn_en:  extended sequence number enable which allows to\n                        use 64-bit sequence number instead of 32-bit when\n                        using an AEAD algorithm\n        :param esn:     extended sequence number (32 MSB)\n        :param iv:      if specified, use this initialization vector for\n                        encryption instead of a random one.\n\n        :returns: the encrypted/encapsulated packet\n        '
        if not isinstance(pkt, self.SUPPORTED_PROTOS):
            raise TypeError('cannot encrypt %s, supported protos are %s' % (pkt.__class__, self.SUPPORTED_PROTOS))
        if self.proto is ESP:
            return self._encrypt_esp(pkt, seq_num=seq_num, iv=iv, esn_en=esn_en, esn=esn)
        else:
            return self._encrypt_ah(pkt, seq_num=seq_num, esn_en=esn_en, esn=esn)

    def _decrypt_esp(self, pkt, verify=True, esn_en=None, esn=None):
        if False:
            i = 10
            return i + 15
        encrypted = pkt[ESP]
        if verify:
            self.check_spi(pkt)
            self.auth_algo.verify(encrypted, self.auth_key)
        esp = self.crypt_algo.decrypt(self, encrypted, self.crypt_key, self.crypt_icv_size or self.crypt_algo.icv_size or self.auth_algo.icv_size, esn_en=esn_en or self.esn_en, esn=esn or self.esn)
        if self.tunnel_header:
            pkt.remove_payload()
            if pkt.version == 4:
                pkt.proto = esp.nh
            else:
                pkt.nh = esp.nh
            cls = pkt.guess_payload_class(esp.data)
            return cls(esp.data)
        else:
            ip_header = pkt
            if ip_header.version == 4:
                ip_header.proto = esp.nh
                del ip_header.chksum
                ip_header.remove_payload()
                ip_header.len = len(ip_header) + len(esp.data)
                ip_header = ip_header.__class__(raw(ip_header))
            else:
                encrypted.underlayer.nh = esp.nh
                encrypted.underlayer.remove_payload()
                ip_header.plen = len(ip_header.payload) + len(esp.data)
            cls = ip_header.guess_payload_class(esp.data)
            return ip_header / cls(esp.data)

    def _decrypt_ah(self, pkt, verify=True, esn_en=None, esn=None):
        if False:
            return 10
        if verify:
            self.check_spi(pkt)
            self.auth_algo.verify(pkt, self.auth_key, esn_en=esn_en or self.esn_en, esn=esn or self.esn)
        ah = pkt[AH]
        payload = ah.payload
        payload.remove_underlayer(None)
        if self.tunnel_header:
            return payload
        else:
            ip_header = pkt
            if ip_header.version == 4:
                ip_header.proto = ah.nh
                del ip_header.chksum
                ip_header.remove_payload()
                ip_header.len = len(ip_header) + len(payload)
                ip_header = ip_header.__class__(raw(ip_header))
            else:
                ah.underlayer.nh = ah.nh
                ah.underlayer.remove_payload()
                ip_header.plen = len(ip_header.payload) + len(payload)
            return ip_header / payload

    def decrypt(self, pkt, verify=True, esn_en=None, esn=None):
        if False:
            i = 10
            return i + 15
        '\n        Decrypt (and decapsulate) an IP(v6) packet containing ESP or AH.\n\n        :param pkt:     the packet to decrypt\n        :param verify:  if False, do not perform the integrity check\n        :param esn_en:  extended sequence number enable which allows to use\n                        64-bit sequence number instead of 32-bit when using an\n                        AEAD algorithm\n        :param esn:        extended sequence number (32 MSB)\n        :returns: the decrypted/decapsulated packet\n        :raise scapy.layers.ipsec.IPSecIntegrityError: if the integrity check\n            fails\n        '
        if not isinstance(pkt, self.SUPPORTED_PROTOS):
            raise TypeError('cannot decrypt %s, supported protos are %s' % (pkt.__class__, self.SUPPORTED_PROTOS))
        if self.proto is ESP and pkt.haslayer(ESP):
            return self._decrypt_esp(pkt, verify=verify, esn_en=esn_en, esn=esn)
        elif self.proto is AH and pkt.haslayer(AH):
            return self._decrypt_ah(pkt, verify=verify, esn_en=esn_en, esn=esn)
        else:
            raise TypeError('%s has no %s layer' % (pkt, self.proto.name))