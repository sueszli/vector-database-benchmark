"""
TLS base fields, used for record parsing/building. As several operations depend
upon the TLS version or ciphersuite, the packet has to provide a TLS context.
"""
import struct
from scapy.fields import ByteField, ShortEnumField, ShortField, StrField
from scapy.compat import orb
_tls_type = {20: 'change_cipher_spec', 21: 'alert', 22: 'handshake', 23: 'application_data'}
_tls_version = {2: 'SSLv2', 512: 'SSLv2', 768: 'SSLv3', 769: 'TLS 1.0', 770: 'TLS 1.1', 771: 'TLS 1.2', 32530: 'TLS 1.3-d18', 32531: 'TLS 1.3-d19', 772: 'TLS 1.3'}
_tls_version_options = {'sslv2': 2, 'sslv3': 768, 'tls1': 769, 'tls10': 769, 'tls11': 770, 'tls12': 771, 'tls13-d18': 32530, 'tls13-d19': 32531, 'tls13': 772}

def _tls13_version_filter(version, legacy_version):
    if False:
        i = 10
        return i + 15
    if version < 772:
        return version
    else:
        return legacy_version

class _TLSClientVersionField(ShortEnumField):
    """
    We use the advertised_tls_version if it has been defined,
    and the legacy 0x0303 for TLS 1.3 packets.
    """

    def i2h(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        if x is None:
            v = pkt.tls_session.advertised_tls_version
            if v:
                return _tls13_version_filter(v, 771)
            return ''
        return x

    def i2m(self, pkt, x):
        if False:
            print('Hello World!')
        if x is None:
            v = pkt.tls_session.advertised_tls_version
            if v:
                return _tls13_version_filter(v, 771)
            return b''
        return x

class _TLSVersionField(ShortEnumField):
    """
    We use the tls_version if it has been defined, else the advertised version.
    Also, the legacy 0x0301 is used for TLS 1.3 packets.
    """

    def i2h(self, pkt, x):
        if False:
            return 10
        if x is None:
            v = pkt.tls_session.tls_version
            if v:
                return _tls13_version_filter(v, 769)
            else:
                adv_v = pkt.tls_session.advertised_tls_version
                return _tls13_version_filter(adv_v, 769)
        return x

    def i2m(self, pkt, x):
        if False:
            i = 10
            return i + 15
        if x is None:
            v = pkt.tls_session.tls_version
            if v:
                return _tls13_version_filter(v, 769)
            else:
                adv_v = pkt.tls_session.advertised_tls_version
                return _tls13_version_filter(adv_v, 769)
        return x

class _TLSLengthField(ShortField):

    def i2repr(self, pkt, x):
        if False:
            return 10
        s = super(_TLSLengthField, self).i2repr(pkt, x)
        if pkt.deciphered_len is not None:
            dx = pkt.deciphered_len
            ds = super(_TLSLengthField, self).i2repr(pkt, dx)
            s += '    [deciphered_len= %s]' % ds
        return s

class _TLSIVField(StrField):
    """
    As stated in Section 6.2.3.2. RFC 4346, TLS 1.1 implements an explicit IV
    mechanism. For that reason, the behavior of the field is dependent on the
    TLS version found in the packet if available or otherwise (on build, if
    not overloaded, it is provided by the session). The size of the IV and
    its value are obviously provided by the session. As a side note, for the
    first packets exchanged by peers, NULL being the default enc alg, it is
    empty (except if forced to a specific value). Also note that the field is
    kept empty (unless forced to a specific value) when the cipher is a stream
    cipher (and NULL is considered a stream cipher).
    """

    def i2len(self, pkt, i):
        if False:
            print('Hello World!')
        if i is not None:
            return len(i)
        tmp_len = 0
        cipher_type = pkt.tls_session.rcs.cipher.type
        if cipher_type == 'block':
            if pkt.tls_session.tls_version >= 770:
                tmp_len = pkt.tls_session.rcs.cipher.block_size
        elif cipher_type == 'aead':
            tmp_len = pkt.tls_session.rcs.cipher.nonce_explicit_len
        return tmp_len

    def i2m(self, pkt, x):
        if False:
            for i in range(10):
                print('nop')
        return x or b''

    def addfield(self, pkt, s, val):
        if False:
            i = 10
            return i + 15
        return s + self.i2m(pkt, val)

    def getfield(self, pkt, s):
        if False:
            return 10
        tmp_len = 0
        cipher_type = pkt.tls_session.rcs.cipher.type
        if cipher_type == 'block':
            if pkt.tls_session.tls_version >= 770:
                tmp_len = pkt.tls_session.rcs.cipher.block_size
        elif cipher_type == 'aead':
            tmp_len = pkt.tls_session.rcs.cipher.nonce_explicit_len
        return (s[tmp_len:], self.m2i(pkt, s[:tmp_len]))

    def i2repr(self, pkt, x):
        if False:
            while True:
                i = 10
        return repr(self.i2m(pkt, x))

class _TLSMACField(StrField):

    def i2len(self, pkt, i):
        if False:
            i = 10
            return i + 15
        if i is not None:
            return len(i)
        return pkt.tls_session.wcs.mac_len

    def i2m(self, pkt, x):
        if False:
            i = 10
            return i + 15
        if x is None:
            return b''
        return x

    def addfield(self, pkt, s, val):
        if False:
            i = 10
            return i + 15
        return s

    def getfield(self, pkt, s):
        if False:
            for i in range(10):
                print('nop')
        if pkt.tls_session.rcs.cipher.type != 'aead' and False in pkt.tls_session.rcs.cipher.ready.values():
            return (s, b'')
        tmp_len = pkt.tls_session.rcs.mac_len
        return (s[tmp_len:], self.m2i(pkt, s[:tmp_len]))

    def i2repr(self, pkt, x):
        if False:
            return 10
        return repr(self.i2m(pkt, x))

class _TLSPadField(StrField):

    def i2len(self, pkt, i):
        if False:
            i = 10
            return i + 15
        if i is not None:
            return len(i)
        return 0

    def i2m(self, pkt, x):
        if False:
            return 10
        if x is None:
            return b''
        return x

    def addfield(self, pkt, s, val):
        if False:
            print('Hello World!')
        return s

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        if pkt.tls_session.consider_read_padding():
            tmp_len = orb(s[pkt.padlen - 1])
            return (s[tmp_len:], self.m2i(pkt, s[:tmp_len]))
        return (s, None)

    def i2repr(self, pkt, x):
        if False:
            while True:
                i = 10
        return repr(self.i2m(pkt, x))

class _TLSPadLenField(ByteField):

    def addfield(self, pkt, s, val):
        if False:
            print('Hello World!')
        return s

    def getfield(self, pkt, s):
        if False:
            for i in range(10):
                print('nop')
        if pkt.tls_session.consider_read_padding():
            return ByteField.getfield(self, pkt, s)
        return (s, None)

class _SSLv2LengthField(_TLSLengthField):

    def i2repr(self, pkt, x):
        if False:
            while True:
                i = 10
        s = super(_SSLv2LengthField, self).i2repr(pkt, x)
        if pkt.with_padding:
            x |= 32768
            s += '    [with padding: %s]' % hex(x)
        return s

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        msglen = struct.unpack('!H', s[:2])[0]
        pkt.with_padding = msglen & 32768 == 0
        if pkt.with_padding:
            msglen_clean = msglen & 16383
        else:
            msglen_clean = msglen & 32767
        return (s[2:], msglen_clean)

class _SSLv2MACField(_TLSMACField):
    pass

class _SSLv2PadField(_TLSPadField):

    def getfield(self, pkt, s):
        if False:
            print('Hello World!')
        if pkt.padlen is not None:
            tmp_len = pkt.padlen
            return (s[tmp_len:], self.m2i(pkt, s[:tmp_len]))
        return (s, None)

class _SSLv2PadLenField(_TLSPadLenField):

    def getfield(self, pkt, s):
        if False:
            return 10
        if pkt.with_padding:
            return ByteField.getfield(self, pkt, s)
        return (s, None)