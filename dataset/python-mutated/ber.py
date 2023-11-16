"""
Basic Encoding Rules (BER) for ASN.1
"""
from scapy.error import warning
from scapy.compat import chb, orb, bytes_encode
from scapy.utils import binrepr, inet_aton, inet_ntoa
from scapy.asn1.asn1 import ASN1_BADTAG, ASN1_BadTag_Decoding_Error, ASN1_Class, ASN1_Class_UNIVERSAL, ASN1_Codecs, ASN1_DECODING_ERROR, ASN1_Decoding_Error, ASN1_Encoding_Error, ASN1_Error, ASN1_Object, _ASN1_ERROR
from typing import Any, AnyStr, Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union, cast

class BER_Exception(Exception):
    pass

class BER_Encoding_Error(ASN1_Encoding_Error):

    def __init__(self, msg, encoded=None, remaining=b''):
        if False:
            for i in range(10):
                print('nop')
        Exception.__init__(self, msg)
        self.remaining = remaining
        self.encoded = encoded

    def __str__(self):
        if False:
            i = 10
            return i + 15
        s = Exception.__str__(self)
        if isinstance(self.encoded, ASN1_Object):
            s += '\n### Already encoded ###\n%s' % self.encoded.strshow()
        else:
            s += '\n### Already encoded ###\n%r' % self.encoded
        s += '\n### Remaining ###\n%r' % self.remaining
        return s

class BER_Decoding_Error(ASN1_Decoding_Error):

    def __init__(self, msg, decoded=None, remaining=b''):
        if False:
            i = 10
            return i + 15
        Exception.__init__(self, msg)
        self.remaining = remaining
        self.decoded = decoded

    def __str__(self):
        if False:
            while True:
                i = 10
        s = Exception.__str__(self)
        if isinstance(self.decoded, ASN1_Object):
            s += '\n### Already decoded ###\n%s' % self.decoded.strshow()
        else:
            s += '\n### Already decoded ###\n%r' % self.decoded
        s += '\n### Remaining ###\n%r' % self.remaining
        return s

class BER_BadTag_Decoding_Error(BER_Decoding_Error, ASN1_BadTag_Decoding_Error):
    pass

def BER_len_enc(ll, size=0):
    if False:
        print('Hello World!')
    if ll <= 127 and size == 0:
        return chb(ll)
    s = b''
    while ll or size > 0:
        s = chb(ll & 255) + s
        ll >>= 8
        size -= 1
    if len(s) > 127:
        raise BER_Exception('BER_len_enc: Length too long (%i) to be encoded [%r]' % (len(s), s))
    return chb(len(s) | 128) + s

def BER_len_dec(s):
    if False:
        for i in range(10):
            print('nop')
    tmp_len = orb(s[0])
    if not tmp_len & 128:
        return (tmp_len, s[1:])
    tmp_len &= 127
    if len(s) <= tmp_len:
        raise BER_Decoding_Error('BER_len_dec: Got %i bytes while expecting %i' % (len(s) - 1, tmp_len), remaining=s)
    ll = 0
    for c in s[1:tmp_len + 1]:
        ll <<= 8
        ll |= orb(c)
    return (ll, s[tmp_len + 1:])

def BER_num_enc(ll, size=1):
    if False:
        return 10
    x = []
    while ll or size > 0:
        x.insert(0, ll & 127)
        if len(x) > 1:
            x[0] |= 128
        ll >>= 7
        size -= 1
    return b''.join((chb(k) for k in x))

def BER_num_dec(s, cls_id=0):
    if False:
        while True:
            i = 10
    if len(s) == 0:
        raise BER_Decoding_Error('BER_num_dec: got empty string', remaining=s)
    x = cls_id
    for (i, c) in enumerate(s):
        c = orb(c)
        x <<= 7
        x |= c & 127
        if not c & 128:
            break
    if c & 128:
        raise BER_Decoding_Error('BER_num_dec: unfinished number description', remaining=s)
    return (x, s[i + 1:])

def BER_id_dec(s):
    if False:
        print('Hello World!')
    x = orb(s[0])
    if x & 31 != 31:
        return (x, s[1:])
    else:
        return BER_num_dec(s[1:], cls_id=x >> 5)

def BER_id_enc(n):
    if False:
        print('Hello World!')
    if n < 256:
        return chb(n)
    else:
        s = BER_num_enc(n)
        tag = orb(s[0])
        tag &= 7
        tag <<= 5
        tag |= 31
        return chb(tag) + s[1:]

def BER_tagging_dec(s, hidden_tag=None, implicit_tag=None, explicit_tag=None, safe=False, _fname=''):
    if False:
        print('Hello World!')
    real_tag = None
    if len(s) > 0:
        err_msg = 'BER_tagging_dec: observed tag 0x%.02x does not match expected tag 0x%.02x (%s)'
        if implicit_tag is not None:
            (ber_id, s) = BER_id_dec(s)
            if ber_id != implicit_tag:
                if not safe and ber_id & 31 != implicit_tag & 31:
                    raise BER_Decoding_Error(err_msg % (ber_id, implicit_tag, _fname), remaining=s)
                else:
                    real_tag = ber_id
            s = chb(hash(hidden_tag)) + s
        elif explicit_tag is not None:
            (ber_id, s) = BER_id_dec(s)
            if ber_id != explicit_tag:
                if not safe:
                    raise BER_Decoding_Error(err_msg % (ber_id, explicit_tag, _fname), remaining=s)
                else:
                    real_tag = ber_id
            (l, s) = BER_len_dec(s)
    return (real_tag, s)

def BER_tagging_enc(s, hidden_tag=None, implicit_tag=None, explicit_tag=None):
    if False:
        print('Hello World!')
    if len(s) > 0:
        if implicit_tag is not None:
            s = BER_id_enc(hash(hidden_tag) & ~31 | implicit_tag) + s[1:]
        elif explicit_tag is not None:
            s = BER_id_enc(explicit_tag) + BER_len_enc(len(s)) + s
    return s

class BERcodec_metaclass(type):

    def __new__(cls, name, bases, dct):
        if False:
            print('Hello World!')
        c = cast('Type[BERcodec_Object[Any]]', super(BERcodec_metaclass, cls).__new__(cls, name, bases, dct))
        try:
            c.tag.register(c.codec, c)
        except Exception:
            warning('Error registering %r for %r' % (c.tag, c.codec))
        return c
_K = TypeVar('_K')

class BERcodec_Object(Generic[_K], metaclass=BERcodec_metaclass):
    codec = ASN1_Codecs.BER
    tag = ASN1_Class_UNIVERSAL.ANY

    @classmethod
    def asn1_object(cls, val):
        if False:
            while True:
                i = 10
        return cls.tag.asn1_object(val)

    @classmethod
    def check_string(cls, s):
        if False:
            print('Hello World!')
        if not s:
            raise BER_Decoding_Error('%s: Got empty object while expecting tag %r' % (cls.__name__, cls.tag), remaining=s)

    @classmethod
    def check_type(cls, s):
        if False:
            return 10
        cls.check_string(s)
        (tag, remainder) = BER_id_dec(s)
        if not isinstance(tag, int) or cls.tag != tag:
            raise BER_BadTag_Decoding_Error('%s: Got tag [%i/%#x] while expecting %r' % (cls.__name__, tag, tag, cls.tag), remaining=s)
        return remainder

    @classmethod
    def check_type_get_len(cls, s):
        if False:
            return 10
        s2 = cls.check_type(s)
        if not s2:
            raise BER_Decoding_Error('%s: No bytes while expecting a length' % cls.__name__, remaining=s)
        return BER_len_dec(s2)

    @classmethod
    def check_type_check_len(cls, s):
        if False:
            i = 10
            return i + 15
        (l, s3) = cls.check_type_get_len(s)
        if len(s3) < l:
            raise BER_Decoding_Error('%s: Got %i bytes while expecting %i' % (cls.__name__, len(s3), l), remaining=s)
        return (l, s3[:l], s3[l:])

    @classmethod
    def do_dec(cls, s, context=None, safe=False):
        if False:
            i = 10
            return i + 15
        if context is not None:
            _context = context
        else:
            _context = cls.tag.context
        cls.check_string(s)
        (p, remainder) = BER_id_dec(s)
        if p not in _context:
            t = s
            if len(t) > 18:
                t = t[:15] + b'...'
            raise BER_Decoding_Error('Unknown prefix [%02x] for [%r]' % (p, t), remaining=s)
        tag = _context[p]
        codec = cast('Type[BERcodec_Object[_K]]', tag.get_codec(ASN1_Codecs.BER))
        if codec == BERcodec_Object:
            (l, s) = BER_num_dec(remainder)
            return (ASN1_BADTAG(s[:l]), s[l:])
        return codec.dec(s, _context, safe)

    @classmethod
    def dec(cls, s, context=None, safe=False):
        if False:
            print('Hello World!')
        if not safe:
            return cls.do_dec(s, context, safe)
        try:
            return cls.do_dec(s, context, safe)
        except BER_BadTag_Decoding_Error as e:
            (o, remain) = BERcodec_Object.dec(e.remaining, context, safe)
            return (ASN1_BADTAG(o), remain)
        except BER_Decoding_Error as e:
            return (ASN1_DECODING_ERROR(s, exc=e), b'')
        except ASN1_Error as e:
            return (ASN1_DECODING_ERROR(s, exc=e), b'')

    @classmethod
    def safedec(cls, s, context=None):
        if False:
            print('Hello World!')
        return cls.dec(s, context, safe=True)

    @classmethod
    def enc(cls, s):
        if False:
            while True:
                i = 10
        if isinstance(s, (str, bytes)):
            return BERcodec_STRING.enc(s)
        else:
            try:
                return BERcodec_INTEGER.enc(int(s))
            except TypeError:
                raise TypeError('Trying to encode an invalid value !')
ASN1_Codecs.BER.register_stem(BERcodec_Object)

class BERcodec_INTEGER(BERcodec_Object[int]):
    tag = ASN1_Class_UNIVERSAL.INTEGER

    @classmethod
    def enc(cls, i):
        if False:
            return 10
        ls = []
        while True:
            ls.append(i & 255)
            if -127 <= i < 0:
                break
            if 128 <= i <= 255:
                ls.append(0)
            i >>= 8
            if not i:
                break
        s = [chb(hash(c)) for c in ls]
        s.append(BER_len_enc(len(s)))
        s.append(chb(hash(cls.tag)))
        s.reverse()
        return b''.join(s)

    @classmethod
    def do_dec(cls, s, context=None, safe=False):
        if False:
            while True:
                i = 10
        (l, s, t) = cls.check_type_check_len(s)
        x = 0
        if s:
            if orb(s[0]) & 128:
                x = -1
            for c in s:
                x <<= 8
                x |= orb(c)
        return (cls.asn1_object(x), t)

class BERcodec_BOOLEAN(BERcodec_INTEGER):
    tag = ASN1_Class_UNIVERSAL.BOOLEAN

class BERcodec_BIT_STRING(BERcodec_Object[str]):
    tag = ASN1_Class_UNIVERSAL.BIT_STRING

    @classmethod
    def do_dec(cls, s, context=None, safe=False):
        if False:
            while True:
                i = 10
        (l, s, t) = cls.check_type_check_len(s)
        if len(s) > 0:
            unused_bits = orb(s[0])
            if safe and unused_bits > 7:
                raise BER_Decoding_Error('BERcodec_BIT_STRING: too many unused_bits advertised', remaining=s)
            fs = ''.join((binrepr(orb(x)).zfill(8) for x in s[1:]))
            if unused_bits > 0:
                fs = fs[:-unused_bits]
            return (cls.tag.asn1_object(fs), t)
        else:
            raise BER_Decoding_Error('BERcodec_BIT_STRING found no content (not even unused_bits byte)', remaining=s)

    @classmethod
    def enc(cls, _s):
        if False:
            return 10
        s = bytes_encode(_s)
        if len(s) % 8 == 0:
            unused_bits = 0
        else:
            unused_bits = 8 - len(s) % 8
            s += b'0' * unused_bits
        s = b''.join((chb(int(b''.join((chb(y) for y in x)), 2)) for x in zip(*[iter(s)] * 8)))
        s = chb(unused_bits) + s
        return chb(hash(cls.tag)) + BER_len_enc(len(s)) + s

class BERcodec_STRING(BERcodec_Object[str]):
    tag = ASN1_Class_UNIVERSAL.STRING

    @classmethod
    def enc(cls, _s):
        if False:
            for i in range(10):
                print('nop')
        s = bytes_encode(_s)
        return chb(hash(cls.tag)) + BER_len_enc(len(s)) + s

    @classmethod
    def do_dec(cls, s, context=None, safe=False):
        if False:
            for i in range(10):
                print('nop')
        (l, s, t) = cls.check_type_check_len(s)
        return (cls.tag.asn1_object(s), t)

class BERcodec_NULL(BERcodec_INTEGER):
    tag = ASN1_Class_UNIVERSAL.NULL

    @classmethod
    def enc(cls, i):
        if False:
            print('Hello World!')
        if i == 0:
            return chb(hash(cls.tag)) + b'\x00'
        else:
            return super(cls, cls).enc(i)

class BERcodec_OID(BERcodec_Object[bytes]):
    tag = ASN1_Class_UNIVERSAL.OID

    @classmethod
    def enc(cls, _oid):
        if False:
            return 10
        oid = bytes_encode(_oid)
        if oid:
            lst = [int(x) for x in oid.strip(b'.').split(b'.')]
        else:
            lst = list()
        if len(lst) >= 2:
            lst[1] += 40 * lst[0]
            del lst[0]
        s = b''.join((BER_num_enc(k) for k in lst))
        return chb(hash(cls.tag)) + BER_len_enc(len(s)) + s

    @classmethod
    def do_dec(cls, s, context=None, safe=False):
        if False:
            return 10
        (l, s, t) = cls.check_type_check_len(s)
        lst = []
        while s:
            (l, s) = BER_num_dec(s)
            lst.append(l)
        if len(lst) > 0:
            lst.insert(0, lst[0] // 40)
            lst[1] %= 40
        return (cls.asn1_object(b'.'.join((str(k).encode('ascii') for k in lst))), t)

class BERcodec_ENUMERATED(BERcodec_INTEGER):
    tag = ASN1_Class_UNIVERSAL.ENUMERATED

class BERcodec_UTF8_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.UTF8_STRING

class BERcodec_NUMERIC_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.NUMERIC_STRING

class BERcodec_PRINTABLE_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.PRINTABLE_STRING

class BERcodec_T61_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.T61_STRING

class BERcodec_VIDEOTEX_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.VIDEOTEX_STRING

class BERcodec_IA5_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.IA5_STRING

class BERcodec_GENERAL_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.GENERAL_STRING

class BERcodec_UTC_TIME(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.UTC_TIME

class BERcodec_GENERALIZED_TIME(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.GENERALIZED_TIME

class BERcodec_ISO646_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.ISO646_STRING

class BERcodec_UNIVERSAL_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.UNIVERSAL_STRING

class BERcodec_BMP_STRING(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.BMP_STRING

class BERcodec_SEQUENCE(BERcodec_Object[Union[bytes, List[BERcodec_Object[Any]]]]):
    tag = ASN1_Class_UNIVERSAL.SEQUENCE

    @classmethod
    def enc(cls, _ll):
        if False:
            while True:
                i = 10
        if isinstance(_ll, bytes):
            ll = _ll
        else:
            ll = b''.join((x.enc(cls.codec) for x in _ll))
        return chb(hash(cls.tag)) + BER_len_enc(len(ll)) + ll

    @classmethod
    def do_dec(cls, s, context=None, safe=False):
        if False:
            return 10
        if context is None:
            context = cls.tag.context
        (ll, st) = cls.check_type_get_len(s)
        (s, t) = (st[:ll], st[ll:])
        obj = []
        while s:
            try:
                (o, remain) = BERcodec_Object.dec(s, context, safe)
                s = remain
            except BER_Decoding_Error as err:
                err.remaining += t
                if err.decoded is not None:
                    obj.append(err.decoded)
                err.decoded = obj
                raise
            obj.append(o)
        if len(st) < ll:
            raise BER_Decoding_Error('Not enough bytes to decode sequence', decoded=obj)
        return (cls.asn1_object(obj), t)

class BERcodec_SET(BERcodec_SEQUENCE):
    tag = ASN1_Class_UNIVERSAL.SET

class BERcodec_IPADDRESS(BERcodec_STRING):
    tag = ASN1_Class_UNIVERSAL.IPADDRESS

    @classmethod
    def enc(cls, ipaddr_ascii):
        if False:
            print('Hello World!')
        try:
            s = inet_aton(ipaddr_ascii)
        except Exception:
            raise BER_Encoding_Error('IPv4 address could not be encoded')
        return chb(hash(cls.tag)) + BER_len_enc(len(s)) + s

    @classmethod
    def do_dec(cls, s, context=None, safe=False):
        if False:
            i = 10
            return i + 15
        (l, s, t) = cls.check_type_check_len(s)
        try:
            ipaddr_ascii = inet_ntoa(s)
        except Exception:
            raise BER_Decoding_Error('IP address could not be decoded', remaining=s)
        return (cls.asn1_object(ipaddr_ascii), t)

class BERcodec_COUNTER32(BERcodec_INTEGER):
    tag = ASN1_Class_UNIVERSAL.COUNTER32

class BERcodec_GAUGE32(BERcodec_INTEGER):
    tag = ASN1_Class_UNIVERSAL.GAUGE32

class BERcodec_TIME_TICKS(BERcodec_INTEGER):
    tag = ASN1_Class_UNIVERSAL.TIME_TICKS