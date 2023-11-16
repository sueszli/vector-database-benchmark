import re
import time
from hashlib import sha256
from binascii import hexlify
from decimal import Decimal
from typing import Optional, TYPE_CHECKING, Type, Dict, Any
import random
import bitstring
from .bitcoin import hash160_to_b58_address, b58_address_to_hash160, TOTAL_COIN_SUPPLY_LIMIT_IN_BTC
from .segwit_addr import bech32_encode, bech32_decode, CHARSET
from . import segwit_addr
from . import constants
from .constants import AbstractNet
from . import ecc
from .bitcoin import COIN
if TYPE_CHECKING:
    from .lnutil import LnFeatures

class LnInvoiceException(Exception):
    pass

class LnDecodeException(LnInvoiceException):
    pass

class LnEncodeException(LnInvoiceException):
    pass

def shorten_amount(amount):
    if False:
        for i in range(10):
            print('nop')
    ' Given an amount in bitcoin, shorten it\n    '
    amount = int(amount * 10 ** 12)
    units = ['p', 'n', 'u', 'm']
    for unit in units:
        if amount % 1000 == 0:
            amount //= 1000
        else:
            break
    else:
        unit = ''
    return str(amount) + unit

def unshorten_amount(amount) -> Decimal:
    if False:
        i = 10
        return i + 15
    ' Given a shortened amount, convert it into a decimal\n    '
    units = {'p': 10 ** 12, 'n': 10 ** 9, 'u': 10 ** 6, 'm': 10 ** 3}
    unit = str(amount)[-1]
    if not re.fullmatch('\\d+[pnum]?', str(amount)):
        raise LnDecodeException("Invalid amount '{}'".format(amount))
    if unit in units.keys():
        return Decimal(amount[:-1]) / units[unit]
    else:
        return Decimal(amount)
_INT_TO_BINSTR = {a: '0' * (5 - len(bin(a)[2:])) + bin(a)[2:] for a in range(32)}

def u5_to_bitarray(arr):
    if False:
        while True:
            i = 10
    b = ''.join((_INT_TO_BINSTR[a] for a in arr))
    return bitstring.BitArray(bin=b)

def bitarray_to_u5(barr):
    if False:
        for i in range(10):
            print('nop')
    assert barr.len % 5 == 0
    ret = []
    s = bitstring.ConstBitStream(barr)
    while s.pos != s.len:
        ret.append(s.read(5).uint)
    return ret

def encode_fallback(fallback: str, net: Type[AbstractNet]):
    if False:
        while True:
            i = 10
    ' Encode all supported fallback addresses.\n    '
    (wver, wprog_ints) = segwit_addr.decode_segwit_address(net.SEGWIT_HRP, fallback)
    if wver is not None:
        wprog = bytes(wprog_ints)
    else:
        (addrtype, addr) = b58_address_to_hash160(fallback)
        if addrtype == net.ADDRTYPE_P2PKH:
            wver = 17
        elif addrtype == net.ADDRTYPE_P2SH:
            wver = 18
        else:
            raise LnEncodeException(f'Unknown address type {addrtype} for {net}')
        wprog = addr
    return tagged('f', bitstring.pack('uint:5', wver) + wprog)

def parse_fallback(fallback, net: Type[AbstractNet]):
    if False:
        print('Hello World!')
    wver = fallback[0:5].uint
    if wver == 17:
        addr = hash160_to_b58_address(fallback[5:].tobytes(), net.ADDRTYPE_P2PKH)
    elif wver == 18:
        addr = hash160_to_b58_address(fallback[5:].tobytes(), net.ADDRTYPE_P2SH)
    elif wver <= 16:
        witprog = fallback[5:]
        witprog = witprog[:len(witprog) // 8 * 8]
        witprog = witprog.tobytes()
        addr = segwit_addr.encode_segwit_address(net.SEGWIT_HRP, wver, witprog)
    else:
        return None
    return addr
BOLT11_HRP_INV_DICT = {net.BOLT11_HRP: net for net in constants.NETS_LIST}

def tagged(char, l):
    if False:
        for i in range(10):
            print('nop')
    while l.len % 5 != 0:
        l.append('0b0')
    return bitstring.pack('uint:5, uint:5, uint:5', CHARSET.find(char), l.len / 5 / 32, l.len / 5 % 32) + l

def tagged_bytes(char, l):
    if False:
        return 10
    return tagged(char, bitstring.BitArray(l))

def trim_to_min_length(bits):
    if False:
        return 10
    "Ensures 'bits' have min number of leading zeroes.\n    Assumes 'bits' is big-endian, and that it needs to be encoded in 5 bit blocks.\n    "
    bits = bits[:]
    while bits.len % 5 != 0:
        bits.prepend('0b0')
    while bits.startswith('0b00000'):
        if len(bits) == 5:
            break
        bits = bits[5:]
    return bits

def trim_to_bytes(barr):
    if False:
        i = 10
        return i + 15
    b = barr.tobytes()
    if barr.len % 8 != 0:
        return b[:-1]
    return b

def pull_tagged(stream):
    if False:
        while True:
            i = 10
    tag = stream.read(5).uint
    length = stream.read(5).uint * 32 + stream.read(5).uint
    return (CHARSET[tag], stream.read(length * 5), stream)

def lnencode(addr: 'LnAddr', privkey) -> str:
    if False:
        i = 10
        return i + 15
    if addr.amount:
        amount = addr.net.BOLT11_HRP + shorten_amount(addr.amount)
    else:
        amount = addr.net.BOLT11_HRP if addr.net else ''
    hrp = 'ln' + amount
    data = bitstring.pack('uint:35', addr.date)
    tags_set = set()
    assert addr.paymenthash is not None
    data += tagged_bytes('p', addr.paymenthash)
    tags_set.add('p')
    if addr.payment_secret is not None:
        data += tagged_bytes('s', addr.payment_secret)
        tags_set.add('s')
    for (k, v) in addr.tags:
        if k in ('d', 'h', 'n', 'x', 'p', 's', '9'):
            if k in tags_set:
                raise LnEncodeException("Duplicate '{}' tag".format(k))
        if k == 'r':
            route = bitstring.BitArray()
            for step in v:
                (pubkey, channel, feebase, feerate, cltv) = step
                route.append(bitstring.BitArray(pubkey) + bitstring.BitArray(channel) + bitstring.pack('intbe:32', feebase) + bitstring.pack('intbe:32', feerate) + bitstring.pack('intbe:16', cltv))
            data += tagged('r', route)
        elif k == 't':
            (pubkey, feebase, feerate, cltv) = v
            route = bitstring.BitArray(pubkey) + bitstring.pack('intbe:32', feebase) + bitstring.pack('intbe:32', feerate) + bitstring.pack('intbe:16', cltv)
            data += tagged('t', route)
        elif k == 'f':
            if v is not None:
                data += encode_fallback(v, addr.net)
        elif k == 'd':
            data += tagged_bytes('d', v.encode()[0:639])
        elif k == 'x':
            expirybits = bitstring.pack('intbe:64', v)
            expirybits = trim_to_min_length(expirybits)
            data += tagged('x', expirybits)
        elif k == 'h':
            data += tagged_bytes('h', sha256(v.encode('utf-8')).digest())
        elif k == 'n':
            data += tagged_bytes('n', v)
        elif k == 'c':
            finalcltvbits = bitstring.pack('intbe:64', v)
            finalcltvbits = trim_to_min_length(finalcltvbits)
            data += tagged('c', finalcltvbits)
        elif k == '9':
            if v == 0:
                continue
            feature_bits = bitstring.BitArray(uint=v, length=v.bit_length())
            feature_bits = trim_to_min_length(feature_bits)
            data += tagged('9', feature_bits)
        else:
            raise LnEncodeException('Unknown tag {}'.format(k))
        tags_set.add(k)
    if 'd' in tags_set and 'h' in tags_set:
        raise ValueError("Cannot include both 'd' and 'h'")
    if 'd' not in tags_set and 'h' not in tags_set:
        raise ValueError("Must include either 'd' or 'h'")
    msg = hrp.encode('ascii') + data.tobytes()
    privkey = ecc.ECPrivkey(privkey)
    sig = privkey.sign_message(msg, is_compressed=False, algo=lambda x: sha256(x).digest())
    recovery_flag = bytes([sig[0] - 27])
    sig = bytes(sig[1:]) + recovery_flag
    data += sig
    return bech32_encode(segwit_addr.Encoding.BECH32, hrp, bitarray_to_u5(data))

class LnAddr(object):

    def __init__(self, *, paymenthash: bytes=None, amount=None, net: Type[AbstractNet]=None, tags=None, date=None, payment_secret: bytes=None):
        if False:
            return 10
        self.date = int(time.time()) if not date else int(date)
        self.tags = [] if not tags else tags
        self.unknown_tags = []
        self.paymenthash = paymenthash
        self.payment_secret = payment_secret
        self.signature = None
        self.pubkey = None
        self.net = constants.net if net is None else net
        self._amount = amount

    @property
    def amount(self) -> Optional[Decimal]:
        if False:
            while True:
                i = 10
        return self._amount

    @amount.setter
    def amount(self, value):
        if False:
            while True:
                i = 10
        if not (isinstance(value, Decimal) or value is None):
            raise LnInvoiceException(f'amount must be Decimal or None, not {value!r}')
        if value is None:
            self._amount = None
            return
        assert isinstance(value, Decimal)
        if value.is_nan() or not 0 <= value <= TOTAL_COIN_SUPPLY_LIMIT_IN_BTC:
            raise LnInvoiceException(f'amount is out-of-bounds: {value!r} BTC')
        if value * 10 ** 12 % 10:
            raise LnInvoiceException(f'Cannot encode {value!r}: too many decimal places')
        self._amount = value

    def get_amount_sat(self) -> Optional[Decimal]:
        if False:
            return 10
        if self.amount is None:
            return None
        return self.amount * COIN

    def get_routing_info(self, tag):
        if False:
            return 10
        r_tags = list(filter(lambda x: x[0] == tag, self.tags))
        r_tags = list(map(lambda x: x[1], r_tags))
        random.shuffle(r_tags)
        return r_tags

    def get_amount_msat(self) -> Optional[int]:
        if False:
            i = 10
            return i + 15
        if self.amount is None:
            return None
        return int(self.amount * COIN * 1000)

    def get_features(self) -> 'LnFeatures':
        if False:
            print('Hello World!')
        from .lnutil import LnFeatures
        return LnFeatures(self.get_tag('9') or 0)

    def validate_and_compare_features(self, myfeatures: 'LnFeatures') -> None:
        if False:
            while True:
                i = 10
        'Raises IncompatibleOrInsaneFeatures.\n\n        note: these checks are not done by the parser (in lndecode), as then when we started requiring a new feature,\n              old saved already paid invoices could no longer be parsed.\n        '
        from .lnutil import validate_features, ln_compare_features
        invoice_features = self.get_features()
        validate_features(invoice_features)
        ln_compare_features(myfeatures.for_invoice(), invoice_features)

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        return 'LnAddr[{}, amount={}{} tags=[{}]]'.format(hexlify(self.pubkey.serialize()).decode('utf-8') if self.pubkey else None, self.amount, self.net.BOLT11_HRP, ', '.join([k + '=' + str(v) for (k, v) in self.tags]))

    def get_min_final_cltv_delta(self) -> int:
        if False:
            i = 10
            return i + 15
        cltv = self.get_tag('c')
        if cltv is None:
            return 18
        return int(cltv)

    def get_tag(self, tag):
        if False:
            print('Hello World!')
        for (k, v) in self.tags:
            if k == tag:
                return v
        return None

    def get_description(self) -> str:
        if False:
            print('Hello World!')
        return self.get_tag('d') or ''

    def get_fallback_address(self) -> str:
        if False:
            while True:
                i = 10
        return self.get_tag('f') or ''

    def get_expiry(self) -> int:
        if False:
            while True:
                i = 10
        exp = self.get_tag('x')
        if exp is None:
            exp = 3600
        return int(exp)

    def is_expired(self) -> bool:
        if False:
            while True:
                i = 10
        now = time.time()
        return now > self.get_expiry() + self.date

    def to_debug_json(self) -> Dict[str, Any]:
        if False:
            print('Hello World!')
        d = {'pubkey': self.pubkey.serialize().hex(), 'amount_BTC': str(self.amount), 'rhash': self.paymenthash.hex(), 'payment_secret': self.payment_secret.hex() if self.payment_secret else None, 'description': self.get_description(), 'exp': self.get_expiry(), 'time': self.date, 'min_final_cltv_delta': self.get_min_final_cltv_delta(), 'features': self.get_features().get_names(), 'tags': self.tags, 'unknown_tags': self.unknown_tags}
        if (ln_routing_info := self.get_routing_info('r')):
            d['r_tags'] = [str((a.hex(), b.hex(), c, d, e)) for (a, b, c, d, e) in ln_routing_info[-1]]
        return d

class SerializableKey:

    def __init__(self, pubkey):
        if False:
            i = 10
            return i + 15
        self.pubkey = pubkey

    def serialize(self):
        if False:
            while True:
                i = 10
        return self.pubkey.get_public_key_bytes(True)

def lndecode(invoice: str, *, verbose=False, net=None) -> LnAddr:
    if False:
        i = 10
        return i + 15
    'Parses a string into an LnAddr object.\n    Can raise LnDecodeException or IncompatibleOrInsaneFeatures.\n    '
    if net is None:
        net = constants.net
    decoded_bech32 = bech32_decode(invoice, ignore_long_length=True)
    hrp = decoded_bech32.hrp
    data = decoded_bech32.data
    if decoded_bech32.encoding is None:
        raise LnDecodeException('Bad bech32 checksum')
    if decoded_bech32.encoding != segwit_addr.Encoding.BECH32:
        raise LnDecodeException('Bad bech32 encoding: must be using vanilla BECH32')
    if not hrp.startswith('ln'):
        raise LnDecodeException('Does not start with ln')
    if not hrp[2:].startswith(net.BOLT11_HRP):
        raise LnDecodeException(f'Wrong Lightning invoice HRP {hrp[2:]}, should be {net.BOLT11_HRP}')
    data = u5_to_bitarray(data)
    if len(data) < 65 * 8:
        raise LnDecodeException('Too short to contain signature')
    sigdecoded = data[-65 * 8:].tobytes()
    data = bitstring.ConstBitStream(data[:-65 * 8])
    addr = LnAddr()
    addr.pubkey = None
    m = re.search('[^\\d]+', hrp[2:])
    if m:
        addr.net = BOLT11_HRP_INV_DICT[m.group(0)]
        amountstr = hrp[2 + m.end():]
        if amountstr != '':
            addr.amount = unshorten_amount(amountstr)
    addr.date = data.read(35).uint
    while data.pos != data.len:
        (tag, tagdata, data) = pull_tagged(data)
        data_length = len(tagdata) / 5
        if tag == 'r':
            route = []
            s = bitstring.ConstBitStream(tagdata)
            while s.pos + 264 + 64 + 32 + 32 + 16 < s.len:
                route.append((s.read(264).tobytes(), s.read(64).tobytes(), s.read(32).uintbe, s.read(32).uintbe, s.read(16).uintbe))
            addr.tags.append(('r', route))
        elif tag == 't':
            s = bitstring.ConstBitStream(tagdata)
            e = (s.read(264).tobytes(), s.read(32).uintbe, s.read(32).uintbe, s.read(16).uintbe)
            addr.tags.append(('t', e))
        elif tag == 'f':
            fallback = parse_fallback(tagdata, addr.net)
            if fallback:
                addr.tags.append(('f', fallback))
            else:
                addr.unknown_tags.append((tag, tagdata))
                continue
        elif tag == 'd':
            addr.tags.append(('d', trim_to_bytes(tagdata).decode('utf-8')))
        elif tag == 'h':
            if data_length != 52:
                addr.unknown_tags.append((tag, tagdata))
                continue
            addr.tags.append(('h', trim_to_bytes(tagdata)))
        elif tag == 'x':
            addr.tags.append(('x', tagdata.uint))
        elif tag == 'p':
            if data_length != 52:
                addr.unknown_tags.append((tag, tagdata))
                continue
            addr.paymenthash = trim_to_bytes(tagdata)
        elif tag == 's':
            if data_length != 52:
                addr.unknown_tags.append((tag, tagdata))
                continue
            addr.payment_secret = trim_to_bytes(tagdata)
        elif tag == 'n':
            if data_length != 53:
                addr.unknown_tags.append((tag, tagdata))
                continue
            pubkeybytes = trim_to_bytes(tagdata)
            addr.pubkey = pubkeybytes
        elif tag == 'c':
            addr.tags.append(('c', tagdata.uint))
        elif tag == '9':
            features = tagdata.uint
            addr.tags.append(('9', features))
        else:
            addr.unknown_tags.append((tag, tagdata))
    if verbose:
        print('hex of signature data (32 byte r, 32 byte s): {}'.format(hexlify(sigdecoded[0:64])))
        print('recovery flag: {}'.format(sigdecoded[64]))
        print('hex of data for signing: {}'.format(hexlify(hrp.encode('ascii') + data.tobytes())))
        print('SHA256 of above: {}'.format(sha256(hrp.encode('ascii') + data.tobytes()).hexdigest()))
    addr.signature = sigdecoded[:65]
    hrp_hash = sha256(hrp.encode('ascii') + data.tobytes()).digest()
    if addr.pubkey:
        if not ecc.ECPubkey(addr.pubkey).verify_message_hash(sigdecoded[:64], hrp_hash):
            raise LnDecodeException('bad signature')
        pubkey_copy = addr.pubkey

        class WrappedBytesKey:
            serialize = lambda : pubkey_copy
        addr.pubkey = WrappedBytesKey
    else:
        addr.pubkey = SerializableKey(ecc.ECPubkey.from_sig_string(sigdecoded[:64], sigdecoded[64], hrp_hash))
    return addr