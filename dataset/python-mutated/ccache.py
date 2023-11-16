import base64
import struct
from typing import Any, Dict, List, Optional

def der_encode_length(length: int) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    if length <= 127:
        return struct.pack('!B', length)
    out = b''
    while length > 0:
        out = struct.pack('!B', length & 255) + out
        length >>= 8
    out = struct.pack('!B', len(out) | 128) + out
    return out

def der_encode_tlv(tag: int, value: bytes) -> bytes:
    if False:
        return 10
    return struct.pack('!B', tag) + der_encode_length(len(value)) + value

def der_encode_integer_value(val: int) -> bytes:
    if False:
        while True:
            i = 10
    if not isinstance(val, int):
        raise TypeError('int')
    if val == 0:
        return b'\x00'
    sign = 0
    out = b''
    while val != sign:
        byte = val & 255
        out = struct.pack('!B', byte) + out
        sign = -1 if byte & 128 == 128 else 0
        val >>= 8
    return out

def der_encode_integer(val: int) -> bytes:
    if False:
        while True:
            i = 10
    return der_encode_tlv(2, der_encode_integer_value(val))

def der_encode_int32(val: int) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    if val < -2147483648 or val > 2147483647:
        raise ValueError('Bad value')
    return der_encode_integer(val)

def der_encode_uint32(val: int) -> bytes:
    if False:
        print('Hello World!')
    if val < 0 or val > 4294967295:
        raise ValueError('Bad value')
    return der_encode_integer(val)

def der_encode_string(val: str) -> bytes:
    if False:
        print('Hello World!')
    if not isinstance(val, str):
        raise TypeError('unicode')
    return der_encode_tlv(27, val.encode())

def der_encode_octet_string(val: bytes) -> bytes:
    if False:
        while True:
            i = 10
    if not isinstance(val, bytes):
        raise TypeError('bytes')
    return der_encode_tlv(4, val)

def der_encode_sequence(tlvs: List[Optional[bytes]], tagged: bool=True) -> bytes:
    if False:
        while True:
            i = 10
    body = []
    for (i, tlv) in enumerate(tlvs):
        if tlv is None:
            continue
        if tagged:
            tlv = der_encode_tlv(160 | i, tlv)
        body.append(tlv)
    return der_encode_tlv(48, b''.join(body))

def der_encode_ticket(tkt: Dict[str, Any]) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    return der_encode_tlv(97, der_encode_sequence([der_encode_integer(5), der_encode_string(tkt['realm']), der_encode_sequence([der_encode_int32(tkt['sname']['nameType']), der_encode_sequence([der_encode_string(c) for c in tkt['sname']['nameString']], tagged=False)]), der_encode_sequence([der_encode_int32(tkt['encPart']['etype']), der_encode_uint32(tkt['encPart']['kvno']) if 'kvno' in tkt['encPart'] else None, der_encode_octet_string(base64.b64decode(tkt['encPart']['cipher']))])]))

def ccache_counted_octet_string(data: bytes) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(data, bytes):
        raise TypeError('bytes')
    return struct.pack('!I', len(data)) + data

def ccache_principal(name: Dict[str, str], realm: str) -> bytes:
    if False:
        print('Hello World!')
    header = struct.pack('!II', name['nameType'], len(name['nameString']))
    return header + ccache_counted_octet_string(realm.encode()) + b''.join((ccache_counted_octet_string(c.encode()) for c in name['nameString']))

def ccache_key(key: Dict[str, str]) -> bytes:
    if False:
        return 10
    return struct.pack('!H', key['keytype']) + ccache_counted_octet_string(base64.b64decode(key['keyvalue']))

def flags_to_uint32(flags: List[str]) -> int:
    if False:
        for i in range(10):
            print('nop')
    ret = 0
    for (i, v) in enumerate(flags):
        if v:
            ret |= 1 << 31 - i
    return ret

def ccache_credential(cred: Dict[str, Any]) -> bytes:
    if False:
        return 10
    out = ccache_principal(cred['cname'], cred['crealm'])
    out += ccache_principal(cred['sname'], cred['srealm'])
    out += ccache_key(cred['key'])
    out += struct.pack('!IIII', cred['authtime'] // 1000, cred.get('starttime', cred['authtime']) // 1000, cred['endtime'] // 1000, cred.get('renewTill', 0) // 1000)
    out += struct.pack('!B', 0)
    out += struct.pack('!I', flags_to_uint32(cred['flags']))
    out += struct.pack('!II', 0, 0)
    out += ccache_counted_octet_string(der_encode_ticket(cred['ticket']))
    out += ccache_counted_octet_string(b'')
    return out

def make_ccache(cred: Dict[str, Any]) -> bytes:
    if False:
        i = 10
        return i + 15
    out = struct.pack('!HHHHII', 1284, 12, 1, 8, 0, 0)
    out += ccache_principal(cred['cname'], cred['crealm'])
    out += ccache_credential(cred)
    return out