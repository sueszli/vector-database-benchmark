import base64
import struct
from typing import Any, Dict, List, Optional

# This file is adapted from samples/shellinabox/ssh-krb-wrapper in
# https://github.com/davidben/webathena, which has the following
# license:
#
# Copyright (c) 2013 David Benjamin and Alan Huang
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation files
# (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# Some DER encoding stuff. Bleh. This is because the ccache contains a
# DER-encoded krb5 Ticket structure, whereas Webathena deserializes
# into the various fields. Re-encoding in the client would be easy as
# there is already an ASN.1 implementation, but in the interest of
# limiting MIT Kerberos's exposure to malformed ccaches, encode it
# ourselves. To that end, here's the laziest DER encoder ever.
def der_encode_length(length: int) -> bytes:
    if length <= 127:
        return struct.pack("!B", length)
    out = b""
    while length > 0:
        out = struct.pack("!B", length & 0xFF) + out
        length >>= 8
    out = struct.pack("!B", len(out) | 0x80) + out
    return out


def der_encode_tlv(tag: int, value: bytes) -> bytes:
    return struct.pack("!B", tag) + der_encode_length(len(value)) + value


def der_encode_integer_value(val: int) -> bytes:
    if not isinstance(val, int):
        raise TypeError("int")
    # base 256, MSB first, two's complement, minimum number of octets
    # necessary. This has a number of annoying edge cases:
    # * 0 and -1 are 0x00 and 0xFF, not the empty string.
    # * 255 is 0x00 0xFF, not 0xFF
    # * -256 is 0xFF 0x00, not 0x00

    # Special-case to avoid an empty encoding.
    if val == 0:
        return b"\x00"
    sign = 0  # What you would get if you sign-extended the current high bit.
    out = b""
    # We can stop once sign-extension matches the remaining value.
    while val != sign:
        byte = val & 0xFF
        out = struct.pack("!B", byte) + out
        sign = -1 if byte & 0x80 == 0x80 else 0
        val >>= 8
    return out


def der_encode_integer(val: int) -> bytes:
    return der_encode_tlv(0x02, der_encode_integer_value(val))


def der_encode_int32(val: int) -> bytes:
    if val < -2147483648 or val > 2147483647:
        raise ValueError("Bad value")
    return der_encode_integer(val)


def der_encode_uint32(val: int) -> bytes:
    if val < 0 or val > 4294967295:
        raise ValueError("Bad value")
    return der_encode_integer(val)


def der_encode_string(val: str) -> bytes:
    if not isinstance(val, str):
        raise TypeError("unicode")
    return der_encode_tlv(0x1B, val.encode())


def der_encode_octet_string(val: bytes) -> bytes:
    if not isinstance(val, bytes):
        raise TypeError("bytes")
    return der_encode_tlv(0x04, val)


def der_encode_sequence(tlvs: List[Optional[bytes]], tagged: bool = True) -> bytes:
    body = []
    for i, tlv in enumerate(tlvs):
        # Missing optional elements represented as None.
        if tlv is None:
            continue
        if tagged:
            # Assume kerberos-style explicit tagging of components.
            tlv = der_encode_tlv(0xA0 | i, tlv)
        body.append(tlv)
    return der_encode_tlv(0x30, b"".join(body))


def der_encode_ticket(tkt: Dict[str, Any]) -> bytes:
    return der_encode_tlv(
        0x61,  # Ticket
        der_encode_sequence(
            [
                der_encode_integer(5),  # tktVno
                der_encode_string(tkt["realm"]),
                der_encode_sequence(  # PrincipalName
                    [
                        der_encode_int32(tkt["sname"]["nameType"]),
                        der_encode_sequence(
                            [der_encode_string(c) for c in tkt["sname"]["nameString"]], tagged=False
                        ),
                    ]
                ),
                der_encode_sequence(  # EncryptedData
                    [
                        der_encode_int32(tkt["encPart"]["etype"]),
                        der_encode_uint32(tkt["encPart"]["kvno"])
                        if "kvno" in tkt["encPart"]
                        else None,
                        der_encode_octet_string(base64.b64decode(tkt["encPart"]["cipher"])),
                    ]
                ),
            ]
        ),
    )


# Kerberos ccache writing code. Using format documentation from here:
# https://www.gnu.org/software/shishi/manual/html_node/The-Credential-Cache-Binary-File-Format.html


def ccache_counted_octet_string(data: bytes) -> bytes:
    if not isinstance(data, bytes):
        raise TypeError("bytes")
    return struct.pack("!I", len(data)) + data


def ccache_principal(name: Dict[str, str], realm: str) -> bytes:
    header = struct.pack("!II", name["nameType"], len(name["nameString"]))
    return (
        header
        + ccache_counted_octet_string(realm.encode())
        + b"".join(ccache_counted_octet_string(c.encode()) for c in name["nameString"])
    )


def ccache_key(key: Dict[str, str]) -> bytes:
    return struct.pack("!H", key["keytype"]) + ccache_counted_octet_string(
        base64.b64decode(key["keyvalue"])
    )


def flags_to_uint32(flags: List[str]) -> int:
    ret = 0
    for i, v in enumerate(flags):
        if v:
            ret |= 1 << (31 - i)
    return ret


def ccache_credential(cred: Dict[str, Any]) -> bytes:
    out = ccache_principal(cred["cname"], cred["crealm"])
    out += ccache_principal(cred["sname"], cred["srealm"])
    out += ccache_key(cred["key"])
    out += struct.pack(
        "!IIII",
        cred["authtime"] // 1000,
        cred.get("starttime", cred["authtime"]) // 1000,
        cred["endtime"] // 1000,
        cred.get("renewTill", 0) // 1000,
    )
    out += struct.pack("!B", 0)
    out += struct.pack("!I", flags_to_uint32(cred["flags"]))
    # TODO: Care about addrs or authdata? Former is "caddr" key.
    out += struct.pack("!II", 0, 0)
    out += ccache_counted_octet_string(der_encode_ticket(cred["ticket"]))
    # No second_ticket.
    out += ccache_counted_octet_string(b"")
    return out


def make_ccache(cred: Dict[str, Any]) -> bytes:
    # Do we need a DeltaTime header? The ccache I get just puts zero
    # in there, so do the same.
    out = struct.pack(
        "!HHHHII",
        0x0504,  # file_format_version
        12,  # headerlen
        1,  # tag (DeltaTime)
        8,  # taglen (two uint32_ts)
        0,
        0,  # time_offset / usec_offset
    )
    out += ccache_principal(cred["cname"], cred["crealm"])
    out += ccache_credential(cred)
    return out
