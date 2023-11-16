#! /usr/bin/env python
# -*- coding: utf-8 -*-

# This file is part of IVRE.
# Copyright 2011 - 2022 Pierre LALET <pierre@droids-corp.org>
#
# IVRE is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# IVRE is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public
# License for more details.
#
# You should have received a copy of the GNU General Public License
# along with IVRE. If not, see <http://www.gnu.org/licenses/>.


import hashlib
from typing import Any, Dict, List, Optional

from ivre import utils

HAS_SCAPY = None


# https://datatracker.ietf.org/doc/html/draft-ietf-tls-grease
GREASE = {
    0x0A0A,
    0x1A1A,
    0x2A2A,
    0x3A3A,
    0x4A4A,
    0x5A5A,
    0x6A6A,
    0x7A7A,
    0x8A8A,
    0x9A9A,
    0xAAAA,
    0xBABA,
    0xCACA,
    0xDADA,
    0xEAEA,
    0xFAFA,
}


def banner2ja3c(banner: bytes) -> Optional[str]:
    # "lazy" import for scapy, as this import is slow.
    # TLS is assigned by the import statement, but pylint seems to miss it.
    global HAS_SCAPY, TLS
    if HAS_SCAPY is None:
        try:
            # noqa: E402
            # pylint: disable=import-outside-toplevel
            from scapy.layers.tls.record import TLS  # type: ignore
        except ImportError:
            HAS_SCAPY = False
        else:
            HAS_SCAPY = True
    if not HAS_SCAPY:
        utils.LOGGER.warning("Scapy not found: cannot parse TLS banners")
        return None
    data = TLS(banner)  # type: ignore
    try:
        if data.type != 22:  # handshake
            return None
    except AttributeError:
        return None
    output = []
    for msg in data.msg:
        try:
            if msg.msgtype != 1:  # TLSClientHello
                continue
        except AttributeError:
            utils.LOGGER.warning("Cannot parse TLS message [%r]", msg)
            continue
        output.append(str(msg.version))
        output.append("-".join(str(c) for c in msg.ciphers or [] if c not in GREASE))
        output.append(
            "-".join(str(e.type) for e in msg.ext or [] if e.type not in GREASE)
        )
        ecsg: List[str] = []
        ecpf: List[str] = []
        for ext in msg.ext or []:
            if ext.type == 10:  # supported_groups / elliptic_curves
                ecsg.extend(str(g) for g in ext.groups if g not in GREASE)
            elif ext.type == 11:  # ec_point_formats
                ecpf.extend(str(p) for p in ext.ecpl if p not in GREASE)
        output.append("-".join(ecsg))
        output.append("-".join(ecpf))
        break
    if not output:
        return None
    return ",".join(output)


def banner2script(banner: bytes) -> Optional[Dict[str, Any]]:
    ja3c = banner2ja3c(banner)
    if not ja3c:
        return None
    structured = {"raw": ja3c}
    script: Dict[str, Any] = {"id": "ssl-ja3-client"}
    for hashtype in ["md5", "sha1", "sha256"]:
        structured[hashtype] = hashlib.new(hashtype, ja3c.encode()).hexdigest()
    script["output"] = structured["md5"]
    script["ssl-ja3-client"] = [structured]
    return script
