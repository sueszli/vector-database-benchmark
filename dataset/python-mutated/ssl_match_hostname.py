"""The match_hostname() function from Python 3.5, essential when using SSL."""
from __future__ import annotations
import ipaddress
import re
import typing
from ipaddress import IPv4Address, IPv6Address
if typing.TYPE_CHECKING:
    from .ssl_ import _TYPE_PEER_CERT_RET_DICT
__version__ = '3.5.0.1'

class CertificateError(ValueError):
    pass

def _dnsname_match(dn: typing.Any, hostname: str, max_wildcards: int=1) -> typing.Match[str] | None | bool:
    if False:
        return 10
    'Matching according to RFC 6125, section 6.4.3\n\n    http://tools.ietf.org/html/rfc6125#section-6.4.3\n    '
    pats = []
    if not dn:
        return False
    parts = dn.split('.')
    leftmost = parts[0]
    remainder = parts[1:]
    wildcards = leftmost.count('*')
    if wildcards > max_wildcards:
        raise CertificateError('too many wildcards in certificate DNS name: ' + repr(dn))
    if not wildcards:
        return bool(dn.lower() == hostname.lower())
    if leftmost == '*':
        pats.append('[^.]+')
    elif leftmost.startswith('xn--') or hostname.startswith('xn--'):
        pats.append(re.escape(leftmost))
    else:
        pats.append(re.escape(leftmost).replace('\\*', '[^.]*'))
    for frag in remainder:
        pats.append(re.escape(frag))
    pat = re.compile('\\A' + '\\.'.join(pats) + '\\Z', re.IGNORECASE)
    return pat.match(hostname)

def _ipaddress_match(ipname: str, host_ip: IPv4Address | IPv6Address) -> bool:
    if False:
        for i in range(10):
            print('nop')
    'Exact matching of IP addresses.\n\n    RFC 9110 section 4.3.5: "A reference identity of IP-ID contains the decoded\n    bytes of the IP address. An IP version 4 address is 4 octets, and an IP\n    version 6 address is 16 octets. [...] A reference identity of type IP-ID\n    matches if the address is identical to an iPAddress value of the\n    subjectAltName extension of the certificate."\n    '
    ip = ipaddress.ip_address(ipname.rstrip())
    return bool(ip.packed == host_ip.packed)

def match_hostname(cert: _TYPE_PEER_CERT_RET_DICT | None, hostname: str, hostname_checks_common_name: bool=False) -> None:
    if False:
        print('Hello World!')
    'Verify that *cert* (in decoded format as returned by\n    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125\n    rules are followed, but IP addresses are not accepted for *hostname*.\n\n    CertificateError is raised on failure. On success, the function\n    returns nothing.\n    '
    if not cert:
        raise ValueError('empty or no certificate, match_hostname needs a SSL socket or SSL context with either CERT_OPTIONAL or CERT_REQUIRED')
    try:
        if '%' in hostname:
            host_ip = ipaddress.ip_address(hostname[:hostname.rfind('%')])
        else:
            host_ip = ipaddress.ip_address(hostname)
    except ValueError:
        host_ip = None
    dnsnames = []
    san: tuple[tuple[str, str], ...] = cert.get('subjectAltName', ())
    key: str
    value: str
    for (key, value) in san:
        if key == 'DNS':
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == 'IP Address':
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)
    if hostname_checks_common_name and host_ip is None and (not dnsnames):
        for sub in cert.get('subject', ()):
            for (key, value) in sub:
                if key == 'commonName':
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)
    if len(dnsnames) > 1:
        raise CertificateError("hostname %r doesn't match either of %s" % (hostname, ', '.join(map(repr, dnsnames))))
    elif len(dnsnames) == 1:
        raise CertificateError(f"hostname {hostname!r} doesn't match {dnsnames[0]!r}")
    else:
        raise CertificateError('no appropriate subjectAltName fields were found')