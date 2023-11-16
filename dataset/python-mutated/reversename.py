"""DNS Reverse Map Names."""
import binascii
import dns.ipv4
import dns.ipv6
import dns.name
ipv4_reverse_domain = dns.name.from_text('in-addr.arpa.')
ipv6_reverse_domain = dns.name.from_text('ip6.arpa.')

def from_address(text: str, v4_origin: dns.name.Name=ipv4_reverse_domain, v6_origin: dns.name.Name=ipv6_reverse_domain) -> dns.name.Name:
    if False:
        i = 10
        return i + 15
    "Convert an IPv4 or IPv6 address in textual form into a Name object whose\n    value is the reverse-map domain name of the address.\n\n    *text*, a ``str``, is an IPv4 or IPv6 address in textual form\n    (e.g. '127.0.0.1', '::1')\n\n    *v4_origin*, a ``dns.name.Name`` to append to the labels corresponding to\n    the address if the address is an IPv4 address, instead of the default\n    (in-addr.arpa.)\n\n    *v6_origin*, a ``dns.name.Name`` to append to the labels corresponding to\n    the address if the address is an IPv6 address, instead of the default\n    (ip6.arpa.)\n\n    Raises ``dns.exception.SyntaxError`` if the address is badly formed.\n\n    Returns a ``dns.name.Name``.\n    "
    try:
        v6 = dns.ipv6.inet_aton(text)
        if dns.ipv6.is_mapped(v6):
            parts = ['%d' % byte for byte in v6[12:]]
            origin = v4_origin
        else:
            parts = [x for x in str(binascii.hexlify(v6).decode())]
            origin = v6_origin
    except Exception:
        parts = ['%d' % byte for byte in dns.ipv4.inet_aton(text)]
        origin = v4_origin
    return dns.name.from_text('.'.join(reversed(parts)), origin=origin)

def to_address(name: dns.name.Name, v4_origin: dns.name.Name=ipv4_reverse_domain, v6_origin: dns.name.Name=ipv6_reverse_domain) -> str:
    if False:
        print('Hello World!')
    'Convert a reverse map domain name into textual address form.\n\n    *name*, a ``dns.name.Name``, an IPv4 or IPv6 address in reverse-map name\n    form.\n\n    *v4_origin*, a ``dns.name.Name`` representing the top-level domain for\n    IPv4 addresses, instead of the default (in-addr.arpa.)\n\n    *v6_origin*, a ``dns.name.Name`` representing the top-level domain for\n    IPv4 addresses, instead of the default (ip6.arpa.)\n\n    Raises ``dns.exception.SyntaxError`` if the name does not have a\n    reverse-map form.\n\n    Returns a ``str``.\n    '
    if name.is_subdomain(v4_origin):
        name = name.relativize(v4_origin)
        text = b'.'.join(reversed(name.labels))
        return dns.ipv4.inet_ntoa(dns.ipv4.inet_aton(text))
    elif name.is_subdomain(v6_origin):
        name = name.relativize(v6_origin)
        labels = list(reversed(name.labels))
        parts = []
        for i in range(0, len(labels), 4):
            parts.append(b''.join(labels[i:i + 4]))
        text = b':'.join(parts)
        return dns.ipv6.inet_ntoa(dns.ipv6.inet_aton(text))
    else:
        raise dns.exception.SyntaxError('unknown reverse-map address family')