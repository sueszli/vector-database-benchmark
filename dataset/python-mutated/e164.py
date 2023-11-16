"""DNS E.164 helpers."""
from typing import Iterable, Optional, Union
import dns.exception
import dns.name
import dns.resolver
public_enum_domain = dns.name.from_text('e164.arpa.')

def from_e164(text: str, origin: Optional[dns.name.Name]=public_enum_domain) -> dns.name.Name:
    if False:
        print('Hello World!')
    'Convert an E.164 number in textual form into a Name object whose\n    value is the ENUM domain name for that number.\n\n    Non-digits in the text are ignored, i.e. "16505551212",\n    "+1.650.555.1212" and "1 (650) 555-1212" are all the same.\n\n    *text*, a ``str``, is an E.164 number in textual form.\n\n    *origin*, a ``dns.name.Name``, the domain in which the number\n    should be constructed.  The default is ``e164.arpa.``.\n\n    Returns a ``dns.name.Name``.\n    '
    parts = [d for d in text if d.isdigit()]
    parts.reverse()
    return dns.name.from_text('.'.join(parts), origin=origin)

def to_e164(name: dns.name.Name, origin: Optional[dns.name.Name]=public_enum_domain, want_plus_prefix: bool=True) -> str:
    if False:
        i = 10
        return i + 15
    "Convert an ENUM domain name into an E.164 number.\n\n    Note that dnspython does not have any information about preferred\n    number formats within national numbering plans, so all numbers are\n    emitted as a simple string of digits, prefixed by a '+' (unless\n    *want_plus_prefix* is ``False``).\n\n    *name* is a ``dns.name.Name``, the ENUM domain name.\n\n    *origin* is a ``dns.name.Name``, a domain containing the ENUM\n    domain name.  The name is relativized to this domain before being\n    converted to text.  If ``None``, no relativization is done.\n\n    *want_plus_prefix* is a ``bool``.  If True, add a '+' to the beginning of\n    the returned number.\n\n    Returns a ``str``.\n\n    "
    if origin is not None:
        name = name.relativize(origin)
    dlabels = [d for d in name.labels if d.isdigit() and len(d) == 1]
    if len(dlabels) != len(name.labels):
        raise dns.exception.SyntaxError('non-digit labels in ENUM domain name')
    dlabels.reverse()
    text = b''.join(dlabels)
    if want_plus_prefix:
        text = b'+' + text
    return text.decode()

def query(number: str, domains: Iterable[Union[dns.name.Name, str]], resolver: Optional[dns.resolver.Resolver]=None) -> dns.resolver.Answer:
    if False:
        return 10
    "Look for NAPTR RRs for the specified number in the specified domains.\n\n    e.g. lookup('16505551212', ['e164.dnspython.org.', 'e164.arpa.'])\n\n    *number*, a ``str`` is the number to look for.\n\n    *domains* is an iterable containing ``dns.name.Name`` values.\n\n    *resolver*, a ``dns.resolver.Resolver``, is the resolver to use.  If\n    ``None``, the default resolver is used.\n    "
    if resolver is None:
        resolver = dns.resolver.get_default_resolver()
    e_nx = dns.resolver.NXDOMAIN()
    for domain in domains:
        if isinstance(domain, str):
            domain = dns.name.from_text(domain)
        qname = dns.e164.from_e164(number, domain)
        try:
            return resolver.resolve(qname, 'NAPTR')
        except dns.resolver.NXDOMAIN as e:
            e_nx += e
    raise e_nx