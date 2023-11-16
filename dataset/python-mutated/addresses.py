from __future__ import annotations
import re
from ansible.errors import AnsibleParserError, AnsibleError
numeric_range = '\n    \\[\n        (?:[0-9]+:[0-9]+)               # numeric begin:end\n        (?::[0-9]+)?                    # numeric :step (optional)\n    \\]\n'
hexadecimal_range = '\n    \\[\n        (?:[0-9a-f]+:[0-9a-f]+)         # hexadecimal begin:end\n        (?::[0-9]+)?                    # numeric :step (optional)\n    \\]\n'
alphanumeric_range = '\n    \\[\n        (?:\n            [a-z]:[a-z]|                # one-char alphabetic range\n            [0-9]+:[0-9]+               # ...or a numeric one\n        )\n        (?::[0-9]+)?                    # numeric :step (optional)\n    \\]\n'
ipv6_component = '\n    (?:\n        [0-9a-f]{{1,4}}|                # 0..ffff\n        {range}                         # or a numeric range\n    )\n'.format(range=hexadecimal_range)
ipv4_component = '\n    (?:\n        [01]?[0-9]{{1,2}}|              # 0..199\n        2[0-4][0-9]|                    # 200..249\n        25[0-5]|                        # 250..255\n        {range}                         # or a numeric range\n    )\n'.format(range=numeric_range)
label = "\n    (?:[\\w]|{range})                    # Starts with an alphanumeric or a range\n    (?:[\\w_-]|{range})*                 # Then zero or more of the same or [_-]\n    (?<![_-])                           # ...as long as it didn't end with [_-]\n".format(range=alphanumeric_range)
patterns = {'bracketed_hostport': re.compile('^\n            \\[(.+)\\]                    # [host identifier]\n            :([0-9]+)                   # :port number\n            $\n        ', re.X), 'hostport': re.compile('^\n            ((?:                        # We want to match:\n                [^:\\[\\]]                # (a non-range character\n                |                       # ...or...\n                \\[[^\\]]*\\]              # a complete bracketed expression)\n            )*)                         # repeated as many times as possible\n            :([0-9]+)                   # followed by a port number\n            $\n        ', re.X), 'ipv4': re.compile('^\n            (?:{i4}\\.){{3}}{i4}         # Three parts followed by dots plus one\n            $\n        '.format(i4=ipv4_component), re.X | re.I), 'ipv6': re.compile('^\n            (?:{0}:){{7}}{0}|           # uncompressed: 1:2:3:4:5:6:7:8\n            (?:{0}:){{1,6}}:|           # compressed variants, which are all\n            (?:{0}:)(?::{0}){{1,6}}|    # a::b for various lengths of a,b\n            (?:{0}:){{2}}(?::{0}){{1,5}}|\n            (?:{0}:){{3}}(?::{0}){{1,4}}|\n            (?:{0}:){{4}}(?::{0}){{1,3}}|\n            (?:{0}:){{5}}(?::{0}){{1,2}}|\n            (?:{0}:){{6}}(?::{0})|      # ...all with 2 <= a+b <= 7\n            :(?::{0}){{1,6}}|           # ::ffff(:ffff...)\n            {0}?::|                     # ffff::, ::\n                                        # ipv4-in-ipv6 variants\n            (?:0:){{6}}(?:{0}\\.){{3}}{0}|\n            ::(?:ffff:)?(?:{0}\\.){{3}}{0}|\n            (?:0:){{5}}ffff:(?:{0}\\.){{3}}{0}\n            $\n        '.format(ipv6_component), re.X | re.I), 'hostname': re.compile('^\n            {label}                     # We must have at least one label\n            (?:\\.{label})*              # Followed by zero or more .labels\n            $\n        '.format(label=label), re.X | re.I | re.UNICODE)}

def parse_address(address, allow_ranges=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Takes a string and returns a (host, port) tuple. If the host is None, then\n    the string could not be parsed as a host identifier with an optional port\n    specification. If the port is None, then no port was specified.\n\n    The host identifier may be a hostname (qualified or not), an IPv4 address,\n    or an IPv6 address. If allow_ranges is True, then any of those may contain\n    [x:y] range specifications, e.g. foo[1:3] or foo[0:5]-bar[x-z].\n\n    The port number is an optional :NN suffix on an IPv4 address or host name,\n    or a mandatory :NN suffix on any square-bracketed expression: IPv6 address,\n    IPv4 address, or host name. (This means the only way to specify a port for\n    an IPv6 address is to enclose it in square brackets.)\n    '
    port = None
    for matching in ['bracketed_hostport', 'hostport']:
        m = patterns[matching].match(address)
        if m:
            (address, port) = m.groups()
            port = int(port)
            continue
    host = None
    for matching in ['ipv4', 'ipv6', 'hostname']:
        m = patterns[matching].match(address)
        if m:
            host = address
            continue
    if not host:
        raise AnsibleError('Not a valid network hostname: %s' % address)
    if not allow_ranges and '[' in host:
        raise AnsibleParserError('Detected range in host but was asked to ignore ranges')
    return (host, port)