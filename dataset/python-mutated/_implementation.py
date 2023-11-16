"""The match_hostname() function from Python 3.3.3, essential when using SSL."""
import re
import sys
try:
    from pip._vendor import ipaddress
except ImportError:
    ipaddress = None
__version__ = '3.5.0.1'

class CertificateError(ValueError):
    pass

def _dnsname_match(dn, hostname, max_wildcards=1):
    if False:
        print('Hello World!')
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
        return dn.lower() == hostname.lower()
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

def _to_unicode(obj):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(obj, str) and sys.version_info < (3,):
        obj = unicode(obj, encoding='ascii', errors='strict')
    return obj

def _ipaddress_match(ipname, host_ip):
    if False:
        return 10
    'Exact matching of IP addresses.\n\n    RFC 6125 explicitly doesn\'t define an algorithm for this\n    (section 1.7.2 - "Out of Scope").\n    '
    ip = ipaddress.ip_address(_to_unicode(ipname).rstrip())
    return ip == host_ip

def match_hostname(cert, hostname):
    if False:
        print('Hello World!')
    'Verify that *cert* (in decoded format as returned by\n    SSLSocket.getpeercert()) matches the *hostname*.  RFC 2818 and RFC 6125\n    rules are followed, but IP addresses are not accepted for *hostname*.\n\n    CertificateError is raised on failure. On success, the function\n    returns nothing.\n    '
    if not cert:
        raise ValueError('empty or no certificate, match_hostname needs a SSL socket or SSL context with either CERT_OPTIONAL or CERT_REQUIRED')
    try:
        host_ip = ipaddress.ip_address(_to_unicode(hostname))
    except ValueError:
        host_ip = None
    except UnicodeError:
        host_ip = None
    except AttributeError:
        if ipaddress is None:
            host_ip = None
        else:
            raise
    dnsnames = []
    san = cert.get('subjectAltName', ())
    for (key, value) in san:
        if key == 'DNS':
            if host_ip is None and _dnsname_match(value, hostname):
                return
            dnsnames.append(value)
        elif key == 'IP Address':
            if host_ip is not None and _ipaddress_match(value, host_ip):
                return
            dnsnames.append(value)
    if not dnsnames:
        for sub in cert.get('subject', ()):
            for (key, value) in sub:
                if key == 'commonName':
                    if _dnsname_match(value, hostname):
                        return
                    dnsnames.append(value)
    if len(dnsnames) > 1:
        raise CertificateError("hostname %r doesn't match either of %s" % (hostname, ', '.join(map(repr, dnsnames))))
    elif len(dnsnames) == 1:
        raise CertificateError("hostname %r doesn't match %r" % (hostname, dnsnames[0]))
    else:
        raise CertificateError('no appropriate commonName or subjectAltName fields were found')