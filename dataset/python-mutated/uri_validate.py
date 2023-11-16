"""
Regex for URIs

These regex are directly derived from the collected ABNF in RFC3986
(except for DIGIT, ALPHA and HEXDIG, defined by RFC2234).

They should be processed with re.VERBOSE.

Thanks Mark Nottingham for this code - https://gist.github.com/138549
"""
import re
DIGIT = '[\\x30-\\x39]'
ALPHA = '[\\x41-\\x5A\\x61-\\x7A]'
HEXDIG = '[\\x30-\\x39A-Fa-f]'
pct_encoded = ' %% %(HEXDIG)s %(HEXDIG)s' % locals()
unreserved = '(?: %(ALPHA)s | %(DIGIT)s | \\- | \\. | _ | ~ )' % locals()
gen_delims = '(?: : | / | \\? | \\# | \\[ | \\] | @ )'
sub_delims = "(?: ! | \\$ | & | ' | \\( | \\) |\n                     \\* | \\+ | , | ; | = )"
pchar = '(?: %(unreserved)s | %(pct_encoded)s | %(sub_delims)s | : | @ )' % locals()
reserved = '(?: %(gen_delims)s | %(sub_delims)s )' % locals()
scheme = '%(ALPHA)s (?: %(ALPHA)s | %(DIGIT)s | \\+ | \\- | \\. )*' % locals()
dec_octet = '(?: %(DIGIT)s |\n                    [\\x31-\\x39] %(DIGIT)s |\n                    1 %(DIGIT)s{2} |\n                    2 [\\x30-\\x34] %(DIGIT)s |\n                    25 [\\x30-\\x35]\n                )\n' % locals()
IPv4address = '%(dec_octet)s \\. %(dec_octet)s \\. %(dec_octet)s \\. %(dec_octet)s' % locals()
IPv6address = '([A-Fa-f0-9:]+:+)+[A-Fa-f0-9]+'
IPvFuture = 'v %(HEXDIG)s+ \\. (?: %(unreserved)s | %(sub_delims)s | : )+' % locals()
IP_literal = '\\[ (?: %(IPv6address)s | %(IPvFuture)s ) \\]' % locals()
reg_name = '(?: %(unreserved)s | %(pct_encoded)s | %(sub_delims)s )*' % locals()
userinfo = '(?: %(unreserved)s | %(pct_encoded)s | %(sub_delims)s | : )' % locals()
host = '(?: %(IP_literal)s | %(IPv4address)s | %(reg_name)s )' % locals()
port = '(?: %(DIGIT)s )*' % locals()
authority = '(?: %(userinfo)s @)? %(host)s (?: : %(port)s)?' % locals()
segment = '%(pchar)s*' % locals()
segment_nz = '%(pchar)s+' % locals()
segment_nz_nc = '(?: %(unreserved)s | %(pct_encoded)s | %(sub_delims)s | @ )+' % locals()
path_abempty = '(?: / %(segment)s )*' % locals()
path_absolute = '/ (?: %(segment_nz)s (?: / %(segment)s )* )?' % locals()
path_noscheme = '%(segment_nz_nc)s (?: / %(segment)s )*' % locals()
path_rootless = '%(segment_nz)s (?: / %(segment)s )*' % locals()
path_empty = ''
path = '(?: %(path_abempty)s |\n               %(path_absolute)s |\n               %(path_noscheme)s |\n               %(path_rootless)s |\n               %(path_empty)s\n            )\n' % locals()
query = '(?: %(pchar)s | / | \\? )*' % locals()
fragment = '(?: %(pchar)s | / | \\? )*' % locals()
hier_part = '(?: (?: // %(authority)s %(path_abempty)s ) |\n                    %(path_absolute)s |\n                    %(path_rootless)s |\n                    %(path_empty)s\n                )\n' % locals()
relative_part = '(?: (?: // %(authority)s %(path_abempty)s ) |\n                        %(path_absolute)s |\n                        %(path_noscheme)s |\n                        %(path_empty)s\n                    )\n' % locals()
relative_ref = '%(relative_part)s (?: \\? %(query)s)? (?: \\# %(fragment)s)?' % locals()
URI = '^(?: %(scheme)s : %(hier_part)s (?: \\? %(query)s )? (?: \\# %(fragment)s )? )$' % locals()
URI_reference = '^(?: %(URI)s | %(relative_ref)s )$' % locals()
absolute_URI = '^(?: %(scheme)s : %(hier_part)s (?: \\? %(query)s )? )$' % locals()

def is_uri(uri):
    if False:
        while True:
            i = 10
    return re.match(URI, uri, re.VERBOSE)

def is_uri_reference(uri):
    if False:
        while True:
            i = 10
    return re.match(URI_reference, uri, re.VERBOSE)

def is_absolute_uri(uri):
    if False:
        print('Hello World!')
    return re.match(absolute_URI, uri, re.VERBOSE)