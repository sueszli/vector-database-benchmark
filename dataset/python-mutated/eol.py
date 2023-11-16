"""End of Line Conversion filters.

See bzr help eol for details.
"""
from __future__ import absolute_import
import re, sys
from bzrlib.errors import BzrError
from bzrlib.filters import ContentFilter
_UNIX_NL_RE = re.compile('(?<!\\r)\\n')

def _to_lf_converter(chunks, context=None):
    if False:
        i = 10
        return i + 15
    'A content file that converts crlf to lf.'
    content = ''.join(chunks)
    if '\x00' in content:
        return [content]
    else:
        return [content.replace('\r\n', '\n')]

def _to_crlf_converter(chunks, context=None):
    if False:
        return 10
    'A content file that converts lf to crlf.'
    content = ''.join(chunks)
    if '\x00' in content:
        return [content]
    else:
        return [_UNIX_NL_RE.sub('\r\n', content)]
if sys.platform == 'win32':
    _native_output = _to_crlf_converter
else:
    _native_output = _to_lf_converter
_eol_filter_stack_map = {'exact': [], 'native': [ContentFilter(_to_lf_converter, _native_output)], 'lf': [ContentFilter(_to_lf_converter, _to_lf_converter)], 'crlf': [ContentFilter(_to_lf_converter, _to_crlf_converter)], 'native-with-crlf-in-repo': [ContentFilter(_to_crlf_converter, _native_output)], 'lf-with-crlf-in-repo': [ContentFilter(_to_crlf_converter, _to_lf_converter)], 'crlf-with-crlf-in-repo': [ContentFilter(_to_crlf_converter, _to_crlf_converter)]}

def eol_lookup(key):
    if False:
        return 10
    filter = _eol_filter_stack_map.get(key)
    if filter is None:
        raise BzrError("Unknown eol value '%s'" % key)
    return filter