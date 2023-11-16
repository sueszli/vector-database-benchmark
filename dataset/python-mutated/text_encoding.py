"""Encoding related utilities."""
import re
import six
_cescape_utf8_to_str = [chr(i) for i in range(0, 256)]
_cescape_utf8_to_str[9] = '\\t'
_cescape_utf8_to_str[10] = '\\n'
_cescape_utf8_to_str[13] = '\\r'
_cescape_utf8_to_str[39] = "\\'"
_cescape_utf8_to_str[34] = '\\"'
_cescape_utf8_to_str[92] = '\\\\'
_cescape_byte_to_str = ['\\%03o' % i for i in range(0, 32)] + [chr(i) for i in range(32, 127)] + ['\\%03o' % i for i in range(127, 256)]
_cescape_byte_to_str[9] = '\\t'
_cescape_byte_to_str[10] = '\\n'
_cescape_byte_to_str[13] = '\\r'
_cescape_byte_to_str[39] = "\\'"
_cescape_byte_to_str[34] = '\\"'
_cescape_byte_to_str[92] = '\\\\'

def CEscape(text, as_utf8):
    if False:
        i = 10
        return i + 15
    'Escape a bytes string for use in an ascii protocol buffer.\n\n  text.encode(\'string_escape\') does not seem to satisfy our needs as it\n  encodes unprintable characters using two-digit hex escapes whereas our\n  C++ unescaping function allows hex escapes to be any length.  So,\n  "\x011".encode(\'string_escape\') ends up being "\\x011", which will be\n  decoded in C++ as a single-character string with char code 0x11.\n\n  Args:\n    text: A byte string to be escaped\n    as_utf8: Specifies if result should be returned in UTF-8 encoding\n  Returns:\n    Escaped string\n  '
    Ord = ord if isinstance(text, six.string_types) else lambda x: x
    if as_utf8:
        return ''.join((_cescape_utf8_to_str[Ord(c)] for c in text))
    return ''.join((_cescape_byte_to_str[Ord(c)] for c in text))
_CUNESCAPE_HEX = re.compile('(\\\\+)x([0-9a-fA-F])(?![0-9a-fA-F])')
_cescape_highbit_to_str = [chr(i) for i in range(0, 127)] + ['\\%03o' % i for i in range(127, 256)]

def CUnescape(text):
    if False:
        i = 10
        return i + 15
    'Unescape a text string with C-style escape sequences to UTF-8 bytes.'

    def ReplaceHex(m):
        if False:
            while True:
                i = 10
        if len(m.group(1)) & 1:
            return m.group(1) + 'x0' + m.group(2)
        return m.group(0)
    result = _CUNESCAPE_HEX.sub(ReplaceHex, text)
    if str is bytes:
        return result.decode('string_escape')
    result = ''.join((_cescape_highbit_to_str[ord(c)] for c in result))
    return result.encode('ascii').decode('unicode_escape').encode('raw_unicode_escape')