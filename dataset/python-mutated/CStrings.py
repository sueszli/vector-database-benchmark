""" C string encoding

This contains the code to create string literals for C to represent the given
values.
"""
import codecs
import re
from nuitka.__past__ import unicode

def _identifierEncode(c):
    if False:
        return 10
    'Nuitka handler to encode unicode to ASCII identifiers for C compiler.'
    return ('$%02x$' % ord(c.object[c.end - 1]), c.end)
codecs.register_error('c_identifier', _identifierEncode)

def _encodePythonStringToC(value):
    if False:
        i = 10
        return i + 15
    "Encode a string, so that it gives a C string literal.\n\n    This doesn't handle limits.\n    "
    assert type(value) is bytes, type(value)
    result = ''
    octal = False
    for c in value:
        if str is bytes:
            cv = ord(c)
        else:
            cv = c
        if c in b'\\\t\r\n"?':
            result += '\\%o' % cv
            octal = True
        elif 32 <= cv <= 127:
            if octal and c in b'0123456789':
                result += '" "'
            result += chr(cv)
            octal = False
        else:
            result += '\\%o' % cv
            octal = True
    result = result.replace('" "\\', '\\')
    return '"%s"' % result

def encodePythonUnicodeToC(value):
    if False:
        return 10
    'Encode a string, so that it gives a wide C string literal.'
    assert type(value) is unicode, type(value)
    result = ''
    for c in value:
        cv = ord(c)
        result += '\\%o' % cv
    return 'L"%s"' % result

def encodePythonStringToC(value):
    if False:
        i = 10
        return i + 15
    'Encode bytes, so that it gives a C string literal.'
    result = _encodePythonStringToC(value[:16000])
    value = value[16000:]
    while value:
        result += ' '
        result += _encodePythonStringToC(value[:16000])
        value = value[16000:]
    return result

def encodePythonIdentifierToC(value):
    if False:
        for i in range(10):
            print('nop')
    'Encode an identifier from a given Python string.'

    def r(match):
        if False:
            i = 10
            return i + 15
        c = match.group()
        if c == '.':
            return '$'
        else:
            return '$$%d$' % ord(c)
    return ''.join((re.sub('[^a-zA-Z0-9_]', r, c) for c in value))