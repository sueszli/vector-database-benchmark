"""Some functions to enable caching the conversion between unicode to utf8"""
from __future__ import absolute_import
import codecs
_utf8_encode = codecs.utf_8_encode
_utf8_decode = codecs.utf_8_decode

def _utf8_decode_with_None(bytestring, _utf8_decode=_utf8_decode):
    if False:
        while True:
            i = 10
    "wrap _utf8_decode to support None->None for optional strings.\n\n    Also, only return the Unicode portion, since we don't care about the second\n    return value.\n    "
    if bytestring is None:
        return None
    else:
        return _utf8_decode(bytestring)[0]
_unicode_to_utf8_map = {}
_utf8_to_unicode_map = {}

def encode(unicode_str, _uni_to_utf8=_unicode_to_utf8_map, _utf8_to_uni=_utf8_to_unicode_map, _utf8_encode=_utf8_encode):
    if False:
        while True:
            i = 10
    'Take this unicode revision id, and get a unicode version'
    try:
        return _uni_to_utf8[unicode_str]
    except KeyError:
        _uni_to_utf8[unicode_str] = utf8_str = _utf8_encode(unicode_str)[0]
        _utf8_to_uni[utf8_str] = unicode_str
        return utf8_str

def decode(utf8_str, _uni_to_utf8=_unicode_to_utf8_map, _utf8_to_uni=_utf8_to_unicode_map, _utf8_decode=_utf8_decode):
    if False:
        print('Hello World!')
    'Take a utf8 revision id, and decode it, but cache the result'
    try:
        return _utf8_to_uni[utf8_str]
    except KeyError:
        unicode_str = _utf8_decode(utf8_str)[0]
        _utf8_to_uni[utf8_str] = unicode_str
        _uni_to_utf8[unicode_str] = utf8_str
        return unicode_str

def get_cached_unicode(unicode_str):
    if False:
        for i in range(10):
            print('nop')
    'Return a cached version of the unicode string.\n\n    This has a similar idea to that of intern() in that it tries\n    to return a singleton string. Only it works for unicode strings.\n    '
    return decode(encode(unicode_str))

def get_cached_utf8(utf8_str):
    if False:
        while True:
            i = 10
    'Return a cached version of the utf-8 string.\n\n    Get a cached version of this string (similar to intern()).\n    At present, this will be decoded to ensure it is a utf-8 string. In the\n    future this might change to simply caching the string.\n    '
    return encode(decode(utf8_str))

def get_cached_ascii(ascii_str, _uni_to_utf8=_unicode_to_utf8_map, _utf8_to_uni=_utf8_to_unicode_map):
    if False:
        i = 10
        return i + 15
    'This is a string which is identical in utf-8 and unicode.'
    ascii_str = _uni_to_utf8.setdefault(ascii_str, ascii_str)
    _utf8_to_uni.setdefault(ascii_str, unicode(ascii_str))
    return ascii_str

def clear_encoding_cache():
    if False:
        for i in range(10):
            print('nop')
    'Clear the encoding and decoding caches'
    _unicode_to_utf8_map.clear()
    _utf8_to_unicode_map.clear()