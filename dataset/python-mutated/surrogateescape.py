"""
This is Victor Stinner's pure-Python implementation of PEP 383: the "surrogateescape" error
handler of Python 3.

Source: misc/python/surrogateescape.py in https://bitbucket.org/haypo/misc
"""
import codecs
import sys
from future import utils
FS_ERRORS = 'surrogateescape'

def u(text):
    if False:
        for i in range(10):
            print('nop')
    if utils.PY3:
        return text
    else:
        return text.decode('unicode_escape')

def b(data):
    if False:
        while True:
            i = 10
    if utils.PY3:
        return data.encode('latin1')
    else:
        return data
if utils.PY3:
    _unichr = chr
    bytes_chr = lambda code: bytes((code,))
else:
    _unichr = unichr
    bytes_chr = chr

def surrogateescape_handler(exc):
    if False:
        print('Hello World!')
    '\n    Pure Python implementation of the PEP 383: the "surrogateescape" error\n    handler of Python 3. Undecodable bytes will be replaced by a Unicode\n    character U+DCxx on decoding, and these are translated into the\n    original bytes on encoding.\n    '
    mystring = exc.object[exc.start:exc.end]
    try:
        if isinstance(exc, UnicodeDecodeError):
            decoded = replace_surrogate_decode(mystring)
        elif isinstance(exc, UnicodeEncodeError):
            decoded = replace_surrogate_encode(mystring)
        else:
            raise exc
    except NotASurrogateError:
        raise exc
    return (decoded, exc.end)

class NotASurrogateError(Exception):
    pass

def replace_surrogate_encode(mystring):
    if False:
        for i in range(10):
            print('nop')
    '\n    Returns a (unicode) string, not the more logical bytes, because the codecs\n    register_error functionality expects this.\n    '
    decoded = []
    for ch in mystring:
        code = ord(ch)
        if not 55296 <= code <= 56575:
            raise NotASurrogateError
        if 56320 <= code <= 56447:
            decoded.append(_unichr(code - 56320))
        elif code <= 56575:
            decoded.append(_unichr(code - 56320))
        else:
            raise NotASurrogateError
    return str().join(decoded)

def replace_surrogate_decode(mybytes):
    if False:
        i = 10
        return i + 15
    '\n    Returns a (unicode) string\n    '
    decoded = []
    for ch in mybytes:
        if isinstance(ch, int):
            code = ch
        else:
            code = ord(ch)
        if 128 <= code <= 255:
            decoded.append(_unichr(56320 + code))
        elif code <= 127:
            decoded.append(_unichr(code))
        else:
            raise NotASurrogateError
    return str().join(decoded)

def encodefilename(fn):
    if False:
        return 10
    if FS_ENCODING == 'ascii':
        encoded = []
        for (index, ch) in enumerate(fn):
            code = ord(ch)
            if code < 128:
                ch = bytes_chr(code)
            elif 56448 <= code <= 56575:
                ch = bytes_chr(code - 56320)
            else:
                raise UnicodeEncodeError(FS_ENCODING, fn, index, index + 1, 'ordinal not in range(128)')
            encoded.append(ch)
        return bytes().join(encoded)
    elif FS_ENCODING == 'utf-8':
        encoded = []
        for (index, ch) in enumerate(fn):
            code = ord(ch)
            if 55296 <= code <= 57343:
                if 56448 <= code <= 56575:
                    ch = bytes_chr(code - 56320)
                    encoded.append(ch)
                else:
                    raise UnicodeEncodeError(FS_ENCODING, fn, index, index + 1, 'surrogates not allowed')
            else:
                ch_utf8 = ch.encode('utf-8')
                encoded.append(ch_utf8)
        return bytes().join(encoded)
    else:
        return fn.encode(FS_ENCODING, FS_ERRORS)

def decodefilename(fn):
    if False:
        return 10
    return fn.decode(FS_ENCODING, FS_ERRORS)
FS_ENCODING = 'ascii'
fn = b('[abcÿ]')
encoded = u('[abc\udcff]')
FS_ENCODING = codecs.lookup(FS_ENCODING).name

def register_surrogateescape():
    if False:
        print('Hello World!')
    '\n    Registers the surrogateescape error handler on Python 2 (only)\n    '
    if utils.PY3:
        return
    try:
        codecs.lookup_error(FS_ERRORS)
    except LookupError:
        codecs.register_error(FS_ERRORS, surrogateescape_handler)
if __name__ == '__main__':
    pass