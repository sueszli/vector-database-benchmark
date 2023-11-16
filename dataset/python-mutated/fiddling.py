from __future__ import absolute_import
from __future__ import division
import base64
import binascii
import random
import re
import os
import six
import string
from six import BytesIO
from six.moves import range
from pwnlib.context import LocalNoarchContext
from pwnlib.context import context
from pwnlib.log import getLogger
from pwnlib.term import text
from pwnlib.util import iters
from pwnlib.util import lists
from pwnlib.util import packing
from pwnlib.util.cyclic import cyclic
from pwnlib.util.cyclic import de_bruijn
from pwnlib.util.cyclic import cyclic_find
log = getLogger(__name__)

def unhex(s):
    if False:
        while True:
            i = 10
    'unhex(s) -> str\n\n    Hex-decodes a string.\n\n    Example:\n\n        >>> unhex("74657374")\n        b\'test\'\n        >>> unhex("F\\n")\n        b\'\\x0f\'\n    '
    s = s.strip()
    if len(s) % 2 != 0:
        s = '0' + s
    return binascii.unhexlify(s)

def enhex(x):
    if False:
        i = 10
        return i + 15
    'enhex(x) -> str\n\n    Hex-encodes a string.\n\n    Example:\n\n        >>> enhex(b"test")\n        \'74657374\'\n    '
    x = binascii.hexlify(x)
    if not hasattr(x, 'encode'):
        x = x.decode('ascii')
    return x

def urlencode(s):
    if False:
        for i in range(10):
            print('nop')
    'urlencode(s) -> str\n\n    URL-encodes a string.\n\n    Example:\n\n        >>> urlencode("test")\n        \'%74%65%73%74\'\n    '
    return ''.join(['%%%02x' % ord(c) for c in s])

def urldecode(s, ignore_invalid=False):
    if False:
        while True:
            i = 10
    'urldecode(s, ignore_invalid = False) -> str\n\n    URL-decodes a string.\n\n    Example:\n\n        >>> urldecode("test%20%41")\n        \'test A\'\n        >>> urldecode("%qq")\n        Traceback (most recent call last):\n        ...\n        ValueError: Invalid input to urldecode\n        >>> urldecode("%qq", ignore_invalid = True)\n        \'%qq\'\n    '
    res = ''
    n = 0
    while n < len(s):
        if s[n] != '%':
            res += s[n]
            n += 1
        else:
            cur = s[n + 1:n + 3]
            if re.match('[0-9a-fA-F]{2}', cur):
                res += chr(int(cur, 16))
                n += 3
            elif ignore_invalid:
                res += '%'
                n += 1
            else:
                raise ValueError('Invalid input to urldecode')
    return res

def bits(s, endian='big', zero=0, one=1):
    if False:
        return 10
    'bits(s, endian = \'big\', zero = 0, one = 1) -> list\n\n    Converts the argument into a list of bits.\n\n    Arguments:\n        s: A string or number to be converted into bits.\n        endian (str): The binary endian, default \'big\'.\n        zero: The representing a 0-bit.\n        one: The representing a 1-bit.\n\n    Returns:\n        A list consisting of the values specified in `zero` and `one`.\n\n    Examples:\n\n        >>> bits(511, zero = "+", one = "-")\n        [\'+\', \'+\', \'+\', \'+\', \'+\', \'+\', \'+\', \'-\', \'-\', \'-\', \'-\', \'-\', \'-\', \'-\', \'-\', \'-\']\n        >>> sum(bits(b"test"))\n        17\n        >>> bits(0)\n        [0, 0, 0, 0, 0, 0, 0, 0]\n    '
    if endian not in ['little', 'big']:
        raise ValueError("bits(): 'endian' must be either 'little' or 'big'")
    else:
        little = endian == 'little'
    out = []
    if isinstance(s, bytes):
        for b in bytearray(s):
            byte = []
            for _ in range(8):
                byte.append(one if b & 1 else zero)
                b >>= 1
            if little:
                out += byte
            else:
                out += byte[::-1]
    elif isinstance(s, six.integer_types):
        if s < 0:
            s = s & (1 << context.bits) - 1
        if s == 0:
            out.append(zero)
        while s:
            (bit, s) = (one if s & 1 else zero, s >> 1)
            out.append(bit)
        while len(out) % 8:
            out.append(zero)
        if not little:
            out = out[::-1]
    else:
        raise ValueError("bits(): 's' must be either a string or a number")
    return out

def bits_str(s, endian='big', zero='0', one='1'):
    if False:
        return 10
    'bits_str(s, endian = \'big\', zero = \'0\', one = \'1\') -> str\n\n    A wrapper around :func:`bits`, which converts the output into a string.\n\n    Examples:\n\n       >>> bits_str(511)\n       \'0000000111111111\'\n       >>> bits_str(b"bits_str", endian = "little")\n       \'0100011010010110001011101100111011111010110011100010111001001110\'\n    '
    return ''.join(bits(s, endian, zero, one))

def unbits(s, endian='big'):
    if False:
        print('Hello World!')
    'unbits(s, endian = \'big\') -> str\n\n    Converts an iterable of bits into a string.\n\n    Arguments:\n       s: Iterable of bits\n       endian (str):  The string "little" or "big", which specifies the bits endianness.\n\n    Returns:\n       A string of the decoded bits.\n\n    Example:\n       >>> unbits([1])\n       b\'\\x80\'\n       >>> unbits([1], endian = \'little\')\n       b\'\\x01\'\n       >>> unbits(bits(b\'hello\'), endian = \'little\')\n       b\'\\x16\\xa666\\xf6\'\n    '
    if endian == 'little':
        u = lambda s: packing._p8lu(int(s[::-1], 2))
    elif endian == 'big':
        u = lambda s: packing._p8lu(int(s, 2))
    else:
        raise ValueError("unbits(): 'endian' must be either 'little' or 'big'")
    out = b''
    cur = b''
    for c in s:
        if c in ['1', 1, True]:
            cur += b'1'
        elif c in ['0', 0, False]:
            cur += b'0'
        else:
            raise ValueError('unbits(): cannot decode the value %r into a bit' % c)
        if len(cur) == 8:
            out += u(cur)
            cur = b''
    if cur:
        out += u(cur.ljust(8, b'0'))
    return out

def bitswap(s):
    if False:
        while True:
            i = 10
    'bitswap(s) -> str\n\n    Reverses the bits in every byte of a given string.\n\n    Example:\n        >>> bitswap(b"1234")\n        b\'\\x8cL\\xcc,\'\n    '
    out = []
    for c in s:
        out.append(unbits(bits_str(c)[::-1]))
    return b''.join(out)

def bitswap_int(n, width):
    if False:
        print('Hello World!')
    "bitswap_int(n) -> int\n\n    Reverses the bits of a numbers and returns the result as a new number.\n\n    Arguments:\n        n (int): The number to swap.\n        width (int): The width of the integer\n\n    Examples:\n        >>> hex(bitswap_int(0x1234, 8))\n        '0x2c'\n        >>> hex(bitswap_int(0x1234, 16))\n        '0x2c48'\n        >>> hex(bitswap_int(0x1234, 24))\n        '0x2c4800'\n        >>> hex(bitswap_int(0x1234, 25))\n        '0x589000'\n    "
    n &= (1 << width) - 1
    s = bits_str(n, endian='little').ljust(width, '0')[:width]
    return int(s, 2)

def b64e(s):
    if False:
        i = 10
        return i + 15
    'b64e(s) -> str\n\n    Base64 encodes a string\n\n    Example:\n\n       >>> b64e(b"test")\n       \'dGVzdA==\'\n       '
    x = base64.b64encode(s)
    if not hasattr(x, 'encode'):
        x = x.decode('ascii')
    return x

def b64d(s):
    if False:
        i = 10
        return i + 15
    "b64d(s) -> str\n\n    Base64 decodes a string\n\n    Example:\n\n       >>> b64d('dGVzdA==')\n       b'test'\n    "
    return base64.b64decode(s)

def xor(*args, **kwargs):
    if False:
        print('Hello World!')
    "xor(*args, cut = 'max') -> str\n\n    Flattens its arguments using :func:`pwnlib.util.packing.flat` and\n    then xors them together. If the end of a string is reached, it wraps\n    around in the string.\n\n    Arguments:\n       args: The arguments to be xor'ed together.\n       cut: How long a string should be returned.\n            Can be either 'min'/'max'/'left'/'right' or a number.\n\n    Returns:\n       The string of the arguments xor'ed together.\n\n    Example:\n       >>> xor(b'lol', b'hello', 42)\n       b'. ***'\n    "
    cut = kwargs.pop('cut', 'max')
    if kwargs != {}:
        raise TypeError("xor() got an unexpected keyword argument '%s'" % kwargs.pop()[0])
    if len(args) == 0:
        raise ValueError('Must have something to xor')
    strs = [packing.flat(s, word_size=8, sign=False, endianness='little') for s in args]
    strs = [bytearray(s) for s in strs if s]
    if strs == []:
        return b''
    if isinstance(cut, six.integer_types):
        cut = cut
    elif cut == 'left':
        cut = len(strs[0])
    elif cut == 'right':
        cut = len(strs[-1])
    elif cut == 'min':
        cut = min((len(s) for s in strs))
    elif cut == 'max':
        cut = max((len(s) for s in strs))
    else:
        raise ValueError("Not a valid argument for 'cut'")

    def get(n):
        if False:
            return 10
        rv = 0
        for s in strs:
            rv ^= s[n % len(s)]
        return packing._p8lu(rv)
    return b''.join(map(get, range(cut)))

def xor_pair(data, avoid=b'\x00\n'):
    if False:
        while True:
            i = 10
    'xor_pair(data, avoid = \'\\x00\\n\') -> None or (str, str)\n\n    Finds two strings that will xor into a given string, while only\n    using a given alphabet.\n\n    Arguments:\n        data (str): The desired string.\n        avoid: The list of disallowed characters. Defaults to nulls and newlines.\n\n    Returns:\n        Two strings which will xor to the given string. If no such two strings exist, then None is returned.\n\n    Example:\n\n        >>> xor_pair(b"test")\n        (b\'\\x01\\x01\\x01\\x01\', b\'udru\')\n    '
    if isinstance(data, six.integer_types):
        data = packing.pack(data)
    if not isinstance(avoid, (bytes, bytearray)):
        avoid = avoid.encode('utf-8')
    avoid = bytearray(avoid)
    alphabet = list((packing._p8lu(n) for n in range(256) if n not in avoid))
    res1 = b''
    res2 = b''
    for c1 in bytearray(data):
        if context.randomize:
            random.shuffle(alphabet)
        for c2 in alphabet:
            c3 = packing._p8lu(c1 ^ packing.u8(c2))
            if c3 in alphabet:
                res1 += c2
                res2 += c3
                break
        else:
            return None
    return (res1, res2)

def xor_key(data, avoid=b'\x00\n', size=None):
    if False:
        print('Hello World!')
    'xor_key(data, size=None, avoid=\'\\x00\\n\') -> None or (int, str)\n\n    Finds a ``size``-width value that can be XORed with a string\n    to produce ``data``, while neither the XOR value or XOR string\n    contain any bytes in ``avoid``.\n\n    Arguments:\n        data (str): The desired string.\n        avoid: The list of disallowed characters. Defaults to nulls and newlines.\n        size (int): Size of the desired output value, default is word size.\n\n    Returns:\n        A tuple containing two strings; the XOR key and the XOR string.\n        If no such pair exists, None is returned.\n\n    Example:\n\n        >>> xor_key(b"Hello, world")\n        (b\'\\x01\\x01\\x01\\x01\', b\'Idmmn-!vnsme\')\n    '
    size = size or context.bytes
    if len(data) % size:
        log.error('Data must be padded to size for xor_key')
    words = lists.group(size, data)
    columns = [b''] * size
    for word in words:
        for (i, byte) in enumerate(bytearray(word)):
            columns[i] += bytearray((byte,))
    avoid = bytearray(avoid)
    alphabet = bytearray((n for n in range(256) if n not in avoid))
    result = b''
    for column in columns:
        if context.randomize:
            random.shuffle(alphabet)
        for c2 in alphabet:
            if all((c ^ c2 in alphabet for c in column)):
                result += packing._p8lu(c2)
                break
        else:
            return None
    return (result, xor(data, result))

def randoms(count, alphabet=string.ascii_lowercase):
    if False:
        while True:
            i = 10
    "randoms(count, alphabet = string.ascii_lowercase) -> str\n\n    Returns a random string of a given length using only the specified alphabet.\n\n    Arguments:\n        count (int): The length of the desired string.\n        alphabet: The alphabet of allowed characters. Defaults to all lowercase characters.\n\n    Returns:\n        A random string.\n\n    Example:\n\n        >>> randoms(10) #doctest: +SKIP\n        'evafjilupm'\n    "
    return ''.join((random.choice(alphabet) for _ in range(count)))

def rol(n, k, word_size=None):
    if False:
        i = 10
        return i + 15
    "Returns a rotation by `k` of `n`.\n\n    When `n` is a number, then means ``((n << k) | (n >> (word_size - k)))`` truncated to `word_size` bits.\n\n    When `n` is a list, tuple or string, this is ``n[k % len(n):] + n[:k % len(n)]``.\n\n    Arguments:\n        n: The value to rotate.\n        k(int): The rotation amount. Can be a positive or negative number.\n        word_size(int): If `n` is a number, then this is the assumed bitsize of `n`.  Defaults to :data:`pwnlib.context.word_size` if `None` .\n\n    Example:\n\n        >>> rol('abcdefg', 2)\n        'cdefgab'\n        >>> rol('abcdefg', -2)\n        'fgabcde'\n        >>> hex(rol(0x86, 3, 8))\n        '0x34'\n        >>> hex(rol(0x86, -3, 8))\n        '0xd0'\n    "
    word_size = word_size or context.word_size
    if not isinstance(word_size, six.integer_types) or word_size <= 0:
        raise ValueError("rol(): 'word_size' must be a strictly positive integer")
    if not isinstance(k, six.integer_types):
        raise ValueError("rol(): 'k' must be an integer")
    if isinstance(n, (bytes, six.text_type, list, tuple)):
        return n[k % len(n):] + n[:k % len(n)]
    elif isinstance(n, six.integer_types):
        k = k % word_size
        n = n << k | n >> word_size - k
        n &= (1 << word_size) - 1
        return n
    else:
        raise ValueError("rol(): 'n' must be an integer, string, list or tuple")

def ror(n, k, word_size=None):
    if False:
        i = 10
        return i + 15
    'A simple wrapper around :func:`rol`, which negates the values of `k`.'
    return rol(n, -k, word_size)

def naf(n):
    if False:
        i = 10
        return i + 15
    'naf(int) -> int generator\n\n    Returns a generator for the non-adjacent form (NAF[1]) of a number, `n`.  If\n    `naf(n)` generates `z_0, z_1, ...`, then `n == z_0 + z_1 * 2 + z_2 * 2**2,\n    ...`.\n\n    [1] https://en.wikipedia.org/wiki/Non-adjacent_form\n\n    Example:\n\n      >>> n = 45\n      >>> m = 0\n      >>> x = 1\n      >>> for z in naf(n):\n      ...     m += x * z\n      ...     x *= 2\n      >>> n == m\n      True\n\n    '
    while n:
        z = 2 - n % 4 if n & 1 else 0
        n = (n - z) // 2
        yield z

def isprint(c):
    if False:
        return 10
    'isprint(c) -> bool\n\n    Return True if a character is printable'
    if isinstance(c, six.text_type):
        c = ord(c)
    t = bytearray(string.ascii_letters + string.digits + string.punctuation + ' ', 'ascii')
    return c in t

def hexii(s, width=16, skip=True):
    if False:
        while True:
            i = 10
    'hexii(s, width = 16, skip = True) -> str\n\n    Return a HEXII-dump of a string.\n\n    Arguments:\n        s(str): The string to dump\n        width(int): The number of characters per line\n        skip(bool): Should repeated lines be replaced by a "*"\n\n    Returns:\n        A HEXII-dump in the form of a string.\n    '
    return hexdump(s, width, skip, True)

def _hexiichar(c):
    if False:
        return 10
    HEXII = bytearray((string.punctuation + string.digits + string.ascii_letters).encode())
    if c in HEXII:
        return '.%c ' % c
    elif c == 0:
        return '   '
    elif c == 255:
        return '## '
    else:
        return '%02x ' % c
default_style = {'marker': text.gray if text.has_gray else text.blue, 'nonprintable': text.gray if text.has_gray else text.blue, '00': text.red, '0a': text.red, 'ff': text.green}
cyclic_pregen = b''
de_bruijn_gen = de_bruijn()

def sequential_lines(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a + b in cyclic_pregen

def update_cyclic_pregenerated(size):
    if False:
        i = 10
        return i + 15
    global cyclic_pregen
    while size > len(cyclic_pregen):
        cyclic_pregen += packing._p8lu(next(de_bruijn_gen))

def hexdump_iter(fd, width=16, skip=True, hexii=False, begin=0, style=None, highlight=None, cyclic=False, groupsize=4, total=True):
    if False:
        print('Hello World!')
    'hexdump_iter(s, width = 16, skip = True, hexii = False, begin = 0, style = None,\n                    highlight = None, cyclic = False, groupsize=4, total = True) -> str generator\n\n    Return a hexdump-dump of a string as a generator of lines.  Unless you have\n    massive amounts of data you probably want to use :meth:`hexdump`.\n\n    Arguments:\n        fd(file): File object to dump.  Use :meth:`StringIO.StringIO` or :meth:`hexdump` to dump a string.\n        width(int): The number of characters per line\n        groupsize(int): The number of characters per group\n        skip(bool): Set to True, if repeated lines should be replaced by a "*"\n        hexii(bool): Set to True, if a hexii-dump should be returned instead of a hexdump.\n        begin(int):  Offset of the first byte to print in the left column\n        style(dict): Color scheme to use.\n        highlight(iterable): Byte values to highlight.\n        cyclic(bool): Attempt to skip consecutive, unmodified cyclic lines\n        total(bool): Set to True, if total bytes should be printed\n\n    Returns:\n        A generator producing the hexdump-dump one line at a time.\n\n    Example:\n\n        >>> tmp = tempfile.NamedTemporaryFile()\n        >>> _ = tmp.write(b\'XXXXHELLO, WORLD\')\n        >>> tmp.flush()\n        >>> _ = tmp.seek(4)\n        >>> print(\'\\n\'.join(hexdump_iter(tmp)))\n        00000000  48 45 4c 4c  4f 2c 20 57  4f 52 4c 44               │HELL│O, W│ORLD│\n        0000000c\n\n        >>> t = tube()\n        >>> t.unrecv(b\'I know kung fu\')\n        >>> print(\'\\n\'.join(hexdump_iter(t)))\n        00000000  49 20 6b 6e  6f 77 20 6b  75 6e 67 20  66 75        │I kn│ow k│ung │fu│\n        0000000e\n    '
    style = style or {}
    highlight = highlight or []
    if groupsize < 1:
        groupsize = width
    for b in highlight:
        if isinstance(b, str):
            b = ord(b)
        style['%02x' % b] = text.white_on_red
    _style = style
    style = default_style.copy()
    style.update(_style)
    skipping = False
    lines = []
    last_unique = ''
    byte_width = len('00 ')
    spacer = ' '
    marker = (style.get('marker') or (lambda s: s))('│')
    if not hexii:

        def style_byte(by):
            if False:
                while True:
                    i = 10
            hbyte = '%02x' % by
            b = packing._p8lu(by)
            abyte = chr(by) if isprint(b) else '·'
            if hbyte in style:
                st = style[hbyte]
            elif isprint(b):
                st = style.get('printable')
            else:
                st = style.get('nonprintable')
            if st:
                hbyte = st(hbyte)
                abyte = st(abyte)
            return (hbyte, abyte)
        cache = [style_byte(b) for b in range(256)]
    numb = 0
    while True:
        offset = begin + numb
        try:
            chunk = fd.read(width)
        except EOFError:
            chunk = b''
        if chunk == b'':
            break
        numb += len(chunk)
        if cyclic:
            update_cyclic_pregenerated(numb)
        if skip and last_unique:
            same_as_last_line = last_unique == chunk
            lines_are_sequential = cyclic and sequential_lines(last_unique, chunk)
            last_unique = chunk
            if same_as_last_line or lines_are_sequential:
                if not skipping:
                    yield '*'
                    skipping = True
                continue
        skipping = False
        last_unique = chunk
        hexbytes = ''
        printable = ''
        color_chars = 0
        abyte = abyte_previous = ''
        for (i, b) in enumerate(bytearray(chunk)):
            if not hexii:
                abyte_previous = abyte
                (hbyte, abyte) = cache[b]
                color_chars += len(hbyte) - 2
            else:
                (hbyte, abyte) = (_hexiichar(b), '')
            if (i + 1) % groupsize == 0 and i < width - 1:
                hbyte += spacer
                abyte_previous += abyte
                abyte = marker
            hexbytes += hbyte + ' '
            printable += abyte_previous
        if abyte != marker:
            printable += abyte
        dividers_per_line = width // groupsize
        if width % groupsize == 0:
            dividers_per_line -= 1
        if hexii:
            line_fmt = '%%(offset)08x  %%(hexbytes)-%is│' % (width * byte_width)
        else:
            line_fmt = '%%(offset)08x  %%(hexbytes)-%is │%%(printable)s│' % (width * byte_width + color_chars + dividers_per_line)
        line = line_fmt % {'offset': offset, 'hexbytes': hexbytes, 'printable': printable}
        yield line
    if total:
        line = '%08x' % (begin + numb)
        yield line

def hexdump(s, width=16, skip=True, hexii=False, begin=0, style=None, highlight=None, cyclic=False, groupsize=4, total=True):
    if False:
        i = 10
        return i + 15
    'hexdump(s, width = 16, skip = True, hexii = False, begin = 0, style = None,\n                highlight = None, cyclic = False, groupsize=4, total = True) -> str\n\n    Return a hexdump-dump of a string.\n\n    Arguments:\n        s(bytes): The data to hexdump.\n        width(int): The number of characters per line\n        groupsize(int): The number of characters per group\n        skip(bool): Set to True, if repeated lines should be replaced by a "*"\n        hexii(bool): Set to True, if a hexii-dump should be returned instead of a hexdump.\n        begin(int):  Offset of the first byte to print in the left column\n        style(dict): Color scheme to use.\n        highlight(iterable): Byte values to highlight.\n        cyclic(bool): Attempt to skip consecutive, unmodified cyclic lines\n        total(bool): Set to True, if total bytes should be printed\n\n    Returns:\n        A hexdump-dump in the form of a string.\n\n    Examples:\n\n        >>> print(hexdump(b"abc"))\n        00000000  61 62 63                                            │abc│\n        00000003\n\n        >>> print(hexdump(b\'A\'*32))\n        00000000  41 41 41 41  41 41 41 41  41 41 41 41  41 41 41 41  │AAAA│AAAA│AAAA│AAAA│\n        *\n        00000020\n\n        >>> print(hexdump(b\'A\'*32, width=8))\n        00000000  41 41 41 41  41 41 41 41  │AAAA│AAAA│\n        *\n        00000020\n\n        >>> print(hexdump(cyclic(32), width=8, begin=0xdead0000, hexii=True))\n        dead0000  .a  .a  .a  .a   .b  .a  .a  .a  │\n        dead0008  .c  .a  .a  .a   .d  .a  .a  .a  │\n        dead0010  .e  .a  .a  .a   .f  .a  .a  .a  │\n        dead0018  .g  .a  .a  .a   .h  .a  .a  .a  │\n        dead0020\n\n        >>> print(hexdump(bytearray(range(256))))\n        00000000  00 01 02 03  04 05 06 07  08 09 0a 0b  0c 0d 0e 0f  │····│····│····│····│\n        00000010  10 11 12 13  14 15 16 17  18 19 1a 1b  1c 1d 1e 1f  │····│····│····│····│\n        00000020  20 21 22 23  24 25 26 27  28 29 2a 2b  2c 2d 2e 2f  │ !"#│$%&\'│()*+│,-./│\n        00000030  30 31 32 33  34 35 36 37  38 39 3a 3b  3c 3d 3e 3f  │0123│4567│89:;│<=>?│\n        00000040  40 41 42 43  44 45 46 47  48 49 4a 4b  4c 4d 4e 4f  │@ABC│DEFG│HIJK│LMNO│\n        00000050  50 51 52 53  54 55 56 57  58 59 5a 5b  5c 5d 5e 5f  │PQRS│TUVW│XYZ[│\\]^_│\n        00000060  60 61 62 63  64 65 66 67  68 69 6a 6b  6c 6d 6e 6f  │`abc│defg│hijk│lmno│\n        00000070  70 71 72 73  74 75 76 77  78 79 7a 7b  7c 7d 7e 7f  │pqrs│tuvw│xyz{│|}~·│\n        00000080  80 81 82 83  84 85 86 87  88 89 8a 8b  8c 8d 8e 8f  │····│····│····│····│\n        00000090  90 91 92 93  94 95 96 97  98 99 9a 9b  9c 9d 9e 9f  │····│····│····│····│\n        000000a0  a0 a1 a2 a3  a4 a5 a6 a7  a8 a9 aa ab  ac ad ae af  │····│····│····│····│\n        000000b0  b0 b1 b2 b3  b4 b5 b6 b7  b8 b9 ba bb  bc bd be bf  │····│····│····│····│\n        000000c0  c0 c1 c2 c3  c4 c5 c6 c7  c8 c9 ca cb  cc cd ce cf  │····│····│····│····│\n        000000d0  d0 d1 d2 d3  d4 d5 d6 d7  d8 d9 da db  dc dd de df  │····│····│····│····│\n        000000e0  e0 e1 e2 e3  e4 e5 e6 e7  e8 e9 ea eb  ec ed ee ef  │····│····│····│····│\n        000000f0  f0 f1 f2 f3  f4 f5 f6 f7  f8 f9 fa fb  fc fd fe ff  │····│····│····│····│\n        00000100\n\n        >>> print(hexdump(bytearray(range(256)), hexii=True))\n        00000000      01  02  03   04  05  06  07   08  09  0a  0b   0c  0d  0e  0f  │\n        00000010  10  11  12  13   14  15  16  17   18  19  1a  1b   1c  1d  1e  1f  │\n        00000020  20  .!  ."  .#   .$  .%  .&  .\'   .(  .)  .*  .+   .,  .-  ..  ./  │\n        00000030  .0  .1  .2  .3   .4  .5  .6  .7   .8  .9  .:  .;   .<  .=  .>  .?  │\n        00000040  .@  .A  .B  .C   .D  .E  .F  .G   .H  .I  .J  .K   .L  .M  .N  .O  │\n        00000050  .P  .Q  .R  .S   .T  .U  .V  .W   .X  .Y  .Z  .[   .\\  .]  .^  ._  │\n        00000060  .`  .a  .b  .c   .d  .e  .f  .g   .h  .i  .j  .k   .l  .m  .n  .o  │\n        00000070  .p  .q  .r  .s   .t  .u  .v  .w   .x  .y  .z  .{   .|  .}  .~  7f  │\n        00000080  80  81  82  83   84  85  86  87   88  89  8a  8b   8c  8d  8e  8f  │\n        00000090  90  91  92  93   94  95  96  97   98  99  9a  9b   9c  9d  9e  9f  │\n        000000a0  a0  a1  a2  a3   a4  a5  a6  a7   a8  a9  aa  ab   ac  ad  ae  af  │\n        000000b0  b0  b1  b2  b3   b4  b5  b6  b7   b8  b9  ba  bb   bc  bd  be  bf  │\n        000000c0  c0  c1  c2  c3   c4  c5  c6  c7   c8  c9  ca  cb   cc  cd  ce  cf  │\n        000000d0  d0  d1  d2  d3   d4  d5  d6  d7   d8  d9  da  db   dc  dd  de  df  │\n        000000e0  e0  e1  e2  e3   e4  e5  e6  e7   e8  e9  ea  eb   ec  ed  ee  ef  │\n        000000f0  f0  f1  f2  f3   f4  f5  f6  f7   f8  f9  fa  fb   fc  fd  fe  ##  │\n        00000100\n\n        >>> print(hexdump(b\'X\' * 64))\n        00000000  58 58 58 58  58 58 58 58  58 58 58 58  58 58 58 58  │XXXX│XXXX│XXXX│XXXX│\n        *\n        00000040\n\n        >>> print(hexdump(b\'X\' * 64, skip=False))\n        00000000  58 58 58 58  58 58 58 58  58 58 58 58  58 58 58 58  │XXXX│XXXX│XXXX│XXXX│\n        00000010  58 58 58 58  58 58 58 58  58 58 58 58  58 58 58 58  │XXXX│XXXX│XXXX│XXXX│\n        00000020  58 58 58 58  58 58 58 58  58 58 58 58  58 58 58 58  │XXXX│XXXX│XXXX│XXXX│\n        00000030  58 58 58 58  58 58 58 58  58 58 58 58  58 58 58 58  │XXXX│XXXX│XXXX│XXXX│\n        00000040\n\n        >>> print(hexdump(fit({0x10: b\'X\'*0x20, 0x50-1: b\'\\xff\'*20}, length=0xc0) + b\'\\x00\'*32))\n        00000000  61 61 61 61  62 61 61 61  63 61 61 61  64 61 61 61  │aaaa│baaa│caaa│daaa│\n        00000010  58 58 58 58  58 58 58 58  58 58 58 58  58 58 58 58  │XXXX│XXXX│XXXX│XXXX│\n        *\n        00000030  6d 61 61 61  6e 61 61 61  6f 61 61 61  70 61 61 61  │maaa│naaa│oaaa│paaa│\n        00000040  71 61 61 61  72 61 61 61  73 61 61 61  74 61 61 ff  │qaaa│raaa│saaa│taa·│\n        00000050  ff ff ff ff  ff ff ff ff  ff ff ff ff  ff ff ff ff  │····│····│····│····│\n        00000060  ff ff ff 61  7a 61 61 62  62 61 61 62  63 61 61 62  │···a│zaab│baab│caab│\n        00000070  64 61 61 62  65 61 61 62  66 61 61 62  67 61 61 62  │daab│eaab│faab│gaab│\n        00000080  68 61 61 62  69 61 61 62  6a 61 61 62  6b 61 61 62  │haab│iaab│jaab│kaab│\n        00000090  6c 61 61 62  6d 61 61 62  6e 61 61 62  6f 61 61 62  │laab│maab│naab│oaab│\n        000000a0  70 61 61 62  71 61 61 62  72 61 61 62  73 61 61 62  │paab│qaab│raab│saab│\n        000000b0  74 61 61 62  75 61 61 62  76 61 61 62  77 61 61 62  │taab│uaab│vaab│waab│\n        000000c0  00 00 00 00  00 00 00 00  00 00 00 00  00 00 00 00  │····│····│····│····│\n        *\n        000000e0\n\n        >>> print(hexdump(fit({0x10: b\'X\'*0x20, 0x50-1: b\'\\xff\'*20}, length=0xc0) + b\'\\x00\'*32, cyclic=1))\n        00000000  61 61 61 61  62 61 61 61  63 61 61 61  64 61 61 61  │aaaa│baaa│caaa│daaa│\n        00000010  58 58 58 58  58 58 58 58  58 58 58 58  58 58 58 58  │XXXX│XXXX│XXXX│XXXX│\n        *\n        00000030  6d 61 61 61  6e 61 61 61  6f 61 61 61  70 61 61 61  │maaa│naaa│oaaa│paaa│\n        00000040  71 61 61 61  72 61 61 61  73 61 61 61  74 61 61 ff  │qaaa│raaa│saaa│taa·│\n        00000050  ff ff ff ff  ff ff ff ff  ff ff ff ff  ff ff ff ff  │····│····│····│····│\n        00000060  ff ff ff 61  7a 61 61 62  62 61 61 62  63 61 61 62  │···a│zaab│baab│caab│\n        00000070  64 61 61 62  65 61 61 62  66 61 61 62  67 61 61 62  │daab│eaab│faab│gaab│\n        *\n        000000c0  00 00 00 00  00 00 00 00  00 00 00 00  00 00 00 00  │····│····│····│····│\n        *\n        000000e0\n\n        >>> print(hexdump(fit({0x10: b\'X\'*0x20, 0x50-1: b\'\\xff\'*20}, length=0xc0) + b\'\\x00\'*32, cyclic=1, hexii=1))\n        00000000  .a  .a  .a  .a   .b  .a  .a  .a   .c  .a  .a  .a   .d  .a  .a  .a  │\n        00000010  .X  .X  .X  .X   .X  .X  .X  .X   .X  .X  .X  .X   .X  .X  .X  .X  │\n        *\n        00000030  .m  .a  .a  .a   .n  .a  .a  .a   .o  .a  .a  .a   .p  .a  .a  .a  │\n        00000040  .q  .a  .a  .a   .r  .a  .a  .a   .s  .a  .a  .a   .t  .a  .a  ##  │\n        00000050  ##  ##  ##  ##   ##  ##  ##  ##   ##  ##  ##  ##   ##  ##  ##  ##  │\n        00000060  ##  ##  ##  .a   .z  .a  .a  .b   .b  .a  .a  .b   .c  .a  .a  .b  │\n        00000070  .d  .a  .a  .b   .e  .a  .a  .b   .f  .a  .a  .b   .g  .a  .a  .b  │\n        *\n        000000c0                                                                     │\n        *\n        000000e0\n\n        >>> print(hexdump(b\'A\'*16, width=9))\n        00000000  41 41 41 41  41 41 41 41  41  │AAAA│AAAA│A│\n        00000009  41 41 41 41  41 41 41         │AAAA│AAA│\n        00000010\n        >>> print(hexdump(b\'A\'*16, width=10))\n        00000000  41 41 41 41  41 41 41 41  41 41  │AAAA│AAAA│AA│\n        0000000a  41 41 41 41  41 41               │AAAA│AA│\n        00000010\n        >>> print(hexdump(b\'A\'*16, width=11))\n        00000000  41 41 41 41  41 41 41 41  41 41 41  │AAAA│AAAA│AAA│\n        0000000b  41 41 41 41  41                     │AAAA│A│\n        00000010\n        >>> print(hexdump(b\'A\'*16, width=12))\n        00000000  41 41 41 41  41 41 41 41  41 41 41 41  │AAAA│AAAA│AAAA│\n        0000000c  41 41 41 41                            │AAAA│\n        00000010\n        >>> print(hexdump(b\'A\'*16, width=13))\n        00000000  41 41 41 41  41 41 41 41  41 41 41 41  41  │AAAA│AAAA│AAAA│A│\n        0000000d  41 41 41                                   │AAA│\n        00000010\n        >>> print(hexdump(b\'A\'*16, width=14))\n        00000000  41 41 41 41  41 41 41 41  41 41 41 41  41 41  │AAAA│AAAA│AAAA│AA│\n        0000000e  41 41                                         │AA│\n        00000010\n        >>> print(hexdump(b\'A\'*16, width=15))\n        00000000  41 41 41 41  41 41 41 41  41 41 41 41  41 41 41  │AAAA│AAAA│AAAA│AAA│\n        0000000f  41                                               │A│\n        00000010\n\n        >>> print(hexdump(b\'A\'*24, width=16, groupsize=8))\n        00000000  41 41 41 41 41 41 41 41  41 41 41 41 41 41 41 41  │AAAAAAAA│AAAAAAAA│\n        00000010  41 41 41 41 41 41 41 41                           │AAAAAAAA│\n        00000018\n        >>> print(hexdump(b\'A\'*24, width=16, groupsize=-1))\n        00000000  41 41 41 41 41 41 41 41 41 41 41 41 41 41 41 41  │AAAAAAAAAAAAAAAA│\n        00000010  41 41 41 41 41 41 41 41                          │AAAAAAAA│\n        00000018\n\n        >>> print(hexdump(b\'A\'*24, width=16, total=False))\n        00000000  41 41 41 41  41 41 41 41  41 41 41 41  41 41 41 41  │AAAA│AAAA│AAAA│AAAA│\n        00000010  41 41 41 41  41 41 41 41                            │AAAA│AAAA│\n        >>> print(hexdump(b\'A\'*24, width=16, groupsize=8, total=False))\n        00000000  41 41 41 41 41 41 41 41  41 41 41 41 41 41 41 41  │AAAAAAAA│AAAAAAAA│\n        00000010  41 41 41 41 41 41 41 41                           │AAAAAAAA│\n    '
    s = packing.flat(s, stacklevel=1)
    return '\n'.join(hexdump_iter(BytesIO(s), width, skip, hexii, begin, style, highlight, cyclic, groupsize, total))

def negate(value, width=None):
    if False:
        i = 10
        return i + 15
    "\n    Returns the two's complement of 'value'.\n    "
    if width is None:
        width = context.bits
    mask = (1 << width) - 1
    return mask + 1 - value & mask

def bnot(value, width=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Returns the binary inverse of 'value'.\n    "
    if width is None:
        width = context.bits
    mask = (1 << width) - 1
    return mask ^ value

@LocalNoarchContext
def js_escape(data, padding=context.cyclic_alphabet[0:1], **kwargs):
    if False:
        while True:
            i = 10
    'js_escape(data, padding=context.cyclic_alphabet[0:1], endian = None, **kwargs) -> str\n\n    Pack data as an escaped Unicode string for use in JavaScript\'s `unescape()` function\n\n    Arguments:\n        data (bytes): Bytes to pack\n        padding (bytes): A single byte to use as padding if data is of uneven length\n        endian (str): Endianness with which to pack the string ("little"/"big")\n\n    Returns:\n        A string representation of the packed data\n\n    >>> js_escape(b\'\\xde\\xad\\xbe\\xef\')\n    \'%uadde%uefbe\'\n\n    >>> js_escape(b\'\\xde\\xad\\xbe\\xef\', endian=\'big\')\n    \'%udead%ubeef\'\n\n    >>> js_escape(b\'\\xde\\xad\\xbe\')\n    \'%uadde%u61be\'\n\n    >>> js_escape(b\'aaaa\')\n    \'%u6161%u6161\'\n    '
    data = packing._need_bytes(data)
    padding = packing._need_bytes(padding)
    if len(padding) != 1:
        raise ValueError('Padding must be a single byte')
    if len(data) % 2:
        data += padding[0:1]
    data = bytearray(data)
    if context.endian == 'little':
        return ''.join(('%u{a:02x}{b:02x}'.format(a=a, b=b) for (b, a) in iters.group(2, data)))
    else:
        return ''.join(('%u{a:02x}{b:02x}'.format(a=a, b=b) for (a, b) in iters.group(2, data)))

@LocalNoarchContext
def js_unescape(s, **kwargs):
    if False:
        while True:
            i = 10
    'js_unescape(s, endian = None, **kwargs) -> bytes\n\n    Unpack an escaped Unicode string from JavaScript\'s `escape()` function\n\n    Arguments:\n        s (str): Escaped string to unpack\n        endian (str): Endianness with which to unpack the string ("little"/"big")\n\n    Returns:\n        A bytes representation of the unpacked data\n\n    >>> js_unescape(\'%uadde%uefbe\')\n    b\'\\xde\\xad\\xbe\\xef\'\n\n    >>> js_unescape(\'%udead%ubeef\', endian=\'big\')\n    b\'\\xde\\xad\\xbe\\xef\'\n\n    >>> js_unescape(\'abc%u4141123\')\n    b\'a\\x00b\\x00c\\x00AA1\\x002\\x003\\x00\'\n\n    >>> data = b\'abcdABCD1234!@#$\\x00\\x01\\x02\\x03\\x80\\x81\\x82\\x83\'\n    >>> js_unescape(js_escape(data)) == data\n    True\n\n    >>> js_unescape(\'%u4141%u42\')\n    Traceback (most recent call last):\n    ValueError: Incomplete Unicode token: %u42\n\n    >>> js_unescape(\'%u4141%uwoot%4141\')\n    Traceback (most recent call last):\n    ValueError: Failed to decode token: %uwoot\n\n    >>> js_unescape(\'%u4141%E4%F6%FC%u4141\')\n    Traceback (most recent call last):\n    NotImplementedError: Non-Unicode % tokens are not supported: %E4\n\n    >>> js_unescape(\'%u4141%zz%u4141\')\n    Traceback (most recent call last):\n    ValueError: Bad % token: %zz\n    '
    s = packing._decode(s)
    res = []
    p = 0
    while p < len(s):
        if s[p] == '%':
            if s[p + 1] == 'u':
                n = s[p + 2:p + 6]
                if len(n) < 4:
                    raise ValueError('Incomplete Unicode token: %s' % s[p:])
                try:
                    n = int(n, 16)
                except ValueError:
                    raise ValueError('Failed to decode token: %s' % s[p:p + 6])
                res.append(packing.p16(n))
                p += 6
            elif s[p + 1] in string.hexdigits and s[p + 2] in string.hexdigits:
                raise NotImplementedError('Non-Unicode %% tokens are not supported: %s' % s[p:p + 3])
            else:
                raise ValueError('Bad %% token: %s' % s[p:p + 3])
        else:
            res.append(packing.p16(ord(s[p])))
            p += 1
    return b''.join(res)