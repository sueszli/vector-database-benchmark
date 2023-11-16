"""
Misc string helper functions; this includes encoding, decoding,
manipulation, ...
"""
from sys import stdout

def decode_until_null(data: bytes, encoding: str='utf-8') -> str:
    if False:
        return 10
    '\n    decodes a bytes object, aborting at the first \\0 character.\n\n    >>> decode_until_null(b"foo\\0bar")\n    \'foo\'\n    '
    end = data.find(0)
    if end != -1:
        data = data[:end]
    return data.decode(encoding)

def try_decode(data: bytes) -> str:
    if False:
        print('Hello World!')
    '\n    does its best to attempt decoding the given string of unknown encoding.\n    '
    try:
        return data.decode('utf-8')
    except UnicodeDecodeError:
        pass
    return data.decode('iso-8859-1')

def binstr(num: int, bits: int=None, group: int=8) -> str:
    if False:
        return 10
    "\n    Similar to the built-in bin(), but optionally takes\n    the number of bits as an argument, and prints underscores instead of\n    zeroes.\n\n    >>> binstr(1337, 16)\n    '_____1_1 __111__1'\n    "
    result = bin(num)[2:]
    if bits is not None:
        result = result.rjust(bits, '0')
    result = result.replace('0', '_')
    if group is not None:
        grouped = [result[i:i + group] for i in range(0, len(result), group)]
        result = ' '.join(grouped)
    return result

def colorize(string: str, colorcode: str) -> str:
    if False:
        while True:
            i = 10
    "\n    Colorizes string with the given EMCA-48 SGR code.\n\n    >>> colorize('foo', '31;1')\n    '\\x1b[31;1mfoo\\x1b[m'\n    "
    if colorcode:
        colorized = f'\x1b[{colorcode}m{string}\x1b[m'
    else:
        colorized = string
    return colorized

def lstrip_once(string: str, substr: str) -> str:
    if False:
        while True:
            i = 10
    '\n    Removes substr at the start of string, and raises ValueError on failure.\n\n    >>> lstrip_once("openage.test", "openage.")\n    \'test\'\n    >>> lstrip_once("libopenage.test", "openage.")\n    Traceback (most recent call last):\n    ValueError: \'libopenage.test\' doesn\'t start with \'openage.\'\n    '
    if not string.startswith(substr):
        raise ValueError(f"{repr(string)} doesn't start with {repr(substr)}")
    return string[len(substr):]

def rstrip_once(string: str, substr: str) -> str:
    if False:
        return 10
    '\n    Removes substr at the end of string, and raises ValueError on failure.\n\n    >>> rstrip_once("test.cpp", ".cpp")\n    \'test\'\n    '
    if not string.endswith(substr):
        raise ValueError(f"{repr(string)} doesn't end with {repr(substr)}")
    return string[:-len(substr)]

def format_progress(progress: int, total: int) -> str:
    if False:
        while True:
            i = 10
    '\n    Formats an "x out of y" string with fixed width.\n\n    >>> format_progress(5, 20)\n    \' 5/20\'\n    '
    return f'{progress:>{len(str(total))}}/{total}'

def print_progress(progress: int, total: int) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Print an "x out of y" string with fixed width to stdout.\n    The output overwrites itself.\n    '
    stdout.write(format_progress(progress, total) + '\r')