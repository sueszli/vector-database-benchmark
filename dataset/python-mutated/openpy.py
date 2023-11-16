"""Tools to open ``*.py`` files as Unicode.

Uses the encoding specified within the file, as per PEP 263.

Much of the code is taken from the tokenize module in Python 3.2.

This file was forked from the IPython project:

* Copyright (c) 2008-2014, IPython Development Team
* Copyright (C) 2001-2007 Fernando Perez <fperez@colorado.edu>
* Copyright (c) 2001, Janko Hauser <jhauser@zscout.de>
* Copyright (c) 2001, Nathaniel Gray <n8gray@caltech.edu>
"""
import io
import re
from xonsh.lazyasd import LazyObject
from xonsh.tokenize import detect_encoding, tokopen
cookie_comment_re = LazyObject(lambda : re.compile('^\\s*#.*coding[:=]\\s*([-\\w.]+)', re.UNICODE), globals(), 'cookie_comment_re')

def source_to_unicode(txt, errors='replace', skip_encoding_cookie=True):
    if False:
        i = 10
        return i + 15
    'Converts a bytes string with python source code to unicode.\n\n    Unicode strings are passed through unchanged. Byte strings are checked\n    for the python source file encoding cookie to determine encoding.\n    txt can be either a bytes buffer or a string containing the source\n    code.\n    '
    if isinstance(txt, str):
        return txt
    if isinstance(txt, bytes):
        buf = io.BytesIO(txt)
    else:
        buf = txt
    try:
        (encoding, _) = detect_encoding(buf.readline)
    except SyntaxError:
        encoding = 'ascii'
    buf.seek(0)
    text = io.TextIOWrapper(buf, encoding, errors=errors, line_buffering=True)
    text.mode = 'r'
    if skip_encoding_cookie:
        return ''.join(strip_encoding_cookie(text))
    else:
        return text.read()

def strip_encoding_cookie(filelike):
    if False:
        print('Hello World!')
    'Generator to pull lines from a text-mode file, skipping the encoding\n    cookie if it is found in the first two lines.\n    '
    it = iter(filelike)
    try:
        first = next(it)
        if not cookie_comment_re.match(first):
            yield first
        second = next(it)
        if not cookie_comment_re.match(second):
            yield second
    except StopIteration:
        return
    yield from it

def read_py_file(filename, skip_encoding_cookie=True):
    if False:
        i = 10
        return i + 15
    'Read a Python file, using the encoding declared inside the file.\n\n    Parameters\n    ----------\n    filename : str\n        The path to the file to read.\n    skip_encoding_cookie : bool\n        If True (the default), and the encoding declaration is found in the first\n        two lines, that line will be excluded from the output - compiling a\n        unicode string with an encoding declaration is a SyntaxError in Python 2.\n\n    Returns\n    -------\n    A unicode string containing the contents of the file.\n    '
    with tokopen(filename) as f:
        if skip_encoding_cookie:
            return ''.join(strip_encoding_cookie(f))
        else:
            return f.read()

def read_py_url(url, errors='replace', skip_encoding_cookie=True):
    if False:
        i = 10
        return i + 15
    "Read a Python file from a URL, using the encoding declared inside the file.\n\n    Parameters\n    ----------\n    url : str\n        The URL from which to fetch the file.\n    errors : str\n        How to handle decoding errors in the file. Options are the same as for\n        bytes.decode(), but here 'replace' is the default.\n    skip_encoding_cookie : bool\n        If True (the default), and the encoding declaration is found in the first\n        two lines, that line will be excluded from the output - compiling a\n        unicode string with an encoding declaration is a SyntaxError in Python 2.\n\n    Returns\n    -------\n    A unicode string containing the contents of the file.\n    "
    try:
        from urllib.request import urlopen
    except ImportError:
        from urllib import urlopen
    response = urlopen(url)
    buf = io.BytesIO(response.read())
    return source_to_unicode(buf, errors, skip_encoding_cookie)

def _list_readline(x):
    if False:
        for i in range(10):
            print('nop')
    'Given a list, returns a readline() function that returns the next element\n    with each call.\n    '
    x = iter(x)

    def readline():
        if False:
            print('Hello World!')
        return next(x)
    return readline