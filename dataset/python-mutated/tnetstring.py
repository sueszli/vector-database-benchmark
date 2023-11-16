"""
tnetstring:  data serialization using typed netstrings
======================================================

This is a custom Python 3 implementation of tnetstrings.
Compared to other implementations, the main difference
is that this implementation supports a custom unicode datatype.

An ordinary tnetstring is a blob of data prefixed with its length and postfixed
with its type. Here are some examples:

    >>> tnetstring.dumps("hello world")
    11:hello world,
    >>> tnetstring.dumps(12345)
    5:12345#
    >>> tnetstring.dumps([12345, True, 0])
    19:5:12345#4:true!1:0#]

This module gives you the following functions:

    :dump:    dump an object as a tnetstring to a file
    :dumps:   dump an object as a tnetstring to a string
    :load:    load a tnetstring-encoded object from a file
    :loads:   load a tnetstring-encoded object from a string

Note that since parsing a tnetstring requires reading all the data into memory
at once, there's no efficiency gain from using the file-based versions of these
functions.  They're only here so you can use load() to read precisely one
item from a file or socket without consuming any extra data.

The tnetstrings specification explicitly states that strings are binary blobs
and forbids the use of unicode at the protocol level.
**This implementation decodes dictionary keys as surrogate-escaped ASCII**,
all other strings are returned as plain bytes.

:Copyright: (c) 2012-2013 by Ryan Kelly <ryan@rfk.id.au>.
:Copyright: (c) 2014 by Carlo Pires <carlopires@gmail.com>.
:Copyright: (c) 2016 by Maximilian Hils <tnetstring3@maximilianhils.com>.

:License: MIT
"""
import collections
from typing import BinaryIO
from typing import Union
TSerializable = Union[None, str, bool, int, float, bytes, list, tuple, dict]

def dumps(value: TSerializable) -> bytes:
    if False:
        while True:
            i = 10
    '\n    This function dumps a python object as a tnetstring.\n    '
    q: collections.deque = collections.deque()
    _rdumpq(q, 0, value)
    return b''.join(q)

def dump(value: TSerializable, file_handle: BinaryIO) -> None:
    if False:
        while True:
            i = 10
    '\n    This function dumps a python object as a tnetstring and\n    writes it to the given file.\n    '
    file_handle.write(dumps(value))

def _rdumpq(q: collections.deque, size: int, value: TSerializable) -> int:
    if False:
        return 10
    '\n    Dump value as a tnetstring, to a deque instance, last chunks first.\n\n    This function generates the tnetstring representation of the given value,\n    pushing chunks of the output onto the given deque instance.  It pushes\n    the last chunk first, then recursively generates more chunks.\n\n    When passed in the current size of the string in the queue, it will return\n    the new size of the string in the queue.\n\n    Operating last-chunk-first makes it easy to calculate the size written\n    for recursive structures without having to build their representation as\n    a string.  This is measurably faster than generating the intermediate\n    strings, especially on deeply nested structures.\n    '
    write = q.appendleft
    if value is None:
        write(b'0:~')
        return size + 3
    elif value is True:
        write(b'4:true!')
        return size + 7
    elif value is False:
        write(b'5:false!')
        return size + 8
    elif isinstance(value, int):
        data = str(value).encode()
        ldata = len(data)
        span = str(ldata).encode()
        write(b'%s:%s#' % (span, data))
        return size + 2 + len(span) + ldata
    elif isinstance(value, float):
        data = repr(value).encode()
        ldata = len(data)
        span = str(ldata).encode()
        write(b'%s:%s^' % (span, data))
        return size + 2 + len(span) + ldata
    elif isinstance(value, bytes):
        data = value
        ldata = len(data)
        span = str(ldata).encode()
        write(b',')
        write(data)
        write(b':')
        write(span)
        return size + 2 + len(span) + ldata
    elif isinstance(value, str):
        data = value.encode('utf8')
        ldata = len(data)
        span = str(ldata).encode()
        write(b';')
        write(data)
        write(b':')
        write(span)
        return size + 2 + len(span) + ldata
    elif isinstance(value, (list, tuple)):
        write(b']')
        init_size = size = size + 1
        for item in reversed(value):
            size = _rdumpq(q, size, item)
        span = str(size - init_size).encode()
        write(b':')
        write(span)
        return size + 1 + len(span)
    elif isinstance(value, dict):
        write(b'}')
        init_size = size = size + 1
        for (k, v) in value.items():
            size = _rdumpq(q, size, v)
            size = _rdumpq(q, size, k)
        span = str(size - init_size).encode()
        write(b':')
        write(span)
        return size + 1 + len(span)
    else:
        raise ValueError(f'unserializable object: {value} ({type(value)})')

def loads(string: bytes) -> TSerializable:
    if False:
        while True:
            i = 10
    '\n    This function parses a tnetstring into a python object.\n    '
    return pop(string)[0]

def load(file_handle: BinaryIO) -> TSerializable:
    if False:
        print('Hello World!')
    'load(file) -> object\n\n    This function reads a tnetstring from a file and parses it into a\n    python object.  The file must support the read() method, and this\n    function promises not to read more data than necessary.\n    '
    c = file_handle.read(1)
    if c == b'':
        raise ValueError('not a tnetstring: empty file')
    data_length = b''
    while c.isdigit():
        data_length += c
        if len(data_length) > 12:
            raise ValueError('not a tnetstring: absurdly large length prefix')
        c = file_handle.read(1)
    if c != b':':
        raise ValueError('not a tnetstring: missing or invalid length prefix')
    data = file_handle.read(int(data_length))
    data_type = file_handle.read(1)[0]
    return parse(data_type, data)

def parse(data_type: int, data: bytes) -> TSerializable:
    if False:
        while True:
            i = 10
    if data_type == ord(b','):
        return data
    if data_type == ord(b';'):
        return data.decode('utf8')
    if data_type == ord(b'#'):
        try:
            return int(data)
        except ValueError:
            raise ValueError(f'not a tnetstring: invalid integer literal: {data!r}')
    if data_type == ord(b'^'):
        try:
            return float(data)
        except ValueError:
            raise ValueError(f'not a tnetstring: invalid float literal: {data!r}')
    if data_type == ord(b'!'):
        if data == b'true':
            return True
        elif data == b'false':
            return False
        else:
            raise ValueError(f'not a tnetstring: invalid boolean literal: {data!r}')
    if data_type == ord(b'~'):
        if data:
            raise ValueError(f'not a tnetstring: invalid null literal: {data!r}')
        return None
    if data_type == ord(b']'):
        lst = []
        while data:
            (item, data) = pop(data)
            lst.append(item)
        return lst
    if data_type == ord(b'}'):
        d = {}
        while data:
            (key, data) = pop(data)
            (val, data) = pop(data)
            d[key] = val
        return d
    raise ValueError(f'unknown type tag: {data_type}')

def pop(data: bytes) -> tuple[TSerializable, bytes]:
    if False:
        print('Hello World!')
    '\n    This function parses a tnetstring into a python object.\n    It returns a tuple giving the parsed object and a string\n    containing any unparsed data from the end of the string.\n    '
    try:
        (blength, data) = data.split(b':', 1)
        length = int(blength)
    except ValueError:
        raise ValueError(f'not a tnetstring: missing or invalid length prefix: {data!r}')
    try:
        (data, data_type, remain) = (data[:length], data[length], data[length + 1:])
    except IndexError:
        raise ValueError(f'not a tnetstring: invalid length prefix: {length}')
    return (parse(data_type, data), remain)
__all__ = ['dump', 'dumps', 'load', 'loads', 'pop']