"""Encode valid C string literals from Python strings.

If a character is not allowed in C string literals, it is either emitted
as a simple escape sequence (e.g. '\\n'), or an octal escape sequence
with exactly three digits ('\\oXXX'). Question marks are escaped to
prevent trigraphs in the string literal from being interpreted. Note
that '\\?' is an invalid escape sequence in Python.

Consider the string literal "AB\\xCDEF". As one would expect, Python
parses it as ['A', 'B', 0xCD, 'E', 'F']. However, the C standard
specifies that all hexadecimal digits immediately following '\\x' will
be interpreted as part of the escape sequence. Therefore, it is
unexpectedly parsed as ['A', 'B', 0xCDEF].

Emitting ("AB\\xCD" "EF") would avoid this behaviour. However, we opt
for simplicity and use octal escape sequences instead. They do not
suffer from the same issue as they are defined to parse at most three
octal digits.
"""
from __future__ import annotations
import string
from typing import Final
CHAR_MAP: Final = [f'\\{i:03o}' for i in range(256)]
for c in string.printable:
    CHAR_MAP[ord(c)] = c
for c in ("'", '"', '\\', 'a', 'b', 'f', 'n', 'r', 't', 'v'):
    escaped = f'\\{c}'
    decoded = escaped.encode('ascii').decode('unicode_escape')
    CHAR_MAP[ord(decoded)] = escaped
CHAR_MAP[ord('?')] = '\\?'

def encode_bytes_as_c_string(b: bytes) -> str:
    if False:
        print('Hello World!')
    'Produce contents of a C string literal for a byte string, without quotes.'
    escaped = ''.join([CHAR_MAP[i] for i in b])
    return escaped

def c_string_initializer(value: bytes) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Create initializer for a C char[]/ char * variable from a string.\n\n    For example, if value if b\'foo\', the result would be \'"foo"\'.\n    '
    return '"' + encode_bytes_as_c_string(value) + '"'