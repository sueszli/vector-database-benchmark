"""
Some file handling utilities
"""
from __future__ import annotations
import typing
import os
from typing import Union
if typing.TYPE_CHECKING:
    from openage.util.fslike.abstract import FSLikeObject

def read_guaranteed(fileobj: FSLikeObject, size: int) -> bytes:
    if False:
        return 10
    '\n    As regular fileobj.read(size), but raises EOFError if fewer bytes\n    than requested are returned.\n    '
    remaining = size
    result = []
    while remaining:
        data = fileobj.read(remaining)
        if not data:
            raise EOFError()
        remaining -= len(data)
        result.append(data)
    return b''.join(result)

def read_nullterminated_string(fileobj: FSLikeObject, maxlen: int=255) -> bytes:
    if False:
        for i in range(10):
            print('nop')
    '\n    Reads bytes until a null terminator is reached.\n    '
    result = bytearray()
    while True:
        char = ord(read_guaranteed(fileobj, 1))
        if char == 0:
            break
        result.append(char)
        if len(result) > maxlen:
            raise SyntaxError('Null-terminated string too long.')
    return bytes(result)

def which(filename: str) -> Union[str, None]:
    if False:
        while True:
            i = 10
    '\n    Like the which (1) tool to get the full path of a command\n    by looking at the PATH environment variable.\n    '

    def is_executable(fpath: str) -> bool:
        if False:
            i = 10
            return i + 15
        '\n        Test if the given file exists and has an executable bit.\n        '
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)
    fpath = os.path.split(filename)[0]
    if fpath:
        if is_executable(filename):
            return filename
    else:
        for path in os.environ['PATH'].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, filename)
            if is_executable(exe_file):
                return exe_file
    return None