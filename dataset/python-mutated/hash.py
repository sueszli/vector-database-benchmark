"""
Functions for hashing files.
"""
from __future__ import annotations
import typing
import hashlib
if typing.TYPE_CHECKING:
    from openage.util.fslike.path import Path

def hash_file(path: Path, hash_algo: str='sha3_256', bufsize: int=32768) -> str:
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the hash value of a given file.\n\n    :param path: Path of the file.\n    :type path: .fslike.path.Path\n    :param hash_algo: Hashing algorithm identifier.\n    :type hash_algo: str\n    :param bufsize: Buffer size for reading files.\n    :type bufsize: int\n    '
    hashfunc = hashlib.new(hash_algo)
    with path.open_r() as f_in:
        while True:
            data = f_in.read(bufsize)
            if not data:
                break
            hashfunc.update(data)
    return hashfunc.hexdigest()