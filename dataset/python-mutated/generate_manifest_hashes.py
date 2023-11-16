"""
Provides functions for traversing a directory and
generating hash values for all the items inside.
"""
from __future__ import annotations
import typing
import os
from openage.util.hash import hash_file
if typing.TYPE_CHECKING:
    from openage.util.fslike.directory import Directory
    from openage.util.fslike.path import Path
    from openage.convert.entity_object.conversion.modpack import Modpack

def bfs_directory(root: Path) -> typing.Generator[Path, None, None]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Traverse the given directory with breadth-first way.\n\n    :param root: The directory to traverse.\n    :type root: ...util.fslike.path.Path\n    '
    dirs = [root]
    while dirs:
        next_level = []
        for directory in dirs:
            for item in directory.iterdir():
                if item.is_dir():
                    next_level.append(item)
                else:
                    yield item
        dirs = next_level

def generate_hashes(modpack: Modpack, exportdir: Directory, hash_algo: str='sha3_256', bufsize: int=32768) -> None:
    if False:
        print('Hello World!')
    '\n    Generate hashes for all the items in a\n    given modpack and adds them to the manifest\n    instance.\n\n    :param modpack: The target modpack.\n    :type modpack: ..dataformats.modpack.Modpack\n    :param exportdir: Directory wheere modpacks are stored.\n    :type exportdir: ...util.fslike.path.Path\n    :param hash_algo: Hashing algorithm used.\n    :type hash_algo: str\n    :param bufsize: Buffer size for reading files.\n    :type bufsize: int\n    '
    modpack.manifest.set_hashing_func(hash_algo)
    for file in bfs_directory(exportdir):
        hash_val = hash_file(file, hash_algo=hash_algo, bufsize=bufsize)
        relative_path = os.path.relpath(str(file), str(exportdir))
        modpack.manifest.add_hash_value(hash_val, relative_path)