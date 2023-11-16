"""Module for extracting object files from a compiled archive (.a) file.

This module provides functionality almost identical to the 'ar -x' command,
which extracts out all object files from a given archive file. This module
assumes the archive is in the BSD variant format used in Apple platforms.

See: https://en.wikipedia.org/wiki/Ar_(Unix)#BSD_variant

This extractor has two important differences compared to the 'ar -x' command
shipped with Xcode.

1.  When there are multiple object files with the same name in a given archive,
    each file is renamed so that they are all correctly extracted without
    overwriting each other.

2.  This module takes the destination directory as an additional parameter.

    Example Usage:

    archive_path = ...
    dest_dir = ...
    extract_object_files(archive_path, dest_dir)
"""
import hashlib
import io
import itertools
import os
import struct
from typing import Iterator, Tuple

def extract_object_files(archive_file: io.BufferedIOBase, dest_dir: str) -> None:
    if False:
        i = 10
        return i + 15
    'Extracts object files from the archive path to the destination directory.\n\n  Extracts object files from the given BSD variant archive file. The extracted\n  files are written to the destination directory, which will be created if the\n  directory does not exist.\n\n  Colliding object file names are automatically renamed upon extraction in order\n  to avoid unintended overwriting.\n\n  Args:\n    archive_file: The archive file object pointing at its beginning.\n    dest_dir: The destination directory path in which the extracted object files\n      will be written. The directory will be created if it does not exist.\n  '
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    _check_archive_signature(archive_file)
    extracted_files = dict()
    for (name, file_content) in _extract_next_file(archive_file):
        digest = hashlib.md5(file_content).digest()
        for final_name in _generate_modified_filenames(name):
            if final_name not in extracted_files:
                extracted_files[final_name] = digest
                with open(os.path.join(dest_dir, final_name), 'wb') as object_file:
                    object_file.write(file_content)
                break
            elif extracted_files[final_name] == digest:
                break

def _generate_modified_filenames(filename: str) -> Iterator[str]:
    if False:
        for i in range(10):
            print('nop')
    'Generates the modified filenames with incremental name suffix added.\n\n  This helper function first yields the given filename itself, and subsequently\n  yields modified filenames by incrementing number suffix to the basename.\n\n  Args:\n    filename: The original filename to be modified.\n\n  Yields:\n    The original filename and then modified filenames with incremental suffix.\n  '
    yield filename
    (base, ext) = os.path.splitext(filename)
    for name_suffix in itertools.count(1, 1):
        yield '{}_{}{}'.format(base, name_suffix, ext)

def _check_archive_signature(archive_file: io.BufferedIOBase) -> None:
    if False:
        i = 10
        return i + 15
    'Checks if the file has the correct archive header signature.\n\n  The cursor is moved to the first available file header section after\n  successfully checking the signature.\n\n  Args:\n    archive_file: The archive file object pointing at its beginning.\n\n  Raises:\n    RuntimeError: The archive signature is invalid.\n  '
    signature = archive_file.read(8)
    if signature != b'!<arch>\n':
        raise RuntimeError('Invalid archive file format.')

def _extract_next_file(archive_file: io.BufferedIOBase) -> Iterator[Tuple[str, bytes]]:
    if False:
        for i in range(10):
            print('nop')
    'Extracts the next available file from the archive.\n\n  Reads the next available file header section and yields its filename and\n  content in bytes as a tuple. Stops when there are no more available files in\n  the provided archive_file.\n\n  Args:\n    archive_file: The archive file object, of which cursor is pointing to the\n      next available file header section.\n\n  Yields:\n    The name and content of the next available file in the given archive file.\n\n  Raises:\n    RuntimeError: The archive_file is in an unknown format.\n  '
    while True:
        header = archive_file.read(60)
        if not header:
            return
        elif len(header) < 60:
            raise RuntimeError('Invalid file header format.')
        (name, _, _, _, _, size, end) = struct.unpack('=16s12s6s6s8s10s2s', header)
        if end != b'`\n':
            raise RuntimeError('Invalid file header format.')
        name = name.decode('ascii').strip()
        size = int(size, base=10)
        odd_size = size % 2 == 1
        if name.startswith('#1/'):
            filename_size = int(name[3:])
            name = archive_file.read(filename_size).decode('utf-8').strip(' \x00')
            size -= filename_size
        file_content = archive_file.read(size)
        if odd_size:
            archive_file.read(1)
        yield (name, file_content)