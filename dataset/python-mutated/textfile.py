"""Utilities for distinguishing binary files from text files"""
from __future__ import absolute_import
from itertools import chain
from bzrlib.errors import BinaryFile
from bzrlib.iterablefile import IterableFile
from bzrlib.osutils import file_iterator

def text_file(input):
    if False:
        for i in range(10):
            print('nop')
    'Produce a file iterator that is guaranteed to be text, without seeking.\n    BinaryFile is raised if the file contains a NUL in the first 1024 bytes.\n    '
    first_chunk = input.read(1024)
    if '\x00' in first_chunk:
        raise BinaryFile()
    return IterableFile(chain((first_chunk,), file_iterator(input)))

def check_text_lines(lines):
    if False:
        while True:
            i = 10
    'Raise BinaryFile if the supplied lines contain NULs.\n    Only the first 1024 characters are checked.\n    '
    f = IterableFile(lines)
    if '\x00' in f.read(1024):
        raise BinaryFile()

def check_text_path(path):
    if False:
        print('Hello World!')
    'Check whether the supplied path is a text, not binary file.\n    Raise BinaryFile if a NUL occurs in the first 1024 bytes.\n    '
    f = open(path, 'rb')
    try:
        text_file(f)
    finally:
        f.close()