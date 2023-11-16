"""
Created on May 10, 2012

@author: sean
"""
from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from imp import get_magic
import time
import struct
import marshal

def extract(binary):
    if False:
        while True:
            i = 10
    '\n    Extract a code object from a binary pyc file.\n\n    :param binary: a sequence of bytes from a pyc file.\n    '
    if len(binary) <= 8:
        raise Exception('Binary pyc must be greater than 8 bytes (got %i)' % len(binary))
    magic = binary[:4]
    MAGIC = get_magic()
    if magic != MAGIC:
        raise Exception('Python version mismatch (%r != %r) Is this a pyc file?' % (magic, MAGIC))
    modtime = time.asctime(time.localtime(struct.unpack('i', binary[4:8])[0]))
    code = marshal.loads(binary[8:])
    return (modtime, code)