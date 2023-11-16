from __future__ import absolute_import
from __future__ import division
from pwnlib.context import context
from pwnlib.util.fiddling import hexdump
from pwnlib.util.packing import unpack

class linux_dirent(object):

    def __init__(self, buf):
        if False:
            return 10
        n = context.bytes
        self.d_ino = unpack(buf[:n])
        buf = buf[n:]
        self.d_off = unpack(buf[:n])
        buf = buf[n:]
        self.d_reclen = unpack(buf[:2], 16)
        buf = buf[2:]
        self.d_name = buf[:buf.index(b'\x00')].decode('utf-8')

    def __len__(self):
        if False:
            return 10
        return self.d_reclen

    def __str__(self):
        if False:
            i = 10
            return i + 15
        return 'inode=%i %r' % (self.d_ino, self.d_name)

def dirents(buf):
    if False:
        while True:
            i = 10
    "unpack_dents(buf) -> list\n\n    Extracts data from a buffer emitted by getdents()\n\n    Arguments:\n        buf(str): Byte array\n\n    Returns:\n        A list of filenames.\n\n    Example:\n\n        >>> data = '5ade6d010100000010002e0000000004010000000200000010002e2e006e3d04092b6d010300000010007461736b00045bde6d010400000010006664003b3504'\n        >>> data = unhex(data)\n        >>> print(dirents(data))\n        ['.', '..', 'fd', 'task']\n    "
    d = []
    while buf:
        try:
            ent = linux_dirent(buf)
        except ValueError:
            break
        d.append(ent.d_name)
        buf = buf[len(ent):]
    return sorted(d)