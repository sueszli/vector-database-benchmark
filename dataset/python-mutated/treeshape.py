"""Test helper for constructing and testing directories.

This module transforms filesystem directories to and from Python lists.
As a Python list the descriptions can be stored in test cases, compared,
etc.
"""
import os
import stat
from bzrlib.trace import warning
from bzrlib.osutils import pathjoin

def build_tree_contents(template):
    if False:
        while True:
            i = 10
    "Reconstitute some files from a text description.\n\n    Each element of template is a tuple.  The first element is a filename,\n    with an optional ending character indicating the type.\n\n    The template is built relative to the Python process's current\n    working directory.\n\n    ('foo/',) will build a directory.\n    ('foo', 'bar') will write 'bar' to 'foo'\n    ('foo@', 'linktarget') will raise an error\n    "
    for tt in template:
        name = tt[0]
        if name[-1] == '/':
            os.mkdir(name)
        elif name[-1] == '@':
            os.symlink(tt[1], tt[0][:-1])
        else:
            f = file(name, 'wb')
            try:
                f.write(tt[1])
            finally:
                f.close()

def capture_tree_contents(top):
    if False:
        i = 10
        return i + 15
    'Make a Python datastructure description of a tree.\n\n    If top is an absolute path the descriptions will be absolute.'
    for (dirpath, dirnames, filenames) in os.walk(top):
        yield (dirpath + '/',)
        filenames.sort()
        for fn in filenames:
            fullpath = pathjoin(dirpath, fn)
            if fullpath[-1] in '@/':
                raise AssertionError(fullpath)
            info = os.lstat(fullpath)
            if stat.S_ISLNK(info.st_mode):
                yield (fullpath + '@', os.readlink(fullpath))
            elif stat.S_ISREG(info.st_mode):
                yield (fullpath, file(fullpath, 'rb').read())
            else:
                warning("can't capture file %s with mode %#o", fullpath, info.st_mode)
                pass