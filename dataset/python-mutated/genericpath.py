"""
Path operations common to more than one OS
Do not use directly.  The OS specific modules import the appropriate
functions from this module themselves.
"""
import os
import stat
__all__ = ['commonprefix', 'exists', 'getatime', 'getctime', 'getmtime', 'getsize', 'isdir', 'isfile', 'samefile', 'sameopenfile', 'samestat']

def exists(path):
    if False:
        print('Hello World!')
    'Test whether a path exists.  Returns False for broken symbolic links'
    try:
        os.stat(path)
    except (OSError, ValueError):
        return False
    return True

def isfile(path):
    if False:
        print('Hello World!')
    'Test whether a path is a regular file'
    try:
        st = os.stat(path)
    except (OSError, ValueError):
        return False
    return stat.S_ISREG(st.st_mode)

def isdir(s):
    if False:
        while True:
            i = 10
    'Return true if the pathname refers to an existing directory.'
    try:
        st = os.stat(s)
    except (OSError, ValueError):
        return False
    return stat.S_ISDIR(st.st_mode)

def getsize(filename):
    if False:
        while True:
            i = 10
    'Return the size of a file, reported by os.stat().'
    return os.stat(filename).st_size

def getmtime(filename):
    if False:
        return 10
    'Return the last modification time of a file, reported by os.stat().'
    return os.stat(filename).st_mtime

def getatime(filename):
    if False:
        while True:
            i = 10
    'Return the last access time of a file, reported by os.stat().'
    return os.stat(filename).st_atime

def getctime(filename):
    if False:
        print('Hello World!')
    'Return the metadata change time of a file, reported by os.stat().'
    return os.stat(filename).st_ctime

def commonprefix(m):
    if False:
        for i in range(10):
            print('nop')
    'Given a list of pathnames, returns the longest common leading component'
    if not m:
        return ''
    if not isinstance(m[0], (list, tuple)):
        m = tuple(map(os.fspath, m))
    s1 = min(m)
    s2 = max(m)
    for (i, c) in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1

def samestat(s1, s2):
    if False:
        for i in range(10):
            print('nop')
    'Test whether two stat buffers reference the same file'
    return s1.st_ino == s2.st_ino and s1.st_dev == s2.st_dev

def samefile(f1, f2):
    if False:
        for i in range(10):
            print('nop')
    'Test whether two pathnames reference the same actual file or directory\n\n    This is determined by the device number and i-node number and\n    raises an exception if an os.stat() call on either pathname fails.\n    '
    s1 = os.stat(f1)
    s2 = os.stat(f2)
    return samestat(s1, s2)

def sameopenfile(fp1, fp2):
    if False:
        i = 10
        return i + 15
    'Test whether two open file objects reference the same file'
    s1 = os.fstat(fp1)
    s2 = os.fstat(fp2)
    return samestat(s1, s2)

def _splitext(p, sep, altsep, extsep):
    if False:
        while True:
            i = 10
    'Split the extension from a pathname.\n\n    Extension is everything from the last dot to the end, ignoring\n    leading dots.  Returns "(root, ext)"; ext may be empty.'
    sepIndex = p.rfind(sep)
    if altsep:
        altsepIndex = p.rfind(altsep)
        sepIndex = max(sepIndex, altsepIndex)
    dotIndex = p.rfind(extsep)
    if dotIndex > sepIndex:
        filenameIndex = sepIndex + 1
        while filenameIndex < dotIndex:
            if p[filenameIndex:filenameIndex + 1] != extsep:
                return (p[:dotIndex], p[dotIndex:])
            filenameIndex += 1
    return (p, p[:0])

def _check_arg_types(funcname, *args):
    if False:
        while True:
            i = 10
    hasstr = hasbytes = False
    for s in args:
        if isinstance(s, str):
            hasstr = True
        elif isinstance(s, bytes):
            hasbytes = True
        else:
            raise TypeError(f'{funcname}() argument must be str, bytes, or os.PathLike object, not {s.__class__.__name__!r}') from None
    if hasstr and hasbytes:
        raise TypeError("Can't mix strings and bytes in path components") from None