"""
DBM-style interface to a directory.

Each key is stored as a single file.  This is not expected to be very fast or
efficient, but it's good for easy debugging.

DirDBMs are *not* thread-safe, they should only be accessed by one thread at
a time.

No files should be placed in the working directory of a DirDBM save those
created by the DirDBM itself!

Maintainer: Itamar Shtull-Trauring
"""
import base64
import glob
import os
import pickle
from twisted.python.filepath import FilePath
try:
    _open
except NameError:
    _open = open

class DirDBM:
    """
    A directory with a DBM interface.

    This class presents a hash-like interface to a directory of small,
    flat files. It can only use strings as keys or values.
    """

    def __init__(self, name):
        if False:
            print('Hello World!')
        '\n        @type name: str\n        @param name: Base path to use for the directory storage.\n        '
        self.dname = os.path.abspath(name)
        self._dnamePath = FilePath(name)
        if not self._dnamePath.isdir():
            self._dnamePath.createDirectory()
        else:
            for f in glob.glob(self._dnamePath.child('*.new').path):
                os.remove(f)
            replacements = glob.glob(self._dnamePath.child('*.rpl').path)
            for f in replacements:
                old = f[:-4]
                if os.path.exists(old):
                    os.remove(f)
                else:
                    os.rename(f, old)

    def _encode(self, k):
        if False:
            i = 10
            return i + 15
        '\n        Encode a key so it can be used as a filename.\n        '
        return base64.encodebytes(k).replace(b'\n', b'_').replace(b'/', b'-')

    def _decode(self, k):
        if False:
            print('Hello World!')
        '\n        Decode a filename to get the key.\n        '
        return base64.decodebytes(k.replace(b'_', b'\n').replace(b'-', b'/'))

    def _readFile(self, path):
        if False:
            i = 10
            return i + 15
        '\n        Read in the contents of a file.\n\n        Override in subclasses to e.g. provide transparently encrypted dirdbm.\n        '
        with _open(path.path, 'rb') as f:
            s = f.read()
        return s

    def _writeFile(self, path, data):
        if False:
            return 10
        '\n        Write data to a file.\n\n        Override in subclasses to e.g. provide transparently encrypted dirdbm.\n        '
        with _open(path.path, 'wb') as f:
            f.write(data)
            f.flush()

    def __len__(self):
        if False:
            print('Hello World!')
        '\n        @return: The number of key/value pairs in this Shelf\n        '
        return len(self._dnamePath.listdir())

    def __setitem__(self, k, v):
        if False:
            while True:
                i = 10
        '\n        C{dirdbm[k] = v}\n        Create or modify a textfile in this directory\n\n        @type k: bytes\n        @param k: key to set\n\n        @type v: bytes\n        @param v: value to associate with C{k}\n        '
        if not type(k) == bytes:
            raise TypeError('DirDBM key must be bytes')
        if not type(v) == bytes:
            raise TypeError('DirDBM value must be bytes')
        k = self._encode(k)
        old = self._dnamePath.child(k)
        if old.exists():
            new = old.siblingExtension('.rpl')
        else:
            new = old.siblingExtension('.new')
        try:
            self._writeFile(new, v)
        except BaseException:
            new.remove()
            raise
        else:
            if old.exists():
                old.remove()
            new.moveTo(old)

    def __getitem__(self, k):
        if False:
            return 10
        '\n        C{dirdbm[k]}\n        Get the contents of a file in this directory as a string.\n\n        @type k: bytes\n        @param k: key to lookup\n\n        @return: The value associated with C{k}\n        @raise KeyError: Raised when there is no such key\n        '
        if not type(k) == bytes:
            raise TypeError('DirDBM key must be bytes')
        path = self._dnamePath.child(self._encode(k))
        try:
            return self._readFile(path)
        except OSError:
            raise KeyError(k)

    def __delitem__(self, k):
        if False:
            while True:
                i = 10
        '\n        C{del dirdbm[foo]}\n        Delete a file in this directory.\n\n        @type k: bytes\n        @param k: key to delete\n\n        @raise KeyError: Raised when there is no such key\n        '
        if not type(k) == bytes:
            raise TypeError('DirDBM key must be bytes')
        k = self._encode(k)
        try:
            self._dnamePath.child(k).remove()
        except OSError:
            raise KeyError(self._decode(k))

    def keys(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @return: a L{list} of filenames (keys).\n        '
        return list(map(self._decode, self._dnamePath.asBytesMode().listdir()))

    def values(self):
        if False:
            print('Hello World!')
        '\n        @return: a L{list} of file-contents (values).\n        '
        vals = []
        keys = self.keys()
        for key in keys:
            vals.append(self[key])
        return vals

    def items(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        @return: a L{list} of 2-tuples containing key/value pairs.\n        '
        items = []
        keys = self.keys()
        for key in keys:
            items.append((key, self[key]))
        return items

    def has_key(self, key):
        if False:
            print('Hello World!')
        '\n        @type key: bytes\n        @param key: The key to test\n\n        @return: A true value if this dirdbm has the specified key, a false\n        value otherwise.\n        '
        if not type(key) == bytes:
            raise TypeError('DirDBM key must be bytes')
        key = self._encode(key)
        return self._dnamePath.child(key).isfile()

    def setdefault(self, key, value):
        if False:
            print('Hello World!')
        '\n        @type key: bytes\n        @param key: The key to lookup\n\n        @param value: The value to associate with key if key is not already\n        associated with a value.\n        '
        if key not in self:
            self[key] = value
            return value
        return self[key]

    def get(self, key, default=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        @type key: bytes\n        @param key: The key to lookup\n\n        @param default: The value to return if the given key does not exist\n\n        @return: The value associated with C{key} or C{default} if not\n        L{DirDBM.has_key(key)}\n        '
        if key in self:
            return self[key]
        else:
            return default

    def __contains__(self, key):
        if False:
            while True:
                i = 10
        '\n        @see: L{DirDBM.has_key}\n        '
        return self.has_key(key)

    def update(self, dict):
        if False:
            i = 10
            return i + 15
        '\n        Add all the key/value pairs in L{dict} to this dirdbm.  Any conflicting\n        keys will be overwritten with the values from L{dict}.\n\n        @type dict: mapping\n        @param dict: A mapping of key/value pairs to add to this dirdbm.\n        '
        for (key, val) in dict.items():
            self[key] = val

    def copyTo(self, path):
        if False:
            while True:
                i = 10
        '\n        Copy the contents of this dirdbm to the dirdbm at C{path}.\n\n        @type path: L{str}\n        @param path: The path of the dirdbm to copy to.  If a dirdbm\n        exists at the destination path, it is cleared first.\n\n        @rtype: C{DirDBM}\n        @return: The dirdbm this dirdbm was copied to.\n        '
        path = FilePath(path)
        assert path != self._dnamePath
        d = self.__class__(path.path)
        d.clear()
        for k in self.keys():
            d[k] = self[k]
        return d

    def clear(self):
        if False:
            while True:
                i = 10
        '\n        Delete all key/value pairs in this dirdbm.\n        '
        for k in self.keys():
            del self[k]

    def close(self):
        if False:
            print('Hello World!')
        '\n        Close this dbm: no-op, for dbm-style interface compliance.\n        '

    def getModificationTime(self, key):
        if False:
            return 10
        '\n        Returns modification time of an entry.\n\n        @return: Last modification date (seconds since epoch) of entry C{key}\n        @raise KeyError: Raised when there is no such key\n        '
        if not type(key) == bytes:
            raise TypeError('DirDBM key must be bytes')
        path = self._dnamePath.child(self._encode(key))
        if path.isfile():
            return path.getModificationTime()
        else:
            raise KeyError(key)

class Shelf(DirDBM):
    """
    A directory with a DBM shelf interface.

    This class presents a hash-like interface to a directory of small,
    flat files. Keys must be strings, but values can be any given object.
    """

    def __setitem__(self, k, v):
        if False:
            for i in range(10):
                print('nop')
        '\n        C{shelf[foo] = bar}\n        Create or modify a textfile in this directory.\n\n        @type k: str\n        @param k: The key to set\n\n        @param v: The value to associate with C{key}\n        '
        v = pickle.dumps(v)
        DirDBM.__setitem__(self, k, v)

    def __getitem__(self, k):
        if False:
            return 10
        '\n        C{dirdbm[foo]}\n        Get and unpickle the contents of a file in this directory.\n\n        @type k: bytes\n        @param k: The key to lookup\n\n        @return: The value associated with the given key\n        @raise KeyError: Raised if the given key does not exist\n        '
        return pickle.loads(DirDBM.__getitem__(self, k))

def open(file, flag=None, mode=None):
    if False:
        i = 10
        return i + 15
    "\n    This is for 'anydbm' compatibility.\n\n    @param file: The parameter to pass to the DirDBM constructor.\n\n    @param flag: ignored\n    @param mode: ignored\n    "
    return DirDBM(file)
__all__ = ['open', 'DirDBM', 'Shelf']