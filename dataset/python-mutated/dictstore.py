"""
Dictionary store
=================

Use a Python dictionary as a store.
"""
__all__ = ('DictStore',)
try:
    import cPickle as pickle
except ImportError:
    import pickle
import errno
from os.path import exists, abspath, dirname
from kivy.compat import iteritems
from kivy.storage import AbstractStore

class DictStore(AbstractStore):
    """Store implementation using a pickled `dict`.
    See the :mod:`kivy.storage` module documentation for more information.
    """

    def __init__(self, filename, data=None, **kwargs):
        if False:
            while True:
                i = 10
        if isinstance(filename, dict):
            self.filename = None
            self._data = filename
        else:
            self.filename = filename
            self._data = data or {}
        self._is_changed = True
        super(DictStore, self).__init__(**kwargs)

    def store_load(self):
        if False:
            print('Hello World!')
        if self.filename is None:
            return
        if not exists(self.filename):
            folder = abspath(dirname(self.filename))
            if not exists(folder):
                not_found = IOError("The folder '{}' doesn't exist!".format(folder))
                not_found.errno = errno.ENOENT
                raise not_found
            return
        with open(self.filename, 'rb') as fd:
            data = fd.read()
            if data:
                self._data = pickle.loads(data)

    def store_sync(self):
        if False:
            i = 10
            return i + 15
        if self.filename is None:
            return
        if not self._is_changed:
            return
        with open(self.filename, 'wb') as fd:
            pickle.dump(self._data, fd)
        self._is_changed = False

    def store_exists(self, key):
        if False:
            return 10
        return key in self._data

    def store_get(self, key):
        if False:
            i = 10
            return i + 15
        return self._data[key]

    def store_put(self, key, value):
        if False:
            while True:
                i = 10
        self._data[key] = value
        self._is_changed = True
        return True

    def store_delete(self, key):
        if False:
            i = 10
            return i + 15
        del self._data[key]
        self._is_changed = True
        return True

    def store_find(self, filters):
        if False:
            return 10
        for (key, values) in iteritems(self._data):
            found = True
            for (fkey, fvalue) in iteritems(filters):
                if fkey not in values:
                    found = False
                    break
                if values[fkey] != fvalue:
                    found = False
                    break
            if found:
                yield (key, values)

    def store_count(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self._data)

    def store_keys(self):
        if False:
            i = 10
            return i + 15
        return list(self._data.keys())