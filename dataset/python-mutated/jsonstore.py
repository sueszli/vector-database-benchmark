"""
JSON store
==========

A :mod:`Storage <kivy.storage>` module used to save/load key-value pairs from
a json file.
"""
__all__ = ('JsonStore',)
import errno
from os.path import exists, abspath, dirname
from kivy.compat import iteritems
from kivy.storage import AbstractStore
from json import loads, dump

class JsonStore(AbstractStore):
    """Store implementation using a json file for storing the key-value pairs.
    See the :mod:`kivy.storage` module documentation for more information.
    """

    def __init__(self, filename, indent=None, sort_keys=False, **kwargs):
        if False:
            return 10
        self.filename = filename
        self.indent = indent
        self.sort_keys = sort_keys
        self._data = {}
        self._is_changed = True
        super(JsonStore, self).__init__(**kwargs)

    def store_load(self):
        if False:
            for i in range(10):
                print('nop')
        if not exists(self.filename):
            folder = abspath(dirname(self.filename))
            if not exists(folder):
                not_found = IOError("The folder '{}' doesn't exist!".format(folder))
                not_found.errno = errno.ENOENT
                raise not_found
            return
        with open(self.filename) as fd:
            data = fd.read()
            if len(data) == 0:
                return
            self._data = loads(data)

    def store_sync(self):
        if False:
            print('Hello World!')
        if not self._is_changed:
            return
        with open(self.filename, 'w') as fd:
            dump(self._data, fd, indent=self.indent, sort_keys=self.sort_keys)
        self._is_changed = False

    def store_exists(self, key):
        if False:
            for i in range(10):
                print('nop')
        return key in self._data

    def store_get(self, key):
        if False:
            return 10
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
            while True:
                i = 10
        del self._data[key]
        self._is_changed = True
        return True

    def store_find(self, filters):
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        return len(self._data)

    def store_keys(self):
        if False:
            while True:
                i = 10
        return list(self._data.keys())