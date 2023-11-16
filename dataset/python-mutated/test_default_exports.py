"""Test the default exports of the top level packages."""
from __future__ import annotations
import inspect
import unittest
import bson
import gridfs
import pymongo
BSON_IGNORE = []
GRIDFS_IGNORE = ['ASCENDING', 'DESCENDING', 'ClientSession', 'Collection', 'ObjectId', 'validate_string', 'Database', 'ConfigurationError', 'WriteConcern']
PYMONGO_IGNORE = []
GLOBAL_INGORE = ['TYPE_CHECKING', 'annotations']

class TestDefaultExports(unittest.TestCase):

    def check_module(self, mod, ignores):
        if False:
            return 10
        names = dir(mod)
        names.remove('__all__')
        for name in mod.__all__:
            if name not in names and name not in ignores:
                self.fail(f'{name} was included in {mod}.__all__ but is not a valid symbol')
        for name in names:
            if name not in mod.__all__ and name not in ignores:
                if name in GLOBAL_INGORE:
                    continue
                value = getattr(mod, name)
                if inspect.ismodule(value):
                    continue
                if getattr(value, '__module__', None) == 'typing':
                    continue
                if not name.startswith('_'):
                    self.fail(f'{name} was not included in {mod}.__all__')

    def test_pymongo(self):
        if False:
            for i in range(10):
                print('nop')
        self.check_module(pymongo, PYMONGO_IGNORE)

    def test_gridfs(self):
        if False:
            while True:
                i = 10
        self.check_module(gridfs, GRIDFS_IGNORE)

    def test_bson(self):
        if False:
            while True:
                i = 10
        self.check_module(bson, BSON_IGNORE)
if __name__ == '__main__':
    unittest.main()