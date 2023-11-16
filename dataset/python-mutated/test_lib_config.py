from __future__ import unicode_literals, division, absolute_import, print_function
import os
from powerline.lib.config import ConfigLoader
from tests.modules import TestCase
from tests.modules.lib.fsconfig import FSTree
FILE_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfglib')

class LoadedList(list):

    def pop_all(self):
        if False:
            i = 10
            return i + 15
        try:
            return self[:]
        finally:
            self[:] = ()
loaded = LoadedList()

def on_load(key):
    if False:
        return 10
    loaded.append(key)

def check_file(path):
    if False:
        for i in range(10):
            print('nop')
    if os.path.exists(path):
        return path
    else:
        raise IOError

class TestLoaderCondition(TestCase):

    def test_update_missing(self):
        if False:
            return 10
        loader = ConfigLoader(run_once=True)
        fpath = os.path.join(FILE_ROOT, 'file.json')
        self.assertRaises(IOError, loader.load, fpath)
        loader.register_missing(check_file, on_load, fpath)
        loader.update()
        with FSTree({'file': {'test': 1}}, root=FILE_ROOT):
            loader.update()
            self.assertEqual(loader.load(fpath), {'test': 1})
            self.assertEqual(loaded.pop_all(), [fpath])
if __name__ == '__main__':
    from tests.modules import main
    main()