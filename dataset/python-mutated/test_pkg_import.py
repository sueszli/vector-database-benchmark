import os
import sys
import shutil
import string
import random
import tempfile
import unittest
from importlib.util import cache_from_source
from test.support.os_helper import create_empty_file

class TestImport(unittest.TestCase):

    def __init__(self, *args, **kw):
        if False:
            return 10
        self.package_name = 'PACKAGE_'
        while self.package_name in sys.modules:
            self.package_name += random.choice(string.ascii_letters)
        self.module_name = self.package_name + '.foo'
        unittest.TestCase.__init__(self, *args, **kw)

    def remove_modules(self):
        if False:
            i = 10
            return i + 15
        for module_name in (self.package_name, self.module_name):
            if module_name in sys.modules:
                del sys.modules[module_name]

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.test_dir = tempfile.mkdtemp()
        sys.path.append(self.test_dir)
        self.package_dir = os.path.join(self.test_dir, self.package_name)
        os.mkdir(self.package_dir)
        create_empty_file(os.path.join(self.package_dir, '__init__.py'))
        self.module_path = os.path.join(self.package_dir, 'foo.py')

    def tearDown(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.test_dir)
        self.assertNotEqual(sys.path.count(self.test_dir), 0)
        sys.path.remove(self.test_dir)
        self.remove_modules()

    def rewrite_file(self, contents):
        if False:
            return 10
        compiled_path = cache_from_source(self.module_path)
        if os.path.exists(compiled_path):
            os.remove(compiled_path)
        with open(self.module_path, 'w', encoding='utf-8') as f:
            f.write(contents)

    def test_package_import__semantics(self):
        if False:
            while True:
                i = 10
        self.rewrite_file('for')
        try:
            __import__(self.module_name)
        except SyntaxError:
            pass
        else:
            raise RuntimeError('Failed to induce SyntaxError')
        self.assertNotIn(self.module_name, sys.modules)
        self.assertFalse(hasattr(sys.modules[self.package_name], 'foo'))
        var = 'a'
        while var in dir(__builtins__):
            var += random.choice(string.ascii_letters)
        self.rewrite_file(var)
        try:
            __import__(self.module_name)
        except NameError:
            pass
        else:
            raise RuntimeError('Failed to induce NameError.')
        self.rewrite_file('%s = 1' % var)
        module = __import__(self.module_name).foo
        self.assertEqual(getattr(module, var), 1)
if __name__ == '__main__':
    unittest.main()