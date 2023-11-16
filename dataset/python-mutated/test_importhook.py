import unittest
import sys, os
import bottle

class TestImportHooks(unittest.TestCase):

    def make_module(self, name, **args):
        if False:
            for i in range(10):
                print('nop')
        mod = sys.modules.setdefault(name, bottle.new_module(name))
        mod.__file__ = '<virtual %s>' % name
        mod.__dict__.update(**args)
        return mod

    def test_direkt_import(self):
        if False:
            print('Hello World!')
        mod = self.make_module('bottle_test')
        import bottle.ext.test
        self.assertEqual(bottle.ext.test, mod)

    def test_from_import(self):
        if False:
            return 10
        mod = self.make_module('bottle_test')
        from bottle.ext import test
        self.assertEqual(test, mod)

    def test_data_import(self):
        if False:
            print('Hello World!')
        mod = self.make_module('bottle_test', item='value')
        from bottle.ext.test import item
        self.assertEqual(item, 'value')

    def test_import_fail(self):
        if False:
            for i in range(10):
                print('nop')
        ' Test a simple static page with this server adapter. '

        def test():
            if False:
                while True:
                    i = 10
            import bottle.ext.doesnotexist
        self.assertRaises(ImportError, test)

    def test_ext_isfile(self):
        if False:
            while True:
                i = 10
        ' The virtual module needs a valid __file__ attribute.\n            If not, the Google app engine development server crashes on windows.\n        '
        from bottle import ext
        self.assertTrue(os.path.isfile(ext.__file__))