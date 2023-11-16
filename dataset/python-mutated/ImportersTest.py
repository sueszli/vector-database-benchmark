import os
import unittest
from collections import OrderedDict
from importlib.machinery import ModuleSpec, SourceFileLoader
from inspect import isclass
from coalib.collecting.Importers import import_objects

class ImportObjectsTest(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        current_dir = os.path.split(__file__)[0]
        self.test_dir = os.path.join(current_dir, 'importers_test_dir')
        self.testfile1_path = os.path.join(self.test_dir, 'file_one.py')
        self.testfile2_path = os.path.join(self.test_dir, 'file_two.py')

    def check_imported_file_one_test(self, obj):
        if False:
            i = 10
            return i + 15
        self.assertTrue(isclass(obj))
        self.assertEqual(obj.__name__, 'test')
        self.assertEqual(obj.__module__, 'file_one')
        instance = obj()
        self.assertIsInstance(instance, list)

    def test_no_file(self):
        if False:
            print('Hello World!')
        self.assertEqual(import_objects([]), [])

    def test_file_one_internal_structure(self):
        if False:
            print('Hello World!')
        objs = import_objects(self.testfile1_path)
        self.assertIsInstance(objs, list)
        self.assertEqual(len(objs), 12)
        self.assertIsInstance(objs[0], dict)
        self.assertIn('__name__', objs[0])
        self.assertEqual(objs[0]['__name__'], 'builtins')
        self.assertIn('copyright', objs[0])
        self.assertIsInstance(objs[1], str)
        self.assertTrue(objs[1].endswith('.pyc'))
        self.assertTrue(objs[1].startswith(self.test_dir))
        self.assertIsNone(objs[2])
        self.assertIsInstance(objs[3], str)
        self.assertEqual(objs[3], self.testfile1_path)
        self.assertIsInstance(objs[4], SourceFileLoader)
        self.assertIsInstance(objs[5], str)
        self.assertEqual(objs[5], 'file_one')
        self.assertIsInstance(objs[6], str)
        self.assertEqual(objs[6], '')
        self.assertIsInstance(objs[7], ModuleSpec)
        self.assertIsInstance(objs[8], list)
        self.assertEqual(objs[8], [1, 2, 3])
        self.assertIsInstance(objs[9], list)
        self.assertEqual(objs[9], [1, 2, 4])
        self.assertIsInstance(objs[10], bool)
        self.assertIs(objs[10], True)
        self.check_imported_file_one_test(objs[11])

    def test_name_import(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(import_objects((self.testfile1_path, self.testfile2_path), names='name')), 2)
        self.assertEqual(len(import_objects((self.testfile1_path, self.testfile2_path), names='last_name')), 0)

    def test_type_import(self):
        if False:
            i = 10
            return i + 15
        objs = import_objects(self.testfile1_path, types=list, verbose=True)
        self.assertEqual(len(objs), 2)
        self.assertIsInstance(objs[0], list)
        self.assertEqual(objs[0], [1, 2, 3])
        self.assertIsInstance(objs[1], list)
        self.assertEqual(objs[1], [1, 2, 4])
        self.assertEqual(len(import_objects((self.testfile1_path, self.testfile2_path), names='name', types=OrderedDict, verbose=True)), 0)

    def test_class_import(self):
        if False:
            print('Hello World!')
        objs = import_objects((self.testfile1_path, self.testfile2_path), supers=list, verbose=True)
        self.assertEqual(len(objs), 1)
        self.check_imported_file_one_test(objs[0])
        self.assertEqual(len(import_objects((self.testfile1_path, self.testfile2_path), supers=str, verbose=True)), 0)

    def test_attribute_import(self):
        if False:
            while True:
                i = 10
        objs = import_objects((self.testfile1_path, self.testfile2_path), attributes='method', local=True, verbose=True)
        self.assertEqual(len(objs), 1)
        self.check_imported_file_one_test(objs[0])
        self.assertEqual(len(import_objects((self.testfile1_path, self.testfile2_path), attributes='something', verbose=True)), 0)

    def test_local_definition(self):
        if False:
            while True:
                i = 10
        self.assertEqual(len(import_objects((self.testfile1_path, self.testfile2_path), attributes='method', verbose=True)), 2)
        self.assertEqual(len(import_objects((self.testfile1_path, self.testfile2_path), attributes='method', local=True, verbose=True)), 1)

    def test_invalid_file(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(ImportError):
            import_objects('some/invalid/path', attributes='method', local=True, verbose=True)
        with self.assertRaises(ImportError):
            import_objects('some/invalid/path', attributes='method', local=True, verbose=False)