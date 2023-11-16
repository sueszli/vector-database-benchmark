import sys
import os
import marshal
import importlib
import importlib.util
import struct
import time
import unittest
import unittest.mock
import warnings
from test import support
from test.support import import_helper
from test.support import os_helper
from zipfile import ZipFile, ZipInfo, ZIP_STORED, ZIP_DEFLATED
import zipimport
import linecache
import doctest
import inspect
import io
from traceback import extract_tb, extract_stack, print_tb
try:
    import zlib
except ImportError:
    zlib = None
test_src = 'def get_name():\n    return __name__\ndef get_file():\n    return __file__\n'
test_co = compile(test_src, '<???>', 'exec')
raise_src = 'def do_raise(): raise TypeError\n'

def make_pyc(co, mtime, size):
    if False:
        return 10
    data = marshal.dumps(co)
    pyc = importlib.util.MAGIC_NUMBER + struct.pack('<iLL', 0, int(mtime) & 4294967295, size & 4294967295) + data
    return pyc

def module_path_to_dotted_name(path):
    if False:
        while True:
            i = 10
    return path.replace(os.sep, '.')
NOW = time.time()
test_pyc = make_pyc(test_co, NOW, len(test_src))
TESTMOD = 'ziptestmodule'
TESTPACK = 'ziptestpackage'
TESTPACK2 = 'ziptestpackage2'
TEMP_DIR = os.path.abspath('junk95142')
TEMP_ZIP = os.path.abspath('junk95142.zip')
pyc_file = importlib.util.cache_from_source(TESTMOD + '.py')
pyc_ext = '.pyc'

class ImportHooksBaseTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.path = sys.path[:]
        self.meta_path = sys.meta_path[:]
        self.path_hooks = sys.path_hooks[:]
        sys.path_importer_cache.clear()
        self.modules_before = import_helper.modules_setup()

    def tearDown(self):
        if False:
            return 10
        sys.path[:] = self.path
        sys.meta_path[:] = self.meta_path
        sys.path_hooks[:] = self.path_hooks
        sys.path_importer_cache.clear()
        import_helper.modules_cleanup(*self.modules_before)

class UncompressedZipImportTestCase(ImportHooksBaseTestCase):
    compression = ZIP_STORED

    def setUp(self):
        if False:
            while True:
                i = 10
        linecache.clearcache()
        zipimport._zip_directory_cache.clear()
        ImportHooksBaseTestCase.setUp(self)

    def makeTree(self, files, dirName=TEMP_DIR):
        if False:
            print('Hello World!')
        self.addCleanup(os_helper.rmtree, dirName)
        for (name, (mtime, data)) in files.items():
            path = os.path.join(dirName, name)
            if path[-1] == os.sep:
                if not os.path.isdir(path):
                    os.makedirs(path)
            else:
                dname = os.path.dirname(path)
                if not os.path.isdir(dname):
                    os.makedirs(dname)
                with open(path, 'wb') as fp:
                    fp.write(data)

    def makeZip(self, files, zipName=TEMP_ZIP, **kw):
        if False:
            while True:
                i = 10
        self.addCleanup(os_helper.unlink, zipName)
        with ZipFile(zipName, 'w') as z:
            for (name, (mtime, data)) in files.items():
                zinfo = ZipInfo(name, time.localtime(mtime))
                zinfo.compress_type = self.compression
                z.writestr(zinfo, data)
            comment = kw.get('comment', None)
            if comment is not None:
                z.comment = comment
        stuff = kw.get('stuff', None)
        if stuff is not None:
            with open(zipName, 'rb') as f:
                data = f.read()
            with open(zipName, 'wb') as f:
                f.write(stuff)
                f.write(data)

    def doTest(self, expected_ext, files, *modules, **kw):
        if False:
            for i in range(10):
                print('nop')
        self.makeZip(files, **kw)
        sys.path.insert(0, TEMP_ZIP)
        mod = importlib.import_module('.'.join(modules))
        call = kw.get('call')
        if call is not None:
            call(mod)
        if expected_ext:
            file = mod.get_file()
            self.assertEqual(file, os.path.join(TEMP_ZIP, *modules) + expected_ext)

    def testAFakeZlib(self):
        if False:
            while True:
                i = 10
        if 'zlib' in sys.builtin_module_names:
            self.skipTest('zlib is a builtin module')
        if 'zlib' in sys.modules:
            del sys.modules['zlib']
        files = {'zlib.py': (NOW, test_src)}
        try:
            self.doTest('.py', files, 'zlib')
        except ImportError:
            if self.compression != ZIP_DEFLATED:
                self.fail('expected test to not raise ImportError')
        else:
            if self.compression != ZIP_STORED:
                self.fail('expected test to raise ImportError')

    def testPy(self):
        if False:
            for i in range(10):
                print('nop')
        files = {TESTMOD + '.py': (NOW, test_src)}
        self.doTest('.py', files, TESTMOD)

    def testPyc(self):
        if False:
            while True:
                i = 10
        files = {TESTMOD + pyc_ext: (NOW, test_pyc)}
        self.doTest(pyc_ext, files, TESTMOD)

    def testBoth(self):
        if False:
            while True:
                i = 10
        files = {TESTMOD + '.py': (NOW, test_src), TESTMOD + pyc_ext: (NOW, test_pyc)}
        self.doTest(pyc_ext, files, TESTMOD)

    def testUncheckedHashBasedPyc(self):
        if False:
            while True:
                i = 10
        source = b"state = 'old'"
        source_hash = importlib.util.source_hash(source)
        bytecode = importlib._bootstrap_external._code_to_hash_pyc(compile(source, '???', 'exec'), source_hash, False)
        files = {TESTMOD + '.py': (NOW, "state = 'new'"), TESTMOD + '.pyc': (NOW - 20, bytecode)}

        def check(mod):
            if False:
                print('Hello World!')
            self.assertEqual(mod.state, 'old')
        self.doTest(None, files, TESTMOD, call=check)

    @unittest.mock.patch('_imp.check_hash_based_pycs', 'always')
    def test_checked_hash_based_change_pyc(self):
        if False:
            i = 10
            return i + 15
        source = b"state = 'old'"
        source_hash = importlib.util.source_hash(source)
        bytecode = importlib._bootstrap_external._code_to_hash_pyc(compile(source, '???', 'exec'), source_hash, False)
        files = {TESTMOD + '.py': (NOW, "state = 'new'"), TESTMOD + '.pyc': (NOW - 20, bytecode)}

        def check(mod):
            if False:
                return 10
            self.assertEqual(mod.state, 'new')
        self.doTest(None, files, TESTMOD, call=check)

    def testEmptyPy(self):
        if False:
            while True:
                i = 10
        files = {TESTMOD + '.py': (NOW, '')}
        self.doTest(None, files, TESTMOD)

    def testBadMagic(self):
        if False:
            i = 10
            return i + 15
        badmagic_pyc = bytearray(test_pyc)
        badmagic_pyc[0] ^= 4
        files = {TESTMOD + '.py': (NOW, test_src), TESTMOD + pyc_ext: (NOW, badmagic_pyc)}
        self.doTest('.py', files, TESTMOD)

    def testBadMagic2(self):
        if False:
            for i in range(10):
                print('nop')
        badmagic_pyc = bytearray(test_pyc)
        badmagic_pyc[0] ^= 4
        files = {TESTMOD + pyc_ext: (NOW, badmagic_pyc)}
        try:
            self.doTest('.py', files, TESTMOD)
            self.fail('This should not be reached')
        except zipimport.ZipImportError as exc:
            self.assertIsInstance(exc.__cause__, ImportError)
            self.assertIn('magic number', exc.__cause__.msg)

    def testBadMTime(self):
        if False:
            print('Hello World!')
        badtime_pyc = bytearray(test_pyc)
        badtime_pyc[11] ^= 2
        files = {TESTMOD + '.py': (NOW, test_src), TESTMOD + pyc_ext: (NOW, badtime_pyc)}
        self.doTest('.py', files, TESTMOD)

    def test2038MTime(self):
        if False:
            for i in range(10):
                print('nop')
        twenty_thirty_eight_pyc = make_pyc(test_co, 2 ** 32 - 1, len(test_src))
        files = {TESTMOD + '.py': (NOW, test_src), TESTMOD + pyc_ext: (NOW, twenty_thirty_eight_pyc)}
        self.doTest('.py', files, TESTMOD)

    def testPackage(self):
        if False:
            for i in range(10):
                print('nop')
        packdir = TESTPACK + os.sep
        files = {packdir + '__init__' + pyc_ext: (NOW, test_pyc), packdir + TESTMOD + pyc_ext: (NOW, test_pyc)}
        self.doTest(pyc_ext, files, TESTPACK, TESTMOD)

    def testSubPackage(self):
        if False:
            return 10
        packdir = TESTPACK + os.sep
        packdir2 = packdir + TESTPACK2 + os.sep
        files = {packdir + '__init__' + pyc_ext: (NOW, test_pyc), packdir2 + '__init__' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc)}
        self.doTest(pyc_ext, files, TESTPACK, TESTPACK2, TESTMOD)

    def testSubNamespacePackage(self):
        if False:
            return 10
        packdir = TESTPACK + os.sep
        packdir2 = packdir + TESTPACK2 + os.sep
        files = {packdir: (NOW, ''), packdir2: (NOW, ''), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc)}
        self.doTest(pyc_ext, files, TESTPACK, TESTPACK2, TESTMOD)

    def testMixedNamespacePackage(self):
        if False:
            while True:
                i = 10
        packdir = TESTPACK + os.sep
        packdir2 = packdir + TESTPACK2 + os.sep
        packdir3 = packdir2 + TESTPACK + '3' + os.sep
        files1 = {packdir: (NOW, ''), packdir + TESTMOD + pyc_ext: (NOW, test_pyc), packdir2: (NOW, ''), packdir3: (NOW, ''), packdir3 + TESTMOD + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + '3' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc)}
        files2 = {packdir: (NOW, ''), packdir + TESTMOD + '2' + pyc_ext: (NOW, test_pyc), packdir2: (NOW, ''), packdir2 + TESTMOD + '2' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc)}
        zip1 = os.path.abspath('path1.zip')
        self.makeZip(files1, zip1)
        zip2 = TEMP_DIR
        self.makeTree(files2, zip2)
        sys.path.insert(0, zip1)
        sys.path.insert(0, zip2)
        mod = importlib.import_module(TESTPACK)
        self.assertEqual(2, len(mod.__path__))
        (p1, p2) = mod.__path__
        self.assertEqual(os.path.basename(TEMP_DIR), p1.split(os.sep)[-2])
        self.assertEqual('path1.zip', p2.split(os.sep)[-2])
        mod = importlib.import_module(packdir3.replace(os.sep, '.')[:-1])
        self.assertEqual(1, len(mod.__path__))
        mpath = list(mod.__path__)[0].split('path1.zip' + os.sep)[1]
        self.assertEqual(packdir3[:-1], mpath)
        mod = importlib.import_module('.'.join((TESTPACK, TESTMOD)))
        self.assertEqual('path1.zip', mod.__file__.split(os.sep)[-3])
        mod = importlib.import_module('.'.join((TESTPACK, TESTMOD + '2')))
        self.assertEqual(os.path.basename(TEMP_DIR), mod.__file__.split(os.sep)[-3])
        subpkg = '.'.join((TESTPACK, TESTPACK2))
        mod = importlib.import_module(subpkg)
        self.assertEqual(2, len(mod.__path__))
        (p1, p2) = mod.__path__
        self.assertEqual(os.path.basename(TEMP_DIR), p1.split(os.sep)[-3])
        self.assertEqual('path1.zip', p2.split(os.sep)[-3])
        mod = importlib.import_module('.'.join((subpkg, TESTMOD)))
        self.assertEqual(os.path.basename(TEMP_DIR), mod.__file__.split(os.sep)[-4])
        mod = importlib.import_module('.'.join((subpkg, TESTMOD + '2')))
        self.assertEqual(os.path.basename(TEMP_DIR), mod.__file__.split(os.sep)[-4])
        mod = importlib.import_module('.'.join((subpkg, TESTMOD + '3')))
        self.assertEqual('path1.zip', mod.__file__.split(os.sep)[-4])

    def testNamespacePackage(self):
        if False:
            return 10
        packdir = TESTPACK + os.sep
        packdir2 = packdir + TESTPACK2 + os.sep
        packdir3 = packdir2 + TESTPACK + '3' + os.sep
        files1 = {packdir: (NOW, ''), packdir + TESTMOD + pyc_ext: (NOW, test_pyc), packdir2: (NOW, ''), packdir3: (NOW, ''), packdir3 + TESTMOD + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + '3' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc)}
        zip1 = os.path.abspath('path1.zip')
        self.makeZip(files1, zip1)
        files2 = {packdir: (NOW, ''), packdir + TESTMOD + '2' + pyc_ext: (NOW, test_pyc), packdir2: (NOW, ''), packdir2 + TESTMOD + '2' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc)}
        zip2 = os.path.abspath('path2.zip')
        self.makeZip(files2, zip2)
        sys.path.insert(0, zip1)
        sys.path.insert(0, zip2)
        mod = importlib.import_module(TESTPACK)
        self.assertEqual(2, len(mod.__path__))
        (p1, p2) = mod.__path__
        self.assertEqual('path2.zip', p1.split(os.sep)[-2])
        self.assertEqual('path1.zip', p2.split(os.sep)[-2])
        mod = importlib.import_module(packdir3.replace(os.sep, '.')[:-1])
        self.assertEqual(1, len(mod.__path__))
        mpath = list(mod.__path__)[0].split('path1.zip' + os.sep)[1]
        self.assertEqual(packdir3[:-1], mpath)
        mod = importlib.import_module('.'.join((TESTPACK, TESTMOD)))
        self.assertEqual('path1.zip', mod.__file__.split(os.sep)[-3])
        mod = importlib.import_module('.'.join((TESTPACK, TESTMOD + '2')))
        self.assertEqual('path2.zip', mod.__file__.split(os.sep)[-3])
        subpkg = '.'.join((TESTPACK, TESTPACK2))
        mod = importlib.import_module(subpkg)
        self.assertEqual(2, len(mod.__path__))
        (p1, p2) = mod.__path__
        self.assertEqual('path2.zip', p1.split(os.sep)[-3])
        self.assertEqual('path1.zip', p2.split(os.sep)[-3])
        mod = importlib.import_module('.'.join((subpkg, TESTMOD)))
        self.assertEqual('path2.zip', mod.__file__.split(os.sep)[-4])
        mod = importlib.import_module('.'.join((subpkg, TESTMOD + '2')))
        self.assertEqual('path2.zip', mod.__file__.split(os.sep)[-4])
        mod = importlib.import_module('.'.join((subpkg, TESTMOD + '3')))
        self.assertEqual('path1.zip', mod.__file__.split(os.sep)[-4])

    def testZipImporterMethods(self):
        if False:
            for i in range(10):
                print('nop')
        packdir = TESTPACK + os.sep
        packdir2 = packdir + TESTPACK2 + os.sep
        files = {packdir + '__init__' + pyc_ext: (NOW, test_pyc), packdir2 + '__init__' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc), 'spam' + pyc_ext: (NOW, test_pyc)}
        self.addCleanup(os_helper.unlink, TEMP_ZIP)
        with ZipFile(TEMP_ZIP, 'w') as z:
            for (name, (mtime, data)) in files.items():
                zinfo = ZipInfo(name, time.localtime(mtime))
                zinfo.compress_type = self.compression
                zinfo.comment = b'spam'
                z.writestr(zinfo, data)
        zi = zipimport.zipimporter(TEMP_ZIP)
        self.assertEqual(zi.archive, TEMP_ZIP)
        self.assertTrue(zi.is_package(TESTPACK))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            find_mod = zi.find_module('spam')
            self.assertIsNotNone(find_mod)
            self.assertIsInstance(find_mod, zipimport.zipimporter)
            self.assertFalse(find_mod.is_package('spam'))
            load_mod = find_mod.load_module('spam')
            self.assertEqual(find_mod.get_filename('spam'), load_mod.__file__)
            mod = zi.load_module(TESTPACK)
            self.assertEqual(zi.get_filename(TESTPACK), mod.__file__)
        spec = zi.find_spec('spam')
        self.assertIsNotNone(spec)
        self.assertIsInstance(spec.loader, zipimport.zipimporter)
        self.assertFalse(spec.loader.is_package('spam'))
        exec_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(exec_mod)
        self.assertEqual(spec.loader.get_filename('spam'), exec_mod.__file__)
        spec = zi.find_spec(TESTPACK)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(zi.get_filename(TESTPACK), mod.__file__)
        existing_pack_path = importlib.import_module(TESTPACK).__path__[0]
        expected_path_path = os.path.join(TEMP_ZIP, TESTPACK)
        self.assertEqual(existing_pack_path, expected_path_path)
        self.assertFalse(zi.is_package(packdir + '__init__'))
        self.assertTrue(zi.is_package(packdir + TESTPACK2))
        self.assertFalse(zi.is_package(packdir2 + TESTMOD))
        mod_path = packdir2 + TESTMOD
        mod_name = module_path_to_dotted_name(mod_path)
        mod = importlib.import_module(mod_name)
        self.assertTrue(mod_name in sys.modules)
        self.assertIsNone(zi.get_source(TESTPACK))
        self.assertIsNone(zi.get_source(mod_path))
        self.assertEqual(zi.get_filename(mod_path), mod.__file__)
        loader = mod.__spec__.loader
        self.assertIsNone(loader.get_source(mod_name))
        self.assertEqual(loader.get_filename(mod_name), mod.__file__)
        zi2 = zipimport.zipimporter(TEMP_ZIP + os.sep + TESTPACK)
        self.assertEqual(zi2.archive, TEMP_ZIP)
        self.assertEqual(zi2.prefix, TESTPACK + os.sep)

    def testInvalidateCaches(self):
        if False:
            for i in range(10):
                print('nop')
        packdir = TESTPACK + os.sep
        packdir2 = packdir + TESTPACK2 + os.sep
        files = {packdir + '__init__' + pyc_ext: (NOW, test_pyc), packdir2 + '__init__' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc), 'spam' + pyc_ext: (NOW, test_pyc)}
        self.addCleanup(os_helper.unlink, TEMP_ZIP)
        with ZipFile(TEMP_ZIP, 'w') as z:
            for (name, (mtime, data)) in files.items():
                zinfo = ZipInfo(name, time.localtime(mtime))
                zinfo.compress_type = self.compression
                zinfo.comment = b'spam'
                z.writestr(zinfo, data)
        zi = zipimport.zipimporter(TEMP_ZIP)
        self.assertEqual(zi._files.keys(), files.keys())
        zi.invalidate_caches()
        self.assertEqual(zi._files.keys(), files.keys())
        newfile = {'spam2' + pyc_ext: (NOW, test_pyc)}
        files.update(newfile)
        with ZipFile(TEMP_ZIP, 'a') as z:
            for (name, (mtime, data)) in newfile.items():
                zinfo = ZipInfo(name, time.localtime(mtime))
                zinfo.compress_type = self.compression
                zinfo.comment = b'spam'
                z.writestr(zinfo, data)
        zi.invalidate_caches()
        self.assertEqual(zi._files.keys(), files.keys())
        spec = zi.find_spec('spam2')
        self.assertIsNotNone(spec)
        self.assertIsInstance(spec.loader, zipimport.zipimporter)
        os.remove(TEMP_ZIP)
        zi.invalidate_caches()
        self.assertFalse(zi._files)
        self.assertIsNone(zipimport._zip_directory_cache.get(zi.archive))
        self.assertIsNone(zi.find_spec('name_does_not_matter'))

    def testZipImporterMethodsInSubDirectory(self):
        if False:
            print('Hello World!')
        packdir = TESTPACK + os.sep
        packdir2 = packdir + TESTPACK2 + os.sep
        files = {packdir2 + '__init__' + pyc_ext: (NOW, test_pyc), packdir2 + TESTMOD + pyc_ext: (NOW, test_pyc)}
        self.addCleanup(os_helper.unlink, TEMP_ZIP)
        with ZipFile(TEMP_ZIP, 'w') as z:
            for (name, (mtime, data)) in files.items():
                zinfo = ZipInfo(name, time.localtime(mtime))
                zinfo.compress_type = self.compression
                zinfo.comment = b'eggs'
                z.writestr(zinfo, data)
        zi = zipimport.zipimporter(TEMP_ZIP + os.sep + packdir)
        self.assertEqual(zi.archive, TEMP_ZIP)
        self.assertEqual(zi.prefix, packdir)
        self.assertTrue(zi.is_package(TESTPACK2))
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            mod = zi.load_module(TESTPACK2)
            self.assertEqual(zi.get_filename(TESTPACK2), mod.__file__)
        spec = zi.find_spec(TESTPACK2)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(spec.loader.get_filename(TESTPACK2), mod.__file__)
        self.assertFalse(zi.is_package(TESTPACK2 + os.sep + '__init__'))
        self.assertFalse(zi.is_package(TESTPACK2 + os.sep + TESTMOD))
        pkg_path = TEMP_ZIP + os.sep + packdir + TESTPACK2
        zi2 = zipimport.zipimporter(pkg_path)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', DeprecationWarning)
            find_mod_dotted = zi2.find_module(TESTMOD)
            self.assertIsNotNone(find_mod_dotted)
            self.assertIsInstance(find_mod_dotted, zipimport.zipimporter)
            self.assertFalse(zi2.is_package(TESTMOD))
            load_mod = find_mod_dotted.load_module(TESTMOD)
            self.assertEqual(find_mod_dotted.get_filename(TESTMOD), load_mod.__file__)
        spec = zi2.find_spec(TESTMOD)
        self.assertIsNotNone(spec)
        self.assertIsInstance(spec.loader, zipimport.zipimporter)
        self.assertFalse(spec.loader.is_package(TESTMOD))
        load_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(load_mod)
        self.assertEqual(spec.loader.get_filename(TESTMOD), load_mod.__file__)
        mod_path = TESTPACK2 + os.sep + TESTMOD
        mod_name = module_path_to_dotted_name(mod_path)
        mod = importlib.import_module(mod_name)
        self.assertTrue(mod_name in sys.modules)
        self.assertIsNone(zi.get_source(TESTPACK2))
        self.assertIsNone(zi.get_source(mod_path))
        self.assertEqual(zi.get_filename(mod_path), mod.__file__)
        loader = mod.__loader__
        self.assertIsNone(loader.get_source(mod_name))
        self.assertEqual(loader.get_filename(mod_name), mod.__file__)

    def testGetData(self):
        if False:
            print('Hello World!')
        self.addCleanup(os_helper.unlink, TEMP_ZIP)
        with ZipFile(TEMP_ZIP, 'w') as z:
            z.compression = self.compression
            name = 'testdata.dat'
            data = bytes((x for x in range(256)))
            z.writestr(name, data)
        zi = zipimport.zipimporter(TEMP_ZIP)
        self.assertEqual(data, zi.get_data(name))
        self.assertIn('zipimporter object', repr(zi))

    def testImporterAttr(self):
        if False:
            return 10
        src = 'if 1:  # indent hack\n        def get_file():\n            return __file__\n        if __loader__.get_data("some.data") != b"some data":\n            raise AssertionError("bad data")\n'
        pyc = make_pyc(compile(src, '<???>', 'exec'), NOW, len(src))
        files = {TESTMOD + pyc_ext: (NOW, pyc), 'some.data': (NOW, 'some data')}
        self.doTest(pyc_ext, files, TESTMOD)

    def testDefaultOptimizationLevel(self):
        if False:
            return 10
        src = 'if 1:  # indent hack\n        def test(val):\n            assert(val)\n            return val\n'
        files = {TESTMOD + '.py': (NOW, src)}
        self.makeZip(files)
        sys.path.insert(0, TEMP_ZIP)
        mod = importlib.import_module(TESTMOD)
        self.assertEqual(mod.test(1), 1)
        self.assertRaises(AssertionError, mod.test, False)

    def testImport_WithStuff(self):
        if False:
            i = 10
            return i + 15
        files = {TESTMOD + '.py': (NOW, test_src)}
        self.doTest('.py', files, TESTMOD, stuff=b'Some Stuff' * 31)

    def assertModuleSource(self, module):
        if False:
            return 10
        self.assertEqual(inspect.getsource(module), test_src)

    def testGetSource(self):
        if False:
            while True:
                i = 10
        files = {TESTMOD + '.py': (NOW, test_src)}
        self.doTest('.py', files, TESTMOD, call=self.assertModuleSource)

    def testGetCompiledSource(self):
        if False:
            i = 10
            return i + 15
        pyc = make_pyc(compile(test_src, '<???>', 'exec'), NOW, len(test_src))
        files = {TESTMOD + '.py': (NOW, test_src), TESTMOD + pyc_ext: (NOW, pyc)}
        self.doTest(pyc_ext, files, TESTMOD, call=self.assertModuleSource)

    def runDoctest(self, callback):
        if False:
            return 10
        files = {TESTMOD + '.py': (NOW, test_src), 'xyz.txt': (NOW, '>>> log.append(True)\n')}
        self.doTest('.py', files, TESTMOD, call=callback)

    def doDoctestFile(self, module):
        if False:
            for i in range(10):
                print('nop')
        log = []
        (old_master, doctest.master) = (doctest.master, None)
        try:
            doctest.testfile('xyz.txt', package=module, module_relative=True, globs=locals())
        finally:
            doctest.master = old_master
        self.assertEqual(log, [True])

    def testDoctestFile(self):
        if False:
            while True:
                i = 10
        self.runDoctest(self.doDoctestFile)

    def doDoctestSuite(self, module):
        if False:
            i = 10
            return i + 15
        log = []
        doctest.DocFileTest('xyz.txt', package=module, module_relative=True, globs=locals()).run()
        self.assertEqual(log, [True])

    def testDoctestSuite(self):
        if False:
            for i in range(10):
                print('nop')
        self.runDoctest(self.doDoctestSuite)

    def doTraceback(self, module):
        if False:
            for i in range(10):
                print('nop')
        try:
            module.do_raise()
        except Exception as e:
            tb = e.__traceback__.tb_next
            (f, lno, n, line) = extract_tb(tb, 1)[0]
            self.assertEqual(line, raise_src.strip())
            (f, lno, n, line) = extract_stack(tb.tb_frame, 1)[0]
            self.assertEqual(line, raise_src.strip())
            s = io.StringIO()
            print_tb(tb, 1, s)
            self.assertTrue(s.getvalue().endswith(raise_src))
        else:
            raise AssertionError('This ought to be impossible')

    def testTraceback(self):
        if False:
            return 10
        files = {TESTMOD + '.py': (NOW, raise_src)}
        self.doTest(None, files, TESTMOD, call=self.doTraceback)

    @unittest.skipIf(os_helper.TESTFN_UNENCODABLE is None, 'need an unencodable filename')
    def testUnencodable(self):
        if False:
            while True:
                i = 10
        filename = os_helper.TESTFN_UNENCODABLE + '.zip'
        self.addCleanup(os_helper.unlink, filename)
        with ZipFile(filename, 'w') as z:
            zinfo = ZipInfo(TESTMOD + '.py', time.localtime(NOW))
            zinfo.compress_type = self.compression
            z.writestr(zinfo, test_src)
        spec = zipimport.zipimporter(filename).find_spec(TESTMOD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    def testBytesPath(self):
        if False:
            while True:
                i = 10
        filename = os_helper.TESTFN + '.zip'
        self.addCleanup(os_helper.unlink, filename)
        with ZipFile(filename, 'w') as z:
            zinfo = ZipInfo(TESTMOD + '.py', time.localtime(NOW))
            zinfo.compress_type = self.compression
            z.writestr(zinfo, test_src)
        zipimport.zipimporter(filename)
        zipimport.zipimporter(os.fsencode(filename))
        with self.assertRaises(TypeError):
            zipimport.zipimporter(bytearray(os.fsencode(filename)))
        with self.assertRaises(TypeError):
            zipimport.zipimporter(memoryview(os.fsencode(filename)))

    def testComment(self):
        if False:
            for i in range(10):
                print('nop')
        files = {TESTMOD + '.py': (NOW, test_src)}
        self.doTest('.py', files, TESTMOD, comment=b'comment')

    def testBeginningCruftAndComment(self):
        if False:
            i = 10
            return i + 15
        files = {TESTMOD + '.py': (NOW, test_src)}
        self.doTest('.py', files, TESTMOD, stuff=b'cruft' * 64, comment=b'hi')

    def testLargestPossibleComment(self):
        if False:
            for i in range(10):
                print('nop')
        files = {TESTMOD + '.py': (NOW, test_src)}
        self.doTest('.py', files, TESTMOD, comment=b'c' * ((1 << 16) - 1))

    def testZip64(self):
        if False:
            i = 10
            return i + 15
        files = {f'f{n}.py': (NOW, test_src) for n in range(65537)}
        self.doTest('.py', files, 'f6')

    def testZip64CruftAndComment(self):
        if False:
            i = 10
            return i + 15
        files = {f'f{n}.py': (NOW, test_src) for n in range(65537)}
        self.doTest('.py', files, 'f65536', comment=b'c' * ((1 << 16) - 1))

@support.requires_zlib()
class CompressedZipImportTestCase(UncompressedZipImportTestCase):
    compression = ZIP_DEFLATED

class BadFileZipImportTestCase(unittest.TestCase):

    def assertZipFailure(self, filename):
        if False:
            return 10
        self.assertRaises(zipimport.ZipImportError, zipimport.zipimporter, filename)

    def testNoFile(self):
        if False:
            return 10
        self.assertZipFailure('AdfjdkFJKDFJjdklfjs')

    def testEmptyFilename(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertZipFailure('')

    def testBadArgs(self):
        if False:
            return 10
        self.assertRaises(TypeError, zipimport.zipimporter, None)
        self.assertRaises(TypeError, zipimport.zipimporter, TESTMOD, kwd=None)
        self.assertRaises(TypeError, zipimport.zipimporter, list(os.fsencode(TESTMOD)))

    def testFilenameTooLong(self):
        if False:
            while True:
                i = 10
        self.assertZipFailure('A' * 33000)

    def testEmptyFile(self):
        if False:
            return 10
        os_helper.unlink(TESTMOD)
        os_helper.create_empty_file(TESTMOD)
        self.assertZipFailure(TESTMOD)

    def testFileUnreadable(self):
        if False:
            while True:
                i = 10
        os_helper.unlink(TESTMOD)
        fd = os.open(TESTMOD, os.O_CREAT, 0)
        try:
            os.close(fd)
            with self.assertRaises(zipimport.ZipImportError) as cm:
                zipimport.zipimporter(TESTMOD)
        finally:
            os.chmod(TESTMOD, 438)
            os_helper.unlink(TESTMOD)

    def testNotZipFile(self):
        if False:
            for i in range(10):
                print('nop')
        os_helper.unlink(TESTMOD)
        fp = open(TESTMOD, 'w+')
        fp.write('a' * 22)
        fp.close()
        self.assertZipFailure(TESTMOD)

    def _testBogusZipFile(self):
        if False:
            i = 10
            return i + 15
        os_helper.unlink(TESTMOD)
        fp = open(TESTMOD, 'w+')
        fp.write(struct.pack('=I', 101010256))
        fp.write('a' * 18)
        fp.close()
        z = zipimport.zipimporter(TESTMOD)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                self.assertRaises(TypeError, z.load_module, None)
            self.assertRaises(TypeError, z.find_module, None)
            self.assertRaises(TypeError, z.find_spec, None)
            self.assertRaises(TypeError, z.exec_module, None)
            self.assertRaises(TypeError, z.is_package, None)
            self.assertRaises(TypeError, z.get_code, None)
            self.assertRaises(TypeError, z.get_data, None)
            self.assertRaises(TypeError, z.get_source, None)
            error = zipimport.ZipImportError
            self.assertIsNone(z.find_module('abc'))
            self.assertIsNone(z.find_spec('abc'))
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                self.assertRaises(error, z.load_module, 'abc')
            self.assertRaises(error, z.get_code, 'abc')
            self.assertRaises(OSError, z.get_data, 'abc')
            self.assertRaises(error, z.get_source, 'abc')
            self.assertRaises(error, z.is_package, 'abc')
        finally:
            zipimport._zip_directory_cache.clear()

def tearDownModule():
    if False:
        return 10
    os_helper.unlink(TESTMOD)
if __name__ == '__main__':
    unittest.main()