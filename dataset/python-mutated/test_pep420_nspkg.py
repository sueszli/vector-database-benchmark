"""
Tests that deal with pep420 namespace packages.

PEP 420 is new in Python 3.3
"""
import os
import shutil
import sys
import subprocess
import textwrap
if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest
from PyInstaller.lib.modulegraph import modulegraph
gRootDir = os.path.dirname(os.path.abspath(__file__))
gSrcDir = os.path.join(gRootDir, 'testpkg-pep420-namespace')
if sys.version_info[:2] >= (3, 3):

    class TestPythonBehaviour(unittest.TestCase):

        def importModule(self, name):
            if False:
                while True:
                    i = 10
            test_dir1 = os.path.join(gSrcDir, 'path1')
            test_dir2 = os.path.join(gSrcDir, 'path2')
            if '.' in name:
                script = textwrap.dedent('                    import site\n                    site.addsitedir(%r)\n                    site.addsitedir(%r)\n                    try:\n                        import %s\n                    except ImportError:\n                        import %s\n                    print (%s.__name__)\n                ') % (test_dir1, test_dir2, name, name.rsplit('.', 1)[0], name)
            else:
                script = textwrap.dedent('                    import site\n                    site.addsitedir(%r)\n                    site.addsitedir(%r)\n                    import %s\n                    print (%s.__name__)\n                ') % (test_dir1, test_dir2, name, name)
            p = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testpkg-relimport'))
            data = p.communicate()[0]
            if sys.version_info[0] != 2:
                data = data.decode('UTF-8')
            data = data.strip()
            if data.endswith(' refs]'):
                data = data.rsplit('\n', 1)[0].strip()
            sts = p.wait()
            if sts != 0:
                print(data)
                self.fail('import of %r failed' % (name,))
            return data

        def testToplevel(self):
            if False:
                i = 10
                return i + 15
            m = self.importModule('package.sub1')
            self.assertEqual(m, 'package.sub1')
            m = self.importModule('package.sub2')
            self.assertEqual(m, 'package.sub2')

        def testSub(self):
            if False:
                return 10
            m = self.importModule('package.subpackage.sub')
            self.assertEqual(m, 'package.subpackage.sub')
            m = self.importModule('package.nspkg.mod')
            self.assertEqual(m, 'package.nspkg.mod')

    class TestModuleGraphImport(unittest.TestCase):
        if not hasattr(unittest.TestCase, 'assertIsInstance'):

            def assertIsInstance(self, value, types):
                if False:
                    i = 10
                    return i + 15
                if not isinstance(value, types):
                    self.fail('%r is not an instance of %r', value, types)

        def setUp(self):
            if False:
                return 10
            self.mf = modulegraph.ModuleGraph(path=[os.path.join(gSrcDir, 'path1'), os.path.join(gSrcDir, 'path2')] + sys.path)

        def testRootPkg(self):
            if False:
                while True:
                    i = 10
            self.mf.import_hook('package')
            node = self.mf.find_node('package')
            self.assertIsInstance(node, modulegraph.NamespacePackage)
            self.assertEqual(node.identifier, 'package')
            self.assertEqual(node.filename, '-')

        def testRootPkgModule(self):
            if False:
                i = 10
                return i + 15
            self.mf.import_hook('package.sub1')
            node = self.mf.find_node('package.sub1')
            self.assertIsInstance(node, modulegraph.SourceModule)
            self.assertEqual(node.identifier, 'package.sub1')
            self.mf.import_hook('package.sub2')
            node = self.mf.find_node('package.sub2')
            self.assertIsInstance(node, modulegraph.SourceModule)
            self.assertEqual(node.identifier, 'package.sub2')

        def testSubRootPkgModule(self):
            if False:
                while True:
                    i = 10
            self.mf.import_hook('package.subpackage.sub')
            node = self.mf.find_node('package.subpackage.sub')
            self.assertIsInstance(node, modulegraph.SourceModule)
            self.assertEqual(node.identifier, 'package.subpackage.sub')
            node = self.mf.find_node('package')
            self.assertIsInstance(node, modulegraph.NamespacePackage)
            self.mf.import_hook('package.nspkg.mod')
            node = self.mf.find_node('package.nspkg.mod')
            self.assertIsInstance(node, modulegraph.SourceModule)
            self.assertEqual(node.identifier, 'package.nspkg.mod')
else:

    class TestPythonBehaviour(unittest.TestCase):

        def importModule(self, name):
            if False:
                print('Hello World!')
            test_dir1 = os.path.join(gSrcDir, 'path1')
            test_dir2 = os.path.join(gSrcDir, 'path2')
            if '.' in name:
                script = textwrap.dedent('                    import site\n                    site.addsitedir(%r)\n                    site.addsitedir(%r)\n                    try:\n                        import %s\n                    except ImportError:\n                        import %s\n                    print (%s.__name__)\n                ') % (test_dir1, test_dir2, name, name.rsplit('.', 1)[0], name)
            else:
                script = textwrap.dedent('                    import site\n                    site.addsitedir(%r)\n                    site.addsitedir(%r)\n                    import %s\n                    print (%s.__name__)\n                ') % (test_dir1, test_dir2, name, name)
            p = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testpkg-relimport'))
            data = p.communicate()[0]
            if sys.version_info[0] != 2:
                data = data.decode('UTF-8')
            data = data.strip()
            if data.endswith(' refs]'):
                data = data.rsplit('\n', 1)[0].strip()
            sts = p.wait()
            if sts != 0:
                raise ImportError(name)
            return data

        def testToplevel(self):
            if False:
                while True:
                    i = 10
            m = self.importModule('sys')
            self.assertEqual(m, 'sys')
            self.assertRaises(ImportError, self.importModule, 'package.sub1')
            self.assertRaises(ImportError, self.importModule, 'package.sub2')

        def testSub(self):
            if False:
                return 10
            self.assertRaises(ImportError, self.importModule, 'package.subpackage.sub')

    class TestModuleGraphImport(unittest.TestCase):
        if not hasattr(unittest.TestCase, 'assertIsInstance'):

            def assertIsInstance(self, value, types):
                if False:
                    print('Hello World!')
                if not isinstance(value, types):
                    self.fail('%r is not an instance of %r', value, types)

        def setUp(self):
            if False:
                print('Hello World!')
            self.mf = modulegraph.ModuleGraph(path=[os.path.join(gSrcDir, 'path1'), os.path.join(gSrcDir, 'path2')] + sys.path)

        def testRootPkg(self):
            if False:
                print('Hello World!')
            self.assertRaises(ImportError, self.mf.import_hook, 'package')
            node = self.mf.find_node('package')
            self.assertIs(node, None)

        def testRootPkgModule(self):
            if False:
                return 10
            self.assertRaises(ImportError, self.mf.import_hook, 'package.sub1')
            node = self.mf.find_node('package.sub1')
            self.assertIs(node, None)
            node = self.mf.find_node('package.sub2')
            self.assertIs(node, None)
if __name__ == '__main__':
    unittest.main()