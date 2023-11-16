import sys
if sys.version_info[:2] <= (2, 6):
    import unittest2 as unittest
else:
    import unittest
import textwrap
import subprocess
import os
from PyInstaller.lib.modulegraph import modulegraph

class TestNativeImport(unittest.TestCase):

    def importModule(self, name):
        if False:
            while True:
                i = 10
        if '.' in name:
            script = textwrap.dedent('                try:\n                    import %s\n                except ImportError:\n                    import %s\n                print (%s.__name__)\n            ') % (name, name.rsplit('.', 1)[0], name)
        else:
            script = textwrap.dedent('                import %s\n                print (%s.__name__)\n            ') % (name, name)
        p = subprocess.Popen([sys.executable, '-c', script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testpkg-import-from-init'))
        data = p.communicate()[0]
        if sys.version_info[0] != 2:
            data = data.decode('UTF-8')
        data = data.strip()
        if data.endswith(' refs]'):
            data = data.rsplit('\n', 1)[0].strip()
        sts = p.wait()
        if sts != 0:
            print(data)
        self.assertEqual(sts, 0)
        return data

    @unittest.skipUnless(sys.version_info[0] == 2, 'Python 2.x test')
    def testRootPkg(self):
        if False:
            return 10
        m = self.importModule('pkg')
        self.assertEqual(m, 'pkg')

    @unittest.skipUnless(sys.version_info[0] == 2, 'Python 2.x test')
    def testSubPackage(self):
        if False:
            for i in range(10):
                print('nop')
        m = self.importModule('pkg.subpkg')
        self.assertEqual(m, 'pkg.subpkg')

    def testRootPkgRelImport(self):
        if False:
            return 10
        m = self.importModule('pkg2')
        self.assertEqual(m, 'pkg2')

    def testSubPackageRelImport(self):
        if False:
            i = 10
            return i + 15
        m = self.importModule('pkg2.subpkg')
        self.assertEqual(m, 'pkg2.subpkg')

class TestModuleGraphImport(unittest.TestCase):
    if not hasattr(unittest.TestCase, 'assertIsInstance'):

        def assertIsInstance(self, value, types):
            if False:
                i = 10
                return i + 15
            if not isinstance(value, types):
                self.fail('%r is not an instance of %r' % (value, types))

    def setUp(self):
        if False:
            while True:
                i = 10
        root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testpkg-import-from-init')
        self.mf = modulegraph.ModuleGraph(path=[root] + sys.path)
        self.mf.add_script(os.path.join(root, 'script.py'))

    @unittest.skipUnless(sys.version_info[0] == 2, 'Python 2.x test')
    def testRootPkg(self):
        if False:
            i = 10
            return i + 15
        node = self.mf.find_node('pkg')
        self.assertIsInstance(node, modulegraph.Package)
        self.assertEqual(node.identifier, 'pkg')

    @unittest.skipUnless(sys.version_info[0] == 2, 'Python 2.x test')
    def testSubPackage(self):
        if False:
            while True:
                i = 10
        node = self.mf.find_node('pkg.subpkg')
        self.assertIsInstance(node, modulegraph.Package)
        self.assertEqual(node.identifier, 'pkg.subpkg')
        node = self.mf.find_node('pkg.subpkg.compat')
        self.assertIsInstance(node, modulegraph.SourceModule)
        self.assertEqual(node.identifier, 'pkg.subpkg.compat')
        node = self.mf.find_node('pkg.subpkg._collections')
        self.assertIsInstance(node, modulegraph.SourceModule)
        self.assertEqual(node.identifier, 'pkg.subpkg._collections')

    def testRootPkgRelImport(self):
        if False:
            while True:
                i = 10
        node = self.mf.find_node('pkg2')
        self.assertIsInstance(node, modulegraph.Package)
        self.assertEqual(node.identifier, 'pkg2')

    def testSubPackageRelImport(self):
        if False:
            print('Hello World!')
        node = self.mf.find_node('pkg2.subpkg')
        self.assertIsInstance(node, modulegraph.Package)
        self.assertEqual(node.identifier, 'pkg2.subpkg')
        node = self.mf.find_node('pkg2.subpkg.compat')
        self.assertIsInstance(node, modulegraph.SourceModule)
        self.assertEqual(node.identifier, 'pkg2.subpkg.compat')
        node = self.mf.find_node('pkg2.subpkg._collections')
        self.assertIsInstance(node, modulegraph.SourceModule)
        self.assertEqual(node.identifier, 'pkg2.subpkg._collections')
if __name__ == '__main__':
    unittest.main()