"""
Tests that deal with setuptools namespace
packages, and in particular the installation
flavour used by pip
"""
import os
import sys
import subprocess
import unittest
import textwrap
import shutil
import pytest
from PyInstaller.lib.modulegraph import modulegraph
gRootDir = os.path.dirname(os.path.abspath(__file__))
gSrcDir = os.path.join(gRootDir, 'testpkg-setuptools-namespace')

@pytest.fixture
def install_testpkg(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    dest_dir = str(tmpdir / 'data')
    shutil.copytree(gSrcDir, dest_dir)
    libdir = str(tmpdir / 'test')
    subprocess.check_call([sys.executable, 'setup.py', 'install', '--install-lib', libdir, '--single-version-externally-managed', '--record', os.path.join(libdir, 'record.lst')], cwd=dest_dir)
    return libdir

class TestPythonBehaviour(object):

    def importModule(self, name, libdir):
        if False:
            while True:
                i = 10
        if '.' in name:
            script = textwrap.dedent('                import site\n                site.addsitedir(%r)\n                try:\n                    import %s\n                except ImportError:\n                    import %s\n                print (%s.__name__)\n            ') % (str(libdir), name, name.rsplit('.', 1)[0], name)
        else:
            script = textwrap.dedent('                import site\n                site.addsitedir(%r)\n                import %s\n                print (%s.__name__)\n            ') % (str(libdir), name, name)
        data = subprocess.check_output([sys.executable, '-c', script], stderr=subprocess.STDOUT, cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'testpkg-relimport'))
        if sys.version_info[0] != 2:
            data = data.decode('UTF-8')
        data = data.strip()
        if data.endswith(' refs]'):
            data = data.rsplit('\n', 1)[0].strip()
        return data

    def testToplevel(self, install_testpkg):
        if False:
            while True:
                i = 10
        m = self.importModule('nspkg.module', install_testpkg)
        assert m == 'nspkg.module'

    def testSub(self, install_testpkg):
        if False:
            return 10
        m = self.importModule('nspkg.nssubpkg.sub', install_testpkg)
        assert m == 'nspkg.nssubpkg.sub'

@pytest.fixture
def install_testpkg_modulegraph(install_testpkg):
    if False:
        print('Hello World!')
    return modulegraph.ModuleGraph(path=[str(install_testpkg)] + sys.path)

class TestModuleGraphImport(object):

    def testRootPkg(self, install_testpkg_modulegraph):
        if False:
            while True:
                i = 10
        install_testpkg_modulegraph.import_hook('nspkg')
        node = install_testpkg_modulegraph.find_node('nspkg')
        assert isinstance(node, modulegraph.NamespacePackage)
        assert node.identifier == 'nspkg'
        assert node.filename == '-'

    def testRootPkgModule(self, install_testpkg_modulegraph):
        if False:
            i = 10
            return i + 15
        install_testpkg_modulegraph.import_hook('nspkg.module')
        node = install_testpkg_modulegraph.find_node('nspkg.module')
        assert isinstance(node, modulegraph.SourceModule)
        assert node.identifier == 'nspkg.module'

    def testSubRootPkgModule(self, install_testpkg_modulegraph):
        if False:
            i = 10
            return i + 15
        install_testpkg_modulegraph.import_hook('nspkg.nssubpkg.sub')
        node = install_testpkg_modulegraph.find_node('nspkg.nssubpkg.sub')
        assert isinstance(node, modulegraph.SourceModule)
        assert node.identifier == 'nspkg.nssubpkg.sub'
        node = install_testpkg_modulegraph.find_node('nspkg')
        assert isinstance(node, modulegraph.NamespacePackage)
if __name__ == '__main__':
    unittest.main()