"""Tests for PAML tools module."""
import unittest
import os
import sys
from Bio.Phylo.PAML import codeml, baseml, yn00
from Bio import MissingExternalDependencyError

def is_exe(filepath):
    if False:
        return 10
    'Test if a file is an executable.'
    return os.path.exists(filepath) and os.access(filepath, os.X_OK)

def which(program):
    if False:
        return 10
    'Find the path to an executable.'
    (filepath, filename) = os.path.split(program)
    os_path = os.environ['PATH'].split(os.pathsep)
    if sys.platform == 'win32':
        try:
            prog_files = os.environ['PROGRAMFILES']
        except KeyError:
            prog_files = 'C:\\Program Files'
        likely_dirs = ['', prog_files, os.path.join(prog_files, 'paml41'), os.path.join(prog_files, 'paml43'), os.path.join(prog_files, 'paml44'), os.path.join(prog_files, 'paml45')] + sys.path
        os_path.extend(likely_dirs)
    for path in os.environ['PATH'].split(os.pathsep):
        exe_file = os.path.join(path, program)
        if is_exe(exe_file):
            return exe_file
    return None
if sys.platform == 'win32':
    binaries = ['codeml.exe', 'baseml.exe', 'yn00.exe']
else:
    binaries = ['codeml', 'baseml', 'yn00']
for binary in binaries:
    if which(binary) is None:
        raise MissingExternalDependencyError('Install PAML if you want to use the Bio.Phylo.PAML wrapper.')

class Common(unittest.TestCase):
    """Base class for PAML unit tests."""
    del_files = []

    def __del__(self):
        if False:
            print('Hello World!')
        'Just in case tool creates some junk files, do a clean-up.'
        del_files = self.del_files
        for filename in del_files:
            if os.path.exists(filename):
                os.remove(filename)

class CodemlTest(Common):
    """Tests for PAML tool codeml."""

    def setUp(self):
        if False:
            return 10
        self.cml = codeml.Codeml()

    def testCodemlBinary(self):
        if False:
            print('Hello World!')
        'Check codeml runs, generates correct output, and is the correct version.'
        ctl_file = os.path.join('PAML', 'Control_files', 'codeml', 'codeml.ctl')
        self.cml.read_ctl_file(ctl_file)
        self.cml.alignment = os.path.join('PAML', 'Alignments', 'alignment.phylip')
        self.cml.tree = os.path.join('PAML', 'Trees', 'species.tree')
        self.cml.out_file = os.path.join('PAML', 'temp.out')
        self.cml.working_dir = os.path.join('PAML', 'codeml_test')
        results = self.cml.run()
        self.assertGreater(results['version'], '4.0')
        self.assertIn('NSsites', results)
        self.assertEqual(len(results['NSsites']), 1)
        self.assertEqual(len(results['NSsites'][0]), 5)

class BasemlTest(Common):
    """Tests for PAML tool baseml."""

    def setUp(self):
        if False:
            print('Hello World!')
        self.bml = baseml.Baseml()

    def testBasemlBinary(self):
        if False:
            print('Hello World!')
        'Check baseml runs, generates correct output, and is the correct version.'
        ctl_file = os.path.join('PAML', 'Control_files', 'baseml', 'baseml.ctl')
        self.bml.read_ctl_file(ctl_file)
        self.bml.alignment = os.path.join('PAML', 'Alignments', 'alignment.phylip')
        self.bml.tree = os.path.join('PAML', 'Trees', 'species.tree')
        self.bml.out_file = os.path.join('PAML', 'temp.out')
        self.bml.working_dir = os.path.join('PAML', 'baseml_test')
        results = self.bml.run()
        self.assertGreater(results['version'], '4.0')
        self.assertIn('parameters', results)
        self.assertEqual(len(results['parameters']), 5)

class Yn00Test(Common):
    """Tests for PAML tool yn00."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.yn = yn00.Yn00()

    def testYn00Binary(self):
        if False:
            print('Hello World!')
        'Check yn00 binary runs and generates correct output.\n\n        yn00 output does not specify the version number.\n        '
        ctl_file = os.path.join('PAML', 'Control_files', 'yn00', 'yn00.ctl')
        self.yn.read_ctl_file(ctl_file)
        self.yn.alignment = os.path.join('PAML', 'Alignments', 'alignment.phylip')
        self.yn.out_file = os.path.join('PAML', 'temp.out')
        self.yn.working_dir = os.path.join('PAML', 'yn00_test')
        results = self.yn.run()
        self.assertEqual(len(results), 5)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)