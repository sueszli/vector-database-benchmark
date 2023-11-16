"""Unit tests for the Bio.PDB.NACCESS submodule."""
import subprocess
import unittest
try:
    import numpy
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.PDB.') from None
from Bio.PDB import PDBParser
from Bio.PDB.NACCESS import NACCESS, process_asa_data, process_rsa_data

class NACCESS_test(unittest.TestCase):
    """Tests for Bio.PDB.NACCESS and output parsing."""

    def test_NACCESS_rsa_file(self):
        if False:
            for i in range(10):
                print('nop')
        'Test parsing of pregenerated rsa NACCESS file.'
        with open('PDB/1A8O.rsa') as rsa:
            naccess = process_rsa_data(rsa)
        self.assertEqual(len(naccess), 66)

    def test_NACCESS_asa_file(self):
        if False:
            return 10
        'Test parsing of pregenerated asa NACCESS file.'
        with open('PDB/1A8O.asa') as asa:
            naccess = process_asa_data(asa)
        self.assertEqual(len(naccess), 524)

    def test_NACCESS(self):
        if False:
            return 10
        'Test calling NACCESS from Bio.PDB.'
        try:
            subprocess.check_call(['naccess', '-q'], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        except OSError:
            raise self.skipTest('Install naccess if you want to use it from Biopython.')
        p = PDBParser()
        pdbfile = 'PDB/1A8O.pdb'
        model = p.get_structure('1A8O', pdbfile)[0]
        naccess = NACCESS(model, pdbfile)
        self.assertEqual(len(naccess), 66)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)