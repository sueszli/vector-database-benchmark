"""Tests for mmtf online module."""
import unittest
import warnings
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from Bio.PDB.mmtf import MMTFParser
import requires_internet
requires_internet.check()

class OnlineMMTF(unittest.TestCase):
    """Online tests for the MMTF code."""

    def test_from_url(self):
        if False:
            while True:
                i = 10
        'Check parser can fetch a record from its PDB ID.'
        parser = MMTFParser()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', PDBConstructionWarning)
            struct = parser.get_structure_from_url('4ZHL')
        atoms = list(struct.get_atoms())
        self.assertEqual(len(atoms), 2080)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)