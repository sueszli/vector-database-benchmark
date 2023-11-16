"""Tests for PDB FragmentMapper module."""
import unittest
try:
    import numpy
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install NumPy if you want to use Bio.PDB.') from None
from Bio.PDB import PDBParser
from Bio.PDB import FragmentMapper
from Bio.PDB import Selection

class FragmentMapperTests(unittest.TestCase):
    """Tests for FragmentMapper module."""

    def test_fragment_mapper(self):
        if False:
            i = 10
            return i + 15
        'Self test for FragmentMapper module.'
        p = PDBParser()
        pdb1 = 'PDB/1A8O.pdb'
        s = p.get_structure('X', pdb1)
        m = s[0]
        fm = FragmentMapper(m, 10, 5, 'PDB')
        for r in Selection.unfold_entities(m, 'R'):
            if r in fm:
                self.assertTrue(str(fm[r]).startswith('<Fragment length=5 id='))
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)