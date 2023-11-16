"""Unit tests for the Bio.PDB.StructureAlignment module."""
import unittest
from Bio.PDB import StructureAlignment
from Bio.PDB import PDBParser
from Bio import Align
from Bio import AlignIO

class StructureAlignTests(unittest.TestCase):
    """Test module StructureAlignment."""

    def test_StructAlign(self):
        if False:
            i = 10
            return i + 15
        'Tests on module to align two proteins according to a FASTA alignment file.'
        p = PDBParser(QUIET=1)
        al_file = 'PDB/alignment_file.fa'
        with open(al_file) as handle:
            records = AlignIO.read(handle, 'fasta')
        with open(al_file) as handle:
            alignment = Align.read(handle, 'fasta')
        s1 = p.get_structure('1', 'PDB/2XHE.pdb')
        s2 = p.get_structure('2', 'PDB/1A8O.pdb')
        m1 = s1[0]
        m2 = s2[0]
        for argument in (records, alignment):
            al = StructureAlignment(argument, m1, m2)
            self.assertNotEqual(al.map12, al.map21)
            self.assertTrue(len(al.map12), 566)
            self.assertTrue(len(al.map21), 70)
            chain1_A = m1['A']
            chain2_A = m2['A']
            self.assertEqual(chain1_A[202].get_resname(), 'ILE')
            self.assertEqual(chain2_A[202].get_resname(), 'LEU')
            self.assertEqual(chain1_A[291].get_resname(), chain2_A[180].get_resname())
            self.assertNotEqual(chain1_A[291].get_resname(), chain2_A[181].get_resname())
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)