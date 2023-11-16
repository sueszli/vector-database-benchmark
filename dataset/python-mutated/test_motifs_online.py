"""Testing online code for Bio.motifs (weblogo etc)."""
import os
import unittest
from Bio import motifs
from Bio.Seq import Seq
import requires_internet
requires_internet.check()

class TestMotifWeblogo(unittest.TestCase):
    """Tests Bio.motifs online code."""

    def check(self, seqs_as_strs, alpha):
        if False:
            for i in range(10):
                print('nop')
        m = motifs.create([Seq(s) for s in seqs_as_strs], alpha)
        m.weblogo(os.devnull)
        m = motifs.create(seqs_as_strs, alpha)
        m.weblogo(os.devnull)

    def test_dna(self):
        if False:
            while True:
                i = 10
        'Test Bio.motifs.weblogo with a DNA sequence.'
        self.check(['TACAA', 'TACGC', 'TACAC', 'TACCC', 'AACCC', 'AATGC', 'AATGC'], 'GATCBDSW')

    def test_rna(self):
        if False:
            return 10
        'Test Bio.motifs.weblogo with an RNA sequence.'
        self.check(['UACAA', 'UACGC', 'UACAC', 'UACCC', 'AACCC', 'AAUGC', 'AAUGC'], 'GAUC')

    def test_protein(self):
        if False:
            i = 10
            return i + 15
        'Test Bio.motifs.weblogo with a protein sequence.'
        self.check(['ACDEG', 'AYCRN', 'HYLID', 'AYHEL', 'ACDEH', 'AYYRN', 'HYIID'], 'ACDEFGHIKLMNPQRSTVWYBXZJUO')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)