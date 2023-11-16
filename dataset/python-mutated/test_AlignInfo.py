"""Bio.Align.AlignInfo related tests."""
import unittest
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import AlignIO
from Bio.Align.AlignInfo import SummaryInfo
from Bio.Data import IUPACData
import math

class AlignInfoTests(unittest.TestCase):
    """Test basic usage."""

    def assertAlmostEqualList(self, list1, list2, **kwargs):
        if False:
            while True:
                i = 10
        self.assertEqual(len(list1), len(list2))
        for (v1, v2) in zip(list1, list2):
            self.assertAlmostEqual(v1, v2, **kwargs)

    def test_nucleotides(self):
        if False:
            while True:
                i = 10
        filename = 'GFF/multi.fna'
        fmt = 'fasta'
        alignment = AlignIO.read(filename, fmt)
        summary = SummaryInfo(alignment)
        c = summary.dumb_consensus(ambiguous='N')
        self.assertEqual(c, 'NNNNNNNN')
        c = summary.gap_consensus(ambiguous='N')
        self.assertEqual(c, 'NNNNNNNN')
        expected = {'A': 0.25, 'G': 0.25, 'T': 0.25, 'C': 0.25}
        m = summary.pos_specific_score_matrix(chars_to_ignore=['-'], axis_seq=c)
        self.assertEqual(str(m), '    A   C   G   T\nN  2.0 0.0 1.0 0.0\nN  1.0 1.0 1.0 0.0\nN  1.0 0.0 2.0 0.0\nN  0.0 1.0 1.0 1.0\nN  1.0 2.0 0.0 0.0\nN  0.0 2.0 1.0 0.0\nN  1.0 2.0 0.0 0.0\nN  0.0 2.0 1.0 0.0\n')
        ic = summary.information_content(e_freq_table=expected, chars_to_ignore=['-'])
        self.assertAlmostEqual(ic, 7.32029999423075, places=6)

    def test_proteins(self):
        if False:
            print('Hello World!')
        a = MultipleSeqAlignment([SeqRecord(Seq('MHQAIFIYQIGYP*LKSGYIQSIRSPEYDNW-'), id='ID001'), SeqRecord(Seq('MH--IFIYQIGYAYLKSGYIQSIRSPEY-NW*'), id='ID002'), SeqRecord(Seq('MHQAIFIYQIGYPYLKSGYIQSIRSPEYDNW*'), id='ID003')])
        self.assertEqual(32, a.get_alignment_length())
        s = SummaryInfo(a)
        c = s.dumb_consensus(ambiguous='X')
        self.assertEqual(c, 'MHQAIFIYQIGYXXLKSGYIQSIRSPEYDNW*')
        c = s.gap_consensus(ambiguous='X')
        self.assertEqual(c, 'MHXXIFIYQIGYXXLKSGYIQSIRSPEYXNWX')
        m = s.pos_specific_score_matrix(chars_to_ignore=['-', '*'], axis_seq=c)
        self.assertEqual(str(m), '    A   D   E   F   G   H   I   K   L   M   N   P   Q   R   S   W   Y\nM  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nH  0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nX  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0\nX  2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nI  0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nF  0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nI  0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nY  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0\nQ  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0\nI  0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nG  0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nY  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0\nX  1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0 0.0 0.0 0.0 0.0 0.0\nX  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 2.0\nL  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nK  0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nS  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0\nG  0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nY  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0\nI  0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nQ  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0\nS  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0\nI  0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nR  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0\nS  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0\nP  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0\nE  0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nY  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0\nX  0.0 2.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\nN  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0 0.0 0.0 0.0 0.0 0.0\nW  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 3.0 0.0\nX  0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0\n')
        letters = IUPACData.protein_letters
        base_freq = 1.0 / len(letters)
        e_freq_table = {letter: base_freq for letter in letters}
        ic = s.information_content(e_freq_table=e_freq_table, chars_to_ignore=['-', '*'])
        self.assertAlmostEqual(ic, 133.061475107, places=6)

    def test_pseudo_count(self):
        if False:
            while True:
                i = 10
        dna_align = MultipleSeqAlignment([SeqRecord(Seq('AACCACGTTTAA'), id='ID001'), SeqRecord(Seq('CACCACGTGGGT'), id='ID002'), SeqRecord(Seq('CACCACGTTCGC'), id='ID003'), SeqRecord(Seq('GCGCACGTGGGG'), id='ID004'), SeqRecord(Seq('TCGCACGTTGTG'), id='ID005'), SeqRecord(Seq('TGGCACGTGTTT'), id='ID006'), SeqRecord(Seq('TGACACGTGGGA'), id='ID007'), SeqRecord(Seq('TTACACGTGCGC'), id='ID008')])
        summary = SummaryInfo(dna_align)
        expected = {'A': 0.325, 'G': 0.175, 'T': 0.325, 'C': 0.175}
        ic = summary.information_content(e_freq_table=expected, log_base=math.exp(1), pseudo_count=1)
        self.assertAlmostEqualList(summary.ic_vector, [0.11, 0.09, 0.36, 1.29, 0.8, 1.29, 1.29, 0.8, 0.61, 0.39, 0.47, 0.04], places=2)
        self.assertAlmostEqual(ic, 7.546, places=3)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)