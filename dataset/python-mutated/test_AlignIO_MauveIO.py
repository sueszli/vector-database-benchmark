"""Tests for Bio.AlignIO.MauveIO module."""
import os
import unittest
from io import StringIO
from Bio import SeqIO
from Bio.AlignIO.MauveIO import MauveIterator
from Bio.AlignIO.MauveIO import MauveWriter

class TestMauveIO(unittest.TestCase):
    MAUVE_TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'Mauve')
    SIMPLE_XMFA = os.path.join(MAUVE_TEST_DATA_DIR, 'simple.xmfa')
    SIMPLE_FA = os.path.join(MAUVE_TEST_DATA_DIR, 'simple.fa')

    def test_one(self):
        if False:
            return 10
        ids = []
        with open(self.SIMPLE_XMFA) as handle:
            for alignment in MauveIterator(handle):
                for record in alignment:
                    ids.append(record.id)
        self.assertEqual(ids, ['1/0-5670', '2/0-5670', '1/5670-9940', '2/7140-11410', '1/9940-14910', '2/5670-7140', '2/11410-12880'])
        expected = 'ATTCGCACAT AAGAATGTAC CTTGCTGTAA TTTATACTCA\n            GCAGGTGGTG CAGACATCAT AACAAAAGAA GACTCTTGTT GTACTAGATA TTGTGTAGCA\n            TCACGACCAC ACACACATGG AATGGAAACA CCTGTCTTAA GATTATCATA AGATAGAGTA\n            CCCATATACA TCACAGCTTC TACACCCGTT AAGGTAGTAG TTTTCTGACC ACAATGTTTA\n            CACACCACAT TAAGAACTCG CTTTGCAGAT TCCAAATTAG CATGCTGTAG AAGATGGGTC\n            ATAGTTTCTC TGACATCACC AAGCTCGCCA ACAGTTTTAT TACTGTAAGC GAGTATGAGT\n            GCACAAAAGT TAGCAGCATC ACCAGCACGG GCTCTATAAT AAGCCTCTTG AAGTGCTGGT\n            GCATTGAATT TGACTTCAAG CTGTTGAAGT GCTAATAAAA CACTAGACAA ATAACAATTG\n            TTATCAGCCC ATTTAATTGA AGTTAAACCA CCAACTTGAG GAAATTTCCA TTTCTTTGTG\n            TGGTTTAAAG CAGACATGTA CCTACCAAGA AAACTCTCAT CAAGAGTATG GTAGTACTCG\n            AAAGCTTCAC TACGTAGTGT GTCATCACTA GGTAGTACAA AGAAAGTCTT ACCCTCATGA\n            TTTACATGAG GTTTAATTTT TGTAACATCA GCACCATCCA AGTATGTTGG ACCAAACTGC\n            TGTCCATATG TCATAGACAT ATCCACAAGC TGTGTGTGGA GATTAGTGTT GTCCACAGTT\n            GTGAACACTT TTATAGTCTT AACCTCCCGC AGGGATAAGA GACTCTTTAG TTTGTCAAGT\n            GAAAGAACCT CACCGTCAAG ATGAAACTCG ACGGGGCTCT CCAGAGTGTG GTACACAATT\n            TTGTCACCAC GCTTAAGAAA TTCAACACCT AACTCTGTAC GCTGTCCTGA ATAGGACCAA\n            TCTCTGTAAG AGCCAGCCAA AGAAACTGTT TCTACAAAGT GCTCCTCAGA TGTCTTTGAT\n            GACGAAGTGA GGTATCCATT ATATGTAGTA ACAGCATCTG GTGATGATAC TGACACTACG\n            GCAGGAGCTT TAAGAGAACG CATACAGCGC GCAGCCTCTT CAAGATTAAA ACCATGTGTC\n            ACATAACCAA TTGGCATTGT GACAAGCGGC TCATTTAGAG AGTTCAGCTT CGTAATAATA\n            GAAGCTACAG GCTCTTTACT AGTATAAAAG AAGAATCGGA CACCATAGTC AACGATGCCC\n            TCTTGAATTT TAATTCCTTT ATACTTACGT TGGATGGTTG CCATTATGGC TCTAACATCC\n            ATGCATATAG GCATTAATTT TCTTGTCTCT TCAGCATGAG CAAGCATTTC TCTCAAATTC\n            CAGGATACAG TTCCTAGAAT CTCTTCCTTA GCATTAGGTG CTTCTGAAGG TAGTACATAA\n            AATGCAGATT TGCATTTCTT AAGAGCAGTC TTAGCTTCCT CAAGTGTATA '
        self.assertEqual(str(record.seq).replace('-', ''), expected.replace(' ', '').replace('\n', ''))

    def test_sequence_positions(self):
        if False:
            while True:
                i = 10
        with open(self.SIMPLE_FA) as handle:
            seqs = list(SeqIO.parse(handle, 'fasta'))
        with open(self.SIMPLE_XMFA) as handle:
            aln_list = list(MauveIterator(handle))
        for aln in aln_list:
            for record in aln:
                if not record.seq.startswith('-'):
                    expected = record.seq[0:10]
                    actual = seqs[int(record.name) - 1].seq
                    actual = actual[record.annotations['start']:record.annotations['end']]
                    if record.annotations['strand'] < 0:
                        actual = actual.reverse_complement()
                    actual = actual[0:10]
                    if len(actual) == 0:
                        continue
                    self.assertEqual(expected, actual)

    def test_write_read(self):
        if False:
            i = 10
            return i + 15
        with open(self.SIMPLE_XMFA) as handle:
            aln_list = list(MauveIterator(handle))
        handle = StringIO()
        MauveWriter(handle).write_file(aln_list)
        handle.seek(0)
        aln_list_out = list(MauveIterator(handle))
        for (a1, a2) in zip(aln_list, aln_list_out):
            self.assertEqual(len(a1), len(a2))
            for (r1, r2) in zip(a1, a2):
                self.assertEqual(r1.id, r2.id)
                self.assertEqual(r1.seq, r2.seq)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)