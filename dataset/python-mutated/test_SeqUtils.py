"""Tests for SeqUtils module."""
import os
import unittest
from Bio import SeqIO
from Bio.Seq import MutableSeq
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction
from Bio.SeqUtils import GC_skew
from Bio.SeqUtils import seq1
from Bio.SeqUtils import seq3
from Bio.SeqUtils.CheckSum import crc32
from Bio.SeqUtils.CheckSum import crc64
from Bio.SeqUtils.CheckSum import gcg
from Bio.SeqUtils.CheckSum import seguid
from Bio.SeqUtils import CodonAdaptationIndex
from Bio.SeqUtils.lcc import lcc_mult
from Bio.SeqUtils.lcc import lcc_simp
import warnings
from Bio import BiopythonDeprecationWarning
with warnings.catch_warnings():
    warnings.simplefilter('ignore', BiopythonDeprecationWarning)
    from Bio.SeqUtils.CodonUsage import CodonAdaptationIndex as OldCodonAdaptationIndex

class SeqUtilsTests(unittest.TestCase):
    str_light_chain_one = 'QSALTQPASVSGSPGQSITISCTGTSSDVGSYNLVSWYQQHPGKAPKLMIYEGSKRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYAGSSTLVFGGGTKLTVL'
    str_light_chain_two = 'QSALTQPASVSGSPGQSITISCTGTSSDVGSYNLVSWYQQHPGKAPKLMIYEGSKRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCCSYAGSSTWVFGGGTKLTVL'

    def test_codon_usage_ecoli(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Codon Adaptation Index (CAI) using default E. coli data.'
        CAI = OldCodonAdaptationIndex()
        value = CAI.cai_for_gene('ATGCGTATCGATCGCGATACGATTAGGCGGATG')
        self.assertAlmostEqual(value, 0.09978, places=5)
        self.assertEqual(str(CAI), 'AAA\t1.000\nAAC\t1.000\nAAG\t0.253\nAAT\t0.051\nACA\t0.076\nACC\t1.000\nACG\t0.099\nACT\t0.965\nAGA\t0.004\nAGC\t0.410\nAGG\t0.002\nAGT\t0.085\nATA\t0.003\nATC\t1.000\nATG\t1.000\nATT\t0.185\nCAA\t0.124\nCAC\t1.000\nCAG\t1.000\nCAT\t0.291\nCCA\t0.135\nCCC\t0.012\nCCG\t1.000\nCCT\t0.070\nCGA\t0.004\nCGC\t0.356\nCGG\t0.004\nCGT\t1.000\nCTA\t0.007\nCTC\t0.037\nCTG\t1.000\nCTT\t0.042\nGAA\t1.000\nGAC\t1.000\nGAG\t0.259\nGAT\t0.434\nGCA\t0.586\nGCC\t0.122\nGCG\t0.424\nGCT\t1.000\nGGA\t0.010\nGGC\t0.724\nGGG\t0.019\nGGT\t1.000\nGTA\t0.495\nGTC\t0.066\nGTG\t0.221\nGTT\t1.000\nTAC\t1.000\nTAT\t0.239\nTCA\t0.077\nTCC\t0.744\nTCG\t0.017\nTCT\t1.000\nTGC\t1.000\nTGG\t1.000\nTGT\t0.500\nTTA\t0.020\nTTC\t1.000\nTTG\t0.020\nTTT\t0.296\n')

    def test_codon_usage_custom_old(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Codon Adaptation Index (CAI) using FASTA file for background.'
        dna_fasta_filename = 'fasta.tmp'
        dna_genbank_filename = 'GenBank/NC_005816.gb'
        record = SeqIO.read(dna_genbank_filename, 'genbank')
        records = []
        for feature in record.features:
            if feature.type == 'CDS' and len(feature.location.parts) == 1:
                start = feature.location.start
                end = feature.location.end
                table = int(feature.qualifiers['transl_table'][0])
                if feature.strand == -1:
                    seq = record.seq[start:end].reverse_complement()
                else:
                    seq = record.seq[start:end]
                a = 'M' + seq[3:].translate(table)
                b = feature.qualifiers['translation'][0] + '*'
                self.assertEqual(a, b)
                records.append(SeqRecord(seq, id=feature.qualifiers['protein_id'][0], description=feature.qualifiers['product'][0]))
        with open(dna_fasta_filename, 'w') as handle:
            SeqIO.write(records, handle, 'fasta')
        CAI = OldCodonAdaptationIndex()
        CAI.generate_index(dna_fasta_filename)
        self.assertEqual(record.annotations['source'], 'Yersinia pestis biovar Microtus str. 91001')
        value = CAI.cai_for_gene('ATGCGTATCGATCGCGATACGATTAGGCGGATG')
        self.assertAlmostEqual(value, 0.67213, places=5)
        self.assertEqual(str(CAI), 'AAA\t1.000\nAAC\t0.385\nAAG\t0.344\nAAT\t1.000\nACA\t1.000\nACC\t0.553\nACG\t0.319\nACT\t0.447\nAGA\t0.595\nAGC\t0.967\nAGG\t0.297\nAGT\t1.000\nATA\t0.581\nATC\t0.930\nATG\t1.000\nATT\t1.000\nCAA\t0.381\nCAC\t0.581\nCAG\t1.000\nCAT\t1.000\nCCA\t0.500\nCCC\t0.500\nCCG\t1.000\nCCT\t0.767\nCGA\t0.568\nCGC\t0.919\nCGG\t0.514\nCGT\t1.000\nCTA\t0.106\nCTC\t0.379\nCTG\t1.000\nCTT\t0.424\nGAA\t1.000\nGAC\t0.633\nGAG\t0.506\nGAT\t1.000\nGCA\t1.000\nGCC\t0.617\nGCG\t0.532\nGCT\t0.809\nGGA\t1.000\nGGC\t0.525\nGGG\t0.575\nGGT\t0.950\nGTA\t0.500\nGTC\t0.618\nGTG\t0.971\nGTT\t1.000\nTAA\t1.000\nTAC\t0.434\nTAG\t0.000\nTAT\t1.000\nTCA\t1.000\nTCC\t0.533\nTCG\t0.233\nTCT\t0.967\nTGA\t0.250\nTGC\t1.000\nTGG\t1.000\nTGT\t0.750\nTTA\t0.455\nTTC\t1.000\nTTG\t0.212\nTTT\t0.886\n')
        os.remove(dna_fasta_filename)

    def test_codon_adaptation_index_initialization(self):
        if False:
            print('Hello World!')
        'Test Codon Adaptation Index (CAI) initialization from sequences.'
        dna_filename = 'GenBank/NC_005816.gb'
        record = SeqIO.read(dna_filename, 'genbank')
        records = []
        for feature in record.features:
            if feature.type == 'CDS' and len(feature.location.parts) == 1:
                start = feature.location.start
                end = feature.location.end
                table = int(feature.qualifiers['transl_table'][0])
                if feature.strand == -1:
                    seq = record.seq[start:end].reverse_complement()
                else:
                    seq = record.seq[start:end]
                a = 'M' + seq[3:].translate(table)
                b = feature.qualifiers['translation'][0] + '*'
                self.assertEqual(a, b)
                records.append(SeqRecord(seq, id=feature.qualifiers['protein_id'][0], description=feature.qualifiers['product'][0]))
        cai = CodonAdaptationIndex(records)
        self.assertEqual(record.annotations['source'], 'Yersinia pestis biovar Microtus str. 91001')
        value = cai.calculate('ATGCGTATCGATCGCGATACGATTAGGCGGATG')
        self.assertAlmostEqual(value, 0.70246, places=5)
        optimized_sequence = cai.optimize('ATGCGTATCGATCGCGATACGATTAGGCGGATG', strict=False)
        optimized_value = cai.calculate(optimized_sequence)
        self.assertEqual(optimized_value, 1.0)
        aa_initial = Seq('ATGCGTATCGATCGCGATACGATTAGGCGGATG').translate()
        aa_optimized = optimized_sequence.translate()
        self.assertEqual(aa_initial, aa_optimized)
        with self.assertRaises(KeyError):
            cai.optimize('CAU', 'protein', strict=False)
        self.maxDiff = None
        self.assertEqual(str(cai), 'AAA\t1.000\nAAC\t0.385\nAAG\t0.344\nAAT\t1.000\nACA\t1.000\nACC\t0.553\nACG\t0.319\nACT\t0.447\nAGA\t0.595\nAGC\t0.967\nAGG\t0.297\nAGT\t1.000\nATA\t0.581\nATC\t0.930\nATG\t1.000\nATT\t1.000\nCAA\t0.381\nCAC\t0.581\nCAG\t1.000\nCAT\t1.000\nCCA\t0.500\nCCC\t0.500\nCCG\t1.000\nCCT\t0.767\nCGA\t0.568\nCGC\t0.919\nCGG\t0.514\nCGT\t1.000\nCTA\t0.106\nCTC\t0.379\nCTG\t1.000\nCTT\t0.424\nGAA\t1.000\nGAC\t0.633\nGAG\t0.506\nGAT\t1.000\nGCA\t1.000\nGCC\t0.617\nGCG\t0.532\nGCT\t0.809\nGGA\t1.000\nGGC\t0.525\nGGG\t0.575\nGGT\t0.950\nGTA\t0.500\nGTC\t0.618\nGTG\t0.971\nGTT\t1.000\nTAA\t1.000\nTAC\t0.434\nTAG\t0.062\nTAT\t1.000\nTCA\t1.000\nTCC\t0.533\nTCG\t0.233\nTCT\t0.967\nTGA\t0.250\nTGC\t1.000\nTGG\t1.000\nTGT\t0.750\nTTA\t0.455\nTTC\t1.000\nTTG\t0.212\nTTT\t0.886\n')

    def test_codon_adaptation_index_calculation(self):
        if False:
            i = 10
            return i + 15
        'Test Codon Adaptation Index (CAI) calculation for an mRNA.'
        cai = CodonAdaptationIndex([])
        cai['TTT'] = 0.296
        cai['TTC'] = 1.0
        cai['TTA'] = 0.02
        cai['TTG'] = 0.02
        cai['CTT'] = 0.042
        cai['CTC'] = 0.037
        cai['CTA'] = 0.007
        cai['CTG'] = 1.0
        cai['ATT'] = 0.185
        cai['ATC'] = 1.0
        cai['ATA'] = 0.003
        cai['ATG'] = 1.0
        cai['GTT'] = 1.0
        cai['GTC'] = 0.066
        cai['GTA'] = 0.495
        cai['GTG'] = 0.221
        cai['TAT'] = 0.239
        cai['TAC'] = 1.0
        cai['CAT'] = 0.291
        cai['CAC'] = 1.0
        cai['CAA'] = 0.124
        cai['CAG'] = 1.0
        cai['AAT'] = 0.051
        cai['AAC'] = 1.0
        cai['AAA'] = 1.0
        cai['AAG'] = 0.253
        cai['GAT'] = 0.434
        cai['GAC'] = 1.0
        cai['GAA'] = 1.0
        cai['GAG'] = 0.259
        cai['TCT'] = 1.0
        cai['TCC'] = 0.744
        cai['TCA'] = 0.077
        cai['TCG'] = 0.017
        cai['CCT'] = 0.07
        cai['CCC'] = 0.012
        cai['CCA'] = 0.135
        cai['CCG'] = 1.0
        cai['ACT'] = 0.965
        cai['ACC'] = 1.0
        cai['ACA'] = 0.076
        cai['ACG'] = 0.099
        cai['GCT'] = 1.0
        cai['GCC'] = 0.122
        cai['GCA'] = 0.586
        cai['GCG'] = 0.424
        cai['TGT'] = 0.5
        cai['TGC'] = 1.0
        cai['TGG'] = 1.0
        cai['CGT'] = 1.0
        cai['CGC'] = 0.356
        cai['CGA'] = 0.004
        cai['CGG'] = 0.004
        cai['AGT'] = 0.085
        cai['AGC'] = 0.41
        cai['AGA'] = 0.004
        cai['AGG'] = 0.002
        cai['GGT'] = 1.0
        cai['GGC'] = 0.724
        cai['GGA'] = 0.01
        cai['GGG'] = 0.019
        rpsU = Seq('CCGGTAATTAAAGTACGTGAAAACGAGCCGTTCGACGTAGCTCTGCGTCGCTTCAAGCGTTCCTGCGAAAAAGCAGGTGTTCTGGCGGAAGTTCGTCGTCGTGAGTTCTATGAAAAACCGACTACCGAACGTAAGCGCGCTAAAGCTTCTGCAGTGAAACGTCACGCGAAGAAACTGGCTCGCGAAAACGCACGCCGCACTCGTCTGTAC')
        self.assertAlmostEqual(cai.calculate(rpsU), 0.726, places=3)
        rpoD = Seq('ATGGAGCAAAACCCGCAGTCACAGCTGAAACTTCTTGTCACCCGTGGTAAGGAGCAAGGCTATCTGACCTATGCCGAGGTCAATGACCATCTGCCGGAAGATATCGTCGATTCAGATCAGATCGAAGACATCATCCAAATGATCAACGACATGGGCATTCAGGTGATGGAAGAAGCACCGGATGCCGATGATCTGATGCTGGCTGAAAACACCGCGGACGAAGATGCTGCCGAAGCCGCCGCGCAGGTGCTTTCCAGCGTGGAATCTGAAATCGGGCGCACGACTGACCCGGTACGCATGTACATGCGTGAAATGGGCACCGTTGAACTGTTGACCCGCGAAGGCGAAATTGACATCGCTAAGCGTATTGAAGACGGGATCAACCAGGTTCAATGCTCCGTTGCTGAATATCCGGAAGCGATCACCTATCTGCTGGAACAGTACGATCGTGTTGAAGCAGAAGAAGCGCGTCTGTCCGATCTGATCACCGGCTTTGTTGACCCGAACGCAGAAGAAGATCTGGCACCTACCGCCACTCACGTCGGTTCTGAGCTTTCCCAGGAAGATCTGGACGATGACGAAGATGAAGACGAAGAAGATGGCGATGACGACAGCGCCGATGATGACAACAGCATCGACCCGGAACTGGCTCGCGAAAAATTTGCGGAACTACGCGCTCAGTACGTTGTAACGCGTGACACCATCAAAGCGAAAGGTCGCAGTCACGCTACCGCTCAGGAAGAGATCCTGAAACTGTCTGAAGTATTCAAACAGTTCCGCCTGGTGCCGAAGCAGTTTGACTACCTGGTCAACAGCATGCGCGTCATGATGGACCGCGTTCGTACGCAAGAACGTCTGATCATGAAGCTCTGCGTTGAGCAGTGCAAAATGCCGAAGAAAAACTTCATTACCCTGTTTACCGGCAACGAAACCAGCGATACCTGGTTCAACGCGGCAATTGCGATGAACAAGCCGTGGTCGGAAAAACTGCACGATGTCTCTGAAGAAGTGCATCGCGCCCTGCAAAAACTGCAGCAGATTGAAGAAGAAACCGGCCTGACCATCGAGCAGGTTAAAGATATCAACCGTCGTATGTCCATCGGTGAAGCGAAAGCCCGCCGTGCGAAGAAAGAGATGGTTGAAGCGAACTTACGTCTGGTTATTTCTATCGCTAAGAAATACACCAACCGTGGCTTGCAGTTCCTTGACCTGATTCAGGAAGGCAACATCGGTCTGATGAAAGCGGTTGATAAATTCGAATACCGCCGTGGTTACAAGTTCTCCACCTACGCAACCTGGTGGATCCGTCAGGCGATCACCCGCTCTATCGCGGATCAGGCGCGCACCATCCGTATTCCGGTGCATATGATTGAGACCATCAACAAGCTCAACCGTATTTCTCGCCAGATGCTGCAAGAGATGGGCCGTGAACCGACGCCGGAAGAACTGGCTGAACGTATGCTGATGCCGGAAGACAAGATCCGCAAAGTGCTGAAGATCGCCAAAGAGCCAATCTCCATGGAAACGCCGATCGGTGATGATGAAGATTCGCATCTGGGGGATTTCATCGAGGATACCACCCTCGAGCTGCCGCTGGATTCTGCGACCACCGAAAGCCTGCGTGCGGCAACGCACGACGTGCTGGCTGGCCTGACCGCGCGTGAAGCAAAAGTTCTGCGTATGCGTTTCGGTATCGATATGAACACCGACTACACGCTGGAAGAAGTGGGTAAACAGTTCGACGTTACCCGCGAACGTATCCGTCAGATCGAAGCGAAGGCGCTGCGCAAACTGCGTCACCCGAGCCGTTCTGAAGTGCTGCGTAGCTTCCTGGACGAT')
        self.assertAlmostEqual(cai.calculate(rpoD), 0.582, places=2)
        dnaG = 'ATGGCTGGACGAATCCCACGCGTATTCATTAATGATCTGCTGGCACGCACTGACATCGTCGATCTGATCGATGCCCGTGTGAAGCTGAAAAAGCAGGGCAAGAATTTCCACGCGTGTTGTCCATTCCACAACGAGAAAACCCCGTCCTTCACCGTTAACGGTGAGAAACAGTTTTACCACTGCTTTGGATGTGGCGCGCACGGCAACGCGATCGACTTCCTGATGAACTACGACAAGCTCGAGTTCGTCGAAACGGTCGAAGAGCTGGCAGCAATGCACAATCTTGAAGTGCCATTTGAAGCAGGCAGCGGCCCCAGCCAGATCGAGCGCCATCAGAGGCAAACGCTTTATCAGTTGATGGACGGTCTGAATACGTTTTACCAACAATCTTTACAACAACCTGTTGCCACGTCTGCGCGCCAGTATCTGGAAAAACGCGGATTAAGCCACGAGGTTATCGCTCGCTTTGCGATTGGTTTTGCGCCCCCCGGCTGGGACAACGTCCTGAAGCGGTTTGGCGGCAATCCAGAAAATCGCCAGTCATTGATTGATGCGGGGATGTTGGTCACTAACGATCAGGGACGCAGTTACGATCGTTTCCGCGAGCGGGTGATGTTCCCCATTCGCGATAAACGCGGTCGGGTGATTGGTTTTGGCGGGCGCGTGCTGGGCAACGATACCCCCAAATACCTGAACTCGCCGGAAACAGACATTTTCCATAAAGGCCGCCAGCTTTACGGTCTTTATGAAGCGCAGCAGGATAACGCTGAACCCAATCGTCTGCTTGTGGTCGAAGGCTATATGGACGTGGTGGCGCTGGCGCAATACGGCATTAATTACGCCGTTGCGTCGTTAGGTACGTCAACCACCGCCGATCACATACAACTGTTGTTCCGCGCGACCAACAATGTCATTTGCTGTTATGACGGCGACCGTGCAGGCCGCGATGCCGCCTGGCGAGCGCTGGAAACGGCGCTGCCTTACATGACAGACGGCCGTCAGCTACGCTTTATGTTTTTGCCTGATGGCGAAGACCCTGACACGCTAGTACGAAAAGAAGGTAAAGAAGCGTTTGAAGCGCGGATGGAGCAGGCGATGCCACTCTCCGCATTTCTGTTTAACAGTCTGATGCCGCAAGTTGATCTGAGTACCCCTGACGGGCGCGCACGTTTGAGTACGCTGGCACTACCATTGATATCGCAAGTGCCGGGCGAAACGCTGCGAATATATCTTCGTCAGGAATTAGGCAACAAATTAGGCATACTTGATGACAGCCAGCTTGAACGATTAATGCCAAAAGCGGCAGAGAGCGGCGTTTCTCGCCCTGTTCCGCAGCTAAAACGCACGACCATGCGTATACTTATAGGGTTGCTGGTGCAAAATCCAGAATTAGCGACGTTGGTCCCGCCGCTTGAGAATCTGGATGAAAATAAGCTCCCTGGACTTGGCTTATTCAGAGAACTGGTCAACACTTGTCTCTCCCAGCCAGGTCTGACCACCGGGCAACTTTTAGAGCACTATCGTGGTACAAATAATGCTGCCACCCTTGAAAAACTGTCGATGTGGGACGATATAGCAGATAAGAATATTGCTGAGCAAACCTTCACCGACTCACTCAACCATATGTTTGATTCGCTGCTTGAACTGCGCCAGGAAGAGTTAATCGCTCGTGAGCGCACGCATGGTTTAAGCAACGAAGAACGCCTGGAGCTCTGGACATTAAACCAGGAGCTGGCGAAAAAG'
        self.assertAlmostEqual(cai.calculate(dnaG), 0.271, places=3)
        lacI = 'GTGAAACCAGTAACGTTATACGATGTCGCAGAGTATGCCGGTGTCTCTTATCAGACCGTTTCCCGCGTGGTGAACCAGGCCAGCCACGTTTCTGCGAAAACGCGGGAAAAAGTGGAAGCGGCGATGGCGGAGCTGAATTACATTCCCAACCGCGTGGCACAACAACTGGCGGGCAAACAGTCGTTGCTGATTGGCGTTGCCACCTCCAGTCTGGCCCTGCACGCGCCGTCGCAAATTGTCGCGGCGATTAAATCTCGCGCCGATCAACTGGGTGCCAGCGTGGTGGTGTCGATGGTAGAACGAAGCGGCGTCGAAGCCTGTAAAGCGGCGGTGCACAATCTTCTCGCGCAACGCGTCAGTGGGCTGATCATTAACTATCCGCTGGATGACCAGGATGCCATTGCTGTGGAAGCTGCCTGCACTAATGTTCCGGCGTTATTTCTTGATGTCTCTGACCAGACACCCATCAACAGTATTATTTTCTCCCATGAAGACGGTACGCGACTGGGCGTGGAGCATCTGGTCGCATTGGGTCACCAGCAAATCGCGCTGTTAGCGGGCCCATTAAGTTCTGTCTCGGCGCGTCTGCGTCTGGCTGGCTGGCATAAATATCTCACTCGCAATCAAATTCAGCCGATAGCGGAACGGGAAGGCGACTGGAGTGCCATGTCCGGTTTTCAACAAACCATGCAAATGCTGAATGAGGGCATCGTTCCCACTGCGATGCTGGTTGCCAACGATCAGATGGCGCTGGGCGCAATGCGCGCCATTACCGAGTCCGGGCTGCGCGTTGGTGCGGATATCTCGGTAGTGGGATACGACGATACCGAAGACAGCTCATGTTATATCCCGCCGTTAACCACCATCAAACAGGATTTTCGCCTGCTGGGGCAAACCAGCGTGGACCGCTTGCTGCAACTCTCTCAGGGCCAGGCGGTGAAGGGCAATCAGCTGTTGCCCGTCTCACTGGTGAAAAGAAAAACCACCCTGGCGCCCAATACGCAAACCGCCTCTCCCCGCGCGTTGGCCGATTCATTAATGCAGCTGGCACGACAGGTTTCCCGACTGGAAAGCGGGCAG'
        self.assertAlmostEqual(cai.calculate(lacI), 0.296, places=2)
        trpR = 'ATGGCCCAACAATCACCCTATTCAGCAGCGATGGCAGAACAGCGTCACCAGGAGTGGTTACGTTTTGTCGACCTGCTTAAGAATGCCTACCAAAACGATCTCCATTTACCGTTGTTAAACCTGATGCTGACGCCAGATGAGCGCGAAGCGTTGGGGACTCGCGTGCGTATTGTCGAAGAGCTGTTGCGCGGCGAAATGAGCCAGCGTGAGTTAAAAAATGAACTCGGCGCAGGCATCGCGACGATTACGCGTGGATCTAACAGCCTGAAAGCCGCGCCCGTCGAGCTGCGCCAGTGGCTGGAAGAGGTGTTGCTGAAAAGCGAT'
        self.assertAlmostEqual(cai.calculate(trpR), 0.267, places=2)
        lpp = 'ATGAAAGCTACTAAACTGGTACTGGGCGCGGTAATCCTGGGTTCTACTCTGCTGGCAGGTTGCTCCAGCAACGCTAAAATCGATCAGCTGTCTTCTGACGTTCAGACTCTGAACGCTAAAGTTGACCAGCTGAGCAACGACGTGAACGCAATGCGTTCCGACGTTCAGGCTGCTAAAGATGACGCAGCTCGTGCTAACCAGCGTCTGGACAACATGGCTACTAAATACCGCAAG'
        self.assertAlmostEqual(cai.calculate(lpp), 0.849, places=3)

    def test_crc_checksum_collision(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(self.str_light_chain_one, self.str_light_chain_two)
        self.assertNotEqual(crc32(self.str_light_chain_one), crc32(self.str_light_chain_two))
        self.assertEqual(crc64(self.str_light_chain_one), crc64(self.str_light_chain_two))
        self.assertNotEqual(gcg(self.str_light_chain_one), gcg(self.str_light_chain_two))
        self.assertNotEqual(seguid(self.str_light_chain_one), seguid(self.str_light_chain_two))

    def seq_checksums(self, seq_str, exp_crc32, exp_crc64, exp_gcg, exp_seguid, exp_simple_LCC, exp_window_LCC):
        if False:
            i = 10
            return i + 15
        for s in [seq_str, Seq(seq_str), MutableSeq(seq_str)]:
            self.assertEqual(exp_crc32, crc32(s))
            self.assertEqual(exp_crc64, crc64(s))
            self.assertEqual(exp_gcg, gcg(s))
            self.assertEqual(exp_seguid, seguid(s))
            self.assertAlmostEqual(exp_simple_LCC, lcc_simp(s), places=4)
            values = lcc_mult(s, 20)
            self.assertEqual(len(exp_window_LCC), len(values), values)
            for (value1, value2) in zip(exp_window_LCC, values):
                self.assertAlmostEqual(value1, value2, places=2)

    def test_checksum1(self):
        if False:
            return 10
        self.seq_checksums(self.str_light_chain_one, 2994980265, 'CRC-44CAAD88706CC153', 9729, 'BpBeDdcNUYNsdk46JoJdw7Pd3BI', 0.516, (0.4982, 0.4794, 0.4794, 0.4794, 0.3241, 0.216, 0.1764, 0.1764, 0.1764, 0.1764, 0.2657, 0.2948, 0.1287))

    def test_checksum2(self):
        if False:
            i = 10
            return i + 15
        self.seq_checksums(self.str_light_chain_two, 802105214, 'CRC-44CAAD88706CC153', 9647, 'X5XEaayob1nZLOc7eVT9qyczarY', 0.5343, (0.4982, 0.4794, 0.4794, 0.4794, 0.3241, 0.216, 0.1764, 0.1764, 0.1764, 0.1764, 0.2657, 0.2948, 0.1287))

    def test_checksum3(self):
        if False:
            for i in range(10):
                print('nop')
        self.seq_checksums('ATGCGTATCGATCGCGATACGATTAGGCGGAT', 817679856, 'CRC-6234FF451DC6DFC6', 7959, '8WCUbVjBgiRmM10gfR7XJNjbwnE', 0.9886, (1.0, 0.9927, 0.9927, 1.0, 0.9927, 0.9854, 0.9927, 0.9927, 0.9927, 0.9794, 0.9794, 0.9794, 0.9794))

    def test_gc_fraction(self):
        if False:
            while True:
                i = 10
        'Tests gc_fraction function.'
        self.assertAlmostEqual(gc_fraction('', 'ignore'), 0, places=3)
        self.assertAlmostEqual(gc_fraction('', 'weighted'), 0, places=3)
        self.assertAlmostEqual(gc_fraction('', 'remove'), 0, places=3)
        seq = 'ACGGGCTACCGTATAGGCAAGAGATGATGCCC'
        self.assertAlmostEqual(gc_fraction(seq, 'ignore'), 0.5625, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'weighted'), 0.5625, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'remove'), 0.5625, places=3)
        seq = 'ACTGSSSS'
        self.assertAlmostEqual(gc_fraction(seq, 'ignore'), 0.75, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'weighted'), 0.75, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'remove'), 0.75, places=3)
        seq = 'CCTGNN'
        self.assertAlmostEqual(gc_fraction(seq, 'ignore'), 0.5, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'weighted'), 0.667, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'remove'), 0.75, places=3)
        seq = 'GDVV'
        self.assertAlmostEqual(gc_fraction(seq, 'ignore'), 0.25, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'weighted'), 0.6667, places=3)
        self.assertAlmostEqual(gc_fraction(seq, 'remove'), 1.0, places=3)
        with self.assertRaises(ValueError):
            gc_fraction(seq, 'other string')

    def test_GC_skew(self):
        if False:
            print('Hello World!')
        s = 'A' * 50
        seq = Seq(s)
        record = SeqRecord(seq)
        self.assertEqual(GC_skew(s)[0], 0)
        self.assertEqual(GC_skew(seq)[0], 0)
        self.assertEqual(GC_skew(record)[0], 0)

    def test_seq1_seq3(self):
        if False:
            i = 10
            return i + 15
        s3 = 'MetAlaTyrtrpcysthrLYSLEUILEGlYPrOGlNaSnaLapRoTyRLySSeRHisTrpLysThr'
        s1 = 'MAYWCTKLIGPQNAPYKSHWKT'
        self.assertEqual(seq1(s3), s1)
        self.assertEqual(seq3(s1).upper(), s3.upper())
        self.assertEqual(seq1(seq3(s1)), s1)
        self.assertEqual(seq3(seq1(s3)).upper(), s3.upper())

    def test_codon_adaptation_index(self):
        if False:
            while True:
                i = 10
        X = OldCodonAdaptationIndex()
        path = os.path.join('CodonUsage', 'HighlyExpressedGenes.txt')
        X.generate_index(path)
        self.assertEqual(len(X.index), 64)
        self.assertAlmostEqual(X.index['AAA'], 1.0, places=3)
        self.assertAlmostEqual(X.index['AAC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['AAG'], 0.219, places=3)
        self.assertAlmostEqual(X.index['AAT'], 0.293, places=3)
        self.assertAlmostEqual(X.index['ACA'], 0.11, places=3)
        self.assertAlmostEqual(X.index['ACC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['ACG'], 0.204, places=3)
        self.assertAlmostEqual(X.index['ACT'], 0.517, places=3)
        self.assertAlmostEqual(X.index['AGA'], 0.018, places=3)
        self.assertAlmostEqual(X.index['AGC'], 0.762, places=3)
        self.assertAlmostEqual(X.index['AGG'], 0.006, places=3)
        self.assertAlmostEqual(X.index['AGT'], 0.195, places=3)
        self.assertAlmostEqual(X.index['ATA'], 0.015, places=3)
        self.assertAlmostEqual(X.index['ATC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['ATG'], 1.0, places=3)
        self.assertAlmostEqual(X.index['ATT'], 0.49, places=3)
        self.assertAlmostEqual(X.index['CAA'], 0.259, places=3)
        self.assertAlmostEqual(X.index['CAC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['CAG'], 1.0, places=3)
        self.assertAlmostEqual(X.index['CAT'], 0.416, places=3)
        self.assertAlmostEqual(X.index['CCA'], 0.247, places=3)
        self.assertAlmostEqual(X.index['CCC'], 0.04, places=3)
        self.assertAlmostEqual(X.index['CCG'], 1.0, places=3)
        self.assertAlmostEqual(X.index['CCT'], 0.161, places=3)
        self.assertAlmostEqual(X.index['CGA'], 0.023, places=3)
        self.assertAlmostEqual(X.index['CGC'], 0.531, places=3)
        self.assertAlmostEqual(X.index['CGG'], 0.014, places=3)
        self.assertAlmostEqual(X.index['CGT'], 1.0, places=3)
        self.assertAlmostEqual(X.index['CTA'], 0.017, places=3)
        self.assertAlmostEqual(X.index['CTC'], 0.1, places=3)
        self.assertAlmostEqual(X.index['CTG'], 1.0, places=3)
        self.assertAlmostEqual(X.index['CTT'], 0.085, places=3)
        self.assertAlmostEqual(X.index['GAA'], 1.0, places=3)
        self.assertAlmostEqual(X.index['GAC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['GAG'], 0.308, places=3)
        self.assertAlmostEqual(X.index['GAT'], 0.886, places=3)
        self.assertAlmostEqual(X.index['GCA'], 0.794, places=3)
        self.assertAlmostEqual(X.index['GCC'], 0.538, places=3)
        self.assertAlmostEqual(X.index['GCG'], 0.937, places=3)
        self.assertAlmostEqual(X.index['GCT'], 1.0, places=3)
        self.assertAlmostEqual(X.index['GGA'], 0.056, places=3)
        self.assertAlmostEqual(X.index['GGC'], 0.892, places=3)
        self.assertAlmostEqual(X.index['GGG'], 0.103, places=3)
        self.assertAlmostEqual(X.index['GGT'], 1.0, places=3)
        self.assertAlmostEqual(X.index['GTA'], 0.465, places=3)
        self.assertAlmostEqual(X.index['GTC'], 0.297, places=3)
        self.assertAlmostEqual(X.index['GTG'], 0.618, places=3)
        self.assertAlmostEqual(X.index['GTT'], 1.0, places=3)
        self.assertAlmostEqual(X.index['TAA'], 1.0, places=3)
        self.assertAlmostEqual(X.index['TAC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['TAG'], 0.012, places=3)
        self.assertAlmostEqual(X.index['TAT'], 0.606, places=3)
        self.assertAlmostEqual(X.index['TCA'], 0.221, places=3)
        self.assertAlmostEqual(X.index['TCC'], 0.785, places=3)
        self.assertAlmostEqual(X.index['TCG'], 0.24, places=3)
        self.assertAlmostEqual(X.index['TCT'], 1.0, places=3)
        self.assertAlmostEqual(X.index['TGA'], 0.081, places=3)
        self.assertAlmostEqual(X.index['TGC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['TGG'], 1.0, places=3)
        self.assertAlmostEqual(X.index['TGT'], 0.721, places=3)
        self.assertAlmostEqual(X.index['TTA'], 0.059, places=3)
        self.assertAlmostEqual(X.index['TTC'], 1.0, places=3)
        self.assertAlmostEqual(X.index['TTG'], 0.072, places=3)
        self.assertAlmostEqual(X.index['TTT'], 0.457, places=3)
        cai = X.cai_for_gene('ATGAAACGCATTAGCACCACCATTACCACCACCATCACCATTACCACAGGTAACGGTGCGGGCTGA')
        self.assertAlmostEqual(cai, 0.6723, places=3)

    def test_lcc_simp(self):
        if False:
            i = 10
            return i + 15
        s = 'ACGATAGC'
        seq = Seq(s)
        record = SeqRecord(seq)
        self.assertAlmostEqual(lcc_simp(s), 0.9528, places=4)
        self.assertAlmostEqual(lcc_simp(seq), 0.9528, places=4)
        self.assertAlmostEqual(lcc_simp(record), 0.9528, places=4)

    def test_lcc_mult(self):
        if False:
            while True:
                i = 10
        s = 'ACGATAGC'
        seq = Seq(s)
        record = SeqRecord(seq)
        llc_lst = lcc_mult(s, len(s))
        self.assertEqual(len(llc_lst), 1)
        self.assertAlmostEqual(llc_lst[0], 0.9528, places=4)
        llc_lst = lcc_mult(seq, len(seq))
        self.assertEqual(len(llc_lst), 1)
        self.assertAlmostEqual(llc_lst[0], 0.9528, places=4)
        llc_lst = lcc_mult(record, len(record))
        self.assertEqual(len(llc_lst), 1)
        self.assertAlmostEqual(llc_lst[0], 0.9528, places=4)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)