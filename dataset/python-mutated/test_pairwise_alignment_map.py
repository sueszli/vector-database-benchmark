"""Tests for mapping pairwise alignments."""
import os
import random
import unittest
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install numpy if you want to use Bio.Align.Alignment.map.') from None
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from Bio import Align
from Bio.Align import PairwiseAligner, Alignment

class TestSimple(unittest.TestCase):
    aligner = PairwiseAligner()
    aligner.internal_open_gap_score = -1
    aligner.internal_extend_gap_score = -0.0
    aligner.match_score = +1
    aligner.mismatch_score = -1
    aligner.mode = 'local'

    def test_internal(self):
        if False:
            while True:
                i = 10
        aligner = self.aligner
        chromosome = Seq('AAAAAAAAAAAAGGGGGGGCCCCCGGGGGGAAAAAAAAAA')
        chromosome.id = 'chromosome'
        transcript = Seq('GGGGGGGCCCCCGGGGGGA')
        transcript.id = 'transcript'
        sequence = Seq('GGCCCCCGGG')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript)
        self.assertEqual(len(alignments1), 1)
        alignment1 = alignments1[0]
        self.assertTrue(np.array_equal(alignment1.coordinates, np.array([[12, 31], [0, 19]])))
        self.assertEqual(str(alignment1), 'chromosom        12 GGGGGGGCCCCCGGGGGGA 31\n                  0 ||||||||||||||||||| 19\ntranscrip         0 GGGGGGGCCCCCGGGGGGA 19\n')
        alignments2 = aligner.align(transcript, sequence)
        self.assertEqual(len(alignments2), 1)
        alignment2 = alignments2[0]
        self.assertTrue(np.array_equal(alignment2.coordinates, np.array([[5, 15], [0, 10]])))
        self.assertEqual(str(alignment2), 'transcrip         5 GGCCCCCGGG 15\n                  0 |||||||||| 10\nsequence          0 GGCCCCCGGG 10\n')
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[17, 27], [0, 10]])))
        self.assertEqual(str(alignment), 'chromosom        17 GGCCCCCGGG 27\n                  0 |||||||||| 10\nsequence          0 GGCCCCCGGG 10\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '10\t0\t0\t0\t0\t0\t0\t0\t+\tsequence\t10\t0\t10\tchromosome\t40\t17\t27\t1\t10,\t0,\t17,\n')

    def test_left_overhang(self):
        if False:
            return 10
        aligner = self.aligner
        chromosome = Seq('GGGCCCCCGGGGGGAAAAAAAAAA')
        chromosome.id = 'chromosome'
        transcript = Seq('AGGGGGCCCCCGGGGGGA')
        transcript.id = 'transcript'
        sequence = Seq('GGGGGCCCCCGGG')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript)
        self.assertEqual(len(alignments1), 1)
        alignment1 = alignments1[0]
        self.assertEqual(str(alignment1), 'chromosom         0 GGGCCCCCGGGGGGA 15\n                  0 ||||||||||||||| 15\ntranscrip         3 GGGCCCCCGGGGGGA 18\n')
        alignments2 = aligner.align(transcript, sequence)
        self.assertEqual(len(alignments2), 1)
        alignment2 = alignments2[0]
        self.assertEqual(str(alignment2), 'transcrip         1 GGGGGCCCCCGGG 14\n                  0 ||||||||||||| 13\nsequence          0 GGGGGCCCCCGGG 13\n')
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[0, 11], [2, 13]])))
        self.assertEqual(str(alignment), 'chromosom         0 GGGCCCCCGGG 11\n                  0 ||||||||||| 11\nsequence          2 GGGCCCCCGGG 13\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '11\t0\t0\t0\t0\t0\t0\t0\t+\tsequence\t13\t2\t13\tchromosome\t24\t0\t11\t1\t11,\t2,\t0,\n')

    def test_right_overhang(self):
        if False:
            return 10
        aligner = self.aligner
        chromosome = Seq('AAAAAAAAAAAAGGGGGGGCCCCCGGG')
        chromosome.id = 'chromosome'
        transcript = Seq('GGGGGGGCCCCCGGGGGGA')
        transcript.id = 'transcript'
        sequence = Seq('GGCCCCCGGGGG')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript)
        self.assertEqual(len(alignments1), 1)
        alignment1 = alignments1[0]
        self.assertEqual(str(alignment1), 'chromosom        12 GGGGGGGCCCCCGGG 27\n                  0 ||||||||||||||| 15\ntranscrip         0 GGGGGGGCCCCCGGG 15\n')
        alignments2 = aligner.align(transcript, sequence)
        self.assertEqual(len(alignments2), 1)
        alignment2 = alignments2[0]
        self.assertEqual(str(alignment2), 'transcrip         5 GGCCCCCGGGGG 17\n                  0 |||||||||||| 12\nsequence          0 GGCCCCCGGGGG 12\n')
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[17, 27], [0, 10]])))
        self.assertEqual(str(alignment), 'chromosom        17 GGCCCCCGGG 27\n                  0 |||||||||| 10\nsequence          0 GGCCCCCGGG 10\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '10\t0\t0\t0\t0\t0\t0\t0\t+\tsequence\t12\t0\t10\tchromosome\t27\t17\t27\t1\t10,\t0,\t17,\n')

    def test_reverse_transcript(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = self.aligner
        chromosome = Seq('AAAAAAAAAAAAGGGGGGGCCCCCGGGGGGAAAAAAAAAA')
        chromosome.id = 'chromosome'
        transcript = Seq('TCCCCCCGGGGGCCCCCCC')
        transcript.id = 'transcript'
        sequence = Seq('GGCCCCCGGG')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript, strand='-')
        self.assertEqual(len(alignments1), 1)
        alignment1 = alignments1[0]
        self.assertTrue(np.array_equal(alignment1.coordinates, np.array([[12, 31], [19, 0]])))
        self.assertEqual(str(alignment1), 'chromosom        12 GGGGGGGCCCCCGGGGGGA 31\n                  0 ||||||||||||||||||| 19\ntranscrip        19 GGGGGGGCCCCCGGGGGGA  0\n')
        alignments2 = aligner.align(transcript, sequence, strand='-')
        self.assertEqual(len(alignments2), 1)
        alignment2 = alignments2[0]
        self.assertTrue(np.array_equal(alignment2.coordinates, np.array([[4, 14], [10, 0]])))
        self.assertEqual(str(alignment2), 'transcrip         4 CCCGGGGGCC 14\n                  0 |||||||||| 10\nsequence         10 CCCGGGGGCC  0\n')
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[17, 27], [0, 10]])))
        self.assertEqual(str(alignment), 'chromosom        17 GGCCCCCGGG 27\n                  0 |||||||||| 10\nsequence          0 GGCCCCCGGG 10\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '10\t0\t0\t0\t0\t0\t0\t0\t+\tsequence\t10\t0\t10\tchromosome\t40\t17\t27\t1\t10,\t0,\t17,\n')

    def test_reverse_sequence(self):
        if False:
            print('Hello World!')
        aligner = self.aligner
        chromosome = Seq('AAAAAAAAAAAAGGGGGGGCCCCCGGGGGGAAAAAAAAAA')
        chromosome.id = 'chromosome'
        transcript = Seq('GGGGGGGCCCCCGGGGGGA')
        transcript.id = 'transcript'
        sequence = Seq('CCCGGGGGCC')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript)
        self.assertEqual(len(alignments1), 1)
        alignment1 = alignments1[0]
        self.assertTrue(np.array_equal(alignment1.coordinates, np.array([[12, 31], [0, 19]])))
        self.assertEqual(str(alignment1), 'chromosom        12 GGGGGGGCCCCCGGGGGGA 31\n                  0 ||||||||||||||||||| 19\ntranscrip         0 GGGGGGGCCCCCGGGGGGA 19\n')
        alignments2 = aligner.align(transcript, sequence, '-')
        self.assertEqual(len(alignments2), 1)
        alignment2 = alignments2[0]
        self.assertTrue(np.array_equal(alignment2.coordinates, np.array([[5, 15], [10, 0]])))
        self.assertEqual(str(alignment2), 'transcrip         5 GGCCCCCGGG 15\n                  0 |||||||||| 10\nsequence         10 GGCCCCCGGG  0\n')
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[17, 27], [10, 0]])))
        self.assertEqual(str(alignment), 'chromosom        17 GGCCCCCGGG 27\n                  0 |||||||||| 10\nsequence         10 GGCCCCCGGG  0\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '10\t0\t0\t0\t0\t0\t0\t0\t-\tsequence\t10\t0\t10\tchromosome\t40\t17\t27\t1\t10,\t0,\t17,\n')

    def test_reverse_transcript_sequence(self):
        if False:
            while True:
                i = 10
        aligner = self.aligner
        chromosome = Seq('AAAAAAAAAAAAGGGGGGGCCCCCGGGGGGAAAAAAAAAA')
        chromosome.id = 'chromosome'
        transcript = Seq('TCCCCCCGGGGGCCCCCCC')
        transcript.id = 'transcript'
        sequence = Seq('CCCGGGGGCC')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript, '-')
        self.assertEqual(len(alignments1), 1)
        alignment1 = alignments1[0]
        self.assertTrue(np.array_equal(alignment1.coordinates, np.array([[12, 31], [19, 0]])))
        self.assertEqual(str(alignment1), 'chromosom        12 GGGGGGGCCCCCGGGGGGA 31\n                  0 ||||||||||||||||||| 19\ntranscrip        19 GGGGGGGCCCCCGGGGGGA  0\n')
        alignments2 = aligner.align(transcript, sequence)
        self.assertEqual(len(alignments2), 1)
        alignment2 = alignments2[0]
        self.assertTrue(np.array_equal(alignment2.coordinates, np.array([[4, 14], [0, 10]])))
        self.assertEqual(str(alignment2), 'transcrip         4 CCCGGGGGCC 14\n                  0 |||||||||| 10\nsequence          0 CCCGGGGGCC 10\n')
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[17, 27], [10, 0]])))
        self.assertEqual(str(alignment), 'chromosom        17 GGCCCCCGGG 27\n                  0 |||||||||| 10\nsequence         10 GGCCCCCGGG  0\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '10\t0\t0\t0\t0\t0\t0\t0\t-\tsequence\t10\t0\t10\tchromosome\t40\t17\t27\t1\t10,\t0,\t17,\n')

class TestComplex(unittest.TestCase):
    aligner = PairwiseAligner()
    aligner.internal_open_gap_score = -1
    aligner.internal_extend_gap_score = -0.0
    aligner.match_score = +1
    aligner.mismatch_score = -1
    aligner.mode = 'local'

    def test1(self):
        if False:
            print('Hello World!')
        aligner = self.aligner
        chromosome = Seq('GCCTACCGTATAACAATGGTTATAATACAAGGCGGTCATAATTAAAGGGAGTGCAGCAACGGCCTGCTCTCCAAAAAAACAGGTTTTATGAAAAGAAAGTGCATTAACTGTTAAAGCCGTCATATCGGTGGGTTCTGCCAGTCACCGGCATACGTCCTGGGACAAAGACTTTTTACTACAATGCCAGGCGGGAGAGTCACCCGCCGCGGTGTCGACCCAGGGGACAGCGGGAAGATGTCGTGGTTTCCTTGTCATTAACCAACTCCATCTTAAAAGCTCCTCTAGCCATGGCATGGTACGTTGCGCGCACCCTTTTATCGGTAAGGCGCGGTGACTCTCTCCCAAAACAGTGCCATAATGGTTCGCTTCCTACCTAAGGCACTTACGGCCAATTAATGCGCAAGCGAGCGGAAGGTCTAACAGGGCACCGAATTCGATTA')
        chromosome.id = 'chromosome'
        transcript = Seq('GGAATTTTAGCAGCCAAAGGACGGATCCTCCAAGGGGCCCCAGCACAGCACATTTTTAACGCGAACTAAGCGGGAGCGCATGTGGGACAGTTGATCCCATCCGCCTCAAAATTTCTCGCAATATCGGTTGGGGCACAGGTCCACTTTACGAATTCATACCGTGGTAGAGACCTTTATTAGATAGATATGACTGTTTGATTGCGGCATAGTACGACGAAGCAAGGGGATGGACGTTTCGGTTGCATTCGACCGGGTTGGGTCGAAAAACAGGTTTTATGAAAAGAAAGTGCATTAACTGTTAAAGCCGTCATATCGGTGGGTTC')
        transcript.id = 'transcript'
        sequence = Seq('TCCAAGGGGCCCCAGCACAGCACATTTTTAACGCGGGGACAGTTGATCCCATCCGCCTTTTACGAATTCATACCGTGGTAGGCGGCATAGTACGACGAAGCGGTTGGGTCGAAAAACAGGTTGCCGTCATATCGGTGGGTTC')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript)
        alignment1 = alignments1[0]
        self.assertEqual(alignment1.coordinates.shape[1], 164)
        self.assertEqual(str(alignment1), 'chromosom        14 AATGGTTATA------ATACAAGG-CGG----TCATAATTAAAGGGAGTG---CAGCAAC\n                  0 |||--||-||------|---||||-|||----||------.|||||---|---|||||--\ntranscrip         2 AAT--TT-TAGCAGCCA---AAGGACGGATCCTC------CAAGGG---GCCCCAGCA--\n\nchromosom        60 GGCCTGCTCTCCAAAAAAACAGGTTTTATGAAAAGAAAGTGCATTAACTGTTAAAGC---\n                 60 ---|.||-----------|||--||||-|------||.|.|----||||----||||---\ntranscrip        45 ---CAGC-----------ACA--TTTT-T------AACGCG----AACT----AAGCGGG\n\nchromosom       117 --CGTCATATCGGTGG----GTTCTGCCAGTCACCGGCATACGTCCTGGGACAAAGACTT\n                120 --||-|||----||||----||--||--|-|--||--|||.||-|||----||||-|---\ntranscrip        74 AGCG-CAT----GTGGGACAGT--TG--A-T--CC--CATCCG-CCT----CAAA-A---\n\nchromosom       171 TTTACT-ACAATGCCAGGCGGGAGAGTCACCCGCCGCGGTGTCGACCCAGGGG-ACAGCG\n                180 |||-||-.||||------------|-|---------||||-|-------||||-||||--\ntranscrip       111 TTT-CTCGCAAT------------A-T---------CGGT-T-------GGGGCACAG--\n\nchromosom       229 GGAAGATGTCGTGGTTTC-CTT---G---TCATTAACC-------A-ACTCCATCTTA--\n                240 -------|||-------|-|||---|---||||--|||-------|-|--||-|-|||--\ntranscrip       138 -------GTC-------CACTTTACGAATTCAT--ACCGTGGTAGAGA--CC-T-TTATT\n\nchromosom       272 AAAGCTCCTCTAGCCATGGCATG---GT---ACGTTGCGCGCACCCTTTTA-T----CG-\n                300 |.|-------|||--||---|||---||---|--|||||-|||------||-|----||-\ntranscrip       178 AGA-------TAG--AT---ATGACTGTTTGA--TTGCG-GCA------TAGTACGACGA\n\nchromosom       320 -GTAAGG-------CG---CGGT-------GACTCTC--------TCCCAAAACAGTGCC\n                360 -|.||||-------||---||||-------|||---|--------||..||||||-----\ntranscrip       217 AGCAAGGGGATGGACGTTTCGGTTGCATTCGAC---CGGGTTGGGTCGAAAAACA-----\n\nchromosom       354 ATAATGGTTCGCTTCCTACCT-------AAG-GCACTT-ACGGCCAATTAATGCGCAAGC\n                420 -----|||----||--||--|-------|||-|||-||-||.|----|||------||||\ntranscrip       269 -----GGT----TT--TA--TGAAAAGAAAGTGCA-TTAACTG----TTA------AAGC\n\nchromosom       405 GAGCGGAAGGTC-TAACAG-GGCACCGAATTC 435\n                480 ---|-----|||-||.|.|-||----|--||| 512\ntranscrip       305 ---C-----GTCATATCGGTGG----G--TTC 323\n')
        alignments2 = aligner.align(transcript, sequence)
        alignment2 = alignments2[0]
        self.assertEqual(alignment2.coordinates.shape[1], 12)
        self.assertEqual(str(alignment2), 'transcrip        28 TCCAAGGGGCCCCAGCACAGCACATTTTTAACGCGAACTAAGCGGGAGCGCATGTGGGAC\n                  0 |||||||||||||||||||||||||||||||||||--------------------|||||\nsequence          0 TCCAAGGGGCCCCAGCACAGCACATTTTTAACGCG--------------------GGGAC\n\ntranscrip        88 AGTTGATCCCATCCGCCTCAAAATTTCTCGCAATATCGGTTGGGGCACAGGTCCACTTTA\n                 60 ||||||||||||||||||--------------------------------------||||\nsequence         40 AGTTGATCCCATCCGCCT--------------------------------------TTTA\n\ntranscrip       148 CGAATTCATACCGTGGTAGAGACCTTTATTAGATAGATATGACTGTTTGATTGCGGCATA\n                120 |||||||||||||||||||---------------------------------||||||||\nsequence         62 CGAATTCATACCGTGGTAG---------------------------------GCGGCATA\n\ntranscrip       208 GTACGACGAAGCAAGGGGATGGACGTTTCGGTTGCATTCGACCGGGTTGGGTCGAAAAAC\n                180 ||||||||||||--------------------------------||||||||||||||||\nsequence         89 GTACGACGAAGC--------------------------------GGTTGGGTCGAAAAAC\n\ntranscrip       268 AGGTTTTATGAAAAGAAAGTGCATTAACTGTTAAAGCCGTCATATCGGTGGGTTC 323\n                240 |||||------------------------------|||||||||||||||||||| 295\nsequence        117 AGGTT------------------------------GCCGTCATATCGGTGGGTTC 142\n')
        alignment = alignment1.map(alignment2)
        self.assertEqual(alignment.coordinates.shape[1], 76)
        self.assertEqual(str(alignment), 'chromosom        35 TCATAATTAAAGGGAGTG---CAGCAACGGCCTGCTCTCCAAAAAAACAGGTTTTATGAA\n                  0 ||------.|||||---|---|||||-----|.||-----------|||--||||-|---\nsequence          0 TC------CAAGGG---GCCCCAGCA-----CAGC-----------ACA--TTTT-T---\n\nchromosom        92 AAGAAAGTGCATTAACTGTTAAAGCCGTCATATCGGTGG----GTTCTGCCAGTCACCGG\n                 60 ---||.|.|----------------------------||----||--||--|-|--||--\nsequence         29 ---AACGCG----------------------------GGGACAGT--TG--A-T--CC--\n\nchromosom       148 CATACGTCCTGGGACAAAGACTTTTTACTACAATGCCAGGCGGGAGAGTCACCCGCCGCG\n                120 |||.||-|||--------------------------------------------------\nsequence         49 CATCCG-CCT--------------------------------------------------\n\nchromosom       208 GTGTCGACCCAGGGGACAGCGGGAAGATGTCGTGGTTTCCTT---G---TCATTAACCAA\n                180 ----------------------------------------||---|---||||--|||--\nsequence         58 ----------------------------------------TTTACGAATTCAT--ACC--\n\nchromosom       262 CTCCATCTTAAAAGCTCCTCTAGCCATGGCATGGTACGTT-------GCGCGCACCCTTT\n                240 -----------------------------------------------|||-|||------\nsequence         74 ----------------------------------------GTGGTAGGCG-GCA------\n\nchromosom       315 TA-T----CG--GTAAGGCGCGGTGACTCTC-------TCCCAAAACAGTGCCATAATGG\n                300 ||-|----||--|.------------------------||..||||||----------||\nsequence         87 TAGTACGACGAAGC-----------------GGTTGGGTCGAAAAACA----------GG\n\nchromosom       361 TTCGCTTCCTACCTAAGGCACTTACGGCCAATTAATGCGCAAGCGAGCGGAAGGTC-TAA\n                360 |----|------------------------------------||---|-----|||-||.\nsequence        120 T----T------------------------------------GC---C-----GTCATAT\n\nchromosom       420 CAG-GGCACCGAATTC 435\n                420 |.|-||----|--||| 436\nsequence        132 CGGTGG----G--TTC 142\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '96\t10\t0\t0\t11\t36\t27\t294\t+\tsequence\t142\t0\t142\tchromosome\t440\t35\t435\t37\t2,6,1,5,4,3,4,1,6,2,2,2,1,1,2,6,3,2,1,4,3,3,3,2,1,2,2,10,3,1,2,1,3,6,2,1,3,\t0,2,8,12,17,21,24,28,29,35,41,43,45,46,47,49,55,58,63,67,71,81,84,87,90,95,99,108,118,121,122,124,125,129,136,138,139,\t35,43,52,53,63,78,83,88,95,129,131,135,139,141,144,148,155,248,250,251,257,302,306,315,317,318,320,339,359,366,403,408,414,417,423,429,432,\n')

    def test2(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = self.aligner
        chromosome = Seq('CTAATGCGCCTTGGTTTTGGCTTAACTAGAAGCAACCTGTAAGATTGCCAATTCTTCAGTCGAAGTAAATCTTCAATGTTTTGGACTCTTAGCGGATATGCGGCTGAGAAGTACGACATGTGTACATTCATACCTGCGTGACGGTCAGCCTCCCCCGGGACCTCATTGGGCGAATCTAGGTGTGATAATTGACACACTCTTGGTAAGAAGCACTCTTTACCCGATCTCCAAGTACCGACGCCAAGGCCAAGCTCTGCGATCTAAAGCTGCCGATCGTAGATCCAAGTCCTCAGCAAGCTCGCACGAATACGCAGTTCGAAGGCTGGGTGTTGTACGACGGTACGGTTGCTATAGCACTTTCGCGGTCTCGCTATTTTCAGTTTGACTCACCAGTCAGTATTGTCATCGACCAACTTGGAATAGTGTAACGCAGCGCTTGA')
        chromosome.id = 'chromosome'
        transcript = Seq('CACCGGCGTCGGTACCAGAGGGCGTGAGTACCTTGTACTAGTACTCATTGGAATAATGCTCTTAGAAGTCATCTAAAAGTGACAACGCCTGTTTGGTTATGACGTTCACGACGCGTCTTAACAGACTAGCATTAGACCGACGGGTTGAGGCGTCTGGGTTGATACAGCCGTTTGCATCAGTGTATCTAACACTCTGAGGGATAATTGATGAACCGTGTTTTCCGATAGGTATGTACAGTACCACCACGCACGACTAAGGACCATTTTCTGCGTGCGACGGTTAAAATAACCTCAATCACT')
        transcript.id = 'transcript'
        sequence = Seq('TCCCCTTCTAATGGAATCCCCCTCCGAAGGTCGCAGAAGCGGCCACGCCGGAGATACCAGTTCCACGCCTCAGGTTGGACTTGTCACACTTGTACGCGAT')
        sequence.id = 'sequence'
        alignments1 = aligner.align(chromosome, transcript)
        alignment1 = alignments1[0]
        self.assertEqual(alignment1.coordinates.shape[1], 126)
        self.assertEqual(str(alignment1), 'chromosom         5 GCGCCTTGGTTTTGGCTTAACTAGA-------AGCAACC-TGTAAGATTGCCAATTCTTC\n                  0 |||--|.|||---------||.|||-------||-.|||-||||------------||--\ntranscrip         5 GCG--TCGGT---------ACCAGAGGGCGTGAG-TACCTTGTA------------CT--\n\nchromosom        57 AGTCGAAGTAAATCTTCAATGTTTTGGA------CTCTTAG----CGGATATGCGGCTGA\n                 60 ------||||---|-|||-----|||||------|||||||----|----||----||-|\ntranscrip        39 ------AGTA---C-TCA-----TTGGAATAATGCTCTTAGAAGTC----AT----CT-A\n\nchromosom       107 GAAGTACGACA-----TGT---GT----ACATTCATAC--CTGCGT-------GACGGTC\n                120 .||||--||||-----|||---||----||.|||--||--|-||||-------|||--|-\ntranscrip        75 AAAGT--GACAACGCCTGTTTGGTTATGACGTTC--ACGAC-GCGTCTTAACAGAC--T-\n\nchromosom       146 AGCCT----CCCCCGGGACCTCATTG-GGCGAATCTAGGTGTGATA-A-----TTGACA-\n                180 |||.|----||..||||------|||-||||--|||.|||-|||||-|-----|||-||-\ntranscrip       127 AGCATTAGACCGACGGG------TTGAGGCG--TCTGGGT-TGATACAGCCGTTTG-CAT\n\nchromosom       194 CA----CTCTTGGTAAGAAGCACTCT---------TTACCCGATCTCCAAGTACCGACGC\n                240 ||----.|||-------||-||||||---------||----|||------|.||||----\ntranscrip       177 CAGTGTATCT-------AA-CACTCTGAGGGATAATT----GAT------GAACCG----\n\nchromosom       241 CAAGGCCAAGCTCTG-----CGATCTAAAGCTGCCGATCGTAGATCCAAGTCCTCAGCAA\n                300 -------------||-----||||----||.|----||-|||----|-|||.|-||.||-\ntranscrip       215 -------------TGTTTTCCGAT----AGGT----AT-GTA----C-AGTAC-CACCA-\n\nchromosom       296 GCTCGCACGAATACGCAG-------TTCGAAGGCTGGGTGTTGTACGACGGTACGGTTGC\n                360 ---|||||||.||---||-------||------|||.|||-----|||||||--------\ntranscrip       246 ---CGCACGACTA---AGGACCATTTT------CTGCGTG-----CGACGGT--------\n\nchromosom       349 TATAGCACTTTCGCGGTCTCGCTATTTTCAGTTTGACTCACCAGTCAGTATTGTCATCGA\n                420 ||-|--|----------------||----------|---|||--|||--------|||--\ntranscrip       281 TA-A--A----------------AT----------A---ACC--TCA--------ATC--\n\nchromosom       409 CCAACT 415\n                480 ---||| 486\ntranscrip       297 ---ACT 300\n')
        alignments2 = aligner.align(transcript, sequence)
        alignment2 = alignments2[0]
        self.assertEqual(alignment2.coordinates.shape[1], 66)
        self.assertEqual(str(alignment2), 'transcrip         8 TCGGTACCAGAGGGCGTGAGTACCTTGTACTAGTACTCATTGGAATAATGCTCTTAGAAG\n                  0 ||------------|-------||||---|||------|-||||||--------------\nsequence          0 TC------------C-------CCTT---CTA------A-TGGAAT--------------\n\ntranscrip        68 TCATCTAAAAGTGACAACGCCTGTTTGGTTATGACGTTCACGACGCGTCTTAACAGACTA\n                 60 -----------------|.||-------------|--||-|||.|-|||---.||||--|\nsequence         17 -----------------CCCC-------------C--TC-CGAAG-GTC---GCAGA--A\n\ntranscrip       128 GCATTAGACCGACG--GGTTGAGGCGTCTGGGTTGATACAGCCGTTTGCATCAGTGTATC\n                120 ||----|.||-|||--|---||------------|||||------------||||---||\nsequence         38 GC----GGCC-ACGCCG---GA------------GATAC------------CAGT---TC\n\ntranscrip       186 TAACA---CTCTGAGGGATAATTGATGAACCGTGTTTTCCGATAGGTATGTACAGTACCA\n                180 ---||---|||--|||-----|||--||--|-----||----------------||----\nsequence         63 ---CACGCCTC--AGG-----TTG--GA--C-----TT----------------GT----\n\ntranscrip       243 CCACGCACGACTAAGGACCATTTTCTG--CGTGCGA 277\n                240 -----|||-|||-------------||--|--|||| 276\nsequence         84 -----CAC-ACT-------------TGTAC--GCGA  99\n')
        alignment = alignment1.map(alignment2)
        self.assertEqual(alignment.coordinates.shape[1], 78)
        self.assertEqual(str(alignment), 'chromosom        10 TTGGTTTTGGCTTAACTAGAAGCAA-CC-TGTAAGATTGCCAATTCTTCAGTCGAAGTAA\n                  0 |.------------------------||-|---------------||--------|----\nsequence          0 TC-----------------------CCCTT---------------CT--------A----\n\nchromosom        68 ATCTTCAATGTTTTGGACTCTTAGCGGATATGCGGCTGAGAAGTACGACATGTGTA----\n                 60 ------|------||||-------------------------------------------\nsequence         10 ------A------TGGA---------------------------------------ATCC\n\nchromosom       124 --CATTCATAC--CTGCGT----GACGGTCAGCCT--CCCCCG--GGACCTCATTGGGCG\n                120 --|--||---|--.-|-||----||-----|||----||-.||--|---------|----\nsequence         19 CCC--TC---CGAA-G-GTCGCAGA-----AGC--GGCC-ACGCCG---------G----\n\nchromosom       172 AATCTAGGTGT-GATAATTGACA-CAC--TCTTGGTAAGAAGCA---CTCT---TTACCC\n                180 ------------||||--------||---||-----------||---|||----||----\nsequence         51 -----------AGATA-------CCA-GTTC-----------CACGCCTC-AGGTT----\n\nchromosom       222 GATCTCCAAGTACCGACGCCAAGGCCAAGCTCTGCGATCTAAAGCTGCCGATCGTAGATC\n                240 |--------|.--|----------------------------------------------\nsequence         76 G--------GA--C----------------------------------------------\n\nchromosom       282 CAA--GTCCTCAGCAAGCTCGCACGAATACGCAGTTCGAAGGCTG--GGTGTTGTACGA\n                300 -----||--------------|||-|.|---------------||--.--|-----|||\nsequence         80 ---TTGT--------------CAC-ACT---------------TGTAC--G-----CGA\n\nchromosom       337\n                359\nsequence         99\n')
        line = format(alignment, 'psl')
        self.assertEqual(line, '61\t6\t0\t0\t14\t32\t28\t260\t+\tsequence\t100\t0\t99\tchromosome\t440\t10\t337\t35\t2,2,1,2,1,1,4,1,2,1,1,1,2,2,3,2,3,1,1,4,2,2,2,3,2,1,2,1,2,3,3,2,1,1,3,\t0,3,6,7,9,10,11,21,22,24,27,28,29,35,37,42,44,49,50,52,57,61,63,68,74,76,77,79,82,84,87,90,94,95,96,\t10,35,37,53,63,74,81,124,127,132,133,135,137,139,146,151,154,157,167,183,194,197,210,212,216,222,231,235,285,301,305,323,325,328,334,\n')

def map_check(alignment1, alignment2):
    if False:
        print('Hello World!')
    line1 = format(alignment1, 'psl')
    handle = open('transcript.psl', 'w')
    handle.write(line1)
    handle.close()
    line2 = format(alignment2, 'psl')
    handle = open('sequence.psl', 'w')
    handle.write(line2)
    handle.close()
    stdout = os.popen('pslMap sequence.psl transcript.psl stdout')
    line = stdout.read()
    os.remove('transcript.psl')
    os.remove('sequence.psl')
    return line

def test_random(aligner, nBlocks1=1, nBlocks2=1, strand1='+', strand2='+'):
    if False:
        i = 10
        return i + 15
    chromosome = ''.join(['ACGT'[random.randint(0, 3)] for i in range(1000)])
    nBlocks = nBlocks1
    transcript = ''
    position = 0
    for i in range(nBlocks):
        position += random.randint(60, 80)
        blockSize = random.randint(60, 80)
        transcript += chromosome[position:position + blockSize]
        position += blockSize
    nBlocks = nBlocks2
    sequence = ''
    position = 0
    for i in range(nBlocks):
        position += random.randint(20, 40)
        blockSize = random.randint(20, 40)
        sequence += transcript[position:position + blockSize]
        position += blockSize
    chromosome = Seq(chromosome)
    transcript = Seq(transcript)
    sequence = Seq(sequence)
    if strand1 == '-':
        chromosome = chromosome.reverse_complement()
    if strand2 == '-':
        sequence = sequence.reverse_complement()
    chromosome.id = 'chromosome'
    transcript.id = 'transcript'
    sequence.id = 'sequence'
    alignments1 = aligner.align(chromosome, transcript, strand=strand1)
    alignment1 = alignments1[0]
    alignments2 = aligner.align(transcript, sequence, strand=strand2)
    alignment2 = alignments2[0]
    alignment = alignment1.map(alignment2)
    line_check = map_check(alignment1, alignment2)
    line = format(alignment, 'psl')
    assert line == line_check
    print('Randomized test %d, %d, %s, %s OK' % (nBlocks1, nBlocks2, strand1, strand2))

def test_random_sequences(aligner, strand1='+', strand2='+'):
    if False:
        while True:
            i = 10
    chromosome = ''.join(['ACGT'[random.randint(0, 3)] for i in range(1000)])
    transcript = ''.join(['ACGT'[random.randint(0, 3)] for i in range(300)])
    sequence = ''.join(['ACGT'[random.randint(0, 3)] for i in range(100)])
    chromosome = Seq(chromosome)
    transcript = Seq(transcript)
    sequence = Seq(sequence)
    chromosome.id = 'chromosome'
    transcript.id = 'transcript'
    sequence.id = 'sequence'
    alignments = aligner.align(chromosome, transcript, strand=strand1)
    alignment1 = alignments[0]
    alignments = aligner.align(transcript, sequence, strand=strand2)
    alignment2 = alignments[0]
    line_check = map_check(alignment1, alignment2)
    alignment = alignment1.map(alignment2)
    line_check = line_check.split()
    line = format(alignment, 'psl')
    line = line.split()
    assert line[8:] == line_check[8:]
    line1 = format(alignment1, 'psl')
    words = line1.split()
    nBlocks1 = int(words[17])
    line2 = format(alignment2, 'psl')
    words = line2.split()
    nBlocks2 = int(words[17])
    print('Randomized sequence test %d, %d, %s, %s OK' % (nBlocks1, nBlocks2, strand1, strand2))

def perform_randomized_tests(n=1000):
    if False:
        print('Hello World!')
    'Perform randomized tests and compare to pslMap.\n\n    Run this function to perform 8 x n mappings for alignments of randomly\n    generated sequences, get the alignment in PSL format, and compare the\n    result to that of pslMap.\n    '
    aligner = PairwiseAligner()
    aligner.internal_open_gap_score = -1
    aligner.internal_extend_gap_score = -0.0
    aligner.match_score = +1
    aligner.mismatch_score = -1
    aligner.mode = 'local'
    for i in range(n):
        nBlocks1 = random.randint(1, 10)
        nBlocks2 = random.randint(1, 10)
        test_random(aligner, nBlocks1, nBlocks2, '+', '+')
        test_random(aligner, nBlocks1, nBlocks2, '+', '-')
        test_random(aligner, nBlocks1, nBlocks2, '-', '+')
        test_random(aligner, nBlocks1, nBlocks2, '-', '-')
        test_random_sequences(aligner, '+', '+')
        test_random_sequences(aligner, '+', '-')
        test_random_sequences(aligner, '-', '+')
        test_random_sequences(aligner, '-', '-')

class TestZeroGaps(unittest.TestCase):

    def test1(self):
        if False:
            for i in range(10):
                print('nop')
        coordinates = np.array([[0, 3, 6, 9], [0, 3, 6, 9]])
        sequences = [SeqRecord(Seq(None, 9), id='genome'), SeqRecord(Seq(None, 9), id='mRNA')]
        alignment1 = Alignment(sequences, coordinates)
        coordinates = np.array([[0, 9], [0, 9]])
        sequences = [SeqRecord(Seq(None, 9), id='mRNA'), SeqRecord(Seq(None, 9), id='tag')]
        alignment2 = Alignment(sequences, coordinates)
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[0, 3, 6, 9], [0, 3, 6, 9]])))

    def test2(self):
        if False:
            while True:
                i = 10
        coordinates = np.array([[0, 69], [0, 69]])
        sequences = [SeqRecord(Seq(None, 69), id='genome'), SeqRecord(Seq(None, 69), id='mRNA')]
        alignment1 = Alignment(sequences, coordinates)
        coordinates = np.array([[0, 24, 24, 69], [0, 24, 24, 69]])
        sequences = [SeqRecord(Seq(None, 69), id='mRNA'), SeqRecord(Seq(None, 68), id='tag')]
        alignment2 = Alignment(sequences, coordinates)
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[0, 24, 69], [0, 24, 69]])))

    def test3(self):
        if False:
            return 10
        coordinates = np.array([[0, 69], [0, 69]])
        sequences = [SeqRecord(Seq(None, 69), id='genome'), SeqRecord(Seq(None, 69), id='mRNA')]
        alignment1 = Alignment(sequences, coordinates)
        coordinates = np.array([[0, 24, 24, 69], [0, 24, 23, 68]])
        sequences = [SeqRecord(Seq(None, 69), id='mRNA'), SeqRecord(Seq(None, 68), id='tag')]
        alignment2 = Alignment(sequences, coordinates)
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[0, 24, 24, 69], [0, 24, 23, 68]])))

    def test4(self):
        if False:
            i = 10
            return i + 15
        coordinates = np.array([[0, 210], [0, 210]])
        sequences = [SeqRecord(Seq(None, 210), id='genome'), SeqRecord(Seq(None, 210), id='mRNA')]
        alignment1 = Alignment(sequences, coordinates)
        coordinates = np.array([[0, 18, 18, 102, 102, 210], [0, 18, 17, 101, 100, 208]])
        sequences = [SeqRecord(Seq(None, 210), id='mRNA'), SeqRecord(Seq(None, 208), id='tag')]
        alignment2 = Alignment(sequences, coordinates)
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[0, 18, 18, 102, 102, 210], [0, 18, 17, 101, 100, 208]])))

    def test5(self):
        if False:
            for i in range(10):
                print('nop')
        coordinates = np.array([[0, 210], [0, 210]])
        sequences = [SeqRecord(Seq(None, 210), id='genome'), SeqRecord(Seq(None, 210), id='mRNA')]
        alignment1 = Alignment(sequences, coordinates)
        coordinates = np.array([[0, 51, 51, 210], [0, 51, 49, 208]])
        sequences = [SeqRecord(Seq(None, 210), id='mRNA'), SeqRecord(Seq(None, 208), id='tag')]
        alignment2 = Alignment(sequences, coordinates)
        alignment = alignment1.map(alignment2)
        self.assertTrue(np.array_equal(alignment.coordinates, np.array([[0, 51, 51, 210], [0, 51, 49, 208]])))

class TestLiftOver(unittest.TestCase):

    def test_chimp(self):
        if False:
            while True:
                i = 10
        chain = Align.read('Blat/panTro5ToPanTro6.over.chain', 'chain')
        alignment = Align.read('Blat/est.panTro5.psl', 'psl')
        self.assertEqual(chain.target.id, alignment.target.id)
        self.assertEqual(len(chain.target.seq), len(alignment.target.seq))
        chain = chain[::-1]
        record = SeqIO.read('Blat/est.fa', 'fasta')
        self.assertEqual(record.id, alignment.query.id)
        self.assertEqual(len(record.seq), len(alignment.query.seq))
        alignment.query = record.seq
        record = SeqIO.read('Blat/panTro5.fa', 'fasta')
        (chromosome, start_end) = record.id.split(':')
        (start, end) = start_end.split('-')
        start = int(start)
        end = int(end)
        data = {start: str(record.seq)}
        length = len(alignment.target.seq)
        seq = Seq(data, length=length)
        record = SeqRecord(seq, id='chr1')
        alignment.target = record
        text = '^chr1      122835789 AGAGATTATTTTGCAGAGGATGATGGGGAGATGGTACCCAGAACGAGTCACACAGCAGGT\n                  0 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::--\nquery            32 AGAGATTATTTTGCAGAGGATGATGGGGAGATGGTACCCAGAACGAGTCACACAGCAG--\n\nchr1      122835849 AAGGATGCTGTGGGCCTTGCCTTGTTAAATTCTTTGtttcttttgtttattcatttggtt\n                 60 ------------------------------------------------------------\nquery            90 ------------------------------------------------------------\n((.|\n)*)\nchr1      122840889 TTGTATTTTGCAGAAACTGAATTCTGCTGGAATGTGCCAGTTAGAATGATCCTAGTGCTG\n               5100 ------------------------------------------------------------\nquery            90 ------------------------------------------------------------\n\nchr1      122840949 TTATTATATAAACCTTTTTTGTTGTTGTTCTGTTTCATTGACAGCTTTTCTTAGTGACAC\n               5160 --------------------------------------------::::::::::::::::\nquery            90 --------------------------------------------CTTTTCTTAGTGACAC\n\nchr1      122841009 TAAAGATCGAGGCCCTCCAGTGCAGTCACAGATCTGGAGAAGTGGTGAAAAGGTCCCGTT\n               5220 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::X:\nquery           106 TAAAGATCGAGGCCCTCCAGTGCAGTCACAGATCTGGAGAAGTGGTGAAAAGGTCCCGNT\n\nchr1      122841069 TGTGCAGACATATTCCTTGAGAGCATTTGAGAAACCCCCTCAGGTACAGACCCAGGCTCT\n               5280 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\nquery           166 TGTGCAGACATATTCCTTGAGAGCATTTGAGAAACCCCCTCAGGTACAGACCCAGGCTCT\n\nchr1      122841129 TCGAGACTTTGAGAAGGTAAGTCATGTGAGTGGATAATTGTTATCCCAATTAGAAGCAGT\n               5340 ::::::::::::::::--------------------------------------------\nquery           226 TCGAGACTTTGAGAAG--------------------------------------------\n\nchr1      122841189 ACTATGGAATAGTGATGCCTGATAAAAATATGACCCATGGATTGGTCCGGATTATGGATG\n               5400 ------------------------------------------------------------\nquery           242 ------------------------------------------------------------\n((.|\n)*)\nchr1      122907129 GTTCTTGGGTTGAGGGGGCAATCGGGCACGCTCCTCCCCATGGGTTGCCCATCATGTCTA\n              71340 ------------------------------------------------------------\nquery           242 ------------------------------------------------------------\n\nchr1      122907189 ATGGATATCGCACTCTGTCCCAGCACCTCAATGACCTGAAGAAGGAGAACTTCAGCCTCA\n              71400 -----------------------:::::::::::::::::::::::::::::::::::::\nquery           242 -----------------------CACCTCAATGACCTGAAGAAGGAGAACTTCAGCCTCA\n\nchr1      122907249 AGCTGCGCATCTACTTCCTGGAGGAGCGCATGCAACAGAAGTATGAGGCCAGCCGGGAGG\n              71460 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\nquery           279 AGCTGCGCATCTACTTCCTGGAGGAGCGCATGCAACAGAAGTATGAGGCCAGCCGGGAGG\n\nchr1      122907309 ACATC 122907314\n              71520 :::::     71525\nquery           339 ACATC       344\n$'
        self.assertRegex(str(alignment).replace('|', ':').replace('.', 'X'), text)
        lifted_alignment = chain.map(alignment)
        self.assertTrue(np.array_equal(lifted_alignment.coordinates, np.array([[111982717, 111982775, 111987921, 111988073, 112009200, 112009302], [32, 90, 90, 242, 242, 344]])))
        record = SeqIO.read('Blat/panTro6.fa', 'fasta')
        (chromosome, start_end) = record.id.split(':')
        (start, end) = start_end.split('-')
        start = int(start)
        end = int(end)
        data = {start: str(record.seq)}
        length = len(alignment.target.seq)
        seq = Seq(data, length=length)
        record = SeqRecord(seq, id='chr1')
        lifted_alignment.target = record
        text = '^chr1      111982717 AGAGATTATTTTGCAGAGGATGATGGGGAGATGGTACCCAGAACGAGTCACACAGCAGGT\n                  0 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::--\nquery            32 AGAGATTATTTTGCAGAGGATGATGGGGAGATGGTACCCAGAACGAGTCACACAGCAG--\n\nchr1      111982777 AAGGATGCTGTGGGCCTTGCCTTGTTAAATTCTTTGtttcttttgtttattcatttggtt\n                 60 ------------------------------------------------------------\nquery            90 ------------------------------------------------------------\n((.|\n)*)\nchr1      111987817 TTGTATTTTGCAGAAACTGAATTCTGCTGGAATGTGCCAGTTAGAATGATCCTAGTGCTG\n               5100 ------------------------------------------------------------\nquery            90 ------------------------------------------------------------\n\nchr1      111987877 TTATTATATAAACCTTTTTTGTTGTTGTTCTGTTTCATTGACAGCTTTTCTTAGTGACAC\n               5160 --------------------------------------------::::::::::::::::\nquery            90 --------------------------------------------CTTTTCTTAGTGACAC\n\nchr1      111987937 TAAAGATCGAGGCCCTCCAGTGCAGTCACAGATCTGGAGAAGTGGTGAAAAGGTCCCGTT\n               5220 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::X:\nquery           106 TAAAGATCGAGGCCCTCCAGTGCAGTCACAGATCTGGAGAAGTGGTGAAAAGGTCCCGNT\n\nchr1      111987997 TGTGCAGACATATTCCTTGAGAGCATTTGAGAAACCCCCTCAGGTACAGACCCAGGCTCT\n               5280 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\nquery           166 TGTGCAGACATATTCCTTGAGAGCATTTGAGAAACCCCCTCAGGTACAGACCCAGGCTCT\n\nchr1      111988057 TCGAGACTTTGAGAAGGTAAGTCATGTGAGTGGATAATTGTTATCCCAATTAGAAGCAGT\n               5340 ::::::::::::::::--------------------------------------------\nquery           226 TCGAGACTTTGAGAAG--------------------------------------------\n\nchr1      111988117 ACTATGGAATAGTGATGCCTGATAAAAATATGACCCATGGATTGGTCCGGATTATGGATG\n               5400 ------------------------------------------------------------\nquery           242 ------------------------------------------------------------\n((.|\n)*)\nchr1      112009117 GTTCTTGGGTTGAGGGGGCAATCGGGCACGCTCCTCCCCATGGGTTGCCCATCATGTCTA\n              26400 ------------------------------------------------------------\nquery           242 ------------------------------------------------------------\n\nchr1      112009177 ATGGATATCGCACTCTGTCCCAGCACCTCAATGACCTGAAGAAGGAGAACTTCAGCCTCA\n              26460 -----------------------:::::::::::::::::::::::::::::::::::::\nquery           242 -----------------------CACCTCAATGACCTGAAGAAGGAGAACTTCAGCCTCA\n\nchr1      112009237 AGCTGCGCATCTACTTCCTGGAGGAGCGCATGCAACAGAAGTATGAGGCCAGCCGGGAGG\n              26520 ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::\nquery           279 AGCTGCGCATCTACTTCCTGGAGGAGCGCATGCAACAGAAGTATGAGGCCAGCCGGGAGG\n\nchr1      112009297 ACATC 112009302\n              26580 :::::     26585\nquery           339 ACATC       344\n$'
        self.assertRegex(str(lifted_alignment).replace('|', ':').replace('.', 'X'), text)
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)