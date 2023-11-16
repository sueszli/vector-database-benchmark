"""Tests for the Alignment class in Bio.Align."""
import os
import unittest
from io import StringIO
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install numpy if you want to use Bio.Align.') from None
from Bio import Align, SeqIO
from Bio.Seq import Seq, reverse_complement, translate
from Bio.SeqRecord import SeqRecord
from Bio.SeqUtils import gc_fraction

class TestAlignment(unittest.TestCase):

    def test_empty_alignment(self):
        if False:
            while True:
                i = 10
        alignment = Align.Alignment([])
        self.assertEqual(repr(alignment), '<Alignment object (0 rows x 0 columns) at 0x%x>' % id(alignment))
        self.assertEqual(len(alignment), 0)
        self.assertEqual(len(alignment.sequences), 0)
        self.assertEqual(alignment.shape, (0, 0))
        self.assertEqual(alignment.coordinates.shape, (0, 0))

class TestPairwiseAlignment(unittest.TestCase):
    target = 'AACCGGGACCG'
    query = 'ACGGAAC'
    query_rc = reverse_complement(query)
    forward_coordinates = np.array([[0, 1, 2, 3, 4, 6, 7, 8, 8, 9, 11], [0, 1, 1, 2, 2, 4, 4, 5, 6, 7, 7]])
    reverse_coordinates = np.array([[0, 1, 2, 3, 4, 6, 7, 8, 8, 9, 11], [7, 6, 6, 5, 5, 3, 3, 2, 1, 0, 0]])

    def check_indexing_slicing(self, alignment, cls, strand):
        if False:
            return 10
        msg = '%s, %s strand' % (cls.__name__, strand)
        self.assertEqual(repr(alignment), '<Alignment object (2 rows x 12 columns) at 0x%x>' % id(alignment))
        if strand == 'forward':
            self.assertEqual(str(alignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             0 A-C-GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(alignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             7 A-C-GG-AAC--  0\n', msg=msg)
        frequencies = alignment.frequencies
        self.assertEqual(list(frequencies.keys()), ['A', 'C', 'G'])
        self.assertTrue(np.array_equal(frequencies['A'], np.array([2, 1, 0, 0, 0, 0, 0, 2, 1, 0, 0, 0])))
        self.assertTrue(np.array_equal(frequencies['C'], np.array([0, 0, 2, 1, 0, 0, 0, 0, 0, 2, 1, 0])))
        self.assertTrue(np.array_equal(frequencies['G'], np.array([0, 0, 0, 0, 2, 2, 1, 0, 0, 0, 0, 1])))
        self.assertAlmostEqual(alignment.score, 6.0)
        self.assertEqual(len(alignment), 2)
        self.assertEqual(alignment.shape, (2, 12))
        self.assertIsInstance(alignment.sequences[0], cls)
        self.assertIsInstance(alignment.sequences[1], cls)
        self.assertEqual(alignment[0], 'AACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1], 'A-C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[-2], 'AACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[-1], 'A-C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 0], 'A', msg=msg)
        self.assertEqual(alignment[0, 1], 'A', msg=msg)
        self.assertEqual(alignment[0, 2], 'C', msg=msg)
        self.assertEqual(alignment[0, 3], 'C', msg=msg)
        self.assertEqual(alignment[0, 4], 'G', msg=msg)
        self.assertEqual(alignment[0, 5], 'G', msg=msg)
        self.assertEqual(alignment[0, 6], 'G', msg=msg)
        self.assertEqual(alignment[0, 7], 'A', msg=msg)
        self.assertEqual(alignment[0, 8], '-', msg=msg)
        self.assertEqual(alignment[0, 9], 'C', msg=msg)
        self.assertEqual(alignment[0, 10], 'C', msg=msg)
        self.assertEqual(alignment[0, 11], 'G', msg=msg)
        self.assertEqual(alignment[1, 0], 'A', msg=msg)
        self.assertEqual(alignment[1, 1], '-', msg=msg)
        self.assertEqual(alignment[1, 2], 'C', msg=msg)
        self.assertEqual(alignment[1, 3], '-', msg=msg)
        self.assertEqual(alignment[1, 4], 'G', msg=msg)
        self.assertEqual(alignment[1, 5], 'G', msg=msg)
        self.assertEqual(alignment[1, 6], '-', msg=msg)
        self.assertEqual(alignment[1, 7], 'A', msg=msg)
        self.assertEqual(alignment[1, 8], 'A', msg=msg)
        self.assertEqual(alignment[1, 9], 'C', msg=msg)
        self.assertEqual(alignment[1, 10], '-', msg=msg)
        self.assertEqual(alignment[1, 11], '-', msg=msg)
        self.assertEqual(alignment[0, :], 'AACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, :], 'A-C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[-2, :], 'AACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[-1, :], 'A-C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 1:2], 'A', msg=msg)
        self.assertEqual(alignment[1, 1:2], '-', msg=msg)
        self.assertEqual(alignment[0, 4:5], 'G', msg=msg)
        self.assertEqual(alignment[1, 4:5], 'G', msg=msg)
        self.assertEqual(alignment[0, 10:11], 'C', msg=msg)
        self.assertEqual(alignment[1, 10:11], '-', msg=msg)
        self.assertEqual(alignment[:, 0], 'AA', msg=msg)
        self.assertEqual(alignment[:, 1], 'A-', msg=msg)
        self.assertEqual(alignment[:, 2], 'CC', msg=msg)
        self.assertEqual(alignment[:, 3], 'C-', msg=msg)
        self.assertEqual(alignment[:, 4], 'GG', msg=msg)
        self.assertEqual(alignment[:, 5], 'GG', msg=msg)
        self.assertEqual(alignment[:, 6], 'G-', msg=msg)
        self.assertEqual(alignment[:, 7], 'AA', msg=msg)
        self.assertEqual(alignment[:, 8], '-A', msg=msg)
        self.assertEqual(alignment[:, 9], 'CC', msg=msg)
        self.assertEqual(alignment[:, 10], 'C-', msg=msg)
        self.assertEqual(alignment[:, 11], 'G-', msg=msg)
        self.assertEqual(alignment[:, -12], 'AA', msg=msg)
        self.assertEqual(alignment[:, -11], 'A-', msg=msg)
        self.assertEqual(alignment[:, -10], 'CC', msg=msg)
        self.assertEqual(alignment[:, -9], 'C-', msg=msg)
        self.assertEqual(alignment[:, -8], 'GG', msg=msg)
        self.assertEqual(alignment[:, -7], 'GG', msg=msg)
        self.assertEqual(alignment[:, -6], 'G-', msg=msg)
        self.assertEqual(alignment[:, -5], 'AA', msg=msg)
        self.assertEqual(alignment[:, -4], '-A', msg=msg)
        self.assertEqual(alignment[:, -3], 'CC', msg=msg)
        self.assertEqual(alignment[:, -2], 'C-', msg=msg)
        self.assertEqual(alignment[:, -1], 'G-', msg=msg)
        self.assertEqual(alignment[1, range(1, 12, 2)], '--GAC-', msg=msg)
        self.assertEqual(alignment[0, (1, 4, 9)], 'AGC', msg=msg)
        self.assertEqual(alignment[1, (1, 4, 9)], '-GC', msg=msg)
        self.assertEqual(alignment[0, range(0, 12, 2)], 'ACGG-C', msg=msg)
        subalignment = alignment[:, :]
        self.assertAlmostEqual(subalignment.score, 6.0, msg=msg)
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             0 A-C-GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             7 A-C-GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        self.assertEqual(alignment[0, 0:12], 'AACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, 0:12], 'A-C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 0:], 'AACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, 0:], 'A-C-GG-AAC--', msg=msg)
        subalignment = alignment[:, 0:]
        self.assertAlmostEqual(subalignment.score, 6.0, msg=msg)
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             0 A-C-GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             7 A-C-GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        self.assertEqual(alignment[0, :12], 'AACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, :12], 'A-C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 1:], 'ACCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, 1:], '-C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 2:], 'CCGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, 2:], 'C-GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 3:], 'CGGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, 3:], '-GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 4:], 'GGGA-CCG', msg=msg)
        self.assertEqual(alignment[1, 4:], 'GG-AAC--', msg=msg)
        self.assertEqual(alignment[0, 5:], 'GGA-CCG', msg=msg)
        self.assertEqual(alignment[1, 5:], 'G-AAC--', msg=msg)
        self.assertEqual(alignment[0, 6:], 'GA-CCG', msg=msg)
        self.assertEqual(alignment[1, 6:], '-AAC--', msg=msg)
        self.assertEqual(alignment[0, 7:], 'A-CCG', msg=msg)
        self.assertEqual(alignment[1, 7:], 'AAC--', msg=msg)
        self.assertEqual(alignment[0, 8:], '-CCG', msg=msg)
        self.assertEqual(alignment[1, 8:], 'AC--', msg=msg)
        self.assertEqual(alignment[0, 9:], 'CCG', msg=msg)
        self.assertEqual(alignment[1, 9:], 'C--', msg=msg)
        self.assertEqual(alignment[0, 10:], 'CG', msg=msg)
        self.assertEqual(alignment[1, 10:], '--', msg=msg)
        self.assertEqual(alignment[0, 11:], 'G', msg=msg)
        self.assertEqual(alignment[1, 11:], '-', msg=msg)
        self.assertEqual(alignment[0, 12:], '', msg=msg)
        self.assertEqual(alignment[1, 12:], '', msg=msg)
        self.assertEqual(alignment[0, :-1], 'AACCGGGA-CC', msg=msg)
        self.assertEqual(alignment[1, :-1], 'A-C-GG-AAC-', msg=msg)
        self.assertEqual(alignment[0, :-2], 'AACCGGGA-C', msg=msg)
        self.assertEqual(alignment[1, :-2], 'A-C-GG-AAC', msg=msg)
        self.assertEqual(alignment[0, :-3], 'AACCGGGA-', msg=msg)
        self.assertEqual(alignment[1, :-3], 'A-C-GG-AA', msg=msg)
        self.assertEqual(alignment[0, 1:-1], 'ACCGGGA-CC', msg=msg)
        self.assertEqual(alignment[1, 1:-1], '-C-GG-AAC-', msg=msg)
        self.assertEqual(alignment[0, 1:-2], 'ACCGGGA-C', msg=msg)
        self.assertEqual(alignment[1, 1:-2], '-C-GG-AAC', msg=msg)
        self.assertEqual(alignment[0, 2:-1], 'CCGGGA-CC', msg=msg)
        self.assertEqual(alignment[1, 2:-1], 'C-GG-AAC-', msg=msg)
        self.assertEqual(alignment[0, 2:-2], 'CCGGGA-C', msg=msg)
        self.assertEqual(alignment[1, 2:-2], 'C-GG-AAC', msg=msg)
        subalignment = alignment[:, :12]
        self.assertAlmostEqual(subalignment.score, 6.0, msg=msg)
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             0 A-C-GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             7 A-C-GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 0:12]
        self.assertAlmostEqual(alignment.score, 6.0, msg=msg)
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             0 A-C-GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             7 A-C-GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 1:]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            1 ACCGGGA-CCG 11\n                  0 -|-||-|-|-- 11\nquery             1 -C-GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            1 ACCGGGA-CCG 11\n                  0 -|-||-|-|-- 11\nquery             6 -C-GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 2:]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            2 CCGGGA-CCG 11\n                  0 |-||-|-|-- 10\nquery             1 C-GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            2 CCGGGA-CCG 11\n                  0 |-||-|-|-- 10\nquery             6 C-GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 3:]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            3 CGGGA-CCG 11\n                  0 -||-|-|--  9\nquery             2 -GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            3 CGGGA-CCG 11\n                  0 -||-|-|--  9\nquery             5 -GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 4:]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            4 GGGA-CCG 11\n                  0 ||-|-|--  8\nquery             2 GG-AAC--  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            4 GGGA-CCG 11\n                  0 ||-|-|--  8\nquery             5 GG-AAC--  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, :-1]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CC 10\n                  0 |-|-||-|-|- 11\nquery             0 A-C-GG-AAC-  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-CC 10\n                  0 |-|-||-|-|- 11\nquery             7 A-C-GG-AAC-  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, :-2]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-C  9\n                  0 |-|-||-|-| 10\nquery             0 A-C-GG-AAC  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA-C  9\n                  0 |-|-||-|-| 10\nquery             7 A-C-GG-AAC  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, :-3]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA- 8\n                  0 |-|-||-|- 9\nquery             0 A-C-GG-AA 6\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            0 AACCGGGA- 8\n                  0 |-|-||-|- 9\nquery             7 A-C-GG-AA 1\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 1:-1]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            1 ACCGGGA-CC 10\n                  0 -|-||-|-|- 10\nquery             1 -C-GG-AAC-  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            1 ACCGGGA-CC 10\n                  0 -|-||-|-|- 10\nquery             6 -C-GG-AAC-  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 1:-2]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            1 ACCGGGA-C 9\n                  0 -|-||-|-| 9\nquery             1 -C-GG-AAC 7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            1 ACCGGGA-C 9\n                  0 -|-||-|-| 9\nquery             6 -C-GG-AAC 0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 2:-1]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            2 CCGGGA-CC 10\n                  0 |-||-|-|-  9\nquery             1 C-GG-AAC-  7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            2 CCGGGA-CC 10\n                  0 |-||-|-|-  9\nquery             6 C-GG-AAC-  0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, 2:-2]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'target            2 CCGGGA-C 9\n                  0 |-||-|-| 8\nquery             1 C-GG-AAC 7\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'target            2 CCGGGA-C 9\n                  0 |-||-|-| 8\nquery             6 C-GG-AAC 0\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, ::2]
        self.assertEqual(str(subalignment), 'target            0 ACGG-C 5\n                  0 |||--- 6\nquery             0 ACG-A- 4\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, range(0, 12, 2)]
        self.assertEqual(str(subalignment), 'target            0 ACGG-C 5\n                  0 |||--- 6\nquery             0 ACG-A- 4\n', msg=msg)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)
        subalignment = alignment[:, (1, 8, 5)]
        self.assertEqual(str(subalignment), 'target            0 A-G 2\n                  0 --| 3\nquery             0 -AG 2\n', msg=msg)
        subalignment = alignment[:1]
        self.assertEqual(len(subalignment.sequences), 1)
        sequence = subalignment.sequences[0]
        self.assertIsInstance(sequence, cls)
        try:
            sequence = sequence.seq
        except AttributeError:
            pass
        self.assertEqual(sequence, 'AACCGGGACCG')
        self.assertTrue(np.array_equal(subalignment.coordinates, np.array([[0, 1, 2, 3, 4, 6, 7, 8, 8, 9, 11]])))
        frequencies = subalignment.frequencies
        self.assertEqual(list(frequencies.keys()), ['A', 'C', 'G'])
        self.assertTrue(np.array_equal(frequencies['A'], np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])))
        self.assertTrue(np.array_equal(frequencies['C'], np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0])))
        self.assertTrue(np.array_equal(frequencies['G'], np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1])))
        subalignment = alignment[:1, :]
        self.assertEqual(len(subalignment.sequences), 1)
        sequence = subalignment.sequences[0]
        self.assertIsInstance(sequence, cls)
        try:
            sequence = sequence.seq
        except AttributeError:
            pass
        self.assertEqual(sequence, 'AACCGGGACCG')
        self.assertTrue(np.array_equal(subalignment.coordinates, np.array([[0, 1, 2, 3, 4, 6, 7, 8, 8, 9, 11]])))
        frequencies = subalignment.frequencies
        self.assertEqual(list(frequencies.keys()), ['A', 'C', 'G'])
        self.assertTrue(np.array_equal(frequencies['A'], np.array([1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0])))
        self.assertTrue(np.array_equal(frequencies['C'], np.array([0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0])))
        self.assertTrue(np.array_equal(frequencies['G'], np.array([0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1])))
        subalignment = alignment[:]
        self.assertEqual(alignment, subalignment)
        self.assertIsInstance(subalignment.sequences[0], cls)
        self.assertIsInstance(subalignment.sequences[1], cls)

    def test_indexing_slicing(self):
        if False:
            for i in range(10):
                print('nop')
        sequences = (self.target, self.query)
        alignment = Align.Alignment(sequences, self.forward_coordinates)
        alignment.score = 6.0
        self.check_indexing_slicing(alignment, str, 'forward')
        sequences = (self.target, self.query_rc)
        alignment = Align.Alignment(sequences, self.reverse_coordinates)
        alignment.score = 6.0
        self.check_indexing_slicing(alignment, str, 'reverse')
        target = Seq(self.target)
        query = Seq(self.query)
        query_rc = Seq(self.query_rc)
        sequences = (target, query)
        alignment = Align.Alignment(sequences, self.forward_coordinates)
        alignment.score = 6.0
        self.check_indexing_slicing(alignment, Seq, 'forward')
        sequences = (target, query_rc)
        alignment = Align.Alignment(sequences, self.reverse_coordinates)
        alignment.score = 6.0
        self.check_indexing_slicing(alignment, Seq, 'reverse')
        target = SeqRecord(target, id=None)
        query = SeqRecord(query, id=None)
        query_rc = SeqRecord(query_rc, id=None)
        sequences = (target, query)
        alignment = Align.Alignment(sequences, self.forward_coordinates)
        alignment.score = 6.0
        self.check_indexing_slicing(alignment, SeqRecord, 'forward')
        sequences = (target, query_rc)
        alignment = Align.Alignment(sequences, self.reverse_coordinates)
        alignment.score = 6.0
        self.check_indexing_slicing(alignment, SeqRecord, 'reverse')

    def test_aligned_indices(self):
        if False:
            i = 10
            return i + 15
        sequences = (self.target, self.query)
        alignment = Align.Alignment(sequences, self.forward_coordinates)
        self.assertEqual(str(alignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             0 A-C-GG-AAC--  7\n')
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3], [4, 6], [7, 8], [8, 9]], [[0, 1], [1, 2], [2, 4], [4, 5], [6, 7]]])))
        self.assertTrue(np.array_equal(alignment.indices, np.array([[0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10], [0, -1, 1, -1, 2, 3, -1, 4, 5, 6, -1, -1]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([0, 2, 4, 5, 7, 8, 9])))
        alignment = Align.Alignment(sequences, self.forward_coordinates[:, 1:])
        self.assertEqual(str(alignment), 'target            1 ACCGGGA-CCG 11\n                  0 -|-||-|-|-- 11\nquery             1 -C-GG-AAC--  7\n')
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 3], [4, 6], [7, 8], [8, 9]], [[1, 2], [2, 4], [4, 5], [6, 7]]])))
        self.assertTrue(np.array_equal(alignment.indices, np.array([[1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10], [-1, 1, -1, 2, 3, -1, 4, 5, 6, -1, -1]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([-1, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([-1, 1, 3, 4, 6, 7, 8])))
        alignment = Align.Alignment(sequences, self.forward_coordinates[:, :-1])
        self.assertEqual(str(alignment), 'target            0 AACCGGGA-C  9\n                  0 |-|-||-|-| 10\nquery             0 A-C-GG-AAC  7\n')
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3], [4, 6], [7, 8], [8, 9]], [[0, 1], [1, 2], [2, 4], [4, 5], [6, 7]]])))
        self.assertTrue(np.array_equal(alignment.indices, np.array([[0, 1, 2, 3, 4, 5, 6, 7, -1, 8], [0, -1, 1, -1, 2, 3, -1, 4, 5, 6]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([0, 2, 4, 5, 7, 8, 9])))
        alignment = Align.Alignment(sequences, self.forward_coordinates[:, 1:-1])
        self.assertEqual(str(alignment), 'target            1 ACCGGGA-C 9\n                  0 -|-||-|-| 9\nquery             1 -C-GG-AAC 7\n')
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 3], [4, 6], [7, 8], [8, 9]], [[1, 2], [2, 4], [4, 5], [6, 7]]])))
        self.assertTrue(np.array_equal(alignment.indices, np.array([[1, 2, 3, 4, 5, 6, 7, -1, 8], [-1, 1, -1, 2, 3, -1, 4, 5, 6]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([-1, 0, 1, 2, 3, 4, 5, 6, 8, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([-1, 1, 3, 4, 6, 7, 8])))
        sequences = (self.target, self.query_rc)
        alignment = Align.Alignment(sequences, self.reverse_coordinates)
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3], [4, 6], [7, 8], [8, 9]], [[7, 6], [6, 5], [5, 3], [3, 2], [1, 0]]])))
        self.assertEqual(str(alignment), 'target            0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nquery             7 A-C-GG-AAC--  0\n')
        self.assertTrue(np.array_equal(alignment.indices, np.array([[0, 1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10], [6, -1, 5, -1, 4, 3, -1, 2, 1, 0, -1, -1]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([9, 8, 7, 5, 4, 2, 0])))
        alignment = Align.Alignment(sequences, self.reverse_coordinates[:, 1:])
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 3], [4, 6], [7, 8], [8, 9]], [[6, 5], [5, 3], [3, 2], [1, 0]]])))
        self.assertEqual(str(alignment), 'target            1 ACCGGGA-CCG 11\n                  0 -|-||-|-|-- 11\nquery             6 -C-GG-AAC--  0\n')
        self.assertTrue(np.array_equal(alignment.indices, np.array([[1, 2, 3, 4, 5, 6, 7, -1, 8, 9, 10], [-1, 5, -1, 4, 3, -1, 2, 1, 0, -1, -1]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([-1, 0, 1, 2, 3, 4, 5, 6, 8, 9, 10])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([8, 7, 6, 4, 3, 1, -1])))
        alignment = Align.Alignment(sequences, self.reverse_coordinates[:, :-1])
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3], [4, 6], [7, 8], [8, 9]], [[7, 6], [6, 5], [5, 3], [3, 2], [1, 0]]])))
        self.assertEqual(str(alignment), 'target            0 AACCGGGA-C  9\n                  0 |-|-||-|-| 10\nquery             7 A-C-GG-AAC  0\n')
        self.assertTrue(np.array_equal(alignment.indices, np.array([[0, 1, 2, 3, 4, 5, 6, 7, -1, 8], [6, -1, 5, -1, 4, 3, -1, 2, 1, 0]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([0, 1, 2, 3, 4, 5, 6, 7, 9, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([9, 8, 7, 5, 4, 2, 0])))
        alignment = Align.Alignment(sequences, self.reverse_coordinates[:, 1:-1])
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 3], [4, 6], [7, 8], [8, 9]], [[6, 5], [5, 3], [3, 2], [1, 0]]])))
        self.assertEqual(str(alignment), 'target            1 ACCGGGA-C 9\n                  0 -|-||-|-| 9\nquery             6 -C-GG-AAC 0\n')
        self.assertTrue(np.array_equal(alignment.indices, np.array([[1, 2, 3, 4, 5, 6, 7, -1, 8], [-1, 5, -1, 4, 3, -1, 2, 1, 0]])))
        inverse_indices = alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([-1, 0, 1, 2, 3, 4, 5, 6, 8, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([8, 7, 6, 4, 3, 1, -1])))

    def test_sort(self):
        if False:
            i = 10
            return i + 15
        target = Seq('ACTT')
        query = Seq('ACCT')
        sequences = (target, query)
        coordinates = np.array([[0, 4], [0, 4]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), 'target            0 ACTT 4\n                  0 ||.| 4\nquery             0 ACCT 4\n')
        alignment.sort()
        self.assertEqual(str(alignment), 'target            0 ACCT 4\n                  0 ||.| 4\nquery             0 ACTT 4\n')
        alignment.sort(reverse=True)
        self.assertEqual(str(alignment), 'target            0 ACTT 4\n                  0 ||.| 4\nquery             0 ACCT 4\n')
        target.id = 'seq1'
        query.id = 'seq2'
        alignment.sort()
        self.assertEqual(str(alignment), 'seq1              0 ACTT 4\n                  0 ||.| 4\nseq2              0 ACCT 4\n')
        alignment.sort(reverse=True)
        self.assertEqual(str(alignment), 'seq2              0 ACCT 4\n                  0 ||.| 4\nseq1              0 ACTT 4\n')
        alignment.sort(key=gc_fraction)
        self.assertEqual(str(alignment), 'seq1              0 ACTT 4\n                  0 ||.| 4\nseq2              0 ACCT 4\n')
        alignment.sort(key=gc_fraction, reverse=True)
        self.assertEqual(str(alignment), 'seq2              0 ACCT 4\n                  0 ||.| 4\nseq1              0 ACTT 4\n')

    def test_substitutions(self):
        if False:
            print('Hello World!')
        path = os.path.join('Align', 'ecoli.fa')
        record = SeqIO.read(path, 'fasta')
        target = record.seq
        path = os.path.join('Align', 'bsubtilis.fa')
        record = SeqIO.read(path, 'fasta')
        query = record.seq
        coordinates = np.array([[503, 744, 744, 747, 748, 820, 820, 822, 822, 823, 823, 828, 828, 833, 833, 845, 848, 850, 851, 854, 857, 1003, 1004, 1011, 1011, 1017, 1017, 1020, 1021, 1116, 1116, 1119, 1120, 1132, 1133, 1242, 1243, 1246, 1246, 1289, 1289, 1292, 1293, 1413], [512, 753, 754, 757, 757, 829, 831, 833, 834, 835, 838, 843, 844, 849, 850, 862, 862, 864, 864, 867, 867, 1013, 1013, 1020, 1021, 1027, 1028, 1031, 1031, 1126, 1127, 1130, 1130, 1142, 1142, 1251, 1251, 1254, 1255, 1298, 1299, 1302, 1302, 1422]])
        sequences = (target, query)
        forward_alignment = Align.Alignment(sequences, coordinates)
        sequences = (target, query.reverse_complement())
        coordinates = coordinates.copy()
        coordinates[1, :] = len(query) - coordinates[1, :]
        reverse_alignment = Align.Alignment(sequences, coordinates)
        for alignment in (forward_alignment, reverse_alignment):
            m = alignment.substitutions
            self.assertEqual(str(m), '      A     C     G     T\nA 191.0   3.0  15.0  13.0\nC   5.0 186.0   9.0  14.0\nG  12.0  11.0 248.0   8.0\nT  11.0  19.0   6.0 145.0\n')
            self.assertAlmostEqual(m['T', 'C'], 19.0)
            self.assertAlmostEqual(m['C', 'T'], 14.0)
            m += m.transpose()
            m /= 2.0
            self.assertEqual(str(m), '      A     C     G     T\nA 191.0   4.0  13.5  12.0\nC   4.0 186.0  10.0  16.5\nG  13.5  10.0 248.0   7.0\nT  12.0  16.5   7.0 145.0\n')
            self.assertAlmostEqual(m['C', 'T'], 16.5)
            self.assertAlmostEqual(m['T', 'C'], 16.5)

    def test_target_query_properties(self):
        if False:
            return 10
        target = 'ABCD'
        query = 'XYZ'
        sequences = [target, query]
        coordinates = np.array([[0, 3, 4], [0, 3, 3]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(alignment.sequences[0], target)
        self.assertEqual(alignment.sequences[1], query)
        self.assertEqual(alignment.target, target)
        self.assertEqual(alignment.query, query)
        target = 'EFGH'
        query = 'UVW'
        sequences = [target, query]
        alignment.sequences = sequences
        self.assertEqual(alignment.sequences[0], target)
        self.assertEqual(alignment.sequences[1], query)
        self.assertEqual(alignment.target, target)
        self.assertEqual(alignment.query, query)
        target = 'IJKL'
        query = 'RST'
        sequences = [target, query]
        alignment.sequences = sequences
        self.assertEqual(alignment.sequences[0], target)
        self.assertEqual(alignment.sequences[1], query)
        self.assertEqual(alignment.target, target)
        self.assertEqual(alignment.query, query)

    def test_reverse_complement(self):
        if False:
            return 10
        target = SeqRecord(Seq(self.target), id='seqA')
        query = SeqRecord(Seq(self.query), id='seqB')
        sequences = [target, query]
        coordinates = self.forward_coordinates
        alignment = Align.Alignment(sequences, coordinates)
        alignment.column_annotations = {'score': [2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1], 'letter': 'ABCDEFGHIJKL'}
        self.assertEqual(str(alignment), 'seqA              0 AACCGGGA-CCG 11\n                  0 |-|-||-|-|-- 12\nseqB              0 A-C-GG-AAC--  7\n')
        self.assertEqual(alignment.column_annotations['score'], [2, 1, 2, 1, 2, 2, 1, 2, 1, 2, 1, 1])
        self.assertEqual(alignment.column_annotations['letter'], 'ABCDEFGHIJKL')
        rc_alignment = alignment.reverse_complement()
        self.assertEqual(str(rc_alignment), '<unknown          0 CGG-TCCCGGTT 11\n                  0 --|-|-||-|-| 12\n<unknown          0 --GTT-CC-G-T  7\n')
        self.assertEqual(len(rc_alignment.column_annotations), 2)
        self.assertEqual(rc_alignment.column_annotations['score'], [1, 1, 2, 1, 2, 1, 2, 2, 1, 2, 1, 2])
        self.assertEqual(rc_alignment.column_annotations['letter'], 'LKJIHGFEDCBA')

    def test_add(self):
        if False:
            print('Hello World!')
        target = Seq('ACTAGG')
        query = Seq('ACCTACG')
        sequences = (target, query)
        coordinates = np.array([[0, 2, 2, 6], [0, 2, 3, 7]])
        alignment1 = Align.Alignment(sequences, coordinates)
        target = Seq('CGTGGGG')
        query = Seq('CGG')
        sequences = (target, query)
        coordinates = np.array([[0, 2, 3, 4, 7], [0, 2, 2, 3, 3]])
        alignment2 = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment1), 'target            0 AC-TAGG 6\n                  0 ||-||.| 7\nquery             0 ACCTACG 7\n')
        self.assertEqual(str(alignment2), 'target            0 CGTGGGG 7\n                  0 ||-|--- 7\nquery             0 CG-G--- 3\n')
        self.assertEqual(str(alignment1 + alignment2), 'target            0 AC-TAGGCGTGGGG 13\n                  0 ||-||.|||-|--- 14\nquery             0 ACCTACGCG-G--- 10\n')

class TestMultipleAlignment(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        path = 'Clustalw/opuntia.aln'
        with open(path) as stream:
            self.alignment = Align.read(stream, 'clustal')

    def tearDown(self):
        if False:
            while True:
                i = 10
        del self.alignment

    def test_target_query_properties(self):
        if False:
            return 10
        target = 'ABCD'
        query = 'XYZ'
        alignment = self.alignment
        with self.assertRaises(ValueError):
            alignment.target
        with self.assertRaises(ValueError):
            alignment.query
        with self.assertRaises(ValueError):
            alignment.target = target
        with self.assertRaises(ValueError):
            alignment.query = query

    def test_comparison(self):
        if False:
            return 10
        alignment = self.alignment
        self.assertEqual(alignment.shape, (7, 156))
        sequences = alignment.sequences
        coordinates = np.array(alignment.coordinates)
        other = Align.Alignment(sequences, coordinates)
        self.assertEqual(alignment, other)
        self.assertLessEqual(alignment, other)
        self.assertGreaterEqual(other, alignment)
        other = Align.Alignment(sequences, coordinates[:, 1:])
        self.assertNotEqual(alignment, other)
        self.assertLess(alignment, other)
        self.assertLessEqual(alignment, other)
        self.assertGreater(other, alignment)
        self.assertGreaterEqual(other, alignment)

    def check_indexing_slicing(self, alignment, strand):
        if False:
            print('Hello World!')
        msg = '%s strand' % strand
        self.assertEqual(repr(alignment), '<Alignment object (7 rows x 156 columns) at 0x%x>' % id(alignment))
        if strand == 'forward':
            self.assertEqual(str(alignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        self.assertEqual(len(alignment), 7)
        self.assertEqual(alignment.shape, (7, 156))
        self.assertEqual(alignment[0], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[-2], 'TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTATACCAGA', msg=msg)
        self.assertEqual(alignment[-1], 'TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(alignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328       146 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        90 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328        36 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA   0\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        self.assertEqual(len(alignment), 7)
        self.assertEqual(alignment.shape, (7, 156))
        self.assertEqual(alignment[0], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[-2], 'TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTATACCAGA', msg=msg)
        self.assertEqual(alignment[-1], 'TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[0, :], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1, :], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[-2, :], 'TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTATACCAGA', msg=msg)
        self.assertEqual(alignment[-1, :], 'TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[:, 0], 'TTTTTTT', msg=msg)
        self.assertEqual(alignment[:, 1], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, 2], 'TTTTTTT', msg=msg)
        self.assertEqual(alignment[:, 3], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, 4], 'CCCCCCC', msg=msg)
        self.assertEqual(alignment[:, 5], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, 6], 'TTTTTTT', msg=msg)
        self.assertEqual(alignment[:, 7], 'TTTATTT', msg=msg)
        self.assertEqual(alignment[:, 8], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, 9], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, 10], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, 11], 'GGGGGGG', msg=msg)
        self.assertEqual(alignment[:, 12], 'AAAAGGG', msg=msg)
        self.assertEqual(alignment[:, -156], 'TTTTTTT', msg=msg)
        self.assertEqual(alignment[:, -155], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, -154], 'TTTTTTT', msg=msg)
        self.assertEqual(alignment[:, -9], 'TTTTTTT', msg=msg)
        self.assertEqual(alignment[:, -8], 'GGGGGAG', msg=msg)
        self.assertEqual(alignment[:, -7], 'TTTTTTT', msg=msg)
        self.assertEqual(alignment[:, -6], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, -5], 'CCCCCCC', msg=msg)
        self.assertEqual(alignment[:, -4], 'CCCCCCC', msg=msg)
        self.assertEqual(alignment[:, -3], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[:, -2], 'GGGGGGG', msg=msg)
        self.assertEqual(alignment[:, -1], 'AAAAAAA', msg=msg)
        self.assertEqual(alignment[0, range(0, 156, 2)], 'TTCTAAAGGGTCGTATGAGCAAAAATTT-----AAATCATTCTTTCCATTAATTTAAATGTATTAAATCTGTTGGACG', msg=msg)
        self.assertEqual(alignment[1, range(1, 156, 2)], 'AAATAGAGGAGGAAAGAAGGAGAGAAAAA----TTTTAATCTAAACAAAAAAACATATAAGAACAGACATATATTCAA', msg=msg)
        self.assertEqual(alignment[0, (1, 4, 9)], 'ACA', msg=msg)
        self.assertEqual(alignment[1, (1, 57, 58)], 'AA-', msg=msg)
        if strand == 'forward':
            self.assertEqual(str(alignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(alignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328       146 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        90 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328        36 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA   0\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        self.assertEqual(alignment[0, 0:156], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1, 0:156], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[0, 0:], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1, 0:], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        if strand == 'forward':
            self.assertEqual(str(alignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(alignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328       146 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        90 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328        36 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA   0\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        self.assertEqual(alignment[0, :156], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1, :156], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[0, 1:], 'ATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1, 1:], 'ATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[0, 2:], 'TACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1, 2:], 'TACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[0, 60:], '------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[1, 60:], '------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAGA', msg=msg)
        self.assertEqual(alignment[0, 156:], '', msg=msg)
        self.assertEqual(alignment[1, 156:], '', msg=msg)
        self.assertEqual(alignment[0, :-1], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAG', msg=msg)
        self.assertEqual(alignment[1, :-1], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAG', msg=msg)
        self.assertEqual(alignment[0, :-2], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCA', msg=msg)
        self.assertEqual(alignment[1, :-2], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCA', msg=msg)
        self.assertEqual(alignment[0, :-3], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACC', msg=msg)
        self.assertEqual(alignment[1, :-3], 'TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACC', msg=msg)
        self.assertEqual(alignment[0, 1:-1], 'ATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAG', msg=msg)
        self.assertEqual(alignment[1, 1:-1], 'ATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAG', msg=msg)
        self.assertEqual(alignment[0, 1:-2], 'ATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCA', msg=msg)
        self.assertEqual(alignment[1, 1:-2], 'ATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCA', msg=msg)
        self.assertEqual(alignment[0, 2:-1], 'TACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCAG', msg=msg)
        self.assertEqual(alignment[1, 2:-1], 'TACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCAG', msg=msg)
        self.assertEqual(alignment[0, 2:-2], 'TACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCCATTGATTTAGTGTACCA', msg=msg)
        self.assertEqual(alignment[1, 2:-2], 'TACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATATCAAAGAATCTATTGATTTAGTGTACCA', msg=msg)
        subalignment = alignment[:, :156]
        self.assertEqual(alignment, subalignment, msg=msg)
        self.assertEqual(alignment.column_annotations, subalignment.column_annotations, msg=msg)
        subalignment = alignment[:, 0:156]
        self.assertEqual(alignment, subalignment, msg=msg)
        self.assertEqual(alignment.column_annotations, subalignment.column_annotations, msg=msg)
        self.assertEqual(len(subalignment.column_annotations), 1)
        self.assertEqual(subalignment.column_annotations['clustal_consensus'], '******* **** *******************************************          ********  **** ********* ********************************************* *********** *******', msg=msg)
        subalignment = alignment[:, 60:]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'gi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'gi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        90 ------ATATATTTCAAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGA\n\ngi|627328       110 TGAATATCAAAGAATCCATTGATTTAGTGTACCAGA 146\ngi|627328       112 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 148\ngi|627328        36 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA   0\ngi|627328       110 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 146\ngi|627329       114 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 150\ngi|627328       114 TGAATATCAAAGAATCTATTGATTTAGTATACCAGA 150\ngi|627329       120 TGAATATCAAAGAATCTATTGATTTAGTGTACCAGA 156\n', msg=msg)
        self.assertEqual(len(subalignment.column_annotations), 1)
        self.assertEqual(subalignment.column_annotations['clustal_consensus'], '      ********  **** ********* ********************************************* *********** *******', msg=msg)
        subalignment = alignment[:, :-60]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAA 86\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAA 88\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATATCCAAA 86\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAA 86\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAA 90\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAA 90\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAA 96\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\ngi|627328       146 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627328         0 TATACATAAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627328         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        56 ------ATATATTTCAAATTTCCTTATATACCCAAA 86\ngi|627328        58 ------ATATATTTCAAATTTCCTTATATACCCAAA 88\ngi|627328        90 ------ATATATTTCAAATTTCCTTATATATCCAAA 60\ngi|627328        56 ------ATATATTTATAATTTCCTTATATATCCAAA 86\ngi|627329        60 ------ATATATTTCAAATTCCCTTATATATCCAAA 90\ngi|627328        60 ------ATATATTTCAAATTCCCTTATATATCCAAA 90\ngi|627329        60 TATATAATATATTTCAAATTCCCTTATATATCCAAA 96\n', msg=msg)
        self.assertEqual(len(subalignment.column_annotations), 1)
        self.assertEqual(subalignment.column_annotations['clustal_consensus'], '******* **** *******************************************          ********  **** ********* *****', msg=msg)
        subalignment = alignment[:, 20:-60]
        if strand == 'forward':
            self.assertEqual(str(subalignment), 'gi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATT\ngi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATT\ngi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATT\ngi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTATAATT\ngi|627329        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA------ATATATTTCAAATT\ngi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA------ATATATTTCAAATT\ngi|627329        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATATATATAATATATTTCAAATT\n\ngi|627328        70 TCCTTATATACCCAAA 86\ngi|627328        72 TCCTTATATACCCAAA 88\ngi|627328        70 TCCTTATATATCCAAA 86\ngi|627328        70 TCCTTATATATCCAAA 86\ngi|627329        74 CCCTTATATATCCAAA 90\ngi|627328        74 CCCTTATATATCCAAA 90\ngi|627329        80 CCCTTATATATCCAAA 96\n', msg=msg)
        if strand == 'reverse':
            self.assertEqual(str(subalignment), 'gi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATT\ngi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--------ATATATTTCAAATT\ngi|627328       126 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTCAAATT\ngi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATA----------ATATATTTATAATT\ngi|627329        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA------ATATATTTCAAATT\ngi|627328        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA------ATATATTTCAAATT\ngi|627329        20 TGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATATATATAATATATTTCAAATT\n\ngi|627328        70 TCCTTATATACCCAAA 86\ngi|627328        72 TCCTTATATACCCAAA 88\ngi|627328        76 TCCTTATATATCCAAA 60\ngi|627328        70 TCCTTATATATCCAAA 86\ngi|627329        74 CCCTTATATATCCAAA 90\ngi|627328        74 CCCTTATATATCCAAA 90\ngi|627329        80 CCCTTATATATCCAAA 96\n', msg=msg)
        self.assertEqual(len(subalignment.column_annotations), 1)
        self.assertEqual(subalignment.column_annotations['clustal_consensus'], '************************************          ********  **** ********* *****', msg=msg)
        subalignment = alignment[:, ::2]
        self.assertEqual(str(subalignment), 'gi|627328         0 TTCTAAAGGGTCGTATGAGCAAAAATTT-----AAATCATTCTTTCCATTAATTTAAATG\ngi|627328         0 TTCTAAAGGGTCGTATGAGCAAAAATTTT----AAATCATTCTTTCCATTAATTTAAATG\ngi|627328         0 TTCTAAAGGGTCGTATGAGCAAAAATTT-----AAATCATTCTTTTCATTAATTTAAATG\ngi|627328         0 TTCTAAAGGGTCGTATGAGCAAAAATTT-----AAATAATTCTTTTCATTAATTTAAATG\ngi|627329         0 TTCTAAGGGGTCGTATGAGCAAAAATTTTT---AAATCATCCTTTTCATTAATTTAAATG\ngi|627328         0 TTCTAAGGGGTCGTATGAGCAAAAATTTTT---AAATCATCCTTTTCATTAATTTAAATG\ngi|627329         0 TTCTAAGGGGTCGTATGAGCAAAAATTTTTTTTAAATCATCCTTTTCATTAATTTAAATG\n\ngi|627328        55 TATTAAATCTGTTGGACG 73\ngi|627328        56 TATTAAATTTGTTGGACG 74\ngi|627328        55 TATTAAATTTGTTGGACG 73\ngi|627328        55 TATTAAATTTGTTGGACG 73\ngi|627329        57 TATTAAATTTGTTGGACG 75\ngi|627328        57 TATTAAATTTGTTGAACG 75\ngi|627329        60 TATTAAATTTGTTGGACG 78\n', msg=msg)
        self.assertEqual(len(subalignment.column_annotations), 1)
        self.assertEqual(subalignment.column_annotations['clustal_consensus'], '****** *********************     **** ** **** ********************** ***** ***', msg=msg)
        subalignment = alignment[:, range(0, 156, 2)]
        self.assertEqual(len(subalignment.column_annotations), 1)
        self.assertEqual(subalignment.column_annotations['clustal_consensus'], '****** *********************     **** ** **** ********************** ***** ***', msg=msg)
        subalignment = alignment[:, (1, 7, 5)]
        self.assertEqual(len(subalignment.column_annotations), 1)
        self.assertEqual(subalignment.column_annotations['clustal_consensus'], '* *', msg=msg)
        self.assertEqual(str(alignment[1::3]), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\n                  0 ||||||||||||.|||||||||||||||||||||||||||||||||||||||||||||--\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        58 ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATA\n                 60 ||||||||||||||.|||||||||.|||||||||||||||||||||||||||||||||||\ngi|627329        60 ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATA\n\ngi|627328       118 TCAAAGAATCTATTGATTTAGTGTACCAGA 148\n                120 |||||||||||||||||||||||||||||| 150\ngi|627329       120 TCAAAGAATCTATTGATTTAGTGTACCAGA 150\n', msg=msg)
        self.assertEqual(str(alignment[1::3, :]), 'gi|627328         0 TATACATTAAAGAAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATA--\n                  0 ||||||||||||.|||||||||||||||||||||||||||||||||||||||||||||--\ngi|627329         0 TATACATTAAAGGAGGGGGATGCGGATAAATGGAAAGGCGAAAGAAAGAATATATATATA\n\ngi|627328        58 ATATATTTCAAATTTCCTTATATACCCAAATATAAAAATATCTAATAAATTAGATGAATA\n                 60 ||||||||||||||.|||||||||.|||||||||||||||||||||||||||||||||||\ngi|627329        60 ATATATTTCAAATTCCCTTATATATCCAAATATAAAAATATCTAATAAATTAGATGAATA\n\ngi|627328       118 TCAAAGAATCTATTGATTTAGTGTACCAGA 148\n                120 |||||||||||||||||||||||||||||| 150\ngi|627329       120 TCAAAGAATCTATTGATTTAGTGTACCAGA 150\n', msg=msg)
        self.assertEqual(alignment, alignment[:])

    def test_indexing_slicing(self):
        if False:
            for i in range(10):
                print('nop')
        alignment = self.alignment
        strand = 'forward'
        self.check_indexing_slicing(alignment, strand)
        name = alignment.sequences[2].id
        alignment.sequences[2] = alignment.sequences[2].reverse_complement()
        alignment.sequences[2].id = name
        n = len(alignment.sequences[2])
        alignment.coordinates[2, :] = n - alignment.coordinates[2, :]
        strand = 'reverse'
        self.check_indexing_slicing(alignment, strand)

    def test_sort(self):
        if False:
            i = 10
            return i + 15
        alignment = self.alignment[:, 40:100]
        self.assertEqual(str(alignment), 'gi|627328        40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATA\ngi|627328        40 AAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTATAATTTCCTTATATATCCAAATATA\ngi|627329        40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\ngi|627329        40 AAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATA\n\ngi|627328        90 \ngi|627328        92 \ngi|627328        90 \ngi|627328        90 \ngi|627329        94 \ngi|627328        94 \ngi|627329       100 \n')
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('gi|6273285|gb|AF191659.1|AF191', 'gi|6273284|gb|AF191658.1|AF191', 'gi|6273287|gb|AF191661.1|AF191', 'gi|6273286|gb|AF191660.1|AF191', 'gi|6273290|gb|AF191664.1|AF191', 'gi|6273289|gb|AF191663.1|AF191', 'gi|6273291|gb|AF191665.1|AF191'))
        alignment.sort()
        self.assertEqual(str(alignment), 'gi|627328        40 AAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTATAATTTCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\ngi|627329        40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\ngi|627329        40 AAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATA\n\ngi|627328        92 \ngi|627328        90 \ngi|627328        90 \ngi|627328        90 \ngi|627328        94 \ngi|627329        94 \ngi|627329       100 \n')
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('gi|6273284|gb|AF191658.1|AF191', 'gi|6273285|gb|AF191659.1|AF191', 'gi|6273286|gb|AF191660.1|AF191', 'gi|6273287|gb|AF191661.1|AF191', 'gi|6273289|gb|AF191663.1|AF191', 'gi|6273290|gb|AF191664.1|AF191', 'gi|6273291|gb|AF191665.1|AF191'))
        alignment.sort(reverse=True)
        self.assertEqual(str(alignment), 'gi|627329        40 AAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATA\ngi|627329        40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTATAATTTCCTTATATATCCAAATATA\ngi|627328        40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATA\ngi|627328        40 AAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATA\n\ngi|627329       100 \ngi|627329        94 \ngi|627328        94 \ngi|627328        90 \ngi|627328        90 \ngi|627328        90 \ngi|627328        92 \n')
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('gi|6273291|gb|AF191665.1|AF191', 'gi|6273290|gb|AF191664.1|AF191', 'gi|6273289|gb|AF191663.1|AF191', 'gi|6273287|gb|AF191661.1|AF191', 'gi|6273286|gb|AF191660.1|AF191', 'gi|6273285|gb|AF191659.1|AF191', 'gi|6273284|gb|AF191658.1|AF191'))
        for (i, sequence) in enumerate(alignment.sequences[::-1]):
            sequence.id = 'seq%d' % (i + 1)
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('seq7', 'seq6', 'seq5', 'seq4', 'seq3', 'seq2', 'seq1'))
        alignment.sort()
        self.assertEqual(str(alignment), 'seq1             40 AAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATA\nseq2             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATA\nseq3             40 AAAGAAAGAATATATA----------ATATATTTATAATTTCCTTATATATCCAAATATA\nseq4             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATATCCAAATATA\nseq5             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq6             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq7             40 AAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATA\n\nseq1             92 \nseq2             90 \nseq3             90 \nseq4             90 \nseq5             94 \nseq6             94 \nseq7            100 \n')
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7'))
        alignment.sort(reverse=True)
        self.assertEqual(str(alignment), 'seq7             40 AAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATA\nseq6             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq5             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq4             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATATCCAAATATA\nseq3             40 AAAGAAAGAATATATA----------ATATATTTATAATTTCCTTATATATCCAAATATA\nseq2             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATA\nseq1             40 AAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATA\n\nseq7            100 \nseq6             94 \nseq5             94 \nseq4             90 \nseq3             90 \nseq2             90 \nseq1             92 \n')
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('seq7', 'seq6', 'seq5', 'seq4', 'seq3', 'seq2', 'seq1'))
        alignment.sort(key=lambda record: gc_fraction(record.seq))
        self.assertEqual(str(alignment), 'seq3             40 AAAGAAAGAATATATA----------ATATATTTATAATTTCCTTATATATCCAAATATA\nseq7             40 AAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATA\nseq4             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATATCCAAATATA\nseq5             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq1             40 AAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATA\nseq6             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq2             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATA\n\nseq3             90 \nseq7            100 \nseq4             90 \nseq5             94 \nseq1             92 \nseq6             94 \nseq2             90 \n')
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('seq3', 'seq7', 'seq4', 'seq5', 'seq1', 'seq6', 'seq2'))
        alignment.sort(key=lambda record: gc_fraction(record.seq), reverse=True)
        self.assertEqual(str(alignment), 'seq2             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATACCCAAATATA\nseq6             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq1             40 AAAGAAAGAATATATATA--------ATATATTTCAAATTTCCTTATATACCCAAATATA\nseq5             40 AAAGAAAGAATATATATATA------ATATATTTCAAATTCCCTTATATATCCAAATATA\nseq4             40 AAAGAAAGAATATATA----------ATATATTTCAAATTTCCTTATATATCCAAATATA\nseq7             40 AAAGAAAGAATATATATATATATATAATATATTTCAAATTCCCTTATATATCCAAATATA\nseq3             40 AAAGAAAGAATATATA----------ATATATTTATAATTTCCTTATATATCCAAATATA\n\nseq2             90 \nseq6             94 \nseq1             92 \nseq5             94 \nseq4             90 \nseq7            100 \nseq3             90 \n')
        self.assertEqual(tuple((sequence.id for sequence in alignment.sequences)), ('seq2', 'seq6', 'seq1', 'seq5', 'seq4', 'seq7', 'seq3'))

    def test_substitutions(self):
        if False:
            return 10
        alignment = self.alignment
        m = alignment.substitutions
        self.assertEqual(str(m), '       A     C     G     T\nA 1395.0   3.0  13.0   6.0\nC    3.0 271.0   0.0  16.0\nG    5.0   0.0 480.0   0.0\nT    6.0  12.0   0.0 874.0\n')
        self.assertAlmostEqual(m['T', 'C'], 12.0)
        self.assertAlmostEqual(m['C', 'T'], 16.0)
        m += m.transpose()
        m /= 2.0
        self.assertEqual(str(m), '       A     C     G     T\nA 1395.0   3.0   9.0   6.0\nC    3.0 271.0   0.0  14.0\nG    9.0   0.0 480.0   0.0\nT    6.0  14.0   0.0 874.0\n')
        self.assertAlmostEqual(m['C', 'T'], 14.0)
        self.assertAlmostEqual(m['T', 'C'], 14.0)

    def test_add(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(str(self.alignment[:, 50:60]), 'gi|627328        50 TATATA---- 56\ngi|627328        50 TATATATA-- 58\ngi|627328        50 TATATA---- 56\ngi|627328        50 TATATA---- 56\ngi|627329        50 TATATATATA 60\ngi|627328        50 TATATATATA 60\ngi|627329        50 TATATATATA 60\n')
        self.assertEqual(str(self.alignment[:, 65:75]), 'gi|627328        56 -ATATATTTC 65\ngi|627328        58 -ATATATTTC 67\ngi|627328        56 -ATATATTTC 65\ngi|627328        56 -ATATATTTA 65\ngi|627329        60 -ATATATTTC 69\ngi|627328        60 -ATATATTTC 69\ngi|627329        65 AATATATTTC 75\n')
        alignment = self.alignment[:, 50:60] + self.alignment[:, 65:75]
        self.assertEqual(str(alignment), 'gi|627328         0 TATATA-----ATATATTTC 15\ngi|627328         0 TATATATA---ATATATTTC 17\ngi|627328         0 TATATA-----ATATATTTC 15\ngi|627328         0 TATATA-----ATATATTTA 15\ngi|627329         0 TATATATATA-ATATATTTC 19\ngi|627328         0 TATATATATA-ATATATTTC 19\ngi|627329         0 TATATATATAAATATATTTC 20\n')

class TestAlignment_format(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        path = 'Clustalw/muscle.a2m'
        with open(path) as stream:
            alignments = Align.parse(stream, 'a2m')
            alignment = next(alignments)
        self.alignment = alignment[:2, :]

    def test_a2m(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.alignment.format('a2m'), '>Test1seq\n.................................................................AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACAAAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT.\n>AT3G20900.1-SEQ\natgaacaaagtagcgaggaagaacaaaacatcaggtgaacaaaaaaaaaactcaatccacatcaaAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAg\n')

    def test_bed(self):
        if False:
            while True:
                i = 10
        self.alignment.score = 100
        self.assertEqual(self.alignment.format('bed'), 'Test1seq\t0\t621\tAT3G20900.1-SEQ\t100\t+\t0\t621\t0\t6\t213,23,30,9,172,174,\t0,213,236,266,275,447,\n')

    def test_clustal(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.alignment.format('clustal'), 'Test1seq                            --------------------------------------------------\nAT3G20900.1-SEQ                     ATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAA\n\nTest1seq                            ---------------AGTTACAATAACTGACGAAGCTAAGTAGGCTACTA\nAT3G20900.1-SEQ                     CTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAA\n\nTest1seq                            ATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGA\nAT3G20900.1-SEQ                     ATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGA\n\nTest1seq                            AAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATC\nAT3G20900.1-SEQ                     AAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATA\n\nTest1seq                            GTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAA\nAT3G20900.1-SEQ                     GTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAA\n\nTest1seq                            ATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACA\nAT3G20900.1-SEQ                     ATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAA\n\nTest1seq                            AAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACT\nAT3G20900.1-SEQ                     CAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACT\n\nTest1seq                            CTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTA\nAT3G20900.1-SEQ                     CTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTA\n\nTest1seq                            ATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATA\nAT3G20900.1-SEQ                     ATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATA\n\nTest1seq                            ATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTC\nAT3G20900.1-SEQ                     ATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTC\n\nTest1seq                            TATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGT\nAT3G20900.1-SEQ                     TATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGT\n\nTest1seq                            CGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCC\nAT3G20900.1-SEQ                     TTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCC\n\nTest1seq                            GGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGT\nAT3G20900.1-SEQ                     GGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGT\n\nTest1seq                            CAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT-\nAT3G20900.1-SEQ                     CAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\n\n\n')

    def test_bigbed(self):
        if False:
            print('Hello World!')
        self.assertRaisesRegex(ValueError, 'bigbed is a binary file format', self.alignment.format, 'bigbed')

    def test_bigmaf(self):
        if False:
            while True:
                i = 10
        self.assertRaisesRegex(ValueError, 'bigmaf is a binary file format', self.alignment.format, 'bigmaf')

    def test_bigpsl(self):
        if False:
            while True:
                i = 10
        self.assertRaisesRegex(ValueError, 'bigpsl is a binary file format', self.alignment.format, 'bigpsl')

    def test_emboss(self):
        if False:
            print('Hello World!')
        self.assertRaisesRegex(ValueError, 'Formatting alignments has not yet been implemented for the emboss format', self.alignment.format, 'emboss')

    def test_exonerate(self):
        if False:
            print('Hello World!')
        self.alignment.score = 100
        self.assertEqual(self.alignment.format('exonerate'), 'vulgar: AT3G20900.1-SEQ 0 687 + Test1seq 0 621 + 100 G 65 0 M 213 213 M 23 23 M 30 30 M 9 9 M 172 172 M 174 174 G 1 0\n')
        self.assertEqual(self.alignment.format('exonerate', 'vulgar'), 'vulgar: AT3G20900.1-SEQ 0 687 + Test1seq 0 621 + 100 G 65 0 M 213 213 M 23 23 M 30 30 M 9 9 M 172 172 M 174 174 G 1 0\n')
        self.assertEqual(self.alignment.format('exonerate', 'cigar'), 'cigar: AT3G20900.1-SEQ 0 687 + Test1seq 0 621 + 100 I 65 M 213 M 23 M 30 M 9 M 172 M 174 I 1\n')

    def test_fasta(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.alignment.format('fasta'), '>Test1seq\n-----------------------------------------------------------------AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACAAAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT-\n>AT3G20900.1-SEQ\nATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\n')

    def test_hhr(self):
        if False:
            i = 10
            return i + 15
        self.assertRaisesRegex(ValueError, 'Formatting alignments has not yet been implemented for the hhr format', self.alignment.format, 'hhr')

    def test_maf(self):
        if False:
            for i in range(10):
                print('nop')
        self.alignment.score = 100
        self.assertEqual(self.alignment.format('maf'), 'a score=100.000000\ns Test1seq        0 621 + 621 -----------------------------------------------------------------AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACAAAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT-\ns AT3G20900.1-SEQ 0 687 + 687 ATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\n\n')

    def test_mauve(self):
        if False:
            i = 10
            return i + 15
        alignment = self.alignment
        metadata = {'File': 'testfile.fa'}
        for (index, record) in enumerate(alignment.sequences):
            record.id = str(index + 1)
        self.assertEqual(alignment.format('mauve', metadata=metadata), '> 2:1-621 + testfile.fa\n-----------------------------------------------------------------AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACAAAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT-\n> 3:1-687 + testfile.fa\nATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\n=\n')

    def test_msf(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaisesRegex(ValueError, 'Formatting alignments has not yet been implemented for the msf format', self.alignment.format, 'msf')

    def test_nexus(self):
        if False:
            for i in range(10):
                print('nop')
        alignment = self.alignment
        for record in alignment.sequences:
            record.annotations['molecule_type'] = 'DNA'
        self.assertEqual(self.alignment.format('nexus'), "#NEXUS\nbegin data;\ndimensions ntax=2 nchar=687;\nformat datatype=dna missing=? gap=-;\nmatrix\nTest1seq          -----------------------------------------------------------------AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACAAAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT-\n'AT3G20900.1-SEQ' ATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\n;\nend;\n")

    def test_phylip(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.alignment.format('phylip'), '2 687\nTest1seq  -----------------------------------------------------------------AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACAAAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT-\nAT3G20900.ATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\n')

    def test_psl(self):
        if False:
            print('Hello World!')
        self.assertEqual(self.alignment.format('psl'), '589\t32\t0\t0\t0\t0\t0\t0\t+\tAT3G20900.1-SEQ\t687\t65\t686\tTest1seq\t621\t0\t621\t6\t213,23,30,9,172,174,\t65,278,301,331,340,512,\t0,213,236,266,275,447,\n')
        self.assertEqual(self.alignment.format('psl', mask='upper'), '0\t32\t589\t0\t0\t0\t0\t0\t+\tAT3G20900.1-SEQ\t687\t65\t686\tTest1seq\t621\t0\t621\t6\t213,23,30,9,172,174,\t65,278,301,331,340,512,\t0,213,236,266,275,447,\n')
        self.assertEqual(self.alignment.format('psl', wildcard='A'), '362\t13\t0\t246\t0\t0\t0\t0\t+\tAT3G20900.1-SEQ\t687\t65\t686\tTest1seq\t621\t0\t621\t6\t213,23,30,9,172,174,\t65,278,301,331,340,512,\t0,213,236,266,275,447,\n')

    def test_sam(self):
        if False:
            while True:
                i = 10
        self.alignment.score = 100
        self.assertEqual(self.alignment.format('sam'), 'AT3G20900.1-SEQ\t0\tTest1seq\t1\t255\t65I213M23M30M9M172M174M1I\t*\t0\t0\tATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\t*\tAS:i:100\n')
        self.assertEqual(self.alignment.format('sam', md=True), 'AT3G20900.1-SEQ\t0\tTest1seq\t1\t255\t65I213M23M30M9M172M174M1I\t*\t0\t0\tATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\t*\tMD:Z:32C0T6C39T53C2A0T0T0C0C0G0G51T3T0T32C1A78C50C0G0T3C43T66G2C0G0C1A4A0A79G44T\tAS:i:100\n')

    def test_stockholm(self):
        if False:
            for i in range(10):
                print('nop')
        alignment = self.alignment
        del alignment.column_annotations['state']
        self.assertEqual(self.alignment.format('stockholm'), '# STOCKHOLM 1.0\n#=GF SQ   2\nTest1seq                        -----------------------------------------------------------------AGTTACAATAACTGACGAAGCTAAGTAGGCTACTAATTAACGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGTAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATCGTATTCCGGTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAATAAATTAGCGCCAAAATAATGAAAAAAATAATAACAAACAAAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGCTGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAACGTAAACAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTTCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACGGTCGCTAGAGAAACTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCGTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTT-\nAT3G20900.1-SEQ                 ATGAACAAAGTAGCGAGGAAGAACAAAACATCAGGTGAACAAAAAAAAAACTCAATCCACATCAAAGTTACAATAACTGACGAAGCTAAGTAGGCTAGAAATTAAAGTCATCAACCTAATACATAGCACTTAGAAAAAAGTGAAGCAAGAAAATATAAAATAATAAAAGGGTGGGTTATCAATTGATAGTGTAAATCATAGTTGATTTTTGATATACCCTACCACAAAAACTCAAACCGACTTGATTCAAATCATCTCAAAAAACAAGCGCCAAAATAATGAAAAAAATAATAACAAAAACAAACAAACCAAAATAAGAAAAAACATTACGCAAAACATAATAATTTACTCTTCGTTATTGTATTAACAAATCAAAGAGATGAATTTTGATCACCTGCTAATACTACTTTCTGTATTGATCCTATATCAAAAAAAAAAAAGATACTAATAATTAACTAAAAGTACGTTCATCGATCGTGTGCGTTGACGAAGAAGAGCTCTATCTCCGGCGGAGCAAAGAAAACGATCTGTCTCCGTCGTAACACACAGTTTTTCGAGACCCTTTGCTTCTTCGGCGCCGGTGGACACGTCAGCATCTCCGGTATCCTAGACTTCTTGGCTTTCGGGGTACAACAACCGCCTGGTGACGTCAGCACCGCTGCTGGGGATGGAGAGGGAACAGAGTAG\n//\n')

    def test_tabular(self):
        if False:
            i = 10
            return i + 15
        self.assertRaisesRegex(ValueError, 'Formatting alignments has not yet been implemented for the tabular format', self.alignment.format, 'tabular')

class TestAlignment_pairwise_format(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner('blastn')
        aligner.mode = 'local'
        seqA = 'AAAAACCCGGGTTTT'
        seqB = 'CCCTGGG'
        alignments = aligner.align(seqA, seqB)
        self.assertEqual(len(alignments), 2)
        self.plain_alignments = list(alignments)
        seqA = Seq('AAAAACCCGGGTTTT')
        seqB = Seq('CCCTGGG')
        alignments = aligner.align(seqA, seqB)
        self.assertEqual(len(alignments), 2)
        self.seq_alignments = list(alignments)
        alignments = aligner.align(seqA, seqB)
        alignments = list(alignments)
        for alignment in alignments:
            alignment.sequences[0] = SeqRecord(seqA, id='A', description='sequence A')
            alignment.sequences[1] = SeqRecord(seqB, id='B', description='sequence B')
        self.seqrecord_alignments = alignments

    def test_a2m(self):
        if False:
            i = 10
            return i + 15
        for alignment in self.plain_alignments:
            alignment.column_annotations = {'state': 'DDDDDD'}
        for alignment in self.seq_alignments:
            alignment.column_annotations = {'state': 'DDDDDD'}
        for alignment in self.seqrecord_alignments:
            alignment.column_annotations = {'state': 'DDDDDD'}
        self.check('a2m', self.plain_alignments)
        self.check('a2m', self.seq_alignments)
        self.check('a2m', self.seqrecord_alignments, ('A', 'B'), ('sequence A', 'sequence B'))

    def test_bed(self):
        if False:
            return 10
        self.check('bed', self.plain_alignments, ('target', 'query'))
        self.check('bed', self.seq_alignments, ('target', 'query'))
        self.check('bed', self.seqrecord_alignments, ('A', 'B'))

    def test_clustal(self):
        if False:
            print('Hello World!')
        self.check('clustal', self.plain_alignments, ('sequence_0', 'sequence_1'))
        self.check('clustal', self.seq_alignments, ('sequence_0', 'sequence_1'))
        self.check('clustal', self.seqrecord_alignments, ('A', 'B'))

    def test_exonerate(self):
        if False:
            return 10
        self.check('exonerate', self.plain_alignments, ('target', 'query'))
        self.check('exonerate', self.seq_alignments, ('target', 'query'))
        self.check('exonerate', self.seqrecord_alignments, ('A', 'B'))

    def test_fasta(self):
        if False:
            print('Hello World!')
        self.check('fasta', self.plain_alignments)
        self.check('fasta', self.seq_alignments)
        self.check('fasta', self.seqrecord_alignments, ('A', 'B'), ('sequence A', 'sequence B'))

    def test_maf(self):
        if False:
            for i in range(10):
                print('nop')
        self.check('maf', self.plain_alignments, ('sequence_0', 'sequence_1'))
        self.check('maf', self.seq_alignments, ('sequence_0', 'sequence_1'))
        self.check('maf', self.seqrecord_alignments, ('A', 'B'))

    def test_phylip(self):
        if False:
            print('Hello World!')
        self.check('phylip', self.plain_alignments)
        self.check('phylip', self.seq_alignments)
        self.check('phylip', self.seqrecord_alignments, ('A', 'B'))

    def test_psl(self):
        if False:
            while True:
                i = 10
        self.check('psl', self.plain_alignments, ('target', 'query'))
        self.check('psl', self.seq_alignments, ('target', 'query'))
        self.check('psl', self.seqrecord_alignments, ('A', 'B'))

    def test_sam(self):
        if False:
            return 10
        self.check('sam', self.plain_alignments, ('target', 'query'))
        self.check('sam', self.seq_alignments, ('target', 'query'))
        self.check('sam', self.seqrecord_alignments, ('A', 'B'))

    def check(self, fmt, alignments, ids=('', ''), descriptions=('', '')):
        if False:
            print('Hello World!')
        stream = StringIO()
        Align.write(alignments[0], stream, fmt)
        stream.seek(0)
        alignment = Align.read(stream, fmt)
        self.assertEqual(alignment.sequences[0].id, ids[0])
        self.assertEqual(alignment.sequences[1].id, ids[1])
        self.assertEqual(alignment.sequences[0].description, descriptions[0])
        self.assertEqual(alignment.sequences[1].description, descriptions[1])

class TestAlign_out_of_order(unittest.TestCase):
    seq1 = 'AACCCAAAACCAAAAATTTAAATTTTAAA'
    seq2 = 'TGTTTTTCCCCC'
    coordinates = np.array([[16, 19, 22, 26, 2, 5, 9, 11], [0, 3, 3, 7, 7, 10, 10, 12]])
    forward_alignment = Align.Alignment([seq1, seq2], coordinates)
    coordinates = np.array([[13, 10, 7, 3, 27, 24, 20, 18], [0, 3, 3, 7, 7, 10, 10, 12]])
    reverse_alignment = Align.Alignment([reverse_complement(seq1), seq2], coordinates)
    coordinates = np.array([[16, 19, 22, 26, 2, 2, 5, 9, 11], [13, 10, 7, 3, 3, 27, 24, 20, 18], [0, 3, 3, 7, 7, 7, 10, 10, 12]])
    multiple_alignment = Align.Alignment([seq1, reverse_complement(seq1), seq2], coordinates)
    del seq1
    del seq2
    del coordinates
    forward_array = np.array(forward_alignment, 'U')
    reverse_array = np.array(reverse_alignment, 'U')
    multiple_array = np.array(multiple_alignment, 'U')

    def test_array(self):
        if False:
            return 10
        alignments = (self.forward_alignment, self.reverse_alignment)
        arrays = (self.forward_array, self.reverse_array)
        for (alignment, a) in zip(alignments, arrays):
            self.assertEqual(alignment.shape, (2, 19))
            self.assertTrue(np.array_equal(a, np.array([['T', 'T', 'T', 'A', 'A', 'A', 'T', 'T', 'T', 'T', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'C', 'C'], ['T', 'G', 'T', '-', '-', '-', 'T', 'T', 'T', 'T', 'C', 'C', 'C', '-', '-', '-', '-', 'C', 'C']], dtype='U')))
        self.assertEqual(self.multiple_alignment.shape, (3, 19))
        self.assertTrue(np.array_equal(self.multiple_array, np.array([['T', 'T', 'T', 'A', 'A', 'A', 'T', 'T', 'T', 'T', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'C', 'C'], ['T', 'T', 'T', 'A', 'A', 'A', 'T', 'T', 'T', 'T', 'C', 'C', 'C', 'A', 'A', 'A', 'A', 'C', 'C'], ['T', 'G', 'T', '-', '-', '-', 'T', 'T', 'T', 'T', 'C', 'C', 'C', '-', '-', '-', '-', 'C', 'C']], dtype='U')))

    def test_row(self):
        if False:
            return 10
        alignments = (self.forward_alignment, self.reverse_alignment)
        arrays = (self.forward_array, self.reverse_array)
        for (alignment, a) in zip(alignments, arrays):
            n = len(alignment)
            for i in range(n):
                s = ''.join(a[i, :])
                self.assertEqual(alignment[i], s)
            for i in range(-n, 0):
                s = ''.join(a[i, :])
                self.assertEqual(alignment[i], s)
        alignment = self.multiple_alignment
        a = self.multiple_array
        n = len(alignment)
        for i in range(n):
            s = ''.join(a[i, :])
            self.assertEqual(alignment[i], s)
        for i in range(-n, 0):
            s = ''.join(a[i, :])
            self.assertEqual(alignment[i], s)

    def test_row_col(self):
        if False:
            return 10
        alignments = (self.forward_alignment, self.reverse_alignment)
        arrays = (self.forward_array, self.reverse_array)
        for (alignment, a) in zip(alignments, arrays):
            (n, m) = alignment.shape
            for i in range(n):
                for j in range(m):
                    self.assertEqual(alignment[i, j], a[i, j])
                for j in range(-m, 0):
                    self.assertEqual(alignment[i, j], a[i, j])
        alignment = self.multiple_alignment
        a = self.multiple_array
        (n, m) = alignment.shape
        for i in range(n):
            for j in range(m):
                self.assertEqual(alignment[i, j], a[i, j])
            for j in range(-m, 0):
                self.assertEqual(alignment[i, j], a[i, j])

    def test_row_slice(self):
        if False:
            for i in range(10):
                print('nop')
        alignments = (self.forward_alignment, self.reverse_alignment)
        arrays = (self.forward_array, self.reverse_array)
        for (alignment, a) in zip(alignments, arrays):
            (n, m) = alignment.shape
            for i in range(n):
                s = ''.join(a[i, :])
                for j in range(m):
                    self.assertEqual(alignment[i, j:], s[j:])
                for j in range(-m, 0):
                    self.assertEqual(alignment[i, j:], s[j:])
                for j in range(m):
                    self.assertEqual(alignment[i, j:-1], s[j:-1])
                for j in range(-m, 0):
                    self.assertEqual(alignment[i, j:-1], s[j:-1])
        alignment = self.multiple_alignment
        a = self.multiple_array
        (n, m) = alignment.shape
        for i in range(n):
            s = ''.join(a[i, :])
            for j in range(m):
                self.assertEqual(alignment[i, j:], s[j:])
            for j in range(-m, 0):
                self.assertEqual(alignment[i, j:], s[j:])
            for j in range(m):
                self.assertEqual(alignment[i, j:-1], s[j:-1])
            for j in range(-m, 0):
                self.assertEqual(alignment[i, j:-1], s[j:-1])

    def test_row_iterable(self):
        if False:
            i = 10
            return i + 15
        alignments = (self.forward_alignment, self.reverse_alignment)
        arrays = (self.forward_array, self.reverse_array)
        for (alignment, a) in zip(alignments, arrays):
            n = len(alignment)
            for i in range(n):
                jj = (1, 2, 6, 8)
                s = ''.join([a[i, j] for j in jj])
                self.assertEqual(alignment[i, jj], s)
                jj = (3, 3, 2, 7)
                s = ''.join([a[i, j] for j in jj])
                self.assertEqual(alignment[i, jj], s)
        alignment = self.multiple_alignment
        a = self.multiple_array
        n = len(alignment)
        for i in range(n):
            jj = (1, 2, 6, 8)
            s = ''.join([a[i, j] for j in jj])
            self.assertEqual(alignment[i, jj], s)
            jj = (3, 3, 2, 7)
            s = ''.join([a[i, j] for j in jj])
            self.assertEqual(alignment[i, jj], s)

    def test_rows_col(self):
        if False:
            i = 10
            return i + 15
        alignments = (self.forward_alignment, self.reverse_alignment)
        arrays = (self.forward_array, self.reverse_array)
        for (alignment, a) in zip(alignments, arrays):
            (n, m) = alignment.shape
            for j in range(m):
                s = ''.join(a[:, j])
                self.assertEqual(alignment[:, j], s)
            for j in range(-m, 0):
                s = ''.join(a[:, j])
                self.assertEqual(alignment[:, j], s)
        alignment = self.multiple_alignment
        a = self.multiple_array
        (n, m) = alignment.shape
        for j in range(m):
            s = ''.join(a[:, j])
            self.assertEqual(alignment[:, j], s)
        for j in range(-m, 0):
            s = ''.join(a[:, j])
            self.assertEqual(alignment[:, j], s)

    def test_rows_cols(self):
        if False:
            for i in range(10):
                print('nop')
        alignment = self.forward_alignment[:, 1:]
        self.assertEqual(str(alignment), 'target           17 TTAAATTTT 26\n                  0 .|---||||\nquery             1 GT---TTTT 7\n\ntarget            2 CCCAAAACC 11\n                  9 |||----|| 18\nquery             7 CCC----CC 12\n')
        alignment = self.forward_alignment[:, :-1]
        self.assertEqual(str(alignment), 'target           16 TTTAAATTTT 26\n                  0 |.|---||||\nquery             0 TGT---TTTT 7\n\ntarget            2 CCCAAAAC 10\n                 10 |||----| 18\nquery             7 CCC----C 11\n')
        alignment = self.forward_alignment[:, 2:-2]
        self.assertEqual(str(alignment), 'target           18 TAAATTTT 26\n                  0 |---||||\nquery             2 T---TTTT 7\n\ntarget            2 CCCAAAA  9\n                  8 |||---- 15\nquery             7 CCC---- 10\n')
        alignment = self.reverse_alignment[:, 1:]
        self.assertEqual(str(alignment), 'target           12 TTAAATTTT 3\n                  0 .|---||||\nquery             1 GT---TTTT 7\n\ntarget           27 CCCAAAACC 18\n                  9 |||----|| 18\nquery             7 CCC----CC 12\n')
        alignment = self.reverse_alignment[:, :-1]
        self.assertEqual(str(alignment), 'target           13 TTTAAATTTT 3\n                  0 |.|---||||\nquery             0 TGT---TTTT 7\n\ntarget           27 CCCAAAAC 19\n                 10 |||----| 18\nquery             7 CCC----C 11\n')
        alignment = self.reverse_alignment[:, 2:-2]
        self.assertEqual(str(alignment), 'target           11 TAAATTTT 3\n                  0 |---||||\nquery             2 T---TTTT 7\n\ntarget           27 CCCAAAA 20\n                  8 |||---- 15\nquery             7 CCC---- 10\n')

    def test_aligned(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(np.array_equal(self.forward_alignment.aligned, np.array([[[16, 19], [22, 26], [2, 5], [9, 11]], [[0, 3], [3, 7], [7, 10], [10, 12]]])))
        self.assertTrue(np.array_equal(self.reverse_alignment.aligned, np.array([[[13, 10], [7, 3], [27, 24], [20, 18]], [[0, 3], [3, 7], [7, 10], [10, 12]]])))

    def test_indices(self):
        if False:
            print('Hello World!')
        indices = self.forward_alignment.indices
        self.assertTrue(np.array_equal(indices, np.array([[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 2, 3, 4, 5, 6, 7, 8, 9, 10], [0, 1, 2, -1, -1, -1, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, 10, 11]])))
        inverse_indices = self.forward_alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([-1, -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 17, 18])))
        indices = self.reverse_alignment.indices
        self.assertTrue(np.array_equal(indices, np.array([[12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 26, 25, 24, 23, 22, 21, 20, 19, 18], [0, 1, 2, -1, -1, -1, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, 10, 11]])))
        inverse_indices = self.reverse_alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 2)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([-1, -1, -1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, -1, 18, 17, 16, 15, 14, 13, 12, 11, 10, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 17, 18])))
        indices = self.multiple_alignment.indices
        self.assertTrue(np.array_equal(indices, np.array([[16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 2, 3, 4, 5, 6, 7, 8, 9, 10], [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 26, 25, 24, 23, 22, 21, 20, 19, 18], [0, 1, 2, -1, -1, -1, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1, -1, 10, 11]])))
        inverse_indices = self.multiple_alignment.inverse_indices
        self.assertEqual(len(inverse_indices), 3)
        self.assertTrue(np.array_equal(inverse_indices[0], np.array([-1, -1, 10, 11, 12, 13, 14, 15, 16, 17, 18, -1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, -1, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[1], np.array([-1, -1, -1, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, -1, -1, -1, -1, -1, 18, 17, 16, 15, 14, 13, 12, 11, 10, -1, -1])))
        self.assertTrue(np.array_equal(inverse_indices[2], np.array([0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 17, 18])))

    def test_substitutions(self):
        if False:
            while True:
                i = 10
        for alignment in (self.forward_alignment, self.reverse_alignment):
            self.assertEqual(str(alignment.substitutions), '    A   C   G   T\nA 0.0 0.0 0.0 0.0\nC 0.0 5.0 0.0 0.0\nG 0.0 0.0 0.0 0.0\nT 0.0 0.0 1.0 6.0\n')
        self.assertEqual(str(self.multiple_alignment.substitutions), '    A    C   G    T\nA 7.0  0.0 0.0  0.0\nC 0.0 15.0 0.0  0.0\nG 0.0  0.0 0.0  0.0\nT 0.0  0.0 2.0 19.0\n')

    def test_str(self):
        if False:
            while True:
                i = 10
        self.assertEqual(str(self.forward_alignment), 'target           16 TTTAAATTTT 26\n                  0 |.|---||||\nquery             0 TGT---TTTT 7\n\ntarget            2 CCCAAAACC 11\n                 10 |||----|| 19\nquery             7 CCC----CC 12\n')
        self.assertEqual(str(self.reverse_alignment), 'target           13 TTTAAATTTT 3\n                  0 |.|---||||\nquery             0 TGT---TTTT 7\n\ntarget           27 CCCAAAACC 18\n                 10 |||----|| 19\nquery             7 CCC----CC 12\n')
        self.assertEqual(str(self.multiple_alignment), '                 16 TTTAAATTTT 26 2\n                 13 TTTAAATTTT 3 3\n                  0 TGT---TTTT 7 7\n\n                  2 CCCAAAACC 11\n                 27 CCCAAAACC 18\n                  7 CCC----CC 12\n')

class TestAlign_nucleotide_protein_str(unittest.TestCase):
    s1 = 'ATGCGGAGCTTTCGAGCGACGTTTGGCTTTGACGACGGA' * 6
    s2 = 'ATGCGGAGCCGAGCGACGTTTACGGCTTTGACGACGGA' * 6
    t1 = translate(s1)
    aligner = Align.PairwiseAligner('blastn')
    alignments = aligner.align(s1, s2)
    alignment = next(alignments)
    del aligner

    def test_nucleotide_nucleotide_str(self):
        if False:
            print('Hello World!')
        self.assertEqual(len(self.alignments), 1)
        self.assertEqual(str(self.alignment), 'target            0 ATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAGCGA\n                  0 |||||||||---||||||||||||--||||||||||||||||||||||||---|||||||\nquery             0 ATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGA\n\ntarget           58 CGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGACGAC\n                 60 |||||--||||||||||||||||||||||||---||||||||||||--||||||||||||\nquery            54 CGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGAC\n\ntarget          114 GGAATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAG\n                120 ||||||||||||---||||||||||||--||||||||||||||||||||||||---||||\nquery           111 GGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAG\n\ntarget          172 CGACGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGAC\n                180 ||||||||--||||||||||||||||||||||||---||||||||||||--|||||||||\nquery           165 CGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGAC\n\ntarget          228 GACGGA 234\n                240 |||||| 246\nquery           222 GACGGA 228\n')

    def test_protein_nucleotide_str(self):
        if False:
            return 10
        (coordinates_s1, coordinates_s2) = self.alignment.coordinates
        coordinates_t1 = coordinates_s1 // 3
        sequences = [self.t1, self.s1]
        coordinates = np.array([coordinates_t1, coordinates_s1])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), 'target            0 M  R  S  F  R  A  T  F  G  F  D  D  G  M  R  S  F  R  A  T  \nquery             0 ATGCGGAGCTTTCGAGCGACGTTTGGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACG\n\ntarget           20 F  G  F  D  D  G  M  R  S  F  R  A  T  F  G  F  D  D  G  M  \nquery            60 TTTGGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACGTTTGGCTTTGACGACGGAATG\n\ntarget           40 R  S  F  R  A  T  F  G  F  D  D  G  M  R  S  F  R  A  T  F  \nquery           120 CGGAGCTTTCGAGCGACGTTTGGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACGTTT\n\ntarget           60 G  F  D  D  G  M  R  S  F  R  A  T  F  G  F  D  D  G    78\nquery           180 GGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACGTTTGGCTTTGACGACGGA 234\n')
        sequences = [self.t1, self.s2]
        coordinates = np.array([coordinates_t1, coordinates_s2])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), 'target            0 M  R  S  F  R  A  T  F  --G  F  D  D  G  M  R  S  F  R  A  T\nquery             0 ATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGA\n\ntarget           20   F  --G  F  D  D  G  M  R  S  F  R  A  T  F  --G  F  D  D  \nquery            54 CGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGAC\n\ntarget           38 G  M  R  S  F  R  A  T  F  --G  F  D  D  G  M  R  S  F  R  A\nquery           111 GGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAG\n\ntarget           58   T  F  --G  F  D  D  G  M  R  S  F  R  A  T  F  --G  F  D  \nquery           165 CGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGAC\n\ntarget           76 D  G    78\nquery           222 GACGGA 228\n')
        sequences = [self.t1, self.s1, self.s2]
        coordinates = np.array([coordinates_t1, coordinates_s1, coordinates_s2])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  R  S  F  R  A  T  F  --G  F  D  D  G  M  R  S  F  R  A  T\n                  0 ATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAGCGA\n                  0 ATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGA\n\n                 20   F  --G  F  D  D  G  M  R  S  F  R  A  T  F  --G  F  D  D  \n                 58 CGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGACGAC\n                 54 CGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGAC\n\n                 38 G  M  R  S  F  R  A  T  F  --G  F  D  D  G  M  R  S  F  R  A\n                114 GGAATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAG\n                111 GGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAG\n\n                 58   T  F  --G  F  D  D  G  M  R  S  F  R  A  T  F  --G  F  D  \n                172 CGACGTTT--GGCTTTGACGACGGAATGCGGAGCTTTCGAGCGACGTTT--GGCTTTGAC\n                165 CGACGTTTACGGCTTTGACGACGGAATGCGGAGC---CGAGCGACGTTTACGGCTTTGAC\n\n                 76 D  G    78\n                228 GACGGA 234\n                222 GACGGA 228\n')

    def test_protein_nucleotide_many_str(self):
        if False:
            print('Hello World!')
        t = 'MMA'
        s = 'ATGATGGCC'
        sequences = [t, s]
        coordinates = np.array([[0, 3], [0, 9]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), 'target            0 M  M  A   3\nquery             0 ATGATGGCC 9\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'ATGATGGCC')
        sequences = [t, t, s]
        coordinates = np.array([[0, 3], [0, 3], [0, 9]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M  A   3\n                  0 M  M  A   3\n                  0 ATGATGGCC 9\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        sequences = [t, t, s, s]
        coordinates = np.array([[0, 3], [0, 3], [0, 9], [0, 9]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M  A   3\n                  0 M  M  A   3\n                  0 ATGATGGCC 9\n                  0 ATGATGGCC 9\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        self.assertEqual(alignment[3], 'ATGATGGCC')
        s = 'ATGATGCC'
        sequences = [t, s]
        coordinates = np.array([[0, 2, 2, 3], [0, 6, 5, 8]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), 'target            0 M  M   2\nquery             0 ATGATG 6\n\ntarget            2 A   3\nquery             5 GCC 8\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'ATGATGGCC')
        sequences = [t, t, s]
        coordinates = np.array([[0, 2, 2, 3], [0, 2, 2, 3], [0, 6, 5, 8]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M   2\n                  0 M  M   2\n                  0 ATGATG 6\n\n                  2 A   3\n                  2 A   3\n                  5 GCC 8\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        sequences = [t, s, s]
        coordinates = np.array([[0, 2, 2, 3], [0, 6, 5, 8], [0, 6, 5, 8]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M   2\n                  0 ATGATG 6\n                  0 ATGATG 6\n\n                  2 A   3\n                  5 GCC 8\n                  5 GCC 8\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'ATGATGGCC')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        sequences = [t, t, s, s]
        coordinates = np.array([[0, 2, 2, 3], [0, 2, 2, 3], [0, 6, 5, 8], [0, 6, 5, 8]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M   2\n                  0 M  M   2\n                  0 ATGATG 6\n                  0 ATGATG 6\n\n                  2 A   3\n                  2 A   3\n                  5 GCC 8\n                  5 GCC 8\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        self.assertEqual(alignment[3], 'ATGATGGCC')
        t = 'MMA'
        s = 'GGCCATCAT'
        sequences = [t, s]
        coordinates = np.array([[0, 3], [9, 0]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), 'target            0 M  M  A   3\nquery             9 ATGATGGCC 0\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'ATGATGGCC')
        sequences = [t, t, s]
        coordinates = np.array([[0, 3], [0, 3], [9, 0]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M  A   3\n                  0 M  M  A   3\n                  9 ATGATGGCC 0\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        sequences = [t, t, s, s]
        coordinates = np.array([[0, 3], [0, 3], [9, 0], [9, 0]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M  A   3\n                  0 M  M  A   3\n                  9 ATGATGGCC 0\n                  9 ATGATGGCC 0\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        self.assertEqual(alignment[3], 'ATGATGGCC')
        s = 'GGCATCAT'
        sequences = [t, s]
        coordinates = np.array([[0, 2, 2, 3], [8, 2, 3, 0]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), 'target            0 M  M   2\nquery             8 ATGATG 2\n\ntarget            2 A   3\nquery             3 GCC 0\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'ATGATGGCC')
        sequences = [t, t, s]
        coordinates = np.array([[0, 2, 2, 3], [0, 2, 2, 3], [8, 2, 3, 0]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M   2\n                  0 M  M   2\n                  8 ATGATG 2\n\n                  2 A   3\n                  2 A   3\n                  3 GCC 0\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        sequences = [t, s, s]
        coordinates = np.array([[0, 2, 2, 3], [8, 2, 3, 0], [8, 2, 3, 0]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M   2\n                  8 ATGATG 2\n                  8 ATGATG 2\n\n                  2 A   3\n                  3 GCC 0\n                  3 GCC 0\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'ATGATGGCC')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        sequences = [t, t, s, s]
        coordinates = np.array([[0, 2, 2, 3], [0, 2, 2, 3], [8, 2, 3, 0], [8, 2, 3, 0]])
        alignment = Align.Alignment(sequences, coordinates)
        self.assertEqual(str(alignment), '                  0 M  M   2\n                  0 M  M   2\n                  8 ATGATG 2\n                  8 ATGATG 2\n\n                  2 A   3\n                  2 A   3\n                  3 GCC 0\n                  3 GCC 0\n')
        self.assertEqual(alignment[0], 'MMA')
        self.assertEqual(alignment[1], 'MMA')
        self.assertEqual(alignment[2], 'ATGATGGCC')
        self.assertEqual(alignment[3], 'ATGATGGCC')

class TestAlign_mapall(unittest.TestCase):

    def test_mapall(self):
        if False:
            while True:
                i = 10
        assemblies = (('panTro5', 'panTro6'), ('hg19', 'hg38'), ('rheMac8', 'rheMac10'), ('calJac3', 'calJac4'), ('mm10', 'mm39'), ('rn6', 'rn7'))
        alignments = []
        records = []
        for (old_assembly, new_assembly) in assemblies:
            new_assembly_capitalized = new_assembly[0].upper() + new_assembly[1:]
            filename = f'{old_assembly}To{new_assembly_capitalized}.chain'
            path = os.path.join('Blat', filename)
            alignment = Align.read(path, 'chain')
            alignments.append(alignment)
            filename = '%s.fa' % new_assembly
            path = os.path.join('Align', filename)
            record = SeqIO.read(path, 'fasta')
            (chromosome, location) = record.id.split(':')
            (start, end) = location.split('-')
            start = int(start)
            end = int(end)
            data = {start: str(record.seq)}
            length = len(alignment.query)
            seq = Seq(data, length=length)
            name = '%s.%s' % (new_assembly, chromosome)
            record = SeqRecord(seq, id=name)
            records.append(record)
        path = os.path.join('Blat', 'panTro5.maf')
        alignment = Align.read(path, 'maf')
        self.assertEqual(str(alignment), 'panTro5.c 133922962 ---ACTAGTTA--CA----GTAACAGAAAATAAAATTTAAATAGAAACTTAAAggcc\nhg19.chr1 155784573 ---ACTAGTTA--CA----GTAACAGAAAATAAAATTTAAATAGAAACTTAAAggcc\nrheMac8.c 130383910 ---ACTAGTTA--CA----GTAACAGAAAATAAAATTTAAATAGAAACTTAAAggcc\ncalJac3.c   9790455 ---ACTAGTTA--CA----GTAACAGAAAATAAAATTTAAATAGAAGCTTAAAggct\nmm10.chr3  88858039 TATAATAATTGTATATGTCACAGAAAAAAATGAATTTTCAAT---GACTTAATAGCC\nrn6.chr2  188162970 TACAATAATTG--TATGTCATAGAAAAAAATGAATTTTCAAT---AACTTAATAGCC\n\npanTro5.c 133923010\nhg19.chr1 155784621\nrheMac8.c 130383958\ncalJac3.c   9790503\nmm10.chr3  88857985\nrn6.chr2  188162918\n')
        alignment = alignment.mapall(alignments)
        for (i, record) in enumerate(records):
            sequence = alignment.sequences[i]
            self.assertEqual(len(record), len(sequence))
            (name, chromosome) = record.id.split('.')
            self.assertEqual(sequence.id, chromosome)
            alignment.sequences[i] = record
        self.assertEqual(str(alignment), 'panTro6.c 130611000 ---ACTAGTTA--CA----GTAACAGAAAATAAAATTTAAATAGAAACTTAAAggcc\nhg38.chr1 155814782 ---ACTAGTTA--CA----GTAACAGAAAATAAAATTTAAATAGAAACTTAAAggcc\nrheMac10.  95186253 ---ACTAGTTA--CA----GTAACAGAAAATAAAATTTAAATAGAAACTTAAAggcc\ncalJac4.c   9758318 ---ACTAGTTA--CA----GTAACAGAaaataaaatttaaatagaagcttaaaggct\nmm39.chr3  88765346 TATAATAATTGTATATGTCACAGAAAAAAATGAATTTTCAAT---GACTTAATAGCC\nrn7.chr2  174256702 TACAATAATTG--TATGTCATAGAAAAAAATGAATTTTCAAT---AACTTAATAGCC\n\npanTro6.c 130611048\nhg38.chr1 155814830\nrheMac10.  95186205\ncalJac4.c   9758366\nmm39.chr3  88765292\nrn7.chr2  174256650\n')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)