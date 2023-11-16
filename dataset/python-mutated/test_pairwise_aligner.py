"""Tests for the PairwiseAligner in Bio.Align."""
import array
import os
import unittest
try:
    import numpy as np
except ImportError:
    from Bio import MissingPythonDependencyError
    raise MissingPythonDependencyError('Install numpy if you want to use Bio.Align.') from None
from Bio import Align, SeqIO
from Bio.Seq import Seq, reverse_complement
from Bio.SeqRecord import SeqRecord

class TestAlignerProperties(unittest.TestCase):

    def test_aligner_property_epsilon(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        self.assertAlmostEqual(aligner.epsilon, 1e-06)
        aligner.epsilon = 0.0001
        self.assertAlmostEqual(aligner.epsilon, 0.0001)
        aligner.epsilon = 1e-08
        self.assertAlmostEqual(aligner.epsilon, 1e-08)
        with self.assertRaises(TypeError):
            aligner.epsilon = 'not a number'
        with self.assertRaises(TypeError):
            aligner.epsilon = None

    def test_aligner_property_mode(self):
        if False:
            return 10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        self.assertEqual(aligner.mode, 'global')
        aligner.mode = 'local'
        self.assertEqual(aligner.mode, 'local')
        with self.assertRaises(ValueError):
            aligner.mode = 'wrong'

    def test_aligner_property_match_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        aligner.match_score = 3.0
        self.assertAlmostEqual(aligner.match_score, 3.0)
        aligner.mismatch_score = -2.0
        self.assertAlmostEqual(aligner.mismatch_score, -2.0)
        with self.assertRaises(ValueError):
            aligner.match_score = 'not a number'
        with self.assertRaises(ValueError):
            aligner.mismatch_score = 'not a number'
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 3.000000\n  mismatch_score: -2.000000\n  target_internal_open_gap_score: 0.000000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: 0.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: 0.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: 0.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: 0.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: 0.000000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')

    def test_aligner_property_gapscores(self):
        if False:
            return 10
        aligner = Align.PairwiseAligner()
        (open_score, extend_score) = (-5, -1)
        aligner.target_open_gap_score = open_score
        aligner.target_extend_gap_score = extend_score
        self.assertAlmostEqual(aligner.target_open_gap_score, open_score)
        self.assertAlmostEqual(aligner.target_extend_gap_score, extend_score)
        (open_score, extend_score) = (-6, -7)
        aligner.query_open_gap_score = open_score
        aligner.query_extend_gap_score = extend_score
        self.assertAlmostEqual(aligner.query_open_gap_score, open_score)
        self.assertAlmostEqual(aligner.query_extend_gap_score, extend_score)
        (open_score, extend_score) = (-3, -9)
        aligner.target_end_open_gap_score = open_score
        aligner.target_end_extend_gap_score = extend_score
        self.assertAlmostEqual(aligner.target_end_open_gap_score, open_score)
        self.assertAlmostEqual(aligner.target_end_extend_gap_score, extend_score)
        (open_score, extend_score) = (-1, -2)
        aligner.query_end_open_gap_score = open_score
        aligner.query_end_extend_gap_score = extend_score
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -5.000000\n  target_internal_extend_gap_score: -1.000000\n  target_left_open_gap_score: -3.000000\n  target_left_extend_gap_score: -9.000000\n  target_right_open_gap_score: -3.000000\n  target_right_extend_gap_score: -9.000000\n  query_internal_open_gap_score: -6.000000\n  query_internal_extend_gap_score: -7.000000\n  query_left_open_gap_score: -1.000000\n  query_left_extend_gap_score: -2.000000\n  query_right_open_gap_score: -1.000000\n  query_right_extend_gap_score: -2.000000\n  mode: global\n')
        self.assertAlmostEqual(aligner.query_end_open_gap_score, open_score)
        self.assertAlmostEqual(aligner.query_end_extend_gap_score, extend_score)
        score = -3
        aligner.target_gap_score = score
        self.assertAlmostEqual(aligner.target_gap_score, score)
        self.assertAlmostEqual(aligner.target_open_gap_score, score)
        self.assertAlmostEqual(aligner.target_extend_gap_score, score)
        score = -2
        aligner.query_gap_score = score
        self.assertAlmostEqual(aligner.query_gap_score, score)
        self.assertAlmostEqual(aligner.query_open_gap_score, score)
        self.assertAlmostEqual(aligner.query_extend_gap_score, score)
        score = -4
        aligner.target_end_gap_score = score
        self.assertAlmostEqual(aligner.target_end_gap_score, score)
        self.assertAlmostEqual(aligner.target_end_open_gap_score, score)
        self.assertAlmostEqual(aligner.target_end_extend_gap_score, score)
        self.assertAlmostEqual(aligner.target_left_gap_score, score)
        self.assertAlmostEqual(aligner.target_left_open_gap_score, score)
        self.assertAlmostEqual(aligner.target_left_extend_gap_score, score)
        self.assertAlmostEqual(aligner.target_right_gap_score, score)
        self.assertAlmostEqual(aligner.target_right_open_gap_score, score)
        self.assertAlmostEqual(aligner.target_right_extend_gap_score, score)
        score = -5
        aligner.query_end_gap_score = score
        self.assertAlmostEqual(aligner.query_end_gap_score, score)
        self.assertAlmostEqual(aligner.query_end_open_gap_score, score)
        self.assertAlmostEqual(aligner.query_end_extend_gap_score, score)
        self.assertAlmostEqual(aligner.query_left_gap_score, score)
        self.assertAlmostEqual(aligner.query_left_open_gap_score, score)
        self.assertAlmostEqual(aligner.query_left_extend_gap_score, score)
        self.assertAlmostEqual(aligner.query_right_gap_score, score)
        self.assertAlmostEqual(aligner.query_right_open_gap_score, score)
        self.assertAlmostEqual(aligner.query_right_extend_gap_score, score)
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -3.000000\n  target_internal_extend_gap_score: -3.000000\n  target_left_open_gap_score: -4.000000\n  target_left_extend_gap_score: -4.000000\n  target_right_open_gap_score: -4.000000\n  target_right_extend_gap_score: -4.000000\n  query_internal_open_gap_score: -2.000000\n  query_internal_extend_gap_score: -2.000000\n  query_left_open_gap_score: -5.000000\n  query_left_extend_gap_score: -5.000000\n  query_right_open_gap_score: -5.000000\n  query_right_extend_gap_score: -5.000000\n  mode: global\n')
        with self.assertRaises(ValueError):
            aligner.target_gap_score = 'wrong'
        with self.assertRaises(ValueError):
            aligner.query_gap_score = 'wrong'
        with self.assertRaises(TypeError):
            aligner.target_end_gap_score = 'wrong'
        with self.assertRaises(TypeError):
            aligner.query_end_gap_score = 'wrong'

    def test_aligner_nonexisting_property(self):
        if False:
            while True:
                i = 10
        aligner = Align.PairwiseAligner()
        with self.assertRaises(AttributeError) as cm:
            aligner.no_such_property
        self.assertEqual(str(cm.exception), "'PairwiseAligner' object has no attribute 'no_such_property'")
        with self.assertRaises(AttributeError) as cm:
            aligner.no_such_property = 1
        self.assertEqual(str(cm.exception), "'PairwiseAligner' object has no attribute 'no_such_property'")

class TestPairwiseGlobal(unittest.TestCase):

    def test_needlemanwunsch_simple1(self):
        if False:
            print('Hello World!')
        seq1 = 'GAACT'
        seq2 = 'GAT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: 0.000000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: 0.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: 0.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: 0.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: 0.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: 0.000000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')
        self.assertEqual(aligner.algorithm, 'Needleman-Wunsch')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), '-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GAACT 5\n                  0 ||--| 5\nquery             0 GA--T 3\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 5]], [[0, 2], [2, 3]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GAACT 5\n                  0 |-|-| 5\nquery             0 G-A-T 3\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3], [4, 5]], [[0, 1], [1, 2], [2, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GAACT 5\n                  0 ||--| 5\nquery             3 GA--T 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 5]], [[3, 1], [1, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GAACT 5\n                  0 |-|-| 5\nquery             3 G-A-T 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3], [4, 5]], [[3, 2], [2, 1], [1, 0]]])))

    def test_align_affine1_score(self):
        if False:
            i = 10
            return i + 15
        seq1 = 'CC'
        seq2 = 'ACCT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 0
        aligner.mismatch_score = -1
        aligner.open_gap_score = -5
        aligner.extend_gap_score = -1
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 0.000000\n  mismatch_score: -1.000000\n  target_internal_open_gap_score: -5.000000\n  target_internal_extend_gap_score: -1.000000\n  target_left_open_gap_score: -5.000000\n  target_left_extend_gap_score: -1.000000\n  target_right_open_gap_score: -5.000000\n  target_right_extend_gap_score: -1.000000\n  query_internal_open_gap_score: -5.000000\n  query_internal_extend_gap_score: -1.000000\n  query_left_open_gap_score: -5.000000\n  query_left_extend_gap_score: -1.000000\n  query_right_open_gap_score: -5.000000\n  query_right_extend_gap_score: -1.000000\n  mode: global\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, -7.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, -7.0)

class TestPairwiseLocal(unittest.TestCase):

    def test_smithwaterman(self):
        if False:
            while True:
                i = 10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.gap_score = -0.1
        self.assertEqual(aligner.algorithm, 'Smith-Waterman')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.100000\n  target_internal_extend_gap_score: -0.100000\n  target_left_open_gap_score: -0.100000\n  target_left_extend_gap_score: -0.100000\n  target_right_open_gap_score: -0.100000\n  target_right_extend_gap_score: -0.100000\n  query_internal_open_gap_score: -0.100000\n  query_internal_extend_gap_score: -0.100000\n  query_left_open_gap_score: -0.100000\n  query_left_extend_gap_score: -0.100000\n  query_right_open_gap_score: -0.100000\n  query_right_extend_gap_score: -0.100000\n  mode: local\n')
        score = aligner.score('AwBw', 'zABz')
        self.assertAlmostEqual(score, 1.9)
        alignments = aligner.align('AwBw', 'zABz')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 AwB 3\n                  0 |-| 3\nquery             1 A-B 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[1, 2], [2, 3]]])))

    def test_gotoh_local(self):
        if False:
            return 10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = -0.1
        aligner.extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.100000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.100000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.100000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.100000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.100000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.100000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n')
        score = aligner.score('AwBw', 'zABz')
        self.assertAlmostEqual(score, 1.9)
        alignments = aligner.align('AwBw', 'zABz')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 AwB 3\n                  0 |-| 3\nquery             1 A-B 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[1, 2], [2, 3]]])))

class TestUnknownCharacter(unittest.TestCase):

    def test_needlemanwunsch_simple1(self):
        if False:
            i = 10
            return i + 15
        seq1 = 'GACT'
        seq2 = 'GA?T'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.gap_score = -1.0
        aligner.mismatch_score = -1.0
        aligner.wildcard = '?'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 ||.| 4\nquery             0 GA?T 4\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 4]], [[0, 4]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 ||.| 4\nquery             4 GA?T 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 4]], [[4, 0]]])))
        seq2 = 'GAXT'
        aligner.wildcard = 'X'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 ||.| 4\nquery             0 GAXT 4\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 4]], [[0, 4]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 ||.| 4\nquery             4 GAXT 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 4]], [[4, 0]]])))
        aligner.wildcard = None
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 2.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 2.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 ||.| 4\nquery             0 GAXT 4\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 4]], [[0, 4]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 ||.| 4\nquery             4 GAXT 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 4]], [[4, 0]]])))

    def test_needlemanwunsch_simple2(self):
        if False:
            i = 10
            return i + 15
        seq1 = 'GA?AT'
        seq2 = 'GAA?T'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.wildcard = '?'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 4.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 4.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 4.0)
        self.assertEqual(str(alignment), 'target            0 GA?A-T 5\n                  0 ||-|-| 6\nquery             0 GA-A?T 5\n')
        self.assertEqual(alignment.shape, (2, 6))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4], [4, 5]], [[0, 2], [2, 3], [4, 5]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 4.0)
        self.assertEqual(str(alignment), 'target            0 GA?A-T 5\n                  0 ||-|-| 6\nquery             5 GA-A?T 0\n')
        self.assertEqual(alignment.shape, (2, 6))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4], [4, 5]], [[5, 3], [3, 2], [1, 0]]])))
        seq1 = 'GAXAT'
        seq2 = 'GAAXT'
        aligner.wildcard = 'X'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 4.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 4.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 4.0)
        self.assertEqual(str(alignment), 'target            0 GAXA-T 5\n                  0 ||-|-| 6\nquery             0 GA-AXT 5\n')
        self.assertEqual(alignment.shape, (2, 6))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4], [4, 5]], [[0, 2], [2, 3], [4, 5]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 4.0)
        self.assertEqual(str(alignment), 'target            0 GAXA-T 5\n                  0 ||-|-| 6\nquery             5 GA-AXT 0\n')
        self.assertEqual(alignment.shape, (2, 6))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4], [4, 5]], [[5, 3], [3, 2], [1, 0]]])))

class TestPairwiseOpenPenalty(unittest.TestCase):

    def test_match_score_open_penalty1(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 2
        aligner.mismatch_score = -1
        aligner.open_gap_score = -0.1
        aligner.extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 2.000000\n  mismatch_score: -1.000000\n  target_internal_open_gap_score: -0.100000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.100000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.100000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.100000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.100000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.100000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')
        seq1 = 'AA'
        seq2 = 'A'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 1.9)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 1.9)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 AA 2\n                  0 -| 2\nquery             0 -A 1\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[1, 2]], [[0, 1]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 AA 2\n                  0 |- 2\nquery             0 A- 1\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1]], [[0, 1]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 AA 2\n                  0 -| 2\nquery             1 -A 0\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[1, 2]], [[1, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 AA 2\n                  0 |- 2\nquery             1 A- 0\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1]], [[1, 0]]])))

    def test_match_score_open_penalty2(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 1.5
        aligner.mismatch_score = 0.0
        aligner.open_gap_score = -0.1
        aligner.extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.500000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.100000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.100000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.100000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.100000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.100000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.100000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')
        seq1 = 'GAA'
        seq2 = 'GA'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 2.9)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 2.9)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.9)
        self.assertEqual(str(alignment), 'target            0 GAA 3\n                  0 |-| 3\nquery             0 G-A 2\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[0, 1], [1, 2]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 2.9)
        self.assertEqual(str(alignment), 'target            0 GAA 3\n                  0 ||- 3\nquery             0 GA- 2\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[0, 2]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.9)
        self.assertEqual(str(alignment), 'target            0 GAA 3\n                  0 |-| 3\nquery             2 G-A 0\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[2, 1], [1, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 2.9)
        self.assertEqual(str(alignment), 'target            0 GAA 3\n                  0 ||- 3\nquery             2 GA- 0\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[2, 0]]])))

    def test_match_score_open_penalty3(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.query_open_gap_score = -0.1
        aligner.query_extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: 0.000000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: 0.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: 0.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.100000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.100000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.100000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')
        seq1 = 'GAACT'
        seq2 = 'GAT'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 2.9)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 2.9)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.9)
        self.assertEqual(str(alignment), 'target            0 GAACT 5\n                  0 ||--| 5\nquery             0 GA--T 3\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 5]], [[0, 2], [2, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.9)
        self.assertEqual(str(alignment), 'target            0 GAACT 5\n                  0 ||--| 5\nquery             3 GA--T 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 5]], [[3, 1], [1, 0]]])))

    def test_match_score_open_penalty4(self):
        if False:
            return 10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.mismatch_score = -2.0
        aligner.open_gap_score = -0.1
        aligner.extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -2.000000\n  target_internal_open_gap_score: -0.100000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.100000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.100000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.100000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.100000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.100000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')
        seq1 = 'GCT'
        seq2 = 'GATA'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 1.7)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 1.7)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 G-CT- 3\n                  0 |--|- 5\nquery             0 GA-TA 4\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 GC-T- 3\n                  0 |--|- 5\nquery             0 G-ATA 4\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[0, 1], [2, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 G-CT- 3\n                  0 |--|- 5\nquery             4 GA-TA 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[4, 3], [2, 1]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 GC-T- 3\n                  0 |--|- 5\nquery             4 G-ATA 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[4, 3], [2, 1]]])))

class TestPairwiseExtendPenalty(unittest.TestCase):

    def test_extend_penalty1(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -0.2
        aligner.extend_gap_score = -0.5
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.200000\n  target_internal_extend_gap_score: -0.500000\n  target_left_open_gap_score: -0.200000\n  target_left_extend_gap_score: -0.500000\n  target_right_open_gap_score: -0.200000\n  target_right_extend_gap_score: -0.500000\n  query_internal_open_gap_score: -0.200000\n  query_internal_extend_gap_score: -0.500000\n  query_left_open_gap_score: -0.200000\n  query_left_extend_gap_score: -0.500000\n  query_right_open_gap_score: -0.200000\n  query_right_extend_gap_score: -0.500000\n  mode: global\n')
        seq1 = 'GACT'
        seq2 = 'GT'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 1.3)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 1.3)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.3)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |--| 4\nquery             0 G--T 2\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [3, 4]], [[0, 1], [1, 2]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.3)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |--| 4\nquery             2 G--T 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [3, 4]], [[2, 1], [1, 0]]])))

    def test_extend_penalty2(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -0.2
        aligner.extend_gap_score = -1.5
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.200000\n  target_internal_extend_gap_score: -1.500000\n  target_left_open_gap_score: -0.200000\n  target_left_extend_gap_score: -1.500000\n  target_right_open_gap_score: -0.200000\n  target_right_extend_gap_score: -1.500000\n  query_internal_open_gap_score: -0.200000\n  query_internal_extend_gap_score: -1.500000\n  query_left_open_gap_score: -0.200000\n  query_left_extend_gap_score: -1.500000\n  query_right_open_gap_score: -0.200000\n  query_right_extend_gap_score: -1.500000\n  mode: global\n')
        seq1 = 'GACT'
        seq2 = 'GT'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 0.6)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 0.6)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 0.6)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 -.-| 4\nquery             0 -G-T 2\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[1, 2], [3, 4]], [[0, 1], [1, 2]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 0.6)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |-.- 4\nquery             0 G-T- 2\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[0, 1], [1, 2]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 0.6)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 -.-| 4\nquery             2 -G-T 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[1, 2], [3, 4]], [[2, 1], [1, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 0.6)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |-.- 4\nquery             2 G-T- 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[2, 1], [1, 0]]])))

class TestPairwisePenalizeExtendWhenOpening(unittest.TestCase):

    def test_penalize_extend_when_opening(self):
        if False:
            while True:
                i = 10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -1.7
        aligner.extend_gap_score = -1.5
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -1.700000\n  target_internal_extend_gap_score: -1.500000\n  target_left_open_gap_score: -1.700000\n  target_left_extend_gap_score: -1.500000\n  target_right_open_gap_score: -1.700000\n  target_right_extend_gap_score: -1.500000\n  query_internal_open_gap_score: -1.700000\n  query_internal_extend_gap_score: -1.500000\n  query_left_open_gap_score: -1.700000\n  query_left_extend_gap_score: -1.500000\n  query_right_open_gap_score: -1.700000\n  query_right_extend_gap_score: -1.500000\n  mode: global\n')
        seq1 = 'GACT'
        seq2 = 'GT'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, -1.2)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, -1.2)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, -1.2)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |--| 4\nquery             0 G--T 2\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [3, 4]], [[0, 1], [1, 2]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, -1.2)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |--| 4\nquery             2 G--T 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [3, 4]], [[2, 1], [1, 0]]])))

class TestPairwisePenalizeEndgaps(unittest.TestCase):

    def test_penalize_end_gaps(self):
        if False:
            while True:
                i = 10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -0.2
        aligner.extend_gap_score = -0.8
        end_score = 0.0
        aligner.target_end_gap_score = end_score
        aligner.query_end_gap_score = end_score
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.200000\n  target_internal_extend_gap_score: -0.800000\n  target_left_open_gap_score: 0.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: 0.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.200000\n  query_internal_extend_gap_score: -0.800000\n  query_left_open_gap_score: 0.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: 0.000000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        seq1 = 'GACT'
        seq2 = 'GT'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 1.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 1.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 3)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 --.| 4\nquery             0 --GT 2\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 4]], [[0, 2]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |--| 4\nquery             0 G--T 2\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [3, 4]], [[0, 1], [1, 2]]])))
        alignment = alignments[2]
        self.assertAlmostEqual(alignment.score, 1.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |.-- 4\nquery             0 GT-- 2\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[0, 2]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 3)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 --.| 4\nquery             2 --GT 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 4]], [[2, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |--| 4\nquery             2 G--T 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [3, 4]], [[2, 1], [1, 0]]])))
        alignment = alignments[2]
        self.assertAlmostEqual(alignment.score, 1.0)
        self.assertEqual(str(alignment), 'target            0 GACT 4\n                  0 |.-- 4\nquery             2 GT-- 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[2, 0]]])))

class TestPairwiseSeparateGapPenalties(unittest.TestCase):

    def test_separate_gap_penalties1(self):
        if False:
            for i in range(10):
                print('nop')
        seq1 = 'GAT'
        seq2 = 'GTCT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        (open_score, extend_score) = (-0.3, 0)
        aligner.target_open_gap_score = open_score
        aligner.target_extend_gap_score = extend_score
        aligner.target_end_open_gap_score = open_score
        aligner.target_end_extend_gap_score = extend_score
        (open_score, extend_score) = (-0.8, 0)
        aligner.query_open_gap_score = open_score
        aligner.query_extend_gap_score = extend_score
        aligner.query_end_open_gap_score = open_score
        aligner.query_end_extend_gap_score = extend_score
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.300000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.300000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.300000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.800000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.800000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.800000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 1.7)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 1.7)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 G-AT 3\n                  0 |-.| 4\nquery             0 GTCT 4\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [1, 3]], [[0, 1], [2, 4]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 GA-T 3\n                  0 |.-| 4\nquery             0 GTCT 4\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [2, 3]], [[0, 2], [3, 4]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 G-AT 3\n                  0 |-.| 4\nquery             4 GTCT 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [1, 3]], [[4, 3], [2, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.7)
        self.assertEqual(str(alignment), 'target            0 GA-T 3\n                  0 |.-| 4\nquery             4 GTCT 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [2, 3]], [[4, 2], [1, 0]]])))

    def test_separate_gap_penalties2(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.target_open_gap_score = -0.3
        aligner.target_extend_gap_score = 0.0
        aligner.query_open_gap_score = -0.2
        aligner.query_extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.300000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.300000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.300000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.200000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.200000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.200000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n')
        seq1 = 'GAT'
        seq2 = 'GTCT'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 1.8)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 1.8)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.8)
        self.assertEqual(str(alignment), 'target            0 GAT 3\n                  0 |-| 3\nquery             0 G-T 2\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[0, 1], [1, 2]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.8)
        self.assertEqual(str(alignment), 'target            0 GAT 3\n                  0 |-| 3\nquery             4 G-T 2\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[4, 3], [3, 2]]])))

class TestPairwiseSeparateGapPenaltiesWithExtension(unittest.TestCase):

    def test_separate_gap_penalties_with_extension(self):
        if False:
            return 10
        seq1 = 'GAAT'
        seq2 = 'GTCCT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        (open_score, extend_score) = (-0.1, 0)
        aligner.target_open_gap_score = open_score
        aligner.target_extend_gap_score = extend_score
        aligner.target_end_open_gap_score = open_score
        aligner.target_end_extend_gap_score = extend_score
        score = -0.1
        aligner.query_gap_score = score
        aligner.query_end_gap_score = score
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.100000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.100000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.100000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.100000\n  query_internal_extend_gap_score: -0.100000\n  query_left_open_gap_score: -0.100000\n  query_left_extend_gap_score: -0.100000\n  query_right_open_gap_score: -0.100000\n  query_right_extend_gap_score: -0.100000\n  mode: local\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 1.9)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 1.9)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 3)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 G-AAT 4\n                  0 |-..| 5\nquery             0 GTCCT 5\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [1, 4]], [[0, 1], [2, 5]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 GA-AT 4\n                  0 |.-.| 5\nquery             0 GTCCT 5\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [2, 4]], [[0, 2], [3, 5]]])))
        alignment = alignments[2]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 GAA-T 4\n                  0 |..-| 5\nquery             0 GTCCT 5\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3], [3, 4]], [[0, 3], [4, 5]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 3)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 G-AAT 4\n                  0 |-..| 5\nquery             5 GTCCT 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [1, 4]], [[5, 4], [3, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 GA-AT 4\n                  0 |.-.| 5\nquery             5 GTCCT 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [2, 4]], [[5, 3], [2, 0]]])))
        alignment = alignments[2]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'target            0 GAA-T 4\n                  0 |..-| 5\nquery             5 GTCCT 0\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3], [3, 4]], [[5, 2], [1, 0]]])))

class TestPairwiseMatchDictionary(unittest.TestCase):
    match_dict = {('A', 'A'): 1.5, ('A', 'T'): 0.5, ('T', 'A'): 0.5, ('T', 'T'): 1.0}

    def test_match_dictionary1(self):
        if False:
            while True:
                i = 10
        try:
            from Bio.Align import substitution_matrices
        except ImportError:
            return
        substitution_matrix = substitution_matrices.Array(data=self.match_dict)
        seq1 = 'ATAT'
        seq2 = 'ATT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrix
        aligner.open_gap_score = -0.5
        aligner.extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*>\n  target_internal_open_gap_score: -0.500000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.500000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.500000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.500000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.500000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.500000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n$')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             0 ATT 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[0, 3]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATAT 4\n                  0 ||-| 4\nquery             0 AT-T 3\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4]], [[0, 2], [2, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             3 ATT 0\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[3, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATAT 4\n                  0 ||-| 4\nquery             3 AT-T 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4]], [[3, 1], [1, 0]]])))

    def test_match_dictionary2(self):
        if False:
            while True:
                i = 10
        try:
            from Bio.Align import substitution_matrices
        except ImportError:
            return
        substitution_matrix = substitution_matrices.Array(data=self.match_dict)
        seq1 = 'ATAT'
        seq2 = 'ATT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrix
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = 0.0
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*>\n  target_internal_open_gap_score: -1.000000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -1.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -1.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -1.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -1.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -1.000000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n$')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             0 ATT 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[0, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             3 ATT 0\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[3, 0]]])))

    def test_match_dictionary3(self):
        if False:
            while True:
                i = 10
        try:
            from Bio.Align import substitution_matrices
        except ImportError:
            return
        substitution_matrix = substitution_matrices.Array(data=self.match_dict)
        seq1 = 'ATT'
        seq2 = 'ATAT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrix
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = 0.0
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*>\n  target_internal_open_gap_score: -1.000000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -1.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -1.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -1.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -1.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -1.000000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n$')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATT 3\n                  0 ||. 3\nquery             0 ATA 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[0, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATT 3\n                  0 ||. 3\nquery             4 ATA 1\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[4, 1]]])))

    def test_match_dictionary4(self):
        if False:
            while True:
                i = 10
        try:
            from Bio.Align import substitution_matrices
        except ImportError:
            return
        substitution_matrix = substitution_matrices.Array(alphabet='AT', dims=2)
        self.assertEqual(substitution_matrix.shape, (2, 2))
        substitution_matrix.update(self.match_dict)
        seq1 = 'ATAT'
        seq2 = 'ATT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrix
        aligner.open_gap_score = -0.5
        aligner.extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*>\n  target_internal_open_gap_score: -0.500000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -0.500000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -0.500000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -0.500000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -0.500000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -0.500000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n$')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             0 ATT 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[0, 3]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATAT 4\n                  0 ||-| 4\nquery             0 AT-T 3\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4]], [[0, 2], [2, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             3 ATT 0\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[3, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATAT 4\n                  0 ||-| 4\nquery             3 AT-T 0\n')
        self.assertEqual(alignment.shape, (2, 4))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4]], [[3, 1], [1, 0]]])))

    def test_match_dictionary5(self):
        if False:
            i = 10
            return i + 15
        try:
            from Bio.Align import substitution_matrices
        except ImportError:
            return
        substitution_matrix = substitution_matrices.Array(alphabet='AT', dims=2)
        self.assertEqual(substitution_matrix.shape, (2, 2))
        substitution_matrix.update(self.match_dict)
        seq1 = 'ATAT'
        seq2 = 'ATT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrix
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = 0.0
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*\n  target_internal_open_gap_score: -1.000000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -1.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -1.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -1.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -1.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -1.000000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n$')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             0 ATT 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[0, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATA 3\n                  0 ||. 3\nquery             3 ATT 0\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[3, 0]]])))

    def test_match_dictionary6(self):
        if False:
            print('Hello World!')
        try:
            from Bio.Align import substitution_matrices
        except ImportError:
            return
        substitution_matrix = substitution_matrices.Array(alphabet='AT', dims=2)
        self.assertEqual(substitution_matrix.shape, (2, 2))
        substitution_matrix.update(self.match_dict)
        seq1 = 'ATT'
        seq2 = 'ATAT'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.substitution_matrix = substitution_matrix
        aligner.open_gap_score = -1.0
        aligner.extend_gap_score = 0.0
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*>\n  target_internal_open_gap_score: -1.000000\n  target_internal_extend_gap_score: 0.000000\n  target_left_open_gap_score: -1.000000\n  target_left_extend_gap_score: 0.000000\n  target_right_open_gap_score: -1.000000\n  target_right_extend_gap_score: 0.000000\n  query_internal_open_gap_score: -1.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: -1.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: -1.000000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n$')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATT 3\n                  0 ||. 3\nquery             0 ATA 3\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[0, 3]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'target            0 ATT 3\n                  0 ||. 3\nquery             4 ATA 1\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3]], [[4, 1]]])))

class TestPairwiseOneCharacter(unittest.TestCase):

    def test_align_one_char1(self):
        if False:
            i = 10
            return i + 15
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = -0.3
        aligner.extend_gap_score = -0.1
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.300000\n  target_internal_extend_gap_score: -0.100000\n  target_left_open_gap_score: -0.300000\n  target_left_extend_gap_score: -0.100000\n  target_right_open_gap_score: -0.300000\n  target_right_extend_gap_score: -0.100000\n  query_internal_open_gap_score: -0.300000\n  query_internal_extend_gap_score: -0.100000\n  query_left_open_gap_score: -0.300000\n  query_left_extend_gap_score: -0.100000\n  query_right_open_gap_score: -0.300000\n  query_right_extend_gap_score: -0.100000\n  mode: local\n')
        score = aligner.score('abcde', 'c')
        self.assertAlmostEqual(score, 1)
        alignments = aligner.align('abcde', 'c')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1)
        self.assertEqual(str(alignment), 'target            2 c 3\n                  0 | 1\nquery             0 c 1\n')
        self.assertEqual(alignment.shape, (2, 1))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 3]], [[0, 1]]])))

    def test_align_one_char2(self):
        if False:
            i = 10
            return i + 15
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.open_gap_score = -0.3
        aligner.extend_gap_score = -0.1
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.300000\n  target_internal_extend_gap_score: -0.100000\n  target_left_open_gap_score: -0.300000\n  target_left_extend_gap_score: -0.100000\n  target_right_open_gap_score: -0.300000\n  target_right_extend_gap_score: -0.100000\n  query_internal_open_gap_score: -0.300000\n  query_internal_extend_gap_score: -0.100000\n  query_left_open_gap_score: -0.300000\n  query_left_extend_gap_score: -0.100000\n  query_right_open_gap_score: -0.300000\n  query_right_extend_gap_score: -0.100000\n  mode: local\n')
        score = aligner.score('abcce', 'c')
        self.assertAlmostEqual(score, 1)
        alignments = aligner.align('abcce', 'c')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1)
        self.assertEqual(str(alignment), 'target            2 c 3\n                  0 | 1\nquery             0 c 1\n')
        self.assertEqual(alignment.shape, (2, 1))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 3]], [[0, 1]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 1)
        self.assertEqual(str(alignment), 'target            3 c 4\n                  0 | 1\nquery             0 c 1\n')
        self.assertEqual(alignment.shape, (2, 1))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[3, 4]], [[0, 1]]])))

    def test_align_one_char3(self):
        if False:
            i = 10
            return i + 15
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -0.3
        aligner.extend_gap_score = -0.1
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.300000\n  target_internal_extend_gap_score: -0.100000\n  target_left_open_gap_score: -0.300000\n  target_left_extend_gap_score: -0.100000\n  target_right_open_gap_score: -0.300000\n  target_right_extend_gap_score: -0.100000\n  query_internal_open_gap_score: -0.300000\n  query_internal_extend_gap_score: -0.100000\n  query_left_open_gap_score: -0.300000\n  query_left_extend_gap_score: -0.100000\n  query_right_open_gap_score: -0.300000\n  query_right_extend_gap_score: -0.100000\n  mode: global\n')
        seq1 = 'abcde'
        seq2 = 'c'
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 0.2)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 0.2)
        self.assertEqual(str(alignment), 'target            0 abcde 5\n                  0 --|-- 5\nquery             0 --c-- 1\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 3]], [[0, 1]]])))

    def test_align_one_char_score3(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.open_gap_score = -0.3
        aligner.extend_gap_score = -0.1
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.300000\n  target_internal_extend_gap_score: -0.100000\n  target_left_open_gap_score: -0.300000\n  target_left_extend_gap_score: -0.100000\n  target_right_open_gap_score: -0.300000\n  target_right_extend_gap_score: -0.100000\n  query_internal_open_gap_score: -0.300000\n  query_internal_extend_gap_score: -0.100000\n  query_left_open_gap_score: -0.300000\n  query_left_extend_gap_score: -0.100000\n  query_right_open_gap_score: -0.300000\n  query_right_extend_gap_score: -0.100000\n  mode: global\n')
        score = aligner.score('abcde', 'c')
        self.assertAlmostEqual(score, 0.2)

class TestPerSiteGapPenalties(unittest.TestCase):
    """Check gap penalty callbacks use correct gap opening position.

    This tests that the gap penalty callbacks are really being used
    with the correct gap opening position.
    """

    def test_gap_here_only_1(self):
        if False:
            return 10
        seq1 = 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA'
        seq2 = 'AABBBAAAACCCCAAAABBBAA'
        breaks = [0, 11, len(seq2)]

        def nogaps(x, y):
            if False:
                return 10
            return -2000 - y

        def specificgaps(x, y):
            if False:
                print('Hello World!')
            if x in breaks:
                return -2 - y
            else:
                return -2000 - y
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.target_gap_score = nogaps
        aligner.query_gap_score = specificgaps
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -1.000000\n  target_gap_function: {nogaps}\n  query_gap_function: {specificgaps}\n  mode: global\n')
        self.assertEqual(aligner.algorithm, 'Waterman-Smith-Beyer global alignment algorithm')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 2)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 2)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2)
        self.assertEqual(str(alignment), 'target            0 AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA 36\n                  0 --|||||||||||----------|||||||||||-- 36\nquery             0 --AABBBAAAACC----------CCAAAABBBAA-- 22\n')
        self.assertEqual(alignment.shape, (2, 36))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 13], [23, 34]], [[0, 11], [11, 22]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2)
        self.assertEqual(str(alignment), 'target            0 AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA 36\n                  0 --|||||||||||----------|||||||||||-- 36\nquery            22 --AABBBAAAACC----------CCAAAABBBAA--  0\n')
        self.assertEqual(alignment.shape, (2, 36))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 13], [23, 34]], [[22, 11], [11, 0]]])))

    def test_gap_here_only_2(self):
        if False:
            while True:
                i = 10
        seq1 = 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA'
        seq2 = 'AABBBAAAACCCCAAAABBBAA'
        breaks = [0, 3, len(seq2)]

        def nogaps(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return -2000 - y

        def specificgaps(x, y):
            if False:
                return 10
            if x in breaks:
                return -2 - y
            else:
                return -2000 - y
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.target_gap_score = nogaps
        aligner.query_gap_score = specificgaps
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -1.000000\n  target_gap_function: {nogaps}\n  query_gap_function: {specificgaps}\n  mode: global\n')
        self.assertEqual(aligner.algorithm, 'Waterman-Smith-Beyer global alignment algorithm')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, -10)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, -10)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, -10)
        self.assertEqual(str(alignment), 'target            0 AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA 36\n                  0 --|||----------......|||||||||||||-- 36\nquery             0 --AAB----------BBAAAACCCCAAAABBBAA-- 22\n')
        self.assertEqual(alignment.shape, (2, 36))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 5], [15, 34]], [[0, 3], [3, 22]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, -10)
        self.assertEqual(str(alignment), 'target            0 AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA 36\n                  0 ||.------------......|||||||||||||-- 36\nquery             0 AAB------------BBAAAACCCCAAAABBBAA-- 22\n')
        self.assertEqual(alignment.shape, (2, 36))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 3], [15, 34]], [[0, 3], [3, 22]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, -10)
        self.assertEqual(str(alignment), 'target            0 AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA 36\n                  0 --|||||||||||||......------------.|| 36\nquery            22 --AABBBAAAACCCCAAAABB------------BAA  0\n')
        self.assertEqual(alignment.shape, (2, 36))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 21], [33, 36]], [[22, 3], [3, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, -10)
        self.assertEqual(str(alignment), 'target            0 AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA 36\n                  0 --|||||||||||||......----------|||-- 36\nquery            22 --AABBBAAAACCCCAAAABB----------BAA--  0\n')
        self.assertEqual(alignment.shape, (2, 36))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 21], [31, 34]], [[22, 3], [3, 0]]])))

    def test_gap_here_only_3(self):
        if False:
            for i in range(10):
                print('nop')
        seq1 = 'TTCCAA'
        seq2 = 'TTGGAA'

        def gap_score(i, n):
            if False:
                for i in range(10):
                    print('nop')
            if i == 3:
                return -10
            if n == 1:
                return -1
            return -10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.match_score = 1
        aligner.mismatch_score = -10
        aligner.target_gap_score = gap_score
        self.assertEqual(aligner.algorithm, 'Waterman-Smith-Beyer global alignment algorithm')
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -10.000000\n  target_gap_function: {gap_score}\n  query_internal_open_gap_score: 0.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: 0.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: 0.000000\n  query_right_extend_gap_score: 0.000000\n  mode: global\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 2.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 2.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 TT-CC-AA 6\n                  0 ||----|| 8\nquery             0 TTG--GAA 6\n')
        self.assertEqual(alignment.shape, (2, 8))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 6]], [[0, 2], [4, 6]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 TT-CC-AA 6\n                  0 ||----|| 8\nquery             6 TTG--GAA 0\n')
        self.assertEqual(alignment.shape, (2, 8))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 6]], [[6, 4], [2, 0]]])))
        aligner.query_gap_score = gap_score
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -10.000000\n  target_gap_function: {gap_score}\n  query_gap_function: {gap_score}\n  mode: global\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, -8.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, -8.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 4)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TT-CCAA 6\n                  0 ||-.-|| 7\nquery             0 TTGG-AA 6\n')
        self.assertEqual(alignment.shape, (2, 7))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [2, 3], [4, 6]], [[0, 2], [3, 4], [4, 6]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TTC--CAA 6\n                  0 ||----|| 8\nquery             0 TT-GG-AA 6\n')
        self.assertEqual(alignment.shape, (2, 8))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 6]], [[0, 2], [4, 6]]])))
        alignment = alignments[2]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TTCC-AA 6\n                  0 ||-.-|| 7\nquery             0 TT-GGAA 6\n')
        self.assertEqual(alignment.shape, (2, 7))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4], [4, 6]], [[0, 2], [2, 3], [4, 6]]])))
        alignment = alignments[3]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TT-CC-AA 6\n                  0 ||----|| 8\nquery             0 TTG--GAA 6\n')
        self.assertEqual(alignment.shape, (2, 8))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 6]], [[0, 2], [4, 6]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 4)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TT-CCAA 6\n                  0 ||-.-|| 7\nquery             6 TTGG-AA 0\n')
        self.assertEqual(alignment.shape, (2, 7))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [2, 3], [4, 6]], [[6, 4], [3, 2], [2, 0]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TTC--CAA 6\n                  0 ||----|| 8\nquery             6 TT-GG-AA 0\n')
        self.assertEqual(alignment.shape, (2, 8))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 6]], [[6, 4], [2, 0]]])))
        alignment = alignments[2]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TTCC-AA 6\n                  0 ||-.-|| 7\nquery             6 TT-GGAA 0\n')
        self.assertEqual(alignment.shape, (2, 7))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [3, 4], [4, 6]], [[6, 4], [4, 3], [2, 0]]])))
        alignment = alignments[3]
        self.assertAlmostEqual(alignment.score, -8.0)
        self.assertEqual(str(alignment), 'target            0 TT-CC-AA 6\n                  0 ||----|| 8\nquery             6 TTG--GAA 0\n')
        self.assertEqual(alignment.shape, (2, 8))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 6]], [[6, 4], [2, 0]]])))

    def test_gap_here_only_local_1(self):
        if False:
            return 10
        seq1 = 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA'
        seq2 = 'AABBBAAAACCCCAAAABBBAA'
        breaks = [0, 11, len(seq2)]

        def nogaps(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return -2000 - y

        def specificgaps(x, y):
            if False:
                return 10
            if x in breaks:
                return -2 - y
            else:
                return -2000 - y
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.target_gap_score = nogaps
        aligner.query_gap_score = specificgaps
        self.assertEqual(aligner.algorithm, 'Waterman-Smith-Beyer local alignment algorithm')
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -1.000000\n  target_gap_function: {nogaps}\n  query_gap_function: {specificgaps}\n  mode: local\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 13)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 13)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target            2 AABBBAAAACCCC 15\n                  0 ||||||||||||| 13\nquery             0 AABBBAAAACCCC 13\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 15]], [[0, 13]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target           21 CCCCAAAABBBAA 34\n                  0 ||||||||||||| 13\nquery             9 CCCCAAAABBBAA 22\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[21, 34]], [[9, 22]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target            2 AABBBAAAACCCC 15\n                  0 ||||||||||||| 13\nquery            22 AABBBAAAACCCC  9\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 15]], [[22, 9]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target           21 CCCCAAAABBBAA 34\n                  0 ||||||||||||| 13\nquery            13 CCCCAAAABBBAA  0\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[21, 34]], [[13, 0]]])))

    def test_gap_here_only_local_2(self):
        if False:
            while True:
                i = 10
        seq1 = 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA'
        seq2 = 'AABBBAAAACCCCAAAABBBAA'
        breaks = [0, 3, len(seq2)]

        def nogaps(x, y):
            if False:
                print('Hello World!')
            return -2000 - y

        def specificgaps(x, y):
            if False:
                return 10
            if x in breaks:
                return -2 - y
            else:
                return -2000 - y
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1
        aligner.mismatch_score = -1
        aligner.target_gap_score = nogaps
        aligner.query_gap_score = specificgaps
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -1.000000\n  target_gap_function: {nogaps}\n  query_gap_function: {specificgaps}\n  mode: local\n')
        self.assertEqual(aligner.algorithm, 'Waterman-Smith-Beyer local alignment algorithm')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 13)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 13)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target            2 AABBBAAAACCCC 15\n                  0 ||||||||||||| 13\nquery             0 AABBBAAAACCCC 13\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 15]], [[0, 13]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target           21 CCCCAAAABBBAA 34\n                  0 ||||||||||||| 13\nquery             9 CCCCAAAABBBAA 22\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[21, 34]], [[9, 22]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target            2 AABBBAAAACCCC 15\n                  0 ||||||||||||| 13\nquery            22 AABBBAAAACCCC  9\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[2, 15]], [[22, 9]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 13)
        self.assertEqual(str(alignment), 'target           21 CCCCAAAABBBAA 34\n                  0 ||||||||||||| 13\nquery            13 CCCCAAAABBBAA  0\n')
        self.assertEqual(alignment.shape, (2, 13))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[21, 34]], [[13, 0]]])))

    def test_gap_here_only_local_3(self):
        if False:
            i = 10
            return i + 15
        seq1 = 'TTCCAA'
        seq2 = 'TTGGAA'

        def gap_score(i, n):
            if False:
                while True:
                    i = 10
            if i == 3:
                return -10
            if n == 1:
                return -1
            return -10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.match_score = 1
        aligner.mismatch_score = -10
        aligner.target_gap_score = gap_score
        self.assertEqual(aligner.algorithm, 'Waterman-Smith-Beyer local alignment algorithm')
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -10.000000\n  target_gap_function: {gap_score}\n  query_internal_open_gap_score: 0.000000\n  query_internal_extend_gap_score: 0.000000\n  query_left_open_gap_score: 0.000000\n  query_left_extend_gap_score: 0.000000\n  query_right_open_gap_score: 0.000000\n  query_right_extend_gap_score: 0.000000\n  mode: local\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 2.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 2.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 TT 2\n                  0 || 2\nquery             0 TT 2\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[0, 2]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            4 AA 6\n                  0 || 2\nquery             4 AA 6\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[4, 6]], [[4, 6]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 TT 2\n                  0 || 2\nquery             6 TT 4\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[6, 4]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            4 AA 6\n                  0 || 2\nquery             2 AA 0\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[4, 6]], [[2, 0]]])))
        aligner.query_gap_score = gap_score
        self.assertEqual(str(aligner), f'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: -10.000000\n  target_gap_function: {gap_score}\n  query_gap_function: {gap_score}\n  mode: local\n')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 2.0)
        score = aligner.score(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(score, 2.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 TT 2\n                  0 || 2\nquery             0 TT 2\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[0, 2]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            4 AA 6\n                  0 || 2\nquery             4 AA 6\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[4, 6]], [[4, 6]]])))
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            0 TT 2\n                  0 || 2\nquery             6 TT 4\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2]], [[6, 4]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 2.0)
        self.assertEqual(str(alignment), 'target            4 AA 6\n                  0 || 2\nquery             2 AA 0\n')
        self.assertEqual(alignment.shape, (2, 2))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[4, 6]], [[2, 0]]])))

    def test_broken_gap_function(self):
        if False:
            print('Hello World!')
        seq1 = 'TTCCAA'
        seq2 = 'TTGGAA'

        def gap_score(i, n):
            if False:
                for i in range(10):
                    print('nop')
            raise RuntimeError('broken gap function')
        aligner = Align.PairwiseAligner()
        aligner.target_gap_score = gap_score
        aligner.query_gap_score = -1
        aligner.mode = 'global'
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, seq2)
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, reverse_complement(seq2), strand='-')
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, seq2)
            alignments = list(alignments)
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
            alignments = list(alignments)
        aligner.mode = 'local'
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, seq2)
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, reverse_complement(seq2), strand='-')
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, seq2)
            alignments = list(alignments)
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
            alignments = list(alignments)
        aligner.target_gap_score = -1
        aligner.query_gap_score = gap_score
        aligner.mode = 'global'
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, seq2)
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, reverse_complement(seq2), strand='-')
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, seq2)
            alignments = list(alignments)
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
            alignments = list(alignments)
        aligner.mode = 'local'
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, seq2)
        with self.assertRaises(RuntimeError):
            aligner.score(seq1, reverse_complement(seq2), strand='-')
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, seq2)
            alignments = list(alignments)
        with self.assertRaises(RuntimeError):
            alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
            alignments = list(alignments)

class TestAlignerInput(unittest.TestCase):
    """Check aligning sequences provided as lists, str, Seq, or SeqRecord objects."""

    def test_three_letter_amino_acids_global(self):
        if False:
            i = 10
            return i + 15
        'Test aligning sequences provided as lists of three-letter amino acids.'
        seq1 = ['Gly', 'Ala', 'Thr']
        seq2 = ['Gly', 'Ala', 'Ala', 'Cys', 'Thr']
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.alphabet = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        self.assertEqual(str(alignments[0]), 'Gly Ala --- --- Thr\n||| ||| --- --- |||\nGly Ala Ala Cys Thr\n')
        self.assertEqual(str(alignments[1]), 'Gly --- Ala --- Thr\n||| --- ||| --- |||\nGly Ala Ala Cys Thr\n')
        self.assertAlmostEqual(alignments[0].score, 3.0)
        self.assertAlmostEqual(alignments[1].score, 3.0)
        seq1 = ['Pro', 'Pro', 'Gly', 'Ala', 'Thr']
        seq2 = ['Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Asn', 'Asn']
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'Pro Pro Gly Ala --- --- Thr --- ---\n--- --- ||| ||| --- --- ||| --- ---\n--- --- Gly Ala Ala Cys Thr Asn Asn\n')
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(alignment[0], ['Pro', 'Pro', 'Gly', 'Ala', None, None, 'Thr', None, None])
        self.assertEqual(alignment[0, :], ['Pro', 'Pro', 'Gly', 'Ala', None, None, 'Thr', None, None])
        self.assertEqual(alignment[0, 1:], ['Pro', 'Gly', 'Ala', None, None, 'Thr', None, None])
        self.assertEqual(alignment[0, ::2], ['Pro', 'Gly', None, 'Thr', None])
        self.assertEqual(alignment[1], [None, None, 'Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Asn', 'Asn'])
        self.assertEqual(alignment[1, :], [None, None, 'Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Asn', 'Asn'])
        self.assertEqual(alignment[1, 1:], [None, 'Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Asn', 'Asn'])
        self.assertEqual(alignment[1, ::2], [None, 'Gly', 'Ala', 'Thr', 'Asn'])
        alignment = alignments[1]
        self.assertEqual(str(alignment), 'Pro Pro Gly --- Ala --- Thr --- ---\n--- --- ||| --- ||| --- ||| --- ---\n--- --- Gly Ala Ala Cys Thr Asn Asn\n')
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(alignment[0], ['Pro', 'Pro', 'Gly', None, 'Ala', None, 'Thr', None, None])
        self.assertEqual(alignment[0, :], ['Pro', 'Pro', 'Gly', None, 'Ala', None, 'Thr', None, None])
        self.assertEqual(alignment[0, 1:-1], ['Pro', 'Gly', None, 'Ala', None, 'Thr', None])
        self.assertEqual(alignment[0, 1::2], ['Pro', None, None, None])
        self.assertEqual(alignment[1], [None, None, 'Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Asn', 'Asn'])
        self.assertEqual(alignment[1, :], [None, None, 'Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Asn', 'Asn'])
        self.assertEqual(alignment[1, 1:-1], [None, 'Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Asn'])
        self.assertEqual(alignment[1, 1::2], [None, 'Ala', 'Cys', 'Asn'])

    def test_three_letter_amino_acids_local(self):
        if False:
            i = 10
            return i + 15
        seq1 = ['Asn', 'Asn', 'Gly', 'Ala', 'Thr', 'Glu', 'Glu']
        seq2 = ['Pro', 'Pro', 'Gly', 'Ala', 'Ala', 'Cys', 'Thr', 'Leu']
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.alphabet = ['Ala', 'Arg', 'Asn', 'Asp', 'Cys', 'Gln', 'Glu', 'Gly', 'His', 'Ile', 'Leu', 'Lys', 'Met', 'Phe', 'Pro', 'Ser', 'Thr', 'Trp', 'Tyr', 'Val']
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'Gly Ala --- --- Thr\n||| ||| --- --- |||\nGly Ala Ala Cys Thr\n')
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(alignment[0], ['Gly', 'Ala', None, None, 'Thr'])
        self.assertEqual(alignment[0, :], ['Gly', 'Ala', None, None, 'Thr'])
        self.assertEqual(alignment[0, 1:], ['Ala', None, None, 'Thr'])
        self.assertEqual(alignment[0, :-1], ['Gly', 'Ala', None, None])
        self.assertEqual(alignment[0, ::2], ['Gly', None, 'Thr'])
        self.assertEqual(alignment[1], ['Gly', 'Ala', 'Ala', 'Cys', 'Thr'])
        self.assertEqual(alignment[1, :], ['Gly', 'Ala', 'Ala', 'Cys', 'Thr'])
        self.assertEqual(alignment[1, 1:], ['Ala', 'Ala', 'Cys', 'Thr'])
        self.assertEqual(alignment[1, :-1], ['Gly', 'Ala', 'Ala', 'Cys'])
        self.assertEqual(alignment[1, ::2], ['Gly', 'Ala', 'Thr'])
        alignment = alignments[1]
        self.assertEqual(str(alignment), 'Gly --- Ala --- Thr\n||| --- ||| --- |||\nGly Ala Ala Cys Thr\n')
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(alignment[0], ['Gly', None, 'Ala', None, 'Thr'])
        self.assertEqual(alignment[0, :], ['Gly', None, 'Ala', None, 'Thr'])
        self.assertEqual(alignment[0, 1:], [None, 'Ala', None, 'Thr'])
        self.assertEqual(alignment[0, :-1], ['Gly', None, 'Ala', None])
        self.assertEqual(alignment[0, ::2], ['Gly', 'Ala', 'Thr'])
        self.assertEqual(alignment[1], ['Gly', 'Ala', 'Ala', 'Cys', 'Thr'])
        self.assertEqual(alignment[1, :], ['Gly', 'Ala', 'Ala', 'Cys', 'Thr'])
        self.assertEqual(alignment[1, 1:], ['Ala', 'Ala', 'Cys', 'Thr'])
        self.assertEqual(alignment[1, :-1], ['Gly', 'Ala', 'Ala', 'Cys'])
        self.assertEqual(alignment[1, ::2], ['Gly', 'Ala', 'Thr'])

    def test_str_seq_seqrecord(self):
        if False:
            return 10
        'Test aligning sequences provided as str, Seq, or SeqRecord objects.'
        aligner = Align.PairwiseAligner('blastn')
        t1 = 'ACGT'
        t2 = 'CGTT'
        s1 = Seq(t1)
        s2 = Seq(t2)
        r1 = SeqRecord(s1, id='first', description='1st sequence')
        r2 = SeqRecord(s2, id='second', description='2nd sequence')
        alignments = aligner.align(t1, t2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ACGT 4\n                  0 ...| 4\nquery             0 CGTT 4\n')
        self.assertEqual(format(alignment, 'fasta'), '>\nACGT\n>\nCGTT\n')
        alignments = aligner.align(s1, s2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ACGT 4\n                  0 ...| 4\nquery             0 CGTT 4\n')
        self.assertEqual(format(alignment, 'fasta'), '>\nACGT\n>\nCGTT\n')
        alignments = aligner.align(r1, r2)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'first             0 ACGT 4\n                  0 ...| 4\nsecond            0 CGTT 4\n')
        self.assertEqual(format(alignment, 'fasta'), '>first 1st sequence\nACGT\n>second 2nd sequence\nCGTT\n')

class TestArgumentErrors(unittest.TestCase):

    def test_aligner_string_errors(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        message = '^sequence has unexpected type int$'
        with self.assertRaisesRegex(TypeError, message):
            aligner.score('AAA', 3)
        message = '^sequence has zero length$'
        with self.assertRaisesRegex(ValueError, message):
            aligner.score('AAA', '')
        with self.assertRaisesRegex(ValueError, message):
            aligner.score('AAA', '', strand='-')
        message = '^sequence contains letters not in the alphabet$'
        aligner.alphabet = 'ABCD'
        with self.assertRaisesRegex(ValueError, message):
            aligner.score('AAA', 'AAE')

    def test_aligner_array_errors(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        s1 = 'GGG'
        s2 = array.array('i', [ord('G'), ord('A'), ord('G')])
        score = aligner.score(s1, s2)
        self.assertAlmostEqual(score, 2.0)
        s2 = array.array('f', [1.0, 0.0, 1.0])
        message = "^sequence has incorrect data type 'f'$"
        with self.assertRaisesRegex(ValueError, message):
            aligner.score(s1, s2)
        aligner.wildcard = chr(99)
        s1 = array.array('i', [1, 5, 6])
        s2 = array.array('i', [1, 8, 6])
        s2a = array.array('i', [1, 8, 99])
        s2b = array.array('i', [1, 28, 6])
        aligner.match = 3.0
        aligner.mismatch = -2.0
        aligner.gap_score = -10.0
        score = aligner.score(s1, s2)
        self.assertAlmostEqual(score, 4.0)
        score = aligner.score(s1, s2a)
        self.assertAlmostEqual(score, 1.0)
        score = aligner.score(s1, s2b)
        self.assertAlmostEqual(score, 4.0)
        try:
            import numpy as np
        except ImportError:
            return
        aligner = Align.PairwiseAligner()
        aligner.wildcard = chr(99)
        s1 = 'GGG'
        s2 = np.array([ord('G'), ord('A'), ord('G')], np.int32)
        score = aligner.score(s1, s2)
        self.assertAlmostEqual(score, 2.0)
        s2 = np.array([1.0, 0.0, 1.0])
        message = "^sequence has incorrect data type 'd'$"
        with self.assertRaisesRegex(ValueError, message):
            aligner.score(s1, s2)
        s2 = np.zeros((3, 2), np.int32)
        message = '^sequence has incorrect rank \\(2 expected 1\\)$'
        with self.assertRaisesRegex(ValueError, message):
            aligner.score(s1, s2)
        s1 = np.array([1, 5, 6], np.int32)
        s2 = np.array([1, 8, 6], np.int32)
        s2a = np.array([1, 8, 99], np.int32)
        s2b = np.array([1, 28, 6], np.int32)
        s2c = np.array([1, 8, -6], np.int32)
        aligner.match = 3.0
        aligner.mismatch = -2.0
        aligner.gap_score = -10.0
        score = aligner.score(s1, s2)
        self.assertAlmostEqual(score, 4.0)
        score = aligner.score(s1, s2a)
        self.assertAlmostEqual(score, 1.0)
        score = aligner.score(s1, s2b)
        self.assertAlmostEqual(score, 4.0)
        m = 5 * np.eye(10)
        aligner.substitution_matrix = m
        score = aligner.score(s1, s2)
        self.assertAlmostEqual(score, 10.0)
        message = '^sequence item 2 is negative \\(-6\\)$'
        with self.assertRaisesRegex(ValueError, message):
            aligner.score(s1, s2c)
        message = '^sequence item 1 is out of bound \\(28, should be < 10\\)$'
        with self.assertRaisesRegex(ValueError, message):
            aligner.score(s1, s2b)
        message = '^sequence item 2 is out of bound \\(99, should be < 10\\)$'
        with self.assertRaisesRegex(ValueError, message):
            aligner.score(s1, s2a)

class TestOverflowError(unittest.TestCase):

    def test_align_overflow_error(self):
        if False:
            return 10
        aligner = Align.PairwiseAligner()
        path = os.path.join('Align', 'bsubtilis.fa')
        record = SeqIO.read(path, 'fasta')
        seq1 = record.seq
        path = os.path.join('Align', 'ecoli.fa')
        record = SeqIO.read(path, 'fasta')
        seq2 = record.seq
        alignments = aligner.align(seq1, seq2)
        self.assertAlmostEqual(alignments.score, 1286.0)
        message = '^number of optimal alignments is larger than (%d|%d)$' % (2147483647, 9223372036854775807)
        with self.assertRaisesRegex(OverflowError, message):
            n = len(alignments)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ATTTA-TC-GGA-GAGTTTGATCC-TGGCTCAGGAC--GAACGCTGGCGGC-GTGCCTAA\n                  0 |---|-|--|-|-||||||||||--||||||||-|---|||||||||||||-|-||||||\nquery             0 A---AAT-TG-AAGAGTTTGATC-ATGGCTCAG-A-TTGAACGCTGGCGGCAG-GCCTAA\n\ntarget           53 T-ACATGCAAGTCGAG-CGG-A-CAG-AT-GGGA-GCTTGCT-C----CCTGAT-GTTAG\n                 60 --|||||||||||||--|||-|-|||-|--|--|-|||||||-|----|-|||--|--||\nquery            51 -CACATGCAAGTCGA-ACGGTAACAGGA-AG--AAGCTTGCTTCTTTGC-TGA-CG--AG\n\ntarget          100 C-GGCGGACGGGTGAGTAACAC-GT--GGGTAA-CCTGCCTGTAA-G-ACTGGG--ATAA\n                120 --|||||||||||||||||----||--|||-||-|-||||||-|--|-|--|||--||||\nquery           102 -TGGCGGACGGGTGAGTAA---TGTCTGGG-AAAC-TGCCTG-A-TGGA--GGGGGATAA\n\ntarget          151 CT-CC-GGGAAACCGG--GGCTAATACCGG-ATGGTTGTTTGAACCGCAT-GGTTCAA-A\n                180 ||-|--||-||||-||--|-|||||||||--||---------|||-|--|-|---|||-|\nquery           152 CTAC-TGG-AAAC-GGTAG-CTAATACCG-CAT---------AAC-G--TCG---CAAGA\n\ntarget          204 C-ATAA-AAGGTGG--C-TTCGG-C-TACCACTTA-C-A--G-ATG-GACCC-GC--GGC\n                240 |-|-||-|-||-||--|-|||||-|-|-|---||--|-|--|-|||-|-|||-|---||-\nquery           192 CCA-AAGA-GG-GGGACCTTCGGGCCT-C---TT-GCCATCGGATGTG-CCCAG-ATGG-\n\ntarget          248 GCATTAGCTAGTT-GGTGAGG-TAACGGCTCACC-AAGGCGACGATGCG--TAGCC-GA-\n                300 |-||||||||||--||||-||-||||||||||||-|-|||||||||-|---||||--|--\nquery           241 G-ATTAGCTAGT-AGGTG-GGGTAACGGCTCACCTA-GGCGACGAT-C-CCTAGC-TG-G\n\ntarget          301 -CCTGAGAGGG-TGATC--GGCCACACTGGGA-CTGAGACACGG-CCCAGACTCCTACGG\n                360 -|-|||||||--|||-|--|-|||||||||-|-|||||||||||-||-||||||||||||\nquery           293 TC-TGAGAGG-ATGA-CCAG-CCACACTGG-AACTGAGACACGGTCC-AGACTCCTACGG\n\ntarget          355 GAGGCAGCAGTAGGG-AATC-TTCCGCA-A-TGGA-CG-AAAGTC-TGAC-GG-AGCAAC\n                420 |||||||||||-|||-|||--||--|||-|-|||--||-||-|-|-|||--|--|||--|\nquery           347 GAGGCAGCAGT-GGGGAAT-ATT--GCACAATGG-GCGCAA-G-CCTGA-TG-CAGC--C\n\ntarget          406 --GCCGCGTG-AGTGAT-GAAGG--TTTTCGGA-TC-GTAAAGCT-CTGTTGTT-AG-GG\n                480 --||||||||-|-|||--|||||--||--|||--|--||||||-|-||-||----||-||\nquery           396 ATGCCGCGTGTA-TGA-AGAAGGCCTT--CGG-GT-TGTAAAG-TACT-TT---CAGCGG\n\ntarget          455 --A--A-G--A--ACAAGTGCCGTTCGAATAGGGC----GG-TACC-TTGACGGT-ACCT\n                540 --|--|-|--|--|-||||----|---||||---|----|--|-|--||||||-|-|||-\nquery           445 GGAGGAAGGGAGTA-AAGT----T---AATA---CCTTTG-CT-C-ATTGACG-TTACC-\n\ntarget          499 AAC-CAGAA-A-GCCAC-GGCTAACTAC-GTGCCAGCAGCCGCGGTAATACGT-AGG-TG\n                600 --|-|||||-|-||-||-||||||||-|-|||||||||||||||||||||||--|||-||\nquery           489 --CGCAGAAGAAGC-ACCGGCTAACT-CCGTGCCAGCAGCCGCGGTAATACG-GAGGGTG\n\ntarget          552 GCAAGCGTTG--TCCGGAATTA-TTGGGCGTAAAG-GGCT-CGCAGGCGGTTTC-TTAAG\n                660 -||||||||---||-|||||||-|-||||||||||-|-|--||||||||||||--|||||\nquery           544 -CAAGCGTT-AATC-GGAATTACT-GGGCGTAAAGCG-C-ACGCAGGCGGTTT-GTTAAG\n\ntarget          606 TCT-GATGTGAAAG-CCCCCGG-CTCAACC-GGGGAGGG--T-CAT-TGGA-AACTGGGG\n                720 ||--|||||||||--||||-||-|||||||-|||-|-----|-|||-||-|-|-||||--\nquery           597 TC-AGATGTGAAA-TCCCC-GGGCTCAACCTGGG-A---ACTGCATCTG-ATA-CTGG--\n\ntarget          657 -AA-CTTGAGTGCA--G-AAGAGGAGAGTGG-A-A-TTCCACG-TGTAGCGGTGAAATGC\n                780 -||-|||||||-|---|-|-||||-|-|-||-|-|-|||||-|-||||||||||||||||\nquery           646 CAAGCTTGAGT-C-TCGTA-GAGG-G-G-GGTAGAATTCCA-GGTGTAGCGGTGAAATGC\n\ntarget          708 GTAGAGATG-TGGAGGAAC-ACCAG-TGGCGAAGGCGA-CTCTC--TGGT-CTGTAA--C\n                840 ||||||||--||||||||--|||-|-|||||||||||--|-|-|--|||--|-|-||--|\nquery           699 GTAGAGAT-CTGGAGGAA-TACC-GGTGGCGAAGGCG-GC-C-CCCTGG-AC-G-AAGAC\n\ntarget          759 TGACGCTG-AGGA-GCGAAAGCGTGGGGAGCGAA-CAGGATTAGATACCCTGGTAGTCCA\n                900 |||||||--|||--|||||||||||||||||-||-|||||||||||||||||||||||||\nquery           750 TGACGCT-CAGG-TGCGAAAGCGTGGGGAGC-AAACAGGATTAGATACCCTGGTAGTCCA\n\ntarget          816 CGCCGTAAACGATGAGT-G-CTAAGTGTT-AGGGGGTT-TCCGCCCCTT-AGTGC-TG-C\n                960 ||||||||||||||--|-|-||---||---|||---||-|--||||-||-||-||-||-|\nquery           807 CGCCGTAAACGATG--TCGACT---TG--GAGG---TTGT--GCCC-TTGAG-GCGTGGC\n\ntarget          869 ------AGCTAACGCA-TTAAG-C-ACTCCGCCTGGGGAGTACGGTC-GCAAGACTG--A\n               1020 ------|||||||||--|||||-|-||-|-|||||||||||||||-|-|||||---|--|\nquery           853 TTCCGGAGCTAACGC-GTTAAGTCGAC-C-GCCTGGGGAGTACGG-CCGCAAG---GTTA\n\ntarget          917 AA-CTCAAA-GGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGAA\n               1080 ||-||||||-|-|||||||||||||||||||||||||||||||||||||||||||||||-\nquery           906 AAACTCAAATG-AATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGA-\n\ntarget          975 -GCAACGCGAAGAACCTTACCA-GGTCTTGACATCCTCTGACA-A--T--CCTAGAGATA\n               1140 -||||||||||||||||||||--|||||||||||||----|||-|--|--||-||||||-\nquery           964 TGCAACGCGAAGAACCTTACC-TGGTCTTGACATCC----ACAGAACTTTCC-AGAGAT-\n\ntarget         1028 GGAC--G-T-CCCCTTCGGGGGCAGA--GTGA--CAGGTGG-TGCATGG-TTGTCGTCAG\n               1200 |||---|-|-||--||||||---|-|--||||--||||||--|||||||-|-||||||||\nquery          1017 GGA-TTGGTGCC--TTCGGG---A-ACTGTGAGACAGGTG-CTGCATGGCT-GTCGTCAG\n\ntarget         1078 CTCGTGTC-GTGAGA-TGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTGATCTTA--G\n               1260 |||||||--||||-|-||||||||||||||||||||||||||||||||||-|||||---|\nquery          1068 CTCGTGT-TGTGA-AATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTT-ATCTT-TTG\n\ntarget         1134 TTGCCAGCA--TTCA-GTTG--GGC-A-CTCTAA-GGT-GACTGCC-GGTGAC-AAACC-\n               1320 ||||||||---|-|--|--|--||--|-|||-||-||--|||||||-|-|||--||||--\nquery          1124 TTGCCAGC-GGT-C-CG--GCCGG-GAACTC-AAAGG-AGACTGCCAG-TGA-TAAAC-T\n\ntarget         1182 GGAGGAAGGTGGGGATGACGTCAAA-TCATCATG-CCCCTTAT-GACCT-GGGCTACACA\n               1380 ||||||||||||||||||||||||--||||||||-|||-|||--||||--||||||||||\nquery          1173 GGAGGAAGGTGGGGATGACGTCAA-GTCATCATGGCCC-TTA-CGACC-AGGGCTACACA\n\ntarget         1238 CGTGCTACAATGGACAG-A-ACAAAG-GGCA-GCGAAACC--GCGAG-GTT-AAGCC--A\n               1440 |||||||||||||-|-|-|-||||||-|--|-||||--||--|||||-|---||||---|\nquery          1229 CGTGCTACAATGG-C-GCATACAAAGAG--AAGCGA--CCTCGCGAGAG--CAAGC-GGA\n\ntarget         1288 ATCC-CAC-AAA-T-CTGTTC-TCAGTTC-GGATC-GC-AGTCTGCAACTCGACTGCG--\n               1500 --||-||--|||-|-|-||-|-|-|||-|-||||--|--||||||||||||||||-|---\nquery          1280 --CCTCA-TAAAGTGC-GT-CGT-AGT-CCGGAT-TG-GAGTCTGCAACTCGACT-C-CA\n\ntarget         1338 TGAAGCT-GGAATCGCTAGTAATCGC-GGATCAGCA-TGCCG-CGGTGAATACGTTCCCG\n               1560 |||||-|-|||||||||||||||||--|||||||-|-||||--|||||||||||||||||\nquery          1329 TGAAG-TCGGAATCGCTAGTAATCG-TGGATCAG-AATGCC-ACGGTGAATACGTTCCCG\n\ntarget         1394 GGCCTTGTACACACCGCCCGTCACACCAC-GAG-AGT---TTGT-AACACCC-GAAGTC-\n               1620 ||||||||||||||||||||||||||||--|-|-|||---|||--||-|----|||||--\nquery          1385 GGCCTTGTACACACCGCCCGTCACACCA-TG-GGAGTGGGTTG-CAA-A---AGAAGT-A\n\ntarget         1446 GGTGAGG-T-AACCTTTTA-GG-AG--C-C--AGCCG-CC---GAAGGTGGGA--CAGAT\n               1680 |||-||--|-||||||----||-||--|-|--|-||--|----|----||--|--||--|\nquery          1437 GGT-AG-CTTAACCTT---CGGGAGGGCGCTTA-CC-AC-TTTG----TG--ATTCA--T\n\ntarget         1491 GA-TTGGGGTGAAGTCGTAACAAGGTAG-CCGTATCGGAAGG----TGCGGCT-GGATCA\n               1740 ||-|-||||||||||||||||||||||--|||||--||--||----|||||-|-||||||\nquery          1481 GACT-GGGGTGAAGTCGTAACAAGGTA-ACCGTA--GG--GGAACCTGCGG-TTGGATCA\n\ntarget         1544 CCTCCTTTCTA 1555\n               1800 |||||||---| 1811\nquery          1534 CCTCCTT---A 1542\n')
        self.assertEqual(alignment.shape, (2, 1811))
        self.assertAlmostEqual(alignment.score, 1286.0)
        alignments = aligner.align(seq1, reverse_complement(seq2), strand='-')
        self.assertAlmostEqual(alignments.score, 1286.0)
        message = '^number of optimal alignments is larger than (%d|%d)$' % (2147483647, 9223372036854775807)
        with self.assertRaisesRegex(OverflowError, message):
            n = len(alignments)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ATTTA-TC-GGA-GAGTTTGATCC-TGGCTCAGGAC--GAACGCTGGCGGC-GTGCCTAA\n                  0 |---|-|--|-|-||||||||||--||||||||-|---|||||||||||||-|-||||||\nquery          1542 A---AAT-TG-AAGAGTTTGATC-ATGGCTCAG-A-TTGAACGCTGGCGGCAG-GCCTAA\n\ntarget           53 T-ACATGCAAGTCGAG-CGG-A-CAG-AT-GGGA-GCTTGCT-C----CCTGAT-GTTAG\n                 60 --|||||||||||||--|||-|-|||-|--|--|-|||||||-|----|-|||--|--||\nquery          1491 -CACATGCAAGTCGA-ACGGTAACAGGA-AG--AAGCTTGCTTCTTTGC-TGA-CG--AG\n\ntarget          100 C-GGCGGACGGGTGAGTAACAC-GT--GGGTAA-CCTGCCTGTAA-G-ACTGGG--ATAA\n                120 --|||||||||||||||||----||--|||-||-|-||||||-|--|-|--|||--||||\nquery          1440 -TGGCGGACGGGTGAGTAA---TGTCTGGG-AAAC-TGCCTG-A-TGGA--GGGGGATAA\n\ntarget          151 CT-CC-GGGAAACCGG--GGCTAATACCGG-ATGGTTGTTTGAACCGCAT-GGTTCAA-A\n                180 ||-|--||-||||-||--|-|||||||||--||---------|||-|--|-|---|||-|\nquery          1390 CTAC-TGG-AAAC-GGTAG-CTAATACCG-CAT---------AAC-G--TCG---CAAGA\n\ntarget          204 C-ATAA-AAGGTGG--C-TTCGG-C-TACCACTTA-C-A--G-ATG-GACCC-GC--GGC\n                240 |-|-||-|-||-||--|-|||||-|-|-|---||--|-|--|-|||-|-|||-|---||-\nquery          1350 CCA-AAGA-GG-GGGACCTTCGGGCCT-C---TT-GCCATCGGATGTG-CCCAG-ATGG-\n\ntarget          248 GCATTAGCTAGTT-GGTGAGG-TAACGGCTCACC-AAGGCGACGATGCG--TAGCC-GA-\n                300 |-||||||||||--||||-||-||||||||||||-|-|||||||||-|---||||--|--\nquery          1301 G-ATTAGCTAGT-AGGTG-GGGTAACGGCTCACCTA-GGCGACGAT-C-CCTAGC-TG-G\n\ntarget          301 -CCTGAGAGGG-TGATC--GGCCACACTGGGA-CTGAGACACGG-CCCAGACTCCTACGG\n                360 -|-|||||||--|||-|--|-|||||||||-|-|||||||||||-||-||||||||||||\nquery          1249 TC-TGAGAGG-ATGA-CCAG-CCACACTGG-AACTGAGACACGGTCC-AGACTCCTACGG\n\ntarget          355 GAGGCAGCAGTAGGG-AATC-TTCCGCA-A-TGGA-CG-AAAGTC-TGAC-GG-AGCAAC\n                420 |||||||||||-|||-|||--||--|||-|-|||--||-||-|-|-|||--|--|||--|\nquery          1195 GAGGCAGCAGT-GGGGAAT-ATT--GCACAATGG-GCGCAA-G-CCTGA-TG-CAGC--C\n\ntarget          406 --GCCGCGTG-AGTGAT-GAAGG--TTTTCGGA-TC-GTAAAGCT-CTGTTGTT-AG-GG\n                480 --||||||||-|-|||--|||||--||--|||--|--||||||-|-||-||----||-||\nquery          1146 ATGCCGCGTGTA-TGA-AGAAGGCCTT--CGG-GT-TGTAAAG-TACT-TT---CAGCGG\n\ntarget          455 --A--A-G--A--ACAAGTGCCGTTCGAATAGGGC----GG-TACC-TTGACGGT-ACCT\n                540 --|--|-|--|--|-||||----|---||||---|----|--|-|--||||||-|-|||-\nquery          1097 GGAGGAAGGGAGTA-AAGT----T---AATA---CCTTTG-CT-C-ATTGACG-TTACC-\n\ntarget          499 AAC-CAGAA-A-GCCAC-GGCTAACTAC-GTGCCAGCAGCCGCGGTAATACGT-AGG-TG\n                600 --|-|||||-|-||-||-||||||||-|-|||||||||||||||||||||||--|||-||\nquery          1053 --CGCAGAAGAAGC-ACCGGCTAACT-CCGTGCCAGCAGCCGCGGTAATACG-GAGGGTG\n\ntarget          552 GCAAGCGTTG--TCCGGAATTA-TTGGGCGTAAAG-GGCT-CGCAGGCGGTTTC-TTAAG\n                660 -||||||||---||-|||||||-|-||||||||||-|-|--||||||||||||--|||||\nquery           998 -CAAGCGTT-AATC-GGAATTACT-GGGCGTAAAGCG-C-ACGCAGGCGGTTT-GTTAAG\n\ntarget          606 TCT-GATGTGAAAG-CCCCCGG-CTCAACC-GGGGAGGG--T-CAT-TGGA-AACTGGGG\n                720 ||--|||||||||--||||-||-|||||||-|||-|-----|-|||-||-|-|-||||--\nquery           945 TC-AGATGTGAAA-TCCCC-GGGCTCAACCTGGG-A---ACTGCATCTG-ATA-CTGG--\n\ntarget          657 -AA-CTTGAGTGCA--G-AAGAGGAGAGTGG-A-A-TTCCACG-TGTAGCGGTGAAATGC\n                780 -||-|||||||-|---|-|-||||-|-|-||-|-|-|||||-|-||||||||||||||||\nquery           896 CAAGCTTGAGT-C-TCGTA-GAGG-G-G-GGTAGAATTCCA-GGTGTAGCGGTGAAATGC\n\ntarget          708 GTAGAGATG-TGGAGGAAC-ACCAG-TGGCGAAGGCGA-CTCTC--TGGT-CTGTAA--C\n                840 ||||||||--||||||||--|||-|-|||||||||||--|-|-|--|||--|-|-||--|\nquery           843 GTAGAGAT-CTGGAGGAA-TACC-GGTGGCGAAGGCG-GC-C-CCCTGG-AC-G-AAGAC\n\ntarget          759 TGACGCTG-AGGA-GCGAAAGCGTGGGGAGCGAA-CAGGATTAGATACCCTGGTAGTCCA\n                900 |||||||--|||--|||||||||||||||||-||-|||||||||||||||||||||||||\nquery           792 TGACGCT-CAGG-TGCGAAAGCGTGGGGAGC-AAACAGGATTAGATACCCTGGTAGTCCA\n\ntarget          816 CGCCGTAAACGATGAGT-G-CTAAGTGTT-AGGGGGTT-TCCGCCCCTT-AGTGC-TG-C\n                960 ||||||||||||||--|-|-||---||---|||---||-|--||||-||-||-||-||-|\nquery           735 CGCCGTAAACGATG--TCGACT---TG--GAGG---TTGT--GCCC-TTGAG-GCGTGGC\n\ntarget          869 ------AGCTAACGCA-TTAAG-C-ACTCCGCCTGGGGAGTACGGTC-GCAAGACTG--A\n               1020 ------|||||||||--|||||-|-||-|-|||||||||||||||-|-|||||---|--|\nquery           689 TTCCGGAGCTAACGC-GTTAAGTCGAC-C-GCCTGGGGAGTACGG-CCGCAAG---GTTA\n\ntarget          917 AA-CTCAAA-GGAATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGAA\n               1080 ||-||||||-|-|||||||||||||||||||||||||||||||||||||||||||||||-\nquery           636 AAACTCAAATG-AATTGACGGGGGCCCGCACAAGCGGTGGAGCATGTGGTTTAATTCGA-\n\ntarget          975 -GCAACGCGAAGAACCTTACCA-GGTCTTGACATCCTCTGACA-A--T--CCTAGAGATA\n               1140 -||||||||||||||||||||--|||||||||||||----|||-|--|--||-||||||-\nquery           578 TGCAACGCGAAGAACCTTACC-TGGTCTTGACATCC----ACAGAACTTTCC-AGAGAT-\n\ntarget         1028 GGAC--G-T-CCCCTTCGGGGGCAGA--GTGA--CAGGTGG-TGCATGG-TTGTCGTCAG\n               1200 |||---|-|-||--||||||---|-|--||||--||||||--|||||||-|-||||||||\nquery           525 GGA-TTGGTGCC--TTCGGG---A-ACTGTGAGACAGGTG-CTGCATGGCT-GTCGTCAG\n\ntarget         1078 CTCGTGTC-GTGAGA-TGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTTGATCTTA--G\n               1260 |||||||--||||-|-||||||||||||||||||||||||||||||||||-|||||---|\nquery           474 CTCGTGT-TGTGA-AATGTTGGGTTAAGTCCCGCAACGAGCGCAACCCTT-ATCTT-TTG\n\ntarget         1134 TTGCCAGCA--TTCA-GTTG--GGC-A-CTCTAA-GGT-GACTGCC-GGTGAC-AAACC-\n               1320 ||||||||---|-|--|--|--||--|-|||-||-||--|||||||-|-|||--||||--\nquery           418 TTGCCAGC-GGT-C-CG--GCCGG-GAACTC-AAAGG-AGACTGCCAG-TGA-TAAAC-T\n\ntarget         1182 GGAGGAAGGTGGGGATGACGTCAAA-TCATCATG-CCCCTTAT-GACCT-GGGCTACACA\n               1380 ||||||||||||||||||||||||--||||||||-|||-|||--||||--||||||||||\nquery           369 GGAGGAAGGTGGGGATGACGTCAA-GTCATCATGGCCC-TTA-CGACC-AGGGCTACACA\n\ntarget         1238 CGTGCTACAATGGACAG-A-ACAAAG-GGCA-GCGAAACC--GCGAG-GTT-AAGCC--A\n               1440 |||||||||||||-|-|-|-||||||-|--|-||||--||--|||||-|---||||---|\nquery           313 CGTGCTACAATGG-C-GCATACAAAGAG--AAGCGA--CCTCGCGAGAG--CAAGC-GGA\n\ntarget         1288 ATCC-CAC-AAA-T-CTGTTC-TCAGTTC-GGATC-GC-AGTCTGCAACTCGACTGCG--\n               1500 --||-||--|||-|-|-||-|-|-|||-|-||||--|--||||||||||||||||-|---\nquery           262 --CCTCA-TAAAGTGC-GT-CGT-AGT-CCGGAT-TG-GAGTCTGCAACTCGACT-C-CA\n\ntarget         1338 TGAAGCT-GGAATCGCTAGTAATCGC-GGATCAGCA-TGCCG-CGGTGAATACGTTCCCG\n               1560 |||||-|-|||||||||||||||||--|||||||-|-||||--|||||||||||||||||\nquery           213 TGAAG-TCGGAATCGCTAGTAATCG-TGGATCAG-AATGCC-ACGGTGAATACGTTCCCG\n\ntarget         1394 GGCCTTGTACACACCGCCCGTCACACCAC-GAG-AGT---TTGT-AACACCC-GAAGTC-\n               1620 ||||||||||||||||||||||||||||--|-|-|||---|||--||-|----|||||--\nquery           157 GGCCTTGTACACACCGCCCGTCACACCA-TG-GGAGTGGGTTG-CAA-A---AGAAGT-A\n\ntarget         1446 GGTGAGG-T-AACCTTTTA-GG-AG--C-C--AGCCG-CC---GAAGGTGGGA--CAGAT\n               1680 |||-||--|-||||||----||-||--|-|--|-||--|----|----||--|--||--|\nquery           105 GGT-AG-CTTAACCTT---CGGGAGGGCGCTTA-CC-AC-TTTG----TG--ATTCA--T\n\ntarget         1491 GA-TTGGGGTGAAGTCGTAACAAGGTAG-CCGTATCGGAAGG----TGCGGCT-GGATCA\n               1740 ||-|-||||||||||||||||||||||--|||||--||--||----|||||-|-||||||\nquery            61 GACT-GGGGTGAAGTCGTAACAAGGTA-ACCGTA--GG--GGAACCTGCGG-TTGGATCA\n\ntarget         1544 CCTCCTTTCTA 1555\n               1800 |||||||---| 1811\nquery             8 CCTCCTT---A    0\n')
        self.assertAlmostEqual(alignment.score, 1286.0)
        self.assertEqual(alignment.shape, (2, 1811))

class TestKeywordArgumentsConstructor(unittest.TestCase):

    def test_confusing_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner(mode='local', open_gap_score=-0.3, extend_gap_score=-0.1, target_open_gap_score=-0.2)
        self.assertEqual(str(aligner), 'Pairwise sequence aligner with parameters\n  wildcard: None\n  match_score: 1.000000\n  mismatch_score: 0.000000\n  target_internal_open_gap_score: -0.200000\n  target_internal_extend_gap_score: -0.100000\n  target_left_open_gap_score: -0.200000\n  target_left_extend_gap_score: -0.100000\n  target_right_open_gap_score: -0.200000\n  target_right_extend_gap_score: -0.100000\n  query_internal_open_gap_score: -0.300000\n  query_internal_extend_gap_score: -0.100000\n  query_left_open_gap_score: -0.300000\n  query_left_extend_gap_score: -0.100000\n  query_right_open_gap_score: -0.300000\n  query_right_extend_gap_score: -0.100000\n  mode: local\n')

class TestPredefinedScoringSchemes(unittest.TestCase):

    def test_blastn(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner(scoring='blastn')
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*\n  target_internal_open_gap_score: -7.000000\n  target_internal_extend_gap_score: -2.000000\n  target_left_open_gap_score: -7.000000\n  target_left_extend_gap_score: -2.000000\n  target_right_open_gap_score: -7.000000\n  target_right_extend_gap_score: -2.000000\n  query_internal_open_gap_score: -7.000000\n  query_internal_extend_gap_score: -2.000000\n  query_left_open_gap_score: -7.000000\n  query_left_extend_gap_score: -2.000000\n  query_right_open_gap_score: -7.000000\n  query_right_extend_gap_score: -2.000000\n  mode: global\n$')
        self.assertEqual(str(aligner.substitution_matrix[:, :]), '     A    T    G    C    S    W    R    Y    K    M    B    V    H    D    N\nA  2.0 -3.0 -3.0 -3.0 -3.0 -1.0 -1.0 -3.0 -3.0 -1.0 -3.0 -1.0 -1.0 -1.0 -2.0\nT -3.0  2.0 -3.0 -3.0 -3.0 -1.0 -3.0 -1.0 -1.0 -3.0 -1.0 -3.0 -1.0 -1.0 -2.0\nG -3.0 -3.0  2.0 -3.0 -1.0 -3.0 -1.0 -3.0 -1.0 -3.0 -1.0 -1.0 -3.0 -1.0 -2.0\nC -3.0 -3.0 -3.0  2.0 -1.0 -3.0 -3.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -3.0 -2.0\nS -3.0 -3.0 -1.0 -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nW -1.0 -1.0 -3.0 -3.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nR -1.0 -3.0 -1.0 -3.0 -1.0 -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nY -3.0 -1.0 -3.0 -1.0 -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nK -3.0 -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -2.0\nM -1.0 -3.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nB -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nV -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nH -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nD -1.0 -1.0 -1.0 -3.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0\nN -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0 -2.0\n')

    def test_megablast(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner(scoring='megablast')
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*\n  target_internal_open_gap_score: -2.500000\n  target_internal_extend_gap_score: -2.500000\n  target_left_open_gap_score: -2.500000\n  target_left_extend_gap_score: -2.500000\n  target_right_open_gap_score: -2.500000\n  target_right_extend_gap_score: -2.500000\n  query_internal_open_gap_score: -2.500000\n  query_internal_extend_gap_score: -2.500000\n  query_left_open_gap_score: -2.500000\n  query_left_extend_gap_score: -2.500000\n  query_right_open_gap_score: -2.500000\n  query_right_extend_gap_score: -2.500000\n  mode: global\n$')
        self.assertEqual(str(aligner.substitution_matrix[:, :]), '     A    T    G    C    S    W    R    Y    K    M    B    V    H    D    N\nA  1.0 -2.0 -2.0 -2.0 -2.0 -1.0 -1.0 -2.0 -2.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0\nT -2.0  1.0 -2.0 -2.0 -2.0 -1.0 -2.0 -1.0 -1.0 -2.0 -1.0 -2.0 -1.0 -1.0 -1.0\nG -2.0 -2.0  1.0 -2.0 -1.0 -2.0 -1.0 -2.0 -1.0 -2.0 -1.0 -1.0 -2.0 -1.0 -1.0\nC -2.0 -2.0 -2.0  1.0 -1.0 -2.0 -2.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -2.0 -1.0\nS -2.0 -2.0 -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nW -1.0 -1.0 -2.0 -2.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nR -1.0 -2.0 -1.0 -2.0 -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nY -2.0 -1.0 -2.0 -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nK -2.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0\nM -1.0 -2.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nB -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nV -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nH -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nD -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\nN -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0\n')

    def test_blastp(self):
        if False:
            return 10
        aligner = Align.PairwiseAligner(scoring='blastp')
        self.assertRegex(str(aligner), '^Pairwise sequence aligner with parameters\n  substitution_matrix: <Array object at .*\n  target_internal_open_gap_score: -12.000000\n  target_internal_extend_gap_score: -1.000000\n  target_left_open_gap_score: -12.000000\n  target_left_extend_gap_score: -1.000000\n  target_right_open_gap_score: -12.000000\n  target_right_extend_gap_score: -1.000000\n  query_internal_open_gap_score: -12.000000\n  query_internal_extend_gap_score: -1.000000\n  query_left_open_gap_score: -12.000000\n  query_left_extend_gap_score: -1.000000\n  query_right_open_gap_score: -12.000000\n  query_right_extend_gap_score: -1.000000\n  mode: global\n$')
        self.assertEqual(str(aligner.substitution_matrix[:, :]), '     A    B    C    D    E    F    G    H    I    J    K    L    M    N    O    P    Q    R    S    T    U    V    W    X    Y    Z    *\nA  4.0 -2.0  0.0 -2.0 -1.0 -2.0  0.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0 -2.0 -1.0 -1.0 -1.0 -1.0  1.0  0.0  0.0  0.0 -3.0 -1.0 -2.0 -1.0 -4.0\nB -2.0  4.0 -3.0  4.0  1.0 -3.0 -1.0  0.0 -3.0 -3.0  0.0 -4.0 -3.0  4.0 -1.0 -2.0  0.0 -1.0  0.0 -1.0 -3.0 -3.0 -4.0 -1.0 -3.0  0.0 -4.0\nC  0.0 -3.0  9.0 -3.0 -4.0 -2.0 -3.0 -3.0 -1.0 -1.0 -3.0 -1.0 -1.0 -3.0 -1.0 -3.0 -3.0 -3.0 -1.0 -1.0  9.0 -1.0 -2.0 -1.0 -2.0 -3.0 -4.0\nD -2.0  4.0 -3.0  6.0  2.0 -3.0 -1.0 -1.0 -3.0 -3.0 -1.0 -4.0 -3.0  1.0 -1.0 -1.0  0.0 -2.0  0.0 -1.0 -3.0 -3.0 -4.0 -1.0 -3.0  1.0 -4.0\nE -1.0  1.0 -4.0  2.0  5.0 -3.0 -2.0  0.0 -3.0 -3.0  1.0 -3.0 -2.0  0.0 -1.0 -1.0  2.0  0.0  0.0 -1.0 -4.0 -2.0 -3.0 -1.0 -2.0  4.0 -4.0\nF -2.0 -3.0 -2.0 -3.0 -3.0  6.0 -3.0 -1.0  0.0  0.0 -3.0  0.0  0.0 -3.0 -1.0 -4.0 -3.0 -3.0 -2.0 -2.0 -2.0 -1.0  1.0 -1.0  3.0 -3.0 -4.0\nG  0.0 -1.0 -3.0 -1.0 -2.0 -3.0  6.0 -2.0 -4.0 -4.0 -2.0 -4.0 -3.0  0.0 -1.0 -2.0 -2.0 -2.0  0.0 -2.0 -3.0 -3.0 -2.0 -1.0 -3.0 -2.0 -4.0\nH -2.0  0.0 -3.0 -1.0  0.0 -1.0 -2.0  8.0 -3.0 -3.0 -1.0 -3.0 -2.0  1.0 -1.0 -2.0  0.0  0.0 -1.0 -2.0 -3.0 -3.0 -2.0 -1.0  2.0  0.0 -4.0\nI -1.0 -3.0 -1.0 -3.0 -3.0  0.0 -4.0 -3.0  4.0  3.0 -3.0  2.0  1.0 -3.0 -1.0 -3.0 -3.0 -3.0 -2.0 -1.0 -1.0  3.0 -3.0 -1.0 -1.0 -3.0 -4.0\nJ -1.0 -3.0 -1.0 -3.0 -3.0  0.0 -4.0 -3.0  3.0  3.0 -3.0  3.0  2.0 -3.0 -1.0 -3.0 -2.0 -2.0 -2.0 -1.0 -1.0  2.0 -2.0 -1.0 -1.0 -3.0 -4.0\nK -1.0  0.0 -3.0 -1.0  1.0 -3.0 -2.0 -1.0 -3.0 -3.0  5.0 -2.0 -1.0  0.0 -1.0 -1.0  1.0  2.0  0.0 -1.0 -3.0 -2.0 -3.0 -1.0 -2.0  1.0 -4.0\nL -1.0 -4.0 -1.0 -4.0 -3.0  0.0 -4.0 -3.0  2.0  3.0 -2.0  4.0  2.0 -3.0 -1.0 -3.0 -2.0 -2.0 -2.0 -1.0 -1.0  1.0 -2.0 -1.0 -1.0 -3.0 -4.0\nM -1.0 -3.0 -1.0 -3.0 -2.0  0.0 -3.0 -2.0  1.0  2.0 -1.0  2.0  5.0 -2.0 -1.0 -2.0  0.0 -1.0 -1.0 -1.0 -1.0  1.0 -1.0 -1.0 -1.0 -1.0 -4.0\nN -2.0  4.0 -3.0  1.0  0.0 -3.0  0.0  1.0 -3.0 -3.0  0.0 -3.0 -2.0  6.0 -1.0 -2.0  0.0  0.0  1.0  0.0 -3.0 -3.0 -4.0 -1.0 -2.0  0.0 -4.0\nO -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -4.0\nP -1.0 -2.0 -3.0 -1.0 -1.0 -4.0 -2.0 -2.0 -3.0 -3.0 -1.0 -3.0 -2.0 -2.0 -1.0  7.0 -1.0 -2.0 -1.0 -1.0 -3.0 -2.0 -4.0 -1.0 -3.0 -1.0 -4.0\nQ -1.0  0.0 -3.0  0.0  2.0 -3.0 -2.0  0.0 -3.0 -2.0  1.0 -2.0  0.0  0.0 -1.0 -1.0  5.0  1.0  0.0 -1.0 -3.0 -2.0 -2.0 -1.0 -1.0  4.0 -4.0\nR -1.0 -1.0 -3.0 -2.0  0.0 -3.0 -2.0  0.0 -3.0 -2.0  2.0 -2.0 -1.0  0.0 -1.0 -2.0  1.0  5.0 -1.0 -1.0 -3.0 -3.0 -3.0 -1.0 -2.0  0.0 -4.0\nS  1.0  0.0 -1.0  0.0  0.0 -2.0  0.0 -1.0 -2.0 -2.0  0.0 -2.0 -1.0  1.0 -1.0 -1.0  0.0 -1.0  4.0  1.0 -1.0 -2.0 -3.0 -1.0 -2.0  0.0 -4.0\nT  0.0 -1.0 -1.0 -1.0 -1.0 -2.0 -2.0 -2.0 -1.0 -1.0 -1.0 -1.0 -1.0  0.0 -1.0 -1.0 -1.0 -1.0  1.0  5.0 -1.0  0.0 -2.0 -1.0 -2.0 -1.0 -4.0\nU  0.0 -3.0  9.0 -3.0 -4.0 -2.0 -3.0 -3.0 -1.0 -1.0 -3.0 -1.0 -1.0 -3.0 -1.0 -3.0 -3.0 -3.0 -1.0 -1.0  9.0 -1.0 -2.0 -1.0 -2.0 -3.0 -4.0\nV  0.0 -3.0 -1.0 -3.0 -2.0 -1.0 -3.0 -3.0  3.0  2.0 -2.0  1.0  1.0 -3.0 -1.0 -2.0 -2.0 -3.0 -2.0  0.0 -1.0  4.0 -3.0 -1.0 -1.0 -2.0 -4.0\nW -3.0 -4.0 -2.0 -4.0 -3.0  1.0 -2.0 -2.0 -3.0 -2.0 -3.0 -2.0 -1.0 -4.0 -1.0 -4.0 -2.0 -3.0 -3.0 -2.0 -2.0 -3.0 11.0 -1.0  2.0 -2.0 -4.0\nX -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -1.0 -4.0\nY -2.0 -3.0 -2.0 -3.0 -2.0  3.0 -3.0  2.0 -1.0 -1.0 -2.0 -1.0 -1.0 -2.0 -1.0 -3.0 -1.0 -2.0 -2.0 -2.0 -2.0 -1.0  2.0 -1.0  7.0 -2.0 -4.0\nZ -1.0  0.0 -3.0  1.0  4.0 -3.0 -2.0  0.0 -3.0 -3.0  1.0 -3.0 -1.0  0.0 -1.0 -1.0  4.0  0.0  0.0 -1.0 -3.0 -2.0 -2.0 -1.0 -2.0  4.0 -4.0\n* -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0 -4.0  1.0\n')

class TestUnicodeStrings(unittest.TestCase):

    def test_needlemanwunsch_simple1(self):
        if False:
            while True:
                i = 10
        seq1 = 'ĞĀĀČŦ'
        seq2 = 'ĞĀŦ'
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.alphabet = None
        self.assertEqual(aligner.algorithm, 'Needleman-Wunsch')
        score = aligner.score(seq1, seq2)
        self.assertAlmostEqual(score, 3.0)
        alignments = aligner.align(seq1, seq2)
        self.assertEqual(len(alignments), 2)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'ĞĀĀČŦ\n||--|\nĞĀ--Ŧ\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 2], [4, 5]], [[0, 2], [2, 3]]])))
        alignment = alignments[1]
        self.assertAlmostEqual(alignment.score, 3.0)
        self.assertEqual(str(alignment), 'ĞĀĀČŦ\n|-|-|\nĞ-Ā-Ŧ\n')
        self.assertEqual(alignment.shape, (2, 5))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3], [4, 5]], [[0, 1], [1, 2], [2, 3]]])))

    def test_align_affine1_score(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        aligner.mode = 'global'
        aligner.alphabet = None
        aligner.match_score = 0
        aligner.mismatch_score = -1
        aligner.open_gap_score = -5
        aligner.extend_gap_score = -1
        self.assertEqual(aligner.algorithm, 'Gotoh global alignment algorithm')
        score = aligner.score('いい', 'あいいう')
        self.assertAlmostEqual(score, -7.0)

    def test_smithwaterman(self):
        if False:
            return 10
        aligner = Align.PairwiseAligner()
        aligner.mode = 'local'
        aligner.alphabet = None
        aligner.gap_score = -0.1
        self.assertEqual(aligner.algorithm, 'Smith-Waterman')
        score = aligner.score('ℵℷℶℷ', 'ℸℵℶℸ')
        self.assertAlmostEqual(score, 1.9)
        alignments = aligner.align('ℵℷℶℷ', 'ℸℵℶℸ')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), 'ℵℷℶ\n|-|\nℵ-ℶ\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[1, 2], [2, 3]]])))

    def test_gotoh_local(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner()
        aligner.alphabet = None
        aligner.mode = 'local'
        aligner.open_gap_score = -0.1
        aligner.extend_gap_score = 0.0
        self.assertEqual(aligner.algorithm, 'Gotoh local alignment algorithm')
        score = aligner.score('生物科物', '学生科学')
        self.assertAlmostEqual(score, 1.9)
        alignments = aligner.align('生物科物', '学生科学')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 1.9)
        self.assertEqual(str(alignment), '生物科\n|-|\n生-科\n')
        self.assertEqual(alignment.shape, (2, 3))
        self.assertTrue(np.array_equal(alignment.aligned, np.array([[[0, 1], [2, 3]], [[1, 2], [2, 3]]])))

class TestAlignerPickling(unittest.TestCase):

    def test_pickle_aligner_match_mismatch(self):
        if False:
            for i in range(10):
                print('nop')
        import pickle
        aligner = Align.PairwiseAligner()
        aligner.wildcard = 'X'
        aligner.match_score = 3
        aligner.mismatch_score = -2
        aligner.target_internal_open_gap_score = -2.5
        aligner.target_internal_extend_gap_score = -3.5
        aligner.target_left_open_gap_score = -2.5
        aligner.target_left_extend_gap_score = -3.5
        aligner.target_right_open_gap_score = -4
        aligner.target_right_extend_gap_score = -4
        aligner.query_internal_open_gap_score = -0.1
        aligner.query_internal_extend_gap_score = -2
        aligner.query_left_open_gap_score = -9
        aligner.query_left_extend_gap_score = +1
        aligner.query_right_open_gap_score = -1
        aligner.query_right_extend_gap_score = -2
        aligner.mode = 'local'
        state = pickle.dumps(aligner)
        pickled_aligner = pickle.loads(state)
        self.assertEqual(aligner.wildcard, pickled_aligner.wildcard)
        self.assertAlmostEqual(aligner.match_score, pickled_aligner.match_score)
        self.assertAlmostEqual(aligner.mismatch_score, pickled_aligner.mismatch_score)
        self.assertIsNone(pickled_aligner.substitution_matrix)
        self.assertAlmostEqual(aligner.target_internal_open_gap_score, pickled_aligner.target_internal_open_gap_score)
        self.assertAlmostEqual(aligner.target_internal_extend_gap_score, pickled_aligner.target_internal_extend_gap_score)
        self.assertAlmostEqual(aligner.target_left_open_gap_score, pickled_aligner.target_left_open_gap_score)
        self.assertAlmostEqual(aligner.target_left_extend_gap_score, pickled_aligner.target_left_extend_gap_score)
        self.assertAlmostEqual(aligner.target_right_open_gap_score, pickled_aligner.target_right_open_gap_score)
        self.assertAlmostEqual(aligner.target_right_extend_gap_score, pickled_aligner.target_right_extend_gap_score)
        self.assertAlmostEqual(aligner.query_internal_open_gap_score, pickled_aligner.query_internal_open_gap_score)
        self.assertAlmostEqual(aligner.query_internal_extend_gap_score, pickled_aligner.query_internal_extend_gap_score)
        self.assertAlmostEqual(aligner.query_left_open_gap_score, pickled_aligner.query_left_open_gap_score)
        self.assertAlmostEqual(aligner.query_left_extend_gap_score, pickled_aligner.query_left_extend_gap_score)
        self.assertAlmostEqual(aligner.query_right_open_gap_score, pickled_aligner.query_right_open_gap_score)
        self.assertAlmostEqual(aligner.query_right_extend_gap_score, pickled_aligner.query_right_extend_gap_score)
        self.assertEqual(aligner.mode, pickled_aligner.mode)

    def test_pickle_aligner_substitution_matrix(self):
        if False:
            while True:
                i = 10
        try:
            from Bio.Align import substitution_matrices
        except ImportError:
            return
        import pickle
        aligner = Align.PairwiseAligner()
        aligner.wildcard = 'N'
        aligner.substitution_matrix = substitution_matrices.load('BLOSUM80')
        aligner.target_internal_open_gap_score = -5
        aligner.target_internal_extend_gap_score = -3
        aligner.target_left_open_gap_score = -2
        aligner.target_left_extend_gap_score = -3
        aligner.target_right_open_gap_score = -4.5
        aligner.target_right_extend_gap_score = -4.3
        aligner.query_internal_open_gap_score = -2
        aligner.query_internal_extend_gap_score = -2.5
        aligner.query_left_open_gap_score = -9.1
        aligner.query_left_extend_gap_score = +1.7
        aligner.query_right_open_gap_score = -1.9
        aligner.query_right_extend_gap_score = -2.0
        aligner.mode = 'global'
        state = pickle.dumps(aligner)
        pickled_aligner = pickle.loads(state)
        self.assertEqual(aligner.wildcard, pickled_aligner.wildcard)
        self.assertIsNone(pickled_aligner.match_score)
        self.assertIsNone(pickled_aligner.mismatch_score)
        self.assertTrue((aligner.substitution_matrix == pickled_aligner.substitution_matrix).all())
        self.assertEqual(aligner.substitution_matrix.alphabet, pickled_aligner.substitution_matrix.alphabet)
        self.assertAlmostEqual(aligner.target_internal_open_gap_score, pickled_aligner.target_internal_open_gap_score)
        self.assertAlmostEqual(aligner.target_internal_extend_gap_score, pickled_aligner.target_internal_extend_gap_score)
        self.assertAlmostEqual(aligner.target_left_open_gap_score, pickled_aligner.target_left_open_gap_score)
        self.assertAlmostEqual(aligner.target_left_extend_gap_score, pickled_aligner.target_left_extend_gap_score)
        self.assertAlmostEqual(aligner.target_right_open_gap_score, pickled_aligner.target_right_open_gap_score)
        self.assertAlmostEqual(aligner.target_right_extend_gap_score, pickled_aligner.target_right_extend_gap_score)
        self.assertAlmostEqual(aligner.query_internal_open_gap_score, pickled_aligner.query_internal_open_gap_score)
        self.assertAlmostEqual(aligner.query_internal_extend_gap_score, pickled_aligner.query_internal_extend_gap_score)
        self.assertAlmostEqual(aligner.query_left_open_gap_score, pickled_aligner.query_left_open_gap_score)
        self.assertAlmostEqual(aligner.query_left_extend_gap_score, pickled_aligner.query_left_extend_gap_score)
        self.assertAlmostEqual(aligner.query_right_open_gap_score, pickled_aligner.query_right_open_gap_score)
        self.assertAlmostEqual(aligner.query_right_extend_gap_score, pickled_aligner.query_right_extend_gap_score)
        self.assertEqual(aligner.mode, pickled_aligner.mode)

class TestAlignmentFormat(unittest.TestCase):

    def test_alignment_simple(self):
        if False:
            return 10
        chromosome = 'ACGATCAGCGAGCATNGAGCACTACGACAGCGAGTGACCACTATTCGCGATCAGGAGCAGATACTTTACGAGCATCGGC'
        transcript = 'AGCATCGAGCGACTTGAGTACTATTCATACTTTCGAGC'
        aligner = Align.PairwiseAligner()
        aligner.query_extend_gap_score = 0
        aligner.query_open_gap_score = -3
        aligner.target_gap_score = -3
        aligner.end_gap_score = 0
        aligner.mismatch = -1
        alignments = aligner.align(chromosome, transcript)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 19.0)
        self.assertEqual(str(alignment), 'target            0 ACGATCAGCGAGCATNGAGC-ACTACGACAGCGAGTGACCACTATTCGCGATCAGGAGCA\n                  0 ----------|||||.||||-|||-----------|||..|||||||-------------\nquery             0 ----------AGCATCGAGCGACT-----------TGAGTACTATTC-------------\n\ntarget           59 GATACTTTACGAGCATCGGC 79\n                 60 -|||||||-|||||------ 80\nquery            26 -ATACTTT-CGAGC------ 38\n')
        self.assertEqual(alignment.shape, (2, 80))
        self.assertEqual(alignment.format('psl'), '34\t2\t0\t1\t1\t1\t3\t26\t+\tquery\t38\t0\t38\ttarget\t79\t10\t73\t5\t10,3,12,7,5,\t0,11,14,26,33,\t10,20,34,60,68,\n')
        self.assertEqual(alignment.format('bed'), 'target\t10\t73\tquery\t19.0\t+\t10\t73\t0\t5\t10,3,12,7,5,\t0,10,24,50,58,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t1\t255\t10D10M1I3M11D12M14D7M1D5M6D\t*\t0\t0\tAGCATCGAGCGACTTGAGTACTATTCATACTTTCGAGC\t*\tAS:i:19\n')
        alignments = aligner.align(chromosome, reverse_complement(transcript), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 19.0)
        self.assertEqual(str(alignment), 'target            0 ACGATCAGCGAGCATNGAGC-ACTACGACAGCGAGTGACCACTATTCGCGATCAGGAGCA\n                  0 ----------|||||.||||-|||-----------|||..|||||||-------------\nquery            38 ----------AGCATCGAGCGACT-----------TGAGTACTATTC-------------\n\ntarget           59 GATACTTTACGAGCATCGGC 79\n                 60 -|||||||-|||||------ 80\nquery            12 -ATACTTT-CGAGC------  0\n')
        self.assertEqual(alignment.shape, (2, 80))
        self.assertEqual(alignment.format('psl'), '34\t2\t0\t1\t1\t1\t3\t26\t-\tquery\t38\t0\t38\ttarget\t79\t10\t73\t5\t10,3,12,7,5,\t0,11,14,26,33,\t10,20,34,60,68,\n')
        self.assertEqual(alignment.format('bed'), 'target\t10\t73\tquery\t19.0\t-\t10\t73\t0\t5\t10,3,12,7,5,\t0,10,24,50,58,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t1\t255\t10D10M1I3M11D12M14D7M1D5M6D\t*\t0\t0\tAGCATCGAGCGACTTGAGTACTATTCATACTTTCGAGC\t*\tAS:i:19\n')

    def test_alignment_end_gap(self):
        if False:
            print('Hello World!')
        aligner = Align.PairwiseAligner()
        aligner.gap_score = -1
        aligner.end_gap_score = 0
        aligner.mismatch = -10
        alignments = aligner.align('ACGTAGCATCAGC', 'CCCCACGTAGCATCAGC')
        self.assertEqual(len(alignments), 1)
        self.assertAlmostEqual(alignments.score, 13.0)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ----ACGTAGCATCAGC 13\n                  0 ----||||||||||||| 17\nquery             0 CCCCACGTAGCATCAGC 17\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t+\tquery\t17\t4\t17\ttarget\t13\t0\t13\t1\t13,\t4,\t0,\n')
        self.assertEqual(alignment.format('bed'), 'target\t0\t13\tquery\t13.0\t+\t0\t13\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t1\t255\t4I13M\t*\t0\t0\tCCCCACGTAGCATCAGC\t*\tAS:i:13\n')
        alignments = aligner.align('ACGTAGCATCAGC', reverse_complement('CCCCACGTAGCATCAGC'), strand='-')
        self.assertEqual(len(alignments), 1)
        self.assertAlmostEqual(alignments.score, 13.0)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ----ACGTAGCATCAGC 13\n                  0 ----||||||||||||| 17\nquery            17 CCCCACGTAGCATCAGC  0\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t-\tquery\t17\t0\t13\ttarget\t13\t0\t13\t1\t13,\t4,\t0,\n')
        self.assertEqual(alignment.format('bed'), 'target\t0\t13\tquery\t13.0\t-\t0\t13\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t1\t255\t4I13M\t*\t0\t0\tCCCCACGTAGCATCAGC\t*\tAS:i:13\n')
        alignments = aligner.align('CCCCACGTAGCATCAGC', 'ACGTAGCATCAGC')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13.0)
        self.assertEqual(str(alignment), 'target            0 CCCCACGTAGCATCAGC 17\n                  0 ----||||||||||||| 17\nquery             0 ----ACGTAGCATCAGC 13\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t+\tquery\t13\t0\t13\ttarget\t17\t4\t17\t1\t13,\t0,\t4,\n')
        self.assertEqual(alignment.format('bed'), 'target\t4\t17\tquery\t13.0\t+\t4\t17\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t1\t255\t4D13M\t*\t0\t0\tACGTAGCATCAGC\t*\tAS:i:13\n')
        alignments = aligner.align('CCCCACGTAGCATCAGC', reverse_complement('ACGTAGCATCAGC'), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13.0)
        self.assertEqual(str(alignment), 'target            0 CCCCACGTAGCATCAGC 17\n                  0 ----||||||||||||| 17\nquery            13 ----ACGTAGCATCAGC  0\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t-\tquery\t13\t0\t13\ttarget\t17\t4\t17\t1\t13,\t0,\t4,\n')
        self.assertEqual(alignment.format('bed'), 'target\t4\t17\tquery\t13.0\t-\t4\t17\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t1\t255\t4D13M\t*\t0\t0\tACGTAGCATCAGC\t*\tAS:i:13\n')
        alignments = aligner.align('ACGTAGCATCAGC', 'ACGTAGCATCAGCGGGG')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ACGTAGCATCAGC---- 13\n                  0 |||||||||||||---- 17\nquery             0 ACGTAGCATCAGCGGGG 17\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t+\tquery\t17\t0\t13\ttarget\t13\t0\t13\t1\t13,\t0,\t0,\n')
        self.assertEqual(alignment.format('bed'), 'target\t0\t13\tquery\t13.0\t+\t0\t13\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t1\t255\t13M4I\t*\t0\t0\tACGTAGCATCAGCGGGG\t*\tAS:i:13\n')
        alignments = aligner.align('ACGTAGCATCAGC', reverse_complement('ACGTAGCATCAGCGGGG'), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 ACGTAGCATCAGC---- 13\n                  0 |||||||||||||---- 17\nquery            17 ACGTAGCATCAGCGGGG  0\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t-\tquery\t17\t4\t17\ttarget\t13\t0\t13\t1\t13,\t0,\t0,\n')
        self.assertEqual(alignment.format('bed'), 'target\t0\t13\tquery\t13.0\t-\t0\t13\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t1\t255\t13M4I\t*\t0\t0\tACGTAGCATCAGCGGGG\t*\tAS:i:13\n')
        alignments = aligner.align('ACGTAGCATCAGCGGGG', 'ACGTAGCATCAGC')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13.0)
        self.assertEqual(str(alignment), 'target            0 ACGTAGCATCAGCGGGG 17\n                  0 |||||||||||||---- 17\nquery             0 ACGTAGCATCAGC---- 13\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t+\tquery\t13\t0\t13\ttarget\t17\t0\t13\t1\t13,\t0,\t0,\n')
        self.assertEqual(alignment.format('bed'), 'target\t0\t13\tquery\t13.0\t+\t0\t13\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t1\t255\t13M4D\t*\t0\t0\tACGTAGCATCAGC\t*\tAS:i:13\n')
        alignments = aligner.align('ACGTAGCATCAGCGGGG', reverse_complement('ACGTAGCATCAGC'), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertAlmostEqual(alignment.score, 13.0)
        self.assertEqual(str(alignment), 'target            0 ACGTAGCATCAGCGGGG 17\n                  0 |||||||||||||---- 17\nquery            13 ACGTAGCATCAGC----  0\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '13\t0\t0\t0\t0\t0\t0\t0\t-\tquery\t13\t0\t13\ttarget\t17\t0\t13\t1\t13,\t0,\t0,\n')
        self.assertEqual(alignment.format('bed'), 'target\t0\t13\tquery\t13.0\t-\t0\t13\t0\t1\t13,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t1\t255\t13M4D\t*\t0\t0\tACGTAGCATCAGC\t*\tAS:i:13\n')

    def test_alignment_wildcard(self):
        if False:
            for i in range(10):
                print('nop')
        aligner = Align.PairwiseAligner()
        aligner.gap_score = -10
        aligner.mismatch = -2
        aligner.wildcard = 'N'
        target = 'TTTTTNACGCTCGAGCAGCTACG'
        query = 'ACGATCGAGCNGCTACGCCCNC'
        aligner.mode = 'local'
        alignments = aligner.align(target, query)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            6 ACGCTCGAGCAGCTACG 23\n                  0 |||.||||||.|||||| 17\nquery             0 ACGATCGAGCNGCTACG 17\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t+\tquery\t22\t0\t17\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t+\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t7\t255\t17M5S\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
        alignments = aligner.align(target, reverse_complement(query), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            6 ACGCTCGAGCAGCTACG 23\n                  0 |||.||||||.|||||| 17\nquery            22 ACGATCGAGCNGCTACG  5\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t-\tquery\t22\t5\t22\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t-\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t7\t255\t17M5S\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
        alignments = aligner.align(Seq(target), Seq(query))
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            6 ACGCTCGAGCAGCTACG 23\n                  0 |||.||||||.|||||| 17\nquery             0 ACGATCGAGCNGCTACG 17\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t+\tquery\t22\t0\t17\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t+\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t7\t255\t17M5S\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
        alignments = aligner.align(Seq(target), Seq(query).reverse_complement(), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            6 ACGCTCGAGCAGCTACG 23\n                  0 |||.||||||.|||||| 17\nquery            22 ACGATCGAGCNGCTACG  5\n')
        self.assertEqual(alignment.shape, (2, 17))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t-\tquery\t22\t5\t22\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t-\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t7\t255\t17M5S\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
        aligner.mode = 'global'
        aligner.end_gap_score = 0
        alignments = aligner.align(target, query)
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 TTTTTNACGCTCGAGCAGCTACG----- 23\n                  0 ------|||.||||||.||||||----- 28\nquery             0 ------ACGATCGAGCNGCTACGCCCNC 22\n')
        self.assertEqual(alignment.shape, (2, 28))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t+\tquery\t22\t0\t17\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t+\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t1\t255\t6D17M5I\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
        alignments = aligner.align(target, reverse_complement(query), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 TTTTTNACGCTCGAGCAGCTACG----- 23\n                  0 ------|||.||||||.||||||----- 28\nquery            22 ------ACGATCGAGCNGCTACGCCCNC  0\n')
        self.assertEqual(alignment.shape, (2, 28))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t-\tquery\t22\t5\t22\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t-\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t1\t255\t6D17M5I\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
        alignments = aligner.align(Seq(target), Seq(query))
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 TTTTTNACGCTCGAGCAGCTACG----- 23\n                  0 ------|||.||||||.||||||----- 28\nquery             0 ------ACGATCGAGCNGCTACGCCCNC 22\n')
        self.assertEqual(alignment.shape, (2, 28))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t+\tquery\t22\t0\t17\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t+\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t0\ttarget\t1\t255\t6D17M5I\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
        alignments = aligner.align(Seq(target), Seq(query).reverse_complement(), strand='-')
        self.assertEqual(len(alignments), 1)
        alignment = alignments[0]
        self.assertEqual(str(alignment), 'target            0 TTTTTNACGCTCGAGCAGCTACG----- 23\n                  0 ------|||.||||||.||||||----- 28\nquery            22 ------ACGATCGAGCNGCTACGCCCNC  0\n')
        self.assertEqual(alignment.shape, (2, 28))
        self.assertEqual(alignment.format('psl'), '15\t1\t0\t1\t0\t0\t0\t0\t-\tquery\t22\t5\t22\ttarget\t23\t6\t23\t1\t17,\t0,\t6,\n')
        self.assertEqual(alignment.format('bed'), 'target\t6\t23\tquery\t13.0\t-\t6\t23\t0\t1\t17,\t0,\n')
        self.assertEqual(alignment.format('sam'), 'query\t16\ttarget\t1\t255\t6D17M5I\t*\t0\t0\tACGATCGAGCNGCTACGCCCNC\t*\tAS:i:13\n')
if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)