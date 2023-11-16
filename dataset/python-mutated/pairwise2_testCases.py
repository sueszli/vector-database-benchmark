"""Tests for pairwise2 module.

Put new test case here, the classes here will be imported and run
as TestCases in ``test_pairwise2.py`` and ``test_pairwise2_no_C.py``
with or without complementing C extensions.

"""
import pickle
import unittest
import warnings
from Bio import BiopythonWarning
from Bio import pairwise2
from Bio.Align import substitution_matrices

class TestPairwiseErrorConditions(unittest.TestCase):
    """Test several error conditions."""

    def test_function_name(self):
        if False:
            print('Hello World!')
        'Test for wrong function names.'
        self.assertRaises(AttributeError, lambda : pairwise2.align.globalxxx)
        self.assertRaises(AttributeError, lambda : pairwise2.align.localxxx)
        self.assertRaises(AttributeError, lambda : pairwise2.align.glocalxx)
        self.assertRaises(AttributeError, lambda : pairwise2.align.globalax)
        self.assertRaises(AttributeError, lambda : pairwise2.align.globalxa)

    def test_function_parameters(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for number of parameters.'
        self.assertRaises(TypeError, pairwise2.align.globalxx, 'A')
        self.assertRaises(TypeError, pairwise2.align.globalxx, 'A', 'C', {'matrix_only': True})
        self.assertRaises(TypeError, pairwise2.align.globalxx, 'A', ['C'])
        self.assertRaises(TypeError, pairwise2.align.globalxx, ['A'], ['C'])
        alignment = pairwise2.align.globalxx('A', '')
        self.assertEqual(alignment, [])
        self.assertRaises(ValueError, pairwise2.align.globalxs, 'A', 'C', 5, -1)
        self.assertRaises(ValueError, pairwise2.align.globalxs, 'A', 'C', -5, 1)
        self.assertRaises(ValueError, pairwise2.align.globalxs, 'A', 'C', -1, -5)

    def test_param_names(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for unknown parameter in parameter names.'
        a = pairwise2.align.alignment_function('globalxx')
        a.param_names = ['Hello']
        self.assertRaises(ValueError, a.decode, 'Bye')

    def test_warnings(self):
        if False:
            print('Hello World!')
        'Test for warnings.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            pairwise2.align.localxx('GA', 'CGA', penalize_end_gaps=True)
            self.assertEqual(len(w), 1)
            self.assertEqual(w[-1].category, BiopythonWarning)
            self.assertIn('should not', str(w[-1].message))

class TestPairwiseKeywordUsage(unittest.TestCase):
    """Tests for keyword usage."""

    def test_keywords(self):
        if False:
            i = 10
            return i + 15
        'Test equality of calls with and without keywords.'
        aligns = pairwise2.align.globalxx('GAACT', 'GAT')
        aligns_kw = pairwise2.align.globalxx(sequenceA='GAACT', sequenceB='GAT')
        self.assertEqual(aligns, aligns_kw)
        aligns = pairwise2.align.globalmx('GAACT', 'GAT', 5, -4)
        aligns_kw = pairwise2.align.globalmx(sequenceA='GAACT', sequenceB='GAT', match=5, mismatch=-4)
        self.assertEqual(aligns, aligns_kw)

class TestPairwiseGlobal(unittest.TestCase):
    """Test some usual global alignments."""

    def test_globalxx_simple(self):
        if False:
            print('Hello World!')
        'Test globalxx.'
        aligns = pairwise2.align.globalxx('GAACT', 'GAT')
        self.assertEqual(len(aligns), 2)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GAACT\n| | |\nG-A-T\n  Score=3\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GAACT\n||  |\nGA--T\n  Score=3\n')

    def test_globalxx_simple2(self):
        if False:
            while True:
                i = 10
        'Do the same test with sequence order reversed.'
        aligns = pairwise2.align.globalxx('GAT', 'GAACT')
        self.assertEqual(len(aligns), 2)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'G-A-T\n| | |\nGAACT\n  Score=3\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GA--T\n||  |\nGAACT\n  Score=3\n')

    def test_one_alignment_only(self):
        if False:
            for i in range(10):
                print('nop')
        'Test one_alignment_only parameter.'
        aligns = pairwise2.align.globalxx('ACCGT', 'ACG')
        self.assertEqual(len(aligns), 2)
        aligns = pairwise2.align.globalxx('ACCGT', 'ACG', one_alignment_only=True)
        self.assertEqual(len(aligns), 1)

    def test_list_input(self):
        if False:
            print('Hello World!')
        'Do a global alignment with sequences supplied as lists.'
        aligns = pairwise2.align.globalxx(['Gly', 'Ala', 'Thr'], ['Gly', 'Ala', 'Ala', 'Cys', 'Thr'], gap_char=['---'])
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        self.assertEqual(score, 3)
        self.assertEqual(seq1, ['Gly', '---', 'Ala', '---', 'Thr'])
        self.assertEqual(seq2, ['Gly', 'Ala', 'Ala', 'Cys', 'Thr'])

class TestPairwiseLocal(unittest.TestCase):
    """Test some simple local alignments."""

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.blosum62 = substitution_matrices.load('BLOSUM62')

    def test_localxs_1(self):
        if False:
            i = 10
            return i + 15
        'Test localxx.'
        aligns = sorted(pairwise2.align.localxs('AxBx', 'zABz', -0.1, 0))
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '1 AxB\n  | |\n2 A-B\n  Score=1.9\n')

    def test_localxs_2(self):
        if False:
            return 10
        'Test localxx with ``full_sequences=True``.'
        aligns = sorted(pairwise2.align.localxs('AxBx', 'zABz', -0.1, 0))
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end, full_sequences=True)
        self.assertEqual(alignment, '-AxBx\n | | \nzA-Bz\n  Score=1.9\n')

    def test_localds_zero_score_segments_symmetric(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if alignment is independent on direction of sequence.'
        aligns1 = pairwise2.align.localds('CWHISLKM', 'CWHGISGLKM', self.blosum62, -11, -1)
        aligns2 = pairwise2.align.localds('MKLSIHWC', 'MKLGSIGHWC', self.blosum62, -11, -1)
        self.assertEqual(len(aligns1), len(aligns2))

    def test_localxs_generic(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the generic method with local alignments.'
        aligns = sorted(pairwise2.align.localxs('AxBx', 'zABz', -0.1, 0, force_generic=True))
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '1 AxB\n  | |\n2 A-B\n  Score=1.9\n')

    def test_localms(self):
        if False:
            for i in range(10):
                print('nop')
        'Two different local alignments.'
        aligns = sorted(pairwise2.align.localms('xxxABCDxxx', 'zzzABzzCDz', 1, -0.5, -3, -1))
        alignment = pairwise2.format_alignment(*aligns[0])
        self.assertEqual(alignment, '6 CD\n  ||\n8 CD\n  Score=2\n')
        alignment = pairwise2.format_alignment(*aligns[1])
        self.assertEqual(alignment, '4 AB\n  ||\n4 AB\n  Score=2\n')

    def test_blosum62(self):
        if False:
            while True:
                i = 10
        'Test localds with blosum62.'
        self.assertEqual(1, self.blosum62['K', 'Q'])
        self.assertEqual(4, self.blosum62['A', 'A'])
        self.assertEqual(8, self.blosum62['H', 'H'])
        alignments = pairwise2.align.localds('VKAHGKKV', 'FQAHCAGV', self.blosum62, -4, -4)
        for a in alignments:
            self.assertEqual(pairwise2.format_alignment(*a), '2 KAH\n  .||\n2 QAH\n  Score=13\n')

    def test_empty_result(self):
        if False:
            while True:
                i = 10
        'Return no alignment.'
        self.assertEqual(pairwise2.align.localxx('AT', 'GC'), [])

class TestScoreOnly(unittest.TestCase):
    """Test parameter ``score_only``."""

    def test_score_only_global(self):
        if False:
            return 10
        'Test ``score_only`` in a global alignment.'
        aligns1 = pairwise2.align.globalxx('GAACT', 'GAT')
        aligns2 = pairwise2.align.globalxx('GAACT', 'GAT', score_only=True)
        self.assertEqual(aligns1[0][2], aligns2)

    def test_score_only_local(self):
        if False:
            i = 10
            return i + 15
        'Test ``score_only`` in a local alignment.'
        aligns1 = pairwise2.align.localms('xxxABCDxxx', 'zzzABzzCDz', 1, -0.5, -3, -1)
        aligns2 = pairwise2.align.localms('xxxABCDxxx', 'zzzABzzCDz', 1, -0.5, -3, -1, score_only=True)
        self.assertEqual(aligns1[0][2], aligns2)

class TestPairwiseOpenPenalty(unittest.TestCase):
    """Alignments with gap-open penalty."""

    def test_match_score_open_penalty1(self):
        if False:
            return 10
        'Test 1.'
        aligns = pairwise2.align.globalms('AA', 'A', 2.0, -1, -0.1, 0)
        self.assertEqual(len(aligns), 2)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'AA\n |\n-A\n  Score=1.9\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'AA\n| \nA-\n  Score=1.9\n')

    def test_match_score_open_penalty2(self):
        if False:
            print('Hello World!')
        'Test 2.'
        aligns = pairwise2.align.globalms('GAA', 'GA', 1.5, 0, -0.1, 0)
        self.assertEqual(len(aligns), 2)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GAA\n| |\nG-A\n  Score=2.9\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GAA\n|| \nGA-\n  Score=2.9\n')

    def test_match_score_open_penalty3(self):
        if False:
            return 10
        'Test 3.'
        aligns = pairwise2.align.globalxs('GAACT', 'GAT', -0.1, 0)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GAACT\n||  |\nGA--T\n  Score=2.9\n')

    def test_match_score_open_penalty4(self):
        if False:
            return 10
        'Test 4.'
        aligns = pairwise2.align.globalms('GCT', 'GATA', 1, -2, -0.1, 0)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GC-T-\n|  | \nG-ATA\n  Score=1.7\n')

class TestPairwiseExtendPenalty(unittest.TestCase):
    """Alignments with gap-extend penalties."""

    def test_extend_penalty1(self):
        if False:
            print('Hello World!')
        'Test 1.'
        aligns = pairwise2.align.globalxs('GACT', 'GT', -0.5, -0.2)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n|  |\nG--T\n  Score=1.3\n')

    def test_extend_penalty2(self):
        if False:
            return 10
        'Test 2.'
        aligns = pairwise2.align.globalxs('GACT', 'GT', -1.5, -0.2)
        self.assertEqual(len(aligns), 1)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n|  |\nG--T\n  Score=0.3\n')

class TestPairwisePenalizeExtendWhenOpening(unittest.TestCase):
    """Alignment with ``penalize_extend_when_opening``."""

    def test_penalize_extend_when_opening(self):
        if False:
            print('Hello World!')
        'Add gap-extend penalty to gap-opening penalty.'
        aligns = pairwise2.align.globalxs('GACT', 'GT', -0.2, -1.5, penalize_extend_when_opening=1)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n|  |\nG--T\n  Score=-1.2\n')

class TestPairwisePenalizeEndgaps(unittest.TestCase):
    """Alignments with end-gaps penalized or not."""

    def test_penalize_end_gaps(self):
        if False:
            print('Hello World!')
        'Turn off end-gap penalties.'
        aligns = pairwise2.align.globalxs('GACT', 'GT', -0.8, -0.2, penalize_end_gaps=0)
        self.assertEqual(len(aligns), 3)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n  .|\n--GT\n  Score=1\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n|  |\nG--T\n  Score=1\n')
        (seq1, seq2, score, begin, end) = aligns[2]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n|.  \nGT--\n  Score=1\n')

    def test_penalize_end_gaps2(self):
        if False:
            i = 10
            return i + 15
        'Do the same, but use the generic method (with the same result).'
        aligns = pairwise2.align.globalxs('GACT', 'GT', -0.8, -0.2, penalize_end_gaps=0, force_generic=True)
        self.assertEqual(len(aligns), 3)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n  .|\n--GT\n  Score=1\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n|  |\nG--T\n  Score=1\n')
        (seq1, seq2, score, begin, end) = aligns[2]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GACT\n|.  \nGT--\n  Score=1\n')

    def test_separate_penalize_end_gaps(self):
        if False:
            i = 10
            return i + 15
        'Test alignment where end-gaps are differently penalized.'
        align = pairwise2.align.globalms('AT', 'AGG', 1.0, -0.5, -1.75, -0.25, penalize_end_gaps=(True, False))
        self.assertEqual(align[0], ('A--T', 'AGG-', -1.0, 0, 4))

class TestPairwiseSeparateGapPenalties(unittest.TestCase):
    """Alignments with separate gap-open penalties for both sequences."""

    def test_separate_gap_penalties1(self):
        if False:
            i = 10
            return i + 15
        'Test 1.'
        aligns = pairwise2.align.localxd('GAT', 'GTCT', -0.3, 0, -0.8, 0)
        self.assertEqual(len(aligns), 2)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'G-AT\n| .|\nGTCT\n  Score=1.7\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'GA-T\n|. |\nGTCT\n  Score=1.7\n')

    def test_separate_gap_penalties2(self):
        if False:
            i = 10
            return i + 15
        'Test 2.'
        aligns = pairwise2.align.localxd('GAT', 'GTCT', -0.5, 0, -0.2, 0)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '1 GAT\n  | |\n1 G-T\n  Score=1.8\n')

class TestPairwiseSeparateGapPenaltiesWithExtension(unittest.TestCase):
    """Alignments with separate gap-extension penalties for both sequences."""

    def test_separate_gap_penalties_with_extension(self):
        if False:
            while True:
                i = 10
        'Test separate gap-extension penalties and list input.'
        aligns = pairwise2.align.localxd(list('GAAT'), list('GTCCT'), -0.1, 0, -0.1, -0.1, gap_char=['-'])
        self.assertEqual(len(aligns), 3)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'G - A A T \n|   . . | \nG T C C T \n  Score=1.9\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'G A - A T \n| .   . | \nG T C C T \n  Score=1.9\n')
        (seq1, seq2, score, begin, end) = aligns[2]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'G A A - T \n| . .   | \nG T C C T \n  Score=1.9\n')

class TestPairwiseMatchDictionary(unittest.TestCase):
    """Alignments with match dictionaries."""
    match_dict = {('A', 'A'): 1.5, ('A', 'T'): 0.5, ('T', 'T'): 1.0}

    def test_match_dictionary1(self):
        if False:
            return 10
        'Test 1.'
        aligns = pairwise2.align.localds('ATAT', 'ATT', self.match_dict, -0.5, 0)
        self.assertEqual(len(aligns), 2)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'ATAT\n|| |\nAT-T\n  Score=3\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '1 ATA\n  ||.\n1 ATT\n  Score=3\n')

    def test_match_dictionary2(self):
        if False:
            print('Hello World!')
        'Test 2.'
        aligns = pairwise2.align.localds('ATAT', 'ATT', self.match_dict, -1, 0)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '1 ATA\n  ||.\n1 ATT\n  Score=3\n')

    def test_match_dictionary3(self):
        if False:
            print('Hello World!')
        'Test 3.'
        aligns = pairwise2.align.localds('ATT', 'ATAT', self.match_dict, -1, 0)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '1 ATT\n  ||.\n1 ATA\n  Score=3\n')

class TestPairwiseOneCharacter(unittest.TestCase):
    """Alignments where one sequence has length 1."""

    def test_align_one_char1(self):
        if False:
            for i in range(10):
                print('nop')
        'Test sequence with only one match.'
        aligns = pairwise2.align.localxs('abcde', 'c', -0.3, -0.1)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '3 c\n  |\n1 c\n  Score=1\n')

    def test_align_one_char2(self):
        if False:
            return 10
        'Test sequences with two possible match positions.'
        aligns = pairwise2.align.localxs('abcce', 'c', -0.3, -0.1)
        self.assertEqual(len(aligns), 2)
        aligns.sort()
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '4 c\n  |\n1 c\n  Score=1\n')
        (seq1, seq2, score, begin, end) = aligns[1]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, '3 c\n  |\n1 c\n  Score=1\n')

    def test_align_one_char3(self):
        if False:
            for i in range(10):
                print('nop')
        'Like test 1, but global alignment.'
        aligns = pairwise2.align.globalxs('abcde', 'c', -0.3, -0.1)
        self.assertEqual(len(aligns), 1)
        (seq1, seq2, score, begin, end) = aligns[0]
        alignment = pairwise2.format_alignment(seq1, seq2, score, begin, end)
        self.assertEqual(alignment, 'abcde\n  |  \n--c--\n  Score=0.2\n')

class TestPersiteGapPenalties(unittest.TestCase):
    """Check gap penalty callbacks use correct gap opening position.

    This tests that the gap penalty callbacks are really being used
    with the correct gap opening position.
    """

    def test_gap_here_only_1(self):
        if False:
            for i in range(10):
                print('nop')
        'Open a gap in second sequence only.'
        seq1 = 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA'
        seq2 = 'AABBBAAAACCCCAAAABBBAA'

        def no_gaps(x, y):
            if False:
                print('Hello World!')
            'Very expensive to open a gap in seq1.'
            return -2000 - y

        def specific_gaps(x, y):
            if False:
                return 10
            'Very expensive to open a gap in seq2.\n\n            ...unless it is in one of the allowed positions:\n            '
            breaks = [0, 11, len(seq2)]
            return -2 - y if x in breaks else -2000 - y
        alignments = pairwise2.align.globalmc(seq1, seq2, 1, -1, no_gaps, specific_gaps)
        self.assertEqual(len(alignments), 1)
        formatted = pairwise2.format_alignment(*alignments[0])
        self.assertEqual(formatted, 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA\n  |||||||||||          |||||||||||  \n--AABBBAAAACC----------CCAAAABBBAA--\n  Score=2\n')

    def test_gap_here_only_2(self):
        if False:
            for i in range(10):
                print('nop')
        'Force a bad alignment.\n\n        Forces a bad alignment by having a very expensive gap penalty\n        where one would normally expect a gap, and a cheap gap penalty\n        in another place.\n        '
        seq1 = 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA'
        seq2 = 'AABBBAAAACCCCAAAABBBAA'

        def no_gaps(x, y):
            if False:
                for i in range(10):
                    print('nop')
            'Very expensive to open a gap in seq1.'
            return -2000 - y

        def specific_gaps(x, y):
            if False:
                return 10
            'Very expensive to open a gap in seq2.\n\n            ...unless it is in one of the allowed positions:\n            '
            breaks = [0, 3, len(seq2)]
            return -2 - y if x in breaks else -2000 - y
        alignments = pairwise2.align.globalmc(seq1, seq2, 1, -1, no_gaps, specific_gaps)
        self.assertEqual(len(alignments), 1)
        formatted = pairwise2.format_alignment(*alignments[0])
        self.assertEqual(formatted, 'AAAABBBAAAACCCCCCCCCCCCCCAAAABBBAAAA\n  |||          ......|||||||||||||  \n--AAB----------BBAAAACCCCAAAABBBAA--\n  Score=-10\n')

class TestOtherFunctions(unittest.TestCase):
    """Test remaining non-tested private methods."""

    def test_clean_alignments(self):
        if False:
            i = 10
            return i + 15
        '``_clean_alignments`` removes redundant and wrong alignments.'
        alns = [('ACCGT', 'AC-G-', 3.0, 0, 4), ('ACCGT', 'AC-G-', 3.0, 1, 1), ('ACCGT', 'A-CG-', 3.0, 0, 4), ('ACCGT', 'AC-G-', 3.0, 0, 4), ('ACCGT', 'A-CG-', 3.0, 0, 4)]
        expected = [('ACCGT', 'AC-G-', 3.0, 0, 4), ('ACCGT', 'A-CG-', 3.0, 0, 4)]
        result = pairwise2._clean_alignments(alns)
        self.assertEqual(expected, result)

    def test_alignments_can_be_pickled(self):
        if False:
            return 10
        alns = [('ACCGT', 'AC-G-', 3.0, 0, 4)]
        expected = [('ACCGT', 'AC-G-', 3.0, 0, 4)]
        result = pickle.loads(pickle.dumps(pairwise2._clean_alignments(alns)))
        self.assertEqual(expected, result)

    def test_print_matrix(self):
        if False:
            while True:
                i = 10
        '``print_matrix`` prints nested lists as nice matrices.'
        import sys
        from io import StringIO
        out = StringIO()
        sys.stdout = out
        pairwise2.print_matrix([[0.0, -1.0, -1.5, -2.0], [-1.0, 4.0, 3.0, 2.5], [-1.5, 3.0, 8.0, 7.0], [-2.0, 2.5, 7.0, 6.0], [-2.5, 2.0, 6.5, 11.0], [-3.0, 1.5, 6.0, 10.0]])
        self.assertEqual(out.getvalue(), ' 0.0  -1.0  -1.5  -2.0 \n-1.0   4.0   3.0   2.5 \n-1.5   3.0   8.0   7.0 \n-2.0   2.5   7.0   6.0 \n-2.5   2.0   6.5  11.0 \n-3.0   1.5   6.0  10.0 \n')
        sys.stdout = sys.__stdout__

    def test_recover_alignments(self):
        if False:
            i = 10
            return i + 15
        'One possible start position in local alignment is not a match.'
        self.assertEqual(len(pairwise2.align.localxx('AC', 'GA')), 1)
if __name__ == '__main__':
    if pairwise2.rint != pairwise2._python_rint:
        runner = unittest.TextTestRunner(verbosity=2)
        unittest.main(testRunner=runner, exit=False)
    else:
        print('Import of C functions failed. Only testing pure Python fallback functions.')
    pairwise2._make_score_matrix_fast = pairwise2._python_make_score_matrix_fast
    pairwise2.rint = pairwise2._python_rint
    runner = unittest.TextTestRunner(verbosity=2)
    unittest.main(testRunner=runner)