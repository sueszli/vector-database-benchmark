"""
Tests for BLEU translation evaluation metric
"""
import io
import unittest
import numpy as np
from nltk.data import find
from nltk.translate.bleu_score import SmoothingFunction, brevity_penalty, closest_ref_length, corpus_bleu, modified_precision, sentence_bleu

class TestBLEU(unittest.TestCase):

    def test_modified_precision(self):
        if False:
            i = 10
            return i + 15
        '\n        Examples from the original BLEU paper\n        https://www.aclweb.org/anthology/P02-1040.pdf\n        '
        ref1 = 'the cat is on the mat'.split()
        ref2 = 'there is a cat on the mat'.split()
        hyp1 = 'the the the the the the the'.split()
        references = [ref1, ref2]
        hyp1_unigram_precision = float(modified_precision(references, hyp1, n=1))
        assert round(hyp1_unigram_precision, 4) == 0.2857
        self.assertAlmostEqual(hyp1_unigram_precision, 0.28571428, places=4)
        assert float(modified_precision(references, hyp1, n=2)) == 0.0
        ref1 = str('It is a guide to action that ensures that the military will forever heed Party commands').split()
        ref2 = str('It is the guiding principle which guarantees the military forces always being under the command of the Party').split()
        ref3 = str('It is the practical guide for the army always to heed the directions of the party').split()
        hyp1 = 'of the'.split()
        references = [ref1, ref2, ref3]
        assert float(modified_precision(references, hyp1, n=1)) == 1.0
        assert float(modified_precision(references, hyp1, n=2)) == 1.0
        hyp1 = str('It is a guide to action which ensures that the military always obeys the commands of the party').split()
        hyp2 = str('It is to insure the troops forever hearing the activity guidebook that party direct').split()
        references = [ref1, ref2, ref3]
        hyp1_unigram_precision = float(modified_precision(references, hyp1, n=1))
        hyp2_unigram_precision = float(modified_precision(references, hyp2, n=1))
        self.assertAlmostEqual(hyp1_unigram_precision, 0.94444444, places=4)
        self.assertAlmostEqual(hyp2_unigram_precision, 0.57142857, places=4)
        assert round(hyp1_unigram_precision, 4) == 0.9444
        assert round(hyp2_unigram_precision, 4) == 0.5714
        hyp1_bigram_precision = float(modified_precision(references, hyp1, n=2))
        hyp2_bigram_precision = float(modified_precision(references, hyp2, n=2))
        self.assertAlmostEqual(hyp1_bigram_precision, 0.58823529, places=4)
        self.assertAlmostEqual(hyp2_bigram_precision, 0.07692307, places=4)
        assert round(hyp1_bigram_precision, 4) == 0.5882
        assert round(hyp2_bigram_precision, 4) == 0.0769

    def test_brevity_penalty(self):
        if False:
            return 10
        references = [['a'] * 11, ['a'] * 8]
        hypothesis = ['a'] * 7
        hyp_len = len(hypothesis)
        closest_ref_len = closest_ref_length(references, hyp_len)
        self.assertAlmostEqual(brevity_penalty(closest_ref_len, hyp_len), 0.8669, places=4)
        references = [['a'] * 11, ['a'] * 8, ['a'] * 6, ['a'] * 7]
        hypothesis = ['a'] * 7
        hyp_len = len(hypothesis)
        closest_ref_len = closest_ref_length(references, hyp_len)
        assert brevity_penalty(closest_ref_len, hyp_len) == 1.0

    def test_zero_matches(self):
        if False:
            print('Hello World!')
        references = ['The candidate has no alignment to any of the references'.split()]
        hypothesis = 'John loves Mary'.split()
        for n in range(1, len(hypothesis)):
            weights = (1.0 / n,) * n
            assert sentence_bleu(references, hypothesis, weights) == 0

    def test_full_matches(self):
        if False:
            return 10
        references = ['John loves Mary'.split()]
        hypothesis = 'John loves Mary'.split()
        for n in range(1, len(hypothesis)):
            weights = (1.0 / n,) * n
            assert sentence_bleu(references, hypothesis, weights) == 1.0

    def test_partial_matches_hypothesis_longer_than_reference(self):
        if False:
            i = 10
            return i + 15
        references = ['John loves Mary'.split()]
        hypothesis = 'John loves Mary who loves Mike'.split()
        self.assertAlmostEqual(sentence_bleu(references, hypothesis), 0.0, places=4)
        try:
            self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
        except AttributeError:
            pass

class TestBLEUFringeCases(unittest.TestCase):

    def test_case_where_n_is_bigger_than_hypothesis_length(self):
        if False:
            while True:
                i = 10
        references = ['John loves Mary ?'.split()]
        hypothesis = 'John loves Mary'.split()
        n = len(hypothesis) + 1
        weights = (1.0 / n,) * n
        self.assertAlmostEqual(sentence_bleu(references, hypothesis, weights), 0.0, places=4)
        try:
            self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
        except AttributeError:
            pass
        references = ['John loves Mary'.split()]
        hypothesis = 'John loves Mary'.split()
        self.assertAlmostEqual(sentence_bleu(references, hypothesis, weights), 0.0, places=4)

    def test_empty_hypothesis(self):
        if False:
            return 10
        references = ['The candidate has no alignment to any of the references'.split()]
        hypothesis = []
        assert sentence_bleu(references, hypothesis) == 0

    def test_length_one_hypothesis(self):
        if False:
            while True:
                i = 10
        references = ['The candidate has no alignment to any of the references'.split()]
        hypothesis = ['Foo']
        method4 = SmoothingFunction().method4
        try:
            sentence_bleu(references, hypothesis, smoothing_function=method4)
        except ValueError:
            pass

    def test_empty_references(self):
        if False:
            print('Hello World!')
        references = [[]]
        hypothesis = 'John loves Mary'.split()
        assert sentence_bleu(references, hypothesis) == 0

    def test_empty_references_and_hypothesis(self):
        if False:
            while True:
                i = 10
        references = [[]]
        hypothesis = []
        assert sentence_bleu(references, hypothesis) == 0

    def test_reference_or_hypothesis_shorter_than_fourgrams(self):
        if False:
            print('Hello World!')
        references = ['let it go'.split()]
        hypothesis = 'let go it'.split()
        self.assertAlmostEqual(sentence_bleu(references, hypothesis), 0.0, places=4)
        try:
            self.assertWarns(UserWarning, sentence_bleu, references, hypothesis)
        except AttributeError:
            pass

    def test_numpy_weights(self):
        if False:
            print('Hello World!')
        references = ['The candidate has no alignment to any of the references'.split()]
        hypothesis = 'John loves Mary'.split()
        weights = np.array([0.25] * 4)
        assert sentence_bleu(references, hypothesis, weights) == 0

class TestBLEUvsMteval13a(unittest.TestCase):

    def test_corpus_bleu(self):
        if False:
            while True:
                i = 10
        ref_file = find('models/wmt15_eval/ref.ru')
        hyp_file = find('models/wmt15_eval/google.ru')
        mteval_output_file = find('models/wmt15_eval/mteval-13a.output')
        with open(mteval_output_file) as mteval_fin:
            mteval_bleu_scores = map(float, mteval_fin.readlines()[-2].split()[1:-1])
        with open(ref_file, encoding='utf8') as ref_fin:
            with open(hyp_file, encoding='utf8') as hyp_fin:
                hypothesis = list(map(lambda x: x.split(), hyp_fin))
                references = list(map(lambda x: [x.split()], ref_fin))
                for (i, mteval_bleu) in zip(range(1, 10), mteval_bleu_scores):
                    nltk_bleu = corpus_bleu(references, hypothesis, weights=(1.0 / i,) * i)
                    assert abs(mteval_bleu - nltk_bleu) < 0.005
                chencherry = SmoothingFunction()
                for (i, mteval_bleu) in zip(range(1, 10), mteval_bleu_scores):
                    nltk_bleu = corpus_bleu(references, hypothesis, weights=(1.0 / i,) * i, smoothing_function=chencherry.method3)
                    assert abs(mteval_bleu - nltk_bleu) < 0.005

class TestBLEUWithBadSentence(unittest.TestCase):

    def test_corpus_bleu_with_bad_sentence(self):
        if False:
            for i in range(10):
                print('nop')
        hyp = 'Teo S yb , oe uNb , R , T t , , t Tue Ar saln S , , 5istsi l , 5oe R ulO sae oR R'
        ref = str('Their tasks include changing a pump on the faulty stokehold .Likewise , two species that are very similar in morphology were distinguished using genetics .')
        references = [[ref.split()]]
        hypotheses = [hyp.split()]
        try:
            with self.assertWarns(UserWarning):
                self.assertAlmostEqual(corpus_bleu(references, hypotheses), 0.0, places=4)
        except AttributeError:
            self.assertAlmostEqual(corpus_bleu(references, hypotheses), 0.0, places=4)

class TestBLEUWithMultipleWeights(unittest.TestCase):

    def test_corpus_bleu_with_multiple_weights(self):
        if False:
            print('Hello World!')
        hyp1 = ['It', 'is', 'a', 'guide', 'to', 'action', 'which', 'ensures', 'that', 'the', 'military', 'always', 'obeys', 'the', 'commands', 'of', 'the', 'party']
        ref1a = ['It', 'is', 'a', 'guide', 'to', 'action', 'that', 'ensures', 'that', 'the', 'military', 'will', 'forever', 'heed', 'Party', 'commands']
        ref1b = ['It', 'is', 'the', 'guiding', 'principle', 'which', 'guarantees', 'the', 'military', 'forces', 'always', 'being', 'under', 'the', 'command', 'of', 'the', 'Party']
        ref1c = ['It', 'is', 'the', 'practical', 'guide', 'for', 'the', 'army', 'always', 'to', 'heed', 'the', 'directions', 'of', 'the', 'party']
        hyp2 = ['he', 'read', 'the', 'book', 'because', 'he', 'was', 'interested', 'in', 'world', 'history']
        ref2a = ['he', 'was', 'interested', 'in', 'world', 'history', 'because', 'he', 'read', 'the', 'book']
        weight_1 = (1, 0, 0, 0)
        weight_2 = (0.25, 0.25, 0.25, 0.25)
        weight_3 = (0, 0, 0, 0, 1)
        bleu_scores = corpus_bleu(list_of_references=[[ref1a, ref1b, ref1c], [ref2a]], hypotheses=[hyp1, hyp2], weights=[weight_1, weight_2, weight_3])
        assert bleu_scores[0] == corpus_bleu([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2], weight_1)
        assert bleu_scores[1] == corpus_bleu([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2], weight_2)
        assert bleu_scores[2] == corpus_bleu([[ref1a, ref1b, ref1c], [ref2a]], [hyp1, hyp2], weight_3)