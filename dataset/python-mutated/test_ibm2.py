"""
Tests for IBM Model 2 training methods
"""
import unittest
from collections import defaultdict
from nltk.translate import AlignedSent, IBMModel, IBMModel2
from nltk.translate.ibm_model import AlignmentInfo

class TestIBMModel2(unittest.TestCase):

    def test_set_uniform_alignment_probabilities(self):
        if False:
            for i in range(10):
                print('nop')
        corpus = [AlignedSent(['ham', 'eggs'], ['schinken', 'schinken', 'eier']), AlignedSent(['spam', 'spam', 'spam', 'spam'], ['spam', 'spam'])]
        model2 = IBMModel2(corpus, 0)
        model2.set_uniform_probabilities(corpus)
        self.assertEqual(model2.alignment_table[0][1][3][2], 1.0 / 4)
        self.assertEqual(model2.alignment_table[2][4][2][4], 1.0 / 3)

    def test_set_uniform_alignment_probabilities_of_non_domain_values(self):
        if False:
            i = 10
            return i + 15
        corpus = [AlignedSent(['ham', 'eggs'], ['schinken', 'schinken', 'eier']), AlignedSent(['spam', 'spam', 'spam', 'spam'], ['spam', 'spam'])]
        model2 = IBMModel2(corpus, 0)
        model2.set_uniform_probabilities(corpus)
        self.assertEqual(model2.alignment_table[99][1][3][2], IBMModel.MIN_PROB)
        self.assertEqual(model2.alignment_table[2][99][2][4], IBMModel.MIN_PROB)

    def test_prob_t_a_given_s(self):
        if False:
            while True:
                i = 10
        src_sentence = ['ich', 'esse', 'ja', 'gern', 'räucherschinken']
        trg_sentence = ['i', 'love', 'to', 'eat', 'smoked', 'ham']
        corpus = [AlignedSent(trg_sentence, src_sentence)]
        alignment_info = AlignmentInfo((0, 1, 4, 0, 2, 5, 5), [None] + src_sentence, ['UNUSED'] + trg_sentence, None)
        translation_table = defaultdict(lambda : defaultdict(float))
        translation_table['i']['ich'] = 0.98
        translation_table['love']['gern'] = 0.98
        translation_table['to'][None] = 0.98
        translation_table['eat']['esse'] = 0.98
        translation_table['smoked']['räucherschinken'] = 0.98
        translation_table['ham']['räucherschinken'] = 0.98
        alignment_table = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda : defaultdict(float))))
        alignment_table[0][3][5][6] = 0.97
        alignment_table[1][1][5][6] = 0.97
        alignment_table[2][4][5][6] = 0.97
        alignment_table[4][2][5][6] = 0.97
        alignment_table[5][5][5][6] = 0.96
        alignment_table[5][6][5][6] = 0.96
        model2 = IBMModel2(corpus, 0)
        model2.translation_table = translation_table
        model2.alignment_table = alignment_table
        probability = model2.prob_t_a_given_s(alignment_info)
        lexical_translation = 0.98 * 0.98 * 0.98 * 0.98 * 0.98 * 0.98
        alignment = 0.97 * 0.97 * 0.97 * 0.97 * 0.96 * 0.96
        expected_probability = lexical_translation * alignment
        self.assertEqual(round(probability, 4), round(expected_probability, 4))