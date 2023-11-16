"""
Automated tests for indirect confirmation measures in the indirect_confirmation_measure module.
"""
import logging
import unittest
import numpy as np
from gensim.corpora.dictionary import Dictionary
from gensim.topic_coherence import indirect_confirmation_measure
from gensim.topic_coherence import text_analysis

class TestIndirectConfirmation(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.topics = [np.array([1, 2])]
        self.segmentation = [[(1, np.array([1, 2])), (2, np.array([1, 2]))]]
        self.gamma = 1
        self.measure = 'nlr'
        self.dictionary = Dictionary()
        self.dictionary.id2token = {1: 'fake', 2: 'tokens'}

    def test_cosine_similarity(self):
        if False:
            for i in range(10):
                print('nop')
        'Test cosine_similarity()'
        accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, self.dictionary)
        accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
        accumulator._num_docs = 5
        obtained = indirect_confirmation_measure.cosine_similarity(self.segmentation, accumulator, self.topics, self.measure, self.gamma)
        expected = (0.623 + 0.623) / 2.0
        self.assertAlmostEqual(expected, obtained[0], 4)
        (mean, std) = indirect_confirmation_measure.cosine_similarity(self.segmentation, accumulator, self.topics, self.measure, self.gamma, with_std=True)[0]
        self.assertAlmostEqual(expected, mean, 4)
        self.assertAlmostEqual(0.0, std, 1)

    def test_word2vec_similarity(self):
        if False:
            for i in range(10):
                print('nop')
        'Sanity check word2vec_similarity.'
        accumulator = text_analysis.WordVectorsAccumulator({1, 2}, self.dictionary)
        accumulator.accumulate([['fake', 'tokens'], ['tokens', 'fake']], 5)
        (mean, std) = indirect_confirmation_measure.word2vec_similarity(self.segmentation, accumulator, with_std=True)[0]
        self.assertNotEqual(0.0, mean)
        self.assertNotEqual(0.0, std)
if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()