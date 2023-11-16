"""
Automated tests for direct confirmation measures in the direct_confirmation_measure module.
"""
import logging
import unittest
from collections import namedtuple
from gensim.topic_coherence import direct_confirmation_measure
from gensim.topic_coherence import text_analysis

class TestDirectConfirmationMeasure(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.segmentation = [[(1, 2)]]
        self.posting_list = {1: {2, 3, 4}, 2: {3, 5}}
        self.num_docs = 5
        id2token = {1: 'test', 2: 'doc'}
        token2id = {v: k for (k, v) in id2token.items()}
        dictionary = namedtuple('Dictionary', 'token2id, id2token')(token2id, id2token)
        self.accumulator = text_analysis.InvertedIndexAccumulator({1, 2}, dictionary)
        self.accumulator._inverted_index = {0: {2, 3, 4}, 1: {3, 5}}
        self.accumulator._num_docs = self.num_docs

    def test_log_conditional_probability(self):
        if False:
            i = 10
            return i + 15
        'Test log_conditional_probability()'
        obtained = direct_confirmation_measure.log_conditional_probability(self.segmentation, self.accumulator)[0]
        expected = -0.693147181
        self.assertAlmostEqual(expected, obtained)
        (mean, std) = direct_confirmation_measure.log_conditional_probability(self.segmentation, self.accumulator, with_std=True)[0]
        self.assertAlmostEqual(expected, mean)
        self.assertEqual(0.0, std)

    def test_log_ratio_measure(self):
        if False:
            for i in range(10):
                print('nop')
        'Test log_ratio_measure()'
        obtained = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator)[0]
        expected = -0.182321557
        self.assertAlmostEqual(expected, obtained)
        (mean, std) = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator, with_std=True)[0]
        self.assertAlmostEqual(expected, mean)
        self.assertEqual(0.0, std)

    def test_normalized_log_ratio_measure(self):
        if False:
            print('Hello World!')
        'Test normalized_log_ratio_measure()'
        obtained = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator, normalize=True)[0]
        expected = -0.113282753
        self.assertAlmostEqual(expected, obtained)
        (mean, std) = direct_confirmation_measure.log_ratio_measure(self.segmentation, self.accumulator, normalize=True, with_std=True)[0]
        self.assertAlmostEqual(expected, mean)
        self.assertEqual(0.0, std)
if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()