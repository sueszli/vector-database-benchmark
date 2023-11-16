import logging
import unittest
import numpy as np
from gensim.models import LdaModel
from gensim.test.utils import common_dictionary, common_corpus

class TestLdaDiff(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.dictionary = common_dictionary
        self.corpus = common_corpus
        self.num_topics = 5
        self.n_ann_terms = 10
        self.model = LdaModel(corpus=self.corpus, id2word=self.dictionary, num_topics=self.num_topics, passes=10)

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        (mdiff, annotation) = self.model.diff(self.model, n_ann_terms=self.n_ann_terms)
        self.assertEqual(mdiff.shape, (self.num_topics, self.num_topics))
        self.assertEqual(len(annotation), self.num_topics)
        self.assertEqual(len(annotation[0]), self.num_topics)
        (mdiff, annotation) = self.model.diff(self.model, n_ann_terms=self.n_ann_terms, diagonal=True)
        self.assertEqual(mdiff.shape, (self.num_topics,))
        self.assertEqual(len(annotation), self.num_topics)

    def test_identity(self):
        if False:
            for i in range(10):
                print('nop')
        for dist_name in ['hellinger', 'kullback_leibler', 'jaccard']:
            (mdiff, annotation) = self.model.diff(self.model, n_ann_terms=self.n_ann_terms, distance=dist_name)
            for row in annotation:
                for (int_tokens, diff_tokens) in row:
                    self.assertEqual(diff_tokens, [])
                    self.assertEqual(len(int_tokens), self.n_ann_terms)
            self.assertTrue(np.allclose(np.diag(mdiff), np.zeros(mdiff.shape[0], dtype=mdiff.dtype)))
            if dist_name == 'jaccard':
                self.assertTrue(np.allclose(mdiff, np.zeros(mdiff.shape, dtype=mdiff.dtype)))
            (mdiff, annotation) = self.model.diff(self.model, n_ann_terms=self.n_ann_terms, distance=dist_name, diagonal=True)
            for (int_tokens, diff_tokens) in annotation:
                self.assertEqual(diff_tokens, [])
                self.assertEqual(len(int_tokens), self.n_ann_terms)
            self.assertTrue(np.allclose(mdiff, np.zeros(mdiff.shape, dtype=mdiff.dtype)))
            if dist_name == 'jaccard':
                self.assertTrue(np.allclose(mdiff, np.zeros(mdiff.shape, dtype=mdiff.dtype)))

    def test_input(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(ValueError, self.model.diff, self.model, n_ann_terms=self.n_ann_terms, distance='something')
        self.assertRaises(ValueError, self.model.diff, [], n_ann_terms=self.n_ann_terms, distance='something')
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()