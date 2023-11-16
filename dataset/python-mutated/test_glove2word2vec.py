"""Test for gensim.scripts.glove2word2vec.py."""
import logging
import unittest
import os
import sys
import numpy
import gensim
from gensim.utils import check_output
from gensim.test.utils import datapath, get_tmpfile

class TestGlove2Word2Vec(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.datapath = datapath('test_glove.txt')
        self.output_file = get_tmpfile('glove2word2vec.test')

    def test_conversion(self):
        if False:
            i = 10
            return i + 15
        check_output(args=[sys.executable, '-m', 'gensim.scripts.glove2word2vec', '--input', self.datapath, '--output', self.output_file])
        try:
            self.test_model = gensim.models.KeyedVectors.load_word2vec_format(self.output_file)
            self.assertTrue(numpy.allclose(self.test_model.n_similarity(['the', 'and'], ['and', 'the']), 1.0))
        except Exception:
            if os.path.isfile(os.path.join(self.output_file)):
                self.fail('model file %s was created but could not be loaded.' % self.output_file)
            else:
                self.fail('model file %s creation failed, check the parameters and input file format.' % self.output_file)
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()