"""
Automated tests for checking transformation algorithms (the models package).
"""
import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import rpmodel
from gensim import matutils
from gensim.test.utils import datapath, get_tmpfile

class TestRpModel(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.corpus = MmCorpus(datapath('testcorpus.mm'))

    def test_transform(self):
        if False:
            return 10
        np.random.seed(13)
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2)
        expected = np.array([-0.70710677, 0.70710677])
        self.assertTrue(np.allclose(vec, expected))

    def test_persistence(self):
        if False:
            return 10
        fname = get_tmpfile('gensim_models.tst')
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        model.save(fname)
        model2 = rpmodel.RpModel.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def test_persistence_compressed(self):
        if False:
            return 10
        fname = get_tmpfile('gensim_models.tst.gz')
        model = rpmodel.RpModel(self.corpus, num_topics=2)
        model.save(fname)
        model2 = rpmodel.RpModel.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection, model2.projection))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()