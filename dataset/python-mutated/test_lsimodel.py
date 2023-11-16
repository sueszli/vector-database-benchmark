"""
Automated tests for checking transformation algorithms (the models package).
"""
import logging
import unittest
import numpy as np
import scipy.linalg
from gensim import matutils
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import lsimodel
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile

class TestLsiModel(unittest.TestCase, basetmtests.TestBaseTopicModel):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.corpus = MmCorpus(datapath('testcorpus.mm'))
        self.model = lsimodel.LsiModel(self.corpus, num_topics=2)

    def test_transform(self):
        if False:
            return 10
        'Test lsi[vector] transformation.'
        model = self.model
        (u, s, vt) = scipy.linalg.svd(matutils.corpus2dense(self.corpus, self.corpus.num_terms), full_matrices=False)
        self.assertTrue(np.allclose(s[:2], model.projection.s))
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2)
        expected = np.array([-0.6594664, 0.142115444])
        self.assertTrue(np.allclose(abs(vec), abs(expected)))

    def test_transform_float32(self):
        if False:
            while True:
                i = 10
        'Test lsi[vector] transformation.'
        model = lsimodel.LsiModel(self.corpus, num_topics=2, dtype=np.float32)
        (u, s, vt) = scipy.linalg.svd(matutils.corpus2dense(self.corpus, self.corpus.num_terms), full_matrices=False)
        self.assertTrue(np.allclose(s[:2], model.projection.s))
        self.assertEqual(model.projection.u.dtype, np.float32)
        self.assertEqual(model.projection.s.dtype, np.float32)
        doc = list(self.corpus)[0]
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, 2)
        expected = np.array([-0.6594664, 0.142115444])
        self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-05))

    def test_corpus_transform(self):
        if False:
            for i in range(10):
                print('nop')
        'Test lsi[corpus] transformation.'
        model = self.model
        got = np.vstack([matutils.sparse2full(doc, 2) for doc in model[self.corpus]])
        expected = np.array([[0.65946639, 0.14211544], [2.02454305, -0.42088759], [1.54655361, 0.32358921], [1.81114125, 0.5890525], [0.9336738, -0.27138939], [0.01274618, -0.49016181], [0.04888203, -1.11294699], [0.08063836, -1.56345594], [0.27381003, -1.34694159]])
        self.assertTrue(np.allclose(abs(got), abs(expected)))

    def test_online_transform(self):
        if False:
            print('Hello World!')
        corpus = list(self.corpus)
        doc = corpus[0]
        model2 = lsimodel.LsiModel(corpus=corpus, num_topics=5)
        model = lsimodel.LsiModel(corpus=None, id2word=model2.id2word, num_topics=5)
        model.add_documents([corpus[0]])
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.num_topics)
        expected = np.array([-1.73205078, 0.0, 0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-06))
        model.add_documents(corpus[1:5], chunksize=2)
        transformed = model[doc]
        vec = matutils.sparse2full(transformed, model.num_topics)
        expected = np.array([-0.66493785, -0.28314203, -1.56376302, 0.05488682, 0.17123269])
        self.assertTrue(np.allclose(abs(vec), abs(expected), atol=1e-06))
        model.add_documents(corpus[5:])
        vec1 = matutils.sparse2full(model[doc], model.num_topics)
        vec2 = matutils.sparse2full(model2[doc], model2.num_topics)
        self.assertTrue(np.allclose(abs(vec1), abs(vec2), atol=1e-05))

    def test_persistence(self):
        if False:
            for i in range(10):
                print('nop')
        fname = get_tmpfile('gensim_models_lsi.tst')
        model = self.model
        model.save(fname)
        model2 = lsimodel.LsiModel.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(np.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def test_persistence_compressed(self):
        if False:
            return 10
        fname = get_tmpfile('gensim_models_lsi.tst.gz')
        model = self.model
        model.save(fname)
        model2 = lsimodel.LsiModel.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(np.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def test_large_mmap(self):
        if False:
            i = 10
            return i + 15
        fname = get_tmpfile('gensim_models_lsi.tst')
        model = self.model
        model.save(fname, sep_limit=0)
        model2 = lsimodel.LsiModel.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.projection.u, np.memmap))
        self.assertTrue(isinstance(model2.projection.s, np.memmap))
        self.assertTrue(np.allclose(model.projection.u, model2.projection.u))
        self.assertTrue(np.allclose(model.projection.s, model2.projection.s))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def test_large_mmap_compressed(self):
        if False:
            for i in range(10):
                print('nop')
        fname = get_tmpfile('gensim_models_lsi.tst.gz')
        model = self.model
        model.save(fname, sep_limit=0)
        return
        self.assertRaises(IOError, lsimodel.LsiModel.load, fname, mmap='r')

    def test_docs_processed(self):
        if False:
            return 10
        self.assertEqual(self.model.docs_processed, 9)
        self.assertEqual(self.model.docs_processed, self.corpus.num_docs)

    def test_get_topics(self):
        if False:
            print('Hello World!')
        topics = self.model.get_topics()
        vocab_size = len(self.model.id2word)
        for topic in topics:
            self.assertTrue(isinstance(topic, np.ndarray))
            self.assertEqual(topic.dtype, np.float64)
            self.assertEqual(vocab_size, topic.shape[0])
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()