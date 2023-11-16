"""
Automated tests for checking transformation algorithms (the models package).
"""
import logging
import unittest
import numpy as np
from gensim.corpora.mmcorpus import MmCorpus
from gensim.models import tfidfmodel
from gensim.test.utils import datapath, get_tmpfile, common_dictionary, common_corpus
from gensim.corpora import Dictionary
texts = [['complier', 'system', 'computer'], ['eulerian', 'node', 'cycle', 'graph', 'tree', 'path'], ['graph', 'flow', 'network', 'graph'], ['loading', 'computer', 'system'], ['user', 'server', 'system'], ['tree', 'hamiltonian'], ['graph', 'trees'], ['computer', 'kernel', 'malfunction', 'computer'], ['server', 'system', 'computer']]
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

class TestTfidfModel(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.corpus = MmCorpus(datapath('testcorpus.mm'))

    def test_transform(self):
        if False:
            print('Hello World!')
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        doc = list(self.corpus)[0]
        transformed = model[doc]
        expected = [(0, 0.5773502691896257), (1, 0.5773502691896257), (2, 0.5773502691896257)]
        self.assertTrue(np.allclose(transformed, expected))

    def test_init(self):
        if False:
            while True:
                i = 10
        model1 = tfidfmodel.TfidfModel(common_corpus)
        dfs = common_dictionary.dfs
        self.assertEqual(model1.dfs, dfs)
        self.assertEqual(model1.idfs, tfidfmodel.precompute_idfs(model1.wglobal, dfs, len(common_corpus)))
        model2 = tfidfmodel.TfidfModel(dictionary=common_dictionary)
        self.assertEqual(model1.idfs, model2.idfs)

    def test_persistence(self):
        if False:
            print('Hello World!')
        fname = get_tmpfile('gensim_models.tst')
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))
        fname = get_tmpfile('gensim_models_smartirs.tst')
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='nfc')
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))
        model3 = tfidfmodel.TfidfModel(self.corpus, smartirs='nfc')
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst'))
        idfs3 = [model3.idfs[key] for key in sorted(model3.idfs.keys())]
        idfs4 = [model4.idfs[key] for key in sorted(model4.idfs.keys())]
        self.assertTrue(np.allclose(idfs3, idfs4))
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))
        self.assertTrue(np.allclose(model3[[]], model4[[]]))
        fname = get_tmpfile('gensim_models_smartirs.tst')
        model = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=1)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        model3 = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=1)
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst'))
        idfs3 = [model3.idfs[key] for key in sorted(model3.idfs.keys())]
        idfs4 = [model4.idfs[key] for key in sorted(model4.idfs.keys())]
        self.assertTrue(np.allclose(idfs3, idfs4))
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))

    def test_persistence_compressed(self):
        if False:
            for i in range(10):
                print('nop')
        fname = get_tmpfile('gensim_models.tst.gz')
        model = tfidfmodel.TfidfModel(self.corpus, normalize=True)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))
        fname = get_tmpfile('gensim_models_smartirs.tst.gz')
        model = tfidfmodel.TfidfModel(self.corpus, smartirs='nfc')
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        self.assertTrue(np.allclose(model[[]], model2[[]]))
        model3 = tfidfmodel.TfidfModel(self.corpus, smartirs='nfc')
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst.bz2'))
        idfs3 = [model3.idfs[key] for key in sorted(model3.idfs.keys())]
        idfs4 = [model4.idfs[key] for key in sorted(model4.idfs.keys())]
        self.assertTrue(np.allclose(idfs3, idfs4))
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))
        self.assertTrue(np.allclose(model3[[]], model4[[]]))
        fname = get_tmpfile('gensim_models_smartirs.tst.gz')
        model = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=1)
        model.save(fname)
        model2 = tfidfmodel.TfidfModel.load(fname, mmap=None)
        self.assertTrue(model.idfs == model2.idfs)
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model[tstvec[0]], model2[tstvec[0]]))
        self.assertTrue(np.allclose(model[tstvec[1]], model2[tstvec[1]]))
        model3 = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=1)
        model4 = tfidfmodel.TfidfModel.load(datapath('tfidf_model.tst.bz2'))
        idfs3 = [model3.idfs[key] for key in sorted(model3.idfs.keys())]
        idfs4 = [model4.idfs[key] for key in sorted(model4.idfs.keys())]
        self.assertTrue(np.allclose(idfs3, idfs4))
        tstvec = [corpus[1], corpus[2]]
        self.assertTrue(np.allclose(model3[tstvec[0]], model4[tstvec[0]]))
        self.assertTrue(np.allclose(model3[tstvec[1]], model4[tstvec[1]]))

    def test_consistency(self):
        if False:
            return 10
        docs = [corpus[1], corpus[2]]
        model = tfidfmodel.TfidfModel(corpus, smartirs='nfc')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        model = tfidfmodel.TfidfModel(corpus)
        expected_docs = [model[docs[0]], model[docs[1]]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='tnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = docs[:]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='nnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = docs[:]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='lnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0)], [(5, 2.0), (9, 1.0), (10, 1.0)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='dnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0)], [(5, 2.0), (9, 1.0), (10, 1.0)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='ann')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0)], [(5, 1.0), (9, 0.75), (10, 0.75)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='bnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (8, 1)], [(5, 1), (9, 1), (10, 1)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='Lnn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 1.0), (4, 1.0), (5, 1.0), (6, 1.0), (7, 1.0), (8, 1.0)], [(5, 1.4133901052), (9, 0.7066950526), (10, 0.7066950526)]]
        model = tfidfmodel.TfidfModel(corpus, smartirs='nxn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = docs[:]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='nfn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 3.169925001442312), (4, 3.169925001442312), (5, 1.584962500721156), (6, 3.169925001442312), (7, 3.169925001442312), (8, 2.169925001442312)], [(5, 3.169925001442312), (9, 3.169925001442312), (10, 3.169925001442312)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='ntn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 3.321928094887362), (4, 3.321928094887362), (5, 1.736965594166206), (6, 3.321928094887362), (7, 3.321928094887362), (8, 2.321928094887362)], [(5, 3.473931188332412), (9, 3.321928094887362), (10, 3.321928094887362)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='npn')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 3.0), (4, 3.0), (5, 1.0), (6, 3.0), (7, 3.0), (8, 1.8073549220576042)], [(5, 2.0), (9, 3.0), (10, 3.0)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='nnx')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = docs[:]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, smartirs='nnc')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(3, 0.4082482905), (4, 0.4082482905), (5, 0.4082482905), (6, 0.4082482905), (7, 0.4082482905), (8, 0.4082482905)], [(5, 0.816496580927726), (9, 0.408248290463863), (10, 0.408248290463863)]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        model = tfidfmodel.TfidfModel(corpus, wlocal=lambda x: x, wglobal=lambda x, y: x * x, smartirs='nnc')
        transformed_docs = [model[docs[0]], model[docs[1]]]
        model = tfidfmodel.TfidfModel(corpus, wlocal=lambda x: x * x, wglobal=lambda x, y: x, smartirs='nnc')
        expected_docs = [model[docs[0]], model[docs[1]]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        slope = 0.2
        model = tfidfmodel.TfidfModel(corpus, smartirs='nnu', slope=slope)
        transformed_docs = [model[docs[0]], model[docs[1]]]
        average_unique_length = 1.0 * sum((len(set(text)) for text in texts)) / len(texts)
        vector_norms = [(1.0 - slope) * average_unique_length + slope * 6.0, (1.0 - slope) * average_unique_length + slope * 3.0]
        expected_docs = [[(termid, weight / vector_norms[0]) for (termid, weight) in docs[0]], [(termid, weight / vector_norms[1]) for (termid, weight) in docs[1]]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))
        slope = 0.2
        model = tfidfmodel.TfidfModel(dictionary=dictionary, smartirs='nnb', slope=slope)
        transformed_docs = [model[docs[0]], model[docs[1]]]
        average_character_length = sum((len(word) + 1.0 for text in texts for word in text)) / len(texts)
        vector_norms = [(1.0 - slope) * average_character_length + slope * 36.0, (1.0 - slope) * average_character_length + slope * 25.0]
        expected_docs = [[(termid, weight / vector_norms[0]) for (termid, weight) in docs[0]], [(termid, weight / vector_norms[1]) for (termid, weight) in docs[1]]]
        self.assertTrue(np.allclose(transformed_docs[0], expected_docs[0]))
        self.assertTrue(np.allclose(transformed_docs[1], expected_docs[1]))

    def test_pivoted_normalization(self):
        if False:
            return 10
        docs = [corpus[1], corpus[2]]
        model = tfidfmodel.TfidfModel(self.corpus)
        transformed_docs = [model[docs[0]], model[docs[1]]]
        model = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=1)
        expected_docs = [model[docs[0]], model[docs[1]]]
        self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
        self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))
        model = tfidfmodel.TfidfModel(self.corpus, pivot=0, slope=0.5)
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(8, 0.8884910505493495), (7, 0.648974041227711), (6, 0.8884910505493495), (5, 0.648974041227711), (4, 0.8884910505493495), (3, 0.8884910505493495)], [(10, 0.8164965809277263), (9, 0.8164965809277263), (5, 1.6329931618554525)]]
        self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
        self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))

    def test_wlocal_wglobal(self):
        if False:
            return 10

        def wlocal(tf):
            if False:
                for i in range(10):
                    print('nop')
            assert isinstance(tf, np.ndarray)
            return iter(tf + 1)

        def wglobal(df, total_docs):
            if False:
                while True:
                    i = 10
            return 1
        docs = [corpus[1], corpus[2]]
        model = tfidfmodel.TfidfModel(corpus, wlocal=wlocal, wglobal=wglobal, normalize=False)
        transformed_docs = [model[docs[0]], model[docs[1]]]
        expected_docs = [[(termid, weight + 1) for (termid, weight) in docs[0]], [(termid, weight + 1) for (termid, weight) in docs[1]]]
        self.assertTrue(np.allclose(sorted(transformed_docs[0]), sorted(expected_docs[0])))
        self.assertTrue(np.allclose(sorted(transformed_docs[1]), sorted(expected_docs[1])))

    def test_backwards_compatibility(self):
        if False:
            while True:
                i = 10
        model = tfidfmodel.TfidfModel.load(datapath('tfidf_model_3_2.tst'))
        attrs = ['pivot', 'slope', 'smartirs']
        for a in attrs:
            self.assertTrue(hasattr(model, a))
        self.assertEqual(len(model[corpus]), len(corpus))
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()