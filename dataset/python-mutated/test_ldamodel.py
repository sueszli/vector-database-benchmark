"""
Automated tests for checking transformation algorithms (the models package).
"""
import logging
import numbers
import os
import unittest
import copy
import numpy as np
from numpy.testing import assert_allclose
from gensim.corpora import mmcorpus, Dictionary
from gensim.models import ldamodel, ldamulticore
from gensim import matutils, utils
from gensim.test import basetmtests
from gensim.test.utils import datapath, get_tmpfile, common_texts
GITHUB_ACTIONS_WINDOWS = os.environ.get('RUNNER_OS') == 'Windows'
dictionary = Dictionary(common_texts)
corpus = [dictionary.doc2bow(text) for text in common_texts]

def test_random_state():
    if False:
        print('Hello World!')
    testcases = [np.random.seed(0), None, np.random.RandomState(0), 0]
    for testcase in testcases:
        assert isinstance(utils.get_random_state(testcase), np.random.RandomState)

class TestLdaModel(unittest.TestCase, basetmtests.TestBaseTopicModel):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamodel.LdaModel
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def test_sync_state(self):
        if False:
            i = 10
            return i + 15
        model2 = self.class_(corpus=self.corpus, id2word=dictionary, num_topics=2, passes=1)
        model2.state = copy.deepcopy(self.model.state)
        model2.sync_state()
        assert_allclose(self.model.get_term_topics(2), model2.get_term_topics(2), rtol=1e-05)
        assert_allclose(self.model.get_topics(), model2.get_topics(), rtol=1e-05)
        self.model.random_state = np.random.RandomState(0)
        model2.random_state = np.random.RandomState(0)
        self.model.passes = 1
        model2.passes = 1
        self.model.update(self.corpus)
        model2.update(self.corpus)
        assert_allclose(self.model.get_term_topics(2), model2.get_term_topics(2), rtol=1e-05)
        assert_allclose(self.model.get_topics(), model2.get_topics(), rtol=1e-05)

    def test_transform(self):
        if False:
            print('Hello World!')
        passed = False
        for i in range(25):
            model = self.class_(id2word=dictionary, num_topics=2, passes=100)
            model.update(self.corpus)
            doc = list(corpus)[0]
            transformed = model[doc]
            vec = matutils.sparse2full(transformed, 2)
            expected = [0.13, 0.87]
            passed = np.allclose(sorted(vec), sorted(expected), atol=0.1)
            if passed:
                break
            logging.warning('LDA failed to converge on attempt %i (got %s, expected %s)', i, sorted(vec), sorted(expected))
        self.assertTrue(passed)

    def test_alpha_auto(self):
        if False:
            i = 10
            return i + 15
        model1 = self.class_(corpus, id2word=dictionary, alpha='symmetric', passes=10)
        modelauto = self.class_(corpus, id2word=dictionary, alpha='auto', passes=10)
        self.assertFalse(all(np.equal(model1.alpha, modelauto.alpha)))

    def test_alpha(self):
        if False:
            while True:
                i = 10
        kwargs = dict(id2word=dictionary, num_topics=2, alpha=None)
        expected_shape = (2,)
        self.class_(**kwargs)
        kwargs['alpha'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        assert_allclose(model.alpha, np.array([0.5, 0.5]))
        kwargs['alpha'] = 'asymmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        assert_allclose(model.alpha, [0.630602, 0.369398], rtol=1e-05)
        kwargs['alpha'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        assert_allclose(model.alpha, np.array([0.3, 0.3]))
        kwargs['alpha'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        assert_allclose(model.alpha, np.array([3, 3]))
        kwargs['alpha'] = [0.3, 0.3]
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        assert_allclose(model.alpha, np.array([0.3, 0.3]))
        kwargs['alpha'] = np.array([0.3, 0.3])
        model = self.class_(**kwargs)
        self.assertEqual(model.alpha.shape, expected_shape)
        assert_allclose(model.alpha, np.array([0.3, 0.3]))
        kwargs['alpha'] = [0.3, 0.3, 0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['alpha'] = [[0.3], [0.3]]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['alpha'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['alpha'] = 'gensim is cool'
        self.assertRaises(ValueError, self.class_, **kwargs)

    def test_eta_auto(self):
        if False:
            print('Hello World!')
        model1 = self.class_(corpus, id2word=dictionary, eta='symmetric', passes=10)
        modelauto = self.class_(corpus, id2word=dictionary, eta='auto', passes=10)
        self.assertFalse(np.allclose(model1.eta, modelauto.eta))

    def test_eta(self):
        if False:
            print('Hello World!')
        kwargs = dict(id2word=dictionary, num_topics=2, eta=None)
        num_terms = len(dictionary)
        expected_shape = (num_terms,)
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.5] * num_terms))
        kwargs['eta'] = 'symmetric'
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.5] * num_terms))
        kwargs['eta'] = 0.3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.3] * num_terms))
        kwargs['eta'] = 3
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([3] * num_terms))
        kwargs['eta'] = [0.3] * num_terms
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.3] * num_terms))
        kwargs['eta'] = np.array([0.3] * num_terms)
        model = self.class_(**kwargs)
        self.assertEqual(model.eta.shape, expected_shape)
        assert_allclose(model.eta, np.array([0.3] * num_terms))
        testeta = np.array([[0.5] * len(dictionary)] * 2)
        kwargs['eta'] = testeta
        self.class_(**kwargs)
        kwargs['eta'] = testeta.reshape(tuple(reversed(testeta.shape)))
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['eta'] = [0.3]
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['eta'] = [0.3] * (num_terms + 1)
        self.assertRaises(AssertionError, self.class_, **kwargs)
        kwargs['eta'] = 'gensim is cool'
        self.assertRaises(ValueError, self.class_, **kwargs)
        kwargs['eta'] = 'asymmetric'
        self.assertRaises(ValueError, self.class_, **kwargs)

    def test_top_topics(self):
        if False:
            while True:
                i = 10
        top_topics = self.model.top_topics(self.corpus)
        for (topic, score) in top_topics:
            self.assertTrue(isinstance(topic, list))
            self.assertTrue(isinstance(score, float))
            for (v, k) in topic:
                self.assertTrue(isinstance(k, str))
                self.assertTrue(np.issubdtype(v, np.floating))

    def test_get_topic_terms(self):
        if False:
            for i in range(10):
                print('nop')
        topic_terms = self.model.get_topic_terms(1)
        for (k, v) in topic_terms:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(np.issubdtype(v, np.floating))

    @unittest.skipIf(GITHUB_ACTIONS_WINDOWS, 'see <https://github.com/RaRe-Technologies/gensim/pull/2836>')
    def test_get_document_topics(self):
        if False:
            print('Hello World!')
        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
        doc_topics = model.get_document_topics(self.corpus)
        for topic in doc_topics:
            self.assertTrue(isinstance(topic, list))
            for (k, v) in topic:
                self.assertTrue(isinstance(k, numbers.Integral))
                self.assertTrue(np.issubdtype(v, np.floating))
        all_topics = model.get_document_topics(self.corpus, per_word_topics=True)
        self.assertEqual(model.state.numdocs, len(corpus))
        for topic in all_topics:
            self.assertTrue(isinstance(topic, tuple))
            for (k, v) in topic[0]:
                self.assertTrue(isinstance(k, numbers.Integral))
                self.assertTrue(np.issubdtype(v, np.floating))
            for (w, topic_list) in topic[1]:
                self.assertTrue(isinstance(w, numbers.Integral))
                self.assertTrue(isinstance(topic_list, list))
            for (w, phi_values) in topic[2]:
                self.assertTrue(isinstance(w, numbers.Integral))
                self.assertTrue(isinstance(phi_values, list))
        doc_topic_count_na = 0
        word_phi_count_na = 0
        all_topics = model.get_document_topics(self.corpus, minimum_probability=0.8, minimum_phi_value=1.0, per_word_topics=True)
        self.assertEqual(model.state.numdocs, len(corpus))
        for topic in all_topics:
            self.assertTrue(isinstance(topic, tuple))
            for (k, v) in topic[0]:
                self.assertTrue(isinstance(k, numbers.Integral))
                self.assertTrue(np.issubdtype(v, np.floating))
                if len(topic[0]) != 0:
                    doc_topic_count_na += 1
            for (w, topic_list) in topic[1]:
                self.assertTrue(isinstance(w, numbers.Integral))
                self.assertTrue(isinstance(topic_list, list))
            for (w, phi_values) in topic[2]:
                self.assertTrue(isinstance(w, numbers.Integral))
                self.assertTrue(isinstance(phi_values, list))
                if len(phi_values) != 0:
                    word_phi_count_na += 1
        self.assertTrue(model.state.numdocs > doc_topic_count_na)
        self.assertTrue(sum((len(i) for i in corpus)) > word_phi_count_na)
        (doc_topics, word_topics, word_phis) = model.get_document_topics(self.corpus[1], per_word_topics=True)
        for (k, v) in doc_topics:
            self.assertTrue(isinstance(k, numbers.Integral))
            self.assertTrue(np.issubdtype(v, np.floating))
        for (w, topic_list) in word_topics:
            self.assertTrue(isinstance(w, numbers.Integral))
            self.assertTrue(isinstance(topic_list, list))
        for (w, phi_values) in word_phis:
            self.assertTrue(isinstance(w, numbers.Integral))
            self.assertTrue(isinstance(phi_values, list))

    def test_term_topics(self):
        if False:
            for i in range(10):
                print('nop')
        model = self.class_(self.corpus, id2word=dictionary, num_topics=2, passes=100, random_state=np.random.seed(0))
        result = model.get_term_topics(2)
        for (topic_no, probability) in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(np.issubdtype(probability, np.floating))
        result = model.get_term_topics(str(model.id2word[2]))
        for (topic_no, probability) in result:
            self.assertTrue(isinstance(topic_no, int))
            self.assertTrue(np.issubdtype(probability, np.floating))

    def test_passes(self):
        if False:
            while True:
                i = 10
        self.longMessage = True
        test_rhots = list()
        model = self.class_(id2word=dictionary, chunksize=1, num_topics=2)

        def final_rhot(model):
            if False:
                for i in range(10):
                    print('nop')
            return pow(model.offset + 1 * model.num_updates / model.chunksize, -model.decay)
        for x in range(5):
            model.update(self.corpus)
            test_rhots.append(final_rhot(model))
        for passes in [1, 5, 10, 50, 100]:
            model = self.class_(id2word=dictionary, chunksize=1, num_topics=2, passes=passes)
            self.assertEqual(final_rhot(model), 1.0)
            for test_rhot in test_rhots:
                model.update(self.corpus)
                msg = ', '.join((str(x) for x in [passes, model.num_updates, model.state.numdocs]))
                self.assertAlmostEqual(final_rhot(model), test_rhot, msg=msg)
            self.assertEqual(model.state.numdocs, len(corpus) * len(test_rhots))
            self.assertEqual(model.num_updates, len(corpus) * len(test_rhots))

    def test_persistence(self):
        if False:
            print('Hello World!')
        fname = get_tmpfile('gensim_models_lda.tst')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def test_model_compatibility_with_python_versions(self):
        if False:
            return 10
        fname_model_2_7 = datapath('ldamodel_python_2_7')
        model_2_7 = self.class_.load(fname_model_2_7)
        fname_model_3_5 = datapath('ldamodel_python_3_5')
        model_3_5 = self.class_.load(fname_model_3_5)
        self.assertEqual(model_2_7.num_topics, model_3_5.num_topics)
        self.assertTrue(np.allclose(model_2_7.expElogbeta, model_3_5.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model_2_7[tstvec], model_3_5[tstvec]))
        id2word_2_7 = dict(model_2_7.id2word.iteritems())
        id2word_3_5 = dict(model_3_5.id2word.iteritems())
        self.assertEqual(set(id2word_2_7.keys()), set(id2word_3_5.keys()))

    def test_persistence_ignore(self):
        if False:
            return 10
        fname = get_tmpfile('gensim_models_lda_testPersistenceIgnore.tst')
        model = ldamodel.LdaModel(self.corpus, num_topics=2)
        model.save(fname, ignore='id2word')
        model2 = ldamodel.LdaModel.load(fname)
        self.assertTrue(model2.id2word is None)
        model.save(fname, ignore=['id2word'])
        model2 = ldamodel.LdaModel.load(fname)
        self.assertTrue(model2.id2word is None)

    def test_persistence_compressed(self):
        if False:
            while True:
                i = 10
        fname = get_tmpfile('gensim_models_lda.tst.gz')
        model = self.model
        model.save(fname)
        model2 = self.class_.load(fname, mmap=None)
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def test_large_mmap(self):
        if False:
            while True:
                i = 10
        fname = get_tmpfile('gensim_models_lda.tst')
        model = self.model
        model.save(fname, sep_limit=0)
        model2 = self.class_.load(fname, mmap='r')
        self.assertEqual(model.num_topics, model2.num_topics)
        self.assertTrue(isinstance(model2.expElogbeta, np.memmap))
        self.assertTrue(np.allclose(model.expElogbeta, model2.expElogbeta))
        tstvec = []
        self.assertTrue(np.allclose(model[tstvec], model2[tstvec]))

    def test_large_mmap_compressed(self):
        if False:
            print('Hello World!')
        fname = get_tmpfile('gensim_models_lda.tst.gz')
        model = self.model
        model.save(fname, sep_limit=0)
        self.assertRaises(IOError, self.class_.load, fname, mmap='r')

    def test_random_state_backward_compatibility(self):
        if False:
            return 10
        pre_0_13_2_fname = datapath('pre_0_13_2_model')
        model_pre_0_13_2 = self.class_.load(pre_0_13_2_fname)
        model_topics = model_pre_0_13_2.print_topics(num_topics=2, num_words=3)
        for i in model_topics:
            self.assertTrue(isinstance(i[0], int))
            self.assertTrue(isinstance(i[1], str))
        post_0_13_2_fname = get_tmpfile('gensim_models_lda_post_0_13_2_model.tst')
        model_pre_0_13_2.save(post_0_13_2_fname)
        model_post_0_13_2 = self.class_.load(post_0_13_2_fname)
        model_topics_new = model_post_0_13_2.print_topics(num_topics=2, num_words=3)
        for i in model_topics_new:
            self.assertTrue(isinstance(i[0], int))
            self.assertTrue(isinstance(i[1], str))

    def test_dtype_backward_compatibility(self):
        if False:
            i = 10
            return i + 15
        lda_3_0_1_fname = datapath('lda_3_0_1_model')
        test_doc = [(0, 1), (1, 1), (2, 1)]
        expected_topics = [(0, 0.8700588697747518), (1, 0.12994113022524822)]
        model = self.class_.load(lda_3_0_1_fname)
        topics = model[test_doc]
        self.assertTrue(np.allclose(expected_topics, topics))

class TestLdaMulticore(TestLdaModel):

    def setUp(self):
        if False:
            print('Hello World!')
        self.corpus = mmcorpus.MmCorpus(datapath('testcorpus.mm'))
        self.class_ = ldamulticore.LdaMulticore
        self.model = self.class_(corpus, id2word=dictionary, num_topics=2, passes=100)

    def test_alpha_auto(self):
        if False:
            return 10
        self.assertRaises(RuntimeError, self.class_, alpha='auto')
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()