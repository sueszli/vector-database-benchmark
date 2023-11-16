"""
Automated tests for checking the EnsembleLda Class
"""
import os
import logging
import unittest
import numpy as np
from copy import deepcopy
import pytest
from gensim.models import EnsembleLda, LdaMulticore, LdaModel
from gensim.test.utils import datapath, get_tmpfile, common_corpus, common_dictionary
NUM_TOPICS = 2
NUM_MODELS = 4
PASSES = 50
RANDOM_STATE = 0
RTOL = 0.0001 if os.name == 'nt' else 1e-05

class TestEnsembleLda(unittest.TestCase):

    def get_elda(self):
        if False:
            return 10
        return EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS, random_state=RANDOM_STATE, topic_model_class=LdaModel)

    def get_elda_mem_unfriendly(self):
        if False:
            for i in range(10):
                print('nop')
        return EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS, random_state=RANDOM_STATE, memory_friendly_ttda=False, topic_model_class=LdaModel)

    def assert_ttda_is_valid(self, elda):
        if False:
            for i in range(10):
                print('nop')
        'Check that ttda has one or more topic and that term probabilities add to one.'
        assert len(elda.ttda) > 0
        sum_over_terms = elda.ttda.sum(axis=1)
        expected_sum_over_terms = np.ones(len(elda.ttda)).astype(np.float32)
        np.testing.assert_allclose(sum_over_terms, expected_sum_over_terms, rtol=0.0001)

    def test_elda(self):
        if False:
            for i in range(10):
                print('nop')
        elda = self.get_elda()
        assert elda.stable_topics.shape[1] == len(common_dictionary)
        assert len(elda.ttda) == NUM_MODELS * NUM_TOPICS
        self.assert_ttda_is_valid(elda)

    def test_backwards_compatibility_with_persisted_model(self):
        if False:
            while True:
                i = 10
        elda = self.get_elda()
        loaded_elda = EnsembleLda.load(datapath('ensemblelda'))
        np.testing.assert_allclose(elda.ttda, loaded_elda.ttda, rtol=RTOL)
        atol = loaded_elda.asymmetric_distance_matrix.max() * 1e-05
        np.testing.assert_allclose(elda.asymmetric_distance_matrix, loaded_elda.asymmetric_distance_matrix, atol=atol)

    def test_recluster(self):
        if False:
            print('Hello World!')
        elda = EnsembleLda.load(datapath('ensemblelda'))
        loaded_cluster_model_results = deepcopy(elda.cluster_model.results)
        loaded_valid_clusters = deepcopy(elda.valid_clusters)
        loaded_stable_topics = deepcopy(elda.get_topics())
        elda.asymmetric_distance_matrix_outdated = True
        elda.recluster()
        self.assert_clustering_results_equal(elda.cluster_model.results, loaded_cluster_model_results)
        assert elda.valid_clusters == loaded_valid_clusters
        np.testing.assert_allclose(elda.get_topics(), loaded_stable_topics, rtol=RTOL)

    def test_recluster_does_nothing_when_stable_topics_already_found(self):
        if False:
            for i in range(10):
                print('nop')
        elda = self.get_elda()
        elda.recluster()
        assert elda.stable_topics.shape[1] == len(common_dictionary)
        assert len(elda.ttda) == NUM_MODELS * NUM_TOPICS
        self.assert_ttda_is_valid(elda)

    def test_not_trained_given_zero_passes(self):
        if False:
            i = 10
            return i + 15
        elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS, passes=0, num_models=NUM_MODELS, random_state=RANDOM_STATE)
        assert len(elda.ttda) == 0

    def test_not_trained_given_no_corpus(self):
        if False:
            i = 10
            return i + 15
        elda = EnsembleLda(id2word=common_dictionary, num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS, random_state=RANDOM_STATE)
        assert len(elda.ttda) == 0

    def test_not_trained_given_zero_iterations(self):
        if False:
            while True:
                i = 10
        elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS, iterations=0, num_models=NUM_MODELS, random_state=RANDOM_STATE)
        assert len(elda.ttda) == 0

    def test_not_trained_given_zero_models(self):
        if False:
            return 10
        elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=NUM_TOPICS, passes=PASSES, num_models=0, random_state=RANDOM_STATE)
        assert len(elda.ttda) == 0

    def test_mem_unfriendly(self):
        if False:
            for i in range(10):
                print('nop')
        elda = self.get_elda()
        elda_mem_unfriendly = self.get_elda_mem_unfriendly()
        assert len(elda_mem_unfriendly.tms) == NUM_MODELS
        np.testing.assert_allclose(elda.ttda, elda_mem_unfriendly.ttda, rtol=RTOL)
        np.testing.assert_allclose(elda.get_topics(), elda_mem_unfriendly.get_topics(), rtol=RTOL)
        self.assert_ttda_is_valid(elda_mem_unfriendly)

    def test_generate_gensim_representation(self):
        if False:
            return 10
        elda = self.get_elda()
        gensim_model = elda.generate_gensim_representation()
        topics = gensim_model.get_topics()
        np.testing.assert_allclose(elda.get_topics(), topics, rtol=RTOL)

    def assert_clustering_results_equal(self, clustering_results_1, clustering_results_2):
        if False:
            i = 10
            return i + 15
        'Assert important attributes of the cluster results'
        np.testing.assert_array_equal([element.label for element in clustering_results_1], [element.label for element in clustering_results_2])
        np.testing.assert_array_equal([element.is_core for element in clustering_results_1], [element.is_core for element in clustering_results_2])

    def test_persisting(self):
        if False:
            for i in range(10):
                print('nop')
        elda = self.get_elda()
        elda_mem_unfriendly = self.get_elda_mem_unfriendly()
        fname = get_tmpfile('gensim_models_ensemblelda')
        elda.save(fname)
        loaded_elda = EnsembleLda.load(fname)
        elda_mem_unfriendly.save(fname)
        loaded_elda_mem_unfriendly = EnsembleLda.load(fname)
        assert loaded_elda.topic_model_class is None
        loaded_elda_representation = loaded_elda.generate_gensim_representation()
        assert loaded_elda.topic_model_class == LdaModel
        topics = loaded_elda_representation.get_topics()
        ttda = loaded_elda.ttda
        amatrix = loaded_elda.asymmetric_distance_matrix
        np.testing.assert_allclose(elda.get_topics(), topics, rtol=RTOL)
        np.testing.assert_allclose(elda.ttda, ttda, rtol=RTOL)
        np.testing.assert_allclose(elda.asymmetric_distance_matrix, amatrix, rtol=RTOL)
        expected_clustering_results = elda.cluster_model.results
        loaded_clustering_results = loaded_elda.cluster_model.results
        self.assert_clustering_results_equal(expected_clustering_results, loaded_clustering_results)
        loaded_elda_mem_unfriendly_representation = loaded_elda_mem_unfriendly.generate_gensim_representation()
        topics = loaded_elda_mem_unfriendly_representation.get_topics()
        np.testing.assert_allclose(elda.get_topics(), topics, rtol=RTOL)

    def test_multiprocessing(self):
        if False:
            print('Hello World!')
        random_state = RANDOM_STATE
        workers = 3
        elda = self.get_elda()
        elda_multiprocessing = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, topic_model_class=LdaModel, num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS, random_state=random_state, ensemble_workers=workers, distance_workers=workers)
        elda_mem_unfriendly = self.get_elda_mem_unfriendly()
        elda_multiprocessing_mem_unfriendly = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, topic_model_class=LdaModel, num_topics=NUM_TOPICS, passes=PASSES, num_models=NUM_MODELS, random_state=random_state, ensemble_workers=workers, distance_workers=workers, memory_friendly_ttda=False)
        np.testing.assert_allclose(elda.get_topics(), elda_multiprocessing.get_topics(), rtol=RTOL)
        np.testing.assert_allclose(elda_mem_unfriendly.get_topics(), elda_multiprocessing_mem_unfriendly.get_topics(), rtol=RTOL)

    def test_add_models_to_empty(self):
        if False:
            return 10
        elda = self.get_elda()
        ensemble = EnsembleLda(id2word=common_dictionary, num_models=0)
        ensemble.add_model(elda.ttda[0:1])
        ensemble.add_model(elda.ttda[1:])
        ensemble.recluster()
        np.testing.assert_allclose(ensemble.get_topics(), elda.get_topics(), rtol=RTOL)
        fname = get_tmpfile('gensim_models_ensemblelda')
        ensemble.save(fname)
        loaded_ensemble = EnsembleLda.load(fname)
        np.testing.assert_allclose(loaded_ensemble.get_topics(), elda.get_topics(), rtol=RTOL)
        self.test_inference(loaded_ensemble)

    def test_add_models(self):
        if False:
            return 10
        num_new_models = 3
        num_new_topics = 3
        base_elda = self.get_elda()
        cumulative_elda = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=1, num_models=num_new_models, iterations=1, random_state=RANDOM_STATE, topic_model_class=LdaMulticore, workers=3, ensemble_workers=2)
        num_topics_before_add_model = len(cumulative_elda.ttda)
        num_models_before_add_model = cumulative_elda.num_models
        cumulative_elda.add_model(base_elda.ttda)
        assert len(cumulative_elda.ttda) == num_topics_before_add_model + len(base_elda.ttda)
        assert cumulative_elda.num_models == num_models_before_add_model + 1
        num_topics_before_add_model = len(cumulative_elda.ttda)
        num_models_before_add_model = cumulative_elda.num_models
        cumulative_elda.add_model(base_elda, 5)
        assert len(cumulative_elda.ttda) == num_topics_before_add_model + len(base_elda.ttda)
        assert cumulative_elda.num_models == num_models_before_add_model + 5
        num_topics_before_add_model = len(cumulative_elda.ttda)
        num_models_before_add_model = cumulative_elda.num_models
        base_elda_mem_unfriendly = self.get_elda_mem_unfriendly()
        cumulative_elda.add_model([base_elda, base_elda_mem_unfriendly])
        assert len(cumulative_elda.ttda) == num_topics_before_add_model + 2 * len(base_elda.ttda)
        assert cumulative_elda.num_models == num_models_before_add_model + 2 * NUM_MODELS
        model = base_elda.classic_model_representation
        num_topics_before_add_model = len(cumulative_elda.ttda)
        num_models_before_add_model = cumulative_elda.num_models
        cumulative_elda.add_model(model)
        assert len(cumulative_elda.ttda) == num_topics_before_add_model + len(model.get_topics())
        assert cumulative_elda.num_models == num_models_before_add_model + 1
        num_topics_before_add_model = len(cumulative_elda.ttda)
        num_models_before_add_model = cumulative_elda.num_models
        cumulative_elda.add_model([model, model])
        assert len(cumulative_elda.ttda) == num_topics_before_add_model + 2 * len(model.get_topics())
        assert cumulative_elda.num_models == num_models_before_add_model + 2
        self.assert_ttda_is_valid(cumulative_elda)
        elda_mem_unfriendly = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=1, num_models=num_new_models, iterations=1, random_state=RANDOM_STATE, topic_model_class=LdaMulticore, workers=3, ensemble_workers=2, memory_friendly_ttda=False)
        num_topics_before_add_model = len(elda_mem_unfriendly.tms)
        num_models_before_add_model = elda_mem_unfriendly.num_models
        elda_mem_unfriendly.add_model(base_elda_mem_unfriendly)
        assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + NUM_MODELS
        assert elda_mem_unfriendly.num_models == num_models_before_add_model + NUM_MODELS
        num_topics_before_add_model = len(elda_mem_unfriendly.tms)
        num_models_before_add_model = elda_mem_unfriendly.num_models
        elda_mem_unfriendly.add_model([base_elda_mem_unfriendly, base_elda_mem_unfriendly])
        assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + 2 * NUM_MODELS
        assert elda_mem_unfriendly.num_models == num_models_before_add_model + 2 * NUM_MODELS
        num_topics_before_add_model = len(elda_mem_unfriendly.tms)
        num_models_before_add_model = elda_mem_unfriendly.num_models
        elda_mem_unfriendly.add_model(base_elda_mem_unfriendly.tms[0])
        assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + 1
        assert elda_mem_unfriendly.num_models == num_models_before_add_model + 1
        num_topics_before_add_model = len(elda_mem_unfriendly.tms)
        num_models_before_add_model = elda_mem_unfriendly.num_models
        elda_mem_unfriendly.add_model(base_elda_mem_unfriendly.tms)
        assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model + NUM_MODELS
        assert elda_mem_unfriendly.num_models == num_models_before_add_model + NUM_MODELS
        num_topics_before_add_model = len(elda_mem_unfriendly.tms)
        num_models_before_add_model = elda_mem_unfriendly.num_models
        with pytest.raises(ValueError):
            elda_mem_unfriendly.add_model(base_elda_mem_unfriendly.tms[0].get_topics())
        assert len(elda_mem_unfriendly.tms) == num_topics_before_add_model
        assert elda_mem_unfriendly.num_models == num_models_before_add_model
        assert elda_mem_unfriendly.num_models == len(elda_mem_unfriendly.tms)
        self.assert_ttda_is_valid(elda_mem_unfriendly)

    def test_add_and_recluster(self):
        if False:
            return 10
        num_new_models = 3
        num_new_topics = 3
        random_state = 1
        elda_1 = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=10, num_models=num_new_models, iterations=30, random_state=random_state, topic_model_class='lda', distance_workers=4)
        elda_mem_unfriendly_1 = EnsembleLda(corpus=common_corpus, id2word=common_dictionary, num_topics=num_new_topics, passes=10, num_models=num_new_models, iterations=30, random_state=random_state, topic_model_class=LdaModel, distance_workers=4, memory_friendly_ttda=False)
        elda_2 = self.get_elda()
        elda_mem_unfriendly_2 = self.get_elda_mem_unfriendly()
        assert elda_1.random_state != elda_2.random_state
        assert elda_mem_unfriendly_1.random_state != elda_mem_unfriendly_2.random_state
        np.testing.assert_allclose(elda_1.ttda, elda_mem_unfriendly_1.ttda, rtol=RTOL)
        np.testing.assert_allclose(elda_1.get_topics(), elda_mem_unfriendly_1.get_topics(), rtol=RTOL)
        elda_1.add_model(elda_2)
        elda_mem_unfriendly_1.add_model(elda_mem_unfriendly_2)
        np.testing.assert_allclose(elda_1.ttda, elda_mem_unfriendly_1.ttda, rtol=RTOL)
        assert len(elda_1.ttda) == len(elda_2.ttda) + num_new_models * num_new_topics
        assert len(elda_mem_unfriendly_1.ttda) == len(elda_mem_unfriendly_2.ttda) + num_new_models * num_new_topics
        assert len(elda_mem_unfriendly_1.tms) == NUM_MODELS + num_new_models
        self.assert_ttda_is_valid(elda_1)
        self.assert_ttda_is_valid(elda_mem_unfriendly_1)
        elda_1._generate_asymmetric_distance_matrix()
        elda_mem_unfriendly_1._generate_asymmetric_distance_matrix()
        np.testing.assert_allclose(elda_1.asymmetric_distance_matrix, elda_mem_unfriendly_1.asymmetric_distance_matrix)
        elda_1._generate_topic_clusters()
        elda_mem_unfriendly_1._generate_topic_clusters()
        clustering_results = elda_1.cluster_model.results
        mem_unfriendly_clustering_results = elda_mem_unfriendly_1.cluster_model.results
        self.assert_clustering_results_equal(clustering_results, mem_unfriendly_clustering_results)
        elda_1._generate_stable_topics()
        elda_mem_unfriendly_1._generate_stable_topics()
        np.testing.assert_allclose(elda_1.get_topics(), elda_mem_unfriendly_1.get_topics())
        elda_1.generate_gensim_representation()
        elda_mem_unfriendly_1.generate_gensim_representation()
        np.testing.assert_allclose(elda_1.get_topics(), elda_mem_unfriendly_1.get_topics(), rtol=RTOL)

    def test_inference(self, elda=None):
        if False:
            return 10
        if elda is None:
            elda = self.get_elda()
        max_id = np.argmax(elda.get_topics()[0, :])
        assert elda.classic_model_representation.iterations > 0
        inferred = elda[[(max_id, 1)]]
        assert inferred[0][1] - 0.3 > inferred[1][1]
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.WARN)
    unittest.main()