"""
Automated tests for probability estimation algorithms in the probability_estimation module.
"""
import logging
import unittest
from gensim.corpora.dictionary import Dictionary
from gensim.corpora.hashdictionary import HashDictionary
from gensim.topic_coherence import probability_estimation

class BaseTestCases:

    class ProbabilityEstimationBase(unittest.TestCase):
        texts = [['human', 'interface', 'computer'], ['eps', 'user', 'interface', 'system'], ['system', 'human', 'system', 'eps'], ['user', 'response', 'time'], ['trees'], ['graph', 'trees']]
        dictionary = None

        def build_segmented_topics(self):
            if False:
                i = 10
                return i + 15
            token2id = self.dictionary.token2id
            computer_id = token2id['computer']
            system_id = token2id['system']
            user_id = token2id['user']
            graph_id = token2id['graph']
            self.segmented_topics = [[(system_id, graph_id), (computer_id, graph_id), (computer_id, system_id)], [(computer_id, graph_id), (user_id, graph_id), (user_id, computer_id)]]
            self.computer_id = computer_id
            self.system_id = system_id
            self.user_id = user_id
            self.graph_id = graph_id

        def setup_dictionary(self):
            if False:
                while True:
                    i = 10
            raise NotImplementedError

        def setUp(self):
            if False:
                while True:
                    i = 10
            self.setup_dictionary()
            self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
            self.build_segmented_topics()

        def test_p_boolean_document(self):
            if False:
                return 10
            'Test p_boolean_document()'
            accumulator = probability_estimation.p_boolean_document(self.corpus, self.segmented_topics)
            obtained = accumulator.index_to_dict()
            expected = {self.graph_id: {5}, self.user_id: {1, 3}, self.system_id: {1, 2}, self.computer_id: {0}}
            self.assertEqual(expected, obtained)

        def test_p_boolean_sliding_window(self):
            if False:
                return 10
            'Test p_boolean_sliding_window()'
            accumulator = probability_estimation.p_boolean_sliding_window(self.texts, self.segmented_topics, self.dictionary, 2)
            self.assertEqual(1, accumulator[self.computer_id])
            self.assertEqual(3, accumulator[self.user_id])
            self.assertEqual(1, accumulator[self.graph_id])
            self.assertEqual(4, accumulator[self.system_id])

class TestProbabilityEstimation(BaseTestCases.ProbabilityEstimationBase):

    def setup_dictionary(self):
        if False:
            while True:
                i = 10
        self.dictionary = HashDictionary(self.texts)

class TestProbabilityEstimationWithNormalDictionary(BaseTestCases.ProbabilityEstimationBase):

    def setup_dictionary(self):
        if False:
            return 10
        self.dictionary = Dictionary(self.texts)
        self.dictionary.id2token = {v: k for (k, v) in self.dictionary.token2id.items()}
if __name__ == '__main__':
    logging.root.setLevel(logging.WARNING)
    unittest.main()