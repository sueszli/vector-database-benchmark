from collections import defaultdict
import math
import unittest
from gensim.models.bm25model import BM25ABC
from gensim.models import OkapiBM25Model, LuceneBM25Model, AtireBM25Model
from gensim.corpora import Dictionary

class BM25Stub(BM25ABC):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super().__init__(*args, **kwargs)

    def precompute_idfs(self, dfs, num_docs):
        if False:
            print('Hello World!')
        return dict()

    def get_term_weights(self, num_tokens, term_frequencies, idfs):
        if False:
            i = 10
            return i + 15
        return term_frequencies

class BM25ABCTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.documents = [['cat', 'dog', 'mouse'], ['cat', 'lion'], ['cat', 'lion']]
        self.dictionary = Dictionary(self.documents)
        self.expected_avgdl = sum(map(len, self.documents)) / len(self.documents)

    def test_avgdl_from_corpus(self):
        if False:
            i = 10
            return i + 15
        corpus = list(map(self.dictionary.doc2bow, self.documents))
        model = BM25Stub(corpus=corpus)
        actual_avgdl = model.avgdl
        self.assertAlmostEqual(self.expected_avgdl, actual_avgdl)

    def test_avgdl_from_dictionary(self):
        if False:
            print('Hello World!')
        model = BM25Stub(dictionary=self.dictionary)
        actual_avgdl = model.avgdl
        self.assertAlmostEqual(self.expected_avgdl, actual_avgdl)

class OkapiBM25ModelTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.documents = [['cat', 'dog', 'mouse'], ['cat', 'lion'], ['cat', 'lion']]
        self.dictionary = Dictionary(self.documents)
        (self.k1, self.b, self.epsilon) = (1.5, 0.75, 0.25)

        def get_idf(word):
            if False:
                while True:
                    i = 10
            frequency = sum(map(lambda document: word in document, self.documents))
            return math.log((len(self.documents) - frequency + 0.5) / (frequency + 0.5))
        dog_idf = get_idf('dog')
        cat_idf = get_idf('cat')
        mouse_idf = get_idf('mouse')
        lion_idf = get_idf('lion')
        average_idf = (dog_idf + cat_idf + mouse_idf + lion_idf) / len(self.dictionary)
        eps = self.epsilon * average_idf
        self.expected_dog_idf = dog_idf if dog_idf > 0 else eps
        self.expected_cat_idf = cat_idf if cat_idf > 0 else eps
        self.expected_mouse_idf = mouse_idf if mouse_idf > 0 else eps
        self.expected_lion_idf = lion_idf if lion_idf > 0 else eps

    def test_idfs_from_corpus(self):
        if False:
            print('Hello World!')
        corpus = list(map(self.dictionary.doc2bow, self.documents))
        model = OkapiBM25Model(corpus=corpus, k1=self.k1, b=self.b, epsilon=self.epsilon)
        actual_dog_idf = model.idfs[self.dictionary.token2id['dog']]
        actual_cat_idf = model.idfs[self.dictionary.token2id['cat']]
        actual_mouse_idf = model.idfs[self.dictionary.token2id['mouse']]
        actual_lion_idf = model.idfs[self.dictionary.token2id['lion']]
        self.assertAlmostEqual(self.expected_dog_idf, actual_dog_idf)
        self.assertAlmostEqual(self.expected_cat_idf, actual_cat_idf)
        self.assertAlmostEqual(self.expected_mouse_idf, actual_mouse_idf)
        self.assertAlmostEqual(self.expected_lion_idf, actual_lion_idf)

    def test_idfs_from_dictionary(self):
        if False:
            print('Hello World!')
        model = OkapiBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b, epsilon=self.epsilon)
        actual_dog_idf = model.idfs[self.dictionary.token2id['dog']]
        actual_cat_idf = model.idfs[self.dictionary.token2id['cat']]
        actual_mouse_idf = model.idfs[self.dictionary.token2id['mouse']]
        actual_lion_idf = model.idfs[self.dictionary.token2id['lion']]
        self.assertAlmostEqual(self.expected_dog_idf, actual_dog_idf)
        self.assertAlmostEqual(self.expected_cat_idf, actual_cat_idf)
        self.assertAlmostEqual(self.expected_mouse_idf, actual_mouse_idf)
        self.assertAlmostEqual(self.expected_lion_idf, actual_lion_idf)

    def test_score(self):
        if False:
            i = 10
            return i + 15
        model = OkapiBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b, epsilon=self.epsilon)
        first_document = self.documents[0]
        first_bow = self.dictionary.doc2bow(first_document)
        weights = defaultdict(lambda : 0.0)
        weights.update(model[first_bow])
        actual_dog_weight = weights[self.dictionary.token2id['dog']]
        actual_cat_weight = weights[self.dictionary.token2id['cat']]
        actual_mouse_weight = weights[self.dictionary.token2id['mouse']]
        actual_lion_weight = weights[self.dictionary.token2id['lion']]

        def get_expected_weight(word):
            if False:
                i = 10
                return i + 15
            idf = model.idfs[self.dictionary.token2id[word]]
            numerator = self.k1 + 1
            denominator = 1 + self.k1 * (1 - self.b + self.b * len(first_document) / model.avgdl)
            return idf * numerator / denominator
        expected_dog_weight = get_expected_weight('dog') if 'dog' in first_document else 0.0
        expected_cat_weight = get_expected_weight('cat') if 'cat' in first_document else 0.0
        expected_mouse_weight = get_expected_weight('mouse') if 'mouse' in first_document else 0.0
        expected_lion_weight = get_expected_weight('lion') if 'lion' in first_document else 0.0
        self.assertAlmostEqual(expected_dog_weight, actual_dog_weight)
        self.assertAlmostEqual(expected_cat_weight, actual_cat_weight)
        self.assertAlmostEqual(expected_mouse_weight, actual_mouse_weight)
        self.assertAlmostEqual(expected_lion_weight, actual_lion_weight)

class LuceneBM25ModelTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.documents = [['cat', 'dog', 'mouse'], ['cat', 'lion'], ['cat', 'lion']]
        self.dictionary = Dictionary(self.documents)
        (self.k1, self.b) = (1.5, 0.75)

        def get_idf(word):
            if False:
                print('Hello World!')
            frequency = sum(map(lambda document: word in document, self.documents))
            return math.log(1.0 + (len(self.documents) - frequency + 0.5) / (frequency + 0.5))
        self.expected_dog_idf = get_idf('dog')
        self.expected_cat_idf = get_idf('cat')
        self.expected_mouse_idf = get_idf('mouse')
        self.expected_lion_idf = get_idf('lion')

    def test_idfs_from_corpus(self):
        if False:
            return 10
        corpus = list(map(self.dictionary.doc2bow, self.documents))
        model = LuceneBM25Model(corpus=corpus, k1=self.k1, b=self.b)
        actual_dog_idf = model.idfs[self.dictionary.token2id['dog']]
        actual_cat_idf = model.idfs[self.dictionary.token2id['cat']]
        actual_mouse_idf = model.idfs[self.dictionary.token2id['mouse']]
        actual_lion_idf = model.idfs[self.dictionary.token2id['lion']]
        self.assertAlmostEqual(self.expected_dog_idf, actual_dog_idf)
        self.assertAlmostEqual(self.expected_cat_idf, actual_cat_idf)
        self.assertAlmostEqual(self.expected_mouse_idf, actual_mouse_idf)
        self.assertAlmostEqual(self.expected_lion_idf, actual_lion_idf)

    def test_idfs_from_dictionary(self):
        if False:
            return 10
        model = LuceneBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b)
        actual_dog_idf = model.idfs[self.dictionary.token2id['dog']]
        actual_cat_idf = model.idfs[self.dictionary.token2id['cat']]
        actual_mouse_idf = model.idfs[self.dictionary.token2id['mouse']]
        actual_lion_idf = model.idfs[self.dictionary.token2id['lion']]
        self.assertAlmostEqual(self.expected_dog_idf, actual_dog_idf)
        self.assertAlmostEqual(self.expected_cat_idf, actual_cat_idf)
        self.assertAlmostEqual(self.expected_mouse_idf, actual_mouse_idf)
        self.assertAlmostEqual(self.expected_lion_idf, actual_lion_idf)

    def test_score(self):
        if False:
            for i in range(10):
                print('nop')
        model = LuceneBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b)
        first_document = self.documents[0]
        first_bow = self.dictionary.doc2bow(first_document)
        weights = defaultdict(lambda : 0.0)
        weights.update(model[first_bow])
        actual_dog_weight = weights[self.dictionary.token2id['dog']]
        actual_cat_weight = weights[self.dictionary.token2id['cat']]
        actual_mouse_weight = weights[self.dictionary.token2id['mouse']]
        actual_lion_weight = weights[self.dictionary.token2id['lion']]

        def get_expected_weight(word):
            if False:
                for i in range(10):
                    print('nop')
            idf = model.idfs[self.dictionary.token2id[word]]
            denominator = 1 + self.k1 * (1 - self.b + self.b * len(first_document) / model.avgdl)
            return idf / denominator
        expected_dog_weight = get_expected_weight('dog') if 'dog' in first_document else 0.0
        expected_cat_weight = get_expected_weight('cat') if 'cat' in first_document else 0.0
        expected_mouse_weight = get_expected_weight('mouse') if 'mouse' in first_document else 0.0
        expected_lion_weight = get_expected_weight('lion') if 'lion' in first_document else 0.0
        self.assertAlmostEqual(expected_dog_weight, actual_dog_weight)
        self.assertAlmostEqual(expected_cat_weight, actual_cat_weight)
        self.assertAlmostEqual(expected_mouse_weight, actual_mouse_weight)
        self.assertAlmostEqual(expected_lion_weight, actual_lion_weight)

class AtireBM25ModelTest(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.documents = [['cat', 'dog', 'mouse'], ['cat', 'lion'], ['cat', 'lion']]
        self.dictionary = Dictionary(self.documents)
        (self.k1, self.b, self.epsilon) = (1.5, 0.75, 0.25)

        def get_idf(word):
            if False:
                while True:
                    i = 10
            frequency = sum(map(lambda document: word in document, self.documents))
            return math.log(len(self.documents) / frequency)
        self.expected_dog_idf = get_idf('dog')
        self.expected_cat_idf = get_idf('cat')
        self.expected_mouse_idf = get_idf('mouse')
        self.expected_lion_idf = get_idf('lion')

    def test_idfs_from_corpus(self):
        if False:
            print('Hello World!')
        corpus = list(map(self.dictionary.doc2bow, self.documents))
        model = AtireBM25Model(corpus=corpus, k1=self.k1, b=self.b)
        actual_dog_idf = model.idfs[self.dictionary.token2id['dog']]
        actual_cat_idf = model.idfs[self.dictionary.token2id['cat']]
        actual_mouse_idf = model.idfs[self.dictionary.token2id['mouse']]
        actual_lion_idf = model.idfs[self.dictionary.token2id['lion']]
        self.assertAlmostEqual(self.expected_dog_idf, actual_dog_idf)
        self.assertAlmostEqual(self.expected_cat_idf, actual_cat_idf)
        self.assertAlmostEqual(self.expected_mouse_idf, actual_mouse_idf)
        self.assertAlmostEqual(self.expected_lion_idf, actual_lion_idf)

    def test_idfs_from_dictionary(self):
        if False:
            return 10
        model = AtireBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b)
        actual_dog_idf = model.idfs[self.dictionary.token2id['dog']]
        actual_cat_idf = model.idfs[self.dictionary.token2id['cat']]
        actual_mouse_idf = model.idfs[self.dictionary.token2id['mouse']]
        actual_lion_idf = model.idfs[self.dictionary.token2id['lion']]
        self.assertAlmostEqual(self.expected_dog_idf, actual_dog_idf)
        self.assertAlmostEqual(self.expected_cat_idf, actual_cat_idf)
        self.assertAlmostEqual(self.expected_mouse_idf, actual_mouse_idf)
        self.assertAlmostEqual(self.expected_lion_idf, actual_lion_idf)

    def test_score(self):
        if False:
            i = 10
            return i + 15
        model = AtireBM25Model(dictionary=self.dictionary, k1=self.k1, b=self.b)
        first_document = self.documents[0]
        first_bow = self.dictionary.doc2bow(first_document)
        weights = defaultdict(lambda : 0.0)
        weights.update(model[first_bow])
        actual_dog_weight = weights[self.dictionary.token2id['dog']]
        actual_cat_weight = weights[self.dictionary.token2id['cat']]
        actual_mouse_weight = weights[self.dictionary.token2id['mouse']]
        actual_lion_weight = weights[self.dictionary.token2id['lion']]

        def get_expected_weight(word):
            if False:
                while True:
                    i = 10
            idf = model.idfs[self.dictionary.token2id[word]]
            numerator = self.k1 + 1
            denominator = 1 + self.k1 * (1 - self.b + self.b * len(first_document) / model.avgdl)
            return idf * numerator / denominator
        expected_dog_weight = get_expected_weight('dog') if 'dog' in first_document else 0.0
        expected_cat_weight = get_expected_weight('cat') if 'cat' in first_document else 0.0
        expected_mouse_weight = get_expected_weight('mouse') if 'mouse' in first_document else 0.0
        expected_lion_weight = get_expected_weight('lion') if 'lion' in first_document else 0.0
        self.assertAlmostEqual(expected_dog_weight, actual_dog_weight)
        self.assertAlmostEqual(expected_cat_weight, actual_cat_weight)
        self.assertAlmostEqual(expected_mouse_weight, actual_mouse_weight)
        self.assertAlmostEqual(expected_lion_weight, actual_lion_weight)