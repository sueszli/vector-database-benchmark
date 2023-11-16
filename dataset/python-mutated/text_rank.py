from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import math
try:
    import numpy
except ImportError:
    numpy = None
from ._summarizer import AbstractSummarizer

class TextRankSummarizer(AbstractSummarizer):
    """An implementation of TextRank algorithm for summarization.

    Source: https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf
    """
    epsilon = 0.0001
    damping = 0.85
    _ZERO_DIVISION_PREVENTION = 1e-07
    _stop_words = frozenset()

    @property
    def stop_words(self):
        if False:
            return 10
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        if False:
            for i in range(10):
                print('nop')
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        if False:
            i = 10
            return i + 15
        self._ensure_dependencies_installed()
        if not document.sentences:
            return ()
        ratings = self.rate_sentences(document)
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    @staticmethod
    def _ensure_dependencies_installed():
        if False:
            for i in range(10):
                print('nop')
        if numpy is None:
            raise ValueError("LexRank summarizer requires NumPy. Please, install it by command 'pip install numpy'.")

    def rate_sentences(self, document):
        if False:
            i = 10
            return i + 15
        matrix = self._create_matrix(document)
        ranks = self.power_method(matrix, self.epsilon)
        return {sent: rank for (sent, rank) in zip(document.sentences, ranks)}

    def _create_matrix(self, document):
        if False:
            for i in range(10):
                print('nop')
        "Create a stochastic matrix for TextRank.\n\n        Element at row i and column j of the matrix corresponds to the similarity of sentence i\n        and j, where the similarity is computed as the number of common words between them, divided\n        by their sum of logarithm of their lengths. After such matrix is created, it is turned into\n        a stochastic matrix by normalizing over columns i.e. making the columns sum to one. TextRank\n        uses PageRank algorithm with damping, so a damping factor is incorporated as explained in\n        TextRank's paper. The resulting matrix is a stochastic matrix ready for power method.\n        "
        sentences_as_words = [self._to_words_set(sent) for sent in document.sentences]
        sentences_count = len(sentences_as_words)
        weights = numpy.zeros((sentences_count, sentences_count))
        for (i, words_i) in enumerate(sentences_as_words):
            for j in range(i, sentences_count):
                rating = self._rate_sentences_edge(words_i, sentences_as_words[j])
                weights[i, j] = rating
                weights[j, i] = rating
        weights /= weights.sum(axis=1)[:, numpy.newaxis] + self._ZERO_DIVISION_PREVENTION
        return numpy.full((sentences_count, sentences_count), (1.0 - self.damping) / sentences_count) + self.damping * weights

    def _to_words_set(self, sentence):
        if False:
            return 10
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    @staticmethod
    def _rate_sentences_edge(words1, words2):
        if False:
            i = 10
            return i + 15
        rank = sum((words2.count(w) for w in words1))
        if rank == 0:
            return 0.0
        assert len(words1) > 0 and len(words2) > 0
        norm = math.log(len(words1)) + math.log(len(words2))
        if numpy.isclose(norm, 0.0):
            assert rank in (0, 1)
            return float(rank)
        else:
            return rank / norm

    @staticmethod
    def power_method(matrix, epsilon):
        if False:
            print('Hello World!')
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0
        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p
        return p_vector