from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals
import math
try:
    import numpy
except ImportError:
    numpy = None
from collections import Counter
from ._summarizer import AbstractSummarizer

class LexRankSummarizer(AbstractSummarizer):
    """
    LexRank: Graph-based Centrality as Salience in Text Summarization
    Source: http://tangra.si.umich.edu/~radev/lexrank/lexrank.pdf
    """
    threshold = 0.1
    epsilon = 0.1
    _stop_words = frozenset()

    @property
    def stop_words(self):
        if False:
            return 10
        return self._stop_words

    @stop_words.setter
    def stop_words(self, words):
        if False:
            while True:
                i = 10
        self._stop_words = frozenset(map(self.normalize_word, words))

    def __call__(self, document, sentences_count):
        if False:
            return 10
        self._ensure_dependencies_installed()
        sentences_words = [self._to_words_set(s) for s in document.sentences]
        if not sentences_words:
            return tuple()
        tf_metrics = self._compute_tf(sentences_words)
        idf_metrics = self._compute_idf(sentences_words)
        matrix = self._create_matrix(sentences_words, self.threshold, tf_metrics, idf_metrics)
        scores = self.power_method(matrix, self.epsilon)
        ratings = dict(zip(document.sentences, scores))
        return self._get_best_sentences(document.sentences, sentences_count, ratings)

    @staticmethod
    def _ensure_dependencies_installed():
        if False:
            for i in range(10):
                print('nop')
        if numpy is None:
            raise ValueError("LexRank summarizer requires NumPy. Please, install it by command 'pip install numpy'.")

    def _to_words_set(self, sentence):
        if False:
            return 10
        words = map(self.normalize_word, sentence.words)
        return [self.stem_word(w) for w in words if w not in self._stop_words]

    def _compute_tf(self, sentences):
        if False:
            for i in range(10):
                print('nop')
        tf_values = map(Counter, sentences)
        tf_metrics = []
        for sentence in tf_values:
            metrics = {}
            max_tf = self._find_tf_max(sentence)
            for (term, tf) in sentence.items():
                metrics[term] = tf / max_tf
            tf_metrics.append(metrics)
        return tf_metrics

    @staticmethod
    def _find_tf_max(terms):
        if False:
            print('Hello World!')
        return max(terms.values()) if terms else 1

    @staticmethod
    def _compute_idf(sentences):
        if False:
            i = 10
            return i + 15
        idf_metrics = {}
        sentences_count = len(sentences)
        for sentence in sentences:
            for term in sentence:
                if term not in idf_metrics:
                    n_j = sum((1 for s in sentences if term in s))
                    idf_metrics[term] = math.log(sentences_count / (1 + n_j))
        return idf_metrics

    def _create_matrix(self, sentences, threshold, tf_metrics, idf_metrics):
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates matrix of shape |sentences|×|sentences|.\n        '
        sentences_count = len(sentences)
        matrix = numpy.zeros((sentences_count, sentences_count))
        degrees = numpy.zeros((sentences_count,))
        for (row, (sentence1, tf1)) in enumerate(zip(sentences, tf_metrics)):
            for (col, (sentence2, tf2)) in enumerate(zip(sentences, tf_metrics)):
                matrix[row, col] = self.cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics)
                if matrix[row, col] > threshold:
                    matrix[row, col] = 1.0
                    degrees[row] += 1
                else:
                    matrix[row, col] = 0
        for row in range(sentences_count):
            for col in range(sentences_count):
                if degrees[row] == 0:
                    degrees[row] = 1
                matrix[row][col] = matrix[row][col] / degrees[row]
        return matrix

    @staticmethod
    def cosine_similarity(sentence1, sentence2, tf1, tf2, idf_metrics):
        if False:
            while True:
                i = 10
        "\n        We compute idf-modified-cosine(sentence1, sentence2) here.\n        It's cosine similarity of these two sentences (vectors) A, B computed as cos(x, y) = A . B / (|A| . |B|)\n        Sentences are represented as vector TF*IDF metrics.\n\n        :param sentence1:\n            Iterable object where every item represents word of 1st sentence.\n        :param sentence2:\n            Iterable object where every item represents word of 2nd sentence.\n        :type tf1: dict\n        :param tf1:\n            Term frequencies of words from 1st sentence.\n        :type tf2: dict\n        :param tf2:\n            Term frequencies of words from 2nd sentence\n        :type idf_metrics: dict\n        :param idf_metrics:\n            Inverted document metrics of the sentences. Every sentence is treated as document for this algorithm.\n        :rtype: float\n        :return:\n            Returns -1.0 for opposite similarity, 1.0 for the same sentence and zero for no similarity between sentences.\n        "
        unique_words1 = frozenset(sentence1)
        unique_words2 = frozenset(sentence2)
        common_words = unique_words1 & unique_words2
        numerator = 0.0
        for term in common_words:
            numerator += tf1[term] * tf2[term] * idf_metrics[term] ** 2
        denominator1 = sum(((tf1[t] * idf_metrics[t]) ** 2 for t in unique_words1))
        denominator2 = sum(((tf2[t] * idf_metrics[t]) ** 2 for t in unique_words2))
        if denominator1 > 0 and denominator2 > 0:
            return numerator / (math.sqrt(denominator1) * math.sqrt(denominator2))
        else:
            return 0.0

    @staticmethod
    def power_method(matrix, epsilon):
        if False:
            while True:
                i = 10
        transposed_matrix = matrix.T
        sentences_count = len(matrix)
        p_vector = numpy.array([1.0 / sentences_count] * sentences_count)
        lambda_val = 1.0
        while lambda_val > epsilon:
            next_p = numpy.dot(transposed_matrix, p_vector)
            next_p /= numpy.linalg.norm(next_p)
            lambda_val = numpy.linalg.norm(numpy.subtract(next_p, p_vector))
            p_vector = next_p
        return p_vector