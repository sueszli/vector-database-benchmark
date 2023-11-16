"""Base class for word-level scorers."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from corpus_processing import scorer

class WordLevelScorer(scorer.Scorer):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(WordLevelScorer, self).__init__()
        self._total_loss = 0
        self._total_words = 0
        self._examples = []
        self._preds = []

    def update(self, examples, predictions, loss):
        if False:
            i = 10
            return i + 15
        super(WordLevelScorer, self).update(examples, predictions, loss)
        n_words = 0
        for (example, preds) in zip(examples, predictions):
            self._examples.append(example)
            self._preds.append(list(preds)[1:len(example.words) - 1])
            n_words += len(example.words) - 2
        self._total_loss += loss * n_words
        self._total_words += n_words

    def get_loss(self):
        if False:
            print('Hello World!')
        return self._total_loss / max(1, self._total_words)