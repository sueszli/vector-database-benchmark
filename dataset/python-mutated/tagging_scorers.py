"""Sequence tagging evaluation."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import abc
from task_specific.word_level import tagging_utils
from task_specific.word_level import word_level_scorer

class AccuracyScorer(word_level_scorer.WordLevelScorer):

    def __init__(self, auto_fail_label=None):
        if False:
            while True:
                i = 10
        super(AccuracyScorer, self).__init__()
        self._auto_fail_label = auto_fail_label

    def _get_results(self):
        if False:
            for i in range(10):
                print('nop')
        (correct, count) = (0, 0)
        for (example, preds) in zip(self._examples, self._preds):
            for (y_true, y_pred) in zip(example.labels, preds):
                count += 1
                correct += 1 if y_pred == y_true and y_true != self._auto_fail_label else 0
        return [('accuracy', 100.0 * correct / count), ('loss', self.get_loss())]

class F1Scorer(word_level_scorer.WordLevelScorer):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(F1Scorer, self).__init__()
        (self._n_correct, self._n_predicted, self._n_gold) = (0, 0, 0)

    def _get_results(self):
        if False:
            for i in range(10):
                print('nop')
        if self._n_correct == 0:
            (p, r, f1) = (0, 0, 0)
        else:
            p = 100.0 * self._n_correct / self._n_predicted
            r = 100.0 * self._n_correct / self._n_gold
            f1 = 2 * p * r / (p + r)
        return [('precision', p), ('recall', r), ('f1', f1), ('loss', self.get_loss())]

class EntityLevelF1Scorer(F1Scorer):

    def __init__(self, label_mapping):
        if False:
            print('Hello World!')
        super(EntityLevelF1Scorer, self).__init__()
        self._inv_label_mapping = {v: k for (k, v) in label_mapping.iteritems()}

    def _get_results(self):
        if False:
            i = 10
            return i + 15
        (self._n_correct, self._n_predicted, self._n_gold) = (0, 0, 0)
        for (example, preds) in zip(self._examples, self._preds):
            sent_spans = set(tagging_utils.get_span_labels(example.labels, self._inv_label_mapping))
            span_preds = set(tagging_utils.get_span_labels(preds, self._inv_label_mapping))
            self._n_correct += len(sent_spans & span_preds)
            self._n_gold += len(sent_spans)
            self._n_predicted += len(span_preds)
        return super(EntityLevelF1Scorer, self)._get_results()