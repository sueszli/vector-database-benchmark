"""Dependency parsing evaluation (computes UAS/LAS)."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from task_specific.word_level import word_level_scorer

class DepparseScorer(word_level_scorer.WordLevelScorer):

    def __init__(self, n_relations, punctuation):
        if False:
            for i in range(10):
                print('nop')
        super(DepparseScorer, self).__init__()
        self._n_relations = n_relations
        self._punctuation = punctuation if punctuation else None

    def _get_results(self):
        if False:
            print('Hello World!')
        (correct_unlabeled, correct_labeled, count) = (0, 0, 0)
        for (example, preds) in zip(self._examples, self._preds):
            for (w, y_true, y_pred) in zip(example.words[1:-1], example.labels, preds):
                if w in self._punctuation:
                    continue
                count += 1
                correct_labeled += 1 if y_pred == y_true else 0
                correct_unlabeled += 1 if int(y_pred // self._n_relations) == int(y_true // self._n_relations) else 0
        return [('las', 100.0 * correct_labeled / count), ('uas', 100.0 * correct_unlabeled / count), ('loss', self.get_loss())]