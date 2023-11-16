import unittest
from types import SimpleNamespace
from typing import Tuple
import numpy as np
import pandas as pd
from snorkel.analysis import Scorer
from snorkel.slicing import SFApplier, slicing_function
from snorkel.utils import preds_to_probs

class ScorerTest(unittest.TestCase):

    def _get_labels(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if False:
            return 10
        golds = np.array([1, 0, 1, 0, 1])
        preds = np.array([1, 0, 1, 1, 0])
        probs = np.array([0.8, 0.6, 0.9, 0.7, 0.4])
        return (golds, preds, probs)

    def test_scorer(self) -> None:
        if False:
            return 10

        def pred_sum(golds, preds, probs):
            if False:
                print('Hello World!')
            return np.sum(preds)
        scorer = Scorer(metrics=['accuracy', 'f1'], custom_metric_funcs=dict(pred_sum=pred_sum))
        results = scorer.score(*self._get_labels())
        results_expected = dict(accuracy=0.6, f1=2 / 3, pred_sum=3)
        self.assertEqual(results, results_expected)

    def test_dict_metric(self) -> None:
        if False:
            print('Hello World!')

        def dict_metric(golds, preds, probs):
            if False:
                return 10
            return dict(a=1, b=2)
        scorer = Scorer(custom_metric_funcs=dict(dict_metric=dict_metric))
        results = scorer.score(*self._get_labels())
        results_expected = dict(a=1, b=2)
        self.assertEqual(results, results_expected)

    def test_invalid_metric(self) -> None:
        if False:
            while True:
                i = 10
        with self.assertRaisesRegex(ValueError, 'Unrecognized metric'):
            Scorer(metrics=['accuracy', 'f2'])

    def test_no_metrics(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        scorer = Scorer()
        self.assertEqual(scorer.score(*self._get_labels()), {})

    def test_no_labels(self) -> None:
        if False:
            while True:
                i = 10
        scorer = Scorer()
        with self.assertRaisesRegex(ValueError, 'Cannot score'):
            scorer.score([], [], [])

    def test_no_probs(self) -> None:
        if False:
            return 10
        scorer = Scorer()
        (golds, preds, probs) = self._get_labels()
        self.assertEqual(scorer.score(golds, preds), scorer.score(golds, preds, probs))

    def test_abstain_labels(self) -> None:
        if False:
            print('Hello World!')
        golds = np.array([1, 0, 1, 0, -1])
        preds = np.array([1, 0, 1, 1, 0])
        probs = np.array([0.8, 0.6, 0.9, 0.7, 0.4])
        scorer = Scorer(metrics=['accuracy'], abstain_label=None)
        results = scorer.score(golds, preds, probs)
        results_expected = dict(accuracy=0.6)
        self.assertEqual(results, results_expected)
        scorer = Scorer(metrics=['accuracy'], abstain_label=-1)
        results = scorer.score(golds, preds, probs)
        results_expected = dict(accuracy=0.75)
        self.assertEqual(results, results_expected)
        abstain_preds = np.array([-1, -1, 1, 1, 0])
        results = scorer.score(golds, abstain_preds)
        results_expected = dict(accuracy=0.5)
        self.assertEqual(results, results_expected)
        scorer = Scorer(metrics=['coverage'], abstain_label=-1)
        results = scorer.score(golds, abstain_preds)
        results_expected = dict(coverage=0.6)
        self.assertEqual(results, results_expected)
        scorer = Scorer(metrics=['accuracy'], abstain_label=10)
        results = scorer.score(golds, preds, probs)
        results_expected = dict(accuracy=0.6)
        self.assertEqual(results, results_expected)

    def test_score_slices(self):
        if False:
            while True:
                i = 10
        DATA = [5, 10, 19, 22, 25]

        @slicing_function()
        def sf(x):
            if False:
                print('Hello World!')
            return x.num < 20
        golds = np.array([0, 1, 0, 1, 0])
        preds = np.array([0, 0, 0, 0, 0])
        probs = preds_to_probs(preds, 2)
        data = [SimpleNamespace(num=x) for x in DATA]
        S = SFApplier([sf]).apply(data)
        scorer = Scorer(metrics=['accuracy'])
        metrics = scorer.score(golds=golds, preds=preds, probs=probs)
        self.assertEqual(metrics['accuracy'], 0.6)
        slice_metrics = scorer.score_slices(S=S, golds=golds, preds=preds, probs=probs)
        self.assertEqual(slice_metrics['overall']['accuracy'], 0.6)
        self.assertEqual(slice_metrics['sf']['accuracy'], 2.0 / 3.0)
        metrics_df = scorer.score_slices(S=S, golds=golds, preds=preds, probs=probs, as_dataframe=True)
        self.assertTrue(isinstance(metrics_df, pd.DataFrame))
        self.assertEqual(metrics_df['accuracy']['overall'], 0.6)
        self.assertEqual(metrics_df['accuracy']['sf'], 2.0 / 3.0)
        with self.assertRaisesRegex(ValueError, 'must have the same number of elements'):
            scorer.score_slices(S=S, golds=golds[:1], preds=preds, probs=probs, as_dataframe=True)
if __name__ == '__main__':
    unittest.main()