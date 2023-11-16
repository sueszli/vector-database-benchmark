from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
import unittest
import turicreate as tc
from turicreate.toolkits._main import ToolkitError
import random

class TreeRegressionTrackingMetricsTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        sf = tc.SFrame({'cat[1]': ['1', '1', '2', '2', '2'] * 20, 'cat[2]': ['1', '3', '3', '1', '1'] * 20, 'target': [random.random() for i in range(100)]})
        (cls.train, cls.test) = sf.random_split(0.5, seed=5)
        cls.default_metric = ['rmse', 'max_error']
        cls.test_metrics = ['rmse', 'max_error']
        cls.models = {'bst': tc.regression.boosted_trees_regression, 'rf': tc.regression.random_forest_regression, 'dt': tc.regression.decision_tree_regression}
        return cls

    def _metric_display_name(self, metric):
        if False:
            return 10
        metric_display_names = {'accuracy': 'Accuracy', 'auc': 'Area Under Curve', 'log_loss': 'Log Loss', 'max_error': 'Max Error', 'rmse': 'Root-Mean-Square Error'}
        if metric in metric_display_names:
            return metric_display_names[metric]
        else:
            return metric

    def _run_test(self, train, valid, metric):
        if False:
            i = 10
            return i + 15
        for (name, model) in self.models.items():
            m = model.create(train, 'target', validation_set=valid, max_depth=2, metric=metric)
            history_header = m.progress.column_names()
            if metric is 'auto':
                metric = self.default_metric
            if type(metric) is str:
                test_metrics = [metric]
            elif type(metric) is list:
                test_metrics = metric
            else:
                raise TypeError('Invalid metric type')
            for name in test_metrics:
                column_name = 'Training %s' % self._metric_display_name(name)
                self.assertTrue(column_name in history_header)
                final_eval = m.evaluate(train, name)[name]
                progress_evals = m.progress[column_name]
                self.assertAlmostEqual(float(progress_evals[-1]), final_eval, delta=0.0001)
                if valid is not None:
                    column_name = 'Validation %s' % self._metric_display_name(name)
                    self.assertTrue(column_name in history_header)

    def test_auto_metric(self):
        if False:
            return 10
        self._run_test(self.train, self.test, 'auto')

    def test_auto_metric_no_validation(self):
        if False:
            return 10
        self._run_test(self.train, None, 'auto')

    def test_single_metric(self):
        if False:
            while True:
                i = 10
        for m in self.test_metrics:
            self._run_test(self.train, self.test, m)

    def test_empty_metric(self):
        if False:
            while True:
                i = 10
        self._run_test(self.train, self.test, [])

    def test_many_metrics(self):
        if False:
            while True:
                i = 10
        self._run_test(self.train, self.test, self.test_metrics)

    def test_tracking_metrics_consistency(self):
        if False:
            print('Hello World!')
        rf_models = []
        for ntrees in [1, 2, 3]:
            m = self.models['rf'].create(self.train, 'target', validation_set=self.test, max_iterations=ntrees, metric=self.test_metrics, random_seed=1)
            rf_models.append(m)
        m_last = rf_models[-1]
        for name in self.test_metrics:
            train_column_name = 'Training %s' % self._metric_display_name(name)
            test_column_name = 'Validation %s' % self._metric_display_name(name)
            train_evals = [float(x) for x in m_last.progress[train_column_name]]
            test_evals = [float(x) for x in m_last.progress[test_column_name]]
            for i in range(len(train_evals) - 1):
                m_current = rf_models[i]
                train_expect = float(m_current.progress[train_column_name][-1])
                test_expect = float(m_current.progress[test_column_name][-1])
                self.assertAlmostEqual(train_evals[i], train_expect, delta=0.0001)
                self.assertAlmostEqual(test_evals[i], test_expect, delta=0.0001)

class BinaryTreeClassifierTrackingMetricsTest(TreeRegressionTrackingMetricsTest):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        sf = tc.SFrame({'cat[1]': ['1', '1', '2', '2', '2'] * 20, 'cat[2]': ['1', '3', '3', '1', '1'] * 20, 'target': ['0', '1'] * 50})
        (cls.train, cls.test) = sf.random_split(0.5, seed=5)
        cls.default_metric = ['log_loss', 'accuracy']
        cls.test_metrics = ['log_loss', 'accuracy', 'auc']
        cls.models = {'bst': tc.classifier.boosted_trees_classifier, 'rf': tc.classifier.random_forest_classifier, 'dt': tc.classifier.decision_tree_classifier}
        return cls

    def test_unseen_label_in_validation(self):
        if False:
            for i in range(10):
                print('nop')
        test = self.test.copy()
        l = len(test)
        test['target'] = test['target'].head(l - 10).append(tc.SArray(['unknown'] * 10))
        self._run_test(self.train, test, self.test_metrics)

    def test_auto_metric_unseen_label_in_validation(self):
        if False:
            for i in range(10):
                print('nop')
        test = self.test.copy()
        l = len(test)
        test['target'] = test['target'].head(l - 10).append(tc.SArray(['unknown'] * 10))
        self._run_test(self.train, test, 'auto')

class MultiClassTreeClassifierTrackingMetricsTest(BinaryTreeClassifierTrackingMetricsTest):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        sf = tc.SFrame({'cat[1]': ['1', '1', '2', '2', '2'] * 20, 'cat[2]': ['1', '3', '3', '1', '1'] * 20, 'target': ['0', '1', '2', '3'] * 25})
        (cls.train, cls.test) = sf.random_split(0.5, seed=5)
        cls.default_metric = ['log_loss', 'accuracy']
        cls.test_metrics = ['log_loss', 'accuracy']
        cls.models = {'bst': tc.classifier.boosted_trees_classifier, 'rf': tc.classifier.random_forest_classifier, 'dt': tc.classifier.decision_tree_classifier}
        return cls

    def test_auc_exception(self):
        if False:
            print('Hello World!')
        test = self.test.copy()
        l = len(test)
        test['target'] = test['target'].head(l - 10).append(tc.SArray(['unknown'] * 10))
        self.assertRaises(ToolkitError, lambda : self._run_test(self.train, test, 'auc'))