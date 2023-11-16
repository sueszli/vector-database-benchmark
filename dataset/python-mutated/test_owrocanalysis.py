import copy
import unittest
from unittest.mock import patch, Mock
import numpy as np
import pyqtgraph as pg
from AnyQt.QtWidgets import QToolTip
from AnyQt.QtCore import QItemSelection
from Orange.data import Table
import Orange.evaluation
import Orange.classification
from Orange.evaluation import Results
from Orange.widgets.evaluate import owrocanalysis
from Orange.widgets.evaluate.owrocanalysis import OWROCAnalysis
from Orange.widgets.evaluate.tests.base import EvaluateTest
from Orange.widgets.tests.utils import mouseMove, simulate
from Orange.tests import test_filename

class TestROC(unittest.TestCase):

    def test_ROCData_from_results(self):
        if False:
            i = 10
            return i + 15
        data = Orange.data.Table('iris')
        learners = [Orange.classification.MajorityLearner(), Orange.classification.LogisticRegressionLearner(), Orange.classification.TreeLearner()]
        cv = Orange.evaluation.CrossValidation(k=10)
        res = cv(data, learners)
        for (i, _) in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.roc_data_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 10)
                self.assertTrue(all((c.is_valid for c in rocdata.folds)))
                self.assertTrue(rocdata.avg_vertical.is_valid)
                self.assertTrue(rocdata.avg_threshold.is_valid)
        data = data[np.random.RandomState(0).choice(len(data), size=20)]
        loo = Orange.evaluation.LeaveOneOut()
        res = loo(data, learners)
        for (i, _) in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.roc_data_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 20)
                self.assertTrue(all((not c.is_valid for c in rocdata.folds)))
                self.assertFalse(rocdata.avg_vertical.is_valid)
                self.assertFalse(rocdata.avg_threshold.is_valid)
        cv = Orange.evaluation.CrossValidation(k=20)
        res = cv(data, learners)
        for (i, _) in enumerate(learners):
            for c in range(len(data.domain.class_var.values)):
                rocdata = owrocanalysis.roc_data_from_results(res, i, target=c)
                self.assertTrue(rocdata.merged.is_valid)
                self.assertEqual(len(rocdata.folds), 20)
                self.assertTrue(all((not c.is_valid for c in rocdata.folds)))
                self.assertFalse(rocdata.avg_vertical.is_valid)
                self.assertFalse(rocdata.avg_threshold.is_valid)

class TestOWROCAnalysis(EvaluateTest):

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        super().setUpClass()
        cls.lenses = data = Table(test_filename('datasets/lenses.tab'))
        totd = Orange.evaluation.TestOnTestData(store_data=True)
        cls.res = totd(data=data[::2], test_data=data[1::2], learners=[Orange.classification.MajorityLearner(), Orange.classification.KNNLearner()])
        try:
            pg.setConfigOption('mouseRateLimit', -1)
        except KeyError:
            pass

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.widget = self.create_widget(OWROCAnalysis, stored_settings={'display_perf_line': True, 'display_def_threshold': True, 'display_convex_hull': True, 'display_convex_curve': True})

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        self.widget.onDeleteWidget()
        self.widgets.remove(self.widget)
        self.widget = None

    @staticmethod
    def _set_list_selection(listview, selection):
        if False:
            i = 10
            return i + 15
        model = listview.model()
        selectionmodel = listview.selectionModel()
        itemselection = QItemSelection()
        for item in selection:
            itemselection.select(model.index(item, 0), model.index(item, 0))
        selectionmodel.select(itemselection, selectionmodel.ClearAndSelect)

    def test_basic(self):
        if False:
            print('Hello World!')
        res = self.res
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()
        self.send_signal(self.widget.Inputs.evaluation_results, None)

    def test_empty_input(self):
        if False:
            return 10
        res = Orange.evaluation.Results(data=self.lenses[:0], nmethods=2, store_data=True)
        res.row_indices = np.array([], dtype=int)
        res.actual = np.array([])
        res.predicted = np.zeros((2, 0))
        res.probabilities = np.zeros((2, 0, 3))
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()
        res.row_indices = np.array([1], dtype=int)
        res.actual = np.array([0.0])
        res.predicted = np.zeros((2, 1))
        res.probabilities = np.zeros((2, 1, 3))
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Vertical
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.Threshold
        self.widget._replot()
        self.widget.roc_averaging = OWROCAnalysis.NoAveraging
        self.widget._replot()

    def test_nan_input(self):
        if False:
            print('Hello World!')
        res = copy.copy(self.res)
        res.actual = res.actual.copy()
        res.predicted = res.predicted.copy()
        res.probabilities = res.probabilities.copy()
        res.actual[0] = np.nan
        res.predicted[:, 1] = np.nan
        res.probabilities[0, 1, :] = np.nan
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.assertTrue(self.widget.Error.invalid_results.is_shown())
        self.send_signal(self.widget.Inputs.evaluation_results, None)
        self.assertFalse(self.widget.Error.invalid_results.is_shown())

    def test_tooltips(self):
        if False:
            return 10
        actual = np.array([float(c == 'n') for c in 'ppnpppnnpnpnpnnnpnpn'])
        p = np.array([0.9, 0.8, 0.7, 0.6, 0.55, 0.54, 0.53, 0.52, 0.51, 0.505, 0.4, 0.39, 0.38, 0.37, 0.36, 0.35, 0.34, 0.33, 0.3, 0.1])
        n = 1 - p
        predicted = (p > 0.5).astype(float)
        p2 = p.copy()
        p2[:4] = [0.7, 0.8, 0.9, 0.59]
        n2 = 1 - p2
        predicted2 = (p2 < 0.5).astype(float)
        data = Orange.data.Table(Orange.data.Domain([], [Orange.data.DiscreteVariable('y', values=tuple('pn'))]), np.empty((len(p), 0), dtype=float), actual)
        res = Results(data=data, actual=actual, predicted=np.array([list(predicted), list(predicted2)]), probabilities=np.array([list(zip(p, n)), list(zip(p2, n2))]))
        self.send_signal(self.widget.Inputs.evaluation_results, res)
        self.widget.roc_averaging = OWROCAnalysis.Merge
        self.widget.target_index = 0
        self.widget.selected_classifiers = [0, 1]
        vb = self.widget.plot.getViewBox()
        vb.childTransform()
        curve = self.widget.plot_curves(self.widget.target_index, 0)
        curve_merge = curve.merge()
        view = self.widget.plotview
        item = curve_merge.curve_item
        with patch.object(QToolTip, 'showText') as show_text:
            pos = item.mapToScene(0.0, 1.0)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            show_text.assert_not_called()
            pos = item.mapToScene(0, 0.1)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            ((_, text), _) = show_text.call_args
            self.assertIn('(#1) 0.900', text)
            self.assertNotIn('#2', text)
            pos = item.mapToScene(0.0, 0.0)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            ((_, text), _) = show_text.call_args
            self.assertIn('(#1) 1.000\n(#2) 1.000', text)
            pos = item.mapToScene(0.1, 0.3)
            pos = view.mapFromScene(pos)
            mouseMove(view.viewport(), pos)
            ((_, text), _) = show_text.call_args
            self.assertIn('(#1) 0.600\n(#2) 0.590', text)
            show_text.reset_mock()
            self.widget.roc_averaging = OWROCAnalysis.Threshold
            self.widget._replot()
            mouseMove(view.viewport(), pos)
            ((_, text), _) = show_text.call_args
            self.assertIn('(#1) 0.600\n(#2) 0.590', text)
            show_text.reset_mock()
            self.widget.roc_averaging = OWROCAnalysis.Vertical
            self.widget._replot()
            mouseMove(view.viewport(), pos)
            show_text.assert_not_called()

    def test_target_prior(self):
        if False:
            print('Hello World!')
        w = self.widget
        self.send_signal(w.Inputs.evaluation_results, self.res)
        self.assertEqual(np.round(4 / 12 * 100), w.target_prior)
        simulate.combobox_activate_item(w.controls.target_index, 'none')
        self.assertEqual(np.round(3 / 12 * 100), w.target_prior)
        simulate.combobox_activate_item(w.controls.target_index, 'soft')
        self.assertEqual(np.round(5 / 12 * 100), w.target_prior)

    @patch('Orange.widgets.evaluate.owrocanalysis.ThresholdClassifier')
    def test_apply_no_output(self, *_):
        if False:
            while True:
                i = 10
        'Test no output warnings'
        widget = self.widget
        model_list = widget.controls.selected_classifiers
        (multiple_folds, multiple_selected, no_models, non_binary_class) = 'abcd'
        messages = {multiple_folds: 'each training data sample produces a different model', no_models: 'test results do not contain stored models - try testing on separate data or on training data', multiple_selected: 'select a single model - the widget can output only one', non_binary_class: 'cannot calibrate non-binary models'}

        def test_shown(shown):
            if False:
                while True:
                    i = 10
            widget_msg = widget.Information.no_output
            output = self.get_output(widget.Outputs.calibrated_model)
            if not shown:
                self.assertFalse(widget_msg.is_shown())
                self.assertIsNotNone(output)
            else:
                self.assertTrue(widget_msg.is_shown())
                self.assertIsNone(output)
                for msg_id in shown:
                    msg = messages[msg_id]
                    self.assertIn(msg, widget_msg.formatted, f'{msg} not included in the message')
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        test_shown({multiple_selected})
        self._set_list_selection(model_list, [0])
        test_shown(())
        widget.controls.display_perf_line.click()
        output = self.get_output(widget.Outputs.calibrated_model)
        self.assertIsNone(output)
        widget.controls.display_perf_line.click()
        output = self.get_output(widget.Outputs.calibrated_model)
        self.assertIsNotNone(output)
        self._set_list_selection(model_list, [0, 1])
        self.results.models = None
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        test_shown({multiple_selected, no_models})
        self.send_signal(widget.Inputs.evaluation_results, self.lenses_results)
        test_shown({multiple_selected, non_binary_class})
        self._set_list_selection(model_list, [0])
        test_shown({non_binary_class})
        self.results.folds = [slice(0, 5), slice(5, 10), slice(10, 19)]
        self.results.models = np.array([[Mock(), Mock()]] * 3)
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        test_shown({multiple_selected, multiple_folds})
        self._set_list_selection(model_list, [0])
        test_shown({multiple_folds})

    @patch('Orange.widgets.evaluate.owrocanalysis.ThresholdClassifier')
    def test_calibrated_output(self, tc):
        if False:
            for i in range(10):
                print('nop')
        widget = self.widget
        model_list = widget.controls.selected_classifiers
        self.send_signal(widget.Inputs.evaluation_results, self.results)
        self._set_list_selection(model_list, [0])
        (model, threshold) = tc.call_args[0]
        self.assertIs(model, self.results.models[0][0])
        self.assertAlmostEqual(threshold, 0.47)
        widget.controls.fp_cost.setValue(1000)
        (model, threshold) = tc.call_args[0]
        self.assertIs(model, self.results.models[0][0])
        self.assertAlmostEqual(threshold, 0.9)
        self._set_list_selection(model_list, [1])
        (model, threshold) = tc.call_args[0]
        self.assertIs(model, self.results.models[0][1])
        self.assertAlmostEqual(threshold, 0.45)
if __name__ == '__main__':
    unittest.main()