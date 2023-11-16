import unittest
from unittest.mock import patch, Mock, call
import numpy as np
from Orange.data import DiscreteVariable, ContinuousVariable, Domain, Table
from Orange.preprocess import Normalize
from Orange.projection import manifold, TSNE
from Orange.projection.manifold import TSNEModel
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, ProjectionWidgetTestMixin
from Orange.widgets.unsupervised.owtsne import OWtSNE, TSNERunner, Task, prepare_tsne_obj

class DummyTSNE(manifold.TSNE):

    def fit(self, X, Y=None):
        if False:
            print('Hello World!')
        return np.ones((len(X), 2), float)

class DummyTSNEModel(manifold.TSNEModel):

    def transform(self, X, **kwargs):
        if False:
            while True:
                i = 10
        return np.ones((len(X), 2), float)

    def optimize(self, n_iter, **kwargs):
        if False:
            i = 10
            return i + 15
        return self

class TestOWtSNE(WidgetTest, ProjectionWidgetTestMixin, WidgetOutputsTestMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.same_input_output_domain = False
        cls.signal_name = OWtSNE.Inputs.data
        cls.signal_data = cls.data

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.tsne = patch('Orange.projection.manifold.TSNE', new=DummyTSNE)
        self.tsne_model = patch('Orange.projection.manifold.TSNEModel', new=DummyTSNEModel)
        self.tsne.start()
        self.tsne_model.start()
        self.widget = self.create_widget(OWtSNE, stored_settings={'multiscale': False})
        self.class_var = DiscreteVariable('Stage name', values=('STG1', 'STG2'))
        self.attributes = [ContinuousVariable('GeneName' + str(i)) for i in range(5)]
        self.domain = Domain(self.attributes, class_vars=self.class_var)
        self.empty_domain = Domain([], class_vars=self.class_var)

    def tearDown(self):
        if False:
            print('Hello World!')
        self.widget.onDeleteWidget()
        try:
            self.restore_mocked_functions()
        except RuntimeError as e:
            if str(e) != 'stop called on unstarted patcher':
                raise e

    def restore_mocked_functions(self):
        if False:
            return 10
        self.tsne.stop()
        self.tsne_model.stop()

    def test_wrong_input(self):
        if False:
            return 10
        data = None
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        data = Table.from_list(self.domain, [[1, 2, 3, 4, 5, 'STG1']])
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_rows.is_shown())
        data = Table.from_list(self.empty_domain, [['STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_cols.is_shown())
        data = Table.from_list(self.empty_domain, [[1, 'STG1'], [2, 'STG1']])
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.not_enough_cols.is_shown())
        data = Table.from_list(self.domain, [[1, 2, 3, 4, 5, 'STG1']] * 2)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNone(self.widget.data)
        self.assertTrue(self.widget.Error.constant_data.is_shown())
        data = Table.from_list(self.domain, [[1, 2, 3, 4, 5, 'STG1'], [5, 4, 3, 2, 1, 'STG1']])
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        self.assertIsNotNone(self.widget.data)
        self.assertFalse(self.widget.Error.not_enough_rows.is_shown())
        self.assertFalse(self.widget.Error.not_enough_cols.is_shown())
        self.assertFalse(self.widget.Error.constant_data.is_shown())

    def test_input(self):
        if False:
            while True:
                i = 10
        data = Table.from_list(self.domain, [[1, 1, 1, 1, 1, 'STG1'], [2, 2, 2, 2, 2, 'STG1'], [4, 4, 4, 4, 4, 'STG2'], [5, 5, 5, 5, 5, 'STG2']])
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()

    def test_attr_models(self):
        if False:
            print('Hello World!')
        "Check possible values for 'Color', 'Shape', 'Size' and 'Label'"
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        controls = self.widget.controls
        for var in self.data.domain.class_vars + self.data.domain.metas:
            self.assertIn(var, controls.attr_color.model())
            self.assertIn(var, controls.attr_label.model())
            if var.is_continuous:
                self.assertIn(var, controls.attr_size.model())
                self.assertNotIn(var, controls.attr_shape.model())
            if var.is_discrete:
                self.assertNotIn(var, controls.attr_size.model())
                self.assertIn(var, controls.attr_shape.model())

    def test_multiscale_changed_updates_ui(self):
        if False:
            return 10
        self.send_signal(self.widget.Inputs.data, self.data)
        self.assertFalse(self.widget.controls.multiscale.isChecked())
        self.assertTrue(self.widget.perplexity_spin.isEnabled())
        self.widget.controls.multiscale.setChecked(True)
        self.assertFalse(self.widget.perplexity_spin.isEnabled())
        settings = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWtSNE, stored_settings=settings)
        self.send_signal(w.Inputs.data, self.data, widget=w)
        self.assertTrue(w.controls.multiscale.isChecked())
        self.assertFalse(w.perplexity_spin.isEnabled())
        w.onDeleteWidget()

    def test_normalize_data(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.widget.controls.normalize.isChecked())
        with patch('Orange.preprocess.preprocess.Normalize', wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            self.wait_until_finished()
            normalize.assert_called_once()
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.controls.normalize.isChecked())
        with patch('Orange.preprocess.preprocess.Normalize', wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, self.data)
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            self.wait_until_finished()
            normalize.assert_not_called()
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.controls.normalize.isChecked())
        sparse_data = self.data.to_sparse()
        with patch('Orange.preprocess.preprocess.Normalize', wraps=Normalize) as normalize:
            self.send_signal(self.widget.Inputs.data, sparse_data)
            self.assertFalse(self.widget.controls.normalize.isEnabled())
            self.wait_until_finished()
            normalize.assert_not_called()

    @patch('Orange.projection.manifold.TSNEModel.optimize')
    def test_exaggeration_is_passed_through_properly(self, optimize):
        if False:
            i = 10
            return i + 15

        def _check_exaggeration(call, exaggeration):
            if False:
                while True:
                    i = 10
            (_, _, kwargs) = call.mock_calls[-1]
            self.assertIn('exaggeration', kwargs)
            self.assertEqual(kwargs['exaggeration'], exaggeration)
        optimize.return_value = DummyTSNE()(self.data)
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.run_button.clicked.emit()
        self.wait_until_stop_blocking()
        self.widget.controls.exaggeration.setValue(1)
        self.widget.run_button.clicked.emit()
        self.wait_until_finished()
        _check_exaggeration(optimize, 1)
        self.send_signal(self.widget.Inputs.data, None)
        optimize.reset_mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.run_button.clicked.emit()
        self.wait_until_stop_blocking()
        self.widget.controls.exaggeration.setValue(3)
        self.widget.run_button.clicked.emit()
        self.wait_until_finished()
        _check_exaggeration(optimize, 3)

    def test_plot_once(self):
        if False:
            print('Hello World!')
        'Test if data is plotted only once but committed on every input change'
        self.widget.setup_plot = Mock()
        self.widget.commit.deferred = self.widget.commit.now = Mock()
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.setup_plot.reset_mock()
        self.widget.commit.deferred.reset_mock()
        self.wait_until_finished()
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()
        self.widget.commit.deferred.reset_mock()
        self.send_signal(self.widget.Inputs.data_subset, self.data[::10])
        self.wait_until_stop_blocking()
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()

    def test_modified_info_message_behaviour(self):
        if False:
            for i in range(10):
                print('nop')
        'Information messages should be cleared if the data changes or if\n        the data is set to None.'
        self.assertFalse(self.widget.Information.modified.is_shown(), 'The modified info message should be hidden by default')
        self.widget.controls.multiscale.setChecked(False)
        self.assertFalse(self.widget.Information.modified.is_shown(), 'The modified info message should be hidden even after toggling options if no data is on input')
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Information.modified.is_shown(), 'The modified info message should be hidden after the widget computes the embedding')
        self.send_signal(self.widget.Inputs.data, self.data)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Information.modified.is_shown(), 'The modified info message should be hidden when reloading the same data set and no previous messages were shown')
        self.widget.controls.multiscale.setChecked(True)
        self.assertTrue(self.widget.Information.modified.is_shown(), 'The modified info message should be shown when a setting is changed, but the embedding is not recomputed')
        self.send_signal(self.widget.Inputs.data, Table('housing'))
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Information.modified.is_shown(), 'The information message was not cleared on new data')
        self.widget.controls.multiscale.setChecked(False)
        assert self.widget.Information.modified.is_shown()
        self.send_signal(self.widget.Inputs.data, None)
        self.wait_until_stop_blocking()
        self.assertFalse(self.widget.Information.modified.is_shown(), 'The information message was not cleared on no data')

    def test_invalidation_flow(self):
        if False:
            return 10
        w = self.widget
        w.controls.multiscale.setChecked(False)
        self.send_signal(w.Inputs.data, self.data)
        self.wait_until_finished()
        self.assertFalse(self.widget.Information.modified.is_shown())
        self.assertIsNotNone(w.pca_projection)
        self.assertIsNotNone(w.affinities)
        self.assertIsNotNone(w.tsne_embedding)
        self.assertFalse(w._invalidated.pca_projection)
        self.assertFalse(w._invalidated.affinities)
        self.assertFalse(w._invalidated.tsne_embedding)
        w.controls.multiscale.setChecked(True)
        self.assertTrue(self.widget.Information.modified.is_shown())
        self.assertFalse(w._invalidated.pca_projection)
        self.assertTrue(w._invalidated.affinities)
        self.assertTrue(w._invalidated.tsne_embedding)
        self.assertIsNotNone(w.pca_projection)
        self.assertIsNotNone(w.affinities)
        self.assertIsNotNone(w.tsne_embedding)
        self.send_signal(w.Inputs.data_subset, self.data[:10])
        self.wait_until_stop_blocking()
        subset = [brush.color().name() == '#46befa' for brush in w.graph.scatterplot_item.data['brush'][:10]]
        other = [brush.color().name() == '#000000' for brush in w.graph.scatterplot_item.data['brush'][10:]]
        self.assertTrue(all(subset))
        self.assertTrue(all(other))
        self.send_signal(w.Inputs.data_subset, None)
        self.widget.run_button.clicked.emit()
        self.wait_until_stop_blocking()
        self.assertFalse(w._invalidated)

class TestTSNERunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls.data = Table('iris')

    def test_run(self):
        if False:
            while True:
                i = 10
        state = Mock()
        state.is_interruption_requested = Mock(return_value=False)
        task = TSNERunner.run(Task(data=self.data, perplexity=30), state)
        self.assertEqual(len(state.set_status.mock_calls), 4)
        state.set_status.assert_has_calls([call('Computing PCA...'), call('Preparing initialization...'), call('Finding nearest neighbors...'), call('Running optimization...')])
        self.assertIsInstance(task.pca_projection, Table)
        self.assertIsInstance(task.tsne, TSNE)
        self.assertIsInstance(task.tsne_embedding, TSNEModel)

    def test_run_do_not_modify_model_inplace(self):
        if False:
            i = 10
            return i + 15
        state = Mock()
        state.is_interruption_requested.return_value = True
        task = Task(data=self.data, perplexity=30, multiscale=False, exaggeration=1)
        task.tsne = prepare_tsne_obj(task.data, task.perplexity, task.multiscale, task.exaggeration)
        TSNERunner.compute_pca(task, state)
        TSNERunner.compute_initialization(task, state)
        TSNERunner.compute_affinities(task, state)
        TSNERunner.compute_tsne(task, state)
        tsne_obj_before = task.tsne_embedding
        state.reset_mock()
        TSNERunner.compute_tsne(task, state)
        tsne_obj_after = task.tsne_embedding
        state.set_partial_result.assert_called_once()
        self.assertIsNot(tsne_obj_before, tsne_obj_after)
if __name__ == '__main__':
    unittest.main()