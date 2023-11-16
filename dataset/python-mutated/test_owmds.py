import os
from itertools import chain
import unittest
from unittest.mock import patch, Mock
import numpy as np
from Orange.data import Table
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.projection.manifold import torgerson
from Orange.widgets.settings import Context
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin, datasets, ProjectionWidgetTestMixin
from Orange.widgets.tests.utils import simulate
from Orange.widgets.unsupervised.owmds import OWMDS, run_mds, Result

class TestOWMDS(WidgetTest, ProjectionWidgetTestMixin, WidgetOutputsTestMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.signal_name = OWMDS.Inputs.distances
        cls.signal_data = Euclidean(cls.data)
        cls.same_input_output_domain = False
        my_dir = os.path.dirname(__file__)
        datasets_dir = os.path.join(my_dir, '..', '..', '..', 'datasets')
        cls.datasets_dir = os.path.realpath(datasets_dir)

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.widget = self.create_widget(OWMDS, stored_settings={'__version__': 2, 'max_iter': 10, 'initialization': OWMDS.PCA})
        self.towns = DistMatrix.from_file(os.path.join(self.datasets_dir, 'slovenian-towns.dst'))

    def tearDown(self):
        if False:
            return 10
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_plot_once(self):
        if False:
            for i in range(10):
                print('nop')
        'Test if data is plotted only once but committed on every input change'
        table = Table('heart_disease')
        self.widget.setup_plot = Mock()
        self.widget.commit.deferred = self.widget.commit.now = Mock()
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.commit.deferred.reset_mock()
        self.wait_until_finished()
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()
        self.widget.commit.deferred.reset_mock()
        self.send_signal(self.widget.Inputs.data_subset, table[::10])
        self.wait_until_stop_blocking()
        self.widget.setup_plot.assert_called_once()
        self.widget.commit.deferred.assert_called_once()

    def test_pca_init(self):
        if False:
            return 10
        self.send_signal(self.signal_name, self.signal_data)
        output = self.get_output(self.widget.Outputs.annotated_data, wait=1000)
        expected = np.array([[-2.69304803, 0.32676458], [-2.7246721, -0.20921726], [-2.90244761, -0.13630526], [-2.75281107, -0.33854819]])
        np.testing.assert_array_almost_equal(output.metas[:4, :2], expected)

    def test_nan_plot(self):
        if False:
            i = 10
            return i + 15

        def combobox_run_through_all():
            if False:
                return 10
            cb = self.widget.controls
            simulate.combobox_run_through_all(cb.attr_color)
            simulate.combobox_run_through_all(cb.attr_size)
        data = datasets.missing_data_1()
        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        combobox_run_through_all()
        self.send_signal(self.widget.Inputs.data, None)
        combobox_run_through_all()
        with data.unlocked():
            data.X[:, 0] = np.nan
            data.Y[:] = np.nan
            data.metas[:, 1] = np.nan
        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        combobox_run_through_all()

    @patch('Orange.projection.MDS.__call__', Mock(side_effect=MemoryError))
    def test_out_of_memory(self):
        if False:
            print('Hello World!')
        with patch('sys.excepthook', Mock()) as hook:
            self.send_signal(self.widget.Inputs.data, self.data, wait=1000)
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.out_of_memory.is_shown())

    @patch('Orange.projection.MDS.__call__', Mock(side_effect=ValueError))
    def test_other_error(self):
        if False:
            return 10
        with patch('sys.excepthook', Mock()) as hook:
            self.send_signal(self.widget.Inputs.data, self.data, wait=1000)
            hook.assert_not_called()
            self.assertTrue(self.widget.Error.optimization_error.is_shown())

    def test_matrix_not_symmetric(self):
        if False:
            return 10
        widget = self.widget
        self.send_signal(self.widget.Inputs.distances, DistMatrix([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(widget.Error.matrix_not_symmetric.is_shown())
        self.send_signal(self.widget.Inputs.distances, None)
        self.assertFalse(widget.Error.matrix_not_symmetric.is_shown())

    def test_matrix_too_small(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.widget
        self.send_signal(self.widget.Inputs.distances, DistMatrix([[1]]))
        self.assertTrue(widget.Error.matrix_too_small.is_shown())
        self.send_signal(self.widget.Inputs.distances, None)
        self.assertFalse(widget.Error.matrix_too_small.is_shown())

    def test_distances_without_data_0(self):
        if False:
            print('Hello World!')
        '\n        Only distances and no data.\n        GH-2335\n        '
        signal_data = Euclidean(self.data, axis=0)
        signal_data.row_items = None
        self.send_signal(self.widget.Inputs.distances, signal_data)

    def test_distances_without_data_1(self):
        if False:
            print('Hello World!')
        '\n        Only distances and no data.\n        GH-2335\n        '
        signal_data = Euclidean(self.data, axis=1)
        signal_data.row_items = None
        self.send_signal(self.widget.Inputs.distances, signal_data)

    def test_small_data(self):
        if False:
            return 10
        data = self.data[:1]
        self.assertFalse(self.widget.Error.not_enough_rows.is_shown())
        self.send_signal(self.widget.Inputs.data, data)

    def test_run(self):
        if False:
            print('Hello World!')
        self.send_signal(self.widget.Inputs.data, self.data)
        self.widget.run_button.click()
        self.widget.initialization = 0
        self.widget._OWMDS__invalidate_embedding()

    @WidgetTest.skipNonEnglish
    def test_migrate_settings_from_version_1(self):
        if False:
            i = 10
            return i + 15
        context_settings = [Context(attributes={'iris': 1, 'petal length': 2, 'petal width': 2, 'sepal length': 2, 'sepal width': 2}, metas={}, ordered_domain=[('sepal length', 2), ('sepal width', 2), ('petal length', 2), ('petal width', 2), ('iris', 1)], time=1500000000, values={'__version__': 1, 'color_value': ('iris', 1), 'shape_value': ('iris', 1), 'size_value': ('Stress', -2), 'label_value': ('sepal length', 2)})]
        settings = {'__version__': 1, 'autocommit': False, 'connected_pairs': 5, 'initialization': 0, 'jitter': 0.5, 'label_only_selected': True, 'legend_anchor': ((1, 0), (1, 0)), 'max_iter': 300, 'refresh_rate': 3, 'symbol_opacity': 230, 'symbol_size': 8, 'context_settings': context_settings, 'savedWidgetGeometry': None}
        w = self.create_widget(OWMDS, stored_settings=settings)
        domain = self.data.domain
        self.send_signal(w.Inputs.data, self.data, widget=w)
        g = w.graph
        for (a, value) in ((w.attr_color, domain['iris']), (w.attr_shape, domain['iris']), (w.attr_size, 'Stress'), (w.attr_label, domain['sepal length']), (g.label_only_selected, True), (g.alpha_value, 230), (g.point_width, 8), (g.jitter_size, 0.5)):
            self.assertEqual(a, value)
        self.assertFalse(w.auto_commit)

    def test_attr_label_from_dist_matrix_from_file(self):
        if False:
            for i in range(10):
                print('nop')
        w = self.widget
        w.start = Mock()
        row_items = self.towns.row_items
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertIn(row_items.domain['label'], w.controls.attr_label.model())
        self.towns.row_items = None
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertEqual(list(w.controls.attr_label.model()), [None])
        self.send_signal(w.Inputs.distances, None)
        self.assertEqual(list(w.controls.attr_label.model()), [None])
        self.towns.row_items = row_items
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertIn(row_items.domain['label'], w.controls.attr_label.model())
        self.towns.row_items = None
        self.send_signal(w.Inputs.distances, self.towns)
        self.assertEqual(list(w.controls.attr_label.model()), [None])

    def test_attr_label_from_dist_matrix_from_data(self):
        if False:
            print('Hello World!')
        w = self.widget
        w.start = Mock()
        data = Table('zoo')
        dist = Euclidean(data)
        self.send_signal(w.Inputs.distances, dist)
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas)) < set(w.controls.attr_label.model()))

    def test_attr_label_from_data(self):
        if False:
            i = 10
            return i + 15
        w = self.widget
        w.start = Mock()
        data = Table('zoo')
        dist = Euclidean(data)
        self.send_signal(w.Inputs.distances, dist)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas)) < set(w.controls.attr_label.model()))

    def test_attr_label_matrix_and_data(self):
        if False:
            for i in range(10):
                print('nop')
        w = self.widget
        w.start = Mock()
        data = Table('zoo')
        dist = Euclidean(data)
        self.send_signal(w.Inputs.distances, dist)
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas)) < set(w.controls.attr_label.model()))
        self.send_signal(w.Inputs.distances, None)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas)) < set(w.controls.attr_label.model()))
        self.send_signal(w.Inputs.data, None)
        self.assertEqual(list(w.controls.attr_label.model()), [None])
        self.send_signal(w.Inputs.data, data)
        self.assertTrue(set(chain(data.domain.variables, data.domain.metas)) < set(w.controls.attr_label.model()))

    def test_saved_matrix_and_data(self):
        if False:
            i = 10
            return i + 15
        towns_data = self.towns.row_items
        attr_label = self.widget.controls.attr_label
        self.widget.start = Mock()
        self.towns.row_items = None
        self.send_signal(self.widget.Inputs.distances, self.towns)
        self.assertIsNotNone(self.widget.graph.scatterplot_item)
        self.assertEqual(list(attr_label.model()), [None])
        self.send_signal(self.widget.Inputs.data, towns_data)
        self.assertIn(towns_data.domain['label'], attr_label.model())

    def test_matrix_columns_tooltip(self):
        if False:
            for i in range(10):
                print('nop')
        dist = Euclidean(self.data, axis=0)
        self.send_signal(self.widget.Inputs.distances, dist)
        self.assertIn('sepal length', self.widget.get_tooltip([0]))

    def test_matrix_columns_labels(self):
        if False:
            print('Hello World!')
        dist = Euclidean(self.data, axis=0)
        self.send_signal(self.widget.Inputs.distances, dist)
        simulate.combobox_activate_index(self.widget.controls.attr_label, 2)

    def test_matrix_columns_default_label(self):
        if False:
            print('Hello World!')
        dist = Euclidean(self.data, axis=0)
        self.send_signal(self.widget.Inputs.distances, dist)
        label_text = self.widget.controls.attr_label.currentText()
        self.assertEqual(label_text, 'labels')

    def test_update_stress(self):
        if False:
            i = 10
            return i + 15
        w = self.widget
        w.effective_matrix = np.array([[0, 4, 1], [4, 0, 1], [1, 1, 0]])
        w.embedding = np.array([[0, 0], [0, 3], [4, 3]])
        w.update_stress()
        expected = np.sqrt(52 / 36)
        self.assertAlmostEqual(w._compute_stress(), expected)
        self.assertIn(f'{expected:.3f}', w.stress_label.text())
        w.embedding = None
        w.update_stress()
        self.assertIsNone(w._compute_stress())
        self.assertIn('-', w.stress_label.text())

class TestOWMDSRunner(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.data = Table('iris')
        cls.distances = Euclidean(cls.data)
        cls.init = torgerson(cls.distances)
        cls.args = (cls.distances, 300, 25, 0, cls.init)

    def test_Result(self):
        if False:
            for i in range(10):
                print('nop')
        result = Result(embedding=self.init)
        self.assertIsInstance(result.embedding, np.ndarray)

    def test_run_mds(self):
        if False:
            return 10
        state = Mock()
        state.is_interruption_requested.return_value = False
        result = run_mds(*self.args + (state,))
        array = np.array([[-2.69280967, 0.32544313], [-2.72409383, -0.21287617], [-2.9022707, -0.13465859], [-2.75267253, -0.33899134], [-2.74108069, 0.35393209]])
        np.testing.assert_almost_equal(array, result.embedding[:5])
        state.set_status.assert_called_once_with('Running...')
        self.assertGreater(state.set_partial_result.call_count, 2)
        self.assertGreater(state.set_progress_value.call_count, 2)

    def test_run_do_not_modify_model_inplace(self):
        if False:
            i = 10
            return i + 15
        state = Mock()
        state.is_interruption_requested.return_value = True
        result = run_mds(*self.args + (state,))
        state.set_partial_result.assert_called_once()
        self.assertIsNot(self.init, result.embedding)
        self.assertTrue((self.init != result.embedding).any())
if __name__ == '__main__':
    unittest.main()