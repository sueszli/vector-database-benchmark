from unittest import skip
from unittest.mock import patch, Mock
import numpy as np
from scipy import sparse
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate, possible_duplicate_table
from Orange.widgets.unsupervised.owmanifoldlearning import OWManifoldLearning

class TestOWManifoldLearning(WidgetTest):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        cls.iris = Table('iris')

    def setUp(self):
        if False:
            while True:
                i = 10
        self.widget = self.create_widget(OWManifoldLearning, stored_settings={'auto_apply': False})

    def click_apply(self):
        if False:
            return 10
        self.widget.apply_button.button.clicked.emit()

    def test_input_data(self):
        if False:
            for i in range(10):
                print('nop')
        "Check widget's data"
        self.assertEqual(self.widget.data, None)
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.data, self.iris)
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.data, None)

    def test_output_data(self):
        if False:
            while True:
                i = 10
        'Check if data is on output after apply'
        self.assertIsNone(self.get_output(self.widget.Outputs.transformed_data))
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.click_apply()
        self.assertIsInstance(self.get_output(self.widget.Outputs.transformed_data), Table)
        self.send_signal(self.widget.Inputs.data, None)
        self.click_apply()
        self.assertIsNone(self.get_output(self.widget.Outputs.transformed_data))

    def test_n_components(self):
        if False:
            for i in range(10):
                print('nop')
        'Check the output for various numbers of components'
        self.send_signal(self.widget.Inputs.data, self.iris)
        for i in range(self.widget.n_components_spin.minimum(), self.widget.n_components_spin.maximum()):
            self.assertEqual(self.widget.data, self.iris)
            self.widget.n_components_spin.setValue(i)
            self.widget.n_components_spin.onEnter()
            self.click_apply()
            self._compare_tables(self.get_output(self.widget.Outputs.transformed_data), i)

    def test_manifold_methods(self):
        if False:
            print('Hello World!')
        'Check output for various manifold methods'
        self.send_signal(self.widget.Inputs.data, self.iris)
        n_comp = self.widget.n_components
        for i in range(len(self.widget.MANIFOLD_METHODS)):
            self.assertEqual(self.widget.data, self.iris)
            self.widget.manifold_methods_combo.activated.emit(i)
            self.click_apply()
            self._compare_tables(self.get_output(self.widget.Outputs.transformed_data), n_comp)

    def _compare_tables(self, _output, n_components):
        if False:
            for i in range(10):
                print('nop')
        'Helper function for table comparison'
        self.assertEqual((len(self.iris), n_components), _output.X.shape)
        np.testing.assert_array_equal(self.iris.Y, _output.Y)
        np.testing.assert_array_equal(self.iris.metas, _output.metas)

    def test_sparse_data(self):
        if False:
            while True:
                i = 10
        data = Table('iris').to_sparse()
        self.assertTrue(sparse.issparse(data.X))

        def __callback():
            if False:
                i = 10
                return i + 15
            self.send_signal(self.widget.Inputs.data, data)
            self.click_apply()
            self.assertTrue(self.widget.Error.sparse_not_supported.is_shown())
            self.send_signal(self.widget.Inputs.data, None)
            self.click_apply()
            self.assertFalse(self.widget.Error.sparse_not_supported.is_shown())
        simulate.combobox_run_through_all(self.widget.manifold_methods_combo, callback=__callback)

    def test_metrics(self):
        if False:
            print('Hello World!')
        simulate.combobox_activate_item(self.widget.manifold_methods_combo, 't-SNE')

        def __callback():
            if False:
                return 10
            self.send_signal(self.widget.Inputs.data, self.iris)
            self.click_apply()
            self.assertFalse(self.widget.Error.manifold_error.is_shown())
            self.send_signal(self.widget.Inputs.data, None)
            self.click_apply()
            self.assertFalse(self.widget.Error.manifold_error.is_shown())
        simulate.combobox_run_through_all(self.widget.tsne_editor.controls.metric_index, callback=__callback)

    def test_unique_domain(self):
        if False:
            print('Hello World!')
        simulate.combobox_activate_item(self.widget.manifold_methods_combo, 'MDS')
        data = possible_duplicate_table('C0', class_var=True)
        self.send_signal(self.widget.Inputs.data, data)
        self.click_apply()
        out = self.get_output(self.widget.Outputs.transformed_data)
        self.assertTrue(out.domain.attributes[0], 'C0 (1)')

    @skip
    def test_singular_matrices(self):
        if False:
            print('Hello World!')
        '\n        Handle singular matrices.\n        GH-2228\n\n        TODO: This test makes sense with the ``Mahalanobis`` distance metric\n        which is currently not supported by tSNE. In case it is ever\n        re-introduced, this test is very much required.\n\n        '
        table = Table(Domain([ContinuousVariable('a'), ContinuousVariable('b')], class_vars=DiscreteVariable('c', values=('0', '1'))), list(zip([1, 1, 1], [0, 1, 2], [0, 1, 1])))
        self.send_signal(self.widget.Inputs.data, table)
        self.widget.manifold_methods_combo.activated.emit(0)
        self.widget.tsne_editor.metric_combo.activated.emit(4)
        self.assertFalse(self.widget.Error.manifold_error.is_shown())
        self.click_apply()
        self.assertTrue(self.widget.Error.manifold_error.is_shown())

    def test_out_of_memory(self):
        if False:
            print('Hello World!')
        '\n        Show error message when out of memory.\n        GH-2441\n        '
        table = Table('iris')
        with patch('Orange.projection.manifold.MDS.__call__', Mock()) as mock:
            mock.side_effect = MemoryError
            self.send_signal('Data', table)
            self.widget.manifold_methods_combo.activated.emit(1)
            self.click_apply()
            self.assertTrue(self.widget.Error.out_of_memory.is_shown())

    def test_unconditional_commit_on_new_signal(self):
        if False:
            for i in range(10):
                print('nop')
        with patch.object(self.widget.commit, 'now') as apply:
            self.widget.auto_apply = False
            apply.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.iris)
            apply.assert_called()

    @patch('Orange.widgets.unsupervised.owmanifoldlearning.OWManifoldLearning.report_items')
    def test_report(self, mocked_report: Mock):
        if False:
            return 10
        for i in range(len(self.widget.MANIFOLD_METHODS)):
            self.send_signal(self.widget.Inputs.data, self.iris)
            self.widget.manifold_methods_combo.activated.emit(i)
            self.wait_until_finished()
            self.widget.send_report()
            mocked_report.assert_called()
            self.assertEqual(mocked_report.call_count, 3)
            mocked_report.reset_mock()
            self.send_signal(self.widget.Inputs.data, None)
            self.widget.send_report()
            self.assertEqual(mocked_report.call_count, 2)
            mocked_report.reset_mock()