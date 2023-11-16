import unittest
from unittest.mock import patch, Mock
import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import svd_flip
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.preprocess import preprocess
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import table_dense_sparse, possible_duplicate_table
from Orange.widgets.unsupervised.owpca import OWPCA
from Orange.tests import test_filename

class TestOWPCA(WidgetTest):

    def setUp(self):
        if False:
            print('Hello World!')
        self.widget = self.create_widget(OWPCA)
        self.iris = Table('iris')

    def test_set_variance100(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget.set_data(self.iris)
        self.widget.variance_covered = 100
        self.widget._update_selection_variance_spin()

    def test_constant_data(self):
        if False:
            print('Hello World!')
        data = self.iris[::5].copy()
        with data.unlocked():
            data.X[:, :] = 1.0
        with np.errstate(invalid='ignore'):
            self.send_signal(self.widget.Inputs.data, data, wait=5000)
        self.assertTrue(self.widget.Warning.trivial_components.is_shown())
        self.assertIsNone(self.get_output(self.widget.Outputs.transformed_data))
        self.assertIsNone(self.get_output(self.widget.Outputs.components))

    def test_empty_data(self):
        if False:
            return 10
        ' Check widget for dataset with no rows and for dataset with no attributes '
        self.send_signal(self.widget.Inputs.data, self.iris[:0])
        self.assertTrue(self.widget.Error.no_instances.is_shown())
        domain = Domain([], None, self.iris.domain.variables)
        new_data = Table.from_table(domain, self.iris)
        self.send_signal(self.widget.Inputs.data, new_data)
        self.assertTrue(self.widget.Error.no_features.is_shown())
        self.assertFalse(self.widget.Error.no_instances.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.no_features.is_shown())

    def test_limit_components(self):
        if False:
            print('Hello World!')
        X = np.random.RandomState(0).rand(101, 101)
        data = Table.from_numpy(None, X)
        self.widget.ncomponents = 100
        self.send_signal(self.widget.Inputs.data, data, wait=5000)
        tran = self.get_output(self.widget.Outputs.transformed_data)
        self.assertEqual(len(tran.domain.attributes), 100)
        self.widget.ncomponents = 101
        with self.assertRaises(IndexError):
            self.widget._setup_plot()

    def test_migrate_settings_limits_components(self):
        if False:
            for i in range(10):
                print('nop')
        settings = {'ncomponents': 10}
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings['ncomponents'], 10)
        settings = {'ncomponents': 101}
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings['ncomponents'], 100)

    def test_migrate_settings_changes_variance_covered_to_int(self):
        if False:
            while True:
                i = 10
        settings = {'variance_covered': 17.5}
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings['variance_covered'], 17)
        settings = {'variance_covered': float('nan')}
        OWPCA.migrate_settings(settings, 0)
        self.assertEqual(settings['variance_covered'], 100)

    def test_variance_shown(self):
        if False:
            while True:
                i = 10
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.widget.maxp = 2
        self.widget._setup_plot()
        self.wait_until_finished()
        var2 = self.widget.variance_covered
        self.widget.ncomponents = 3
        self.widget._update_selection_component_spin()
        self.wait_until_finished()
        var3 = self.widget.variance_covered
        self.assertGreater(var3, var2)

    def test_unique_domain_components(self):
        if False:
            while True:
                i = 10
        table = possible_duplicate_table('components')
        self.send_signal(self.widget.Inputs.data, table)
        out = self.get_output(self.widget.Outputs.components)
        self.assertEqual(out.domain.metas[0].name, 'components (1)')

    def test_variance_attr(self):
        if False:
            print('Hello World!')
        self.widget.ncomponents = 2
        self.send_signal(self.widget.Inputs.data, self.iris, wait=5000)
        self.wait_until_stop_blocking()
        self.widget._variance_ratio = np.array([0.5, 0.25, 0.2, 0.05])
        self.widget.commit.now()
        self.wait_until_finished()
        result = self.get_output(self.widget.Outputs.transformed_data)
        (pc1, pc2) = result.domain.attributes
        self.assertEqual(pc1.attributes['variance'], 0.5)
        self.assertEqual(pc2.attributes['variance'], 0.25)
        result = self.get_output(self.widget.Outputs.data)
        (pc1, pc2) = result.domain.metas
        self.assertEqual(pc1.attributes['variance'], 0.5)
        self.assertEqual(pc2.attributes['variance'], 0.25)
        result = self.get_output(self.widget.Outputs.components)
        np.testing.assert_almost_equal(result.get_column('variance'), [0.5, 0.25])

    def test_sparse_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Check that PCA returns the same results for both dense and sparse data.'
        (dense_data, sparse_data) = (self.iris, self.iris.to_sparse())

        def _compute_projection(data):
            if False:
                i = 10
                return i + 15
            self.send_signal(self.widget.Inputs.data, data)
            self.wait_until_stop_blocking()
            result = self.get_output(self.widget.Outputs.transformed_data)
            self.send_signal(self.widget.Inputs.data, None)
            return result
        self.widget.controls.normalize.setChecked(False)
        dense_pca = _compute_projection(dense_data)
        sparse_pca = _compute_projection(sparse_data)
        np.testing.assert_almost_equal(dense_pca.X, sparse_pca.X)
        self.widget.controls.normalize.setChecked(True)
        dense_pca = _compute_projection(dense_data)
        sparse_pca = _compute_projection(sparse_data)
        np.testing.assert_almost_equal(dense_pca.X, sparse_pca.X)

    def test_all_components_continuous(self):
        if False:
            i = 10
            return i + 15
        data = Table(test_filename('datasets/cyber-security-breaches.tab'))
        self.assertTrue(any((isinstance(a, TimeVariable) for a in data.domain.attributes)))
        self.widget.normalize = False
        self.widget._update_normalize()
        self.widget.set_data(data)
        components = self.get_output(self.widget.Outputs.components)
        self.assertTrue(all((type(a) is ContinuousVariable for a in components.domain.attributes)), "Some variables aren't of type ContinuousVariable")

    @table_dense_sparse
    def test_normalize_data(self, prepare_table):
        if False:
            for i in range(10):
                print('nop')
        'Check that normalization is called at the proper times.'
        data = prepare_table(self.iris)
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.controls.normalize.isChecked())
        with patch.object(preprocess.Normalize, '__call__', wraps=lambda x: x) as normalize:
            self.send_signal(self.widget.Inputs.data, data, wait=5000)
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_called_once()
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.controls.normalize.isChecked())
        with patch.object(preprocess.Normalize, '__call__', wraps=lambda x: x) as normalize:
            self.send_signal(self.widget.Inputs.data, data, wait=5000)
            self.wait_until_stop_blocking()
            self.assertTrue(self.widget.controls.normalize.isEnabled())
            normalize.assert_not_called()

    @table_dense_sparse
    def test_normalization_variance(self, prepare_table):
        if False:
            for i in range(10):
                print('nop')
        data = prepare_table(self.iris)
        self.widget.ncomponents = 2
        self.widget.controls.normalize.setChecked(True)
        self.assertTrue(self.widget.normalize)
        self.send_signal(self.widget.Inputs.data, data, wait=5000)
        self.wait_until_stop_blocking()
        variance_normalized = self.widget.variance_covered
        self.widget.controls.normalize.setChecked(False)
        self.assertFalse(self.widget.normalize)
        self.wait_until_finished()
        self.wait_until_stop_blocking()
        variance_unnormalized = self.widget.variance_covered
        self.assertLess(variance_normalized, variance_unnormalized)

    @table_dense_sparse
    def test_normalized_gives_correct_result(self, prepare_table):
        if False:
            print('Hello World!')
        'Make sure that normalization through widget gives correct result.'
        random_state = check_random_state(42)
        mask = random_state.beta(1, 2, size=self.iris.X.shape) > 0.5
        with self.iris.unlocked():
            self.iris.X[mask] = 0
        data = prepare_table(self.iris)
        self.widget.controls.normalize.setChecked(True)
        self.send_signal(self.widget.Inputs.data, data)
        self.wait_until_stop_blocking()
        widget_result = self.get_output(self.widget.Outputs.transformed_data)
        x = self.iris.X
        x = (x - x.mean(0)) / x.std(0)
        (U, S, Va) = np.linalg.svd(x)
        (U, S, Va) = (U[:, :2], S[:2], Va[:2])
        (U, Va) = svd_flip(U, Va)
        pca_embedding = U * S
        np.testing.assert_almost_equal(widget_result.X, pca_embedding)

    def test_do_not_mask_features(self):
        if False:
            i = 10
            return i + 15
        data = Table('iris.tab')
        self.widget.set_data(data)
        ndata = Table('iris.tab')
        self.assertEqual(data.domain[0], ndata.domain[0])

    def test_on_cut_changed(self):
        if False:
            while True:
                i = 10
        widget = self.widget
        widget.ncomponents = 2
        invalidate = widget._invalidate_selection = Mock()
        widget._on_cut_changed(2)
        invalidate.assert_not_called()
        widget._on_cut_changed(3)
        invalidate.assert_called()
        widget.ncomponents = 0
        invalidate.reset_mock()
        widget._on_cut_changed(1)
        invalidate.assert_not_called()
        self.assertEqual(widget.ncomponents, 0)

    def test_output_data(self):
        if False:
            i = 10
            return i + 15
        widget = self.widget
        widget.ncomponents = 2
        domain = Domain(self.iris.domain.attributes[:3], self.iris.domain.class_var, self.iris.domain.attributes[3:])
        iris = self.iris.transform(domain)
        self.send_signal(widget.Inputs.data, iris)
        output = self.get_output(widget.Outputs.data)
        outdom = output.domain
        self.assertEqual(domain.attributes, outdom.attributes)
        self.assertEqual(domain.class_var, outdom.class_var)
        self.assertEqual(domain.metas, outdom.metas[:1])
        self.assertEqual(len(outdom.metas), 3)
        np.testing.assert_equal(iris.X, output.X)
        np.testing.assert_equal(iris.Y, output.Y)
        np.testing.assert_equal(iris.metas[:, 0], output.metas[:, 0])
        trans = self.get_output(widget.Outputs.transformed_data)
        self.assertEqual(trans.domain.attributes, outdom.metas[1:])
        np.testing.assert_equal(trans.X, output.metas[:, 1:])
        self.send_signal(widget.Inputs.data, None)
        output = self.get_output(widget.Outputs.data)
        self.assertIsNone(output)

    def test_table_subclass(self):
        if False:
            i = 10
            return i + 15
        "\n        When input table is instance of Table's subclass (e.g. Corpus) resulting\n        tables should also be an instance subclasses\n        "

        class TableSub(Table):
            pass
        table_subclass = TableSub(self.iris)
        self.send_signal(self.widget.Inputs.data, table_subclass)
        data_out = self.get_output(self.widget.Outputs.data)
        trans_data_out = self.get_output(self.widget.Outputs.transformed_data)
        self.assertIsInstance(data_out, TableSub)
        self.assertIsInstance(trans_data_out, TableSub)
if __name__ == '__main__':
    unittest.main()