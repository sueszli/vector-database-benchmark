import warnings
import numpy as np
from AnyQt.QtCore import QPoint, Qt
from AnyQt.QtTest import QTest
import Orange.misc
from Orange.data import Table, Domain, ContinuousVariable, DiscreteVariable
from Orange.distance import Euclidean
from Orange.misc import DistMatrix
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.unsupervised.owhierarchicalclustering import OWHierarchicalClustering

class TestOWHierarchicalClustering(WidgetTest, WidgetOutputsTestMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.distances = Euclidean(cls.data)
        cls.signal_name = OWHierarchicalClustering.Inputs.distances
        cls.signal_data = cls.distances
        cls.same_input_output_domain = False
        cls.distances_cols = Euclidean(cls.data, axis=0)

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget = self.create_widget(OWHierarchicalClustering)

    def _select_data(self):
        if False:
            while True:
                i = 10
        items = self.widget.dendrogram._items
        cluster = items[sorted(list(items.keys()))[4]]
        self.widget.dendrogram.set_selected_items([cluster])
        return [14, 15, 32, 33]

    def _select_data_columns(self):
        if False:
            for i in range(10):
                print('nop')
        items = self.widget.dendrogram._items
        cluster = items[sorted(list(items.keys()))[5]]
        self.widget.dendrogram.set_selected_items([cluster])

    def _compare_selected_annotated_domains(self, selected, annotated):
        if False:
            print('Hello World!')
        self.assertEqual(annotated.domain.variables, selected.domain.variables)
        self.assertNotIn('Other', selected.domain.metas[0].values)
        self.assertIn('Other', annotated.domain.metas[0].values)
        self.assertLess(set((var.name for var in selected.domain.metas)), set((var.name for var in annotated.domain.metas)))

    def test_selection_box_output(self):
        if False:
            for i in range(10):
                print('nop')
        'Check output if Selection method changes'
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))
        self.widget.selection_box.buttons[1].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))
        self.widget.selection_box.buttons[2].click()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.assertIsNotNone(self.get_output(self.widget.Outputs.annotated_data))

    def test_all_zero_inputs(self):
        if False:
            while True:
                i = 10
        d = Orange.misc.DistMatrix(np.zeros((10, 10)))
        self.widget.set_distances(d)

    def test_annotation_settings_retrieval(self):
        if False:
            for i in range(10):
                print('nop')
        'Check whether widget retrieves correct settings for annotation'
        widget = self.widget
        dist_names = Orange.misc.DistMatrix(np.zeros((4, 4)), self.data, axis=0)
        dist_no_names = Orange.misc.DistMatrix(np.zeros((10, 10)), axis=1)
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertEqual(widget.annotation, self.data.domain.class_var)
        var2 = self.data.domain[2]
        widget.annotation = var2
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, 'Enumeration')
        widget.annotation = 'None'
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, 'None')
        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, 'Name')
        widget.annotation = 'Enumeration'
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIs(widget.annotation, var2)
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, 'None')
        self.send_signal(self.widget.Inputs.distances, dist_names)
        self.assertEqual(widget.annotation, 'Enumeration')
        self.send_signal(self.widget.Inputs.distances, dist_no_names)
        self.assertEqual(widget.annotation, 'None')

    def test_domain_loses_class(self):
        if False:
            print('Hello World!')
        widget = self.widget
        self.send_signal(self.widget.Inputs.distances, self.distances)
        data = self.data[:, :4]
        distances = Euclidean(data)
        self.send_signal(self.widget.Inputs.distances, distances)

    def test_infinite_distances(self):
        if False:
            while True:
                i = 10
        '\n        Scipy does not accept infinite distances and neither does this widget.\n        Error is shown.\n        GH-2380\n        '
        table = Table.from_list(Domain([ContinuousVariable('a')], [DiscreteVariable('b', values=('y',))]), list(zip([1.79e+308, -1e+120], 'yy')))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', '.*', RuntimeWarning)
            distances = Euclidean(table)
        self.assertFalse(self.widget.Error.not_finite_distances.is_shown())
        self.send_signal(self.widget.Inputs.distances, distances)
        self.assertTrue(self.widget.Error.not_finite_distances.is_shown())
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertFalse(self.widget.Error.not_finite_distances.is_shown())

    def test_not_symmetric(self):
        if False:
            print('Hello World!')
        w = self.widget
        self.send_signal(w.Inputs.distances, DistMatrix([[1, 2, 3], [4, 5, 6]]))
        self.assertTrue(w.Error.not_symmetric.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.not_symmetric.is_shown())

    def test_empty_matrix(self):
        if False:
            while True:
                i = 10
        w = self.widget
        self.send_signal(w.Inputs.distances, DistMatrix([[]]))
        self.assertTrue(w.Error.empty_matrix.is_shown())
        self.send_signal(w.Inputs.distances, None)
        self.assertFalse(w.Error.empty_matrix.is_shown())

    def test_output_cut_ratio(self):
        if False:
            print('Hello World!')
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertIsNotNone(annotated)
        self.widget.grab()
        QTest.mousePress(self.widget.view.headerView().viewport(), Qt.LeftButton, Qt.NoModifier, QPoint(100, 10))
        selected = self.get_output(self.widget.Outputs.selected_data)
        annotated = self.get_output(self.widget.Outputs.annotated_data)
        self.assertEqual(len(selected), len(self.data))
        self.assertIsNotNone(annotated)

    def test_retain_selection(self):
        if False:
            while True:
                i = 10
        "Hierarchical Clustering didn't retain selection. GH-1563"
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self._select_data()
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self.assertIsNotNone(self.get_output(self.widget.Outputs.selected_data))

    def test_restore_state(self):
        if False:
            for i in range(10):
                print('nop')
        self.send_signal(self.widget.Inputs.distances, self.distances)
        self._select_data()
        ids_1 = self.get_output(self.widget.Outputs.selected_data).ids
        state = self.widget.settingsHandler.pack_data(self.widget)
        w = self.create_widget(OWHierarchicalClustering, stored_settings=state)
        self.send_signal(w.Inputs.distances, self.distances, widget=w)
        ids_2 = self.get_output(w.Outputs.selected_data, widget=w).ids
        self.assertSequenceEqual(list(ids_1), list(ids_2))

    def test_column_distances(self):
        if False:
            for i in range(10):
                print('nop')
        self.send_signal(self.widget.Inputs.distances, self.distances_cols)
        self._select_data_columns()
        o = self.get_output(self.widget.Outputs.annotated_data)
        annotated = [(a.name, a.attributes['cluster']) for a in o.domain.attributes]
        self.assertEqual(annotated, [('sepal width', 1), ('petal length', 1), ('sepal length', 0), ('petal width', 0)])
        self.widget.selection_box.buttons[2].click()
        o = self.get_output(self.widget.Outputs.annotated_data)
        annotated = [(a.name, a.attributes['cluster']) for a in o.domain.attributes]
        self.assertEqual(annotated, [('sepal length', 1), ('petal width', 2), ('sepal width', 3), ('petal length', 3)])