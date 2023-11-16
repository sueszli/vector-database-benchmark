from math import isnan
import unittest
from unittest.mock import patch
import numpy as np
from AnyQt.QtCore import QEvent, QPointF, Qt
from AnyQt.QtGui import QMouseEvent
from Orange.data import ContinuousVariable, DiscreteVariable, Domain, Table
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.widgets.tests.utils import simulate
from Orange.widgets.visualize.owsieve import OWSieveDiagram
from Orange.widgets.visualize.owsieve import ChiSqStats
from Orange.widgets.visualize.owsieve import Discretize
from Orange.widgets.widget import AttributeList

class TestOWSieveDiagram(WidgetTest, WidgetOutputsTestMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.signal_name = OWSieveDiagram.Inputs.data
        cls.signal_data = cls.data
        cls.titanic = Table('titanic')
        cls.iris = Table('iris')

    def setUp(self):
        if False:
            return 10
        self.widget = self.create_widget(OWSieveDiagram)

    def test_context_settings(self):
        if False:
            return 10
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertEqual(self.widget.attr_x, self.titanic.domain.class_var)
        self.assertEqual(self.widget.attr_y, self.titanic.domain.attributes[0])
        self.widget.attr_x = self.titanic.domain.attributes[1]
        self.widget.attr_y = self.titanic.domain.attributes[2]
        self.send_signal(self.widget.Inputs.data, None)
        self.assertEqual(self.widget.attr_x, None)
        self.assertEqual(self.widget.attr_y, None)
        self.assertIsNone(self.widget.discrete_data)
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertEqual(self.widget.attr_x, self.titanic.domain.attributes[1])
        self.assertEqual(self.widget.attr_y, self.titanic.domain.attributes[2])

    def test_continuous_data(self):
        if False:
            print('Hello World!')
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(self.widget.attr_x, self.iris.domain.class_var)
        self.assertEqual(self.widget.attr_y, self.iris.domain.attributes[0])
        self.assertTrue(self.widget.discrete_data.domain[0].is_discrete)

    def test_few_attributes(self):
        if False:
            i = 10
            return i + 15
        attr2 = self.titanic.domain[:2]
        domain2 = Domain(attr2)
        data2 = self.titanic.transform(domain2)
        self.send_signal(self.widget.Inputs.data, data2)
        attr1 = self.titanic.domain[:1]
        domain1 = Domain(attr1)
        data1 = self.titanic.transform(domain1)
        self.send_signal(self.widget.Inputs.data, data1)
        attr0 = self.titanic.domain[:0]
        domain0 = Domain(attr0)
        data0 = self.titanic.transform(domain0)
        self.send_signal(self.widget.Inputs.data, data0)

    def _select_data(self):
        if False:
            while True:
                i = 10
        (self.widget.attr_x, self.widget.attr_y) = self.data.domain[:2]
        area = self.widget.areas[0]
        self.widget.select_area(area, QMouseEvent(QEvent.MouseButtonPress, QPointF(), Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))
        return [0, 4, 6, 7, 11, 17, 19, 21, 22, 24, 26, 39, 40, 43, 44, 46]

    def test_missing_values(self):
        if False:
            print('Hello World!')
        'Check widget for dataset with missing values'
        attrs = [DiscreteVariable('c1', ['a', 'b', 'c'])]
        class_var = DiscreteVariable('cls', [])
        X = np.array([1, 2, 0, 1, 0, 2])[:, None]
        data = Table(Domain(attrs, class_var), X, np.array([np.nan] * 6))
        self.send_signal(self.widget.Inputs.data, data)

    def test_single_line(self):
        if False:
            i = 10
            return i + 15
        '\n        Check if it works when a table has only one row or duplicates.\n        Discretizer must have remove_const set to False.\n        '
        data = self.titanic[0:1]
        self.send_signal(self.widget.Inputs.data, data)

    def test_chisquare(self):
        if False:
            return 10
        '\n        Check if it can calculate chi square when there are no attributes\n        which suppose to be.\n        '
        a = DiscreteVariable('a', values=('y', 'n'))
        b = DiscreteVariable('b', values=('y', 'n', 'o'))
        table = Table.from_list(Domain([a, b]), list(zip('yynny', 'ynyyn')))
        chi = ChiSqStats(table, 0, 1)
        self.assertFalse(isnan(chi.chisq))

    def test_metadata(self):
        if False:
            print('Hello World!')
        '\n        Widget should interpret meta data which are continuous or discrete in\n        the same way as features or target. However still one variable should\n        be target or feature.\n        '
        table = Table.from_list(Domain([], [], [ContinuousVariable('a'), DiscreteVariable('b', values=('y', 'n'))]), list(zip([42.48, 16.84, 15.23, 23.8], 'yynn')))
        with patch('Orange.widgets.visualize.owsieve.Discretize', wraps=Discretize) as disc:
            self.send_signal(self.widget.Inputs.data, table)
            self.assertTrue(disc.called)
        metas = self.widget.discrete_data.domain.metas
        self.assertEqual(len(metas), 2)
        self.assertTrue(all((attr.is_discrete for attr in metas)))

    def test_sparse_data(self):
        if False:
            return 10
        '\n        Sparse support.\n        '
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.assertEqual(len(self.widget.discrete_data.domain.variables), len(self.iris.domain.variables))
        output = self.get_output(self.widget.Inputs.data)
        self.assertFalse(output.is_sparse())
        table = self.iris.to_sparse()
        self.send_signal(self.widget.Inputs.data, table)
        self.assertEqual(len(self.widget.discrete_data.domain.variables), 2)
        output = self.get_output(self.widget.Inputs.data)
        self.assertTrue(output.is_sparse())

    @patch('Orange.widgets.visualize.owsieve.SieveRank.on_manual_change')
    def test_vizrank_receives_manual_change(self, on_manual_change):
        if False:
            i = 10
            return i + 15
        self.widget = self.create_widget(OWSieveDiagram)
        data = Table('iris.tab')
        self.send_signal(self.widget.Inputs.data, data)
        model = self.widget.controls.attr_x.model()
        self.widget.attr_x = model[2]
        self.widget.attr_y = model[3]
        simulate.combobox_activate_index(self.widget.controls.attr_x, 4)
        call_args = on_manual_change.call_args[0]
        self.assertEqual(len(call_args), 2)
        self.assertEqual(call_args[0].name, data.domain[2].name)
        self.assertEqual(call_args[1].name, data.domain[1].name)

    def test_input_features(self):
        if False:
            while True:
                i = 10
        self.assertTrue(self.widget.attr_box.isEnabled())
        self.send_signal(self.widget.Inputs.data, self.iris)
        self.send_signal(self.widget.Inputs.features, AttributeList(self.iris.domain.attributes))
        self.assertFalse(self.widget.attr_box.isEnabled())
        self.assertFalse(self.widget.vizrank.isEnabled())
        self.send_signal(self.widget.Inputs.features, None)
        self.assertTrue(self.widget.attr_box.isEnabled())
        self.assertTrue(self.widget.vizrank.isEnabled())
if __name__ == '__main__':
    unittest.main()