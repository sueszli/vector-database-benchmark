import numpy as np
from AnyQt.QtCore import Qt, QItemSelection
from AnyQt.QtTest import QTest
from Orange.data import Table, Domain, ContinuousVariable, TimeVariable
from Orange.preprocess import impute
from Orange.widgets.data.owimpute import OWImpute, AsDefault, Learner, Method
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.widgets.utils.itemmodels import select_row

class Foo(Learner):

    def __call__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        1 / 0

class Bar:

    def __call__(self, args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        1 / 0

class FooBar(Learner):

    def __call__(self, data, *args, **kwargs):
        if False:
            return 10
        bar = Bar()
        bar.domain = data.domain
        return bar

class TestOWImpute(WidgetTest):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.widget = self.create_widget(OWImpute)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.widget.onDeleteWidget()
        super().tearDown()

    def test_empty_data(self):
        if False:
            for i in range(10):
                print('nop')
        'No crash on empty data'
        data = Table('iris')[::3]
        widget = self.widget
        widget.default_method_index = Method.Model
        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        imp_data = self.get_output(self.widget.Outputs.data)
        np.testing.assert_equal(imp_data.X, data.X)
        np.testing.assert_equal(imp_data.Y, data.Y)
        self.send_signal(self.widget.Inputs.data, Table.from_domain(data.domain), wait=1000)
        imp_data = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(imp_data), 0)
        data = data.transform(Domain([], [], data.domain.attributes))
        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        imp_data = self.get_output()
        self.assertEqual(len(imp_data), len(data))
        self.assertEqual(imp_data.domain, data.domain)
        np.testing.assert_equal(imp_data.metas, data.metas)

    def test_model_error(self):
        if False:
            for i in range(10):
                print('nop')
        widget = self.widget
        widget.default_method_index = Method.Model
        data = Table('brown-selected')[::4][:, :4]
        self.send_signal(self.widget.Inputs.data, data, wait=1000)
        self.send_signal(self.widget.Inputs.learner, Foo(), wait=1000)
        self.assertTrue(widget.Error.imputation_failed.is_shown())
        self.send_signal(self.widget.Inputs.learner, FooBar(), wait=1000)
        self.assertTrue(widget.Error.imputation_failed.is_shown())

    def test_select_method(self):
        if False:
            for i in range(10):
                print('nop')
        data = Table('iris')[::5]
        self.send_signal(self.widget.Inputs.data, data)
        widget = self.widget
        model = widget.varmodel
        view = widget.varview
        defbg = widget.default_button_group
        varbg = widget.variable_button_group
        self.assertSequenceEqual(list(model), data.domain.variables)
        defbg.button(Method.Average).click()
        self.assertEqual(widget.default_method_index, Method.Average)
        self.assertTrue(all((isinstance(m, AsDefault) and isinstance(m.method, impute.Average) for m in map(widget.get_method_for_column, range(len(data.domain.variables))))))
        select_row(view, 0)
        varbg.button(Method.Average).click()
        met = widget.get_method_for_column(0)
        self.assertIsInstance(met, impute.Average)
        selmodel = view.selectionModel()
        selmodel.select(model.index(2), selmodel.Select)
        self.assertEqual(varbg.checkedId(), -1)
        varbg.button(Method.Leave).click()
        self.assertIsInstance(widget.get_method_for_column(0), impute.DoNotImpute)
        self.assertIsInstance(widget.get_method_for_column(2), impute.DoNotImpute)
        varbg.button(Method.AsAboveSoBelow).click()
        self.assertIsInstance(widget.get_method_for_column(0), AsDefault)
        self.assertIsInstance(widget.get_method_for_column(2), AsDefault)

    def test_overall_default(self):
        if False:
            print('Hello World!')
        domain = Domain([ContinuousVariable(f'c{i}') for i in range(3)] + [TimeVariable(f't{i}') for i in range(3)], [])
        n = np.nan
        x = np.array([[1, 2, n, 1000, n, n], [2, n, 1, n, 2000, 2000]])
        data = Table(domain, x, np.empty((2, 0)))
        widget = self.widget
        widget.default_numeric_value = 3.14
        widget.default_time = 42
        widget.default_method_index = Method.Default
        self.send_signal(self.widget.Inputs.data, data)
        imp_data = self.get_output(self.widget.Outputs.data)
        np.testing.assert_almost_equal(imp_data.X, [[1, 2, 3.14, 1000, 42, 42], [2, 3.14, 1, 42, 2000, 2000]])
        widget.numeric_value_widget.setValue(100)
        QTest.keyClick(widget.numeric_value_widget, Qt.Key_Enter)
        self.assertEqual(widget.default_numeric_value, 100)

    def test_value_edit(self):
        if False:
            return 10
        data = Table('heart_disease')[::10]
        self.send_signal(self.widget.Inputs.data, data)
        widget = self.widget
        model = widget.varmodel
        view = widget.varview
        selmodel = view.selectionModel()
        varbg = widget.variable_button_group

        def selectvars(varlist, command=selmodel.ClearAndSelect):
            if False:
                return 10
            indices = [data.domain.index(var) for var in varlist]
            itemsel = QItemSelection()
            for ind in indices:
                midx = model.index(ind)
                itemsel.select(midx, midx)
            selmodel.select(itemsel, command)

        def effective_method(var):
            if False:
                while True:
                    i = 10
            return widget.get_method_for_column(data.domain.index(var))
        selectvars(['chest pain'])
        self.assertTrue(widget.value_combo.isVisibleTo(widget) and widget.value_combo.isEnabledTo(widget))
        self.assertEqual(varbg.checkedId(), Method.AsAboveSoBelow)
        simulate.combobox_activate_item(widget.value_combo, data.domain['chest pain'].values[1])
        self.assertEqual(varbg.checkedId(), Method.Default)
        imputer = effective_method('chest pain')
        self.assertIsInstance(imputer, impute.Default)
        self.assertEqual(imputer.default, 1)
        selectvars(['rest SBP', 'cholesterol'])
        self.assertTrue(widget.value_double.isVisibleTo(widget) and widget.value_double.isEnabledTo(widget))
        self.assertEqual(varbg.checkedId(), Method.AsAboveSoBelow)
        widget.value_double.setValue(-1.0)
        QTest.keyClick(self.widget.value_double, Qt.Key_Enter)
        self.assertEqual(varbg.checkedId(), Method.Default)
        imputer = effective_method('rest SBP')
        self.assertIsInstance(imputer, impute.Default)
        self.assertEqual(imputer.default, -1.0)
        imputer = effective_method('cholesterol')
        self.assertIsInstance(imputer, impute.Default)
        self.assertEqual(imputer.default, -1.0)
        selectvars(['chest pain'], selmodel.Select)
        self.assertEqual(varbg.checkedId(), -1)
        self.assertFalse(widget.value_combo.isEnabledTo(widget) and widget.value_double.isEnabledTo(widget))
        selectvars(['chest pain'])
        self.assertTrue(widget.value_combo.isVisibleTo(widget) and widget.value_combo.isEnabledTo(widget))
        self.assertEqual(varbg.checkedId(), Method.Default)
        self.assertEqual(widget.value_combo.currentIndex(), 1)