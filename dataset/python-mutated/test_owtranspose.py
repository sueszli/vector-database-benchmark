import unittest
from unittest.mock import patch, Mock
import numpy as np
from Orange.data import Table
from Orange.widgets.data.owtranspose import OWTranspose, run
from Orange.widgets.tests.base import WidgetTest
from Orange.widgets.tests.utils import simulate
from Orange.tests import test_filename

class TestRunner(WidgetTest):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.zoo = Table('zoo')
        self.state = Mock()
        self.state.is_interruption_requested = Mock(return_value=False)

    def test_run(self):
        if False:
            i = 10
            return i + 15
        result = run(self.zoo, '', 'Feature', False, self.state)
        self.assert_table_equal(Table.transpose(self.zoo), result)

    def test_run_var(self):
        if False:
            print('Hello World!')
        result = run(self.zoo, 'name', 'Feature', False, self.state)
        self.assert_table_equal(Table.transpose(self.zoo, 'name'), result)

    def test_run_name(self):
        if False:
            print('Hello World!')
        result1 = run(self.zoo, '', 'Foo', False, self.state)
        result2 = Table.transpose(self.zoo, feature_name='Foo')
        self.assert_table_equal(result1, result2)

    def test_run_callback(self):
        if False:
            print('Hello World!')
        self.state.set_progress_value = Mock()
        run(self.zoo, '', 'Feature', False, self.state)
        self.state.set_progress_value.assert_called()

class TestOWTranspose(WidgetTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.widget = self.create_widget(OWTranspose)
        self.zoo = Table('zoo')

    def test_transpose(self):
        if False:
            i = 10
            return i + 15
        widget = self.widget
        self.assertIsNone(self.get_output(widget.Outputs.data))
        self.assertEqual(widget.data, None)
        self.send_signal(widget.Inputs.data, self.zoo)
        output = self.get_output(widget.Outputs.data)
        transpose = Table.transpose(self.zoo)
        np.testing.assert_array_equal(output.X, transpose.X)
        np.testing.assert_array_equal(output.Y, transpose.Y)
        np.testing.assert_array_equal(output.metas, transpose.metas)
        self.send_signal(widget.Inputs.data, None)
        self.assertIsNone(self.get_output(widget.Outputs.data))

    def test_feature_type(self):
        if False:
            i = 10
            return i + 15
        widget = self.widget
        data = Table(test_filename('datasets/test_asn_data_working.csv'))
        metas = data.domain.metas
        widget.feature_type = widget.GENERIC
        self.send_signal(widget.Inputs.data, data)
        self.assertEqual(widget.feature_type, widget.FROM_VAR)
        self.assertIs(widget.feature_names_column, metas[0])
        output = self.get_output(widget.Outputs.data)
        self.assertListEqual([a.name for a in output.domain.attributes], [metas[0].to_val(m) for m in data.metas[:, 0]])
        widget.feature_names_column = metas[4]
        widget.commit.now()
        output = self.get_output(widget.Outputs.data)
        self.assertTrue(all((a.name.startswith(metas[1].to_val(m)) for (a, m) in zip(output.domain.attributes, data.metas[:, 4]))))
        self.assertEqual(widget.DEFAULT_PREFIX, 'Feature')
        widget.feature_type = widget.GENERIC
        widget.commit.now()
        output = self.get_output(widget.Outputs.data)
        self.assertTrue(all((x.name.startswith(widget.DEFAULT_PREFIX) for x in output.domain.attributes)))
        widget.feature_name = 'Foo'
        widget.commit.now()
        output = self.get_output(widget.Outputs.data)
        self.assertTrue(all((x.name.startswith('Foo ') for x in output.domain.attributes)))
        widget.feature_name = ''
        widget.commit.now()
        output = self.get_output(widget.Outputs.data)
        self.assertTrue(all((x.name.startswith(widget.DEFAULT_PREFIX) for x in output.domain.attributes)))

    def test_remove_redundant_instance(self):
        if False:
            i = 10
            return i + 15
        cb = self.widget.feature_combo
        data = Table('iris')
        self.send_signal(self.widget.Inputs.data, data)
        simulate.combobox_activate_item(cb, 'petal length')
        self.widget.controls.remove_redundant_inst.setChecked(True)
        self.wait_until_finished()
        output = self.get_output(self.widget.Outputs.data)
        self.assertEqual(len(output), 3)
        self.assertNotIn('petal length', output.metas)
        self.widget.controls.remove_redundant_inst.setChecked(False)
        self.wait_until_finished()
        self.assertEqual(len(self.get_output(self.widget.Outputs.data)), 4)

    def test_send_report(self):
        if False:
            print('Hello World!')
        widget = self.widget
        widget.feature_type = widget.FROM_VAR
        widget.report_button.click()
        widget.feature_type = widget.GENERIC
        widget.report_button.click()
        self.send_signal(widget.Inputs.data, self.zoo)
        widget.feature_type = widget.FROM_VAR
        widget.report_button.click()
        widget.feature_type = widget.GENERIC
        widget.report_button.click()

    def test_gui_behaviour(self):
        if False:
            i = 10
            return i + 15
        widget = self.widget
        widget.commit.now = unittest.mock.Mock()
        widget.auto_apply = False
        self.assertEqual(widget.feature_type, widget.GENERIC)
        self.assertFalse(widget.meta_button.isEnabled())
        self.assertFalse(widget.commit.now.called)
        self.send_signal(widget.Inputs.data, self.zoo)
        self.assertTrue(widget.meta_button.isEnabled())
        self.assertEqual(widget.feature_type, widget.FROM_VAR)
        self.assertIs(widget.feature_names_column, widget.feature_model[0])
        self.assertTrue(widget.commit.now.called)
        widget.commit.now.reset_mock()
        widget.controls.feature_name.editingFinished.emit()
        self.assertEqual(widget.feature_type, widget.GENERIC)
        self.assertFalse(widget.commit.now.called)
        widget.commit.now.reset_mock()
        widget.feature_combo.activated.emit(0)
        self.assertEqual(widget.feature_type, widget.FROM_VAR)
        self.assertFalse(widget.commit.now.called)
        widget.commit.deferred = unittest.mock.Mock()
        widget.commit.deferred.reset_mock()
        widget.controls.feature_name.editingFinished.emit()
        self.assertEqual(widget.feature_type, widget.GENERIC)
        self.assertTrue(widget.commit.deferred.called)
        widget.commit.deferred.reset_mock()
        widget.feature_combo.activated.emit(0)
        self.assertEqual(widget.feature_type, widget.FROM_VAR)
        self.assertTrue(widget.commit.deferred.called)

    def test_all_whitespace(self):
        if False:
            i = 10
            return i + 15
        widget = self.widget
        widget.feature_name = '  '
        widget.controls.feature_name.editingFinished.emit()
        self.assertEqual(widget.feature_name, '')

    def test_error(self):
        if False:
            print('Hello World!')
        widget = self.widget
        with unittest.mock.patch('Orange.data.Table.transpose', side_effect=ValueError('foo')):
            self.send_signal(widget.Inputs.data, self.zoo)
            self.wait_until_finished()
            self.assertTrue(widget.Error.value_error.is_shown())
        self.send_signal(widget.Inputs.data, self.zoo)
        self.wait_until_finished()
        self.assertFalse(widget.Error.value_error.is_shown())

    def test_feature_names_from_cont_vars(self):
        if False:
            return 10
        table = Table('iris')
        self.send_signal(self.widget.Inputs.data, table)
        self.assertListEqual(self.widget.feature_model[:], list(table.domain.attributes))
        self.widget.feature_combo.activated.emit(3)
        output = self.get_output(self.widget.Outputs.data)
        self.assertListEqual([f.name for f in output.domain.attributes[:10]], ['0.2 (1)', '0.2 (2)', '0.2 (3)', '0.2 (4)', '0.2 (5)', '0.4 (1)', '0.3 (1)', '0.2 (6)', '0.2 (7)', '0.1 (1)'])
        self.assertTrue(self.widget.Warning.duplicate_names.is_shown())

    def test_unconditional_commit_on_new_signal(self):
        if False:
            while True:
                i = 10
        with patch.object(self.widget.commit, 'now') as apply:
            self.widget.auto_apply = False
            apply.reset_mock()
            self.send_signal(self.widget.Inputs.data, self.zoo)
            apply.assert_called()
if __name__ == '__main__':
    unittest.main()