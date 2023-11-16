import unittest
from scipy import sparse
from AnyQt.QtWidgets import QButtonGroup, QRadioButton, QSpinBox, QDoubleSpinBox, QComboBox
from Orange.data import Table
from Orange.widgets.model.owrules import OWRuleLearner
from Orange.widgets.tests.base import WidgetTest, WidgetLearnerTestMixin, ParameterMapping

class TestOWRulesClassification(WidgetTest, WidgetLearnerTestMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.widget = self.create_widget(OWRuleLearner, stored_settings={'auto_apply': False})
        self.init()
        self.radio_button_groups = self.widget.findChildren(QButtonGroup)
        self.radio_buttons = self.widget.findChildren(QRadioButton)
        self.spin_boxes = self.widget.findChildren(QSpinBox)
        self.double_spin_boxes = self.widget.findChildren(QDoubleSpinBox)
        self.combo_boxes = self.widget.findChildren(QComboBox)
        self.parameters = [ParameterMapping('Evaluation measure', self.combo_boxes[0], self.widget.storage_measures), ParameterMapping('Beam width', self.spin_boxes[0]), ParameterMapping('Minimum rule coverage', self.spin_boxes[1]), ParameterMapping('Maximum rule length', self.spin_boxes[2])]

    def test_rule_ordering_radio_buttons(self):
        if False:
            return 10
        self.assertFalse(self.radio_buttons[0].isHidden())
        self.assertFalse(self.radio_buttons[1].isHidden())
        self.assertTrue(self.radio_buttons[0].isChecked())
        self.assertFalse(self.radio_buttons[1].isChecked())
        self.assertEqual(self.widget.rule_ordering, 0)
        self.radio_buttons[1].click()
        self.assertFalse(self.radio_buttons[0].isChecked())
        self.assertTrue(self.radio_buttons[1].isChecked())
        self.assertEqual(self.widget.rule_ordering, 1)

    def test_covering_algorithm_radio_buttons(self):
        if False:
            while True:
                i = 10
        self.assertFalse(self.radio_buttons[2].isHidden())
        self.assertFalse(self.radio_buttons[3].isHidden())
        self.assertTrue(self.radio_buttons[2].isChecked())
        self.assertFalse(self.radio_buttons[3].isChecked())
        self.assertEqual(self.widget.covering_algorithm, 0)
        self.assertFalse(self.double_spin_boxes[0].isEnabled())
        self.radio_buttons[3].click()
        self.assertFalse(self.radio_buttons[2].isChecked())
        self.assertTrue(self.radio_buttons[3].isChecked())
        self.assertEqual(self.widget.covering_algorithm, 1)
        self.assertTrue(self.double_spin_boxes[0].isEnabled())
        self.assertEqual(self.double_spin_boxes[0].value(), self.widget.gamma)

    def test_alpha_double_spin_boxes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Due to the checkbox components of the double-spin boxes,\n        standard ParameterMapping cannot be used for this specific\n        widget.\n        '
        self.assertFalse(self.double_spin_boxes[1].box.isHidden())
        self.assertFalse(self.double_spin_boxes[2].box.isHidden())
        self.assertFalse(self.double_spin_boxes[1].isEnabled())
        self.assertFalse(self.double_spin_boxes[2].isEnabled())
        self.double_spin_boxes[1].cbox.click()
        self.double_spin_boxes[2].cbox.click()
        self.assertTrue(self.double_spin_boxes[1].isEnabled())
        self.assertTrue(self.double_spin_boxes[2].isEnabled())
        self.assertEqual(self.double_spin_boxes[1].value(), self.widget.default_alpha)
        self.assertEqual(self.double_spin_boxes[2].value(), self.widget.parent_alpha)

    def test_sparse_data(self):
        if False:
            while True:
                i = 10
        data = Table('iris')
        with data.unlocked():
            data.X = sparse.csr_matrix(data.X)
        self.assertTrue(sparse.issparse(data.X))
        self.send_signal(self.widget.Inputs.data, data)
        self.click_apply()
        self.assertTrue(self.widget.Error.sparse_not_supported.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.click_apply()
        self.assertFalse(self.widget.Error.sparse_not_supported.is_shown())

    def test_out_of_memory(self):
        if False:
            return 10
        '\n        Handling memory error.\n        GH-2397\n        '
        data = Table('iris')[::3]
        self.assertFalse(self.widget.Error.out_of_memory.is_shown())
        with unittest.mock.patch('Orange.widgets.model.owrules.CustomRuleLearner.__call__', side_effect=MemoryError):
            self.send_signal(self.widget.Inputs.data, data)
            self.assertTrue(self.widget.Error.out_of_memory.is_shown())
        self.send_signal(self.widget.Inputs.data, None)
        self.assertFalse(self.widget.Error.out_of_memory.is_shown())

    def test_default_rule(self):
        if False:
            while True:
                i = 10
        data = Table('zoo')
        self.send_signal(self.widget.Inputs.data, data)
        self.click_apply()
        self.assertEqual(sum(self.widget.model.rule_list[-1].curr_class_dist.tolist()), len(data))