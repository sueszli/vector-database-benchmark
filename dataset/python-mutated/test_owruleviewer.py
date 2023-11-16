from AnyQt.QtCore import Qt
from AnyQt.QtWidgets import QApplication
from Orange.data import Table
from Orange.widgets.tests.base import WidgetTest, WidgetOutputsTestMixin
from Orange.classification import CN2Learner
from Orange.widgets.visualize.owruleviewer import OWRuleViewer

class TestOWRuleViewer(WidgetTest, WidgetOutputsTestMixin):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        WidgetOutputsTestMixin.init(cls)
        cls.titanic = Table('titanic')
        cls.learner = CN2Learner()
        cls.classifier = cls.learner(cls.titanic)
        cls.classifier.instances = cls.titanic
        cls.signal_name = OWRuleViewer.Inputs.classifier
        cls.signal_data = cls.classifier
        cls.data = cls.titanic

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.widget = self.create_widget(OWRuleViewer)

    def test_set_data(self):
        if False:
            print('Hello World!')
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.assertEqual(self.titanic, self.widget.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))
        self.send_signal(self.widget.Inputs.data, None)
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_set_classifier(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(self.widget.data)
        self.assertIsNone(self.widget.classifier)
        self.assertIsNone(self.widget.selected)
        self.send_signal(self.widget.Inputs.classifier, self.classifier)
        self.assertIsNone(self.widget.data)
        self.assertIsNotNone(self.widget.classifier)
        self.assertIsNone(self.widget.selected)
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_filtered_data_output(self):
        if False:
            for i in range(10):
                print('nop')
        self.send_signal(self.widget.Inputs.data, self.titanic)
        self.send_signal(self.widget.Inputs.classifier, self.classifier)
        selection_model = self.widget.view.selectionModel()
        selection_model.select(self.widget.proxy_model.index(len(self.classifier.rule_list) - 1, 0), selection_model.Select | selection_model.Rows)
        output = self.get_output(self.widget.Outputs.selected_data)
        self.assertEqual(len(self.titanic), len(output))
        selection_model.clearSelection()
        self.assertIsNone(self.get_output(self.widget.Outputs.selected_data))

    def test_copy_to_clipboard(self):
        if False:
            return 10
        self.send_signal(self.widget.Inputs.classifier, self.classifier)
        selection_model = self.widget.view.selectionModel()
        selection_model.select(self.widget.proxy_model.index(len(self.classifier.rule_list) - 1, 0), selection_model.Select | selection_model.Rows)
        self.widget.copy_to_clipboard()
        clipboard_contents = QApplication.clipboard().text()
        self.assertTrue(self.classifier.rule_list[-1].__str__() == clipboard_contents)

    def test_restore_original_order(self):
        if False:
            return 10
        self.send_signal(self.widget.Inputs.classifier, self.classifier)
        bottom_row = len(self.classifier.rule_list) - 1
        self.widget.proxy_model.sort(0, Qt.AscendingOrder)
        q_index = self.widget.proxy_model.index(bottom_row, 0)
        self.assertEqual(bottom_row, q_index.row())
        q_index = self.widget.proxy_model.mapToSource(q_index)
        self.assertNotEqual(bottom_row, q_index.row())
        self.widget.restore_original_order()
        q_index = self.widget.proxy_model.index(bottom_row, 0)
        self.assertEqual(bottom_row, q_index.row())
        q_index = self.widget.proxy_model.mapToSource(q_index)
        self.assertEqual(bottom_row, q_index.row())

    def test_selection_compact_view(self):
        if False:
            while True:
                i = 10
        self.send_signal(self.widget.Inputs.classifier, self.classifier)
        selection_model = self.widget.view.selectionModel()
        selection_model.select(self.widget.proxy_model.index(0, 0), selection_model.Select | selection_model.Rows)
        self.widget._save_selected(actual=True)
        temp = self.widget.selected
        self.widget.on_update()
        self.widget._save_selected(actual=True)
        self.assertEqual(temp, self.widget.selected)

    def _select_data(self):
        if False:
            while True:
                i = 10
        selection_model = self.widget.view.selectionModel()
        selection_model.select(self.widget.proxy_model.index(2, 0), selection_model.Select | selection_model.Rows)
        return list(range(586, 597))