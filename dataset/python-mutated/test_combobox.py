from AnyQt.QtCore import Qt, QEvent
from AnyQt.QtGui import QFont, QColor, QFocusEvent
from AnyQt.QtWidgets import QApplication
from AnyQt.QtTest import QTest, QSignalSpy
from orangewidget.tests.base import GuiTest
from orangewidget.tests.utils import simulate
from orangewidget.utils.itemmodels import PyListModel
from Orange.widgets.utils.combobox import ItemStyledComboBox, TextEditCombo

class TestItemStyledComboBox(GuiTest):

    def test_combobox(self):
        if False:
            for i in range(10):
                print('nop')
        cb = ItemStyledComboBox()
        cb.setPlaceholderText('...')
        self.assertEqual(cb.placeholderText(), '...')
        cb.grab()
        cb.addItems(['1'])
        cb.setCurrentIndex(0)
        model = cb.model()
        model.setItemData(model.index(0, 0), {Qt.ForegroundRole: QColor(Qt.blue), Qt.FontRole: QFont('Windings')})
        cb.grab()

class TestTextEditCombo(GuiTest):

    def test_texteditcombo(self):
        if False:
            while True:
                i = 10
        cb = TextEditCombo()
        model = PyListModel()
        cb.setModel(model)

        def enter_text(text: str):
            if False:
                return 10
            cb.lineEdit().selectAll()
            spy_act = QSignalSpy(cb.activated[int])
            spy_edit = QSignalSpy(cb.editingFinished)
            QTest.keyClick(cb.lineEdit(), Qt.Key_Delete)
            QTest.keyClicks(cb.lineEdit(), text)
            QApplication.sendEvent(cb, QFocusEvent(QEvent.FocusOut, Qt.TabFocusReason))
            self.assertEqual(len(spy_edit), 1)
            if cb.insertPolicy() != TextEditCombo.NoInsert:
                self.assertEqual(list(spy_act), [[cb.currentIndex()]])
        cb.setInsertPolicy(TextEditCombo.NoInsert)
        enter_text('!!')
        self.assertEqual(list(model), [])
        cb.setInsertPolicy(TextEditCombo.InsertAtTop)
        enter_text('BB')
        enter_text('AA')
        self.assertEqual(list(model), ['AA', 'BB'])
        cb.setInsertPolicy(TextEditCombo.InsertAtBottom)
        enter_text('CC')
        self.assertEqual(list(model), ['AA', 'BB', 'CC'])
        cb.setInsertPolicy(TextEditCombo.InsertBeforeCurrent)
        cb.setCurrentIndex(1)
        enter_text('AB')
        self.assertEqual(list(model), ['AA', 'AB', 'BB', 'CC'])
        cb.setInsertPolicy(TextEditCombo.InsertAfterCurrent)
        cb.setCurrentIndex(2)
        enter_text('BC')
        self.assertEqual(list(model), ['AA', 'AB', 'BB', 'BC', 'CC'])
        cb.setInsertPolicy(TextEditCombo.InsertAtCurrent)
        cb.setCurrentIndex(2)
        enter_text('BBA')
        self.assertEqual(list(model), ['AA', 'AB', 'BBA', 'BC', 'CC'])
        cb.setInsertPolicy(TextEditCombo.InsertAlphabetically)
        enter_text('BCA')
        self.assertEqual(list(model), ['AA', 'AB', 'BBA', 'BC', 'BCA', 'CC'])

    def test_activate_editing_finished_emit_ordering(self):
        if False:
            for i in range(10):
                print('nop')

        def activated():
            if False:
                while True:
                    i = 10
            sigs.append('activated')

        def finished():
            if False:
                for i in range(10):
                    print('nop')
            sigs.append('finished')
        sigs = []
        cb = TextEditCombo(activated=activated, editingFinished=finished)
        cb.insertItem(0, 'AA')
        simulate.combobox_activate_index(cb, 0)
        self.assertEqual(sigs, ['finished', 'activated'])