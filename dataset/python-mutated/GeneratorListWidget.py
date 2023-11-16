from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtGui import QContextMenuEvent, QFocusEvent
from PyQt5.QtWidgets import QListWidget, QMenu, QAction

class GeneratorListWidget(QListWidget):
    item_edit_clicked = pyqtSignal(int)
    edit_all_items_clicked = pyqtSignal()
    lost_focus = pyqtSignal()

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super().__init__(parent)

    def create_context_menu(self) -> QMenu:
        if False:
            while True:
                i = 10
        menu = QMenu()
        sel_indexes = [index.row() for index in self.selectedIndexes()]
        edit_action = QAction('Edit', self)
        edit_action.triggered.connect(self.on_edit_action_triggered)
        if len(sel_indexes) == 0:
            edit_action.setEnabled(False)
        menu.addAction(edit_action)
        if self.count() > 0:
            edit_all_action = QAction('Edit all', self)
            edit_all_action.triggered.connect(self.on_edit_all_action_triggered)
            menu.addAction(edit_all_action)
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent):
        if False:
            for i in range(10):
                print('nop')
        menu = self.create_context_menu()
        menu.exec_(self.mapToGlobal(event.pos()))

    def focusOutEvent(self, event: QFocusEvent):
        if False:
            for i in range(10):
                print('nop')
        self.lost_focus.emit()
        super().focusOutEvent(event)

    @pyqtSlot()
    def on_edit_action_triggered(self):
        if False:
            print('Hello World!')
        if len(self.selectedIndexes()) > 0:
            selected_indx = self.selectedIndexes()[0].row()
            self.item_edit_clicked.emit(selected_indx)

    @pyqtSlot()
    def on_edit_all_action_triggered(self):
        if False:
            print('Hello World!')
        self.edit_all_items_clicked.emit()