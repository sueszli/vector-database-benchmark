from PyQt5.QtCore import pyqtSignal, Qt, pyqtSlot
from PyQt5.QtGui import QContextMenuEvent, QKeyEvent, QIcon
from PyQt5.QtWidgets import QListView, QMenu
from urh.models.GeneratorListModel import GeneratorListModel

class GeneratorListView(QListView):
    selection_changed = pyqtSignal()
    edit_on_item_triggered = pyqtSignal(int)

    def __init__(self, parent):
        if False:
            return 10
        super().__init__(parent)
        self.context_menu_pos = None

    def model(self) -> GeneratorListModel:
        if False:
            for i in range(10):
                print('nop')
        return super().model()

    def create_context_menu(self):
        if False:
            return 10
        menu = QMenu()
        if self.model().message is None or len(self.model().message.message_type) == 0:
            return menu
        edit_action = menu.addAction('Edit fuzzing label')
        edit_action.setIcon(QIcon.fromTheme('configure'))
        edit_action.triggered.connect(self.on_edit_action_triggered)
        del_action = menu.addAction('Delete fuzzing label')
        del_action.setIcon(QIcon.fromTheme('edit-delete'))
        del_action.triggered.connect(self.on_delete_action_triggered)
        menu.addSeparator()
        fuzz_all_action = menu.addAction('Check all')
        fuzz_all_action.triggered.connect(self.model().fuzzAll)
        unfuzz_all_action = menu.addAction('Uncheck all')
        unfuzz_all_action.triggered.connect(self.model().unfuzzAll)
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent):
        if False:
            for i in range(10):
                print('nop')
        self.context_menu_pos = event.pos()
        menu = self.create_context_menu()
        menu.exec(self.mapToGlobal(self.context_menu_pos))
        self.context_menu_pos = None

    def selectionChanged(self, QItemSelection, QItemSelection_1):
        if False:
            return 10
        self.selection_changed.emit()
        super().selectionChanged(QItemSelection, QItemSelection_1)

    def keyPressEvent(self, event: QKeyEvent):
        if False:
            for i in range(10):
                print('nop')
        if event.key() in (Qt.Key_Enter, Qt.Key_Return):
            selected = [index.row() for index in self.selectedIndexes()]
            if len(selected) > 0:
                self.edit_on_item_triggered.emit(min(selected))
        else:
            super().keyPressEvent(event)

    def mouseDoubleClickEvent(self, QMouseEvent):
        if False:
            return 10
        selected = [index.row() for index in self.selectedIndexes()]
        if len(selected) > 0:
            self.edit_on_item_triggered.emit(min(selected))

    @pyqtSlot()
    def on_delete_action_triggered(self):
        if False:
            while True:
                i = 10
        index = self.indexAt(self.context_menu_pos)
        self.model().delete_label_at(index.row())

    @pyqtSlot()
    def on_edit_action_triggered(self):
        if False:
            for i in range(10):
                print('nop')
        self.edit_on_item_triggered.emit(self.indexAt(self.context_menu_pos).row())