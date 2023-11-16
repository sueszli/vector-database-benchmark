from functools import partial
import threading
from PyQt6 import QtCore, QtGui, QtWidgets
from picard.const import DEFAULT_SCRIPT_NAME
from picard.util import unique_numbered_title
from picard.ui import HashableListWidgetItem

class ScriptListWidget(QtWidgets.QListWidget):
    signal_reset_selected_item = QtCore.pyqtSignal()

    def __init__(self, parent):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.itemChanged.connect(self.item_changed)
        self.currentItemChanged.connect(self.current_item_changed)
        self.old_row = -1
        self.bad_row = -1

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        item = self.itemAt(event.x(), event.y())
        if item:
            menu = QtWidgets.QMenu(self)
            rename_action = QtGui.QAction(_('Rename script'), self)
            rename_action.triggered.connect(partial(self.editItem, item))
            menu.addAction(rename_action)
            remove_action = QtGui.QAction(_('Remove script'), self)
            remove_action.triggered.connect(partial(self.remove_script, item))
            menu.addAction(remove_action)
            menu.exec(event.globalPos())

    def keyPressEvent(self, event):
        if False:
            while True:
                i = 10
        if event.matches(QtGui.QKeySequence.StandardKey.Delete):
            self.remove_selected_script()
        elif event.key() == QtCore.Qt.Key.Key_Insert:
            self.add_script()
        else:
            super().keyPressEvent(event)

    def unique_script_name(self):
        if False:
            while True:
                i = 10
        existing_titles = [self.item(i).name for i in range(self.count())]
        return unique_numbered_title(gettext_constants(DEFAULT_SCRIPT_NAME), existing_titles)

    def add_script(self):
        if False:
            for i in range(10):
                print('nop')
        numbered_name = self.unique_script_name()
        list_item = ScriptListWidgetItem(name=numbered_name)
        list_item.setCheckState(QtCore.Qt.CheckState.Checked)
        self.addItem(list_item)
        self.setCurrentItem(list_item, QtCore.QItemSelectionModel.SelectionFlag.Clear | QtCore.QItemSelectionModel.SelectionFlag.SelectCurrent)

    def remove_selected_script(self):
        if False:
            print('Hello World!')
        items = self.selectedItems()
        if items:
            self.remove_script(items[0])

    def remove_script(self, item):
        if False:
            return 10
        row = self.row(item)
        msg = _('Are you sure you want to remove this script?')
        reply = QtWidgets.QMessageBox.question(self, _('Confirm Remove'), msg, QtWidgets.QMessageBox.StandardButton.Yes, QtWidgets.QMessageBox.StandardButton.No)
        if item and reply == QtWidgets.QMessageBox.StandardButton.Yes:
            item = self.takeItem(row)
            del item

    def item_changed(self, item):
        if False:
            for i in range(10):
                print('nop')
        if not item.name.strip():
            item.setText(self.unique_script_name())

    def current_item_changed(self, new_item, old_item):
        if False:
            while True:
                i = 10
        if old_item and old_item.has_error:
            self.bad_row = self.old_row
            threading.Thread(target=self.signal_reset_selected_item.emit).start()
        else:
            self.old_row = self.currentRow()

class ScriptListWidgetItem(HashableListWidgetItem):
    """Holds a script's list and text widget properties"""

    def __init__(self, name=None, enabled=True, script=''):
        if False:
            print('Hello World!')
        super().__init__(name)
        self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEditable)
        if name is None:
            name = gettext_constants(DEFAULT_SCRIPT_NAME)
        self.setText(name)
        self.setCheckState(QtCore.Qt.CheckState.Checked if enabled else QtCore.Qt.CheckState.Unchecked)
        self.script = script
        self.has_error = False

    @property
    def pos(self):
        if False:
            i = 10
            return i + 15
        return self.listWidget().row(self)

    @property
    def name(self):
        if False:
            i = 10
            return i + 15
        return self.text()

    @property
    def enabled(self):
        if False:
            return 10
        return self.checkState() == QtCore.Qt.CheckState.Checked

    def get_all(self):
        if False:
            i = 10
            return i + 15
        return (self.pos, self.name, self.enabled, self.script)