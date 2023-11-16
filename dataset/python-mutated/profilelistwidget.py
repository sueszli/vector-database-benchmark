from functools import partial
import uuid
from PyQt6 import QtCore, QtGui, QtWidgets
from picard.const import DEFAULT_PROFILE_NAME
from picard.util import unique_numbered_title
from picard.ui import HashableListWidgetItem

class ProfileListWidget(QtWidgets.QListWidget):

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        item = self.itemAt(event.x(), event.y())
        if item:
            menu = QtWidgets.QMenu(self)
            rename_action = QtGui.QAction(_('Rename profile'), self)
            rename_action.triggered.connect(partial(self.editItem, item))
            menu.addAction(rename_action)
            remove_action = QtGui.QAction(_('Remove profile'), self)
            remove_action.triggered.connect(partial(self.remove_profile, item))
            menu.addAction(remove_action)
            menu.exec(event.globalPos())

    def keyPressEvent(self, event):
        if False:
            return 10
        if event.matches(QtGui.QKeySequence.StandardKey.Delete):
            self.remove_selected_profile()
        elif event.key() == QtCore.Qt.Key.Key_Insert:
            self.add_profile()
        else:
            super().keyPressEvent(event)

    def unique_profile_name(self, base_name=None):
        if False:
            return 10
        if base_name is None:
            base_name = gettext_constants(DEFAULT_PROFILE_NAME)
        existing_titles = [self.item(i).name for i in range(self.count())]
        return unique_numbered_title(base_name, existing_titles)

    def add_profile(self, name=None, profile_id=''):
        if False:
            i = 10
            return i + 15
        if name is None:
            name = self.unique_profile_name()
        list_item = ProfileListWidgetItem(name=name, profile_id=profile_id)
        list_item.setCheckState(QtCore.Qt.CheckState.Checked)
        self.insertItem(0, list_item)
        self.setCurrentItem(list_item, QtCore.QItemSelectionModel.SelectionFlag.Clear | QtCore.QItemSelectionModel.SelectionFlag.SelectCurrent)

    def remove_selected_profile(self):
        if False:
            i = 10
            return i + 15
        items = self.selectedItems()
        if items:
            self.remove_profile(items[0])

    def remove_profile(self, item):
        if False:
            return 10
        row = self.row(item)
        msg = _('Are you sure you want to remove this profile?')
        reply = QtWidgets.QMessageBox.question(self, _('Confirm Remove'), msg, QtWidgets.QMessageBox.StandardButton.Yes, QtWidgets.QMessageBox.StandardButton.No)
        if item and reply == QtWidgets.QMessageBox.StandardButton.Yes:
            item = self.takeItem(row)
            del item

class ProfileListWidgetItem(HashableListWidgetItem):
    """Holds a profile's list and text widget properties"""

    def __init__(self, name=None, enabled=True, profile_id=''):
        if False:
            print('Hello World!')
        super().__init__(name)
        self.setFlags(self.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsEditable)
        if name is None:
            name = gettext_constants(DEFAULT_PROFILE_NAME)
        self.setText(name)
        self.setCheckState(QtCore.Qt.CheckState.Checked if enabled else QtCore.Qt.CheckState.Unchecked)
        if not profile_id:
            profile_id = str(uuid.uuid4())
        self.profile_id = profile_id

    @property
    def pos(self):
        if False:
            return 10
        return self.listWidget().row(self)

    @property
    def name(self):
        if False:
            print('Hello World!')
        return self.text()

    @property
    def enabled(self):
        if False:
            return 10
        return self.checkState() == QtCore.Qt.CheckState.Checked

    def get_all(self):
        if False:
            while True:
                i = 10
        return (self.pos, self.name, self.enabled, self.profile_id)

    def get_dict(self):
        if False:
            print('Hello World!')
        return {'position': self.pos, 'title': self.name, 'enabled': self.enabled, 'id': self.profile_id}