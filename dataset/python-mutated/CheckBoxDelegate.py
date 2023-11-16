from PyQt5.QtCore import QModelIndex, QAbstractItemModel, Qt, pyqtSlot
from PyQt5.QtWidgets import QStyledItemDelegate, QWidget, QStyleOptionViewItem, QCheckBox

class CheckBoxDelegate(QStyledItemDelegate):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.enabled = True

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex):
        if False:
            for i in range(10):
                print('nop')
        editor = QCheckBox(parent)
        editor.stateChanged.connect(self.stateChanged)
        return editor

    def setEditorData(self, editor: QCheckBox, index: QModelIndex):
        if False:
            print('Hello World!')
        editor.blockSignals(True)
        editor.setChecked(index.model().data(index))
        self.enabled = editor.isChecked()
        editor.blockSignals(False)

    def setModelData(self, editor: QCheckBox, model: QAbstractItemModel, index: QModelIndex):
        if False:
            while True:
                i = 10
        model.setData(index, editor.isChecked(), Qt.EditRole)

    @pyqtSlot()
    def stateChanged(self):
        if False:
            i = 10
            return i + 15
        self.commitData.emit(self.sender())