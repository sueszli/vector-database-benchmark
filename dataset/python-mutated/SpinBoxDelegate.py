from PyQt5.QtCore import QModelIndex, pyqtSlot, QAbstractItemModel, Qt
from PyQt5.QtWidgets import QStyledItemDelegate, QWidget, QStyleOptionViewItem, QSpinBox

class SpinBoxDelegate(QStyledItemDelegate):

    def __init__(self, minimum, maximum, parent=None, suffix=''):
        if False:
            return 10
        super().__init__(parent)
        self.minimum = minimum
        self.maximum = maximum
        self.suffix = suffix

    def _get_editor(self, parent) -> QSpinBox:
        if False:
            i = 10
            return i + 15
        return QSpinBox(parent)

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex):
        if False:
            while True:
                i = 10
        editor = self._get_editor(parent)
        editor.setMinimum(self.minimum)
        editor.setMaximum(self.maximum)
        editor.setSuffix(self.suffix)
        editor.valueChanged.connect(self.valueChanged)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        if False:
            print('Hello World!')
        editor.blockSignals(True)
        try:
            editor.setValue(int(index.model().data(index)))
        except ValueError:
            pass
        editor.blockSignals(False)

    def setModelData(self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex):
        if False:
            i = 10
            return i + 15
        model.setData(index, editor.value(), Qt.EditRole)

    @pyqtSlot()
    def valueChanged(self):
        if False:
            for i in range(10):
                print('nop')
        self.commitData.emit(self.sender())