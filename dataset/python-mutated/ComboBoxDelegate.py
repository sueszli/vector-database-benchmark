import sys
from PyQt5.QtCore import QModelIndex, Qt, QAbstractItemModel, pyqtSlot, QRectF
from PyQt5.QtGui import QImage, QPainter, QColor, QPixmap
from PyQt5.QtWidgets import QStyledItemDelegate, QWidget, QStyleOptionViewItem, QComboBox

class ComboBoxDelegate(QStyledItemDelegate):

    def __init__(self, items, colors=None, is_editable=False, return_index=True, parent=None):
        if False:
            while True:
                i = 10
        '\n\n        :param items:\n        :param colors:\n        :param is_editable:\n        :param return_index: True for returning current index, false for returning current text of editor\n        :param parent:\n        '
        super().__init__(parent)
        self.items = items
        self.colors = colors
        self.return_index = return_index
        self.is_editable = is_editable
        self.current_edit_text = ''
        if colors:
            assert len(items) == len(colors)

    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index: QModelIndex):
        if False:
            return 10
        if self.colors:
            try:
                item = index.model().data(index)
                index = self.items.index(item) if item in self.items else int(item)
                color = self.colors[index]
                (x, y, h) = (option.rect.x(), option.rect.y(), option.rect.height())
                rect = QRectF(x + 8, y + h / 2 - 8, 16, 16)
                painter.fillRect(rect, QColor('black'))
                rect = rect.adjusted(1, 1, -1, -1)
                painter.fillRect(rect, QColor(color.red(), color.green(), color.blue(), 255))
            except:
                super().paint(painter, option, index)
        else:
            super().paint(painter, option, index)

    def createEditor(self, parent: QWidget, option: QStyleOptionViewItem, index: QModelIndex):
        if False:
            return 10
        editor = QComboBox(parent)
        if sys.platform == 'win32':
            editor.setMinimumHeight(self.sizeHint(option, index).height() + 10)
        editor.addItems(self.items)
        if self.is_editable:
            editor.setEditable(True)
            editor.setInsertPolicy(QComboBox.NoInsert)
        if self.current_edit_text:
            editor.setEditText(self.current_edit_text)
        if self.colors:
            img = QImage(16, 16, QImage.Format_RGB32)
            painter = QPainter(img)
            painter.fillRect(img.rect(), Qt.black)
            rect = img.rect().adjusted(1, 1, -1, -1)
            for (i, item) in enumerate(self.items):
                color = self.colors[i]
                painter.fillRect(rect, QColor(color.red(), color.green(), color.blue(), 255))
                editor.setItemData(i, QPixmap.fromImage(img), Qt.DecorationRole)
            del painter
        editor.currentIndexChanged.connect(self.currentIndexChanged)
        editor.editTextChanged.connect(self.on_edit_text_changed)
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex):
        if False:
            print('Hello World!')
        editor.blockSignals(True)
        item = index.model().data(index)
        try:
            indx = self.items.index(item) if item in self.items else int(item)
            editor.setCurrentIndex(indx)
        except ValueError:
            pass
        editor.blockSignals(False)

    def setModelData(self, editor: QWidget, model: QAbstractItemModel, index: QModelIndex):
        if False:
            i = 10
            return i + 15
        if self.return_index:
            model.setData(index, editor.currentIndex(), Qt.EditRole)
        else:
            model.setData(index, editor.currentText(), Qt.EditRole)

    def updateEditorGeometry(self, editor: QWidget, option: QStyleOptionViewItem, index: QModelIndex):
        if False:
            i = 10
            return i + 15
        editor.setGeometry(option.rect)

    @pyqtSlot()
    def currentIndexChanged(self):
        if False:
            for i in range(10):
                print('nop')
        self.commitData.emit(self.sender())

    @pyqtSlot(str)
    def on_edit_text_changed(self, text: str):
        if False:
            return 10
        self.current_edit_text = text