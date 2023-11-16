from PyQt5.QtCore import pyqtSignal, Qt, QRect, QUrl, QModelIndex, QItemSelection
from PyQt5.QtWidgets import QComboBox, QStyledItemDelegate, QStyle, QStyleOptionComboBox, QStyleOptionViewItem, QApplication
from PyQt5.QtGui import QBrush
from hscommon.trans import trget
from qt.tree_model import RefNode, TreeModel
tr = trget('ui')
HEADERS = [tr('Name'), tr('State')]
STATES = [tr('Normal'), tr('Reference'), tr('Excluded')]

class DirectoriesDelegate(QStyledItemDelegate):

    def createEditor(self, parent, option, index):
        if False:
            print('Hello World!')
        editor = QComboBox(parent)
        editor.addItems(STATES)
        return editor

    def paint(self, painter, option, index):
        if False:
            while True:
                i = 10
        self.initStyleOption(option, index)
        option = QStyleOptionViewItem(option)
        if index.column() == 1 and option.state & QStyle.State_Selected:
            cboption = QStyleOptionComboBox()
            cboption.rect = option.rect
            cboption.state |= QStyle.State_Enabled
            QApplication.style().drawComplexControl(QStyle.CC_ComboBox, cboption, painter)
            painter.setBrush(option.palette.text())
            rect = QRect(option.rect)
            rect.setLeft(rect.left() + 4)
            painter.drawText(rect, Qt.AlignLeft, option.text)
        else:
            super().paint(painter, option, index)

    def setEditorData(self, editor, index):
        if False:
            print('Hello World!')
        value = index.model().data(index, Qt.EditRole)
        editor.setCurrentIndex(value)
        editor.showPopup()

    def setModelData(self, editor, model, index):
        if False:
            for i in range(10):
                print('nop')
        value = editor.currentIndex()
        model.setData(index, value, Qt.EditRole)

    def updateEditorGeometry(self, editor, option, index):
        if False:
            while True:
                i = 10
        editor.setGeometry(option.rect)

class DirectoriesModel(TreeModel):
    MIME_TYPE_FORMAT = 'text/uri-list'

    def __init__(self, model, view, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.model = model
        self.model.view = self
        self.view = view
        self.view.setModel(self)
        self.view.selectionModel().selectionChanged[QItemSelection, QItemSelection].connect(self.selectionChanged)

    def _create_node(self, ref, row):
        if False:
            i = 10
            return i + 15
        return RefNode(self, None, ref, row)

    def _get_children(self):
        if False:
            i = 10
            return i + 15
        return list(self.model)

    def columnCount(self, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        return 2

    def data(self, index, role):
        if False:
            i = 10
            return i + 15
        if not index.isValid():
            return None
        node = index.internalPointer()
        ref = node.ref
        if role == Qt.DisplayRole:
            if index.column() == 0:
                return ref.name
            else:
                return STATES[ref.state]
        elif role == Qt.EditRole and index.column() == 1:
            return ref.state
        elif role == Qt.ForegroundRole:
            state = ref.state
            if state == 1:
                return QBrush(Qt.blue)
            elif state == 2:
                return QBrush(Qt.red)
        return None

    def dropMimeData(self, mime_data, action, row, column, parent_index):
        if False:
            print('Hello World!')
        if not mime_data.hasFormat(self.MIME_TYPE_FORMAT):
            return False
        data = bytes(mime_data.data(self.MIME_TYPE_FORMAT)).decode('ascii')
        urls = data.split('\r\n')
        paths = [QUrl(url).toLocalFile() for url in urls if url]
        for path in paths:
            self.model.add_directory(path)
        self.foldersAdded.emit(paths)
        self.reset()
        return True

    def flags(self, index):
        if False:
            for i in range(10):
                print('nop')
        if not index.isValid():
            return Qt.ItemIsEnabled | Qt.ItemIsDropEnabled
        result = Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDropEnabled
        if index.column() == 1:
            result |= Qt.ItemIsEditable
        return result

    def headerData(self, section, orientation, role):
        if False:
            print('Hello World!')
        if orientation == Qt.Horizontal and role == Qt.DisplayRole and (section < len(HEADERS)):
            return HEADERS[section]
        return None

    def mimeTypes(self):
        if False:
            i = 10
            return i + 15
        return [self.MIME_TYPE_FORMAT]

    def setData(self, index, value, role):
        if False:
            while True:
                i = 10
        if not index.isValid() or role != Qt.EditRole or index.column() != 1:
            return False
        node = index.internalPointer()
        ref = node.ref
        ref.state = value
        return True

    def supportedDropActions(self):
        if False:
            return 10
        return Qt.ActionMask

    def selectionChanged(self, selected, deselected):
        if False:
            return 10
        new_nodes = [modelIndex.internalPointer().ref for modelIndex in self.view.selectionModel().selectedRows()]
        self.model.selected_nodes = new_nodes
    foldersAdded = pyqtSignal(list)

    def refresh(self):
        if False:
            return 10
        self.reset()

    def refresh_states(self):
        if False:
            print('Hello World!')
        self.refreshData()