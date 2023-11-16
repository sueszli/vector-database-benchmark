from PyQt5.QtCore import QModelIndex, Qt
from PyQt5.QtGui import QIcon
from urh.models.ProtocolTreeModel import ProtocolTreeModel

class GeneratorTreeModel(ProtocolTreeModel):

    def __init__(self, controller, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(controller, parent)

    def set_root_item(self, root_item):
        if False:
            return 10
        self.rootItem = root_item

    def flags(self, index: QModelIndex):
        if False:
            print('Hello World!')
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def mimeTypes(self):
        if False:
            for i in range(10):
                print('nop')
        return []

    def data(self, index: QModelIndex, role=None):
        if False:
            while True:
                i = 10
        item = self.getItem(index)
        if role == Qt.DisplayRole:
            return item.data()
        elif role == Qt.DecorationRole and item.is_group:
            return QIcon.fromTheme('folder')