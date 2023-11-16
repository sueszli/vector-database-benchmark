from PyQt5.QtCore import QModelIndex, Qt, QSortFilterProxyModel
from PyQt5.QtGui import QFont, QColor

class FileFilterProxyModel(QSortFilterProxyModel):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.open_files = set()

    def filterAcceptsRow(self, source_row: int, source_parent: QModelIndex):
        if False:
            for i in range(10):
                print('nop')
        index0 = self.sourceModel().index(source_row, 0, source_parent)
        return self.sourceModel().fileName(index0) != 'URHProject.xml'

    def get_file_path(self, index: QModelIndex):
        if False:
            for i in range(10):
                print('nop')
        return self.sourceModel().filePath(self.mapToSource(index))

    def data(self, index: QModelIndex, role=None):
        if False:
            while True:
                i = 10
        if role == Qt.FontRole or role == Qt.TextColorRole:
            file_name = self.get_file_path(index)
            if hasattr(self, 'open_files') and file_name in self.open_files:
                if role == Qt.FontRole:
                    font = QFont()
                    font.setBold(True)
                    return font
                elif role == Qt.TextColorRole:
                    return QColor('orange')
        return super().data(index, role)