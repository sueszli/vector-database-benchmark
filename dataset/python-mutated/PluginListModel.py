from PyQt5.QtCore import QAbstractListModel, Qt, QModelIndex
from PyQt5.QtGui import QFont
from urh import settings
from urh.plugins import Plugin

class PluginListModel(QAbstractListModel):

    def __init__(self, plugins, highlighted_plugins=None, parent=None):
        if False:
            return 10
        '\n        :type plugins: list of Plugin\n        :type highlighted_plugins: list of Plugin\n        '
        super().__init__(parent)
        self.plugins = plugins
        self.highlighted_plugins = highlighted_plugins if highlighted_plugins is not None else []

    def rowCount(self, QModelIndex_parent=None, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return len(self.plugins)

    def data(self, index: QModelIndex, role=None):
        if False:
            i = 10
            return i + 15
        row = index.row()
        if role == Qt.DisplayRole:
            return self.plugins[row].name
        elif role == Qt.CheckStateRole:
            return self.plugins[row].enabled
        elif role == Qt.TextColorRole and self.plugins[row] in self.highlighted_plugins:
            return settings.HIGHLIGHT_TEXT_FOREGROUND_COLOR
        elif role == Qt.BackgroundColorRole and self.plugins[row] in self.highlighted_plugins:
            return settings.HIGHLIGHT_TEXT_BACKGROUND_COLOR
        elif role == Qt.FontRole and self.plugins[row] in self.highlighted_plugins:
            font = QFont()
            font.setBold(True)
            return font

    def setData(self, index: QModelIndex, value, role=None):
        if False:
            for i in range(10):
                print('nop')
        if role == Qt.CheckStateRole:
            self.plugins[index.row()].enabled = value
        return True

    def flags(self, index):
        if False:
            for i in range(10):
                print('nop')
        return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable