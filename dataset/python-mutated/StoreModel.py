"""
tableview的模型.
"""
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
PluginFileCol = 0
h1 = 1
h2 = 2
MTime = 3
CTime = 4
AutoStartCol = 5

class FileModel(QFileSystemModel):
    """
    继承QFileSystemModel.
    """
    RE_UN_LoadSignal = pyqtSignal(object)
    AutoStartSignal = pyqtSignal(object)

    def __init__(self, manager=None, *a, **kw):
        if False:
            while True:
                i = 10
        super(FileModel, self).__init__(*a, **kw)
        self.manager = manager

    def columnCount(self, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        添加了两列\n        '
        return 6

    def headerData(self, section, Orientation, role=Qt.DisplayRole):
        if False:
            return 10
        if Orientation == 1:
            if section == PluginFileCol:
                return '文件名'
            elif section == MTime:
                return '修改时间'
            elif section == CTime:
                return '创建时间'
            elif section == AutoStartCol:
                return '允许自启动'
        return super(FileModel, self).headerData(section, Orientation, role)

    def flags(self, index):
        if False:
            print('Hello World!')
        '\n        flag描述了view中数据项的状态信息\n        '
        column = index.column()
        if column == PluginFileCol:
            flag = super(FileModel, self).flags(index)
            return flag | Qt.ItemIsEnabled | Qt.ItemIsUserCheckable | Qt.ItemIsSelectable
        else:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsUserCheckable

    def data(self, index, role=Qt.DisplayRole):
        if False:
            while True:
                i = 10
        '\n        根据值来显示界面信息.\n        '
        if not index.isValid():
            return QVariant()
        column = index.column()
        if role == Qt.CheckStateRole:
            if column == PluginFileCol:
                mod = index.data()[:-3]
                return Qt.Checked if self.manager.pluginsInfo['StartModule'][mod]['active'] else Qt.Unchecked
            elif column == AutoStartCol:
                mod = self.index(index.row(), PluginFileCol, self.manager.index).data()[:-3]
                return Qt.Checked if self.manager.jsonPlugin[mod]['Allow'] else Qt.Unchecked
        if role == Qt.DisplayRole:
            if column == CTime:
                mod = self.index(index.row(), PluginFileCol, self.manager.index).data()[:-3]
                return self.manager.jsonPlugin[mod]['CreateTime']
            elif column == AutoStartCol:
                mod = self.index(index.row(), PluginFileCol, self.manager.index).data()[:-3]
                return str(self.manager.jsonPlugin[mod]['Allow'])
        return super(FileModel, self).data(index, role)

    def setData(self, index, value, role=Qt.DisplayRole):
        if False:
            while True:
                i = 10
        '\n        数据驱动界面 , 发射信号修改数据即可.\n        '
        if not index.isValid():
            return QVariant()
        if role == Qt.CheckStateRole:
            mod = self.index(index.row(), PluginFileCol, self.manager.index).data()[:-3]
            if index.column() == PluginFileCol:
                self.RE_UN_LoadSignal.emit((mod, index))
            elif index.column() == AutoStartCol:
                self.AutoStartSignal.emit(mod)
        return super(FileModel, self).setData(index, value, role)