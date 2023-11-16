"""
Created on 2018年12月27日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: QListView.SortItemByRole
@description: 
"""
from random import choice
try:
    from PyQt5.QtCore import QSortFilterProxyModel, Qt
    from PyQt5.QtGui import QStandardItem, QStandardItemModel
    from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QListView, QPushButton
except ImportError:
    from PySide2.QtCore import QSortFilterProxyModel, Qt
    from PySide2.QtGui import QStandardItem, QStandardItemModel
    from PySide2.QtWidgets import QApplication, QWidget, QVBoxLayout, QListView, QPushButton

class SortFilterProxyModel(QSortFilterProxyModel):

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        super(SortFilterProxyModel, self).__init__(*args, **kwargs)
        self._topIndex = 0

    def setSortIndex(self, index):
        if False:
            print('Hello World!')
        self._topIndex = index
        print('在最前面的序号为:', index)

    def lessThan(self, source_left, source_right):
        if False:
            while True:
                i = 10
        if not source_left.isValid() or not source_right.isValid():
            return False
        if self.sortRole() == ClassifyRole and source_left.column() == self.sortColumn() and (source_right.column() == self.sortColumn()):
            leftIndex = source_left.data(ClassifyRole)
            rightIndex = source_right.data(ClassifyRole)
            if self.sortOrder() == Qt.AscendingOrder:
                if leftIndex == self._topIndex:
                    leftIndex = -1
                if rightIndex == self._topIndex:
                    rightIndex = -1
                return leftIndex < rightIndex
        return super(SortFilterProxyModel, self).lessThan(source_left, source_right)
NameDict = {'唐': ['Tang', 0], '宋': ['Song', 1], '元': ['Yuan', 2], '明': ['Ming', 3], '清': ['Qing', 4]}
IndexDict = {0: '唐', 1: '宋', 2: '元', 3: '明', 4: '清'}
IdRole = Qt.UserRole + 1
ClassifyRole = Qt.UserRole + 2

class Window(QWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(Window, self).__init__(*args, **kwargs)
        self.resize(600, 400)
        layout = QVBoxLayout(self)
        self.listView = QListView(self)
        self.listView.setEditTriggers(QListView.NoEditTriggers)
        layout.addWidget(self.listView)
        layout.addWidget(QPushButton('恢复默认顺序', self, clicked=self.restoreSort))
        layout.addWidget(QPushButton('唐', self, clicked=self.sortByClassify))
        layout.addWidget(QPushButton('宋', self, clicked=self.sortByClassify))
        layout.addWidget(QPushButton('元', self, clicked=self.sortByClassify))
        layout.addWidget(QPushButton('明', self, clicked=self.sortByClassify))
        layout.addWidget(QPushButton('清', self, clicked=self.sortByClassify))
        self._initItems()

    def restoreSort(self):
        if False:
            while True:
                i = 10
        self.fmodel.setSortRole(IdRole)
        self.fmodel.sort(0)

    def sortByClassify(self):
        if False:
            while True:
                i = 10
        self.fmodel.setSortIndex(NameDict.get(self.sender().text(), ['', 100])[1])
        self.fmodel.setSortRole(IdRole)
        self.fmodel.setSortRole(ClassifyRole)
        self.fmodel.sort(0)

    def _initItems(self):
        if False:
            while True:
                i = 10
        self.dmodel = QStandardItemModel(self.listView)
        self.fmodel = SortFilterProxyModel(self.listView)
        self.fmodel.setSourceModel(self.dmodel)
        self.listView.setModel(self.fmodel)
        keys = list(NameDict.keys())
        print(keys)
        classifies = [v[1] for v in NameDict.values()]
        for i in range(5):
            classifies.append(100)
        print(classifies)
        for i in range(50):
            item = QStandardItem()
            item.setData(i, IdRole)
            c = choice(classifies)
            item.setData(c, ClassifyRole)
            item.setText('Name: {}\t\tId: {}\t\tClassify: {}'.format(IndexDict.get(c, '其它'), i, c))
            self.dmodel.appendRow(item)
if __name__ == '__main__':
    import sys
    import cgitb
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())