"""
Created on 2017年4月6日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: CopyContent
@description: 
"""
try:
    from PyQt5.QtCore import Qt
    from PyQt5.QtGui import QStandardItemModel, QStandardItem
    from PyQt5.QtWidgets import QTableView, QApplication, QAction, QMessageBox
except ImportError:
    from PySide2.QtCore import Qt
    from PySide2.QtGui import QStandardItemModel, QStandardItem
    from PySide2.QtWidgets import QTableView, QApplication, QAction, QMessageBox

class TableView(QTableView):

    def __init__(self, parent=None):
        if False:
            return 10
        super(TableView, self).__init__(parent)
        self.resize(800, 600)
        self.setContextMenuPolicy(Qt.ActionsContextMenu)
        self.setEditTriggers(self.NoEditTriggers)
        self.doubleClicked.connect(self.onDoubleClick)
        self.addAction(QAction('复制', self, triggered=self.copyData))
        self.myModel = QStandardItemModel()
        self.initHeader()
        self.setModel(self.myModel)
        self.initData()

    def onDoubleClick(self, index):
        if False:
            for i in range(10):
                print('nop')
        print(index.row(), index.column(), index.data())

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        super(TableView, self).keyPressEvent(event)
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_C:
            self.copyData()

    def copyData(self):
        if False:
            return 10
        count = len(self.selectedIndexes())
        if count == 0:
            return
        if count == 1:
            QApplication.clipboard().setText(self.selectedIndexes()[0].data())
            QMessageBox.information(self, '提示', '已复制一个数据')
            return
        rows = set()
        cols = set()
        for index in self.selectedIndexes():
            rows.add(index.row())
            cols.add(index.column())
        if len(rows) == 1:
            QApplication.clipboard().setText('\t'.join([index.data() for index in self.selectedIndexes()]))
            QMessageBox.information(self, '提示', '已复制一行数据')
            return
        if len(cols) == 1:
            QApplication.clipboard().setText('\r\n'.join([index.data() for index in self.selectedIndexes()]))
            QMessageBox.information(self, '提示', '已复制一列数据')
            return
        (mirow, marow) = (min(rows), max(rows))
        (micol, macol) = (min(cols), max(cols))
        print(mirow, marow, micol, macol)
        arrays = [['' for _ in range(macol - micol + 1)] for _ in range(marow - mirow + 1)]
        print(arrays)
        for index in self.selectedIndexes():
            arrays[index.row() - mirow][index.column() - micol] = index.data()
        print(arrays)
        data = ''
        for row in arrays:
            data += '\t'.join(row) + '\r\n'
        print(data)
        QApplication.clipboard().setText(data)
        QMessageBox.information(self, '提示', '已复制')

    def initHeader(self):
        if False:
            while True:
                i = 10
        for i in range(5):
            self.myModel.setHorizontalHeaderItem(i, QStandardItem('表头' + str(i + 1)))

    def initData(self):
        if False:
            while True:
                i = 10
        for row in range(100):
            for col in range(5):
                self.myModel.setItem(row, col, QStandardItem('row: {row},col: {col}'.format(row=row + 1, col=col + 1)))
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setApplicationName('TableView')
    w = TableView()
    w.show()
    sys.exit(app.exec_())