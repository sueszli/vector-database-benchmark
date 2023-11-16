"""
Created on 2018年3月21日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: Splitter
@description: 
"""
import sys
try:
    from PyQt5.QtCore import Qt, QPointF, pyqtSignal
    from PyQt5.QtGui import QPainter, QPolygonF
    from PyQt5.QtWidgets import QTextEdit, QListWidget, QTreeWidget, QSplitter, QApplication, QMainWindow, QSplitterHandle
except ImportError:
    from PySide2.QtCore import Qt, QPointF, Signal as pyqtSignal
    from PySide2.QtGui import QPainter, QPolygonF
    from PySide2.QtWidgets import QTextEdit, QListWidget, QTreeWidget, QSplitter, QApplication, QMainWindow, QSplitterHandle

class SplitterHandle(QSplitterHandle):
    clicked = pyqtSignal()

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        super(SplitterHandle, self).__init__(*args, **kwargs)
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if False:
            while True:
                i = 10
        super(SplitterHandle, self).mousePressEvent(event)
        if event.pos().y() <= 24:
            self.clicked.emit()

    def mouseMoveEvent(self, event):
        if False:
            return 10
        '鼠标移动事件'
        if event.pos().y() <= 24:
            self.unsetCursor()
            event.accept()
        else:
            self.setCursor(Qt.SplitHCursor if self.orientation() == Qt.Horizontal else Qt.SplitVCursor)
            super(SplitterHandle, self).mouseMoveEvent(event)

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        super(SplitterHandle, self).paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(Qt.red)
        painter.drawRect(0, 0, self.width(), 24)
        painter.setBrush(Qt.red)
        painter.drawPolygon(QPolygonF([QPointF(0, (24 - 8) / 2), QPointF(self.width() - 2, 24 / 2), QPointF(0, (24 + 8) / 2)]))

class Splitter(QSplitter):

    def onClicked(self):
        if False:
            print('Hello World!')
        print('clicked')

    def createHandle(self):
        if False:
            print('Hello World!')
        if self.count() == 1:
            handle = SplitterHandle(self.orientation(), self)
            handle.clicked.connect(self.onClicked)
            return handle
        return super(Splitter, self).createHandle()

class SplitterWindow(QMainWindow):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super(SplitterWindow, self).__init__(parent)
        self.resize(400, 400)
        self.setWindowTitle('PyQt Qsplitter')
        textedit = QTextEdit('QTextEdit', self)
        listwidget = QListWidget(self)
        listwidget.addItem('This is  a \nListWidget!')
        treewidget = QTreeWidget()
        treewidget.setHeaderLabels(['This', 'is', 'a', 'TreeWidgets!'])
        splitter = Splitter(self)
        splitter.setHandleWidth(8)
        splitter.addWidget(textedit)
        splitter.addWidget(listwidget)
        splitter.addWidget(treewidget)
        splitter.setOrientation(Qt.Horizontal)
        self.setCentralWidget(splitter)
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = SplitterWindow()
    main.show()
    sys.exit(app.exec_())