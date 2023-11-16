"""
Created on 2018年9月14日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: DragListWidget
@description: 
"""
try:
    from PyQt5.QtCore import Qt, QSize, QRect, QPoint
    from PyQt5.QtGui import QColor, QPixmap, QDrag, QPainter, QCursor
    from PyQt5.QtWidgets import QListWidget, QListWidgetItem, QLabel, QRubberBand, QApplication
except ImportError:
    from PySide2.QtCore import Qt, QSize, QRect, QPoint
    from PySide2.QtGui import QColor, QPixmap, QDrag, QPainter, QCursor
    from PySide2.QtWidgets import QListWidget, QListWidgetItem, QLabel, QRubberBand, QApplication

class DropListWidget(QListWidget):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(DropListWidget, self).__init__(*args, **kwargs)
        self.resize(400, 400)
        self.setAcceptDrops(True)
        self.setFlow(self.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(self.Adjust)
        self.setSpacing(5)

    def makeItem(self, size, cname):
        if False:
            i = 10
            return i + 15
        item = QListWidgetItem(self)
        item.setData(Qt.UserRole + 1, cname)
        item.setSizeHint(size)
        label = QLabel(self)
        label.setMargin(2)
        label.resize(size)
        pixmap = QPixmap(size.scaled(96, 96, Qt.IgnoreAspectRatio))
        pixmap.fill(QColor(cname))
        label.setPixmap(pixmap)
        self.setItemWidget(item, label)

    def dragEnterEvent(self, event):
        if False:
            i = 10
            return i + 15
        mimeData = event.mimeData()
        if not mimeData.property('myItems'):
            event.ignore()
        else:
            event.acceptProposedAction()

    def dropEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        items = event.mimeData().property('myItems')
        event.accept()
        for item in items:
            self.makeItem(QSize(100, 100), item.data(Qt.UserRole + 1))

class DragListWidget(QListWidget):

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super(DragListWidget, self).__init__(*args, **kwargs)
        self.resize(400, 400)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setEditTriggers(self.NoEditTriggers)
        self.setDragEnabled(True)
        self.setDragDropMode(self.DragOnly)
        self.setDefaultDropAction(Qt.IgnoreAction)
        self.setSelectionMode(self.ContiguousSelection)
        self.setFlow(self.LeftToRight)
        self.setWrapping(True)
        self.setResizeMode(self.Adjust)
        self.setSpacing(5)
        self._rubberPos = None
        self._rubberBand = QRubberBand(QRubberBand.Rectangle, self)
        self.initItems()

    def startDrag(self, supportedActions):
        if False:
            print('Hello World!')
        items = self.selectedItems()
        drag = QDrag(self)
        mimeData = self.mimeData(items)
        mimeData.setProperty('myItems', items)
        drag.setMimeData(mimeData)
        pixmap = QPixmap(self.viewport().visibleRegion().boundingRect().size())
        pixmap.fill(Qt.transparent)
        painter = QPainter()
        painter.begin(pixmap)
        for item in items:
            rect = self.visualRect(self.indexFromItem(item))
            painter.drawPixmap(rect, self.viewport().grab(rect))
        painter.end()
        drag.setPixmap(pixmap)
        drag.setHotSpot(self.viewport().mapFromGlobal(QCursor.pos()))
        drag.exec_(supportedActions)

    def mousePressEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(DragListWidget, self).mousePressEvent(event)
        if event.buttons() != Qt.LeftButton or self.itemAt(event.pos()):
            return
        self._rubberPos = event.pos()
        self._rubberBand.setGeometry(QRect(self._rubberPos, QSize()))
        self._rubberBand.show()

    def mouseReleaseEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        super(DragListWidget, self).mouseReleaseEvent(event)
        self._rubberPos = None
        self._rubberBand.hide()

    def mouseMoveEvent(self, event):
        if False:
            print('Hello World!')
        super(DragListWidget, self).mouseMoveEvent(event)
        if self._rubberPos:
            pos = event.pos()
            (lx, ly) = (self._rubberPos.x(), self._rubberPos.y())
            (rx, ry) = (pos.x(), pos.y())
            size = QSize(abs(rx - lx), abs(ry - ly))
            self._rubberBand.setGeometry(QRect(QPoint(min(lx, rx), min(ly, ry)), size))

    def makeItem(self, size, cname):
        if False:
            return 10
        item = QListWidgetItem(self)
        item.setData(Qt.UserRole + 1, cname)
        item.setSizeHint(size)
        label = QLabel(self)
        label.setMargin(2)
        label.resize(size)
        pixmap = QPixmap(size.scaled(96, 96, Qt.IgnoreAspectRatio))
        pixmap.fill(QColor(cname))
        label.setPixmap(pixmap)
        self.setItemWidget(item, label)

    def initItems(self):
        if False:
            print('Hello World!')
        size = QSize(100, 100)
        for cname in QColor.colorNames():
            self.makeItem(size, cname)
if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    app.setStyleSheet('QListWidget {\n        outline: 0px;\n        background-color: transparent;\n    }\n    QListWidget::item:selected {\n        border-radius: 2px;\n        border: 1px solid rgb(0, 170, 255);\n    }\n    QListWidget::item:selected:!active {\n        border-radius: 2px;\n        border: 1px solid transparent;\n    }\n    QListWidget::item:selected:active {\n        border-radius: 2px;\n        border: 1px solid rgb(0, 170, 255);\n    }\n    QListWidget::item:hover {\n        border-radius: 2px;\n        border: 1px solid rgb(0, 170, 255);\n    }')
    wa = DragListWidget()
    wa.show()
    wo = DropListWidget()
    wo.show()
    sys.exit(app.exec_())