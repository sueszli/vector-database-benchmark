"""
Created on 2018年4月30日
@author: Irony
@site: https://pyqt.site , https://github.com/PyQt5
@email: 892768447@qq.com
@file: NewFramelessWindow
@description:
"""
import sys
try:
    from PyQt5.QtCore import QEvent, QObject, QPoint, Qt, QTimer
    from PyQt5.QtGui import QColor, QMouseEvent, QPainter, QWindow
    from PyQt5.QtWidgets import QApplication, QMessageBox, QWidget
except ImportError:
    from PySide2.QtCore import QTimer, Qt, QEvent, QObject, QPoint
    from PySide2.QtGui import QWindow, QPainter, QColor, QMouseEvent
    from PySide2.QtWidgets import QApplication, QWidget, QMessageBox
from Lib.ui_frameless import Ui_FormFrameless

class FramelessObject(QObject):
    Margins = 3
    TitleHeight = 36
    Widgets = set()

    @classmethod
    def set_margins(cls, margins):
        if False:
            while True:
                i = 10
        cls.Margins = margins

    @classmethod
    def set_title_height(cls, height):
        if False:
            for i in range(10):
                print('nop')
        cls.TitleHeight = height

    @classmethod
    def add_widget(cls, widget):
        if False:
            while True:
                i = 10
        cls.Widgets.add(widget)

    @classmethod
    def del_widget(cls, widget):
        if False:
            while True:
                i = 10
        if widget in cls.Widgets:
            cls.Widgets.remove(widget)

    def _get_edges(self, pos, width, height):
        if False:
            print('Hello World!')
        '根据坐标获取方向\n        :param pos: QPoint\n        :param width: int\n        :param height: int\n        :return: Qt.Edges\n        '
        edge = 0
        (x, y) = (pos.x(), pos.y())
        if y <= self.Margins:
            edge |= Qt.TopEdge
        if x <= self.Margins:
            edge |= Qt.LeftEdge
        if x >= width - self.Margins:
            edge |= Qt.RightEdge
        if y >= height - self.Margins:
            edge |= Qt.BottomEdge
        return edge

    def _get_cursor(self, edges):
        if False:
            i = 10
            return i + 15
        '调整鼠标样式\n        :param edges: int or None\n        :return: Qt.CursorShape\n        '
        if edges == Qt.LeftEdge | Qt.TopEdge or edges == Qt.RightEdge | Qt.BottomEdge:
            return Qt.SizeFDiagCursor
        elif edges == Qt.RightEdge | Qt.TopEdge or edges == Qt.LeftEdge | Qt.BottomEdge:
            return Qt.SizeBDiagCursor
        elif edges == Qt.LeftEdge or edges == Qt.RightEdge:
            return Qt.SizeHorCursor
        elif edges == Qt.TopEdge or edges == Qt.BottomEdge:
            return Qt.SizeVerCursor
        return Qt.ArrowCursor

    def is_titlebar(self, pos):
        if False:
            while True:
                i = 10
        '判断是否是标题栏\n        :param pos: QPoint\n        :return: bool\n        '
        return pos.y() <= self.TitleHeight

    def moveOrResize(self, window, pos, width, height):
        if False:
            return 10
        edges = self._get_edges(pos, width, height)
        if edges:
            if window.windowState() == Qt.WindowNoState:
                window.startSystemResize(edges)
        elif self.is_titlebar(pos):
            window.startSystemMove()
            QApplication.instance().postEvent(window, QMouseEvent(QEvent.MouseButtonRelease, QPoint(-1, -1), Qt.LeftButton, Qt.NoButton, Qt.NoModifier))

    def eventFilter(self, obj, event):
        if False:
            print('Hello World!')
        if obj.isWindowType():
            if event.type() == QEvent.MouseMove and obj.windowState() == Qt.WindowNoState:
                obj.setCursor(self._get_cursor(self._get_edges(event.pos(), obj.width(), obj.height())))
            elif event.type() == QEvent.TouchUpdate:
                self.moveOrResize(obj, event.pos(), obj.width(), obj.height())
        elif obj in self.Widgets and isinstance(event, QMouseEvent) and (event.button() == Qt.LeftButton):
            if event.type() == QEvent.MouseButtonDblClick:
                if self.is_titlebar(event.pos()):
                    if obj.windowState() == Qt.WindowFullScreen:
                        pass
                    elif obj.windowState() == Qt.WindowMaximized:
                        obj.showNormal()
                    else:
                        obj.showMaximized()
            elif event.type() == QEvent.MouseButtonPress:
                self.moveOrResize(obj.windowHandle(), event.pos(), obj.width(), obj.height())
        return False

class FramelessWindow(QWidget, Ui_FormFrameless):

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super(FramelessWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.setWindowFlags(self.windowFlags() | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setMouseTracking(True)
        self.buttonNormal.setVisible(False)
        self.buttonMinimum.clicked.connect(self.showMinimized)
        self.buttonMaximum.clicked.connect(self.showMaximized)
        self.buttonNormal.clicked.connect(self.showNormal)
        self.buttonClose.clicked.connect(self.close)
        self.setStyleSheet('#widgetTitleBar{background: rgb(232, 232, 232);}')

    def showMinimized(self):
        if False:
            for i in range(10):
                print('nop')
        flags = self.windowFlags()
        if sys.platform == 'darwin':
            self.setWindowFlags((self.windowFlags() | Qt.CustomizeWindowHint) & ~Qt.WindowTitleHint)
        super(FramelessWindow, self).showMinimized()
        if sys.platform == 'darwin':
            self.setWindowFlags(flags)
            self.show()

    def changeEvent(self, event):
        if False:
            print('Hello World!')
        '窗口状态改变\n        :param event:\n        '
        super(FramelessWindow, self).changeEvent(event)
        visible = self.isMaximized()
        self.buttonMaximum.setVisible(not visible)
        self.buttonNormal.setVisible(visible)
        if visible:
            self.layout().setContentsMargins(0, 0, 0, 0)
        else:
            m = FramelessObject.Margins
            self.layout().setContentsMargins(m, m, m, m)

    def paintEvent(self, event):
        if False:
            return 10
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(255, 255, 255, 1))
if __name__ == '__main__':
    import cgitb
    import sys
    cgitb.enable(format='text')
    app = QApplication(sys.argv)
    if not hasattr(QWindow, 'startSystemMove'):
        QWindow.startSystemResize()
        QMessageBox.critical(None, '错误', '当前Qt版本不支持该例子')
        QTimer.singleShot(100, app.quit)
    else:
        fo = FramelessObject()
        app.installEventFilter(fo)
        w1 = FramelessWindow()
        fo.add_widget(w1)
        w1.show()
        w2 = FramelessWindow()
        fo.add_widget(w2)
        w2.show()
    sys.exit(app.exec_())