import logging
from PyQt5.QtCore import QPoint, pyqtSignal
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QStyle, QStyleOption, QWidget
from tribler.gui.sentry_mixin import AddBreadcrumbOnShowMixin
from tribler.gui.utilities import connect

class DialogContainer(AddBreadcrumbOnShowMixin, QWidget):
    close_event = pyqtSignal()

    def __init__(self, parent, left_right_margin=100):
        if False:
            print('Hello World!')
        QWidget.__init__(self, parent)
        self.setStyleSheet('background-color: rgba(30, 30, 30, 0.75);')
        self.dialog_widget = QWidget(self)
        self.left_right_margin = left_right_margin
        self.closed = False
        self.logger = logging.getLogger(self.__class__.__name__)
        connect(self.window().resize_event, self.on_main_window_resize)

    def paintEvent(self, _):
        if False:
            while True:
                i = 10
        opt = QStyleOption()
        opt.initFrom(self)
        painter = QPainter(self)
        self.style().drawPrimitive(QStyle.PE_Widget, opt, painter, self)

    def close_dialog(self, checked=False):
        if False:
            i = 10
            return i + 15
        if self.closed:
            return
        try:
            self.close_event.emit()
            self.setParent(None)
            self.deleteLater()
            self.closed = True
        except RuntimeError:
            pass

    def mouseReleaseEvent(self, qevent):
        if False:
            for i in range(10):
                print('nop')
        if not self.dialog_widget.geometry().contains(qevent.localPos().toPoint()):
            self.close_dialog()

    def showEvent(self, _):
        if False:
            i = 10
            return i + 15
        self.on_main_window_resize()

    def on_main_window_resize(self):
        if False:
            print('Hello World!')
        try:
            if not self or not self.parentWidget():
                return
            self.setFixedSize(self.parentWidget().size())
            self.dialog_widget.setFixedWidth(self.width() - self.left_right_margin)
            self.dialog_widget.move(QPoint(int(self.geometry().center().x() - self.dialog_widget.geometry().width() // 2), int(self.geometry().center().y() - self.dialog_widget.geometry().height() // 2)))
        except RuntimeError:
            pass