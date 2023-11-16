from PyQt5.QtCore import QSize, QPoint, Qt, QEvent, QRect
from PyQt5.QtGui import QResizeEvent
from PyQt5.QtWidgets import QLayout, QWidget

class ExpandLayout(QLayout):
    """ Expand layout """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.__items = []
        self.__widgets = []

    def addWidget(self, widget: QWidget):
        if False:
            return 10
        if widget in self.__widgets:
            return
        self.__widgets.append(widget)
        widget.installEventFilter(self)

    def addItem(self, item):
        if False:
            return 10
        self.__items.append(item)

    def count(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.__items)

    def itemAt(self, index):
        if False:
            i = 10
            return i + 15
        if 0 <= index < len(self.__items):
            return self.__items[index]
        return None

    def takeAt(self, index):
        if False:
            return 10
        if 0 <= index < len(self.__items):
            self.__widgets.pop(index)
            return self.__items.pop(index)
        return None

    def expandingDirections(self):
        if False:
            i = 10
            return i + 15
        return Qt.Vertical

    def hasHeightForWidth(self):
        if False:
            return 10
        return True

    def heightForWidth(self, width):
        if False:
            i = 10
            return i + 15
        ' get the minimal height according to width '
        return self.__doLayout(QRect(0, 0, width, 0), False)

    def setGeometry(self, rect):
        if False:
            while True:
                i = 10
        super().setGeometry(rect)
        self.__doLayout(rect, True)

    def sizeHint(self):
        if False:
            while True:
                i = 10
        return self.minimumSize()

    def minimumSize(self):
        if False:
            return 10
        size = QSize()
        for w in self.__widgets:
            size = size.expandedTo(w.minimumSize())
        m = self.contentsMargins()
        size += QSize(m.left() + m.right(), m.top() + m.bottom())
        return size

    def __doLayout(self, rect, move):
        if False:
            return 10
        ' adjust widgets position according to the window size '
        margin = self.contentsMargins()
        x = rect.x() + margin.left()
        y = rect.y() + margin.top() + margin.bottom()
        width = rect.width() - margin.left() - margin.right()
        for (i, w) in enumerate(self.__widgets):
            if w.isHidden():
                continue
            y += (i > 0) * self.spacing()
            if move:
                w.setGeometry(QRect(QPoint(x, y), QSize(width, w.height())))
            y += w.height()
        return y - rect.y()

    def eventFilter(self, obj, e):
        if False:
            i = 10
            return i + 15
        if obj in self.__widgets:
            if e.type() == QEvent.Resize:
                re = QResizeEvent(e)
                ds = re.size() - re.oldSize()
                if ds.height() != 0 and ds.width() == 0:
                    w = self.parentWidget()
                    w.resize(w.width(), w.height() + ds.height())
        return super().eventFilter(obj, e)