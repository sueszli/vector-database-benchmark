import warnings
from ..Qt import QtCore, QtGui, QtWidgets
__all__ = ['VerticalLabel']

class VerticalLabel(QtWidgets.QLabel):

    def __init__(self, text, orientation='vertical', forceWidth=True):
        if False:
            i = 10
            return i + 15
        QtWidgets.QLabel.__init__(self, text)
        self.forceWidth = forceWidth
        self.orientation = None
        self.setOrientation(orientation)

    def setOrientation(self, o):
        if False:
            i = 10
            return i + 15
        if self.orientation == o:
            return
        self.orientation = o
        self.update()
        self.updateGeometry()

    def paintEvent(self, ev):
        if False:
            print('Hello World!')
        p = QtGui.QPainter(self)
        if self.orientation == 'vertical':
            p.rotate(-90)
            rgn = QtCore.QRect(-self.height(), 0, self.height(), self.width())
        else:
            rgn = self.contentsRect()
        align = self.alignment()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.hint = p.drawText(rgn, align, self.text())
        p.end()
        if self.orientation == 'vertical':
            self.setMaximumWidth(self.hint.height())
            self.setMinimumWidth(0)
            self.setMaximumHeight(16777215)
            if self.forceWidth:
                self.setMinimumHeight(self.hint.width())
            else:
                self.setMinimumHeight(0)
        else:
            self.setMaximumHeight(self.hint.height())
            self.setMinimumHeight(0)
            self.setMaximumWidth(16777215)
            if self.forceWidth:
                self.setMinimumWidth(self.hint.width())
            else:
                self.setMinimumWidth(0)

    def sizeHint(self):
        if False:
            for i in range(10):
                print('nop')
        if self.orientation == 'vertical':
            if hasattr(self, 'hint'):
                return QtCore.QSize(self.hint.height(), self.hint.width())
            else:
                return QtCore.QSize(19, 50)
        elif hasattr(self, 'hint'):
            return QtCore.QSize(self.hint.width(), self.hint.height())
        else:
            return QtCore.QSize(50, 19)