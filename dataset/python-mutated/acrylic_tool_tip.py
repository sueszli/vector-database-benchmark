from PyQt5.QtCore import QRect, QRectF
from PyQt5.QtGui import QPainterPath
from PyQt5.QtWidgets import QApplication, QFrame
from .acrylic_widget import AcrylicWidget
from ..widgets.tool_tip import ToolTip, ToolTipFilter

class AcrylicToolTipContainer(AcrylicWidget, QFrame):
    """ Acrylic tool tip container """

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent=parent)
        self.setProperty('transparent', True)

    def acrylicClipPath(self):
        if False:
            while True:
                i = 10
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 3, 3)
        return path

class AcrylicToolTip(ToolTip):
    """ Acrylic tool tip """

    def _createContainer(self):
        if False:
            for i in range(10):
                print('nop')
        return AcrylicToolTipContainer(self)

    def showEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        pos = self.pos() + self.container.pos()
        self.container.acrylicBrush.grabImage(QRect(pos, self.container.size()))
        return super().showEvent(e)

class AcrylicToolTipFilter(ToolTipFilter):
    """ Acrylic tool tip filter """

    def _createToolTip(self):
        if False:
            i = 10
            return i + 15
        return AcrylicToolTip(self.parent().toolTip(), self.parent().window())