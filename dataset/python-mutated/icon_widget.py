from typing import Union
from PyQt5.QtCore import pyqtProperty
from PyQt5.QtGui import QIcon, QPainter
from PyQt5.QtWidgets import QWidget
from ...common.icon import FluentIconBase, drawIcon, toQIcon
from ...common.overload import singledispatchmethod

class IconWidget(QWidget):
    """ Icon widget

    Constructors
    ------------
    * IconWidget(`parent`: QWidget = None)
    * IconWidget(`icon`: QIcon | str | FluentIconBase, `parent`: QWidget = None)
    """

    @singledispatchmethod
    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.setIcon(QIcon())

    @__init__.register
    def _(self, icon: FluentIconBase, parent: QWidget=None):
        if False:
            while True:
                i = 10
        self.__init__(parent)
        self.setIcon(icon)

    @__init__.register
    def _(self, icon: QIcon, parent: QWidget=None):
        if False:
            return 10
        self.__init__(parent)
        self.setIcon(icon)

    @__init__.register
    def _(self, icon: str, parent: QWidget=None):
        if False:
            print('Hello World!')
        self.__init__(parent)
        self.setIcon(icon)

    def getIcon(self):
        if False:
            print('Hello World!')
        return toQIcon(self._icon)

    def setIcon(self, icon: Union[str, QIcon, FluentIconBase]):
        if False:
            i = 10
            return i + 15
        self._icon = icon
        self.update()

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        drawIcon(self._icon, painter, self.rect())
    icon = pyqtProperty(QIcon, getIcon, setIcon)