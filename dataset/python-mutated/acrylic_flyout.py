from typing import Union
from PyQt5.QtCore import QPoint, Qt, QRect, QRectF
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPainterPath, QIcon, QImage
from PyQt5.QtWidgets import QWidget
from ...common.style_sheet import isDarkTheme
from ...common.icon import FluentIconBase
from ..widgets.flyout import FlyoutAnimationType, FlyoutViewBase, FlyoutView, Flyout, FlyoutAnimationManager
from .acrylic_widget import AcrylicWidget

class AcrylicFlyoutViewBase(AcrylicWidget, FlyoutViewBase):
    """ Acrylic flyout view base """

    def acrylicClipPath(self):
        if False:
            print('Hello World!')
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 8, 8)
        return path

    def paintEvent(self, e):
        if False:
            while True:
                i = 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        self._drawAcrylic(painter)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QColor(23, 23, 23) if isDarkTheme() else QColor(195, 195, 195))
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)

class AcrylicFlyoutView(AcrylicWidget, FlyoutView):
    """ Acrylic flyout view """

    def acrylicClipPath(self):
        if False:
            print('Hello World!')
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 8, 8)
        return path

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        self._drawAcrylic(painter)
        painter.setBrush(Qt.NoBrush)
        painter.setPen(QColor(23, 23, 23) if isDarkTheme() else QColor(195, 195, 195))
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)

class AcrylicFlyout(Flyout):
    """ Acrylic flyout """

    @classmethod
    def create(cls, title: str, content: str, icon: Union[FluentIconBase, QIcon, str]=None, image: Union[str, QPixmap, QImage]=None, isClosable=False, target: Union[QWidget, QPoint]=None, parent=None, aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        if False:
            while True:
                i = 10
        ' create and show a flyout using the default view\n\n        Parameters\n        ----------\n        title: str\n            the title of teaching tip\n\n        content: str\n            the content of teaching tip\n\n        icon: InfoBarIcon | FluentIconBase | QIcon | str\n            the icon of teaching tip\n\n        image: str | QPixmap | QImage\n            the image of teaching tip\n\n        isClosable: bool\n            whether to show the close button\n\n        target: QWidget | QPoint\n            the target widget or position to show flyout\n\n        parent: QWidget\n            parent window\n\n        aniType: FlyoutAnimationType\n            flyout animation type\n\n        isDeleteOnClose: bool\n            whether delete flyout automatically when flyout is closed\n        '
        view = AcrylicFlyoutView(title, content, icon, image, isClosable)
        w = cls.make(view, target, parent, aniType, isDeleteOnClose)
        view.closed.connect(w.close)
        return w

    def exec(self, pos: QPoint, aniType=FlyoutAnimationType.PULL_UP):
        if False:
            i = 10
            return i + 15
        ' show calendar view '
        self.aniManager = FlyoutAnimationManager.make(aniType, self)
        if isinstance(self.view, AcrylicWidget):
            pos = self.aniManager._adjustPosition(pos)
            self.view.acrylicBrush.grabImage(QRect(pos, self.layout().sizeHint()))
        self.show()
        self.aniManager.exec(pos)