from enum import Enum
from typing import Union
from PyQt5.QtCore import Qt, QPropertyAnimation, QPoint, QParallelAnimationGroup, QEasingCurve, QMargins, QRectF, QObject, QSize, pyqtSignal
from PyQt5.QtGui import QPixmap, QPainter, QColor, QCursor, QIcon, QImage, QPainterPath, QBrush, QMovie, QImageReader
from PyQt5.QtWidgets import QWidget, QGraphicsDropShadowEffect, QLabel, QHBoxLayout, QVBoxLayout, QApplication
from ...common.auto_wrap import TextWrap
from ...common.style_sheet import isDarkTheme, FluentStyleSheet
from ...common.icon import FluentIconBase, drawIcon, FluentIcon
from .button import TransparentToolButton
from .label import ImageLabel

class FlyoutAnimationType(Enum):
    """ Flyout animation type """
    PULL_UP = 0
    DROP_DOWN = 1
    SLIDE_LEFT = 2
    SLIDE_RIGHT = 3
    FADE_IN = 4
    NONE = 5

class IconWidget(QWidget):

    def __init__(self, icon, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self.setFixedSize(36, 54)
        self.icon = icon

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        if not self.icon:
            return
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        rect = QRectF(8, (self.height() - 20) / 2, 20, 20)
        drawIcon(self.icon, painter, rect)

class FlyoutViewBase(QWidget):
    """ Flyout view base class """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)

    def addWidget(self, widget: QWidget, stretch=0, align=Qt.AlignLeft):
        if False:
            for i in range(10):
                print('nop')
        raise NotImplementedError

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setBrush(QColor(40, 40, 40) if isDarkTheme() else QColor(248, 248, 248))
        painter.setPen(QColor(23, 23, 23) if isDarkTheme() else QColor(195, 195, 195))
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 8, 8)

class FlyoutView(FlyoutViewBase):
    """ Flyout view """
    closed = pyqtSignal()

    def __init__(self, title: str, content: str, icon: Union[FluentIconBase, QIcon, str]=None, image: Union[str, QPixmap, QImage]=None, isClosable=False, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent=parent)
        '\n        Parameters\n        ----------\n        title: str\n            the title of teaching tip\n\n        content: str\n            the content of teaching tip\n\n        icon: InfoBarIcon | FluentIconBase | QIcon | str\n            the icon of teaching tip\n\n        image: str | QPixmap | QImage\n            the image of teaching tip\n\n        isClosable: bool\n            whether to show the close button\n\n        parent: QWidget\n            parent widget\n        '
        self.icon = icon
        self.title = title
        self.image = image
        self.content = content
        self.isClosable = isClosable
        self.vBoxLayout = QVBoxLayout(self)
        self.viewLayout = QHBoxLayout()
        self.widgetLayout = QVBoxLayout()
        self.titleLabel = QLabel(title, self)
        self.contentLabel = QLabel(content, self)
        self.iconWidget = IconWidget(icon, self)
        self.imageLabel = ImageLabel(self)
        self.closeButton = TransparentToolButton(FluentIcon.CLOSE, self)
        self.__initWidgets()

    def __initWidgets(self):
        if False:
            return 10
        self.imageLabel.setImage(self.image)
        self.closeButton.setFixedSize(32, 32)
        self.closeButton.setIconSize(QSize(12, 12))
        self.closeButton.setVisible(self.isClosable)
        self.titleLabel.setVisible(bool(self.title))
        self.contentLabel.setVisible(bool(self.content))
        self.iconWidget.setHidden(self.icon is None)
        self.closeButton.clicked.connect(self.closed)
        self.titleLabel.setObjectName('titleLabel')
        self.contentLabel.setObjectName('contentLabel')
        FluentStyleSheet.TEACHING_TIP.apply(self)
        self.__initLayout()

    def __initLayout(self):
        if False:
            i = 10
            return i + 15
        self.vBoxLayout.setContentsMargins(1, 1, 1, 1)
        self.widgetLayout.setContentsMargins(0, 8, 0, 8)
        self.viewLayout.setSpacing(4)
        self.widgetLayout.setSpacing(0)
        self.vBoxLayout.setSpacing(0)
        if not self.title or not self.content:
            self.iconWidget.setFixedHeight(36)
        self.vBoxLayout.addLayout(self.viewLayout)
        self.viewLayout.addWidget(self.iconWidget, 0, Qt.AlignTop)
        self._adjustText()
        self.widgetLayout.addWidget(self.titleLabel)
        self.widgetLayout.addWidget(self.contentLabel)
        self.viewLayout.addLayout(self.widgetLayout)
        self.closeButton.setVisible(self.isClosable)
        self.viewLayout.addWidget(self.closeButton, 0, Qt.AlignRight | Qt.AlignTop)
        margins = QMargins(6, 5, 6, 5)
        margins.setLeft(20 if not self.icon else 5)
        margins.setRight(20 if not self.isClosable else 6)
        self.viewLayout.setContentsMargins(margins)
        self._adjustImage()
        self._addImageToLayout()

    def addWidget(self, widget: QWidget, stretch=0, align=Qt.AlignLeft):
        if False:
            for i in range(10):
                print('nop')
        ' add widget to view '
        self.widgetLayout.addSpacing(8)
        self.widgetLayout.addWidget(widget, stretch, align)

    def _addImageToLayout(self):
        if False:
            return 10
        self.imageLabel.setBorderRadius(8, 8, 0, 0)
        self.imageLabel.setHidden(self.imageLabel.isNull())
        self.vBoxLayout.insertWidget(0, self.imageLabel)

    def _adjustText(self):
        if False:
            i = 10
            return i + 15
        w = min(900, QApplication.screenAt(QCursor.pos()).geometry().width() - 200)
        chars = max(min(w / 10, 120), 30)
        self.titleLabel.setText(TextWrap.wrap(self.title, chars, False)[0])
        chars = max(min(w / 9, 120), 30)
        self.contentLabel.setText(TextWrap.wrap(self.content, chars, False)[0])

    def _adjustImage(self):
        if False:
            while True:
                i = 10
        w = self.vBoxLayout.sizeHint().width() - 2
        self.imageLabel.scaledToWidth(w)

    def showEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().showEvent(e)
        self._adjustImage()
        self.adjustSize()

class Flyout(QWidget):
    """ Flyout """
    closed = pyqtSignal()

    def __init__(self, view: FlyoutViewBase, parent=None, isDeleteOnClose=True):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self.view = view
        self.hBoxLayout = QHBoxLayout(self)
        self.aniManager = None
        self.isDeleteOnClose = isDeleteOnClose
        self.hBoxLayout.setContentsMargins(15, 8, 15, 20)
        self.hBoxLayout.addWidget(self.view)
        self.setShadowEffect()
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint | Qt.NoDropShadowWindowHint)

    def setShadowEffect(self, blurRadius=35, offset=(0, 8)):
        if False:
            i = 10
            return i + 15
        ' add shadow to dialog '
        color = QColor(0, 0, 0, 80 if isDarkTheme() else 30)
        self.shadowEffect = QGraphicsDropShadowEffect(self.view)
        self.shadowEffect.setBlurRadius(blurRadius)
        self.shadowEffect.setOffset(*offset)
        self.shadowEffect.setColor(color)
        self.view.setGraphicsEffect(None)
        self.view.setGraphicsEffect(self.shadowEffect)

    def closeEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        if self.isDeleteOnClose:
            self.deleteLater()
        super().closeEvent(e)
        self.closed.emit()

    def exec(self, pos: QPoint, aniType=FlyoutAnimationType.PULL_UP):
        if False:
            for i in range(10):
                print('nop')
        ' show calendar view '
        self.aniManager = FlyoutAnimationManager.make(aniType, self)
        self.show()
        self.aniManager.exec(pos)

    @classmethod
    def make(cls, view: FlyoutViewBase, target: Union[QWidget, QPoint]=None, parent=None, aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        if False:
            while True:
                i = 10
        ' create and show a flyout\n\n        Parameters\n        ----------\n        view: FlyoutViewBase\n            flyout view\n\n        target: QWidget | QPoint\n            the target widget or position to show flyout\n\n        parent: QWidget\n            parent window\n\n        aniType: FlyoutAnimationType\n            flyout animation type\n\n        isDeleteOnClose: bool\n            whether delete flyout automatically when flyout is closed\n        '
        w = cls(view, parent, isDeleteOnClose)
        if target is None:
            return w
        w.show()
        if isinstance(target, QWidget):
            target = FlyoutAnimationManager.make(aniType, w).position(target)
        w.exec(target, aniType)
        return w

    @classmethod
    def create(cls, title: str, content: str, icon: Union[FluentIconBase, QIcon, str]=None, image: Union[str, QPixmap, QImage]=None, isClosable=False, target: Union[QWidget, QPoint]=None, parent=None, aniType=FlyoutAnimationType.PULL_UP, isDeleteOnClose=True):
        if False:
            for i in range(10):
                print('nop')
        ' create and show a flyout using the default view\n\n        Parameters\n        ----------\n        title: str\n            the title of teaching tip\n\n        content: str\n            the content of teaching tip\n\n        icon: InfoBarIcon | FluentIconBase | QIcon | str\n            the icon of teaching tip\n\n        image: str | QPixmap | QImage\n            the image of teaching tip\n\n        isClosable: bool\n            whether to show the close button\n\n        target: QWidget | QPoint\n            the target widget or position to show flyout\n\n        parent: QWidget\n            parent window\n\n        aniType: FlyoutAnimationType\n            flyout animation type\n\n        isDeleteOnClose: bool\n            whether delete flyout automatically when flyout is closed\n        '
        view = FlyoutView(title, content, icon, image, isClosable)
        w = cls.make(view, target, parent, aniType, isDeleteOnClose)
        view.closed.connect(w.close)
        return w

class FlyoutAnimationManager(QObject):
    """ Flyout animation manager """
    managers = {}

    def __init__(self, flyout: Flyout):
        if False:
            return 10
        super().__init__()
        self.flyout = flyout
        self.aniGroup = QParallelAnimationGroup(self)
        self.slideAni = QPropertyAnimation(flyout, b'pos', self)
        self.opacityAni = QPropertyAnimation(flyout, b'windowOpacity', self)
        self.slideAni.setDuration(187)
        self.opacityAni.setDuration(187)
        self.opacityAni.setStartValue(0)
        self.opacityAni.setEndValue(1)
        self.slideAni.setEasingCurve(QEasingCurve.OutQuad)
        self.opacityAni.setEasingCurve(QEasingCurve.OutQuad)
        self.aniGroup.addAnimation(self.slideAni)
        self.aniGroup.addAnimation(self.opacityAni)

    @classmethod
    def register(cls, name):
        if False:
            return 10
        ' register menu animation manager\n\n        Parameters\n        ----------\n        name: Any\n            the name of manager, it should be unique\n        '

        def wrapper(Manager):
            if False:
                i = 10
                return i + 15
            if name not in cls.managers:
                cls.managers[name] = Manager
            return Manager
        return wrapper

    def exec(self, pos: QPoint):
        if False:
            print('Hello World!')
        ' start animation '
        raise NotImplementedError

    def _adjustPosition(self, pos):
        if False:
            return 10
        rect = QApplication.screenAt(QCursor.pos()).availableGeometry()
        (w, h) = (self.flyout.sizeHint().width() + 5, self.flyout.sizeHint().height())
        x = max(rect.left(), min(pos.x(), rect.right() - w))
        y = max(rect.top(), min(pos.y() - 4, rect.bottom() - h + 5))
        return QPoint(x, y)

    def position(self, target: QWidget):
        if False:
            print('Hello World!')
        ' return the top left position relative to the target '
        raise NotImplementedError

    @classmethod
    def make(cls, aniType: FlyoutAnimationType, flyout: Flyout):
        if False:
            while True:
                i = 10
        ' mask animation manager '
        if aniType not in cls.managers:
            raise ValueError(f'`{aniType}` is an invalid animation type.')
        return cls.managers[aniType](flyout)

@FlyoutAnimationManager.register(FlyoutAnimationType.PULL_UP)
class PullUpFlyoutAnimationManager(FlyoutAnimationManager):
    """ Pull up flyout animation manager """

    def position(self, target: QWidget):
        if False:
            return 10
        w = self.flyout
        pos = target.mapToGlobal(QPoint())
        x = pos.x() + target.width() // 2 - w.sizeHint().width() // 2
        y = pos.y() - w.sizeHint().height() + w.layout().contentsMargins().bottom()
        return QPoint(x, y)

    def exec(self, pos: QPoint):
        if False:
            for i in range(10):
                print('nop')
        pos = self._adjustPosition(pos)
        self.slideAni.setStartValue(pos + QPoint(0, 8))
        self.slideAni.setEndValue(pos)
        self.aniGroup.start()

@FlyoutAnimationManager.register(FlyoutAnimationType.DROP_DOWN)
class DropDownFlyoutAnimationManager(FlyoutAnimationManager):
    """ Drop down flyout animation manager """

    def position(self, target: QWidget):
        if False:
            for i in range(10):
                print('nop')
        w = self.flyout
        pos = target.mapToGlobal(QPoint(0, target.height()))
        x = pos.x() + target.width() // 2 - w.sizeHint().width() // 2
        y = pos.y() - w.layout().contentsMargins().top() + 8
        return QPoint(x, y)

    def exec(self, pos: QPoint):
        if False:
            for i in range(10):
                print('nop')
        pos = self._adjustPosition(pos)
        self.slideAni.setStartValue(pos - QPoint(0, 8))
        self.slideAni.setEndValue(pos)
        self.aniGroup.start()

@FlyoutAnimationManager.register(FlyoutAnimationType.SLIDE_LEFT)
class SlideLeftFlyoutAnimationManager(FlyoutAnimationManager):
    """ Slide left flyout animation manager """

    def position(self, target: QWidget):
        if False:
            i = 10
            return i + 15
        w = self.flyout
        pos = target.mapToGlobal(QPoint(0, 0))
        x = pos.x() - w.sizeHint().width() + 8
        y = pos.y() - w.sizeHint().height() // 2 + target.height() // 2 + w.layout().contentsMargins().top()
        return QPoint(x, y)

    def exec(self, pos: QPoint):
        if False:
            for i in range(10):
                print('nop')
        pos = self._adjustPosition(pos)
        self.slideAni.setStartValue(pos + QPoint(8, 0))
        self.slideAni.setEndValue(pos)
        self.aniGroup.start()

@FlyoutAnimationManager.register(FlyoutAnimationType.SLIDE_RIGHT)
class SlideRightFlyoutAnimationManager(FlyoutAnimationManager):
    """ Slide right flyout animation manager """

    def position(self, target: QWidget):
        if False:
            while True:
                i = 10
        w = self.flyout
        pos = target.mapToGlobal(QPoint(0, 0))
        x = pos.x() + target.width() - 8
        y = pos.y() - w.sizeHint().height() // 2 + target.height() // 2 + w.layout().contentsMargins().top()
        return QPoint(x, y)

    def exec(self, pos: QPoint):
        if False:
            i = 10
            return i + 15
        pos = self._adjustPosition(pos)
        self.slideAni.setStartValue(pos - QPoint(8, 0))
        self.slideAni.setEndValue(pos)
        self.aniGroup.start()

@FlyoutAnimationManager.register(FlyoutAnimationType.FADE_IN)
class FadeInFlyoutAnimationManager(FlyoutAnimationManager):
    """ Fade in flyout animation manager """

    def position(self, target: QWidget):
        if False:
            print('Hello World!')
        w = self.flyout
        pos = target.mapToGlobal(QPoint())
        x = pos.x() + target.width() // 2 - w.sizeHint().width() // 2
        y = pos.y() - w.sizeHint().height() + w.layout().contentsMargins().bottom()
        return QPoint(x, y)

    def exec(self, pos: QPoint):
        if False:
            i = 10
            return i + 15
        self.flyout.move(self._adjustPosition(pos))
        self.aniGroup.removeAnimation(self.slideAni)
        self.aniGroup.start()

@FlyoutAnimationManager.register(FlyoutAnimationType.NONE)
class DummyFlyoutAnimationManager(FlyoutAnimationManager):
    """ Dummy flyout animation manager """

    def exec(self, pos: QPoint):
        if False:
            print('Hello World!')
        ' start animation '
        self.flyout.move(self._adjustPosition(pos))

    def position(self, target: QWidget):
        if False:
            i = 10
            return i + 15
        ' return the top left position relative to the target '
        m = self.flyout.hBoxLayout.contentsMargins()
        return target.mapToGlobal(QPoint(-m.left(), -self.flyout.sizeHint().height() + m.bottom() - 8))