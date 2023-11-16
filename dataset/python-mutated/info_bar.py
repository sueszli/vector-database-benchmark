from enum import Enum
from typing import Union
import weakref
from PyQt5.QtCore import Qt, QEvent, QSize, QRectF, QObject, QPropertyAnimation, QEasingCurve, QTimer, pyqtSignal, QParallelAnimationGroup, QPoint
from PyQt5.QtGui import QPainter, QIcon, QColor
from PyQt5.QtWidgets import QWidget, QFrame, QLabel, QHBoxLayout, QVBoxLayout, QToolButton, QGraphicsOpacityEffect
from ...common.auto_wrap import TextWrap
from ...common.style_sheet import FluentStyleSheet, themeColor
from ...common.icon import FluentIconBase, Theme, isDarkTheme, writeSvg, drawSvgIcon, drawIcon
from ...common.icon import FluentIcon as FIF
from .button import TransparentToolButton

class InfoBarIcon(FluentIconBase, Enum):
    """ Info bar icon """
    INFORMATION = 'Info'
    SUCCESS = 'Success'
    WARNING = 'Warning'
    ERROR = 'Error'

    def path(self, theme=Theme.AUTO):
        if False:
            print('Hello World!')
        if theme == Theme.AUTO:
            color = 'dark' if isDarkTheme() else 'light'
        else:
            color = theme.value.lower()
        return f':/qfluentwidgets/images/info_bar/{self.value}_{color}.svg'

class InfoBarPosition(Enum):
    """ Info bar position """
    TOP = 0
    BOTTOM = 1
    TOP_LEFT = 2
    TOP_RIGHT = 3
    BOTTOM_LEFT = 4
    BOTTOM_RIGHT = 5
    NONE = 6

class InfoIconWidget(QWidget):
    """ Icon widget """

    def __init__(self, icon: InfoBarIcon, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self.setFixedSize(36, 36)
        self.icon = icon

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        rect = QRectF(10, 10, 15, 15)
        if self.icon != InfoBarIcon.INFORMATION:
            drawIcon(self.icon, painter, rect)
        else:
            drawIcon(self.icon, painter, rect, indexes=[0], fill=themeColor().name())

class InfoBar(QFrame):
    """ Information bar """
    closedSignal = pyqtSignal()

    def __init__(self, icon: Union[InfoBarIcon, FluentIconBase, QIcon, str], title: str, content: str, orient=Qt.Horizontal, isClosable=True, duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        icon: InfoBarIcon | FluentIconBase | QIcon | str\n            the icon of info bar\n\n        title: str\n            the title of info bar\n\n        content: str\n            the content of info bar\n\n        orient: Qt.Orientation\n            the layout direction of info bar, use `Qt.Horizontal` for short content\n\n        isClosable: bool\n            whether to show the close button\n\n        duraction: int\n            the time for info bar to display in milliseconds. If duration is less than zero,\n            info bar will never disappear.\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(parent=parent)
        self.title = title
        self.content = content
        self.orient = orient
        self.icon = icon
        self.duration = duration
        self.isClosable = isClosable
        self.position = position
        self.titleLabel = QLabel(self)
        self.contentLabel = QLabel(self)
        self.closeButton = TransparentToolButton(FIF.CLOSE, self)
        self.iconWidget = InfoIconWidget(icon)
        self.hBoxLayout = QHBoxLayout(self)
        self.textLayout = QHBoxLayout() if self.orient == Qt.Horizontal else QVBoxLayout()
        self.widgetLayout = QHBoxLayout() if self.orient == Qt.Horizontal else QVBoxLayout()
        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.opacityAni = QPropertyAnimation(self.opacityEffect, b'opacity', self)
        self.lightBackgroundColor = None
        self.darkBackgroundColor = None
        self.__initWidget()

    def __initWidget(self):
        if False:
            i = 10
            return i + 15
        self.opacityEffect.setOpacity(1)
        self.setGraphicsEffect(self.opacityEffect)
        self.closeButton.setFixedSize(36, 36)
        self.closeButton.setIconSize(QSize(12, 12))
        self.closeButton.setCursor(Qt.PointingHandCursor)
        self.closeButton.setVisible(self.isClosable)
        self.__setQss()
        self.__initLayout()
        self.closeButton.clicked.connect(self.close)

    def __initLayout(self):
        if False:
            for i in range(10):
                print('nop')
        self.hBoxLayout.setContentsMargins(6, 6, 6, 6)
        self.hBoxLayout.setSizeConstraint(QVBoxLayout.SetMinimumSize)
        self.textLayout.setSizeConstraint(QHBoxLayout.SetMinimumSize)
        self.textLayout.setAlignment(Qt.AlignTop)
        self.textLayout.setContentsMargins(1, 8, 0, 8)
        self.hBoxLayout.setSpacing(0)
        self.textLayout.setSpacing(5)
        self.hBoxLayout.addWidget(self.iconWidget, 0, Qt.AlignTop | Qt.AlignLeft)
        self.textLayout.addWidget(self.titleLabel, 1, Qt.AlignTop)
        self.titleLabel.setVisible(bool(self.title))
        if self.orient == Qt.Horizontal:
            self.textLayout.addSpacing(7)
        self.textLayout.addWidget(self.contentLabel, 1, Qt.AlignTop)
        self.contentLabel.setVisible(bool(self.content))
        self.hBoxLayout.addLayout(self.textLayout)
        if self.orient == Qt.Horizontal:
            self.hBoxLayout.addLayout(self.widgetLayout)
            self.widgetLayout.setSpacing(10)
        else:
            self.textLayout.addLayout(self.widgetLayout)
        self.hBoxLayout.addSpacing(12)
        self.hBoxLayout.addWidget(self.closeButton, 0, Qt.AlignTop | Qt.AlignLeft)
        self._adjustText()

    def __setQss(self):
        if False:
            while True:
                i = 10
        self.titleLabel.setObjectName('titleLabel')
        self.contentLabel.setObjectName('contentLabel')
        if isinstance(self.icon, Enum):
            self.setProperty('type', self.icon.value)
        FluentStyleSheet.INFO_BAR.apply(self)

    def __fadeOut(self):
        if False:
            while True:
                i = 10
        ' fade out '
        self.opacityAni.setDuration(200)
        self.opacityAni.setStartValue(1)
        self.opacityAni.setEndValue(0)
        self.opacityAni.finished.connect(self.close)
        self.opacityAni.start()

    def _adjustText(self):
        if False:
            i = 10
            return i + 15
        w = 900 if not self.parent() else self.parent().width() - 50
        chars = max(min(w / 10, 120), 30)
        self.titleLabel.setText(TextWrap.wrap(self.title, chars, False)[0])
        chars = max(min(w / 9, 120), 30)
        self.contentLabel.setText(TextWrap.wrap(self.content, chars, False)[0])
        self.adjustSize()

    def addWidget(self, widget: QWidget, stretch=0):
        if False:
            print('Hello World!')
        ' add widget to info bar '
        self.widgetLayout.addSpacing(6)
        align = Qt.AlignTop if self.orient == Qt.Vertical else Qt.AlignVCenter
        self.widgetLayout.addWidget(widget, stretch, Qt.AlignLeft | align)

    def setCustomBackgroundColor(self, light, dark):
        if False:
            i = 10
            return i + 15
        ' set the custom background color\n\n        Parameters\n        ----------\n        light, dark: str | Qt.GlobalColor | QColor\n            background color in light/dark theme mode\n        '
        self.lightBackgroundColor = QColor(light)
        self.darkBackgroundColor = QColor(dark)
        self.update()

    def eventFilter(self, obj, e: QEvent):
        if False:
            print('Hello World!')
        if obj is self.parent():
            if e.type() in [QEvent.Resize, QEvent.WindowStateChange]:
                self._adjustText()
        return super().eventFilter(obj, e)

    def closeEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.closedSignal.emit()
        self.deleteLater()

    def showEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self._adjustText()
        super().showEvent(e)
        if self.duration >= 0:
            QTimer.singleShot(self.duration, self.__fadeOut)
        if self.position != InfoBarPosition.NONE:
            manager = InfoBarManager.make(self.position)
            manager.add(self)
        if self.parent():
            self.parent().installEventFilter(self)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().paintEvent(e)
        if self.lightBackgroundColor is None:
            return
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        if isDarkTheme():
            painter.setBrush(self.darkBackgroundColor)
        else:
            painter.setBrush(self.lightBackgroundColor)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.drawRoundedRect(rect, 6, 6)

    @classmethod
    def new(cls, icon, title, content, orient=Qt.Horizontal, isClosable=True, duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        if False:
            print('Hello World!')
        w = InfoBar(icon, title, content, orient, isClosable, duration, position, parent)
        w.show()
        return w

    @classmethod
    def info(cls, title, content, orient=Qt.Horizontal, isClosable=True, duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        if False:
            while True:
                i = 10
        return cls.new(InfoBarIcon.INFORMATION, title, content, orient, isClosable, duration, position, parent)

    @classmethod
    def success(cls, title, content, orient=Qt.Horizontal, isClosable=True, duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        if False:
            for i in range(10):
                print('nop')
        return cls.new(InfoBarIcon.SUCCESS, title, content, orient, isClosable, duration, position, parent)

    @classmethod
    def warning(cls, title, content, orient=Qt.Horizontal, isClosable=True, duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        if False:
            i = 10
            return i + 15
        return cls.new(InfoBarIcon.WARNING, title, content, orient, isClosable, duration, position, parent)

    @classmethod
    def error(cls, title, content, orient=Qt.Horizontal, isClosable=True, duration=1000, position=InfoBarPosition.TOP_RIGHT, parent=None):
        if False:
            return 10
        return cls.new(InfoBarIcon.ERROR, title, content, orient, isClosable, duration, position, parent)

class InfoBarManager(QObject):
    """ Info bar manager """
    _instance = None
    managers = {}

    def __new__(cls, *args, **kwargs):
        if False:
            while True:
                i = 10
        if cls._instance is None:
            cls._instance = super(InfoBarManager, cls).__new__(cls, *args, **kwargs)
            cls._instance.__initialized = False
        return cls._instance

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        if self.__initialized:
            return
        self.spacing = 16
        self.margin = 24
        self.infoBars = weakref.WeakKeyDictionary()
        self.aniGroups = weakref.WeakKeyDictionary()
        self.slideAnis = []
        self.dropAnis = []
        self.__initialized = True

    def add(self, infoBar: InfoBar):
        if False:
            i = 10
            return i + 15
        ' add info bar '
        p = infoBar.parent()
        if not p:
            return
        if p not in self.infoBars:
            p.installEventFilter(self)
            self.infoBars[p] = []
            self.aniGroups[p] = QParallelAnimationGroup(self)
        if infoBar in self.infoBars[p]:
            return
        if self.infoBars[p]:
            dropAni = QPropertyAnimation(infoBar, b'pos')
            dropAni.setDuration(200)
            self.aniGroups[p].addAnimation(dropAni)
            self.dropAnis.append(dropAni)
            infoBar.setProperty('dropAni', dropAni)
        self.infoBars[p].append(infoBar)
        slideAni = self._createSlideAni(infoBar)
        self.slideAnis.append(slideAni)
        infoBar.setProperty('slideAni', slideAni)
        infoBar.closedSignal.connect(lambda : self.remove(infoBar))
        slideAni.start()

    def remove(self, infoBar: InfoBar):
        if False:
            print('Hello World!')
        ' remove info bar '
        p = infoBar.parent()
        if p not in self.infoBars:
            return
        if infoBar not in self.infoBars[p]:
            return
        self.infoBars[p].remove(infoBar)
        dropAni = infoBar.property('dropAni')
        if dropAni:
            self.aniGroups[p].removeAnimation(dropAni)
            self.dropAnis.remove(dropAni)
        slideAni = infoBar.property('slideAni')
        if slideAni:
            self.slideAnis.remove(slideAni)
        self._updateDropAni(p)
        self.aniGroups[p].start()

    def _createSlideAni(self, infoBar: InfoBar):
        if False:
            i = 10
            return i + 15
        slideAni = QPropertyAnimation(infoBar, b'pos')
        slideAni.setEasingCurve(QEasingCurve.OutQuad)
        slideAni.setDuration(200)
        slideAni.setStartValue(self._slideStartPos(infoBar))
        slideAni.setEndValue(self._pos(infoBar))
        return slideAni

    def _updateDropAni(self, parent):
        if False:
            return 10
        for bar in self.infoBars[parent]:
            ani = bar.property('dropAni')
            if not ani:
                continue
            ani.setStartValue(bar.pos())
            ani.setEndValue(self._pos(bar))

    def _pos(self, infoBar: InfoBar, parentSize=None) -> QPoint:
        if False:
            print('Hello World!')
        ' return the position of info bar '
        raise NotImplementedError

    def _slideStartPos(self, infoBar: InfoBar) -> QPoint:
        if False:
            while True:
                i = 10
        ' return the start position of slide animation  '
        raise NotImplementedError

    def eventFilter(self, obj, e: QEvent):
        if False:
            return 10
        if obj not in self.infoBars:
            return False
        if e.type() in [QEvent.Resize, QEvent.WindowStateChange]:
            size = e.size() if e.type() == QEvent.Resize else None
            for bar in self.infoBars[obj]:
                bar.move(self._pos(bar, size))
        return super().eventFilter(obj, e)

    @classmethod
    def register(cls, name):
        if False:
            print('Hello World!')
        ' register menu animation manager\n\n        Parameters\n        ----------\n        name: Any\n            the name of manager, it should be unique\n        '

        def wrapper(Manager):
            if False:
                return 10
            if name not in cls.managers:
                cls.managers[name] = Manager
            return Manager
        return wrapper

    @classmethod
    def make(cls, position: InfoBarPosition):
        if False:
            return 10
        ' mask info bar manager according to the display position '
        if position not in cls.managers:
            raise ValueError(f'`{position}` is an invalid animation type.')
        return cls.managers[position]()

@InfoBarManager.register(InfoBarPosition.TOP)
class TopInfoBarManager(InfoBarManager):
    """ Top position info bar manager """

    def _pos(self, infoBar: InfoBar, parentSize=None):
        if False:
            return 10
        p = infoBar.parent()
        parentSize = parentSize or p.size()
        x = (infoBar.parent().width() - infoBar.width()) // 2
        y = self.margin
        index = self.infoBars[p].index(infoBar)
        for bar in self.infoBars[p][0:index]:
            y += bar.height() + self.spacing
        return QPoint(x, y)

    def _slideStartPos(self, infoBar: InfoBar):
        if False:
            while True:
                i = 10
        pos = self._pos(infoBar)
        return QPoint(pos.x(), pos.y() - 16)

@InfoBarManager.register(InfoBarPosition.TOP_RIGHT)
class TopRightInfoBarManager(InfoBarManager):
    """ Top right position info bar manager """

    def _pos(self, infoBar: InfoBar, parentSize=None):
        if False:
            for i in range(10):
                print('nop')
        p = infoBar.parent()
        parentSize = parentSize or p.size()
        x = parentSize.width() - infoBar.width() - self.margin
        y = self.margin
        index = self.infoBars[p].index(infoBar)
        for bar in self.infoBars[p][0:index]:
            y += bar.height() + self.spacing
        return QPoint(x, y)

    def _slideStartPos(self, infoBar: InfoBar):
        if False:
            for i in range(10):
                print('nop')
        return QPoint(infoBar.parent().width(), self._pos(infoBar).y())

@InfoBarManager.register(InfoBarPosition.BOTTOM_RIGHT)
class BottomRightInfoBarManager(InfoBarManager):
    """ Bottom right position info bar manager """

    def _pos(self, infoBar: InfoBar, parentSize=None) -> QPoint:
        if False:
            return 10
        p = infoBar.parent()
        parentSize = parentSize or p.size()
        x = parentSize.width() - infoBar.width() - self.margin
        y = parentSize.height() - infoBar.height() - self.margin
        index = self.infoBars[p].index(infoBar)
        for bar in self.infoBars[p][0:index]:
            y -= bar.height() + self.spacing
        return QPoint(x, y)

    def _slideStartPos(self, infoBar: InfoBar):
        if False:
            i = 10
            return i + 15
        return QPoint(infoBar.parent().width(), self._pos(infoBar).y())

@InfoBarManager.register(InfoBarPosition.TOP_LEFT)
class TopLeftInfoBarManager(InfoBarManager):
    """ Top left position info bar manager """

    def _pos(self, infoBar: InfoBar, parentSize=None) -> QPoint:
        if False:
            for i in range(10):
                print('nop')
        p = infoBar.parent()
        parentSize = parentSize or p.size()
        y = self.margin
        index = self.infoBars[p].index(infoBar)
        for bar in self.infoBars[p][0:index]:
            y += bar.height() + self.spacing
        return QPoint(self.margin, y)

    def _slideStartPos(self, infoBar: InfoBar):
        if False:
            for i in range(10):
                print('nop')
        return QPoint(-infoBar.width(), self._pos(infoBar).y())

@InfoBarManager.register(InfoBarPosition.BOTTOM_LEFT)
class BottomLeftInfoBarManager(InfoBarManager):
    """ Bottom left position info bar manager """

    def _pos(self, infoBar: InfoBar, parentSize: QSize=None) -> QPoint:
        if False:
            print('Hello World!')
        p = infoBar.parent()
        parentSize = parentSize or p.size()
        y = parentSize.height() - infoBar.height() - self.margin
        index = self.infoBars[p].index(infoBar)
        for bar in self.infoBars[p][0:index]:
            y -= bar.height() + self.spacing
        return QPoint(self.margin, y)

    def _slideStartPos(self, infoBar: InfoBar):
        if False:
            while True:
                i = 10
        return QPoint(-infoBar.width(), self._pos(infoBar).y())

@InfoBarManager.register(InfoBarPosition.BOTTOM)
class BottomInfoBarManager(InfoBarManager):
    """ Bottom position info bar manager """

    def _pos(self, infoBar: InfoBar, parentSize: QSize=None) -> QPoint:
        if False:
            for i in range(10):
                print('nop')
        p = infoBar.parent()
        parentSize = parentSize or p.size()
        x = (parentSize.width() - infoBar.width()) // 2
        y = parentSize.height() - infoBar.height() - self.margin
        index = self.infoBars[p].index(infoBar)
        for bar in self.infoBars[p][0:index]:
            y -= bar.height() + self.spacing
        return QPoint(x, y)

    def _slideStartPos(self, infoBar: InfoBar):
        if False:
            for i in range(10):
                print('nop')
        pos = self._pos(infoBar)
        return QPoint(pos.x(), pos.y() + 16)