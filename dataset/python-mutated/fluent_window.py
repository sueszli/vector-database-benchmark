from typing import Union
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPainter, QColor
from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QLabel, QApplication
from ..common.config import qconfig
from ..common.icon import FluentIconBase
from ..common.router import qrouter
from ..common.style_sheet import FluentStyleSheet, isDarkTheme, setTheme, Theme
from ..common.animation import BackgroundAnimationWidget
from ..components.widgets.frameless_window import FramelessWindow
from ..components.navigation import NavigationInterface, NavigationBar, NavigationItemPosition, NavigationBarPushButton, NavigationTreeWidget
from .stacked_widget import StackedWidget
from qframelesswindow import TitleBar

class FluentWindowBase(BackgroundAnimationWidget, FramelessWindow):
    """ Fluent window base class """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        self._isMicaEnabled = False
        super().__init__(parent=parent)
        self.hBoxLayout = QHBoxLayout(self)
        self.stackedWidget = StackedWidget(self)
        self.navigationInterface = None
        self.hBoxLayout.setSpacing(0)
        self.hBoxLayout.setContentsMargins(0, 0, 0, 0)
        FluentStyleSheet.FLUENT_WINDOW.apply(self.stackedWidget)
        self.setMicaEffectEnabled(True)
        qconfig.themeChangedFinished.connect(self._onThemeChangedFinished)

    def addSubInterface(self, interface: QWidget, icon: Union[FluentIconBase, QIcon, str], text: str, position=NavigationItemPosition.TOP):
        if False:
            for i in range(10):
                print('nop')
        ' add sub interface '
        raise NotImplementedError

    def switchTo(self, interface: QWidget):
        if False:
            while True:
                i = 10
        self.stackedWidget.setCurrentWidget(interface, popOut=False)

    def _onCurrentInterfaceChanged(self, index: int):
        if False:
            print('Hello World!')
        widget = self.stackedWidget.widget(index)
        self.navigationInterface.setCurrentItem(widget.objectName())
        qrouter.push(self.stackedWidget, widget.objectName())
        self._updateStackedBackground()

    def _updateStackedBackground(self):
        if False:
            print('Hello World!')
        isTransparent = self.stackedWidget.currentWidget().property('isStackedTransparent')
        if bool(self.stackedWidget.property('isTransparent')) == isTransparent:
            return
        self.stackedWidget.setProperty('isTransparent', isTransparent)
        self.stackedWidget.setStyle(QApplication.style())

    def _normalBackgroundColor(self):
        if False:
            while True:
                i = 10
        if not self.isMicaEffectEnabled():
            return QColor(32, 32, 32) if isDarkTheme() else QColor(243, 243, 243)
        return QColor(0, 0, 0, 0)

    def _onThemeChangedFinished(self):
        if False:
            while True:
                i = 10
        if self.isMicaEffectEnabled():
            self.windowEffect.setMicaEffect(self.winId(), isDarkTheme())

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.backgroundColor)
        painter.drawRect(self.rect())

    def setMicaEffectEnabled(self, isEnabled: bool):
        if False:
            for i in range(10):
                print('nop')
        ' set whether the mica effect is enabled, only available on Win11 '
        if sys.platform != 'win32' or sys.getwindowsversion().build < 22000:
            return
        self._isMicaEnabled = isEnabled
        if isEnabled:
            self.windowEffect.setMicaEffect(self.winId(), isDarkTheme())
        else:
            self.windowEffect.removeBackgroundEffect(self.winId())
        self.setBackgroundColor(self._normalBackgroundColor())

    def isMicaEffectEnabled(self):
        if False:
            for i in range(10):
                print('nop')
        return self._isMicaEnabled

class FluentTitleBar(TitleBar):
    """ Fluent title bar"""

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.setFixedHeight(48)
        self.hBoxLayout.removeWidget(self.minBtn)
        self.hBoxLayout.removeWidget(self.maxBtn)
        self.hBoxLayout.removeWidget(self.closeBtn)
        self.iconLabel = QLabel(self)
        self.iconLabel.setFixedSize(18, 18)
        self.hBoxLayout.insertWidget(0, self.iconLabel, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.window().windowIconChanged.connect(self.setIcon)
        self.titleLabel = QLabel(self)
        self.hBoxLayout.insertWidget(1, self.titleLabel, 0, Qt.AlignLeft | Qt.AlignVCenter)
        self.titleLabel.setObjectName('titleLabel')
        self.window().windowTitleChanged.connect(self.setTitle)
        self.vBoxLayout = QVBoxLayout()
        self.buttonLayout = QHBoxLayout()
        self.buttonLayout.setSpacing(0)
        self.buttonLayout.setContentsMargins(0, 0, 0, 0)
        self.buttonLayout.setAlignment(Qt.AlignTop)
        self.buttonLayout.addWidget(self.minBtn)
        self.buttonLayout.addWidget(self.maxBtn)
        self.buttonLayout.addWidget(self.closeBtn)
        self.vBoxLayout.addLayout(self.buttonLayout)
        self.vBoxLayout.addStretch(1)
        self.hBoxLayout.addLayout(self.vBoxLayout, 0)
        FluentStyleSheet.FLUENT_WINDOW.apply(self)

    def setTitle(self, title):
        if False:
            i = 10
            return i + 15
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def setIcon(self, icon):
        if False:
            print('Hello World!')
        self.iconLabel.setPixmap(QIcon(icon).pixmap(18, 18))

class FluentWindow(FluentWindowBase):
    """ Fluent window """

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.setTitleBar(FluentTitleBar(self))
        self.navigationInterface = NavigationInterface(self, showReturnButton=True)
        self.widgetLayout = QHBoxLayout()
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addLayout(self.widgetLayout)
        self.hBoxLayout.setStretchFactor(self.widgetLayout, 1)
        self.widgetLayout.addWidget(self.stackedWidget)
        self.widgetLayout.setContentsMargins(0, 48, 0, 0)
        self.navigationInterface.displayModeChanged.connect(self.titleBar.raise_)
        self.titleBar.raise_()

    def addSubInterface(self, interface: QWidget, icon: Union[FluentIconBase, QIcon, str], text: str, position=NavigationItemPosition.TOP, parent=None, isTransparent=False) -> NavigationTreeWidget:
        if False:
            i = 10
            return i + 15
        ' add sub interface, the object name of `interface` should be set already\n        before calling this method\n\n        Parameters\n        ----------\n        interface: QWidget\n            the subinterface to be added\n\n        icon: FluentIconBase | QIcon | str\n            the icon of navigation item\n\n        text: str\n            the text of navigation item\n\n        position: NavigationItemPosition\n            the position of navigation item\n\n        parent: QWidget\n            the parent of navigation item\n\n        isTransparent: bool\n            whether to use transparent background\n        '
        if not interface.objectName():
            raise ValueError("The object name of `interface` can't be empty string.")
        if parent and (not parent.objectName()):
            raise ValueError("The object name of `parent` can't be empty string.")
        interface.setProperty('isStackedTransparent', isTransparent)
        self.stackedWidget.addWidget(interface)
        routeKey = interface.objectName()
        item = self.navigationInterface.addItem(routeKey=routeKey, icon=icon, text=text, onClick=lambda : self.switchTo(interface), position=position, tooltip=text, parentRouteKey=parent.objectName() if parent else None)
        if self.stackedWidget.count() == 1:
            self.stackedWidget.currentChanged.connect(self._onCurrentInterfaceChanged)
            self.navigationInterface.setCurrentItem(routeKey)
            qrouter.setDefaultRouteKey(self.stackedWidget, routeKey)
        self._updateStackedBackground()
        return item

    def resizeEvent(self, e):
        if False:
            print('Hello World!')
        self.titleBar.move(46, 0)
        self.titleBar.resize(self.width() - 46, self.titleBar.height())

class MSFluentTitleBar(FluentTitleBar):

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.hBoxLayout.insertSpacing(0, 20)
        self.hBoxLayout.insertSpacing(2, 2)

class MSFluentWindow(FluentWindowBase):
    """ Fluent window in Microsoft Store style """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.setTitleBar(MSFluentTitleBar(self))
        self.navigationInterface = NavigationBar(self)
        self.hBoxLayout.setContentsMargins(0, 48, 0, 0)
        self.hBoxLayout.addWidget(self.navigationInterface)
        self.hBoxLayout.addWidget(self.stackedWidget, 1)
        self.titleBar.raise_()
        self.titleBar.setAttribute(Qt.WA_StyledBackground)

    def addSubInterface(self, interface: QWidget, icon: Union[FluentIconBase, QIcon, str], text: str, selectedIcon=None, position=NavigationItemPosition.TOP, isTransparent=False) -> NavigationBarPushButton:
        if False:
            return 10
        ' add sub interface, the object name of `interface` should be set already\n        before calling this method\n\n        Parameters\n        ----------\n        interface: QWidget\n            the subinterface to be added\n\n        icon: FluentIconBase | QIcon | str\n            the icon of navigation item\n\n        text: str\n            the text of navigation item\n\n        selectedIcon: str | QIcon | FluentIconBase\n            the icon of navigation item in selected state\n\n        position: NavigationItemPosition\n            the position of navigation item\n        '
        if not interface.objectName():
            raise ValueError("The object name of `interface` can't be empty string.")
        interface.setProperty('isStackedTransparent', isTransparent)
        self.stackedWidget.addWidget(interface)
        routeKey = interface.objectName()
        item = self.navigationInterface.addItem(routeKey=routeKey, icon=icon, text=text, onClick=lambda : self.switchTo(interface), selectedIcon=selectedIcon, position=position)
        if self.stackedWidget.count() == 1:
            self.stackedWidget.currentChanged.connect(self._onCurrentInterfaceChanged)
            self.navigationInterface.setCurrentItem(routeKey)
            qrouter.setDefaultRouteKey(self.stackedWidget, routeKey)
        self._updateStackedBackground()
        return item

class SplitTitleBar(TitleBar):

    def __init__(self, parent):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.iconLabel = QLabel(self)
        self.iconLabel.setFixedSize(18, 18)
        self.hBoxLayout.insertSpacing(0, 12)
        self.hBoxLayout.insertWidget(1, self.iconLabel, 0, Qt.AlignLeft | Qt.AlignBottom)
        self.window().windowIconChanged.connect(self.setIcon)
        self.titleLabel = QLabel(self)
        self.hBoxLayout.insertWidget(2, self.titleLabel, 0, Qt.AlignLeft | Qt.AlignBottom)
        self.titleLabel.setObjectName('titleLabel')
        self.window().windowTitleChanged.connect(self.setTitle)
        FluentStyleSheet.FLUENT_WINDOW.apply(self)

    def setTitle(self, title):
        if False:
            for i in range(10):
                print('nop')
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def setIcon(self, icon):
        if False:
            print('Hello World!')
        self.iconLabel.setPixmap(QIcon(icon).pixmap(18, 18))

class SplitFluentWindow(FluentWindow):
    """ Fluent window with split style """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.setTitleBar(SplitTitleBar(self))
        self.widgetLayout.setContentsMargins(0, 0, 0, 0)
        self.titleBar.raise_()
        self.navigationInterface.displayModeChanged.connect(self.titleBar.raise_)