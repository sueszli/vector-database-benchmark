from typing import Dict, Union
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QPropertyAnimation, QEasingCurve, pyqtProperty, QRectF
from PyQt5.QtGui import QFont, QPainter, QColor, QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from ...common.config import isDarkTheme
from ...common.font import setFont
from ...common.style_sheet import themeColor
from ...common.icon import drawIcon, FluentIconBase, toQIcon
from ...common.icon import FluentIcon as FIF
from ...common.router import qrouter
from ...common.style_sheet import FluentStyleSheet
from ..widgets.scroll_area import SingleDirectionScrollArea
from .navigation_widget import NavigationPushButton, NavigationWidget
from .navigation_panel import RouteKeyError, NavigationItemPosition

class IconSlideAnimation(QPropertyAnimation):
    """ Icon sliding animation """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._offset = 0
        self.maxOffset = 6
        self.setTargetObject(self)
        self.setPropertyName(b'offset')

    def getOffset(self):
        if False:
            while True:
                i = 10
        return self._offset

    def setOffset(self, value: float):
        if False:
            i = 10
            return i + 15
        self._offset = value
        self.parent().update()

    def slideDown(self):
        if False:
            while True:
                i = 10
        ' slide down '
        self.setEndValue(self.maxOffset)
        self.setDuration(100)
        self.start()

    def slideUp(self):
        if False:
            return 10
        ' slide up '
        self.setEndValue(0)
        self.setDuration(100)
        self.start()
    offset = pyqtProperty(float, getOffset, setOffset)

class NavigationBarPushButton(NavigationPushButton):
    """ Navigation bar push button """

    def __init__(self, icon: Union[str, QIcon, FIF], text: str, isSelectable: bool, selectedIcon=None, parent=None):
        if False:
            print('Hello World!')
        super().__init__(icon, text, isSelectable, parent)
        self.iconAni = IconSlideAnimation(self)
        self._selectedIcon = selectedIcon
        self._isSelectedTextVisible = True
        self.setFixedSize(64, 58)
        setFont(self, 11)

    def selectedIcon(self):
        if False:
            print('Hello World!')
        if self._selectedIcon:
            return toQIcon(self._selectedIcon)
        return QIcon()

    def setSelectedIcon(self, icon: Union[str, QIcon, FIF]):
        if False:
            return 10
        self._selectedIcon = icon
        self.update()

    def setSelectedTextVisible(self, isVisible):
        if False:
            for i in range(10):
                print('nop')
        self._isSelectedTextVisible = isVisible
        self.update()

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        if self.isSelected:
            painter.setBrush(QColor(255, 255, 255, 42) if isDarkTheme() else Qt.white)
            painter.drawRoundedRect(self.rect(), 5, 5)
            painter.setBrush(themeColor())
            if not self.isPressed:
                painter.drawRoundedRect(0, 16, 4, 24, 2, 2)
            else:
                painter.drawRoundedRect(0, 19, 4, 18, 2, 2)
        elif self.isPressed or self.isEnter:
            c = 255 if isDarkTheme() else 0
            alpha = 9 if self.isEnter else 6
            painter.setBrush(QColor(c, c, c, alpha))
            painter.drawRoundedRect(self.rect(), 5, 5)
        if (self.isPressed or not self.isEnter) and (not self.isSelected):
            painter.setOpacity(0.6)
        if not self.isEnabled():
            painter.setOpacity(0.4)
        if self._isSelectedTextVisible:
            rect = QRectF(22, 13, 20, 20)
        else:
            rect = QRectF(22, 13 + self.iconAni.offset, 20, 20)
        selectedIcon = self._selectedIcon or self._icon
        if isinstance(selectedIcon, FluentIconBase) and self.isSelected:
            selectedIcon.render(painter, rect, fill=themeColor().name())
        elif self.isSelected:
            drawIcon(selectedIcon, painter, rect)
        else:
            drawIcon(self._icon, painter, rect)
        if self.isSelected and (not self._isSelectedTextVisible):
            return
        if self.isSelected:
            painter.setPen(themeColor())
        else:
            painter.setPen(Qt.white if isDarkTheme() else Qt.black)
        painter.setFont(self.font())
        rect = QRect(0, 32, self.width(), 26)
        painter.drawText(rect, Qt.AlignCenter, self.text())

    def setSelected(self, isSelected: bool):
        if False:
            for i in range(10):
                print('nop')
        if isSelected == self.isSelected:
            return
        self.isSelected = isSelected
        if isSelected:
            self.iconAni.slideDown()
        else:
            self.iconAni.slideUp()

class NavigationBar(QWidget):

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self.scrollArea = SingleDirectionScrollArea(self)
        self.scrollWidget = QWidget()
        self.vBoxLayout = QVBoxLayout(self)
        self.topLayout = QVBoxLayout()
        self.bottomLayout = QVBoxLayout()
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.items = {}
        self.history = qrouter
        self.__initWidget()

    def __initWidget(self):
        if False:
            while True:
                i = 10
        self.resize(48, self.height())
        self.setAttribute(Qt.WA_StyledBackground)
        self.window().installEventFilter(self)
        self.scrollArea.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollArea.setWidget(self.scrollWidget)
        self.scrollArea.setWidgetResizable(True)
        self.scrollWidget.setObjectName('scrollWidget')
        FluentStyleSheet.NAVIGATION_INTERFACE.apply(self)
        FluentStyleSheet.NAVIGATION_INTERFACE.apply(self.scrollWidget)
        self.__initLayout()

    def __initLayout(self):
        if False:
            print('Hello World!')
        self.vBoxLayout.setContentsMargins(0, 5, 0, 5)
        self.topLayout.setContentsMargins(4, 0, 4, 0)
        self.bottomLayout.setContentsMargins(4, 0, 4, 0)
        self.scrollLayout.setContentsMargins(4, 0, 4, 0)
        self.vBoxLayout.setSpacing(4)
        self.topLayout.setSpacing(4)
        self.bottomLayout.setSpacing(4)
        self.scrollLayout.setSpacing(4)
        self.vBoxLayout.addLayout(self.topLayout, 0)
        self.vBoxLayout.addWidget(self.scrollArea)
        self.vBoxLayout.addLayout(self.bottomLayout, 0)
        self.vBoxLayout.setAlignment(Qt.AlignTop)
        self.topLayout.setAlignment(Qt.AlignTop)
        self.scrollLayout.setAlignment(Qt.AlignTop)
        self.bottomLayout.setAlignment(Qt.AlignBottom)

    def widget(self, routeKey: str):
        if False:
            return 10
        if routeKey not in self.items:
            raise RouteKeyError(f'`{routeKey}` is illegal.')
        return self.items[routeKey]

    def addItem(self, routeKey: str, icon: Union[str, QIcon, FluentIconBase], text: str, onClick=None, selectable=True, selectedIcon=None, position=NavigationItemPosition.TOP):
        if False:
            while True:
                i = 10
        ' add navigation item\n\n        Parameters\n        ----------\n        routeKey: str\n            the unique name of item\n\n        icon: str | QIcon | FluentIconBase\n            the icon of navigation item\n\n        text: str\n            the text of navigation item\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        selectable: bool\n            whether the item is selectable\n\n        selectedIcon: str | QIcon | FluentIconBase\n            the icon of navigation item in selected state\n\n        position: NavigationItemPosition\n            where the button is added\n        '
        return self.insertItem(-1, routeKey, icon, text, onClick, selectable, selectedIcon, position)

    def addWidget(self, routeKey: str, widget: NavigationWidget, onClick=None, position=NavigationItemPosition.TOP):
        if False:
            while True:
                i = 10
        ' add custom widget\n\n        Parameters\n        ----------\n        routeKey: str\n            the unique name of item\n\n        widget: NavigationWidget\n            the custom widget to be added\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        position: NavigationItemPosition\n            where the button is added\n        '
        self.insertWidget(-1, routeKey, widget, onClick, position)

    def insertItem(self, index: int, routeKey: str, icon: Union[str, QIcon, FluentIconBase], text: str, onClick=None, selectable=True, selectedIcon=None, position=NavigationItemPosition.TOP):
        if False:
            i = 10
            return i + 15
        ' insert navigation tree item\n\n        Parameters\n        ----------\n        index: int\n            the insert position of parent widget\n\n        routeKey: str\n            the unique name of item\n\n        icon: str | QIcon | FluentIconBase\n            the icon of navigation item\n\n        text: str\n            the text of navigation item\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        selectable: bool\n            whether the item is selectable\n\n        selectedIcon: str | QIcon | FluentIconBase\n            the icon of navigation item in selected state\n\n        position: NavigationItemPosition\n            where the button is added\n        '
        if routeKey in self.items:
            return
        w = NavigationBarPushButton(icon, text, selectable, selectedIcon, self)
        self.insertWidget(index, routeKey, w, onClick, position)
        return w

    def insertWidget(self, index: int, routeKey: str, widget: NavigationWidget, onClick=None, position=NavigationItemPosition.TOP):
        if False:
            for i in range(10):
                print('nop')
        ' insert custom widget\n\n        Parameters\n        ----------\n        index: int\n            insert position\n\n        routeKey: str\n            the unique name of item\n\n        widget: NavigationWidget\n            the custom widget to be added\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        position: NavigationItemPosition\n            where the button is added\n        '
        if routeKey in self.items:
            return
        self._registerWidget(routeKey, widget, onClick)
        self._insertWidgetToLayout(index, widget, position)

    def _registerWidget(self, routeKey: str, widget: NavigationWidget, onClick):
        if False:
            for i in range(10):
                print('nop')
        ' register widget '
        widget.clicked.connect(self._onWidgetClicked)
        if onClick is not None:
            widget.clicked.connect(onClick)
        widget.setProperty('routeKey', routeKey)
        self.items[routeKey] = widget

    def _insertWidgetToLayout(self, index: int, widget: NavigationWidget, position: NavigationItemPosition):
        if False:
            return 10
        ' insert widget to layout '
        if position == NavigationItemPosition.TOP:
            widget.setParent(self)
            self.topLayout.insertWidget(index, widget, 0, Qt.AlignTop | Qt.AlignHCenter)
        elif position == NavigationItemPosition.SCROLL:
            widget.setParent(self.scrollWidget)
            self.scrollLayout.insertWidget(index, widget, 0, Qt.AlignTop | Qt.AlignHCenter)
        else:
            widget.setParent(self)
            self.bottomLayout.insertWidget(index, widget, 0, Qt.AlignBottom | Qt.AlignHCenter)
        widget.show()

    def removeWidget(self, routeKey: str):
        if False:
            while True:
                i = 10
        ' remove widget\n\n        Parameters\n        ----------\n        routeKey: str\n            the unique name of item\n        '
        if routeKey not in self.items:
            return
        widget = self.items.pop(routeKey)
        widget.deleteLater()
        self.history.remove(routeKey)

    def setCurrentItem(self, routeKey: str):
        if False:
            i = 10
            return i + 15
        ' set current selected item\n\n        Parameters\n        ----------\n        routeKey: str\n            the unique name of item\n        '
        if routeKey not in self.items:
            return
        for (k, widget) in self.items.items():
            widget.setSelected(k == routeKey)

    def setFont(self, font: QFont):
        if False:
            return 10
        ' set the font of navigation item '
        super().setFont(font)
        for widget in self.buttons():
            widget.setFont(font)

    def setSelectedTextVisible(self, isVisible: bool):
        if False:
            i = 10
            return i + 15
        ' set whether the text is visible when button is selected '
        for widget in self.buttons():
            widget.setSelectedTextVisible(isVisible)

    def buttons(self):
        if False:
            i = 10
            return i + 15
        return [i for i in self.items.values() if isinstance(i, NavigationPushButton)]

    def _onWidgetClicked(self):
        if False:
            return 10
        widget = self.sender()
        if widget.isSelectable:
            self.setCurrentItem(widget.property('routeKey'))