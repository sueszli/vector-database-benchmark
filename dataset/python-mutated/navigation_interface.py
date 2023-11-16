from typing import Union
from PyQt5.QtCore import Qt, QEvent, pyqtSignal
from PyQt5.QtGui import QResizeEvent, QIcon
from PyQt5.QtWidgets import QWidget
from .navigation_panel import NavigationPanel, NavigationItemPosition, NavigationWidget, NavigationDisplayMode
from .navigation_widget import NavigationTreeWidget
from ...common.style_sheet import FluentStyleSheet
from ...common.icon import FluentIconBase

class NavigationInterface(QWidget):
    """ Navigation interface """
    displayModeChanged = pyqtSignal(NavigationDisplayMode)

    def __init__(self, parent=None, showMenuButton=True, showReturnButton=False, collapsible=True):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        parent: widget\n            parent widget\n\n        showMenuButton: bool\n            whether to show menu button\n\n        showReturnButton: bool\n            whether to show return button\n\n        collapsible: bool\n            Is the navigation interface collapsible\n        '
        super().__init__(parent=parent)
        self.panel = NavigationPanel(self)
        self.panel.setMenuButtonVisible(showMenuButton and collapsible)
        self.panel.setReturnButtonVisible(showReturnButton)
        self.panel.setCollapsible(collapsible)
        self.panel.installEventFilter(self)
        self.panel.displayModeChanged.connect(self.displayModeChanged)
        self.resize(48, self.height())
        self.setMinimumWidth(48)
        self.setAttribute(Qt.WA_TranslucentBackground)

    def addItem(self, routeKey: str, icon: Union[str, QIcon, FluentIconBase], text: str, onClick=None, selectable=True, position=NavigationItemPosition.TOP, tooltip: str=None, parentRouteKey: str=None) -> NavigationTreeWidget:
        if False:
            i = 10
            return i + 15
        ' add navigation item\n\n        Parameters\n        ----------\n        routKey: str\n            the unique name of item\n\n        icon: str | QIcon | FluentIconBase\n            the icon of navigation item\n\n        text: str\n            the text of navigation item\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        selectable: bool\n            whether the item is selectable\n\n        position: NavigationItemPosition\n            where the button is added\n\n        tooltip: str\n            the tooltip of item\n\n        parentRouteKey: str\n            the route key of parent item, the parent item should be `NavigationTreeWidgetBase`\n        '
        return self.insertItem(-1, routeKey, icon, text, onClick, selectable, position, tooltip, parentRouteKey)

    def addWidget(self, routeKey: str, widget: NavigationWidget, onClick=None, position=NavigationItemPosition.TOP, tooltip: str=None, parentRouteKey: str=None):
        if False:
            return 10
        ' add custom widget\n\n        Parameters\n        ----------\n        routKey: str\n            the unique name of item\n\n        widget: NavigationWidget\n            the custom widget to be added\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        position: NavigationItemPosition\n            where the widget is added\n\n        tooltip: str\n            the tooltip of widget\n\n        parentRouteKey: str\n            the route key of parent item, the parent item should be `NavigationTreeWidgetBase`\n        '
        self.insertWidget(-1, routeKey, widget, onClick, position, tooltip, parentRouteKey)

    def insertItem(self, index: int, routeKey: str, icon: Union[str, QIcon, FluentIconBase], text: str, onClick=None, selectable=True, position=NavigationItemPosition.TOP, tooltip: str=None, parentRouteKey: str=None) -> NavigationTreeWidget:
        if False:
            for i in range(10):
                print('nop')
        ' insert navigation item\n\n        Parameters\n        ----------\n        index: int\n            insert position\n\n        routKey: str\n            the unique name of item\n\n        icon: str | QIcon | FluentIconBase\n            the icon of navigation item\n\n        text: str\n            the text of navigation item\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        selectable: bool\n            whether the item is selectable\n\n        position: NavigationItemPosition\n            where the item is added\n\n        tooltip: str\n            the tooltip of item\n\n        parentRouteKey: str\n            the route key of parent item, the parent item should be `NavigationTreeWidgetBase`\n        '
        w = self.panel.insertItem(index, routeKey, icon, text, onClick, selectable, position, tooltip, parentRouteKey)
        self.setMinimumHeight(self.panel.layoutMinHeight())
        return w

    def insertWidget(self, index: int, routeKey: str, widget: NavigationWidget, onClick=None, position=NavigationItemPosition.TOP, tooltip: str=None, parentRouteKey: str=None):
        if False:
            for i in range(10):
                print('nop')
        ' insert custom widget\n\n        Parameters\n        ----------\n        index: int\n            insert position\n\n        routKey: str\n            the unique name of item\n\n        widget: NavigationWidget\n            the custom widget to be added\n\n        onClick: callable\n            the slot connected to item clicked signal\n\n        position: NavigationItemPosition\n            where the widget is added\n\n        tooltip: str\n            the tooltip of widget\n\n        parentRouteKey: str\n            the route key of parent item, the parent item should be `NavigationTreeWidgetBase`\n        '
        self.panel.insertWidget(index, routeKey, widget, onClick, position, tooltip, parentRouteKey)
        self.setMinimumHeight(self.panel.layoutMinHeight())

    def addSeparator(self, position=NavigationItemPosition.TOP):
        if False:
            i = 10
            return i + 15
        ' add separator\n\n        Parameters\n        ----------\n        position: NavigationPostion\n            where to add the separator\n        '
        self.insertSeparator(-1, position)

    def insertSeparator(self, index: int, position=NavigationItemPosition.TOP):
        if False:
            while True:
                i = 10
        ' add separator\n\n        Parameters\n        ----------\n        index: int\n            insert position\n\n        position: NavigationPostion\n            where to add the separator\n        '
        self.panel.insertSeparator(index, position)
        self.setMinimumHeight(self.panel.layoutMinHeight())

    def removeWidget(self, routeKey: str):
        if False:
            print('Hello World!')
        ' remove widget\n\n        Parameters\n        ----------\n        routKey: str\n            the unique name of item\n        '
        self.panel.removeWidget(routeKey)

    def setCurrentItem(self, name: str):
        if False:
            i = 10
            return i + 15
        ' set current selected item\n\n        Parameters\n        ----------\n        name: str\n            the unique name of item\n        '
        self.panel.setCurrentItem(name)

    def setExpandWidth(self, width: int):
        if False:
            return 10
        ' set the maximum width '
        self.panel.setExpandWidth(width)

    def setMenuButtonVisible(self, isVisible: bool):
        if False:
            return 10
        ' set whether the menu button is visible '
        self.panel.setMenuButtonVisible(isVisible)

    def setReturnButtonVisible(self, isVisible: bool):
        if False:
            return 10
        ' set whether the return button is visible '
        self.panel.setReturnButtonVisible(isVisible)

    def setCollapsible(self, collapsible: bool):
        if False:
            i = 10
            return i + 15
        self.panel.setCollapsible(collapsible)

    def isAcrylicEnabled(self):
        if False:
            return 10
        return self.panel.isAcrylicEnabled()

    def setAcrylicEnabled(self, isEnabled: bool):
        if False:
            i = 10
            return i + 15
        ' set whether the acrylic background effect is enabled '
        self.panel.setAcrylicEnabled(isEnabled)

    def widget(self, routeKey: str):
        if False:
            return 10
        return self.panel.widget(routeKey)

    def eventFilter(self, obj, e: QEvent):
        if False:
            print('Hello World!')
        if obj is not self.panel or e.type() != QEvent.Resize:
            return super().eventFilter(obj, e)
        if self.panel.displayMode != NavigationDisplayMode.MENU:
            event = QResizeEvent(e)
            if event.oldSize().width() != event.size().width():
                self.setFixedWidth(event.size().width())
        return super().eventFilter(obj, e)

    def resizeEvent(self, e: QResizeEvent):
        if False:
            i = 10
            return i + 15
        if e.oldSize().height() != self.height():
            self.panel.setFixedHeight(self.height())