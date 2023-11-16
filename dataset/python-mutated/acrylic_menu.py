from PyQt5.QtCore import Qt, QRect, QRectF, QSize
from PyQt5.QtGui import QPainter, QColor, QPainterPath
from PyQt5.QtWidgets import QLineEdit, QListWidgetItem, QListWidget
from ..widgets.menu import RoundMenu, MenuAnimationType, MenuAnimationManager, MenuActionListWidget, IndicatorMenuItemDelegate, LineEditMenu, MenuIndicatorType, CheckableMenu
from ..widgets.line_edit import CompleterMenu, LineEdit
from ..widgets.acrylic_label import AcrylicBrush
from ...common.style_sheet import isDarkTheme

class AcrylicMenuActionListWidget(MenuActionListWidget):

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.acrylicBrush = AcrylicBrush(self.viewport(), 35)
        self.setViewportMargins(0, 0, 0, 0)
        self.setProperty('transparent', True)
        super().addItem(self.createPlaceholderItem(self._topMargin()))
        super().addItem(self.createPlaceholderItem(self._bottomMargin()))

    def _updateAcrylicColor(self):
        if False:
            return 10
        if isDarkTheme():
            tintColor = QColor(32, 32, 32, 200)
            luminosityColor = QColor(0, 0, 0, 0)
        else:
            tintColor = QColor(255, 255, 255, 160)
            luminosityColor = QColor(255, 255, 255, 50)
        self.acrylicBrush.tintColor = tintColor
        self.acrylicBrush.luminosityColor = luminosityColor

    def _topMargin(self):
        if False:
            for i in range(10):
                print('nop')
        return 6

    def _bottomMargin(self):
        if False:
            while True:
                i = 10
        return 6

    def setItemHeight(self, height: int):
        if False:
            i = 10
            return i + 15
        ' set the height of item '
        if height == self._itemHeight:
            return
        for i in range(1, self.count() - 1):
            item = self.item(i)
            if not self.itemWidget(item):
                item.setSizeHint(QSize(item.sizeHint().width(), height))
        self._itemHeight = height
        self.adjustSize()

    def addItem(self, item):
        if False:
            print('Hello World!')
        return super().insertItem(self.count() - 1, item)

    def createPlaceholderItem(self, height=2):
        if False:
            while True:
                i = 10
        item = QListWidgetItem()
        item.setSizeHint(QSize(1, height))
        item.setFlags(Qt.ItemFlag.NoItemFlags)
        return item

    def clipPath(self):
        if False:
            print('Hello World!')
        path = QPainterPath()
        path.addRoundedRect(QRectF(self.rect().adjusted(0, 0, -2.5, -2.5)), 8, 8)
        return path

    def paintEvent(self, e) -> None:
        if False:
            i = 10
            return i + 15
        painter = QPainter(self.viewport())
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.acrylicBrush.clipPath = self.clipPath()
        self._updateAcrylicColor()
        self.acrylicBrush.paint()
        super().paintEvent(e)

class AcrylicMenuBase:

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)

    def setUpMenu(self, view):
        if False:
            return 10
        self.hBoxLayout.removeWidget(self.view)
        self.view.deleteLater()
        self.view = view
        self.hBoxLayout.addWidget(self.view)
        self.setShadowEffect()
        self.view.itemClicked.connect(self._onItemClicked)
        self.view.itemEntered.connect(self._onItemEntered)

    def exec(self, pos, ani=True, aniType=MenuAnimationType.DROP_DOWN):
        if False:
            i = 10
            return i + 15
        p = MenuAnimationManager.make(self, aniType)._endPosition(pos)
        self.view.acrylicBrush.grabImage(QRect(p, self.layout().sizeHint()))
        super().exec(pos, ani, aniType)

class AcrylicMenu(AcrylicMenuBase, RoundMenu):
    """ Acrylic menu """

    def __init__(self, title='', parent=None):
        if False:
            while True:
                i = 10
        super().__init__(title, parent)
        self.setUpMenu(AcrylicMenuActionListWidget(self))

class AcrylicCompleterMenuActionListWidget(AcrylicMenuActionListWidget):

    def clipPath(self):
        if False:
            for i in range(10):
                print('nop')
        path = QPainterPath()
        path.setFillRule(Qt.FillRule.WindingFill)
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -2.5, -2.5)), 8, 8)
        if self.property('dropDown'):
            path.addRect(1, 1, 11, 11)
            path.addRect(self.width() - 12, 1, 11, 11)
        else:
            path.addRect(1, self.height() - 11, 11, 11)
            path.addRect(self.width() - 12, self.height() - 11, 11, 11)
        return path

class AcrylicCompleterMenu(AcrylicMenuBase, CompleterMenu):
    """ Acrylic completer menu """

    def __init__(self, lineEdit: LineEdit):
        if False:
            print('Hello World!')
        super().__init__(lineEdit)
        self.setUpMenu(AcrylicCompleterMenuActionListWidget(self))
        self.view.setObjectName('completerListWidget')
        self.view.setItemDelegate(IndicatorMenuItemDelegate())
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setItemHeight(33)

    def setItems(self, items: str):
        if False:
            return 10
        ' set completion items '
        self.view.clear()
        self.items = items
        QListWidget.addItem(self.view, self.view.createPlaceholderItem(self.view._topMargin()))
        self.view.addItems(items)
        for i in range(1, self.view.count()):
            item = self.view.item(i)
            item.setSizeHint(QSize(1, self.itemHeight))
        QListWidget.addItem(self.view, self.view.createPlaceholderItem(self.view._bottomMargin()))

class AcrylicLineEditMenu(AcrylicMenuBase, LineEditMenu):
    """ Acrylic line edit menu """

    def __init__(self, parent: QLineEdit):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.setUpMenu(AcrylicMenuActionListWidget(self))

class AcrylicCheckableMenu(AcrylicMenuBase, CheckableMenu):
    """ Checkable menu """

    def __init__(self, title='', parent=None, indicatorType=MenuIndicatorType.CHECK):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(title, parent, indicatorType)
        self.setUpMenu(AcrylicMenuActionListWidget(self))
        self.view.setObjectName('checkableListWidget')

class AcrylicSystemTrayMenu(AcrylicMenu):
    """ System tray menu """

    def showEvent(self, e):
        if False:
            return 10
        super().showEvent(e)
        self.adjustPosition()
        self.view.acrylicBrush.grabImage(QRect(self.pos(), self.layout().sizeHint()))

class AcrylicCheckableSystemTrayMenu(AcrylicCheckableMenu):
    """ Checkable system tray menu """

    def showEvent(self, e):
        if False:
            i = 10
            return i + 15
        super().showEvent(e)
        self.adjustPosition()