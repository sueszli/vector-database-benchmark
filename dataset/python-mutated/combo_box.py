import sys
from typing import Union, List, Iterable
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPoint, QObject, QEvent
from PyQt5.QtGui import QPainter, QCursor, QIcon
from PyQt5.QtWidgets import QAction, QPushButton, QApplication
from .menu import RoundMenu, MenuAnimationType, IndicatorMenuItemDelegate
from .line_edit import LineEdit, LineEditButton
from ...common.animation import TranslateYAnimation
from ...common.icon import FluentIconBase, isDarkTheme
from ...common.icon import FluentIcon as FIF
from ...common.font import setFont
from ...common.style_sheet import FluentStyleSheet

class ComboItem:
    """ Combo box item """

    def __init__(self, text: str, icon: Union[str, QIcon, FluentIconBase]=None, userData=None):
        if False:
            i = 10
            return i + 15
        ' add item\n\n        Parameters\n        ----------\n        text: str\n            the text of item\n\n        icon: str | QIcon | FluentIconBase\n            the icon of item\n\n        userData: Any\n            user data\n        '
        self.text = text
        self.userData = userData
        self.icon = icon

    @property
    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self._icon, QIcon):
            return self._icon
        return self._icon.icon()

    @icon.setter
    def icon(self, ico: Union[str, QIcon, FluentIconBase]):
        if False:
            print('Hello World!')
        if ico:
            self._icon = QIcon(ico) if isinstance(ico, str) else ico
        else:
            self._icon = QIcon()

class ComboBoxBase(QObject):
    """ Combo box base """
    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)

    def __init__(self, parent=None, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self.isHover = False
        self.isPressed = False
        self.items = []
        self._currentIndex = -1
        self._maxVisibleItems = -1
        self.dropMenu = None
        self._placeholderText = ''
        FluentStyleSheet.COMBO_BOX.apply(self)
        self.installEventFilter(self)

    def eventFilter(self, obj, e: QEvent):
        if False:
            while True:
                i = 10
        if obj is self:
            if e.type() == QEvent.MouseButtonPress:
                self.isPressed = True
            elif e.type() == QEvent.MouseButtonRelease:
                self.isPressed = False
            elif e.type() == QEvent.Enter:
                self.isHover = True
            elif e.type() == QEvent.Leave:
                self.isHover = False
        return super().eventFilter(obj, e)

    def addItem(self, text: str, icon: Union[str, QIcon, FluentIconBase]=None, userData=None):
        if False:
            for i in range(10):
                print('nop')
        ' add item\n\n        Parameters\n        ----------\n        text: str\n            the text of item\n\n        icon: str | QIcon | FluentIconBase\n        '
        item = ComboItem(text, icon, userData)
        self.items.append(item)
        if len(self.items) == 1:
            self.setCurrentIndex(0)

    def addItems(self, texts: Iterable[str]):
        if False:
            for i in range(10):
                print('nop')
        ' add items\n\n        Parameters\n        ----------\n        text: Iterable[str]\n            the text of item\n        '
        for text in texts:
            self.addItem(text)

    def removeItem(self, index: int):
        if False:
            return 10
        ' Removes the item at the given index from the combobox.\n        This will update the current index if the index is removed.\n        '
        if not 0 <= index < len(self.items):
            return
        self.items.pop(index)
        if index < self.currentIndex():
            self._onItemClicked(self._currentIndex - 1)
        elif index == self.currentIndex():
            if index > 0:
                self._onItemClicked(self._currentIndex - 1)
            else:
                self.setCurrentIndex(0)
                self.currentTextChanged.emit(self.currentText())
                self.currentIndexChanged.emit(0)
        if self.count() == 0:
            self.clear()

    def currentIndex(self):
        if False:
            return 10
        return self._currentIndex

    def setCurrentIndex(self, index: int):
        if False:
            i = 10
            return i + 15
        ' set current index\n\n        Parameters\n        ----------\n        index: int\n            current index\n        '
        if not 0 <= index < len(self.items):
            return
        self._currentIndex = index
        self.setText(self.items[index].text)

    def setText(self, text: str):
        if False:
            i = 10
            return i + 15
        super().setText(text)
        self.adjustSize()

    def currentText(self):
        if False:
            while True:
                i = 10
        if not 0 <= self.currentIndex() < len(self.items):
            return ''
        return self.items[self.currentIndex()].text

    def currentData(self):
        if False:
            i = 10
            return i + 15
        if not 0 <= self.currentIndex() < len(self.items):
            return None
        return self.items[self.currentIndex()].userData

    def setCurrentText(self, text):
        if False:
            for i in range(10):
                print('nop')
        ' set the current text displayed in combo box,\n        text should be in the item list\n\n        Parameters\n        ----------\n        text: str\n            text displayed in combo box\n        '
        if text == self.currentText():
            return
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)

    def setItemText(self, index: int, text: str):
        if False:
            for i in range(10):
                print('nop')
        ' set the text of item\n\n        Parameters\n        ----------\n        index: int\n            the index of item\n\n        text: str\n            new text of item\n        '
        if not 0 <= index < len(self.items):
            return
        self.items[index].text = text
        if self.currentIndex() == index:
            self.setText(text)

    def itemData(self, index: int):
        if False:
            print('Hello World!')
        ' Returns the data in the given index '
        if not 0 <= index < len(self.items):
            return None
        return self.items[index].userData

    def itemText(self, index: int):
        if False:
            print('Hello World!')
        ' Returns the text in the given index '
        if not 0 <= index < len(self.items):
            return ''
        return self.items[index].text

    def itemIcon(self, index: int):
        if False:
            for i in range(10):
                print('nop')
        ' Returns the icon in the given index '
        if not 0 <= index < len(self.items):
            return QIcon()
        return self.items[index].icon

    def setItemData(self, index: int, value):
        if False:
            for i in range(10):
                print('nop')
        ' Sets the data role for the item on the given index '
        if 0 <= index < len(self.items):
            self.items[index].userData = value

    def setItemIcon(self, index: int, icon: Union[str, QIcon, FluentIconBase]):
        if False:
            while True:
                i = 10
        ' Sets the data role for the item on the given index '
        if 0 <= index < len(self.items):
            self.items[index].icon = icon

    def findData(self, data):
        if False:
            return 10
        ' Returns the index of the item containing the given data, otherwise returns -1 '
        for (i, item) in enumerate(self.items):
            if item.userData == data:
                return i
        return -1

    def findText(self, text: str):
        if False:
            while True:
                i = 10
        ' Returns the index of the item containing the given text; otherwise returns -1. '
        for (i, item) in enumerate(self.items):
            if item.text == text:
                return i
        return -1

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        ' Clears the combobox, removing all items. '
        if self.currentIndex() >= 0:
            self.setText('')
        self.items.clear()
        self._currentIndex = -1

    def count(self):
        if False:
            i = 10
            return i + 15
        ' Returns the number of items in the combobox '
        return len(self.items)

    def insertItem(self, index: int, text: str, icon: Union[str, QIcon, FluentIconBase]=None, userData=None):
        if False:
            i = 10
            return i + 15
        ' Inserts item into the combobox at the given index. '
        item = ComboItem(text, icon, userData)
        self.items.insert(index, item)
        if index <= self.currentIndex():
            self._onItemClicked(self.currentIndex() + 1)

    def insertItems(self, index: int, texts: Iterable[str]):
        if False:
            for i in range(10):
                print('nop')
        ' Inserts items into the combobox, starting at the index specified. '
        pos = index
        for text in texts:
            item = ComboItem(text)
            self.items.insert(pos, item)
            pos += 1
        if index <= self.currentIndex():
            self._onItemClicked(self.currentIndex() + pos - index)

    def setMaxVisibleItems(self, num: int):
        if False:
            return 10
        self._maxVisibleItems = num

    def maxVisibleItems(self):
        if False:
            print('Hello World!')
        return self._maxVisibleItems

    def _closeComboMenu(self):
        if False:
            for i in range(10):
                print('nop')
        if not self.dropMenu:
            return
        self.dropMenu.close()
        self.dropMenu = None

    def _onDropMenuClosed(self):
        if False:
            for i in range(10):
                print('nop')
        if sys.platform != 'win32':
            self.dropMenu = None
        else:
            pos = self.mapFromGlobal(QCursor.pos())
            if not self.rect().contains(pos):
                self.dropMenu = None

    def _createComboMenu(self):
        if False:
            i = 10
            return i + 15
        return ComboBoxMenu(self)

    def _showComboMenu(self):
        if False:
            while True:
                i = 10
        if not self.items:
            return
        menu = self._createComboMenu()
        for (i, item) in enumerate(self.items):
            menu.addAction(QAction(item.icon, item.text, triggered=lambda c, x=i: self._onItemClicked(x)))
        if menu.view.width() < self.width():
            menu.view.setMinimumWidth(self.width())
            menu.adjustSize()
        menu.setMaxVisibleItems(self.maxVisibleItems())
        menu.closedSignal.connect(self._onDropMenuClosed)
        self.dropMenu = menu
        if self.currentIndex() >= 0 and self.items:
            menu.setDefaultAction(menu.actions()[self.currentIndex()])
        x = -menu.width() // 2 + menu.layout().contentsMargins().left() + self.width() // 2
        pd = self.mapToGlobal(QPoint(x, self.height()))
        hd = menu.view.heightForAnimation(pd, MenuAnimationType.DROP_DOWN)
        pu = self.mapToGlobal(QPoint(x, 0))
        hu = menu.view.heightForAnimation(pd, MenuAnimationType.PULL_UP)
        if hd >= hu:
            menu.view.adjustSize(pd, MenuAnimationType.DROP_DOWN)
            menu.exec(pd, aniType=MenuAnimationType.DROP_DOWN)
        else:
            menu.view.adjustSize(pu, MenuAnimationType.PULL_UP)
            menu.exec(pu, aniType=MenuAnimationType.PULL_UP)

    def _toggleComboMenu(self):
        if False:
            print('Hello World!')
        if self.dropMenu:
            self._closeComboMenu()
        else:
            self._showComboMenu()

    def _onItemClicked(self, index):
        if False:
            for i in range(10):
                print('nop')
        if index == self.currentIndex():
            return
        self.setCurrentIndex(index)
        self.currentTextChanged.emit(self.currentText())
        self.currentIndexChanged.emit(index)

class ComboBox(QPushButton, ComboBoxBase):
    """ Combo box """
    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self.arrowAni = TranslateYAnimation(self)
        setFont(self)

    def setPlaceholderText(self, text: str):
        if False:
            return 10
        self._placeholderText = text
        if self.currentIndex() <= 0:
            self._updateTextState(True)
            self.setText(text)

    def setCurrentIndex(self, index: int):
        if False:
            while True:
                i = 10
        if index < 0:
            self._currentIndex = -1
            self.setPlaceholderText(self._placeholderText)
        elif 0 <= index < len(self.items):
            self._updateTextState(False)
            super().setCurrentIndex(index)

    def _updateTextState(self, isPlaceholder):
        if False:
            i = 10
            return i + 15
        if self.property('isPlaceholderText') == isPlaceholder:
            return
        self.setProperty('isPlaceholderText', isPlaceholder)
        self.setStyle(QApplication.style())

    def mouseReleaseEvent(self, e):
        if False:
            while True:
                i = 10
        super().mouseReleaseEvent(e)
        self._toggleComboMenu()

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        QPushButton.paintEvent(self, e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if self.isHover:
            painter.setOpacity(0.8)
        elif self.isPressed:
            painter.setOpacity(0.7)
        rect = QRectF(self.width() - 22, self.height() / 2 - 5 + self.arrowAni.y, 10, 10)
        if isDarkTheme():
            FIF.ARROW_DOWN.render(painter, rect)
        else:
            FIF.ARROW_DOWN.render(painter, rect, fill='#646464')

class EditableComboBox(LineEdit, ComboBoxBase):
    """ Editable combo box """
    currentIndexChanged = pyqtSignal(int)
    currentTextChanged = pyqtSignal(str)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self.dropButton = LineEditButton(FIF.ARROW_DOWN, self)
        self.setTextMargins(0, 0, 29, 0)
        self.dropButton.setFixedSize(30, 25)
        self.hBoxLayout.addWidget(self.dropButton, 0, Qt.AlignRight)
        self.dropButton.clicked.connect(self._toggleComboMenu)
        self.textEdited.connect(self._onTextEdited)
        self.returnPressed.connect(self._onReturnPressed)
        self.clearButton.disconnect()
        self.clearButton.clicked.connect(self._onClearButtonClicked)

    def setCompleterMenu(self, menu):
        if False:
            for i in range(10):
                print('nop')
        super().setCompleterMenu(menu)
        menu.activated.connect(self.__onActivated)

    def __onActivated(self, text):
        if False:
            print('Hello World!')
        index = self.findText(text)
        if index >= 0:
            self.setCurrentIndex(index)

    def currentText(self):
        if False:
            for i in range(10):
                print('nop')
        return self.text()

    def setCurrentIndex(self, index: int):
        if False:
            i = 10
            return i + 15
        if index < 0:
            self._currentIndex = -1
            self.setText('')
            self.setPlaceholderText(self._placeholderText)
        else:
            super().setCurrentIndex(index)

    def clear(self):
        if False:
            print('Hello World!')
        ComboBoxBase.clear(self)

    def setPlaceholderText(self, text: str):
        if False:
            return 10
        self._placeholderText = text
        super().setPlaceholderText(text)

    def _onReturnPressed(self):
        if False:
            return 10
        if not self.text():
            return
        index = self.findText(self.text())
        if index >= 0 and index != self.currentIndex():
            self._currentIndex = index
            self.currentIndexChanged.emit(index)
        elif index == -1:
            self.addItem(self.text())
            self.setCurrentIndex(self.count() - 1)

    def _onTextEdited(self, text: str):
        if False:
            while True:
                i = 10
        self._currentIndex = -1
        self.currentTextChanged.emit(text)
        for (i, item) in enumerate(self.items):
            if item.text == text:
                self._currentIndex = i
                self.currentIndexChanged.emit(i)
                return

    def _onDropMenuClosed(self):
        if False:
            return 10
        self.dropMenu = None

    def _onClearButtonClicked(self):
        if False:
            while True:
                i = 10
        LineEdit.clear(self)
        self._currentIndex = -1

class ComboBoxMenu(RoundMenu):
    """ Combo box menu """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(title='', parent=parent)
        self.view.setViewportMargins(0, 2, 0, 6)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.view.setItemDelegate(IndicatorMenuItemDelegate())
        self.view.setObjectName('comboListWidget')
        self.setItemHeight(33)

    def exec(self, pos, ani=True, aniType=MenuAnimationType.DROP_DOWN):
        if False:
            while True:
                i = 10
        self.view.adjustSize(pos, aniType)
        self.adjustSize()
        return super().exec(pos, ani, aniType)