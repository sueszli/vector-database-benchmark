from typing import List, Union
from PyQt5 import QtCore
from PyQt5.QtCore import QSize, Qt, QRectF, pyqtSignal, QPoint, QTimer, QEvent, QAbstractItemModel, pyqtProperty
from PyQt5.QtGui import QPainter, QPainterPath, QIcon, QCursor
from PyQt5.QtWidgets import QApplication, QAction, QHBoxLayout, QLineEdit, QToolButton, QTextEdit, QPlainTextEdit, QCompleter, QStyle, QWidget
from ...common.style_sheet import FluentStyleSheet, themeColor
from ...common.icon import isDarkTheme, FluentIconBase, drawIcon
from ...common.icon import FluentIcon as FIF
from ...common.font import setFont
from .menu import LineEditMenu, TextEditMenu, RoundMenu, MenuAnimationType, IndicatorMenuItemDelegate
from .scroll_bar import SmoothScrollDelegate

class LineEditButton(QToolButton):
    """ Line edit button """

    def __init__(self, icon: Union[str, QIcon, FluentIconBase], parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self._icon = icon
        self.isPressed = False
        self.setFixedSize(31, 23)
        self.setIconSize(QSize(10, 10))
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName('lineEditButton')
        FluentStyleSheet.LINE_EDIT.apply(self)

    def mousePressEvent(self, e):
        if False:
            while True:
                i = 10
        self.isPressed = True
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if False:
            print('Hello World!')
        self.isPressed = False
        super().mouseReleaseEvent(e)

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        (iw, ih) = (self.iconSize().width(), self.iconSize().height())
        (w, h) = (self.width(), self.height())
        rect = QRectF((w - iw) / 2, (h - ih) / 2, iw, ih)
        if self.isPressed:
            painter.setOpacity(0.7)
        if isDarkTheme():
            drawIcon(self._icon, painter, rect)
        else:
            drawIcon(self._icon, painter, rect, fill='#656565')

class LineEdit(QLineEdit):
    """ Line edit """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self._isClearButtonEnabled = False
        self._completer = None
        self._completerMenu = None
        self.setProperty('transparent', True)
        FluentStyleSheet.LINE_EDIT.apply(self)
        self.setFixedHeight(33)
        self.setAttribute(Qt.WA_MacShowFocusRect, False)
        setFont(self)
        self.hBoxLayout = QHBoxLayout(self)
        self.clearButton = LineEditButton(FIF.CLOSE, self)
        self.clearButton.setFixedSize(29, 25)
        self.clearButton.hide()
        self.hBoxLayout.setSpacing(3)
        self.hBoxLayout.setContentsMargins(4, 4, 4, 4)
        self.hBoxLayout.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.hBoxLayout.addWidget(self.clearButton, 0, Qt.AlignRight)
        self.clearButton.clicked.connect(self.clear)
        self.textChanged.connect(self.__onTextChanged)
        self.textEdited.connect(self.__onTextEdited)

    def setClearButtonEnabled(self, enable: bool):
        if False:
            print('Hello World!')
        self._isClearButtonEnabled = enable
        self.setTextMargins(0, 0, 28 * enable, 0)

    def isClearButtonEnabled(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._isClearButtonEnabled

    def setCompleter(self, completer: QCompleter):
        if False:
            for i in range(10):
                print('nop')
        self._completer = completer

    def completer(self):
        if False:
            i = 10
            return i + 15
        return self._completer

    def focusOutEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().focusOutEvent(e)
        self.clearButton.hide()

    def focusInEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().focusInEvent(e)
        if self.isClearButtonEnabled():
            self.clearButton.setVisible(bool(self.text()))

    def __onTextChanged(self, text):
        if False:
            return 10
        ' text changed slot '
        if self.isClearButtonEnabled():
            self.clearButton.setVisible(bool(text) and self.hasFocus())

    def __onTextEdited(self, text):
        if False:
            for i in range(10):
                print('nop')
        if not self.completer():
            return
        if self.text():
            QTimer.singleShot(50, self._showCompleterMenu)
        elif self._completerMenu:
            self._completerMenu.close()

    def setCompleterMenu(self, menu):
        if False:
            print('Hello World!')
        ' set completer menu\n\n        Parameters\n        ----------\n        menu: CompleterMenu\n            completer menu\n        '
        menu.activated.connect(self._completer.activated)
        self._completerMenu = menu

    def _showCompleterMenu(self):
        if False:
            return 10
        if not self.completer() or not self.text():
            return
        if not self._completerMenu:
            self.setCompleterMenu(CompleterMenu(self))
        self.completer().setCompletionPrefix(self.text())
        changed = self._completerMenu.setCompletion(self.completer().completionModel())
        self._completerMenu.setMaxVisibleItems(self.completer().maxVisibleItems())
        if changed:
            self._completerMenu.popup()

    def contextMenuEvent(self, e):
        if False:
            print('Hello World!')
        menu = LineEditMenu(self)
        menu.exec_(e.globalPos())

    def paintEvent(self, e):
        if False:
            while True:
                i = 10
        super().paintEvent(e)
        if not self.hasFocus():
            return
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        m = self.contentsMargins()
        path = QPainterPath()
        (w, h) = (self.width() - m.left() - m.right(), self.height())
        path.addRoundedRect(QRectF(m.left(), h - 10, w, 10), 5, 5)
        rectPath = QPainterPath()
        rectPath.addRect(m.left(), h - 10, w, 8)
        path = path.subtracted(rectPath)
        painter.fillPath(path, themeColor())

class CompleterMenu(RoundMenu):
    """ Completer menu """
    activated = pyqtSignal(str)

    def __init__(self, lineEdit: LineEdit):
        if False:
            print('Hello World!')
        super().__init__()
        self.items = []
        self.lineEdit = lineEdit
        self.view.setViewportMargins(0, 2, 0, 6)
        self.view.setObjectName('completerListWidget')
        self.view.setItemDelegate(IndicatorMenuItemDelegate())
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.installEventFilter(self)
        self.setItemHeight(33)

    def setCompletion(self, model: QAbstractItemModel):
        if False:
            for i in range(10):
                print('nop')
        ' set the completion model '
        items = []
        for i in range(model.rowCount()):
            for j in range(model.columnCount()):
                items.append(model.data(model.index(i, j)))
        if self.items == items and self.isVisible():
            return False
        self.setItems(items)
        return True

    def setItems(self, items: List[str]):
        if False:
            return 10
        ' set completion items '
        self.view.clear()
        self.items = items
        self.view.addItems(items)
        for i in range(self.view.count()):
            item = self.view.item(i)
            item.setSizeHint(QSize(1, self.itemHeight))

    def _onItemClicked(self, item):
        if False:
            i = 10
            return i + 15
        self._hideMenu(False)
        self.__onItemSelected(item.text())

    def eventFilter(self, obj, e: QEvent):
        if False:
            i = 10
            return i + 15
        if e.type() != QEvent.KeyPress:
            return super().eventFilter(obj, e)
        self.lineEdit.event(e)
        self.view.event(e)
        if e.key() == Qt.Key_Escape:
            self.close()
        if e.key() in [Qt.Key_Enter, Qt.Key_Return] and self.view.currentRow() >= 0:
            self.__onItemSelected(self.view.currentItem().text())
            self.close()
        return super().eventFilter(obj, e)

    def __onItemSelected(self, text):
        if False:
            i = 10
            return i + 15
        self.lineEdit.setText(text)
        self.activated.emit(text)

    def popup(self):
        if False:
            while True:
                i = 10
        ' show menu '
        if not self.items:
            return self.close()
        p = self.lineEdit
        if self.view.width() < p.width():
            self.view.setMinimumWidth(p.width())
            self.adjustSize()
        x = -self.width() // 2 + self.layout().contentsMargins().left() + p.width() // 2
        y = p.height() - self.layout().contentsMargins().top() + 2
        pd = p.mapToGlobal(QPoint(x, y))
        hd = self.view.heightForAnimation(pd, MenuAnimationType.FADE_IN_DROP_DOWN)
        pu = p.mapToGlobal(QPoint(x, 7))
        hu = self.view.heightForAnimation(pd, MenuAnimationType.FADE_IN_PULL_UP)
        if hd >= hu:
            pos = pd
            aniType = MenuAnimationType.FADE_IN_DROP_DOWN
        else:
            pos = pu
            aniType = MenuAnimationType.FADE_IN_PULL_UP
        self.view.adjustSize(pos, aniType)
        self.view.setProperty('dropDown', aniType == MenuAnimationType.FADE_IN_DROP_DOWN)
        self.view.setStyle(QApplication.style())
        self.view.update()
        self.adjustSize()
        self.exec(pos, aniType=aniType)
        self.view.setFocusPolicy(Qt.NoFocus)
        self.setFocusPolicy(Qt.NoFocus)
        p.setFocus()

class SearchLineEdit(LineEdit):
    """ Search line edit """
    searchSignal = pyqtSignal(str)
    clearSignal = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.searchButton = LineEditButton(FIF.SEARCH, self)
        self.hBoxLayout.addWidget(self.searchButton, 0, Qt.AlignRight)
        self.setClearButtonEnabled(True)
        self.setTextMargins(0, 0, 59, 0)
        self.searchButton.clicked.connect(self.search)
        self.clearButton.clicked.connect(self.clearSignal)

    def search(self):
        if False:
            for i in range(10):
                print('nop')
        ' emit search signal '
        text = self.text().strip()
        if text:
            self.searchSignal.emit(text)
        else:
            self.clearSignal.emit()

    def setClearButtonEnabled(self, enable: bool):
        if False:
            print('Hello World!')
        self._isClearButtonEnabled = enable
        self.setTextMargins(0, 0, 28 * enable + 30, 0)

class EditLayer(QWidget):
    """ Edit layer """

    def __init__(self, parent):
        if False:
            return 10
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        parent.installEventFilter(self)

    def eventFilter(self, obj, e):
        if False:
            i = 10
            return i + 15
        if obj is self.parent() and e.type() == QEvent.Resize:
            self.resize(e.size())
        return super().eventFilter(obj, e)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        if not self.parent().hasFocus():
            return
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        m = self.contentsMargins()
        path = QPainterPath()
        (w, h) = (self.width() - m.left() - m.right(), self.height())
        path.addRoundedRect(QRectF(m.left(), h - 10, w, 10), 5, 5)
        rectPath = QPainterPath()
        rectPath.addRect(m.left(), h - 10, w, 7.5)
        path = path.subtracted(rectPath)
        painter.fillPath(path, themeColor())

class TextEdit(QTextEdit):
    """ Text edit """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self.layer = EditLayer(self)
        self.scrollDelegate = SmoothScrollDelegate(self)
        FluentStyleSheet.LINE_EDIT.apply(self)
        setFont(self)

    def contextMenuEvent(self, e):
        if False:
            while True:
                i = 10
        menu = TextEditMenu(self)
        menu.exec_(e.globalPos())

class PlainTextEdit(QPlainTextEdit):
    """ Plain text edit """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self.layer = EditLayer(self)
        self.scrollDelegate = SmoothScrollDelegate(self)
        FluentStyleSheet.LINE_EDIT.apply(self)
        setFont(self)

    def contextMenuEvent(self, e):
        if False:
            print('Hello World!')
        menu = TextEditMenu(self)
        menu.exec_(e.globalPos())

class PasswordLineEdit(LineEdit):
    """ Password line edit """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.viewButton = LineEditButton(FIF.VIEW, self)
        self.setEchoMode(QLineEdit.Password)
        self.setContextMenuPolicy(Qt.NoContextMenu)
        self.hBoxLayout.addWidget(self.viewButton, 0, Qt.AlignRight)
        self.setClearButtonEnabled(False)
        self.viewButton.installEventFilter(self)
        self.viewButton.setIconSize(QSize(13, 13))
        self.viewButton.setFixedSize(29, 25)

    def setPasswordVisible(self, isVisible: bool):
        if False:
            print('Hello World!')
        ' set the visibility of password '
        if isVisible:
            self.setEchoMode(QLineEdit.Normal)
        else:
            self.setEchoMode(QLineEdit.Password)

    def isPasswordVisible(self):
        if False:
            while True:
                i = 10
        return self.echoMode() == QLineEdit.Normal

    def setClearButtonEnabled(self, enable: bool):
        if False:
            for i in range(10):
                print('nop')
        self._isClearButtonEnabled = enable
        if self.viewButton.isHidden():
            self.setTextMargins(0, 0, 28 * enable, 0)
        else:
            self.setTextMargins(0, 0, 28 * enable + 30, 0)

    def setViewPasswordButtonVisible(self, isVisible: bool):
        if False:
            print('Hello World!')
        ' set the visibility of view password button '
        self.viewButton.setVisible(isVisible)

    def eventFilter(self, obj, e):
        if False:
            while True:
                i = 10
        if obj is not self.viewButton:
            return super().eventFilter(obj, e)
        if e.type() == QEvent.MouseButtonPress:
            self.setPasswordVisible(True)
        elif e.type() == QEvent.MouseButtonRelease:
            self.setPasswordVisible(False)
        return super().eventFilter(obj, e)
    passwordVisible = pyqtProperty(bool, isPasswordVisible, setPasswordVisible)