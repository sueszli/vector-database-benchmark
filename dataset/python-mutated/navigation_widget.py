from typing import Union, List
from PyQt5.QtCore import Qt, pyqtSignal, QRect, QRectF, QPropertyAnimation, pyqtProperty, QMargins, QEasingCurve, QPoint, QEvent
from PyQt5.QtGui import QColor, QPainter, QPen, QIcon, QCursor, QFont, QBrush, QPixmap, QImage
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from ...common.config import isDarkTheme
from ...common.style_sheet import themeColor
from ...common.icon import drawIcon, toQIcon
from ...common.icon import FluentIcon as FIF
from ...common.font import setFont

class NavigationWidget(QWidget):
    """ Navigation widget """
    clicked = pyqtSignal(bool)
    EXPAND_WIDTH = 312

    def __init__(self, isSelectable: bool, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.isCompacted = True
        self.isSelected = False
        self.isPressed = False
        self.isEnter = False
        self.isSelectable = isSelectable
        self.treeParent = None
        self.nodeDepth = 0
        self.setFixedSize(40, 36)

    def enterEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.isEnter = True
        self.update()

    def leaveEvent(self, e):
        if False:
            while True:
                i = 10
        self.isEnter = False
        self.isPressed = False
        self.update()

    def mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().mousePressEvent(e)
        self.isPressed = True
        self.update()

    def mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().mouseReleaseEvent(e)
        self.isPressed = False
        self.update()
        self.clicked.emit(True)

    def click(self):
        if False:
            return 10
        self.clicked.emit(True)

    def setCompacted(self, isCompacted: bool):
        if False:
            while True:
                i = 10
        ' set whether the widget is compacted '
        if isCompacted == self.isCompacted:
            return
        self.isCompacted = isCompacted
        if isCompacted:
            self.setFixedSize(40, 36)
        else:
            self.setFixedSize(self.EXPAND_WIDTH, 36)
        self.update()

    def setSelected(self, isSelected: bool):
        if False:
            print('Hello World!')
        ' set whether the button is selected\n\n        Parameters\n        ----------\n        isSelected: bool\n            whether the button is selected\n        '
        if not self.isSelectable:
            return
        self.isSelected = isSelected
        self.update()

class NavigationPushButton(NavigationWidget):
    """ Navigation push button """

    def __init__(self, icon: Union[str, QIcon, FIF], text: str, isSelectable: bool, parent=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        icon: str | QIcon | FluentIconBase\n            the icon to be drawn\n\n        text: str\n            the text of button\n        '
        super().__init__(isSelectable=isSelectable, parent=parent)
        self._icon = icon
        self._text = text
        setFont(self)

    def text(self):
        if False:
            return 10
        return self._text

    def setText(self, text: str):
        if False:
            return 10
        self._text = text
        self.update()

    def icon(self):
        if False:
            for i in range(10):
                print('nop')
        return toQIcon(self._icon)

    def setIcon(self, icon: Union[str, QIcon, FIF]):
        if False:
            i = 10
            return i + 15
        self._icon = icon
        self.update()

    def _margins(self):
        if False:
            while True:
                i = 10
        return QMargins(0, 0, 0, 0)

    def _canDrawIndicator(self):
        if False:
            i = 10
            return i + 15
        return self.isSelected

    def paintEvent(self, e):
        if False:
            while True:
                i = 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.TextAntialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        if self.isPressed:
            painter.setOpacity(0.7)
        if not self.isEnabled():
            painter.setOpacity(0.4)
        c = 255 if isDarkTheme() else 0
        m = self._margins()
        (pl, pr) = (m.left(), m.right())
        globalRect = QRect(self.mapToGlobal(QPoint()), self.size())
        if self._canDrawIndicator():
            painter.setBrush(QColor(c, c, c, 6 if self.isEnter else 10))
            painter.drawRoundedRect(self.rect(), 5, 5)
            painter.setBrush(themeColor())
            painter.drawRoundedRect(pl, 10, 3, 16, 1.5, 1.5)
        elif self.isEnter and self.isEnabled() and globalRect.contains(QCursor.pos()):
            painter.setBrush(QColor(c, c, c, 10))
            painter.drawRoundedRect(self.rect(), 5, 5)
        drawIcon(self._icon, painter, QRectF(11.5 + pl, 10, 16, 16))
        if self.isCompacted:
            return
        painter.setFont(self.font())
        painter.setPen(QColor(c, c, c))
        painter.drawText(QRect(44 + pl, 0, self.width() - 57 - pl - pr, self.height()), Qt.AlignVCenter, self.text())

class NavigationToolButton(NavigationPushButton):
    """ Navigation tool button """

    def __init__(self, icon: Union[str, QIcon, FIF], parent=None):
        if False:
            return 10
        super().__init__(icon, '', False, parent)

    def setCompacted(self, isCompacted: bool):
        if False:
            print('Hello World!')
        self.setFixedSize(40, 36)

class NavigationSeparator(NavigationWidget):
    """ Navigation Separator """

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(False, parent=parent)
        self.setCompacted(True)

    def setCompacted(self, isCompacted: bool):
        if False:
            i = 10
            return i + 15
        if isCompacted:
            self.setFixedSize(48, 3)
        else:
            self.setFixedSize(self.EXPAND_WIDTH + 10, 3)
        self.update()

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        painter = QPainter(self)
        c = 255 if isDarkTheme() else 0
        pen = QPen(QColor(c, c, c, 15))
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawLine(0, 1, self.width(), 1)

class NavigationTreeItem(NavigationPushButton):
    """ Navigation tree item widget """
    itemClicked = pyqtSignal(bool, bool)

    def __init__(self, icon: Union[str, QIcon, FIF], text: str, isSelectable: bool, parent=None):
        if False:
            print('Hello World!')
        super().__init__(icon, text, isSelectable, parent)
        self._arrowAngle = 0
        self.rotateAni = QPropertyAnimation(self, b'arrowAngle', self)

    def setExpanded(self, isExpanded: bool):
        if False:
            for i in range(10):
                print('nop')
        self.rotateAni.stop()
        self.rotateAni.setEndValue(180 if isExpanded else 0)
        self.rotateAni.setDuration(150)
        self.rotateAni.start()

    def mouseReleaseEvent(self, e):
        if False:
            while True:
                i = 10
        super().mouseReleaseEvent(e)
        clickArrow = QRectF(self.width() - 30, 8, 20, 20).contains(e.pos())
        self.itemClicked.emit(True, clickArrow and (not self.parent().isLeaf()))
        self.update()

    def _canDrawIndicator(self):
        if False:
            while True:
                i = 10
        p = self.parent()
        if p.isLeaf() or p.isSelected:
            return p.isSelected
        for child in p.treeChildren:
            if child.itemWidget._canDrawIndicator() and (not child.isVisible()):
                return True
        return False

    def _margins(self):
        if False:
            print('Hello World!')
        p = self.parent()
        return QMargins(p.nodeDepth * 28, 0, 20 * bool(p.treeChildren), 0)

    def paintEvent(self, e):
        if False:
            return 10
        super().paintEvent(e)
        if self.isCompacted or not self.parent().treeChildren:
            return
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        if self.isPressed:
            painter.setOpacity(0.7)
        if not self.isEnabled():
            painter.setOpacity(0.4)
        painter.translate(self.width() - 20, 18)
        painter.rotate(self.arrowAngle)
        FIF.ARROW_DOWN.render(painter, QRectF(-5, -5, 9.6, 9.6))

    def getArrowAngle(self):
        if False:
            while True:
                i = 10
        return self._arrowAngle

    def setArrowAngle(self, angle):
        if False:
            return 10
        self._arrowAngle = angle
        self.update()
    arrowAngle = pyqtProperty(float, getArrowAngle, setArrowAngle)

class NavigationTreeWidgetBase(NavigationWidget):
    """ Navigation tree widget base class """

    def addChild(self, child):
        if False:
            while True:
                i = 10
        ' add child\n\n        Parameters\n        ----------\n        child: NavigationTreeWidgetBase\n            child item\n        '
        raise NotImplementedError

    def insertChild(self, index: int, child: NavigationWidget):
        if False:
            for i in range(10):
                print('nop')
        ' insert child\n\n        Parameters\n        ----------\n        child: NavigationTreeWidgetBase\n            child item\n        '
        raise NotImplementedError

    def removeChild(self, child: NavigationWidget):
        if False:
            i = 10
            return i + 15
        ' remove child\n\n        Parameters\n        ----------\n        child: NavigationTreeWidgetBase\n            child item\n        '
        raise NotImplementedError

    def isRoot(self):
        if False:
            return 10
        ' is root node '
        return True

    def isLeaf(self):
        if False:
            print('Hello World!')
        ' is leaf node '
        return True

    def setExpanded(self, isExpanded: bool):
        if False:
            for i in range(10):
                print('nop')
        ' set the expanded status\n\n        Parameters\n        ----------\n        isExpanded: bool\n            whether to expand node\n        '
        raise NotImplementedError

    def childItems(self) -> list:
        if False:
            return 10
        ' return child items '
        raise NotImplementedError

class NavigationTreeWidget(NavigationTreeWidgetBase):
    """ Navigation tree widget """

    def __init__(self, icon: Union[str, QIcon, FIF], text: str, isSelectable: bool, parent=None):
        if False:
            print('Hello World!')
        super().__init__(isSelectable, parent)
        self.treeChildren = []
        self.isExpanded = False
        self.itemWidget = NavigationTreeItem(icon, text, isSelectable, self)
        self.vBoxLayout = QVBoxLayout(self)
        self.expandAni = QPropertyAnimation(self, b'geometry', self)
        self.__initWidget()

    def __initWidget(self):
        if False:
            return 10
        self.vBoxLayout.setSpacing(4)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.itemWidget, 0, Qt.AlignTop)
        self.itemWidget.itemClicked.connect(self._onClicked)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.expandAni.valueChanged.connect(lambda g: self.setFixedSize(g.size()))

    def addChild(self, child):
        if False:
            while True:
                i = 10
        self.insertChild(-1, child)

    def text(self):
        if False:
            print('Hello World!')
        return self.itemWidget.text()

    def icon(self):
        if False:
            print('Hello World!')
        return self.itemWidget.icon()

    def setText(self, text):
        if False:
            return 10
        self.itemWidget.setText(text)

    def setIcon(self, icon: Union[str, QIcon, FIF]):
        if False:
            while True:
                i = 10
        self.itemWidget.setIcon(icon)

    def setFont(self, font: QFont):
        if False:
            while True:
                i = 10
        super().setFont(font)
        self.itemWidget.setFont(font)

    def insertChild(self, index, child):
        if False:
            for i in range(10):
                print('nop')
        if child in self.treeChildren:
            return
        child.treeParent = self
        child.nodeDepth = self.nodeDepth + 1
        child.setVisible(self.isExpanded)
        child.expandAni.valueChanged.connect(lambda : self.setFixedSize(self.sizeHint()))
        p = self.treeParent
        while p:
            child.expandAni.valueChanged.connect(lambda v, p=p: p.setFixedSize(p.sizeHint()))
            p = p.treeParent
        if index < 0:
            index = len(self.treeChildren)
        index += 1
        self.treeChildren.insert(index, child)
        self.vBoxLayout.insertWidget(index, child, 0, Qt.AlignTop)

    def removeChild(self, child):
        if False:
            for i in range(10):
                print('nop')
        self.treeChildren.remove(child)
        self.vBoxLayout.removeWidget(child)

    def childItems(self) -> list:
        if False:
            while True:
                i = 10
        return self.treeChildren

    def setExpanded(self, isExpanded: bool, ani=False):
        if False:
            for i in range(10):
                print('nop')
        ' set the expanded status '
        if isExpanded == self.isExpanded:
            return
        self.isExpanded = isExpanded
        self.itemWidget.setExpanded(isExpanded)
        for child in self.treeChildren:
            child.setVisible(isExpanded)
            child.setFixedSize(child.sizeHint())
        if ani:
            self.expandAni.stop()
            self.expandAni.setStartValue(self.geometry())
            self.expandAni.setEndValue(QRect(self.pos(), self.sizeHint()))
            self.expandAni.setDuration(120)
            self.expandAni.setEasingCurve(QEasingCurve.OutQuad)
            self.expandAni.start()
        else:
            self.setFixedSize(self.sizeHint())

    def isRoot(self):
        if False:
            i = 10
            return i + 15
        return self.treeParent is None

    def isLeaf(self):
        if False:
            print('Hello World!')
        return len(self.treeChildren) == 0

    def setSelected(self, isSelected: bool):
        if False:
            print('Hello World!')
        super().setSelected(isSelected)
        self.itemWidget.setSelected(isSelected)

    def mouseReleaseEvent(self, e):
        if False:
            i = 10
            return i + 15
        pass

    def setCompacted(self, isCompacted: bool):
        if False:
            while True:
                i = 10
        super().setCompacted(isCompacted)
        self.itemWidget.setCompacted(isCompacted)

    def _onClicked(self, triggerByUser, clickArrow):
        if False:
            return 10
        if not self.isCompacted:
            if self.isSelectable and (not self.isSelected) and (not clickArrow):
                self.setExpanded(True, ani=True)
            else:
                self.setExpanded(not self.isExpanded, ani=True)
        if not clickArrow or self.isCompacted:
            self.clicked.emit(triggerByUser)

class NavigationAvatarWidget(NavigationWidget):
    """ Avatar widget """

    def __init__(self, name: str, avatar: Union[str, QPixmap, QImage], parent=None):
        if False:
            return 10
        super().__init__(isSelectable=False, parent=parent)
        self.name = name
        self.setAvatar(avatar)
        setFont(self)

    def setName(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        self.name = name
        self.update()

    def setAvatar(self, avatar: Union[str, QPixmap, QImage]):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(avatar, str):
            avatar = QImage(avatar)
        elif isinstance(avatar, QPixmap):
            avatar = avatar.toImage()
        self.avatar = avatar.scaled(24, 24, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def paintEvent(self, e):
        if False:
            return 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.SmoothPixmapTransform | QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        if self.isPressed:
            painter.setOpacity(0.7)
        if self.isEnter:
            c = 255 if isDarkTheme() else 0
            painter.setBrush(QColor(c, c, c, 10))
            painter.drawRoundedRect(self.rect(), 5, 5)
        painter.setBrush(QBrush(self.avatar))
        painter.translate(8, 6)
        painter.drawEllipse(0, 0, 24, 24)
        painter.translate(-8, -6)
        if not self.isCompacted:
            painter.setPen(Qt.white if isDarkTheme() else Qt.black)
            painter.setFont(self.font())
            painter.drawText(QRect(44, 0, 255, 36), Qt.AlignVCenter, self.name)