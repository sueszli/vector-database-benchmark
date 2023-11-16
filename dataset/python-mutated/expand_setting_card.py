from typing import Union
from PyQt5.QtCore import QEvent, Qt, QPropertyAnimation, pyqtProperty, QEasingCurve, QRectF
from PyQt5.QtGui import QColor, QPainter, QIcon, QPainterPath
from PyQt5.QtWidgets import QFrame, QWidget, QAbstractButton, QApplication, QScrollArea, QVBoxLayout
from ...common.config import isDarkTheme
from ...common.icon import FluentIcon as FIF
from ...common.style_sheet import FluentStyleSheet
from .setting_card import SettingCard
from ..layout.v_box_layout import VBoxLayout

class ExpandButton(QAbstractButton):
    """ Expand button """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.setFixedSize(30, 30)
        self.__angle = 0
        self.isHover = False
        self.isPressed = False
        self.rotateAni = QPropertyAnimation(self, b'angle', self)
        self.clicked.connect(self.__onClicked)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        painter.setPen(Qt.NoPen)
        r = 255 if isDarkTheme() else 0
        if self.isPressed:
            color = QColor(r, r, r, 10)
        elif self.isHover:
            color = QColor(r, r, r, 14)
        else:
            color = Qt.transparent
        painter.setBrush(color)
        painter.drawRoundedRect(self.rect(), 4, 4)
        painter.translate(self.width() // 2, self.height() // 2)
        painter.rotate(self.__angle)
        FIF.ARROW_DOWN.render(painter, QRectF(-5, -5, 9.6, 9.6))

    def enterEvent(self, e):
        if False:
            while True:
                i = 10
        self.setHover(True)

    def leaveEvent(self, e):
        if False:
            return 10
        self.setHover(False)

    def mousePressEvent(self, e):
        if False:
            return 10
        super().mousePressEvent(e)
        self.setPressed(True)

    def mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().mouseReleaseEvent(e)
        self.setPressed(False)

    def setHover(self, isHover: bool):
        if False:
            i = 10
            return i + 15
        self.isHover = isHover
        self.update()

    def setPressed(self, isPressed: bool):
        if False:
            i = 10
            return i + 15
        self.isPressed = isPressed
        self.update()

    def __onClicked(self):
        if False:
            i = 10
            return i + 15
        self.rotateAni.setEndValue(180 if self.angle < 180 else 0)
        self.rotateAni.setDuration(200)
        self.rotateAni.start()

    def getAngle(self):
        if False:
            i = 10
            return i + 15
        return self.__angle

    def setAngle(self, angle):
        if False:
            print('Hello World!')
        self.__angle = angle
        self.update()
    angle = pyqtProperty(float, getAngle, setAngle)

class SpaceWidget(QWidget):
    """ Spacing widget """

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedHeight(1)

class HeaderSettingCard(SettingCard):
    """ Header setting card """

    def __init__(self, icon, title, content=None, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(icon, title, content, parent)
        self.expandButton = ExpandButton(self)
        self.hBoxLayout.addWidget(self.expandButton, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(8)
        self.installEventFilter(self)

    def eventFilter(self, obj, e):
        if False:
            for i in range(10):
                print('nop')
        if obj is self:
            if e.type() == QEvent.Enter:
                self.expandButton.setHover(True)
            elif e.type() == QEvent.Leave:
                self.expandButton.setHover(False)
            elif e.type() == QEvent.MouseButtonPress and e.button() == Qt.LeftButton:
                self.expandButton.setPressed(True)
            elif e.type() == QEvent.MouseButtonRelease and e.button() == Qt.LeftButton:
                self.expandButton.setPressed(False)
                self.expandButton.click()
        return super().eventFilter(obj, e)

    def addWidget(self, widget: QWidget):
        if False:
            while True:
                i = 10
        ' add widget to tail '
        N = self.hBoxLayout.count()
        self.hBoxLayout.removeItem(self.hBoxLayout.itemAt(N - 1))
        self.hBoxLayout.addWidget(widget, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(19)
        self.hBoxLayout.addWidget(self.expandButton, 0, Qt.AlignRight)
        self.hBoxLayout.addSpacing(8)

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        if isDarkTheme():
            painter.setBrush(QColor(255, 255, 255, 13))
        else:
            painter.setBrush(QColor(255, 255, 255, 170))
        p = self.parent()
        path = QPainterPath()
        path.setFillRule(Qt.WindingFill)
        path.addRoundedRect(QRectF(self.rect().adjusted(1, 1, -1, -1)), 6, 6)
        if p.isExpand:
            path.addRect(1, self.height() - 8, self.width() - 2, 8)
        painter.drawPath(path.simplified())

class ExpandBorderWidget(QWidget):
    """ Expand setting card border widget """

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        parent.installEventFilter(self)

    def eventFilter(self, obj, e):
        if False:
            return 10
        if obj is self.parent() and e.type() == QEvent.Resize:
            self.resize(e.size())
        return super().eventFilter(obj, e)

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setBrush(Qt.NoBrush)
        if isDarkTheme():
            painter.setPen(QColor(0, 0, 0, 50))
        else:
            painter.setPen(QColor(0, 0, 0, 19))
        p = self.parent()
        (r, d) = (6, 12)
        (ch, h, w) = (p.card.height(), self.height(), self.width())
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)
        if ch < h:
            painter.drawLine(1, ch, w - 1, ch)

class ExpandSettingCard(QScrollArea):
    """ Expandable setting card """

    def __init__(self, icon: Union[str, QIcon, FIF], title: str, content: str=None, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent=parent)
        self.isExpand = False
        self.scrollWidget = QFrame(self)
        self.view = QFrame(self.scrollWidget)
        self.card = HeaderSettingCard(icon, title, content, self)
        self.scrollLayout = QVBoxLayout(self.scrollWidget)
        self.viewLayout = QVBoxLayout(self.view)
        self.spaceWidget = SpaceWidget(self.scrollWidget)
        self.borderWidget = ExpandBorderWidget(self)
        self.expandAni = QPropertyAnimation(self.verticalScrollBar(), b'value', self)
        self.__initWidget()

    def __initWidget(self):
        if False:
            return 10
        ' initialize widgets '
        self.setWidget(self.scrollWidget)
        self.setWidgetResizable(True)
        self.setFixedHeight(self.card.height())
        self.setViewportMargins(0, self.card.height(), 0, 0)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scrollLayout.setContentsMargins(0, 0, 0, 0)
        self.scrollLayout.setSpacing(0)
        self.scrollLayout.addWidget(self.view)
        self.scrollLayout.addWidget(self.spaceWidget)
        self.expandAni.setEasingCurve(QEasingCurve.OutQuad)
        self.expandAni.setDuration(200)
        self.view.setObjectName('view')
        self.scrollWidget.setObjectName('scrollWidget')
        self.setProperty('isExpand', False)
        FluentStyleSheet.EXPAND_SETTING_CARD.apply(self.card)
        FluentStyleSheet.EXPAND_SETTING_CARD.apply(self)
        self.card.installEventFilter(self)
        self.expandAni.valueChanged.connect(self._onExpandValueChanged)
        self.card.expandButton.clicked.connect(self.toggleExpand)

    def addWidget(self, widget: QWidget):
        if False:
            while True:
                i = 10
        ' add widget to tail '
        self.card.addWidget(widget)

    def wheelEvent(self, e):
        if False:
            i = 10
            return i + 15
        pass

    def setExpand(self, isExpand: bool):
        if False:
            while True:
                i = 10
        ' set the expand status of card '
        if self.isExpand == isExpand:
            return
        self.isExpand = isExpand
        self.setProperty('isExpand', isExpand)
        self.setStyle(QApplication.style())
        if isExpand:
            h = self.viewLayout.sizeHint().height()
            self.verticalScrollBar().setValue(h)
            self.expandAni.setStartValue(h)
            self.expandAni.setEndValue(0)
        else:
            self.expandAni.setStartValue(0)
            self.expandAni.setEndValue(self.verticalScrollBar().maximum())
        self.expandAni.start()

    def toggleExpand(self):
        if False:
            while True:
                i = 10
        ' toggle expand status '
        self.setExpand(not self.isExpand)

    def resizeEvent(self, e):
        if False:
            print('Hello World!')
        self.card.resize(self.width(), self.card.height())
        self.scrollWidget.resize(self.width(), self.scrollWidget.height())

    def _onExpandValueChanged(self):
        if False:
            while True:
                i = 10
        vh = self.viewLayout.sizeHint().height()
        h = self.viewportMargins().top()
        self.setFixedHeight(max(h + vh - self.verticalScrollBar().value(), h))

    def _adjustViewSize(self):
        if False:
            while True:
                i = 10
        ' adjust view size '
        h = self.viewLayout.sizeHint().height()
        self.spaceWidget.setFixedHeight(h)
        if self.isExpand:
            self.setFixedHeight(self.card.height() + h)

    def setValue(self, value):
        if False:
            i = 10
            return i + 15
        ' set the value of config item '
        pass

class GroupSeparator(QWidget):
    """ group separator """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self.setFixedHeight(3)

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if isDarkTheme():
            painter.setPen(QColor(0, 0, 0, 50))
        else:
            painter.setPen(QColor(0, 0, 0, 19))
        painter.drawLine(0, 1, self.width(), 1)

class ExpandGroupSettingCard(ExpandSettingCard):
    """ Expand group setting card """

    def addGroupWidget(self, widget: QWidget):
        if False:
            while True:
                i = 10
        ' add widget to group '
        if self.viewLayout.count() >= 1:
            self.viewLayout.addWidget(GroupSeparator(self.view))
        widget.setParent(self.view)
        self.viewLayout.addWidget(widget)
        self._adjustViewSize()