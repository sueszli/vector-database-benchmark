from PyQt5.QtCore import Qt, pyqtSignal, QRectF, pyqtProperty, QPropertyAnimation, QPoint
from PyQt5.QtGui import QPixmap, QPainter, QColor, QPainterPath, QFont
from PyQt5.QtWidgets import QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel
from ...common.overload import singledispatchmethod
from ...common.style_sheet import isDarkTheme, FluentStyleSheet
from ...common.animation import BackgroundAnimationWidget, DropShadowAnimation
from ...common.font import setFont

class CardWidget(BackgroundAnimationWidget, QFrame):
    """ Card widget """
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent=parent)
        self._isClickEnabled = False
        self._borderRadius = 5

    def mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        super().mouseReleaseEvent(e)
        self.clicked.emit()

    def setClickEnabled(self, isEnabled: bool):
        if False:
            while True:
                i = 10
        self._isClickEnabled = isEnabled
        self.update()

    def isClickEnabled(self):
        if False:
            return 10
        return self._isClickEnabled

    def _normalBackgroundColor(self):
        if False:
            return 10
        return QColor(255, 255, 255, 13 if isDarkTheme() else 170)

    def _hoverBackgroundColor(self):
        if False:
            for i in range(10):
                print('nop')
        return QColor(255, 255, 255, 21 if isDarkTheme() else 64)

    def _pressedBackgroundColor(self):
        if False:
            for i in range(10):
                print('nop')
        return QColor(255, 255, 255, 8 if isDarkTheme() else 64)

    def getBorderRadius(self):
        if False:
            return 10
        return self._borderRadius

    def setBorderRadius(self, radius: int):
        if False:
            return 10
        self._borderRadius = radius
        self.update()

    def paintEvent(self, e):
        if False:
            print('Hello World!')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        (w, h) = (self.width(), self.height())
        r = 5
        d = 2 * r
        isDark = isDarkTheme()
        path = QPainterPath()
        path.arcMoveTo(1, h - d - 1, d, d, 225)
        path.arcTo(1, h - d - 1, d, d, 225, -45)
        path.lineTo(1, r)
        path.arcTo(1, 1, d, d, -180, -90)
        path.lineTo(w - r, 1)
        path.arcTo(w - d - 1, 1, d, d, 90, -90)
        path.lineTo(w - 1, h - r)
        path.arcTo(w - d - 1, h - d - 1, d, d, 0, -45)
        topBorderColor = QColor(0, 0, 0, 20)
        if isDark:
            if self.isPressed:
                topBorderColor = QColor(255, 255, 255, 18)
            elif self.isHover:
                topBorderColor = QColor(255, 255, 255, 13)
        else:
            topBorderColor = QColor(0, 0, 0, 15)
        painter.strokePath(path, topBorderColor)
        path = QPainterPath()
        path.arcMoveTo(1, h - d - 1, d, d, 225)
        path.arcTo(1, h - d - 1, d, d, 225, 45)
        path.lineTo(w - r - 1, h - 1)
        path.arcTo(w - d - 1, h - d - 1, d, d, 270, 45)
        bottomBorderColor = topBorderColor
        if not isDark and self.isHover and (not self.isPressed):
            bottomBorderColor = QColor(0, 0, 0, 27)
        painter.strokePath(path, bottomBorderColor)
        painter.setPen(Qt.NoPen)
        rect = self.rect().adjusted(1, 1, -1, -1)
        painter.setBrush(self.backgroundColor)
        painter.drawRoundedRect(rect, r, r)
    borderRadius = pyqtProperty(int, getBorderRadius, setBorderRadius)

class SimpleCardWidget(CardWidget):
    """ Simple card widget """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)

    def _normalBackgroundColor(self):
        if False:
            for i in range(10):
                print('nop')
        return QColor(255, 255, 255, 13 if isDarkTheme() else 170)

    def _hoverBackgroundColor(self):
        if False:
            while True:
                i = 10
        return self._normalBackgroundColor()

    def _pressedBackgroundColor(self):
        if False:
            while True:
                i = 10
        return self._normalBackgroundColor()

    def paintEvent(self, e):
        if False:
            while True:
                i = 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setBrush(self.backgroundColor)
        if isDarkTheme():
            painter.setPen(QColor(0, 0, 0, 48))
        else:
            painter.setPen(QColor(0, 0, 0, 12))
        r = self.borderRadius
        painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), r, r)

class ElevatedCardWidget(SimpleCardWidget):
    """ Card widget with shadow effect """

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.shadowAni = DropShadowAnimation(self, hoverColor=QColor(0, 0, 0, 20))
        self.shadowAni.setOffset(0, 5)
        self.shadowAni.setBlurRadius(38)
        self.elevatedAni = QPropertyAnimation(self, b'pos', self)
        self.elevatedAni.setDuration(100)
        self._originalPos = self.pos()
        self.setBorderRadius(8)

    def enterEvent(self, e):
        if False:
            return 10
        super().enterEvent(e)
        if self.elevatedAni.state() != QPropertyAnimation.Running:
            self._originalPos = self.pos()
        self._startElevateAni(self.pos(), self.pos() - QPoint(0, 3))

    def leaveEvent(self, e):
        if False:
            i = 10
            return i + 15
        super().leaveEvent(e)
        self._startElevateAni(self.pos(), self._originalPos)

    def mousePressEvent(self, e):
        if False:
            return 10
        super().mousePressEvent(e)
        self._startElevateAni(self.pos(), self._originalPos)

    def _startElevateAni(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        self.elevatedAni.setStartValue(start)
        self.elevatedAni.setEndValue(end)
        self.elevatedAni.start()

    def _hoverBackgroundColor(self):
        if False:
            i = 10
            return i + 15
        return QColor(255, 255, 255, 16) if isDarkTheme() else QColor(255, 255, 255)

    def _pressedBackgroundColor(self):
        if False:
            for i in range(10):
                print('nop')
        return QColor(255, 255, 255, 6 if isDarkTheme() else 118)

class CardSeparator(QWidget):
    """ Card separator """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent=parent)
        self.setFixedHeight(3)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if isDarkTheme():
            painter.setPen(QColor(255, 255, 255, 46))
        else:
            painter.setPen(QColor(0, 0, 0, 12))
        painter.drawLine(2, 1, self.width() - 2, 1)

class HeaderCardWidget(SimpleCardWidget):
    """ Header card widget """

    @singledispatchmethod
    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.headerView = QWidget(self)
        self.headerLabel = QLabel(self)
        self.separator = CardSeparator(self)
        self.view = QWidget(self)
        self.vBoxLayout = QVBoxLayout(self)
        self.headerLayout = QHBoxLayout(self.headerView)
        self.viewLayout = QHBoxLayout(self.view)
        self.headerLayout.addWidget(self.headerLabel)
        self.headerLayout.setContentsMargins(24, 0, 16, 0)
        self.headerView.setFixedHeight(48)
        self.vBoxLayout.setSpacing(0)
        self.vBoxLayout.setContentsMargins(0, 0, 0, 0)
        self.vBoxLayout.addWidget(self.headerView)
        self.vBoxLayout.addWidget(self.separator)
        self.vBoxLayout.addWidget(self.view)
        self.viewLayout.setContentsMargins(24, 24, 24, 24)
        setFont(self.headerLabel, 15, QFont.DemiBold)
        self.view.setObjectName('view')
        self.headerView.setObjectName('headerView')
        self.headerLabel.setObjectName('headerLabel')
        FluentStyleSheet.CARD_WIDGET.apply(self)

    @__init__.register
    def _(self, title: str, parent=None):
        if False:
            return 10
        self.__init__(parent)
        self.setTitle(title)

    def getTitle(self):
        if False:
            return 10
        return self.headerLabel.text()

    def setTitle(self, title: str):
        if False:
            return 10
        self.headerLabel.setText(title)
    title = pyqtProperty(str, getTitle, setTitle)