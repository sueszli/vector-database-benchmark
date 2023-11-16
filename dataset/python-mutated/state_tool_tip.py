from PyQt5.QtCore import QPropertyAnimation, Qt, QTimer, pyqtSignal, QPoint, QRectF
from PyQt5.QtGui import QPainter
from PyQt5.QtWidgets import QLabel, QWidget, QToolButton, QGraphicsOpacityEffect
from ...common import FluentStyleSheet, isDarkTheme, Theme
from ...common.icon import FluentIcon as FIF

class StateCloseButton(QToolButton):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.setFixedSize(12, 12)
        self.isPressed = False
        self.isEnter = False

    def enterEvent(self, e):
        if False:
            while True:
                i = 10
        self.isEnter = True
        self.update()

    def leaveEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.isEnter = False
        self.isPressed = False
        self.update()

    def mousePressEvent(self, e):
        if False:
            print('Hello World!')
        self.isPressed = True
        self.update()
        super().mousePressEvent(e)

    def mouseReleaseEvent(self, e):
        if False:
            print('Hello World!')
        self.isPressed = False
        self.update()
        super().mouseReleaseEvent(e)

    def paintEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        if self.isPressed:
            painter.setOpacity(0.6)
        elif self.isEnter:
            painter.setOpacity(0.8)
        theme = Theme.DARK if not isDarkTheme() else Theme.LIGHT
        FIF.CLOSE.render(painter, self.rect(), theme)

class StateToolTip(QWidget):
    """ State tooltip """
    closedSignal = pyqtSignal()

    def __init__(self, title, content, parent=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        title: str\n            title of tooltip\n\n        content: str\n            content of tooltip\n\n        parant:\n            parent window\n        '
        super().__init__(parent)
        self.title = title
        self.content = content
        self.titleLabel = QLabel(self.title, self)
        self.contentLabel = QLabel(self.content, self)
        self.rotateTimer = QTimer(self)
        self.opacityEffect = QGraphicsOpacityEffect(self)
        self.animation = QPropertyAnimation(self.opacityEffect, b'opacity')
        self.closeButton = StateCloseButton(self)
        self.isDone = False
        self.rotateAngle = 0
        self.deltaAngle = 20
        self.__initWidget()

    def __initWidget(self):
        if False:
            i = 10
            return i + 15
        ' initialize widgets '
        self.setAttribute(Qt.WA_StyledBackground)
        self.setGraphicsEffect(self.opacityEffect)
        self.opacityEffect.setOpacity(1)
        self.rotateTimer.setInterval(50)
        self.contentLabel.setMinimumWidth(200)
        self.closeButton.clicked.connect(self.__onCloseButtonClicked)
        self.rotateTimer.timeout.connect(self.__rotateTimerFlowSlot)
        self.__setQss()
        self.__initLayout()
        self.rotateTimer.start()

    def __initLayout(self):
        if False:
            for i in range(10):
                print('nop')
        ' initialize layout '
        self.setFixedSize(max(self.titleLabel.width(), self.contentLabel.width()) + 56, 51)
        self.titleLabel.move(32, 9)
        self.contentLabel.move(12, 27)
        self.closeButton.move(self.width() - 24, 19)

    def __setQss(self):
        if False:
            while True:
                i = 10
        ' set style sheet '
        self.titleLabel.setObjectName('titleLabel')
        self.contentLabel.setObjectName('contentLabel')
        FluentStyleSheet.STATE_TOOL_TIP.apply(self)
        self.titleLabel.adjustSize()
        self.contentLabel.adjustSize()

    def setTitle(self, title: str):
        if False:
            i = 10
            return i + 15
        ' set the title of tooltip '
        self.title = title
        self.titleLabel.setText(title)
        self.titleLabel.adjustSize()

    def setContent(self, content: str):
        if False:
            while True:
                i = 10
        ' set the content of tooltip '
        self.content = content
        self.contentLabel.setText(content)
        self.contentLabel.adjustSize()

    def setState(self, isDone=False):
        if False:
            print('Hello World!')
        ' set the state of tooltip '
        self.isDone = isDone
        self.update()
        if isDone:
            QTimer.singleShot(1000, self.__fadeOut)

    def __onCloseButtonClicked(self):
        if False:
            i = 10
            return i + 15
        ' close button clicked slot '
        self.closedSignal.emit()
        self.hide()

    def __fadeOut(self):
        if False:
            return 10
        ' fade out '
        self.rotateTimer.stop()
        self.animation.setDuration(200)
        self.animation.setStartValue(1)
        self.animation.setEndValue(0)
        self.animation.finished.connect(self.deleteLater)
        self.animation.start()

    def __rotateTimerFlowSlot(self):
        if False:
            i = 10
            return i + 15
        ' rotate timer time out slot '
        self.rotateAngle = (self.rotateAngle + self.deltaAngle) % 360
        self.update()

    def getSuitablePos(self):
        if False:
            print('Hello World!')
        ' get suitable position in main window '
        for i in range(10):
            dy = i * (self.height() + 16)
            pos = QPoint(self.parent().width() - self.width() - 24, 50 + dy)
            widget = self.parent().childAt(pos + QPoint(2, 2))
            if isinstance(widget, StateToolTip):
                pos += QPoint(0, self.height() + 16)
            else:
                break
        return pos

    def paintEvent(self, e):
        if False:
            i = 10
            return i + 15
        ' paint state tooltip '
        super().paintEvent(e)
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        theme = Theme.DARK if not isDarkTheme() else Theme.LIGHT
        if not self.isDone:
            painter.translate(19, 18)
            painter.rotate(self.rotateAngle)
            FIF.SYNC.render(painter, QRectF(-8, -8, 16, 16), theme)
        else:
            FIF.COMPLETED.render(painter, QRectF(11, 10, 16, 16), theme)