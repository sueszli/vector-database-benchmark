from enum import Enum
from PyQt5.QtCore import QEvent, QObject, QPoint, QTimer, Qt, QPropertyAnimation, QSize
from PyQt5.QtGui import QColor, QCursor
from PyQt5.QtWidgets import QApplication, QFrame, QGraphicsDropShadowEffect, QHBoxLayout, QLabel, QWidget
from ...common import FluentStyleSheet

class ToolTipPosition(Enum):
    """ Info bar position """
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
    TOP_LEFT = 4
    TOP_RIGHT = 5
    BOTTOM_LEFT = 6
    BOTTOM_RIGHT = 7

class ToolTip(QFrame):
    """ Tool tip """

    def __init__(self, text='', parent=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        text: str\n            the text of tool tip\n\n        parent: QWidget\n            parent widget\n        '
        super().__init__(parent=parent)
        self.__text = text
        self.__duration = 1000
        self.container = self._createContainer()
        self.timer = QTimer(self)
        self.setLayout(QHBoxLayout())
        self.containerLayout = QHBoxLayout(self.container)
        self.label = QLabel(text, self)
        self.layout().setContentsMargins(12, 8, 12, 12)
        self.layout().addWidget(self.container)
        self.containerLayout.addWidget(self.label)
        self.containerLayout.setContentsMargins(8, 6, 8, 6)
        self.opacityAni = QPropertyAnimation(self, b'windowOpacity', self)
        self.opacityAni.setDuration(150)
        self.shadowEffect = QGraphicsDropShadowEffect(self)
        self.shadowEffect.setBlurRadius(25)
        self.shadowEffect.setColor(QColor(0, 0, 0, 60))
        self.shadowEffect.setOffset(0, 5)
        self.container.setGraphicsEffect(self.shadowEffect)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.hide)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setWindowFlags(Qt.Tool | Qt.FramelessWindowHint)
        self.__setQss()

    def text(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__text

    def setText(self, text):
        if False:
            while True:
                i = 10
        ' set text on tooltip '
        self.__text = text
        self.label.setText(text)
        self.container.adjustSize()
        self.adjustSize()

    def duration(self):
        if False:
            for i in range(10):
                print('nop')
        return self.__duration

    def setDuration(self, duration: int):
        if False:
            for i in range(10):
                print('nop')
        " set tooltip duration in milliseconds\n\n        Parameters\n        ----------\n        duration: int\n            display duration in milliseconds, if `duration <= 0`, tooltip won't disappear automatically\n        "
        self.__duration = duration

    def __setQss(self):
        if False:
            while True:
                i = 10
        ' set style sheet '
        self.container.setObjectName('container')
        self.label.setObjectName('contentLabel')
        FluentStyleSheet.TOOL_TIP.apply(self)
        self.label.adjustSize()
        self.adjustSize()

    def _createContainer(self):
        if False:
            print('Hello World!')
        return QFrame(self)

    def showEvent(self, e):
        if False:
            print('Hello World!')
        self.opacityAni.setStartValue(0)
        self.opacityAni.setEndValue(1)
        self.opacityAni.start()
        self.timer.stop()
        if self.duration() > 0:
            self.timer.start(self.__duration + self.opacityAni.duration())
        super().showEvent(e)

    def hideEvent(self, e):
        if False:
            while True:
                i = 10
        self.timer.stop()
        super().hideEvent(e)

    def adjustPos(self, widget, position: ToolTipPosition):
        if False:
            print('Hello World!')
        ' adjust the position of tooltip relative to widget '
        manager = ToolTipPositionManager.make(position)
        self.move(manager.position(self, widget))

class ToolTipPositionManager:
    """ Tooltip position manager """

    def position(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            while True:
                i = 10
        pos = self._pos(tooltip, parent)
        (x, y) = (pos.x(), pos.y())
        rect = QApplication.screenAt(QCursor.pos()).availableGeometry()
        x = max(rect.left(), min(pos.x(), rect.right() - tooltip.width() - 4))
        y = max(rect.top(), min(pos.y(), rect.bottom() - tooltip.height() - 4))
        return QPoint(x, y)

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            return 10
        raise NotImplementedError

    @staticmethod
    def make(position: ToolTipPosition):
        if False:
            for i in range(10):
                print('nop')
        ' mask info bar manager according to the display position '
        managers = {ToolTipPosition.TOP: TopToolTipManager, ToolTipPosition.BOTTOM: BottomToolTipManager, ToolTipPosition.LEFT: LeftToolTipManager, ToolTipPosition.RIGHT: RightToolTipManager, ToolTipPosition.TOP_RIGHT: TopRightToolTipManager, ToolTipPosition.BOTTOM_RIGHT: BottomRightToolTipManager, ToolTipPosition.TOP_LEFT: TopLeftToolTipManager, ToolTipPosition.BOTTOM_LEFT: BottomLeftToolTipManager}
        if position not in managers:
            raise ValueError(f'`{position}` is an invalid info bar position.')
        return managers[position]()

class TopToolTipManager(ToolTipPositionManager):
    """ Top tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget):
        if False:
            print('Hello World!')
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() + parent.width() // 2 - tooltip.width() // 2
        y = pos.y() - tooltip.height()
        return QPoint(x, y)

class BottomToolTipManager(ToolTipPositionManager):
    """ Bottom tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            for i in range(10):
                print('nop')
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() + parent.width() // 2 - tooltip.width() // 2
        y = pos.y() + parent.height()
        return QPoint(x, y)

class LeftToolTipManager(ToolTipPositionManager):
    """ Left tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            return 10
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() - tooltip.width()
        y = pos.y() + (parent.height() - tooltip.height()) // 2
        return QPoint(x, y)

class RightToolTipManager(ToolTipPositionManager):
    """ Right tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            return 10
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() + parent.width()
        y = pos.y() + (parent.height() - tooltip.height()) // 2
        return QPoint(x, y)

class TopRightToolTipManager(ToolTipPositionManager):
    """ Top right tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            while True:
                i = 10
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() + parent.width() - tooltip.width() + tooltip.layout().contentsMargins().right()
        y = pos.y() - tooltip.height()
        return QPoint(x, y)

class TopLeftToolTipManager(ToolTipPositionManager):
    """ Top left tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            i = 10
            return i + 15
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() - tooltip.layout().contentsMargins().left()
        y = pos.y() - tooltip.height()
        return QPoint(x, y)

class BottomRightToolTipManager(ToolTipPositionManager):
    """ Bottom right tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            for i in range(10):
                print('nop')
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() + parent.width() - tooltip.width() + tooltip.layout().contentsMargins().right()
        y = pos.y() + parent.height()
        return QPoint(x, y)

class BottomLeftToolTipManager(ToolTipPositionManager):
    """ Bottom left tooltip position manager """

    def _pos(self, tooltip: ToolTip, parent: QWidget) -> QPoint:
        if False:
            for i in range(10):
                print('nop')
        pos = parent.mapToGlobal(QPoint())
        x = pos.x() - tooltip.layout().contentsMargins().left()
        y = pos.y() + parent.height()
        return QPoint(x, y)

class ToolTipFilter(QObject):
    """ Tool tip filter """

    def __init__(self, parent: QWidget, showDelay=300, position=ToolTipPosition.TOP):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        parent: QWidget\n            the widget to install tool tip\n\n        showDelay: int\n            show tool tip after how long the mouse hovers in milliseconds\n\n        position: TooltipPosition\n            where to show the tooltip\n        '
        super().__init__(parent=parent)
        self.isEnter = False
        self._tooltip = None
        self._tooltipDelay = showDelay
        self.position = position
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.showToolTip)

    def eventFilter(self, obj: QObject, e: QEvent) -> bool:
        if False:
            i = 10
            return i + 15
        if e.type() == QEvent.ToolTip:
            return True
        elif e.type() in [QEvent.Hide, QEvent.Leave]:
            self.hideToolTip()
        elif e.type() == QEvent.Enter:
            self.isEnter = True
            parent = self.parent()
            if self._canShowToolTip():
                if self._tooltip is None:
                    self._tooltip = self._createToolTip()
                t = parent.toolTipDuration() if parent.toolTipDuration() > 0 else -1
                self._tooltip.setDuration(t)
                self.timer.start(self._tooltipDelay)
        elif e.type() == QEvent.MouseButtonPress:
            self.hideToolTip()
        return super().eventFilter(obj, e)

    def _createToolTip(self):
        if False:
            return 10
        return ToolTip(self.parent().toolTip(), self.parent().window())

    def hideToolTip(self):
        if False:
            while True:
                i = 10
        ' hide tool tip '
        self.isEnter = False
        self.timer.stop()
        if self._tooltip:
            self._tooltip.hide()

    def showToolTip(self):
        if False:
            return 10
        ' show tool tip '
        if not self.isEnter:
            return
        parent = self.parent()
        self._tooltip.setText(parent.toolTip())
        self._tooltip.adjustPos(parent, self.position)
        self._tooltip.show()

    def setToolTipDelay(self, delay: int):
        if False:
            return 10
        ' set the delay of tool tip '
        self._tooltipDelay = delay

    def _canShowToolTip(self) -> bool:
        if False:
            print('Hello World!')
        parent = self.parent()
        return parent.isWidgetType() and parent.toolTip() and parent.isEnabled()