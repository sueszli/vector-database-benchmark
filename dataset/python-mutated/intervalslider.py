from typing import Tuple
from AnyQt.QtCore import pyqtSignal as Signal
from AnyQt.QtWidgets import QWidget, QStyleOptionSlider, QSizePolicy, QStyle, QSlider
from AnyQt.QtGui import QPainter, QMouseEvent, QPalette, QBrush
from AnyQt.QtCore import QRect, Qt, QSize

class IntervalSlider(QWidget):
    """
    Slider with two handles for setting an interval of values.

    Only horizontal orientation is supported.

    Signals:

    rangeChanged(minimum: int, maximum: int):
        minumum or maximum or both have changed

    intervalChanged(low: int, high: int):
        One or both boundaries have been changed.The tracking determines
        whether this signal is emitted during user interaction or only when
        the mouse is released

    sliderPressed(id: int)
        The user started dragging.
        Id is IntervarlSlider.LowHandle or IntervalSlider.HighHandle

    sliderMoved(id: int, value: int)
        the user drags the slider
        Id is IntervarlSlider.LowHandle or IntervalSlider.HighHandle

    sliderReleased(id: int)
        The user finished dragging.
        Id is IntervarlSlider.LowHandle or IntervalSlider.HighHandle

    """
    (NoHandle, LowHandle, HighHandle) = (0, 1, 2)
    rangeChanged = Signal((int, int))
    intervalChanged = Signal((int, int))
    sliderPressed = Signal(int)
    sliderMoved = Signal(int, int)
    sliderReleased = Signal(int)

    def __init__(self, low=1, high=8, minimum=0, maximum=10, parent: QWidget=None, **args):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed, QSizePolicy.Slider))
        self._dragged = self.NoHandle
        self._start_drag_value = None
        self.opt = QStyleOptionSlider()
        self._interval = self._pos = (low, high)
        self.opt.minimum = minimum
        self.opt.maximum = maximum
        self._tracking = True
        for (opt, value) in args.items():
            getattr(self, f'set{opt[0].upper()}{opt[1:]}')(value)

    def orientation(self):
        if False:
            i = 10
            return i + 15
        'Orientation. Always Qt.Horizontal'
        return Qt.Horizontal

    def setOrientation(self, orientation):
        if False:
            print('Hello World!')
        'Set orientation (to Qt.Horizontal)'
        if orientation != Qt.Horizontal:
            raise ValueError('IntervalSlider supports only horizontal direction')

    def sliderPosition(self) -> Tuple[int, int]:
        if False:
            return 10
        '\n        Current position of sliders.\n        This is the same as `interval` if tracking is enabled.\n        '
        return self._pos

    def setSliderPosition(self, low: int, high: int) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set position of sliders.\n\n        This also changes `interval` if tracking is enabled.\n        '
        self._pos = (low, high)
        self.update()
        if self._tracking:
            self._interval = self._pos
            self.intervalChanged.emit(low, high)
        self.sliderMoved.emit(self.LowHandle, low)
        self.sliderMoved.emit(self.HighHandle, high)

    def interval(self) -> Tuple[int, int]:
        if False:
            print('Hello World!')
        'Current interval'
        return self._interval

    def setInterval(self, low: int, high: int) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Set the current interval'
        self._interval = (low, high)
        self.setSliderPosition(low, high)

    def low(self) -> int:
        if False:
            return 10
        'The lower bound of the interavl'
        return self._interval[0]

    def setLow(self, low: int) -> None:
        if False:
            print('Hello World!')
        'Set the lower bound'
        self.setInterval(low, self.high())

    def high(self) -> int:
        if False:
            while True:
                i = 10
        'The higher bound of the interval'
        return self._interval[1]

    def setHigh(self, high: int) -> None:
        if False:
            i = 10
            return i + 15
        'Set the higher bound of the interval'
        self.setInterval(self.low(), high)

    def setMinimum(self, minimum: int) -> None:
        if False:
            i = 10
            return i + 15
        'Set the minimal value of the lower bound'
        self.opt.minimum = minimum
        self.rangeChanged.emit(minimum, self.opt.maximum)

    def setMaximum(self, maximum: int) -> None:
        if False:
            return 10
        'Set the maximal value of the higher bound'
        self.opt.maximum = maximum
        self.rangeChanged.emit(self.opt.minimum, maximum)

    def setTickPosition(self, position: QSlider.TickPosition) -> None:
        if False:
            i = 10
            return i + 15
        'See QSlider.setTickPosition'
        self.opt.tickPosition = position

    def setTickInterval(self, ti: int) -> None:
        if False:
            while True:
                i = 10
        'See QSlider.setInterval'
        self.opt.tickInterval = ti

    def setTracking(self, enabled: bool) -> None:
        if False:
            while True:
                i = 10
        '\n        If enabled, interval is changed during user interaction and the\n        notifier signal is emitted immediately.\n        '
        self._tracking = enabled

    def hasTracking(self) -> bool:
        if False:
            i = 10
            return i + 15
        return self._tracking

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if False:
            while True:
                i = 10
        self._dragged = self.NoHandle
        args = (QStyle.CC_Slider, self.opt, event.pos(), self)
        self.opt.sliderPosition = self._pos[0]
        if self.style().hitTestComplexControl(*args) == QStyle.SC_SliderHandle:
            self._dragged |= self.LowHandle
        self.opt.sliderPosition = self._pos[1]
        if self.style().hitTestComplexControl(*args) == QStyle.SC_SliderHandle:
            self._dragged |= self.HighHandle
        if self._dragged != self.NoHandle:
            self.sliderPressed.emit(self._dragged)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if False:
            return 10
        distance = self.opt.maximum - self.opt.minimum
        pos = self.style().sliderValueFromPosition(0, distance, event.pos().x(), self.rect().width())
        (low, high) = self._pos
        if self._dragged == self.LowHandle | self.HighHandle:
            if pos < high:
                self._dragged = self.LowHandle
            elif pos > low:
                self._dragged = self.HighHandle
        if self._dragged == self.LowHandle:
            low = min(pos, high)
        elif self._dragged == self.HighHandle:
            high = max(pos, low)
        if self._pos == (low, high):
            return
        self._pos = (low, high)
        self.update()
        if self._tracking:
            self._interval = self._pos
            self.intervalChanged.emit(*self._interval)

    def mouseReleaseEvent(self, _) -> None:
        if False:
            return 10
        if self._dragged == self.NoHandle:
            return
        self.sliderReleased.emit(self._dragged)
        if self._interval != self._pos:
            self._interval = self._pos
            self.intervalChanged.emit(*self._interval)

    def paintEvent(self, _) -> None:
        if False:
            i = 10
            return i + 15
        painter = QPainter(self)
        self.opt.initFrom(self)
        self.opt.rect = self.rect()
        self.opt.sliderPosition = 0
        self.opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderTickmarks
        self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)
        color = self.palette().color(QPalette.Highlight)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.NoPen)
        args = (QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
        self.opt.sliderPosition = self._pos[0]
        x_left_handle = self.style().subControlRect(*args).right()
        self.opt.sliderPosition = self._pos[1]
        x_right_handle = self.style().subControlRect(*args).left()
        groove_rect = self.style().subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderGroove, self)
        selection = QRect(x_left_handle, groove_rect.y() + groove_rect.height() // 2 - 2, x_right_handle - x_left_handle, 4)
        painter.drawRect(selection)
        self.opt.subControls = QStyle.SC_SliderHandle
        for self.opt.sliderPosition in self._pos:
            self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

    def sizeHint(self) -> QSize:
        if False:
            print('Hello World!')
        SliderLength = 84
        TickSpace = 5
        w = SliderLength
        h = self.style().pixelMetric(QStyle.PM_SliderThickness, self.opt, self)
        if self.opt.tickPosition != QSlider.NoTicks:
            h += TickSpace
        return self.style().sizeFromContents(QStyle.CT_Slider, self.opt, QSize(w, h))