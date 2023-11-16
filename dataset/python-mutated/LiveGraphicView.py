from PyQt5.QtCore import pyqtSignal, QEvent
from PyQt5.QtGui import QWheelEvent, QMouseEvent
from urh.ui.painting.GridScene import GridScene
from urh.ui.views.ZoomableGraphicView import ZoomableGraphicView

class LiveGraphicView(ZoomableGraphicView):
    freq_clicked = pyqtSignal(float)
    wheel_event_triggered = pyqtSignal(QWheelEvent)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.capturing_data = True
        self.setMouseTracking(True)

    def wheelEvent(self, event: QWheelEvent):
        if False:
            print('Hello World!')
        self.wheel_event_triggered.emit(event)
        if self.capturing_data:
            return
        super().wheelEvent(event)

    def leaveEvent(self, event: QEvent):
        if False:
            print('Hello World!')
        super().leaveEvent(event)
        if isinstance(self.scene(), GridScene):
            self.scene().clear_frequency_marker()

    def mouseMoveEvent(self, event: QMouseEvent):
        if False:
            i = 10
            return i + 15
        super().mouseMoveEvent(event)
        if isinstance(self.scene(), GridScene):
            x = int(self.mapToScene(event.pos()).x())
            freq = self.scene().get_freq_for_pos(x)
            self.scene().draw_frequency_marker(x, freq)

    def mousePressEvent(self, event: QMouseEvent):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.scene(), GridScene):
            freq = self.scene().get_freq_for_pos(int(self.mapToScene(event.pos()).x()))
            if freq is not None:
                self.freq_clicked.emit(freq)

    def update(self, *__args):
        if False:
            while True:
                i = 10
        try:
            super().update(*__args)
            super().show_full_scene()
        except RuntimeError:
            pass