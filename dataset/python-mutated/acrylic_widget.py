from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainterPath, QPainter, QColor
from PyQt5.QtWidgets import QWidget
from ..widgets.flyout import FlyoutViewBase
from ..widgets.acrylic_label import AcrylicBrush
from ...common.style_sheet import isDarkTheme

class AcrylicWidget:
    """ Acrylic widget """

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(*args, **kwargs)
        self.acrylicBrush = AcrylicBrush(self, 30)

    def _updateAcrylicColor(self):
        if False:
            for i in range(10):
                print('nop')
        if isDarkTheme():
            tintColor = QColor(32, 32, 32, 200)
            luminosityColor = QColor(0, 0, 0, 0)
        else:
            tintColor = QColor(255, 255, 255, 180)
            luminosityColor = QColor(255, 255, 255, 0)
        self.acrylicBrush.tintColor = tintColor
        self.acrylicBrush.luminosityColor = luminosityColor

    def acrylicClipPath(self):
        if False:
            print('Hello World!')
        return QPainterPath()

    def _drawAcrylic(self, painter: QPainter):
        if False:
            print('Hello World!')
        path = self.acrylicClipPath()
        if not path.isEmpty():
            self.acrylicBrush.clipPath = self.acrylicClipPath()
        self._updateAcrylicColor()
        self.acrylicBrush.paint()

    def paintEvent(self, e):
        if False:
            return 10
        painter = QPainter(self)
        painter.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self._drawAcrylic(painter)
        super().paintEvent(e)