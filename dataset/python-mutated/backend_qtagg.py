"""
Render to qt from agg.
"""
import ctypes
from matplotlib.transforms import Bbox
from .qt_compat import QT_API, QtCore, QtGui
from .backend_agg import FigureCanvasAgg
from .backend_qt import _BackendQT, FigureCanvasQT
from .backend_qt import FigureManagerQT, NavigationToolbar2QT

class FigureCanvasQTAgg(FigureCanvasAgg, FigureCanvasQT):

    def paintEvent(self, event):
        if False:
            while True:
                i = 10
        '\n        Copy the image from the Agg canvas to the qt.drawable.\n\n        In Qt, all drawing should be done inside of here when a widget is\n        shown onscreen.\n        '
        self._draw_idle()
        if not hasattr(self, 'renderer'):
            return
        painter = QtGui.QPainter(self)
        try:
            rect = event.rect()
            width = rect.width() * self.device_pixel_ratio
            height = rect.height() * self.device_pixel_ratio
            (left, top) = self.mouseEventCoords(rect.topLeft())
            bottom = top - height
            right = left + width
            bbox = Bbox([[left, bottom], [right, top]])
            buf = memoryview(self.copy_from_bbox(bbox))
            if QT_API == 'PyQt6':
                from PyQt6 import sip
                ptr = int(sip.voidptr(buf))
            else:
                ptr = buf
            painter.eraseRect(rect)
            qimage = QtGui.QImage(ptr, buf.shape[1], buf.shape[0], QtGui.QImage.Format.Format_RGBA8888)
            qimage.setDevicePixelRatio(self.device_pixel_ratio)
            origin = QtCore.QPoint(rect.left(), rect.top())
            painter.drawImage(origin, qimage)
            if QT_API == 'PySide2' and QtCore.__version_info__ < (5, 12):
                ctypes.c_long.from_address(id(buf)).value = 1
            self._draw_rect_callback(painter)
        finally:
            painter.end()

    def print_figure(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().print_figure(*args, **kwargs)
        self._draw_pending = True

@_BackendQT.export
class _BackendQTAgg(_BackendQT):
    FigureCanvas = FigureCanvasQTAgg