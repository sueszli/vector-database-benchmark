from typing import List
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPaintEvent, QPen, QPainterPath, QColor, QTransform
from PyQt5.QtCore import QPoint, QSize, QRect, QRectF, Qt
from electrum.qrreader import QrCodeResult
from .validator import QrReaderValidatorResult

class QrReaderVideoOverlay(QWidget):
    """
    Overlays the QR scanner results over the video
    """
    BG_RECT_PADDING = 10
    BG_RECT_CORNER_RADIUS = 10.0
    BG_RECT_OPACITY = 0.75

    def __init__(self, parent: QWidget=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.results = []
        self.flip_x = False
        self.validator_results = None
        self.crop = None
        self.resolution = None
        self.qr_outline_pen = QPen()
        self.qr_outline_pen.setColor(Qt.red)
        self.qr_outline_pen.setWidth(3)
        self.qr_outline_pen.setStyle(Qt.DotLine)
        self.text_pen = QPen()
        self.text_pen.setColor(Qt.black)
        self.bg_rect_pen = QPen()
        self.bg_rect_pen.setColor(Qt.black)
        self.bg_rect_pen.setStyle(Qt.DotLine)
        self.bg_rect_fill = QColor(255, 255, 255, int(255 * self.BG_RECT_OPACITY))

    def set_results(self, results: List[QrCodeResult], flip_x: bool, validator_results: QrReaderValidatorResult):
        if False:
            for i in range(10):
                print('nop')
        self.results = results
        self.flip_x = flip_x
        self.validator_results = validator_results
        self.update()

    def set_crop(self, crop: QRect):
        if False:
            return 10
        self.crop = crop

    def set_resolution(self, resolution: QSize):
        if False:
            for i in range(10):
                print('nop')
        self.resolution = resolution

    def paintEvent(self, _event: QPaintEvent):
        if False:
            return 10
        if not self.crop or not self.resolution:
            return
        painter = QPainter(self)
        transform = painter.worldTransform()
        transform = transform.scale(self.width() / self.resolution.width(), self.height() / self.resolution.height())
        transform_flip = QTransform()
        if self.flip_x:
            transform_flip = transform_flip.translate(self.resolution.width(), 0.0)
            transform_flip = transform_flip.scale(-1.0, 1.0)

        def toqp(point):
            if False:
                while True:
                    i = 10
            return QPoint(point[0], point[1])
        painter.setRenderHint(QPainter.Antialiasing)
        for res in self.results:
            painter.setWorldTransform(transform_flip * transform, False)
            pen = QPen(self.qr_outline_pen)
            if res in self.validator_results.result_colors:
                pen.setColor(self.validator_results.result_colors[res])
            painter.setPen(pen)
            num_points = len(res.points)
            for i in range(0, num_points):
                i_n = i + 1
                line_from = toqp(res.points[i])
                line_from += self.crop.topLeft()
                line_to = toqp(res.points[i_n] if i_n < num_points else res.points[0])
                line_to += self.crop.topLeft()
                painter.drawLine(line_from, line_to)
            painter.setWorldTransform(transform, False)
            font_metrics = painter.fontMetrics()
            data_metrics = QSize(font_metrics.horizontalAdvance(res.data), font_metrics.capHeight())
            center_pos = toqp(res.center)
            center_pos += self.crop.topLeft()
            center_pos = transform_flip.map(center_pos)
            text_offset = QPoint(data_metrics.width(), data_metrics.height())
            text_offset = text_offset / 2
            text_offset.setX(-text_offset.x())
            center_pos += text_offset
            padding = self.BG_RECT_PADDING
            bg_rect_pos = center_pos - QPoint(padding, data_metrics.height() + padding)
            bg_rect_size = data_metrics + QSize(padding, padding) * 2
            bg_rect = QRect(bg_rect_pos, bg_rect_size)
            bg_rect_path = QPainterPath()
            radius = self.BG_RECT_CORNER_RADIUS
            bg_rect_path.addRoundedRect(QRectF(bg_rect), radius, radius, Qt.AbsoluteSize)
            painter.setPen(self.bg_rect_pen)
            painter.fillPath(bg_rect_path, self.bg_rect_fill)
            painter.drawPath(bg_rect_path)
            painter.setPen(self.text_pen)
            painter.drawText(center_pos, res.data)