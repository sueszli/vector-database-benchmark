from PyQt5.QtWidgets import QPushButton
from PyQt5.QtGui import QPainter, QPolygonF, QColor, QTextOption
from PyQt5.QtCore import QPointF, QRectF, Qt
from feeluown.player import PlaylistMode
from feeluown.gui.helpers import SOLARIZED_COLORS

class PlaylistButton(QPushButton):

    def __init__(self, app, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent=parent)
        self._app = app
        self._app.playlist.mode_changed.connect(lambda *args: self.update(), weak=False)

    def paintEvent(self, e):
        if False:
            return 10
        width = height = 18
        painter = QPainter(self)
        painter.save()
        painter.setRenderHints(QPainter.Antialiasing)
        painter.translate((self.width() - width) // 2, (self.height() - height) // 2)
        line_height = 1.3
        line_margin = (height - 3 * line_height) / 3
        h1 = line_margin
        h2 = h1 + line_margin + line_height
        h3 = h2 + line_margin + line_height
        pen = painter.pen()
        pen.setColor(QColor('#6F6F6F'))
        pen.setWidthF(line_height)
        painter.setPen(pen)
        painter.setBrush(pen.color())
        triangle_side_length_half = line_margin * 0.6
        triangle_height = triangle_side_length_half * 1.7
        triangle = QPolygonF([QPointF(0, h1 - triangle_side_length_half), QPointF(triangle_height, h1), QPointF(0, h1 + triangle_side_length_half)])
        painter.drawPolygon(triangle)
        painter.drawLine(QPointF(triangle_height + triangle_side_length_half, h1), QPointF(width, h1))
        painter.drawLine(QPointF(0, h2), QPointF(width, h2))
        if self._app.playlist.mode is PlaylistMode.fm:
            painter.drawLine(QPointF(0, h3), QPointF(width // 2, h3))
            painter.pen()
            pen.setColor(QColor(SOLARIZED_COLORS['blue']))
            painter.setPen(pen)
            font = painter.font()
            rect_h_half = line_margin // 2
            font.setPixelSize(int(rect_h_half * 2))
            painter.setFont(font)
            rect = QRectF(width // 2 + rect_h_half, h3 - rect_h_half, width // 2 - rect_h_half, rect_h_half * 2)
            painter.drawText(rect, 'FM', QTextOption(Qt.AlignCenter))
        else:
            painter.drawLine(QPointF(0, h3), QPointF(width, h3))
        painter.restore()