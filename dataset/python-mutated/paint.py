from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5 import QtGui, QtWidgets, QtCore
from PyQt5.QtGui import QPainter, QBitmap, QPolygon, QPen, QBrush, QColor
from PyQt5.QtCore import Qt
from MainWindow import Ui_MainWindow
import os
import sys
import random
import types
try:
    from PyQt5.QtWinExtras import QtWin
    myappid = 'com.learnpyqt.minute-apps.paint'
    QtWin.setCurrentProcessExplicitAppUserModelID(myappid)
except ImportError:
    pass
BRUSH_MULT = 3
SPRAY_PAINT_MULT = 5
SPRAY_PAINT_N = 100
COLORS = ['#000000', '#82817f', '#820300', '#868417', '#007e03', '#037e7b', '#040079', '#81067a', '#7f7e45', '#05403c', '#0a7cf6', '#093c7e', '#7e07f9', '#7c4002', '#ffffff', '#c1c1c1', '#f70406', '#fffd00', '#08fb01', '#0bf8ee', '#0000fa', '#b92fc2', '#fffc91', '#00fd83', '#87f9f9', '#8481c4', '#dc137d', '#fb803c']
FONT_SIZES = [7, 8, 9, 10, 11, 12, 13, 14, 18, 24, 36, 48, 64, 72, 96, 144, 288]
MODES = ['selectpoly', 'selectrect', 'eraser', 'fill', 'dropper', 'stamp', 'pen', 'brush', 'spray', 'text', 'line', 'polyline', 'rect', 'polygon', 'ellipse', 'roundrect']
CANVAS_DIMENSIONS = (600, 400)
STAMPS = [':/stamps/pie-apple.png', ':/stamps/pie-cherry.png', ':/stamps/pie-cherry2.png', ':/stamps/pie-lemon.png', ':/stamps/pie-moon.png', ':/stamps/pie-pork.png', ':/stamps/pie-pumpkin.png', ':/stamps/pie-walnut.png']
SELECTION_PEN = QPen(QColor(255, 255, 255), 1, Qt.DashLine)
PREVIEW_PEN = QPen(QColor(255, 255, 255), 1, Qt.SolidLine)

def build_font(config):
    if False:
        while True:
            i = 10
    '\n    Construct a complete font from the configuration options\n    :param self:\n    :param config:\n    :return: QFont\n    '
    font = config['font']
    font.setPointSize(config['fontsize'])
    font.setBold(config['bold'])
    font.setItalic(config['italic'])
    font.setUnderline(config['underline'])
    return font

class Canvas(QLabel):
    mode = 'rectangle'
    primary_color = QColor(Qt.black)
    secondary_color = None
    primary_color_updated = pyqtSignal(str)
    secondary_color_updated = pyqtSignal(str)
    config = {'size': 1, 'fill': True, 'font': QFont('Times'), 'fontsize': 12, 'bold': False, 'italic': False, 'underline': False}
    active_color = None
    preview_pen = None
    timer_event = None
    current_stamp = None

    def initialize(self):
        if False:
            while True:
                i = 10
        self.background_color = QColor(self.secondary_color) if self.secondary_color else QColor(Qt.white)
        self.eraser_color = QColor(self.secondary_color) if self.secondary_color else QColor(Qt.white)
        self.eraser_color.setAlpha(100)
        self.reset()

    def reset(self):
        if False:
            print('Hello World!')
        self.setPixmap(QPixmap(*CANVAS_DIMENSIONS))
        self.pixmap().fill(self.background_color)

    def set_primary_color(self, hex):
        if False:
            print('Hello World!')
        self.primary_color = QColor(hex)

    def set_secondary_color(self, hex):
        if False:
            print('Hello World!')
        self.secondary_color = QColor(hex)

    def set_config(self, key, value):
        if False:
            i = 10
            return i + 15
        self.config[key] = value

    def set_mode(self, mode):
        if False:
            i = 10
            return i + 15
        self.timer_cleanup()
        self.active_shape_fn = None
        self.active_shape_args = ()
        self.origin_pos = None
        self.current_pos = None
        self.last_pos = None
        self.history_pos = None
        self.last_history = []
        self.current_text = ''
        self.last_text = ''
        self.last_config = {}
        self.dash_offset = 0
        self.locked = False
        self.mode = mode

    def reset_mode(self):
        if False:
            i = 10
            return i + 15
        self.set_mode(self.mode)

    def on_timer(self):
        if False:
            for i in range(10):
                print('nop')
        if self.timer_event:
            self.timer_event()

    def timer_cleanup(self):
        if False:
            i = 10
            return i + 15
        if self.timer_event:
            timer_event = self.timer_event
            self.timer_event = None
            timer_event(final=True)

    def mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        fn = getattr(self, '%s_mousePressEvent' % self.mode, None)
        if fn:
            return fn(e)

    def mouseMoveEvent(self, e):
        if False:
            i = 10
            return i + 15
        fn = getattr(self, '%s_mouseMoveEvent' % self.mode, None)
        if fn:
            return fn(e)

    def mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        fn = getattr(self, '%s_mouseReleaseEvent' % self.mode, None)
        if fn:
            return fn(e)

    def mouseDoubleClickEvent(self, e):
        if False:
            i = 10
            return i + 15
        fn = getattr(self, '%s_mouseDoubleClickEvent' % self.mode, None)
        if fn:
            return fn(e)

    def generic_mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.last_pos = e.pos()
        if e.button() == Qt.LeftButton:
            self.active_color = self.primary_color
        else:
            self.active_color = self.secondary_color

    def generic_mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.last_pos = None

    def selectpoly_mousePressEvent(self, e):
        if False:
            print('Hello World!')
        if not self.locked or e.button == Qt.RightButton:
            self.active_shape_fn = 'drawPolygon'
            self.preview_pen = SELECTION_PEN
            self.generic_poly_mousePressEvent(e)

    def selectpoly_timerEvent(self, final=False):
        if False:
            return 10
        self.generic_poly_timerEvent(final)

    def selectpoly_mouseMoveEvent(self, e):
        if False:
            print('Hello World!')
        if not self.locked:
            self.generic_poly_mouseMoveEvent(e)

    def selectpoly_mouseDoubleClickEvent(self, e):
        if False:
            print('Hello World!')
        self.current_pos = e.pos()
        self.locked = True

    def selectpoly_copy(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Copy a polygon region from the current image, returning it.\n\n        Create a mask for the selected area, and use it to blank\n        out non-selected regions. Then get the bounding rect of the\n        selection and crop to produce the smallest possible image.\n\n        :return: QPixmap of the copied region.\n        '
        self.timer_cleanup()
        pixmap = self.pixmap().copy()
        bitmap = QBitmap(*CANVAS_DIMENSIONS)
        bitmap.clear()
        p = QPainter(bitmap)
        userpoly = QPolygon(self.history_pos + [self.current_pos])
        p.setPen(QPen(Qt.color1))
        p.setBrush(QBrush(Qt.color1))
        p.drawPolygon(userpoly)
        p.end()
        pixmap.setMask(bitmap)
        return pixmap.copy(userpoly.boundingRect())

    def selectrect_mousePressEvent(self, e):
        if False:
            print('Hello World!')
        self.active_shape_fn = 'drawRect'
        self.preview_pen = SELECTION_PEN
        self.generic_shape_mousePressEvent(e)

    def selectrect_timerEvent(self, final=False):
        if False:
            while True:
                i = 10
        self.generic_shape_timerEvent(final)

    def selectrect_mouseMoveEvent(self, e):
        if False:
            print('Hello World!')
        if not self.locked:
            self.current_pos = e.pos()

    def selectrect_mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.current_pos = e.pos()
        self.locked = True

    def selectrect_copy(self):
        if False:
            return 10
        '\n        Copy a rectangle region of the current image, returning it.\n\n        :return: QPixmap of the copied region.\n        '
        self.timer_cleanup()
        return self.pixmap().copy(QRect(self.origin_pos, self.current_pos))

    def eraser_mousePressEvent(self, e):
        if False:
            while True:
                i = 10
        self.generic_mousePressEvent(e)

    def eraser_mouseMoveEvent(self, e):
        if False:
            print('Hello World!')
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.eraser_color, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())
            self.last_pos = e.pos()
            self.update()

    def eraser_mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.generic_mouseReleaseEvent(e)

    def stamp_mousePressEvent(self, e):
        if False:
            while True:
                i = 10
        p = QPainter(self.pixmap())
        stamp = self.current_stamp
        p.drawPixmap(e.x() - stamp.width() // 2, e.y() - stamp.height() // 2, stamp)
        self.update()

    def pen_mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.generic_mousePressEvent(e)

    def pen_mouseMoveEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.active_color, self.config['size'], Qt.SolidLine, Qt.SquareCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())
            self.last_pos = e.pos()
            self.update()

    def pen_mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.generic_mouseReleaseEvent(e)

    def brush_mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.generic_mousePressEvent(e)

    def brush_mouseMoveEvent(self, e):
        if False:
            return 10
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.active_color, self.config['size'] * BRUSH_MULT, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(self.last_pos, e.pos())
            self.last_pos = e.pos()
            self.update()

    def brush_mouseReleaseEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.generic_mouseReleaseEvent(e)

    def spray_mousePressEvent(self, e):
        if False:
            return 10
        self.generic_mousePressEvent(e)

    def spray_mouseMoveEvent(self, e):
        if False:
            return 10
        if self.last_pos:
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.active_color, 1))
            for n in range(self.config['size'] * SPRAY_PAINT_N):
                xo = random.gauss(0, self.config['size'] * SPRAY_PAINT_MULT)
                yo = random.gauss(0, self.config['size'] * SPRAY_PAINT_MULT)
                p.drawPoint(e.x() + xo, e.y() + yo)
        self.update()

    def spray_mouseReleaseEvent(self, e):
        if False:
            return 10
        self.generic_mouseReleaseEvent(e)

    def keyPressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        if self.mode == 'text':
            if e.key() == Qt.Key_Backspace:
                self.current_text = self.current_text[:-1]
            else:
                self.current_text = self.current_text + e.text()

    def text_mousePressEvent(self, e):
        if False:
            print('Hello World!')
        if e.button() == Qt.LeftButton and self.current_pos is None:
            self.current_pos = e.pos()
            self.current_text = ''
            self.timer_event = self.text_timerEvent
        elif e.button() == Qt.LeftButton:
            self.timer_cleanup()
            p = QPainter(self.pixmap())
            p.setRenderHints(QPainter.Antialiasing)
            font = build_font(self.config)
            p.setFont(font)
            pen = QPen(self.primary_color, 1, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            p.setPen(pen)
            p.drawText(self.current_pos, self.current_text)
            self.update()
            self.reset_mode()
        elif e.button() == Qt.RightButton and self.current_pos:
            self.reset_mode()

    def text_timerEvent(self, final=False):
        if False:
            for i in range(10):
                print('nop')
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = PREVIEW_PEN
        p.setPen(pen)
        if self.last_text:
            font = build_font(self.last_config)
            p.setFont(font)
            p.drawText(self.current_pos, self.last_text)
        if not final:
            font = build_font(self.config)
            p.setFont(font)
            p.drawText(self.current_pos, self.current_text)
        self.last_text = self.current_text
        self.last_config = self.config.copy()
        self.update()

    def fill_mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        if e.button() == Qt.LeftButton:
            self.active_color = self.primary_color
        else:
            self.active_color = self.secondary_color
        image = self.pixmap().toImage()
        (w, h) = (image.width(), image.height())
        (x, y) = (e.x(), e.y())
        target_color = image.pixel(x, y)
        have_seen = set()
        queue = [(x, y)]

        def get_cardinal_points(have_seen, center_pos):
            if False:
                return 10
            points = []
            (cx, cy) = center_pos
            for (x, y) in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
                (xx, yy) = (cx + x, cy + y)
                if xx >= 0 and xx < w and (yy >= 0) and (yy < h) and ((xx, yy) not in have_seen):
                    points.append((xx, yy))
                    have_seen.add((xx, yy))
            return points
        p = QPainter(self.pixmap())
        p.setPen(QPen(self.active_color))
        while queue:
            (x, y) = queue.pop()
            if image.pixel(x, y) == target_color:
                p.drawPoint(QPoint(x, y))
                queue.extend(get_cardinal_points(have_seen, (x, y)))
        self.update()

    def dropper_mousePressEvent(self, e):
        if False:
            while True:
                i = 10
        c = self.pixmap().toImage().pixel(e.pos())
        hex = QColor(c).name()
        if e.button() == Qt.LeftButton:
            self.set_primary_color(hex)
            self.primary_color_updated.emit(hex)
        elif e.button() == Qt.RightButton:
            self.set_secondary_color(hex)
            self.secondary_color_updated.emit(hex)

    def generic_shape_mousePressEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.origin_pos = e.pos()
        self.current_pos = e.pos()
        self.timer_event = self.generic_shape_timerEvent

    def generic_shape_timerEvent(self, final=False):
        if False:
            print('Hello World!')
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = self.preview_pen
        pen.setDashOffset(self.dash_offset)
        p.setPen(pen)
        if self.last_pos:
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, self.last_pos), *self.active_shape_args)
        if not final:
            self.dash_offset -= 1
            pen.setDashOffset(self.dash_offset)
            p.setPen(pen)
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, self.current_pos), *self.active_shape_args)
        self.update()
        self.last_pos = self.current_pos

    def generic_shape_mouseMoveEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.current_pos = e.pos()

    def generic_shape_mouseReleaseEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        if self.last_pos:
            self.timer_cleanup()
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.primary_color, self.config['size'], Qt.SolidLine, Qt.SquareCap, Qt.MiterJoin))
            if self.config['fill']:
                p.setBrush(QBrush(self.secondary_color))
            getattr(p, self.active_shape_fn)(QRect(self.origin_pos, e.pos()), *self.active_shape_args)
            self.update()
        self.reset_mode()

    def line_mousePressEvent(self, e):
        if False:
            while True:
                i = 10
        self.origin_pos = e.pos()
        self.current_pos = e.pos()
        self.preview_pen = PREVIEW_PEN
        self.timer_event = self.line_timerEvent

    def line_timerEvent(self, final=False):
        if False:
            i = 10
            return i + 15
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = self.preview_pen
        p.setPen(pen)
        if self.last_pos:
            p.drawLine(self.origin_pos, self.last_pos)
        if not final:
            p.drawLine(self.origin_pos, self.current_pos)
        self.update()
        self.last_pos = self.current_pos

    def line_mouseMoveEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.current_pos = e.pos()

    def line_mouseReleaseEvent(self, e):
        if False:
            print('Hello World!')
        if self.last_pos:
            self.timer_cleanup()
            p = QPainter(self.pixmap())
            p.setPen(QPen(self.primary_color, self.config['size'], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            p.drawLine(self.origin_pos, e.pos())
            self.update()
        self.reset_mode()

    def generic_poly_mousePressEvent(self, e):
        if False:
            print('Hello World!')
        if e.button() == Qt.LeftButton:
            if self.history_pos:
                self.history_pos.append(e.pos())
            else:
                self.history_pos = [e.pos()]
                self.current_pos = e.pos()
                self.timer_event = self.generic_poly_timerEvent
        elif e.button() == Qt.RightButton and self.history_pos:
            self.timer_cleanup()
            self.reset_mode()

    def generic_poly_timerEvent(self, final=False):
        if False:
            for i in range(10):
                print('nop')
        p = QPainter(self.pixmap())
        p.setCompositionMode(QPainter.RasterOp_SourceXorDestination)
        pen = self.preview_pen
        pen.setDashOffset(self.dash_offset)
        p.setPen(pen)
        if self.last_history:
            getattr(p, self.active_shape_fn)(*self.last_history)
        if not final:
            self.dash_offset -= 1
            pen.setDashOffset(self.dash_offset)
            p.setPen(pen)
            getattr(p, self.active_shape_fn)(*self.history_pos + [self.current_pos])
        self.update()
        self.last_pos = self.current_pos
        self.last_history = self.history_pos + [self.current_pos]

    def generic_poly_mouseMoveEvent(self, e):
        if False:
            return 10
        self.current_pos = e.pos()

    def generic_poly_mouseDoubleClickEvent(self, e):
        if False:
            print('Hello World!')
        self.timer_cleanup()
        p = QPainter(self.pixmap())
        p.setPen(QPen(self.primary_color, self.config['size'], Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        if self.secondary_color:
            p.setBrush(QBrush(self.secondary_color))
        getattr(p, self.active_shape_fn)(*self.history_pos + [e.pos()])
        self.update()
        self.reset_mode()

    def polyline_mousePressEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.active_shape_fn = 'drawPolyline'
        self.preview_pen = PREVIEW_PEN
        self.generic_poly_mousePressEvent(e)

    def polyline_timerEvent(self, final=False):
        if False:
            return 10
        self.generic_poly_timerEvent(final)

    def polyline_mouseMoveEvent(self, e):
        if False:
            return 10
        self.generic_poly_mouseMoveEvent(e)

    def polyline_mouseDoubleClickEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.generic_poly_mouseDoubleClickEvent(e)

    def rect_mousePressEvent(self, e):
        if False:
            print('Hello World!')
        self.active_shape_fn = 'drawRect'
        self.active_shape_args = ()
        self.preview_pen = PREVIEW_PEN
        self.generic_shape_mousePressEvent(e)

    def rect_timerEvent(self, final=False):
        if False:
            for i in range(10):
                print('nop')
        self.generic_shape_timerEvent(final)

    def rect_mouseMoveEvent(self, e):
        if False:
            for i in range(10):
                print('nop')
        self.generic_shape_mouseMoveEvent(e)

    def rect_mouseReleaseEvent(self, e):
        if False:
            while True:
                i = 10
        self.generic_shape_mouseReleaseEvent(e)

    def polygon_mousePressEvent(self, e):
        if False:
            print('Hello World!')
        self.active_shape_fn = 'drawPolygon'
        self.preview_pen = PREVIEW_PEN
        self.generic_poly_mousePressEvent(e)

    def polygon_timerEvent(self, final=False):
        if False:
            while True:
                i = 10
        self.generic_poly_timerEvent(final)

    def polygon_mouseMoveEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.generic_poly_mouseMoveEvent(e)

    def polygon_mouseDoubleClickEvent(self, e):
        if False:
            print('Hello World!')
        self.generic_poly_mouseDoubleClickEvent(e)

    def ellipse_mousePressEvent(self, e):
        if False:
            print('Hello World!')
        self.active_shape_fn = 'drawEllipse'
        self.active_shape_args = ()
        self.preview_pen = PREVIEW_PEN
        self.generic_shape_mousePressEvent(e)

    def ellipse_timerEvent(self, final=False):
        if False:
            while True:
                i = 10
        self.generic_shape_timerEvent(final)

    def ellipse_mouseMoveEvent(self, e):
        if False:
            print('Hello World!')
        self.generic_shape_mouseMoveEvent(e)

    def ellipse_mouseReleaseEvent(self, e):
        if False:
            while True:
                i = 10
        self.generic_shape_mouseReleaseEvent(e)

    def roundrect_mousePressEvent(self, e):
        if False:
            while True:
                i = 10
        self.active_shape_fn = 'drawRoundedRect'
        self.active_shape_args = (25, 25)
        self.preview_pen = PREVIEW_PEN
        self.generic_shape_mousePressEvent(e)

    def roundrect_timerEvent(self, final=False):
        if False:
            i = 10
            return i + 15
        self.generic_shape_timerEvent(final)

    def roundrect_mouseMoveEvent(self, e):
        if False:
            while True:
                i = 10
        self.generic_shape_mouseMoveEvent(e)

    def roundrect_mouseReleaseEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.generic_shape_mouseReleaseEvent(e)

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setupUi(self)
        self.horizontalLayout.removeWidget(self.canvas)
        self.canvas = Canvas()
        self.canvas.initialize()
        self.canvas.setMouseTracking(True)
        self.canvas.setFocusPolicy(Qt.StrongFocus)
        self.horizontalLayout.addWidget(self.canvas)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        for mode in MODES:
            btn = getattr(self, '%sButton' % mode)
            btn.pressed.connect(lambda mode=mode: self.canvas.set_mode(mode))
            mode_group.addButton(btn)
        self.primaryButton.pressed.connect(lambda : self.choose_color(self.set_primary_color))
        self.secondaryButton.pressed.connect(lambda : self.choose_color(self.set_secondary_color))
        for (n, hex) in enumerate(COLORS, 1):
            btn = getattr(self, 'colorButton_%d' % n)
            btn.setStyleSheet('QPushButton { background-color: %s; }' % hex)
            btn.hex = hex

            def patch_mousePressEvent(self_, e):
                if False:
                    for i in range(10):
                        print('nop')
                if e.button() == Qt.LeftButton:
                    self.set_primary_color(self_.hex)
                elif e.button() == Qt.RightButton:
                    self.set_secondary_color(self_.hex)
            btn.mousePressEvent = types.MethodType(patch_mousePressEvent, btn)
        self.actionCopy.triggered.connect(self.copy_to_clipboard)
        self.timer = QTimer()
        self.timer.timeout.connect(self.canvas.on_timer)
        self.timer.setInterval(100)
        self.timer.start()
        self.set_primary_color('#000000')
        self.set_secondary_color('#ffffff')
        self.canvas.primary_color_updated.connect(self.set_primary_color)
        self.canvas.secondary_color_updated.connect(self.set_secondary_color)
        self.current_stamp_n = -1
        self.next_stamp()
        self.stampnextButton.pressed.connect(self.next_stamp)
        self.actionNewImage.triggered.connect(self.canvas.initialize)
        self.actionOpenImage.triggered.connect(self.open_file)
        self.actionSaveImage.triggered.connect(self.save_file)
        self.actionClearImage.triggered.connect(self.canvas.reset)
        self.actionInvertColors.triggered.connect(self.invert)
        self.actionFlipHorizontal.triggered.connect(self.flip_horizontal)
        self.actionFlipVertical.triggered.connect(self.flip_vertical)
        self.fontselect = QFontComboBox()
        self.fontToolbar.addWidget(self.fontselect)
        self.fontselect.currentFontChanged.connect(lambda f: self.canvas.set_config('font', f))
        self.fontselect.setCurrentFont(QFont('Times'))
        self.fontsize = QComboBox()
        self.fontsize.addItems([str(s) for s in FONT_SIZES])
        self.fontsize.currentTextChanged.connect(lambda f: self.canvas.set_config('fontsize', int(f)))
        self.fontToolbar.addWidget(self.fontsize)
        self.fontToolbar.addAction(self.actionBold)
        self.actionBold.triggered.connect(lambda s: self.canvas.set_config('bold', s))
        self.fontToolbar.addAction(self.actionItalic)
        self.actionItalic.triggered.connect(lambda s: self.canvas.set_config('italic', s))
        self.fontToolbar.addAction(self.actionUnderline)
        self.actionUnderline.triggered.connect(lambda s: self.canvas.set_config('underline', s))
        sizeicon = QLabel()
        sizeicon.setPixmap(QPixmap(':/icons/border-weight.png'))
        self.drawingToolbar.addWidget(sizeicon)
        self.sizeselect = QSlider()
        self.sizeselect.setRange(1, 20)
        self.sizeselect.setOrientation(Qt.Horizontal)
        self.sizeselect.valueChanged.connect(lambda s: self.canvas.set_config('size', s))
        self.drawingToolbar.addWidget(self.sizeselect)
        self.actionFillShapes.triggered.connect(lambda s: self.canvas.set_config('fill', s))
        self.drawingToolbar.addAction(self.actionFillShapes)
        self.actionFillShapes.setChecked(True)
        self.show()

    def choose_color(self, callback):
        if False:
            while True:
                i = 10
        dlg = QColorDialog()
        if dlg.exec():
            callback(dlg.selectedColor().name())

    def set_primary_color(self, hex):
        if False:
            while True:
                i = 10
        self.canvas.set_primary_color(hex)
        self.primaryButton.setStyleSheet('QPushButton { background-color: %s; }' % hex)

    def set_secondary_color(self, hex):
        if False:
            print('Hello World!')
        self.canvas.set_secondary_color(hex)
        self.secondaryButton.setStyleSheet('QPushButton { background-color: %s; }' % hex)

    def next_stamp(self):
        if False:
            print('Hello World!')
        self.current_stamp_n += 1
        if self.current_stamp_n >= len(STAMPS):
            self.current_stamp_n = 0
        pixmap = QPixmap(STAMPS[self.current_stamp_n])
        self.stampnextButton.setIcon(QIcon(pixmap))
        self.canvas.current_stamp = pixmap

    def copy_to_clipboard(self):
        if False:
            print('Hello World!')
        clipboard = QApplication.clipboard()
        if self.canvas.mode == 'selectrect' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectrect_copy())
        elif self.canvas.mode == 'selectpoly' and self.canvas.locked:
            clipboard.setPixmap(self.canvas.selectpoly_copy())
        else:
            clipboard.setPixmap(self.canvas.pixmap())

    def open_file(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Open image file for editing, scaling the smaller dimension and cropping the remainder.\n        :return:\n        '
        (path, _) = QFileDialog.getOpenFileName(self, 'Open file', '', 'PNG image files (*.png); JPEG image files (*jpg); All files (*.*)')
        if path:
            pixmap = QPixmap()
            pixmap.load(path)
            iw = pixmap.width()
            ih = pixmap.height()
            (cw, ch) = CANVAS_DIMENSIONS
            if iw / cw < ih / ch:
                pixmap = pixmap.scaledToWidth(cw)
                hoff = (pixmap.height() - ch) // 2
                pixmap = pixmap.copy(QRect(QPoint(0, hoff), QPoint(cw, pixmap.height() - hoff)))
            elif iw / cw > ih / ch:
                pixmap = pixmap.scaledToHeight(ch)
                woff = (pixmap.width() - cw) // 2
                pixmap = pixmap.copy(QRect(QPoint(woff, 0), QPoint(pixmap.width() - woff, ch)))
            self.canvas.setPixmap(pixmap)

    def save_file(self):
        if False:
            while True:
                i = 10
        '\n        Save active canvas to image file.\n        :return:\n        '
        (path, _) = QFileDialog.getSaveFileName(self, 'Save file', '', 'PNG Image file (*.png)')
        if path:
            pixmap = self.canvas.pixmap()
            pixmap.save(path, 'PNG')

    def invert(self):
        if False:
            for i in range(10):
                print('nop')
        img = QImage(self.canvas.pixmap())
        img.invertPixels()
        pixmap = QPixmap()
        pixmap.convertFromImage(img)
        self.canvas.setPixmap(pixmap)

    def flip_horizontal(self):
        if False:
            return 10
        pixmap = self.canvas.pixmap()
        self.canvas.setPixmap(pixmap.transformed(QTransform().scale(-1, 1)))

    def flip_vertical(self):
        if False:
            for i in range(10):
                print('nop')
        pixmap = self.canvas.pixmap()
        self.canvas.setPixmap(pixmap.transformed(QTransform().scale(1, -1)))
if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setWindowIcon(QtGui.QIcon(':/icons/piecasso.ico'))
    window = MainWindow()
    app.exec_()