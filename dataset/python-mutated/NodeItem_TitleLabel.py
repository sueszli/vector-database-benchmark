from qtpy.QtCore import QRectF, QPointF, QSizeF, Property
from qtpy.QtGui import QFont, QFontMetricsF, QColor
from qtpy.QtWidgets import QGraphicsWidget, QGraphicsLayoutItem, QGraphicsItem
from ...utils import get_longest_line

class TitleLabel(QGraphicsWidget):

    def __init__(self, node_gui, node_item):
        if False:
            while True:
                i = 10
        super(TitleLabel, self).__init__(parent=node_item)
        self.setGraphicsItem(self)
        self.node_gui = node_gui
        self.node_item = node_item
        font = QFont('Poppins', 15) if self.node_gui.style == 'normal' else QFont('K2D', 20, QFont.Bold, True)
        self.fm = QFontMetricsF(font)
        (self.title_str, self.width, self.height) = (None, None, None)
        self.update_shape()
        self.color = QColor(30, 43, 48)
        self.pen_width = 1.5
        self.hovering = False

    def update_shape(self):
        if False:
            while True:
                i = 10
        self.title_str = self.node_gui.display_title
        self.width = self.fm.width(get_longest_line(self.title_str) + '___')
        self.height = self.fm.height() * 0.7 * (self.title_str.count('\n') + 1)

    def boundingRect(self):
        if False:
            i = 10
            return i + 15
        return QRectF(QPointF(0, 0), self.geometry().size())

    def setGeometry(self, rect):
        if False:
            return 10
        self.prepareGeometryChange()
        QGraphicsLayoutItem.setGeometry(self, rect)
        self.setPos(rect.topLeft())

    def sizeHint(self, which, constraint=...):
        if False:
            i = 10
            return i + 15
        return QSizeF(self.width, self.height)

    def paint(self, painter, option, widget=None):
        if False:
            print('Hello World!')
        self.node_item.session_design.flow_theme.paint_NI_title_label(self.node_gui, self.node_item.isSelected(), self.hovering, painter, option, self.design_style(), self.title_str, self.node_item.color, self.boundingRect())

    def design_style(self):
        if False:
            i = 10
            return i + 15
        return self.node_gui.style

    def set_NI_hover_state(self, hovering: bool):
        if False:
            print('Hello World!')
        self.hovering = hovering
        self.update()

    def get_color(self):
        if False:
            print('Hello World!')
        return self.color

    def set_color(self, val):
        if False:
            return 10
        self.color = val
        QGraphicsItem.update(self)
    p_color = Property(QColor, get_color, set_color)