from qtpy.QtCore import QSize, QRectF, QPointF, QSizeF, Qt
from qtpy.QtWidgets import QGraphicsWidget, QGraphicsLayoutItem
from qtpy.QtGui import QColor
from ...GlobalAttributes import Location
from ...utils import change_svg_color, get_resource

class NodeItem_CollapseButton(QGraphicsWidget):

    def __init__(self, node_gui, node_item):
        if False:
            print('Hello World!')
        super().__init__(parent=node_item)
        self.node_gui = node_gui
        self.node_item = node_item
        self.size = QSizeF(14, 7)
        self.setGraphicsItem(self)
        self.setCursor(Qt.PointingHandCursor)
        self.collapse_pixmap = change_svg_color(get_resource('node_collapse_icon.svg'), self.node_gui.color)
        self.expand_pixmap = change_svg_color(get_resource('node_expand_icon.svg'), self.node_gui.color)

    def boundingRect(self):
        if False:
            return 10
        return QRectF(QPointF(0, 0), self.size)

    def setGeometry(self, rect):
        if False:
            for i in range(10):
                print('nop')
        self.prepareGeometryChange()
        QGraphicsLayoutItem.setGeometry(self, rect)
        self.setPos(rect.topLeft())

    def sizeHint(self, which, constraint=...):
        if False:
            i = 10
            return i + 15
        return QSizeF(self.size.width(), self.size.height())

    def mousePressEvent(self, event):
        if False:
            return 10
        event.accept()
        self.node_item.flow_view.mouse_event_taken = True
        if self.node_item.collapsed:
            self.node_item.expand()
        else:
            self.node_item.collapse()

    def paint(self, painter, option, widget=None):
        if False:
            for i in range(10):
                print('nop')
        if not self.node_item.hovered:
            return
        if self.node_item.collapsed:
            pixmap = self.expand_pixmap
        else:
            pixmap = self.collapse_pixmap
        painter.drawPixmap(0, 0, self.size.width(), self.size.height(), pixmap)