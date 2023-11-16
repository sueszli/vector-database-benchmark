from qtpy.QtCore import QSize, QRectF, QPointF, QSizeF
from qtpy.QtGui import QPixmap, QImage, QPainter, QIcon, QPicture
from qtpy.QtWidgets import QGraphicsPixmapItem, QGraphicsWidget, QGraphicsLayoutItem
from ...utils import change_svg_color

class NodeItem_Icon(QGraphicsWidget):

    def __init__(self, node_gui, node_item):
        if False:
            while True:
                i = 10
        super().__init__(parent=node_item)
        if node_gui.style == 'normal':
            self.size = QSize(20, 20)
        else:
            self.size = QSize(50, 50)
        self.setGraphicsItem(self)
        image = QImage(node_gui.icon)
        self.pixmap = QPixmap.fromImage(image)

    def boundingRect(self):
        if False:
            i = 10
            return i + 15
        return QRectF(QPointF(0, 0), self.size)

    def setGeometry(self, rect):
        if False:
            return 10
        self.prepareGeometryChange()
        QGraphicsLayoutItem.setGeometry(self, rect)
        self.setPos(rect.topLeft())

    def sizeHint(self, which, constraint=...):
        if False:
            return 10
        return QSizeF(self.size.width(), self.size.height())

    def paint(self, painter, option, widget=None):
        if False:
            while True:
                i = 10
        painter.drawPixmap(0, 0, self.size.width(), self.size.height(), self.pixmap)