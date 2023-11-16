import traceback
from qtpy.QtWidgets import QGraphicsPixmapItem
from qtpy.QtGui import QPixmap, QImage
from ...GUIBase import GUIBase
from ...utils import get_resource

class NodeErrorIndicator(GUIBase, QGraphicsPixmapItem):

    def __init__(self, node_item):
        if False:
            while True:
                i = 10
        GUIBase.__init__(self)
        QGraphicsPixmapItem.__init__(self, parent=node_item)
        self.node = node_item
        self.pix = QPixmap(str(get_resource('pics/warning.png')))
        self.setPixmap(self.pix)
        self.setScale(0.1)
        self.setOffset(-self.boundingRect().width() / 2, -self.boundingRect().width() / 2)

    def set_error(self, e):
        if False:
            i = 10
            return i + 15
        error_msg = ''.join([f'<p>{line}</p>' for line in traceback.format_exc().splitlines()])
        self.setToolTip(f'<html><head/><body>{error_msg}</body></html>')