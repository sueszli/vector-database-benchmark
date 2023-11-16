from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
__all__ = ['ItemGroup']

class ItemGroup(GraphicsObject):
    """
    Replacement for QGraphicsItemGroup
    """

    def __init__(self, *args):
        if False:
            while True:
                i = 10
        GraphicsObject.__init__(self, *args)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)

    def boundingRect(self):
        if False:
            print('Hello World!')
        return QtCore.QRectF()

    def paint(self, *args):
        if False:
            return 10
        pass

    def addItem(self, item):
        if False:
            i = 10
            return i + 15
        item.setParentItem(self)