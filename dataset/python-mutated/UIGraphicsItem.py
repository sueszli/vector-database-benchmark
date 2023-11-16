from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
__all__ = ['UIGraphicsItem']

class UIGraphicsItem(GraphicsObject):
    """
    Base class for graphics items with boundaries relative to a GraphicsView or ViewBox.
    The purpose of this class is to allow the creation of GraphicsItems which live inside 
    a scalable view, but whose boundaries will always stay fixed relative to the view's boundaries.
    For example: GridItem, InfiniteLine
    
    The view can be specified on initialization or it can be automatically detected when the item is painted.
    
    NOTE: Only the item's boundingRect is affected; the item is not transformed in any way. Use viewRangeChanged
    to respond to changes in the view.
    """

    def __init__(self, bounds=None, parent=None):
        if False:
            i = 10
            return i + 15
        '\n        ============== =============================================================================\n        **Arguments:**\n        bounds         QRectF with coordinates relative to view box. The default is QRectF(0,0,1,1),\n                       which means the item will have the same bounds as the view.\n        ============== =============================================================================\n        '
        GraphicsObject.__init__(self, parent)
        self.setFlag(self.GraphicsItemFlag.ItemSendsScenePositionChanges)
        if bounds is None:
            self._bounds = QtCore.QRectF(0, 0, 1, 1)
        else:
            self._bounds = bounds
        self._boundingRect = None
        self._updateView()

    def paint(self, *args):
        if False:
            i = 10
            return i + 15
        pass

    def itemChange(self, change, value):
        if False:
            while True:
                i = 10
        ret = GraphicsObject.itemChange(self, change, value)
        if change == self.GraphicsItemChange.ItemScenePositionHasChanged:
            self.setNewBounds()
        return ret

    def boundingRect(self):
        if False:
            i = 10
            return i + 15
        if self._boundingRect is None:
            br = self.viewRect()
            if br is None:
                return QtCore.QRectF()
            else:
                self._boundingRect = br
        return QtCore.QRectF(self._boundingRect)

    def dataBounds(self, axis, frac=1.0, orthoRange=None):
        if False:
            return 10
        'Called by ViewBox for determining the auto-range bounds.\n        By default, UIGraphicsItems are excluded from autoRange.'
        return None

    def viewRangeChanged(self):
        if False:
            for i in range(10):
                print('nop')
        'Called when the view widget/viewbox is resized/rescaled'
        self.setNewBounds()
        self.update()

    def setNewBounds(self):
        if False:
            while True:
                i = 10
        "Update the item's bounding rect to match the viewport"
        self._boundingRect = None
        self.prepareGeometryChange()

    def setPos(self, *args):
        if False:
            for i in range(10):
                print('nop')
        GraphicsObject.setPos(self, *args)
        self.setNewBounds()

    def mouseShape(self):
        if False:
            i = 10
            return i + 15
        'Return the shape of this item after expanding by 2 pixels'
        shape = self.shape()
        ds = self.mapToDevice(shape)
        stroker = QtGui.QPainterPathStroker()
        stroker.setWidh(2)
        ds2 = stroker.createStroke(ds).united(ds)
        return self.mapFromDevice(ds2)