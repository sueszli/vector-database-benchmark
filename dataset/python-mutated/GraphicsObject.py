import warnings
from ..Qt import QT_LIB, QtCore, QtWidgets
from .GraphicsItem import GraphicsItem
__all__ = ['GraphicsObject']

class GraphicsObject(GraphicsItem, QtWidgets.QGraphicsObject):
    """
    **Bases:** :class:`GraphicsItem <pyqtgraph.GraphicsItem>`, :class:`QtWidgets.QGraphicsObject`

    Extension of QGraphicsObject with some useful methods (provided by :class:`GraphicsItem <pyqtgraph.GraphicsItem>`)
    """
    _qtBaseClass = QtWidgets.QGraphicsObject

    def __init__(self, *args):
        if False:
            print('Hello World!')
        self.__inform_view_on_changes = True
        QtWidgets.QGraphicsObject.__init__(self, *args)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges)
        GraphicsItem.__init__(self)

    def itemChange(self, change, value):
        if False:
            return 10
        ret = super().itemChange(change, value)
        if change in [self.GraphicsItemChange.ItemParentHasChanged, self.GraphicsItemChange.ItemSceneHasChanged]:
            if self.__class__.__dict__.get('parentChanged') is not None:
                warnings.warn('parentChanged() is deprecated and will be removed in the future. Use changeParent() instead.', DeprecationWarning, stacklevel=2)
                if QT_LIB == 'PySide6' and QtCore.__version_info__ == (6, 2, 2):
                    getattr(self.__class__, 'parentChanged')(self)
                else:
                    self.parentChanged()
            else:
                self.changeParent()
        try:
            inform_view_on_change = self.__inform_view_on_changes
        except AttributeError:
            pass
        else:
            if inform_view_on_change and change in [self.GraphicsItemChange.ItemPositionHasChanged, self.GraphicsItemChange.ItemTransformHasChanged]:
                self.informViewBoundsChanged()
        return ret