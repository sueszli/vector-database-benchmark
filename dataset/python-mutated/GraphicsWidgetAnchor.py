from ..Point import Point
__all__ = ['GraphicsWidgetAnchor']

class GraphicsWidgetAnchor(object):
    """
    Class used to allow GraphicsWidgets to anchor to a specific position on their
    parent. The item will be automatically repositioned if the parent is resized. 
    This is used, for example, to anchor a LegendItem to a corner of its parent 
    PlotItem.

    """

    def __init__(self):
        if False:
            return 10
        self.__parent = None
        self.__parentAnchor = None
        self.__itemAnchor = None
        self.__offset = (0, 0)
        if hasattr(self, 'geometryChanged'):
            self.geometryChanged.connect(self.__geometryChanged)

    def anchor(self, itemPos, parentPos, offset=(0, 0)):
        if False:
            return 10
        "\n        Anchors the item at its local itemPos to the item's parent at parentPos.\n        Both positions are expressed in values relative to the size of the item or parent;\n        a value of 0 indicates left or top edge, while 1 indicates right or bottom edge.\n        \n        Optionally, offset may be specified to introduce an absolute offset. \n        \n        Example: anchor a box such that its upper-right corner is fixed 10px left\n        and 10px down from its parent's upper-right corner::\n        \n            box.anchor(itemPos=(1,0), parentPos=(1,0), offset=(-10,10))\n        "
        parent = self.parentItem()
        if parent is None:
            raise Exception('Cannot anchor; parent is not set.')
        if self.__parent is not parent:
            if self.__parent is not None:
                self.__parent.geometryChanged.disconnect(self.__geometryChanged)
            self.__parent = parent
            parent.geometryChanged.connect(self.__geometryChanged)
        self.__itemAnchor = itemPos
        self.__parentAnchor = parentPos
        self.__offset = offset
        self.__geometryChanged()

    def autoAnchor(self, pos, relative=True):
        if False:
            print('Hello World!')
        "\n        Set the position of this item relative to its parent by automatically \n        choosing appropriate anchor settings.\n        \n        If relative is True, one corner of the item will be anchored to \n        the appropriate location on the parent with no offset. The anchored\n        corner will be whichever is closest to the parent's boundary.\n        \n        If relative is False, one corner of the item will be anchored to the same\n        corner of the parent, with an absolute offset to achieve the correct\n        position. \n        "
        pos = Point(pos)
        br = self.mapRectToParent(self.boundingRect()).translated(pos - self.pos())
        pbr = self.parentItem().boundingRect()
        anchorPos = [0, 0]
        parentPos = Point()
        itemPos = Point()
        if abs(br.left() - pbr.left()) < abs(br.right() - pbr.right()):
            anchorPos[0] = 0
            parentPos[0] = pbr.left()
            itemPos[0] = br.left()
        else:
            anchorPos[0] = 1
            parentPos[0] = pbr.right()
            itemPos[0] = br.right()
        if abs(br.top() - pbr.top()) < abs(br.bottom() - pbr.bottom()):
            anchorPos[1] = 0
            parentPos[1] = pbr.top()
            itemPos[1] = br.top()
        else:
            anchorPos[1] = 1
            parentPos[1] = pbr.bottom()
            itemPos[1] = br.bottom()
        if relative:
            relPos = [(itemPos[0] - pbr.left()) / pbr.width(), (itemPos[1] - pbr.top()) / pbr.height()]
            self.anchor(anchorPos, relPos)
        else:
            offset = itemPos - parentPos
            self.anchor(anchorPos, anchorPos, offset)

    def __geometryChanged(self):
        if False:
            i = 10
            return i + 15
        if self.__parent is None:
            return
        if self.__itemAnchor is None:
            return
        o = self.mapToParent(Point(0, 0))
        a = self.boundingRect().bottomRight() * Point(self.__itemAnchor)
        a = self.mapToParent(a)
        p = self.__parent.boundingRect().bottomRight() * Point(self.__parentAnchor)
        off = Point(self.__offset)
        pos = p + (o - a) + off
        self.setPos(pos)