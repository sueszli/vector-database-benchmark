__all__ = ['MouseDragEvent', 'MouseClickEvent', 'HoverEvent']
import weakref
from time import perf_counter
from ..Point import Point
from ..Qt import QtCore

class MouseDragEvent(object):
    """
    Instances of this class are delivered to items in a :class:`GraphicsScene <pyqtgraph.GraphicsScene>` via their mouseDragEvent() method when the item is being mouse-dragged. 
    
    """

    def __init__(self, moveEvent, pressEvent, lastEvent, start=False, finish=False):
        if False:
            print('Hello World!')
        self.start = start
        self.finish = finish
        self.accepted = False
        self.currentItem = None
        self._buttonDownScenePos = {}
        self._buttonDownScreenPos = {}
        for btn in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.RightButton]:
            self._buttonDownScenePos[btn] = moveEvent.buttonDownScenePos(btn)
            self._buttonDownScreenPos[btn] = moveEvent.buttonDownScreenPos(btn)
        self._scenePos = moveEvent.scenePos()
        self._screenPos = moveEvent.screenPos()
        if lastEvent is None:
            self._lastScenePos = pressEvent.scenePos()
            self._lastScreenPos = pressEvent.screenPos()
        else:
            self._lastScenePos = lastEvent.scenePos()
            self._lastScreenPos = lastEvent.screenPos()
        self._buttons = moveEvent.buttons()
        self._button = pressEvent.button()
        self._modifiers = moveEvent.modifiers()
        self.acceptedItem = None

    def accept(self):
        if False:
            return 10
        'An item should call this method if it can handle the event. This will prevent the event being delivered to any other items.'
        self.accepted = True
        self.acceptedItem = self.currentItem

    def ignore(self):
        if False:
            return 10
        'An item should call this method if it cannot handle the event. This will allow the event to be delivered to other items.'
        self.accepted = False

    def isAccepted(self):
        if False:
            print('Hello World!')
        return self.accepted

    def scenePos(self):
        if False:
            i = 10
            return i + 15
        'Return the current scene position of the mouse.'
        return Point(self._scenePos)

    def screenPos(self):
        if False:
            while True:
                i = 10
        'Return the current screen position (pixels relative to widget) of the mouse.'
        return Point(self._screenPos)

    def buttonDownScenePos(self, btn=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the scene position of the mouse at the time *btn* was pressed.\n        If *btn* is omitted, then the button that initiated the drag is assumed.\n        '
        if btn is None:
            btn = self.button()
        return Point(self._buttonDownScenePos[btn])

    def buttonDownScreenPos(self, btn=None):
        if False:
            print('Hello World!')
        '\n        Return the screen position (pixels relative to widget) of the mouse at the time *btn* was pressed.\n        If *btn* is omitted, then the button that initiated the drag is assumed.\n        '
        if btn is None:
            btn = self.button()
        return Point(self._buttonDownScreenPos[btn])

    def lastScenePos(self):
        if False:
            return 10
        '\n        Return the scene position of the mouse immediately prior to this event.\n        '
        return Point(self._lastScenePos)

    def lastScreenPos(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the screen position of the mouse immediately prior to this event.\n        '
        return Point(self._lastScreenPos)

    def buttons(self):
        if False:
            while True:
                i = 10
        '\n        Return the buttons currently pressed on the mouse.\n        (see QGraphicsSceneMouseEvent::buttons in the Qt documentation)\n        '
        return self._buttons

    def button(self):
        if False:
            print('Hello World!')
        'Return the button that initiated the drag (may be different from the buttons currently pressed)\n        (see QGraphicsSceneMouseEvent::button in the Qt documentation)\n        \n        '
        return self._button

    def pos(self):
        if False:
            print('Hello World!')
        '\n        Return the current position of the mouse in the coordinate system of the item\n        that the event was delivered to.\n        '
        return Point(self.currentItem.mapFromScene(self._scenePos))

    def lastPos(self):
        if False:
            while True:
                i = 10
        '\n        Return the previous position of the mouse in the coordinate system of the item\n        that the event was delivered to.\n        '
        return Point(self.currentItem.mapFromScene(self._lastScenePos))

    def buttonDownPos(self, btn=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the position of the mouse at the time the drag was initiated\n        in the coordinate system of the item that the event was delivered to.\n        '
        if btn is None:
            btn = self.button()
        return Point(self.currentItem.mapFromScene(self._buttonDownScenePos[btn]))

    def isStart(self):
        if False:
            i = 10
            return i + 15
        'Returns True if this event is the first since a drag was initiated.'
        return self.start

    def isFinish(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns False if this is the last event in a drag. Note that this\n        event will have the same position as the previous one.'
        return self.finish

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self.currentItem is None:
            lp = self._lastScenePos
            p = self._scenePos
        else:
            lp = self.lastPos()
            p = self.pos()
        return '<MouseDragEvent (%g,%g)->(%g,%g) buttons=%s start=%s finish=%s>' % (lp.x(), lp.y(), p.x(), p.y(), str(self.buttons()), str(self.isStart()), str(self.isFinish()))

    def modifiers(self):
        if False:
            print('Hello World!')
        'Return any keyboard modifiers currently pressed.\n        (see QGraphicsSceneMouseEvent::modifiers in the Qt documentation)\n        \n        '
        return self._modifiers

class MouseClickEvent(object):
    """
    Instances of this class are delivered to items in a :class:`GraphicsScene <pyqtgraph.GraphicsScene>` via their mouseClickEvent() method when the item is clicked. 
    
    
    """

    def __init__(self, pressEvent, double=False):
        if False:
            print('Hello World!')
        self.accepted = False
        self.currentItem = None
        self._double = double
        self._scenePos = pressEvent.scenePos()
        self._screenPos = pressEvent.screenPos()
        self._button = pressEvent.button()
        self._buttons = pressEvent.buttons()
        self._modifiers = pressEvent.modifiers()
        self._time = perf_counter()
        self.acceptedItem = None

    def accept(self):
        if False:
            return 10
        'An item should call this method if it can handle the event. This will prevent the event being delivered to any other items.'
        self.accepted = True
        self.acceptedItem = self.currentItem

    def ignore(self):
        if False:
            while True:
                i = 10
        'An item should call this method if it cannot handle the event. This will allow the event to be delivered to other items.'
        self.accepted = False

    def isAccepted(self):
        if False:
            for i in range(10):
                print('nop')
        return self.accepted

    def scenePos(self):
        if False:
            while True:
                i = 10
        'Return the current scene position of the mouse.'
        return Point(self._scenePos)

    def screenPos(self):
        if False:
            print('Hello World!')
        'Return the current screen position (pixels relative to widget) of the mouse.'
        return Point(self._screenPos)

    def buttons(self):
        if False:
            while True:
                i = 10
        '\n        Return the buttons currently pressed on the mouse.\n        (see QGraphicsSceneMouseEvent::buttons in the Qt documentation)\n        '
        return self._buttons

    def button(self):
        if False:
            while True:
                i = 10
        'Return the mouse button that generated the click event.\n        (see QGraphicsSceneMouseEvent::button in the Qt documentation)\n        '
        return self._button

    def double(self):
        if False:
            print('Hello World!')
        'Return True if this is a double-click.'
        return self._double

    def pos(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the current position of the mouse in the coordinate system of the item\n        that the event was delivered to.\n        '
        return Point(self.currentItem.mapFromScene(self._scenePos))

    def lastPos(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the previous position of the mouse in the coordinate system of the item\n        that the event was delivered to.\n        '
        return Point(self.currentItem.mapFromScene(self._lastScenePos))

    def modifiers(self):
        if False:
            for i in range(10):
                print('nop')
        'Return any keyboard modifiers currently pressed.\n        (see QGraphicsSceneMouseEvent::modifiers in the Qt documentation)        \n        '
        return self._modifiers

    def __repr__(self):
        if False:
            print('Hello World!')
        try:
            if self.currentItem is None:
                p = self._scenePos
            else:
                p = self.pos()
            return '<MouseClickEvent (%g,%g) button=%s>' % (p.x(), p.y(), str(self.button()))
        except:
            return '<MouseClickEvent button=%s>' % str(self.button())

    def time(self):
        if False:
            while True:
                i = 10
        return self._time

class HoverEvent(object):
    """
    Instances of this class are delivered to items in a :class:`GraphicsScene <pyqtgraph.GraphicsScene>` via their hoverEvent() method when the mouse is hovering over the item.
    This event class both informs items that the mouse cursor is nearby and allows items to 
    communicate with one another about whether each item will accept *potential* mouse events. 
    
    It is common for multiple overlapping items to receive hover events and respond by changing 
    their appearance. This can be misleading to the user since, in general, only one item will
    respond to mouse events. To avoid this, items make calls to event.acceptClicks(button) 
    and/or acceptDrags(button).
    
    Each item may make multiple calls to acceptClicks/Drags, each time for a different button. 
    If the method returns True, then the item is guaranteed to be
    the recipient of the claimed event IF the user presses the specified mouse button before
    moving. If claimEvent returns False, then this item is guaranteed NOT to get the specified
    event (because another has already claimed it) and the item should change its appearance 
    accordingly.
    
    event.isEnter() returns True if the mouse has just entered the item's shape;
    event.isExit() returns True if the mouse has just left.
    """

    def __init__(self, moveEvent, acceptable):
        if False:
            return 10
        self.enter = False
        self.acceptable = acceptable
        self.exit = False
        self.__clickItems = weakref.WeakValueDictionary()
        self.__dragItems = weakref.WeakValueDictionary()
        self.currentItem = None
        if moveEvent is not None:
            self._scenePos = moveEvent.scenePos()
            self._screenPos = moveEvent.screenPos()
            self._lastScenePos = moveEvent.lastScenePos()
            self._lastScreenPos = moveEvent.lastScreenPos()
            self._buttons = moveEvent.buttons()
            self._modifiers = moveEvent.modifiers()
        else:
            self.exit = True

    def isEnter(self):
        if False:
            while True:
                i = 10
        "Returns True if the mouse has just entered the item's shape"
        return self.enter

    def isExit(self):
        if False:
            for i in range(10):
                print('nop')
        "Returns True if the mouse has just exited the item's shape"
        return self.exit

    def acceptClicks(self, button):
        if False:
            return 10
        'Inform the scene that the item (that the event was delivered to)\n        would accept a mouse click event if the user were to click before\n        moving the mouse again.\n        \n        Returns True if the request is successful, otherwise returns False (indicating\n        that some other item would receive an incoming click).\n        '
        if not self.acceptable:
            return False
        if button not in self.__clickItems:
            self.__clickItems[button] = self.currentItem
            return True
        return False

    def acceptDrags(self, button):
        if False:
            i = 10
            return i + 15
        'Inform the scene that the item (that the event was delivered to)\n        would accept a mouse drag event if the user were to drag before\n        the next hover event.\n        \n        Returns True if the request is successful, otherwise returns False (indicating\n        that some other item would receive an incoming drag event).\n        '
        if not self.acceptable:
            return False
        if button not in self.__dragItems:
            self.__dragItems[button] = self.currentItem
            return True
        return False

    def scenePos(self):
        if False:
            while True:
                i = 10
        'Return the current scene position of the mouse.'
        return Point(self._scenePos)

    def screenPos(self):
        if False:
            i = 10
            return i + 15
        'Return the current screen position of the mouse.'
        return Point(self._screenPos)

    def lastScenePos(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the previous scene position of the mouse.'
        return Point(self._lastScenePos)

    def lastScreenPos(self):
        if False:
            return 10
        'Return the previous screen position of the mouse.'
        return Point(self._lastScreenPos)

    def buttons(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the buttons currently pressed on the mouse.\n        (see QGraphicsSceneMouseEvent::buttons in the Qt documentation)\n        '
        return self._buttons

    def pos(self):
        if False:
            return 10
        '\n        Return the current position of the mouse in the coordinate system of the item\n        that the event was delivered to.\n        '
        return Point(self.currentItem.mapFromScene(self._scenePos))

    def lastPos(self):
        if False:
            return 10
        '\n        Return the previous position of the mouse in the coordinate system of the item\n        that the event was delivered to.\n        '
        return Point(self.currentItem.mapFromScene(self._lastScenePos))

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if self.exit:
            return '<HoverEvent exit=True>'
        if self.currentItem is None:
            lp = self._lastScenePos
            p = self._scenePos
        else:
            lp = self.lastPos()
            p = self.pos()
        return '<HoverEvent (%g,%g)->(%g,%g) buttons=%s enter=%s exit=%s>' % (lp.x(), lp.y(), p.x(), p.y(), str(self.buttons()), str(self.isEnter()), str(self.isExit()))

    def modifiers(self):
        if False:
            while True:
                i = 10
        'Return any keyboard modifiers currently pressed.\n        (see QGraphicsSceneMouseEvent::modifiers in the Qt documentation)        \n        '
        return self._modifiers

    def clickItems(self):
        if False:
            print('Hello World!')
        return self.__clickItems

    def dragItems(self):
        if False:
            while True:
                i = 10
        return self.__dragItems