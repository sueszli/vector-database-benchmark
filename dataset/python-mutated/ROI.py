"""
ROI.py -  Interactive graphics items for GraphicsView (ROI widgets)
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.

Implements a series of graphics items which display movable/scalable/rotatable shapes
for use as region-of-interest markers. ROI class automatically handles extraction 
of array data from ImageItems.

The ROI class is meant to serve as the base for more specific types; see several examples
of how to build an ROI at the bottom of the file.
"""
import sys
from math import atan2, cos, degrees, hypot, sin
import numpy as np
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from ..SRTTransform import SRTTransform
from .GraphicsObject import GraphicsObject
from .UIGraphicsItem import UIGraphicsItem
translate = QtCore.QCoreApplication.translate
__all__ = ['ROI', 'TestROI', 'RectROI', 'EllipseROI', 'CircleROI', 'LineROI', 'MultiLineROI', 'MultiRectROI', 'LineSegmentROI', 'PolyLineROI', 'CrosshairROI', 'TriangleROI']

def rectStr(r):
    if False:
        i = 10
        return i + 15
    return '[%f, %f] + [%f, %f]' % (r.x(), r.y(), r.width(), r.height())

class ROI(GraphicsObject):
    """
    Generic region-of-interest widget.
    
    Can be used for implementing many types of selection box with 
    rotate/translate/scale handles.
    ROIs can be customized to have a variety of shapes (by subclassing or using
    any of the built-in subclasses) and any combination of draggable handles
    that allow the user to manipulate the ROI.

    Default mouse interaction:

      * Left drag moves the ROI
      * Left drag + Ctrl moves the ROI with position snapping
      * Left drag + Alt rotates the ROI
      * Left drag + Alt + Ctrl rotates the ROI with angle snapping
      * Left drag + Shift scales the ROI
      * Left drag + Shift + Ctrl scales the ROI with size snapping

    In addition to the above interaction modes, it is possible to attach any
    number of handles to the ROI that can be dragged to change the ROI in
    various ways (see the ROI.add____Handle methods).


    ================ ===========================================================
    **Arguments**
    pos              (length-2 sequence) Indicates the position of the ROI's 
                     origin. For most ROIs, this is the lower-left corner of
                     its bounding rectangle.
    size             (length-2 sequence) Indicates the width and height of the 
                     ROI.
    angle            (float) The rotation of the ROI in degrees. Default is 0.
    invertible       (bool) If True, the user may resize the ROI to have 
                     negative width or height (assuming the ROI has scale
                     handles). Default is False.
    maxBounds        (QRect, QRectF, or None) Specifies boundaries that the ROI 
                     cannot be dragged outside of by the user. Default is None.
    snapSize         (float) The spacing of snap positions used when *scaleSnap*
                     or *translateSnap* are enabled. Default is 1.0.
    scaleSnap        (bool) If True, the width and height of the ROI are forced
                     to be integer multiples of *snapSize* when being resized
                     by the user. Default is False.
    translateSnap    (bool) If True, the x and y positions of the ROI are forced
                     to be integer multiples of *snapSize* when being resized
                     by the user. Default is False.
    rotateSnap       (bool) If True, the ROI angle is forced to a multiple of 
                     the ROI's snap angle (default is 15 degrees) when rotated
                     by the user. Default is False.
    parent           (QGraphicsItem) The graphics item parent of this ROI. It
                     is generally not necessary to specify the parent.
    pen              (QPen or argument to pg.mkPen) The pen to use when drawing
                     the shape of the ROI.
    hoverPen         (QPen or argument to mkPen) The pen to use while the
                     mouse is hovering over the ROI shape.
    handlePen        (QPen or argument to mkPen) The pen to use when drawing
                     the ROI handles.
    handleHoverPen   (QPen or argument to mkPen) The pen to use while the mouse
                     is hovering over an ROI handle.
    movable          (bool) If True, the ROI can be moved by dragging anywhere 
                     inside the ROI. Default is True.
    rotatable        (bool) If True, the ROI can be rotated by mouse drag + ALT
    resizable        (bool) If True, the ROI can be resized by mouse drag + 
                     SHIFT
    removable        (bool) If True, the ROI will be given a context menu with
                     an option to remove the ROI. The ROI emits
                     sigRemoveRequested when this menu action is selected.
                     Default is False.
    ================ ===========================================================
    
    
    
    ======================= ====================================================
    **Signals**
    sigRegionChangeFinished Emitted when the user stops dragging the ROI (or
                            one of its handles) or if the ROI is changed
                            programatically.
    sigRegionChangeStarted  Emitted when the user starts dragging the ROI (or
                            one of its handles).
    sigRegionChanged        Emitted any time the position of the ROI changes,
                            including while it is being dragged by the user.
    sigHoverEvent           Emitted when the mouse hovers over the ROI.
    sigClicked              Emitted when the user clicks on the ROI.
                            Note that clicking is disabled by default to prevent
                            stealing clicks from objects behind the ROI. To 
                            enable clicking, call 
                            roi.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton). 
                            See QtWidgets.QGraphicsItem documentation for more 
                            details.
    sigRemoveRequested      Emitted when the user selects 'remove' from the 
                            ROI's context menu (if available).
    ======================= ====================================================
    """
    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChangeStarted = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)
    sigHoverEvent = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    sigRemoveRequested = QtCore.Signal(object)

    def __init__(self, pos, size=Point(1, 1), angle=0.0, invertible=False, maxBounds=None, snapSize=1.0, scaleSnap=False, translateSnap=False, rotateSnap=False, parent=None, pen=None, hoverPen=None, handlePen=None, handleHoverPen=None, movable=True, rotatable=True, resizable=True, removable=False, aspectLocked=False):
        if False:
            return 10
        GraphicsObject.__init__(self, parent)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        pos = Point(pos)
        size = Point(size)
        self.aspectLocked = aspectLocked
        self.translatable = movable
        self.rotatable = rotatable
        self.resizable = resizable
        self.removable = removable
        self.menu = None
        self.freeHandleMoved = False
        self.mouseHovering = False
        if pen is None:
            pen = (255, 255, 255)
        self.setPen(pen)
        if hoverPen is None:
            hoverPen = (255, 255, 0)
        self.hoverPen = fn.mkPen(hoverPen)
        if handlePen is None:
            handlePen = (150, 255, 255)
        self.handlePen = fn.mkPen(handlePen)
        if handleHoverPen is None:
            handleHoverPen = (255, 255, 0)
        self.handleHoverPen = handleHoverPen
        self.handles = []
        self.state = {'pos': Point(0, 0), 'size': Point(1, 1), 'angle': 0}
        self.lastState = None
        self.setPos(pos)
        self.setAngle(angle)
        self.setSize(size)
        self.setZValue(10)
        self.isMoving = False
        self.handleSize = 5
        self.invertible = invertible
        self.maxBounds = maxBounds
        self.snapSize = snapSize
        self.translateSnap = translateSnap
        self.rotateSnap = rotateSnap
        self.rotateSnapAngle = 15.0
        self.scaleSnap = scaleSnap
        self.scaleSnapSize = snapSize
        self.mouseDragHandler = MouseDragHandler(self)

    def getState(self):
        if False:
            i = 10
            return i + 15
        return self.stateCopy()

    def stateCopy(self):
        if False:
            for i in range(10):
                print('nop')
        sc = {}
        sc['pos'] = Point(self.state['pos'])
        sc['size'] = Point(self.state['size'])
        sc['angle'] = self.state['angle']
        return sc

    def saveState(self):
        if False:
            print('Hello World!')
        'Return the state of the widget in a format suitable for storing to \n        disk. (Points are converted to tuple)\n        \n        Combined with setState(), this allows ROIs to be easily saved and \n        restored.'
        state = {}
        state['pos'] = tuple(self.state['pos'])
        state['size'] = tuple(self.state['size'])
        state['angle'] = self.state['angle']
        return state

    def setState(self, state, update=True):
        if False:
            print('Hello World!')
        '\n        Set the state of the ROI from a structure generated by saveState() or\n        getState().\n        '
        self.setPos(state['pos'], update=False)
        self.setSize(state['size'], update=False)
        self.setAngle(state['angle'], update=update)

    def setZValue(self, z):
        if False:
            while True:
                i = 10
        QtWidgets.QGraphicsItem.setZValue(self, z)
        for h in self.handles:
            h['item'].setZValue(z + 1)

    def parentBounds(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the bounding rectangle of this ROI in the coordinate system\n        of its parent.        \n        '
        return self.mapToParent(self.boundingRect()).boundingRect()

    def setPen(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Set the pen to use when drawing the ROI shape.\n        For arguments, see :func:`mkPen <pyqtgraph.mkPen>`.\n        '
        self.pen = fn.mkPen(*args, **kwargs)
        self.currentPen = self.pen
        self.update()

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the size (w,h) of the ROI.'
        return self.getState()['size']

    def pos(self):
        if False:
            for i in range(10):
                print('nop')
        "Return the position (x,y) of the ROI's origin. \n        For most ROIs, this will be the lower-left corner."
        return self.getState()['pos']

    def angle(self):
        if False:
            while True:
                i = 10
        'Return the angle of the ROI in degrees.'
        return self.getState()['angle']

    def setPos(self, pos, y=None, update=True, finish=True):
        if False:
            for i in range(10):
                print('nop')
        "Set the position of the ROI (in the parent's coordinate system).\n        \n        Accepts either separate (x, y) arguments or a single :class:`Point` or\n        ``QPointF`` argument. \n        \n        By default, this method causes both ``sigRegionChanged`` and\n        ``sigRegionChangeFinished`` to be emitted. If *finish* is False, then\n        ``sigRegionChangeFinished`` will not be emitted. You can then use \n        stateChangeFinished() to cause the signal to be emitted after a series\n        of state changes.\n        \n        If *update* is False, the state change will be remembered but not processed and no signals \n        will be emitted. You can then use stateChanged() to complete the state change. This allows\n        multiple change functions to be called sequentially while minimizing processing overhead\n        and repeated signals. Setting ``update=False`` also forces ``finish=False``.\n        "
        if update not in (True, False):
            raise TypeError('update argument must be bool')
        if y is None:
            pos = Point(pos)
        else:
            if isinstance(y, bool):
                raise TypeError('Positional arguments to setPos() must be numerical.')
            pos = Point(pos, y)
        self.state['pos'] = pos
        QtWidgets.QGraphicsItem.setPos(self, pos)
        if update:
            self.stateChanged(finish=finish)

    def setSize(self, size, center=None, centerLocal=None, snap=False, update=True, finish=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the ROI's size.\n        \n        =============== ==========================================================================\n        **Arguments**\n        size            (Point | QPointF | sequence) The final size of the ROI\n        center          (None | Point) Optional center point around which the ROI is scaled,\n                        expressed as [0-1, 0-1] over the size of the ROI.\n        centerLocal     (None | Point) Same as *center*, but the position is expressed in the\n                        local coordinate system of the ROI\n        snap            (bool) If True, the final size is snapped to the nearest increment (see\n                        ROI.scaleSnapSize)\n        update          (bool) See setPos()\n        finish          (bool) See setPos()\n        =============== ==========================================================================\n        "
        if update not in (True, False):
            raise TypeError('update argument must be bool')
        size = Point(size)
        if snap:
            size[0] = round(size[0] / self.scaleSnapSize) * self.scaleSnapSize
            size[1] = round(size[1] / self.scaleSnapSize) * self.scaleSnapSize
        if centerLocal is not None:
            oldSize = Point(self.state['size'])
            oldSize[0] = 1 if oldSize[0] == 0 else oldSize[0]
            oldSize[1] = 1 if oldSize[1] == 0 else oldSize[1]
            center = Point(centerLocal) / oldSize
        if center is not None:
            center = Point(center)
            c = self.mapToParent(Point(center) * self.state['size'])
            c1 = self.mapToParent(Point(center) * size)
            newPos = self.state['pos'] + c - c1
            self.setPos(newPos, update=False, finish=False)
        self.prepareGeometryChange()
        self.state['size'] = size
        if update:
            self.stateChanged(finish=finish)

    def setAngle(self, angle, center=None, centerLocal=None, snap=False, update=True, finish=True):
        if False:
            print('Hello World!')
        "\n        Set the ROI's rotation angle.\n        \n        =============== ==========================================================================\n        **Arguments**\n        angle           (float) The final ROI angle in degrees\n        center          (None | Point) Optional center point around which the ROI is rotated,\n                        expressed as [0-1, 0-1] over the size of the ROI.\n        centerLocal     (None | Point) Same as *center*, but the position is expressed in the\n                        local coordinate system of the ROI\n        snap            (bool) If True, the final ROI angle is snapped to the nearest increment\n                        (default is 15 degrees; see ROI.rotateSnapAngle)\n        update          (bool) See setPos()\n        finish          (bool) See setPos()\n        =============== ==========================================================================\n        "
        if update not in (True, False):
            raise TypeError('update argument must be bool')
        if snap is True:
            angle = round(angle / self.rotateSnapAngle) * self.rotateSnapAngle
        self.state['angle'] = angle
        tr = QtGui.QTransform()
        tr.rotate(angle)
        if center is not None:
            centerLocal = Point(center) * self.state['size']
        if centerLocal is not None:
            centerLocal = Point(centerLocal)
            cc = self.mapToParent(centerLocal) - (tr.map(centerLocal) + self.state['pos'])
            self.translate(cc, update=False)
        self.setTransform(tr)
        if update:
            self.stateChanged(finish=finish)

    def scale(self, s, center=None, centerLocal=None, snap=False, update=True, finish=True):
        if False:
            i = 10
            return i + 15
        '\n        Resize the ROI by scaling relative to *center*.\n        See setPos() for an explanation of the *update* and *finish* arguments.\n        '
        newSize = self.state['size'] * s
        self.setSize(newSize, center=center, centerLocal=centerLocal, snap=snap, update=update, finish=finish)

    def translate(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        '\n        Move the ROI to a new position.\n        Accepts either (x, y, snap) or ([x,y], snap) as arguments\n        If the ROI is bounded and the move would exceed boundaries, then the ROI\n        is moved to the nearest acceptable position instead.\n        \n        *snap* can be:\n        \n        =============== ==========================================================================\n        None (default)  use self.translateSnap and self.snapSize to determine whether/how to snap\n        False           do not snap\n        Point(w,h)      snap to rectangular grid with spacing (w,h)\n        True            snap using self.snapSize (and ignoring self.translateSnap)\n        =============== ==========================================================================\n           \n        Also accepts *update* and *finish* arguments (see setPos() for a description of these).\n        '
        if len(args) == 1:
            pt = args[0]
        else:
            pt = args
        newState = self.stateCopy()
        newState['pos'] = newState['pos'] + pt
        snap = kargs.get('snap', None)
        if snap is None:
            snap = self.translateSnap
        if snap is not False:
            newState['pos'] = self.getSnapPosition(newState['pos'], snap=snap)
        if self.maxBounds is not None:
            r = self.stateRect(newState)
            d = Point(0, 0)
            if self.maxBounds.left() > r.left():
                d[0] = self.maxBounds.left() - r.left()
            elif self.maxBounds.right() < r.right():
                d[0] = self.maxBounds.right() - r.right()
            if self.maxBounds.top() > r.top():
                d[1] = self.maxBounds.top() - r.top()
            elif self.maxBounds.bottom() < r.bottom():
                d[1] = self.maxBounds.bottom() - r.bottom()
            newState['pos'] += d
        update = kargs.get('update', True)
        finish = kargs.get('finish', True)
        self.setPos(newState['pos'], update=update, finish=finish)

    def rotate(self, angle, center=None, snap=False, update=True, finish=True):
        if False:
            return 10
        '\n        Rotate the ROI by *angle* degrees. \n        \n        =============== ==========================================================================\n        **Arguments**\n        angle           (float) The angle in degrees to rotate\n        center          (None | Point) Optional center point around which the ROI is rotated, in\n                        the local coordinate system of the ROI\n        snap            (bool) If True, the final ROI angle is snapped to the nearest increment\n                        (default is 15 degrees; see ROI.rotateSnapAngle)\n        update          (bool) See setPos()\n        finish          (bool) See setPos()\n        =============== ==========================================================================\n        '
        self.setAngle(self.angle() + angle, center=center, snap=snap, update=update, finish=finish)

    def handleMoveStarted(self):
        if False:
            print('Hello World!')
        self.preMoveState = self.getState()
        self.sigRegionChangeStarted.emit(self)

    def addTranslateHandle(self, pos, axes=None, item=None, name=None, index=None):
        if False:
            while True:
                i = 10
        "\n        Add a new translation handle to the ROI. Dragging the handle will move \n        the entire ROI without changing its angle or shape. \n        \n        Note that, by default, ROIs may be moved by dragging anywhere inside the\n        ROI. However, for larger ROIs it may be desirable to disable this and\n        instead provide one or more translation handles.\n        \n        =================== ====================================================\n        **Arguments**\n        pos                 (length-2 sequence) The position of the handle \n                            relative to the shape of the ROI. A value of (0,0)\n                            indicates the origin, whereas (1, 1) indicates the\n                            upper-right corner, regardless of the ROI's size.\n        item                The Handle instance to add. If None, a new handle\n                            will be created.\n        name                The name of this handle (optional). Handles are \n                            identified by name when calling \n                            getLocalHandlePositions and getSceneHandlePositions.\n        =================== ====================================================\n        "
        pos = Point(pos)
        return self.addHandle({'name': name, 'type': 't', 'pos': pos, 'item': item}, index=index)

    def addFreeHandle(self, pos=None, axes=None, item=None, name=None, index=None):
        if False:
            print('Hello World!')
        "\n        Add a new free handle to the ROI. Dragging free handles has no effect\n        on the position or shape of the ROI. \n        \n        =================== ====================================================\n        **Arguments**\n        pos                 (length-2 sequence) The position of the handle \n                            relative to the shape of the ROI. A value of (0,0)\n                            indicates the origin, whereas (1, 1) indicates the\n                            upper-right corner, regardless of the ROI's size.\n        item                The Handle instance to add. If None, a new handle\n                            will be created.\n        name                The name of this handle (optional). Handles are \n                            identified by name when calling \n                            getLocalHandlePositions and getSceneHandlePositions.\n        =================== ====================================================\n        "
        if pos is not None:
            pos = Point(pos)
        return self.addHandle({'name': name, 'type': 'f', 'pos': pos, 'item': item}, index=index)

    def addScaleHandle(self, pos, center, axes=None, item=None, name=None, lockAspect=False, index=None):
        if False:
            i = 10
            return i + 15
        "\n        Add a new scale handle to the ROI. Dragging a scale handle allows the\n        user to change the height and/or width of the ROI.\n        \n        =================== ====================================================\n        **Arguments**\n        pos                 (length-2 sequence) The position of the handle \n                            relative to the shape of the ROI. A value of (0,0)\n                            indicates the origin, whereas (1, 1) indicates the\n                            upper-right corner, regardless of the ROI's size.\n        center              (length-2 sequence) The center point around which \n                            scaling takes place. If the center point has the\n                            same x or y value as the handle position, then \n                            scaling will be disabled for that axis.\n        item                The Handle instance to add. If None, a new handle\n                            will be created.\n        name                The name of this handle (optional). Handles are \n                            identified by name when calling \n                            getLocalHandlePositions and getSceneHandlePositions.\n        =================== ====================================================\n        "
        pos = Point(pos)
        center = Point(center)
        info = {'name': name, 'type': 's', 'center': center, 'pos': pos, 'item': item, 'lockAspect': lockAspect}
        if pos.x() == center.x():
            info['xoff'] = True
        if pos.y() == center.y():
            info['yoff'] = True
        return self.addHandle(info, index=index)

    def addRotateHandle(self, pos, center, item=None, name=None, index=None):
        if False:
            i = 10
            return i + 15
        "\n        Add a new rotation handle to the ROI. Dragging a rotation handle allows \n        the user to change the angle of the ROI.\n        \n        =================== ====================================================\n        **Arguments**\n        pos                 (length-2 sequence) The position of the handle \n                            relative to the shape of the ROI. A value of (0,0)\n                            indicates the origin, whereas (1, 1) indicates the\n                            upper-right corner, regardless of the ROI's size.\n        center              (length-2 sequence) The center point around which \n                            rotation takes place.\n        item                The Handle instance to add. If None, a new handle\n                            will be created.\n        name                The name of this handle (optional). Handles are \n                            identified by name when calling \n                            getLocalHandlePositions and getSceneHandlePositions.\n        =================== ====================================================\n        "
        pos = Point(pos)
        center = Point(center)
        return self.addHandle({'name': name, 'type': 'r', 'center': center, 'pos': pos, 'item': item}, index=index)

    def addScaleRotateHandle(self, pos, center, item=None, name=None, index=None):
        if False:
            return 10
        "\n        Add a new scale+rotation handle to the ROI. When dragging a handle of \n        this type, the user can simultaneously rotate the ROI around an \n        arbitrary center point as well as scale the ROI by dragging the handle\n        toward or away from the center point.\n        \n        =================== ====================================================\n        **Arguments**\n        pos                 (length-2 sequence) The position of the handle \n                            relative to the shape of the ROI. A value of (0,0)\n                            indicates the origin, whereas (1, 1) indicates the\n                            upper-right corner, regardless of the ROI's size.\n        center              (length-2 sequence) The center point around which \n                            scaling and rotation take place.\n        item                The Handle instance to add. If None, a new handle\n                            will be created.\n        name                The name of this handle (optional). Handles are \n                            identified by name when calling \n                            getLocalHandlePositions and getSceneHandlePositions.\n        =================== ====================================================\n        "
        pos = Point(pos)
        center = Point(center)
        if pos[0] == center[0] and pos[1] == center[1]:
            raise Exception('Scale/rotate handles cannot be at their center point.')
        return self.addHandle({'name': name, 'type': 'sr', 'center': center, 'pos': pos, 'item': item}, index=index)

    def addRotateFreeHandle(self, pos, center, axes=None, item=None, name=None, index=None):
        if False:
            print('Hello World!')
        "\n        Add a new rotation+free handle to the ROI. When dragging a handle of \n        this type, the user can rotate the ROI around an \n        arbitrary center point, while moving toward or away from the center \n        point has no effect on the shape of the ROI.\n        \n        =================== ====================================================\n        **Arguments**\n        pos                 (length-2 sequence) The position of the handle \n                            relative to the shape of the ROI. A value of (0,0)\n                            indicates the origin, whereas (1, 1) indicates the\n                            upper-right corner, regardless of the ROI's size.\n        center              (length-2 sequence) The center point around which \n                            rotation takes place.\n        item                The Handle instance to add. If None, a new handle\n                            will be created.\n        name                The name of this handle (optional). Handles are \n                            identified by name when calling \n                            getLocalHandlePositions and getSceneHandlePositions.\n        =================== ====================================================\n        "
        pos = Point(pos)
        center = Point(center)
        return self.addHandle({'name': name, 'type': 'rf', 'center': center, 'pos': pos, 'item': item}, index=index)

    def addHandle(self, info, index=None):
        if False:
            print('Hello World!')
        if 'item' not in info or info['item'] is None:
            h = Handle(self.handleSize, typ=info['type'], pen=self.handlePen, hoverPen=self.handleHoverPen, parent=self)
            info['item'] = h
        else:
            h = info['item']
            if info['pos'] is None:
                info['pos'] = h.pos()
        h.setPos(info['pos'] * self.state['size'])
        h.connectROI(self)
        if index is None:
            self.handles.append(info)
        else:
            self.handles.insert(index, info)
        h.setZValue(self.zValue() + 1)
        self.stateChanged()
        return h

    def indexOfHandle(self, handle):
        if False:
            return 10
        "\n        Return the index of *handle* in the list of this ROI's handles.\n        "
        if isinstance(handle, Handle):
            index = [i for (i, info) in enumerate(self.handles) if info['item'] is handle]
            if len(index) == 0:
                raise Exception('Cannot return handle index; not attached to this ROI')
            return index[0]
        else:
            return handle

    def removeHandle(self, handle):
        if False:
            return 10
        'Remove a handle from this ROI. Argument may be either a Handle \n        instance or the integer index of the handle.'
        index = self.indexOfHandle(handle)
        handle = self.handles[index]['item']
        self.handles.pop(index)
        handle.disconnectROI(self)
        if len(handle.rois) == 0 and self.scene() is not None:
            self.scene().removeItem(handle)
        self.stateChanged()

    def replaceHandle(self, oldHandle, newHandle):
        if False:
            print('Hello World!')
        'Replace one handle in the ROI for another. This is useful when \n        connecting multiple ROIs together.\n        \n        *oldHandle* may be a Handle instance or the index of a handle to be\n        replaced.'
        index = self.indexOfHandle(oldHandle)
        info = self.handles[index]
        self.removeHandle(index)
        info['item'] = newHandle
        info['pos'] = newHandle.pos()
        self.addHandle(info, index=index)

    def checkRemoveHandle(self, handle):
        if False:
            return 10
        return True

    def getLocalHandlePositions(self, index=None):
        if False:
            i = 10
            return i + 15
        "Returns the position of handles in the ROI's coordinate system.\n        \n        The format returned is a list of (name, pos) tuples.\n        "
        if index is None:
            positions = []
            for h in self.handles:
                positions.append((h['name'], h['pos']))
            return positions
        else:
            return (self.handles[index]['name'], self.handles[index]['pos'])

    def getSceneHandlePositions(self, index=None):
        if False:
            while True:
                i = 10
        'Returns the position of handles in the scene coordinate system.\n        \n        The format returned is a list of (name, pos) tuples.\n        '
        if index is None:
            positions = []
            for h in self.handles:
                positions.append((h['name'], h['item'].scenePos()))
            return positions
        else:
            return (self.handles[index]['name'], self.handles[index]['item'].scenePos())

    def getHandles(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a list of this ROI's Handles.\n        "
        return [h['item'] for h in self.handles]

    def mapSceneToParent(self, pt):
        if False:
            print('Hello World!')
        return self.mapToParent(self.mapFromScene(pt))

    def setSelected(self, s):
        if False:
            i = 10
            return i + 15
        QtWidgets.QGraphicsItem.setSelected(self, s)
        if s:
            for h in self.handles:
                h['item'].show()
        else:
            for h in self.handles:
                h['item'].hide()

    def hoverEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        hover = False
        if not ev.isExit():
            if self.translatable and ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
                hover = True
            for btn in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton, QtCore.Qt.MouseButton.MiddleButton]:
                if self.acceptedMouseButtons() & btn and ev.acceptClicks(btn):
                    hover = True
            if self.contextMenuEnabled():
                ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
        if hover:
            self.setMouseHover(True)
            ev.acceptClicks(QtCore.Qt.MouseButton.LeftButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.RightButton)
            ev.acceptClicks(QtCore.Qt.MouseButton.MiddleButton)
            self.sigHoverEvent.emit(self)
        else:
            self.setMouseHover(False)

    def setMouseHover(self, hover):
        if False:
            for i in range(10):
                print('nop')
        if self.mouseHovering == hover:
            return
        self.mouseHovering = hover
        self._updateHoverColor()

    def _updateHoverColor(self):
        if False:
            return 10
        pen = self._makePen()
        if self.currentPen != pen:
            self.currentPen = pen
            self.update()

    def _makePen(self):
        if False:
            print('Hello World!')
        if self.mouseHovering:
            return self.hoverPen
        else:
            return self.pen

    def contextMenuEnabled(self):
        if False:
            while True:
                i = 10
        return self.removable

    def raiseContextMenu(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if not self.contextMenuEnabled():
            return
        menu = self.getMenu()
        menu = self.scene().addParentContextMenus(self, menu, ev)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def getMenu(self):
        if False:
            for i in range(10):
                print('nop')
        if self.menu is None:
            self.menu = QtWidgets.QMenu()
            self.menu.setTitle(translate('ROI', 'ROI'))
            remAct = QtGui.QAction(translate('ROI', 'Remove ROI'), self.menu)
            remAct.triggered.connect(self.removeClicked)
            self.menu.addAction(remAct)
            self.menu.remAct = remAct
        self.menu.setEnabled(self.contextMenuEnabled())
        return self.menu

    def removeClicked(self):
        if False:
            i = 10
            return i + 15
        QtCore.QTimer.singleShot(0, self._emitRemoveRequest)

    def _emitRemoveRequest(self):
        if False:
            return 10
        self.sigRemoveRequested.emit(self)

    def mouseDragEvent(self, ev):
        if False:
            i = 10
            return i + 15
        self.mouseDragHandler.mouseDragEvent(ev)

    def mouseClickEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.isMoving:
            ev.accept()
            self.cancelMove()
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.contextMenuEnabled():
            self.raiseContextMenu(ev)
            ev.accept()
        elif self.acceptedMouseButtons() & ev.button():
            ev.accept()
            self.sigClicked.emit(self, ev)
        else:
            ev.ignore()

    def _moveStarted(self):
        if False:
            print('Hello World!')
        self.isMoving = True
        self.preMoveState = self.getState()
        self.sigRegionChangeStarted.emit(self)

    def _moveFinished(self):
        if False:
            for i in range(10):
                print('nop')
        if self.isMoving:
            self.stateChangeFinished()
        self.isMoving = False

    def cancelMove(self):
        if False:
            return 10
        self.isMoving = False
        self.setState(self.preMoveState)

    def checkPointMove(self, handle, pos, modifiers):
        if False:
            i = 10
            return i + 15
        'When handles move, they must ask the ROI if the move is acceptable.\n        By default, this always returns True. Subclasses may wish override.\n        '
        return True

    def movePoint(self, handle, pos, modifiers=None, finish=True, coords='parent'):
        if False:
            for i in range(10):
                print('nop')
        if modifiers is None:
            modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        newState = self.stateCopy()
        index = self.indexOfHandle(handle)
        h = self.handles[index]
        p0 = self.mapToParent(h['pos'] * self.state['size'])
        p1 = Point(pos)
        if coords == 'parent':
            pass
        elif coords == 'scene':
            p1 = self.mapSceneToParent(p1)
        else:
            raise Exception("New point location must be given in either 'parent' or 'scene' coordinates.")
        if 'center' in h:
            c = h['center']
            cs = c * self.state['size']
            lp0 = self.mapFromParent(p0) - cs
            lp1 = self.mapFromParent(p1) - cs
        if h['type'] == 't':
            snap = True if modifiers & QtCore.Qt.KeyboardModifier.ControlModifier else None
            self.translate(p1 - p0, snap=snap, update=False)
        elif h['type'] == 'f':
            newPos = self.mapFromParent(p1)
            h['item'].setPos(newPos)
            h['pos'] = newPos
            self.freeHandleMoved = True
        elif h['type'] == 's':
            if h['center'][0] == h['pos'][0]:
                lp1[0] = 0
            if h['center'][1] == h['pos'][1]:
                lp1[1] = 0
            if self.scaleSnap or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                lp1[0] = round(lp1[0] / self.scaleSnapSize) * self.scaleSnapSize
                lp1[1] = round(lp1[1] / self.scaleSnapSize) * self.scaleSnapSize
            if h['lockAspect'] or modifiers & QtCore.Qt.KeyboardModifier.AltModifier:
                lp1 = lp1.proj(lp0)
            hs = h['pos'] - c
            if hs[0] == 0:
                hs[0] = 1
            if hs[1] == 0:
                hs[1] = 1
            newSize = lp1 / hs
            if newSize[0] == 0:
                newSize[0] = newState['size'][0]
            if newSize[1] == 0:
                newSize[1] = newState['size'][1]
            if not self.invertible:
                if newSize[0] < 0:
                    newSize[0] = newState['size'][0]
                if newSize[1] < 0:
                    newSize[1] = newState['size'][1]
            if self.aspectLocked:
                newSize[0] = newSize[1]
            s0 = c * self.state['size']
            s1 = c * newSize
            cc = self.mapToParent(s0 - s1) - self.mapToParent(Point(0, 0))
            newState['size'] = newSize
            newState['pos'] = newState['pos'] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return
            self.setPos(newState['pos'], update=False)
            self.setSize(newState['size'], update=False)
        elif h['type'] in ['r', 'rf']:
            if h['type'] == 'rf':
                self.freeHandleMoved = True
            if not self.rotatable:
                return
            try:
                if lp1.length() == 0 or lp0.length() == 0:
                    return
            except OverflowError:
                return
            ang = newState['angle'] - lp0.angle(lp1)
            if ang is None:
                return
            if self.rotateSnap or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                ang = round(ang / self.rotateSnapAngle) * self.rotateSnapAngle
            tr = QtGui.QTransform()
            tr.rotate(ang)
            cc = self.mapToParent(cs) - (tr.map(cs) + self.state['pos'])
            newState['angle'] = ang
            newState['pos'] = newState['pos'] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return
            self.setPos(newState['pos'], update=False)
            self.setAngle(ang, update=False)
            if h['type'] == 'rf':
                h['item'].setPos(self.mapFromScene(p1))
                h['pos'] = self.mapFromParent(p1)
        elif h['type'] == 'sr':
            try:
                if lp1.length() == 0 or lp0.length() == 0:
                    return
            except OverflowError:
                return
            ang = newState['angle'] - lp0.angle(lp1)
            if ang is None:
                return
            if self.rotateSnap or modifiers & QtCore.Qt.KeyboardModifier.ControlModifier:
                ang = round(ang / self.rotateSnapAngle) * self.rotateSnapAngle
            if self.aspectLocked or h['center'][0] != h['pos'][0]:
                newState['size'][0] = self.state['size'][0] * lp1.length() / lp0.length()
                if self.scaleSnap:
                    newState['size'][0] = round(newState['size'][0] / self.snapSize) * self.snapSize
            if self.aspectLocked or h['center'][1] != h['pos'][1]:
                newState['size'][1] = self.state['size'][1] * lp1.length() / lp0.length()
                if self.scaleSnap:
                    newState['size'][1] = round(newState['size'][1] / self.snapSize) * self.snapSize
            if newState['size'][0] == 0:
                newState['size'][0] = 1
            if newState['size'][1] == 0:
                newState['size'][1] = 1
            c1 = c * newState['size']
            tr = QtGui.QTransform()
            tr.rotate(ang)
            cc = self.mapToParent(cs) - (tr.map(c1) + self.state['pos'])
            newState['angle'] = ang
            newState['pos'] = newState['pos'] + cc
            if self.maxBounds is not None:
                r = self.stateRect(newState)
                if not self.maxBounds.contains(r):
                    return
            self.setState(newState, update=False)
        self.stateChanged(finish=finish)

    def stateChanged(self, finish=True):
        if False:
            while True:
                i = 10
        'Process changes to the state of the ROI.\n        If there are any changes, then the positions of handles are updated accordingly\n        and sigRegionChanged is emitted. If finish is True, then \n        sigRegionChangeFinished will also be emitted.'
        changed = False
        if self.lastState is None:
            changed = True
        else:
            state = self.getState()
            for k in list(state.keys()):
                if state[k] != self.lastState[k]:
                    changed = True
        self.prepareGeometryChange()
        if changed:
            for h in self.handles:
                if h['item'] in self.childItems():
                    h['item'].setPos(h['pos'] * self.state['size'])
            self.update()
            self.sigRegionChanged.emit(self)
        elif self.freeHandleMoved:
            self.sigRegionChanged.emit(self)
        self.freeHandleMoved = False
        self.lastState = self.getState()
        if finish:
            self.stateChangeFinished()
            self.informViewBoundsChanged()

    def stateChangeFinished(self):
        if False:
            i = 10
            return i + 15
        self.sigRegionChangeFinished.emit(self)

    def stateRect(self, state):
        if False:
            return 10
        r = QtCore.QRectF(0, 0, state['size'][0], state['size'][1])
        tr = QtGui.QTransform()
        tr.rotate(-state['angle'])
        r = tr.mapRect(r)
        return r.adjusted(state['pos'][0], state['pos'][1], state['pos'][0], state['pos'][1])

    def getSnapPosition(self, pos, snap=None):
        if False:
            i = 10
            return i + 15
        if snap is None or snap is True:
            if self.snapSize is None:
                return pos
            snap = Point(self.snapSize, self.snapSize)
        return Point(round(pos[0] / snap[0]) * snap[0], round(pos[1] / snap[1]) * snap[1])

    def boundingRect(self):
        if False:
            print('Hello World!')
        return QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()

    def paint(self, p, opt, widget):
        if False:
            print('Hello World!')
        r = QtCore.QRectF(0, 0, self.state['size'][0], self.state['size'][1]).normalized()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.translate(r.left(), r.top())
        p.scale(r.width(), r.height())
        p.drawRect(0, 0, 1, 1)

    def getArraySlice(self, data, img, axes=(0, 1), returnSlice=True):
        if False:
            for i in range(10):
                print('nop')
        'Return a tuple of slice objects that can be used to slice the region\n        from *data* that is covered by the bounding rectangle of this ROI.\n        Also returns the transform that maps the ROI into data coordinates.\n        \n        If returnSlice is set to False, the function returns a pair of tuples with the values that would have \n        been used to generate the slice objects. ((ax0Start, ax0Stop), (ax1Start, ax1Stop))\n        \n        If the slice cannot be computed (usually because the scene/transforms are not properly\n        constructed yet), then the method returns None.\n        '
        dShape = (data.shape[axes[0]], data.shape[axes[1]])
        try:
            tr = self.sceneTransform() * fn.invertQTransform(img.sceneTransform())
        except np.linalg.linalg.LinAlgError:
            return None
        axisOrder = img.axisOrder
        if axisOrder == 'row-major':
            tr.scale(float(dShape[1]) / img.width(), float(dShape[0]) / img.height())
        else:
            tr.scale(float(dShape[0]) / img.width(), float(dShape[1]) / img.height())
        dataBounds = tr.mapRect(self.boundingRect())
        if axisOrder == 'row-major':
            intBounds = dataBounds.intersected(QtCore.QRectF(0, 0, dShape[1], dShape[0]))
        else:
            intBounds = dataBounds.intersected(QtCore.QRectF(0, 0, dShape[0], dShape[1]))
        bounds = ((int(min(intBounds.left(), intBounds.right())), int(1 + max(intBounds.left(), intBounds.right()))), (int(min(intBounds.bottom(), intBounds.top())), int(1 + max(intBounds.bottom(), intBounds.top()))))
        if axisOrder == 'row-major':
            bounds = bounds[::-1]
        if returnSlice:
            sl = [slice(None)] * data.ndim
            sl[axes[0]] = slice(*bounds[0])
            sl[axes[1]] = slice(*bounds[1])
            return (tuple(sl), tr)
        else:
            return (bounds, tr)

    def getArrayRegion(self, data, img, axes=(0, 1), returnMappedCoords=False, **kwds):
        if False:
            for i in range(10):
                print('nop')
        "Use the position and orientation of this ROI relative to an imageItem\n        to pull a slice from an array.\n\n        =================== ====================================================\n        **Arguments**\n        data                The array to slice from. Note that this array does\n                            *not* have to be the same data that is represented\n                            in *img*.\n        img                 (ImageItem or other suitable QGraphicsItem)\n                            Used to determine the relationship between the \n                            ROI and the boundaries of *data*.\n        axes                (length-2 tuple) Specifies the axes in *data* that\n                            correspond to the (x, y) axes of *img*. If the\n                            image's axis order is set to\n                            'row-major', then the axes are instead specified in\n                            (y, x) order.\n        returnMappedCoords  (bool) If True, the array slice is returned along\n                            with a corresponding array of coordinates that were\n                            used to extract data from the original array.\n        \\**kwds             All keyword arguments are passed to \n                            :func:`affineSlice <pyqtgraph.affineSlice>`.\n        =================== ====================================================\n        \n        This method uses :func:`affineSlice <pyqtgraph.affineSlice>` to generate\n        the slice from *data* and uses :func:`getAffineSliceParams <pyqtgraph.ROI.getAffineSliceParams>`\n        to determine the parameters to pass to :func:`affineSlice <pyqtgraph.affineSlice>`.\n        \n        If *returnMappedCoords* is True, then the method returns a tuple (result, coords) \n        such that coords is the set of coordinates used to interpolate values from the original\n        data, mapped into the parent coordinate system of the image. This is useful, when slicing\n        data from images that have been transformed, for determining the location of each value\n        in the sliced data.\n        \n        All extra keyword arguments are passed to :func:`affineSlice <pyqtgraph.affineSlice>`.\n        "
        fromBR = kwds.pop('fromBoundingRect', False)
        (_shape, _vectors, _origin) = self.getAffineSliceParams(data, img, axes, fromBoundingRect=fromBR)
        shape = kwds.pop('shape', _shape)
        vectors = kwds.pop('vectors', _vectors)
        origin = kwds.pop('origin', _origin)
        if not returnMappedCoords:
            rgn = fn.affineSlice(data, shape=shape, vectors=vectors, origin=origin, axes=axes, **kwds)
            return rgn
        else:
            kwds['returnCoords'] = True
            (result, coords) = fn.affineSlice(data, shape=shape, vectors=vectors, origin=origin, axes=axes, **kwds)
            mapped = fn.transformCoordinates(img.transform(), coords)
            return (result, mapped)

    def _getArrayRegionForArbitraryShape(self, data, img, axes=(0, 1), returnMappedCoords=False, **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the result of :meth:`~pyqtgraph.ROI.getArrayRegion`, masked by\n        the shape of the ROI. Values outside the ROI shape are set to 0.\n\n        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the\n        arguments.\n        '
        if returnMappedCoords:
            (sliced, mappedCoords) = ROI.getArrayRegion(self, data, img, axes, returnMappedCoords, fromBoundingRect=True, **kwds)
        else:
            sliced = ROI.getArrayRegion(self, data, img, axes, returnMappedCoords, fromBoundingRect=True, **kwds)
        if img.axisOrder == 'col-major':
            mask = self.renderShapeMask(sliced.shape[axes[0]], sliced.shape[axes[1]])
        else:
            mask = self.renderShapeMask(sliced.shape[axes[1]], sliced.shape[axes[0]])
            mask = mask.T
        shape = [1] * data.ndim
        shape[axes[0]] = sliced.shape[axes[0]]
        shape[axes[1]] = sliced.shape[axes[1]]
        mask = mask.reshape(shape)
        if returnMappedCoords:
            return (sliced * mask, mappedCoords)
        else:
            return sliced * mask

    def getAffineSliceParams(self, data, img, axes=(0, 1), fromBoundingRect=False):
        if False:
            i = 10
            return i + 15
        "\n        Returns the parameters needed to use :func:`affineSlice <pyqtgraph.affineSlice>`\n        (shape, vectors, origin) to extract a subset of *data* using this ROI \n        and *img* to specify the subset.\n        \n        If *fromBoundingRect* is True, then the ROI's bounding rectangle is used\n        rather than the shape of the ROI.\n        \n        See :func:`getArrayRegion <pyqtgraph.ROI.getArrayRegion>` for more information.\n        "
        if self.scene() is not img.scene():
            raise Exception('ROI and target item must be members of the same scene.')
        origin = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 0)))
        vx = img.mapToData(self.mapToItem(img, QtCore.QPointF(1, 0))) - origin
        vy = img.mapToData(self.mapToItem(img, QtCore.QPointF(0, 1))) - origin
        lvx = hypot(vx.x(), vx.y())
        lvy = hypot(vy.x(), vy.y())
        sx = 1.0 / lvx
        sy = 1.0 / lvy
        vectors = ((vx.x() * sx, vx.y() * sx), (vy.x() * sy, vy.y() * sy))
        if fromBoundingRect is True:
            shape = (self.boundingRect().width(), self.boundingRect().height())
            origin = img.mapToData(self.mapToItem(img, self.boundingRect().topLeft()))
            origin = (origin.x(), origin.y())
        else:
            shape = self.state['size']
            origin = (origin.x(), origin.y())
        shape = [abs(shape[0] / sx), abs(shape[1] / sy)]
        if img.axisOrder == 'row-major':
            vectors = vectors[::-1]
            shape = shape[::-1]
        return (shape, vectors, origin)

    def renderShapeMask(self, width, height):
        if False:
            for i in range(10):
                print('nop')
        'Return an array of 0.0-1.0 into which the shape of the item has been drawn.\n        \n        This can be used to mask array selections.\n        '
        if width == 0 or height == 0:
            return np.empty((width, height), dtype=float)
        im = QtGui.QImage(width, height, QtGui.QImage.Format.Format_ARGB32)
        im.fill(QtCore.Qt.GlobalColor.transparent)
        p = QtGui.QPainter(im)
        p.setPen(fn.mkPen(None))
        p.setBrush(fn.mkBrush('w'))
        shape = self.shape()
        bounds = shape.boundingRect()
        p.scale(im.width() / bounds.width(), im.height() / bounds.height())
        p.translate(-bounds.topLeft())
        p.drawPath(shape)
        p.end()
        cidx = 0 if sys.byteorder == 'little' else 3
        mask = fn.ndarray_from_qimage(im)[..., cidx].T
        return mask.astype(float) / 255

    def getGlobalTransform(self, relativeTo=None):
        if False:
            return 10
        "Return global transformation (rotation angle+translation) required to move \n        from relative state to current state. If relative state isn't specified,\n        then we use the state of the ROI when mouse is pressed."
        if relativeTo is None:
            relativeTo = self.preMoveState
        st = self.getState()
        relativeTo['scale'] = relativeTo['size']
        st['scale'] = st['size']
        t1 = SRTTransform(relativeTo)
        t2 = SRTTransform(st)
        return t2 / t1

    def applyGlobalTransform(self, tr):
        if False:
            for i in range(10):
                print('nop')
        st = self.getState()
        st['scale'] = st['size']
        st = SRTTransform(st)
        st = (st * tr).saveState()
        st['size'] = st['scale']
        self.setState(st)

class Handle(UIGraphicsItem):
    """
    Handle represents a single user-interactable point attached to an ROI. They
    are usually created by a call to one of the ROI.add___Handle() methods.
    
    Handles are represented as a square, diamond, or circle, and are drawn with 
    fixed pixel size regardless of the scaling of the view they are displayed in.
    
    Handles may be dragged to change the position, size, orientation, or other
    properties of the ROI they are attached to.
    """
    types = {'t': (4, np.pi / 4), 'f': (4, np.pi / 4), 's': (4, 0), 'r': (12, 0), 'sr': (12, 0), 'rf': (12, 0)}
    sigClicked = QtCore.Signal(object, object)
    sigRemoveRequested = QtCore.Signal(object)

    def __init__(self, radius, typ=None, pen=(200, 200, 220), hoverPen=(255, 255, 0), parent=None, deletable=False):
        if False:
            for i in range(10):
                print('nop')
        self.rois = []
        self.radius = radius
        self.typ = typ
        self.pen = fn.mkPen(pen)
        self.hoverPen = fn.mkPen(hoverPen)
        self.currentPen = self.pen
        self.pen.setWidth(0)
        self.pen.setCosmetic(True)
        self.isMoving = False
        (self.sides, self.startAng) = self.types[typ]
        self.buildPath()
        self._shape = None
        self.menu = self.buildMenu()
        UIGraphicsItem.__init__(self, parent=parent)
        self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.NoButton)
        self.deletable = deletable
        if deletable:
            self.setAcceptedMouseButtons(QtCore.Qt.MouseButton.RightButton)
        self.setZValue(11)

    def connectROI(self, roi):
        if False:
            i = 10
            return i + 15
        self.rois.append(roi)

    def disconnectROI(self, roi):
        if False:
            print('Hello World!')
        self.rois.remove(roi)

    def setDeletable(self, b):
        if False:
            for i in range(10):
                print('nop')
        self.deletable = b
        if b:
            self.setAcceptedMouseButtons(self.acceptedMouseButtons() | QtCore.Qt.MouseButton.RightButton)
        else:
            self.setAcceptedMouseButtons(self.acceptedMouseButtons() & ~QtCore.Qt.MouseButton.RightButton)

    def removeClicked(self):
        if False:
            while True:
                i = 10
        self.sigRemoveRequested.emit(self)

    def hoverEvent(self, ev):
        if False:
            print('Hello World!')
        hover = False
        if not ev.isExit():
            if ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton):
                hover = True
            for btn in [QtCore.Qt.MouseButton.LeftButton, QtCore.Qt.MouseButton.RightButton, QtCore.Qt.MouseButton.MiddleButton]:
                if self.acceptedMouseButtons() & btn and ev.acceptClicks(btn):
                    hover = True
        if hover:
            self.currentPen = self.hoverPen
        else:
            self.currentPen = self.pen
        self.update()

    def mouseClickEvent(self, ev):
        if False:
            print('Hello World!')
        if ev.button() == QtCore.Qt.MouseButton.RightButton and self.isMoving:
            self.isMoving = False
            self.movePoint(self.startPos, finish=True)
            ev.accept()
        elif self.acceptedMouseButtons() & ev.button():
            ev.accept()
            if ev.button() == QtCore.Qt.MouseButton.RightButton and self.deletable:
                self.raiseContextMenu(ev)
            self.sigClicked.emit(self, ev)
        else:
            ev.ignore()

    def buildMenu(self):
        if False:
            for i in range(10):
                print('nop')
        menu = QtWidgets.QMenu()
        menu.setTitle(translate('ROI', 'Handle'))
        self.removeAction = menu.addAction(translate('ROI', 'Remove handle'), self.removeClicked)
        return menu

    def getMenu(self):
        if False:
            while True:
                i = 10
        return self.menu

    def raiseContextMenu(self, ev):
        if False:
            return 10
        menu = self.scene().addParentContextMenus(self, self.getMenu(), ev)
        removeAllowed = all((r.checkRemoveHandle(self) for r in self.rois))
        self.removeAction.setEnabled(removeAllowed)
        pos = ev.screenPos()
        menu.popup(QtCore.QPoint(int(pos.x()), int(pos.y())))

    def mouseDragEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        ev.accept()
        if ev.isFinish():
            if self.isMoving:
                for r in self.rois:
                    r.stateChangeFinished()
            self.isMoving = False
            self.currentPen = self.pen
            self.update()
        elif ev.isStart():
            for r in self.rois:
                r.handleMoveStarted()
            self.isMoving = True
            self.startPos = self.scenePos()
            self.cursorOffset = self.scenePos() - ev.buttonDownScenePos()
            self.currentPen = self.hoverPen
        if self.isMoving:
            pos = ev.scenePos() + self.cursorOffset
            self.currentPen = self.hoverPen
            self.movePoint(pos, ev.modifiers(), finish=False)

    def movePoint(self, pos, modifiers=None, finish=True):
        if False:
            while True:
                i = 10
        if modifiers is None:
            modifiers = QtCore.Qt.KeyboardModifier.NoModifier
        for r in self.rois:
            if not r.checkPointMove(self, pos, modifiers):
                return
        for r in self.rois:
            r.movePoint(self, pos, modifiers, finish=finish, coords='scene')

    def buildPath(self):
        if False:
            while True:
                i = 10
        size = self.radius
        self.path = QtGui.QPainterPath()
        ang = self.startAng
        dt = 2 * np.pi / self.sides
        for i in range(0, self.sides + 1):
            x = size * cos(ang)
            y = size * sin(ang)
            ang += dt
            if i == 0:
                self.path.moveTo(x, y)
            else:
                self.path.lineTo(x, y)

    def paint(self, p, opt, widget):
        if False:
            while True:
                i = 10
        p.setRenderHints(p.RenderHint.Antialiasing, True)
        p.setPen(self.currentPen)
        p.drawPath(self.shape())

    def shape(self):
        if False:
            i = 10
            return i + 15
        if self._shape is None:
            s = self.generateShape()
            if s is None:
                return self.path
            self._shape = s
            self.prepareGeometryChange()
        return self._shape

    def boundingRect(self):
        if False:
            i = 10
            return i + 15
        s1 = self.shape()
        return self.shape().boundingRect()

    def generateShape(self):
        if False:
            while True:
                i = 10
        dt = self.deviceTransform()
        if dt is None:
            self._shape = self.path
            return None
        v = dt.map(QtCore.QPointF(1, 0)) - dt.map(QtCore.QPointF(0, 0))
        va = atan2(v.y(), v.x())
        dti = fn.invertQTransform(dt)
        devPos = dt.map(QtCore.QPointF(0, 0))
        tr = QtGui.QTransform()
        tr.translate(devPos.x(), devPos.y())
        tr.rotateRadians(va)
        return dti.map(tr.map(self.path))

    def viewTransformChanged(self):
        if False:
            print('Hello World!')
        GraphicsObject.viewTransformChanged(self)
        self._shape = None
        self.update()

class MouseDragHandler(object):
    """Implements default mouse drag behavior for ROI (not for ROI handles).
    """

    def __init__(self, roi):
        if False:
            i = 10
            return i + 15
        self.roi = roi
        self.dragMode = None
        self.startState = None
        self.snapModifier = QtCore.Qt.KeyboardModifier.ControlModifier
        self.translateModifier = QtCore.Qt.KeyboardModifier.NoModifier
        self.rotateModifier = QtCore.Qt.KeyboardModifier.AltModifier
        self.scaleModifier = QtCore.Qt.KeyboardModifier.ShiftModifier
        self.rotateSpeed = 0.5
        self.scaleSpeed = 1.01

    def mouseDragEvent(self, ev):
        if False:
            i = 10
            return i + 15
        roi = self.roi
        if ev.isStart():
            if ev.button() == QtCore.Qt.MouseButton.LeftButton:
                roi.setSelected(True)
                mods = ev.modifiers()
                try:
                    mods &= ~self.snapModifier
                except ValueError:
                    if mods & self.snapModifier:
                        mods ^= self.snapModifier
                if roi.translatable and mods == self.translateModifier:
                    self.dragMode = 'translate'
                elif roi.rotatable and mods == self.rotateModifier:
                    self.dragMode = 'rotate'
                elif roi.resizable and mods == self.scaleModifier:
                    self.dragMode = 'scale'
                else:
                    self.dragMode = None
                if self.dragMode is not None:
                    roi._moveStarted()
                    self.startPos = roi.mapToParent(ev.buttonDownPos())
                    self.startState = roi.saveState()
                    self.cursorOffset = roi.pos() - self.startPos
                    ev.accept()
                else:
                    ev.ignore()
            else:
                self.dragMode = None
                ev.ignore()
        if ev.isFinish() and self.dragMode is not None:
            roi._moveFinished()
            return
        if not roi.isMoving or self.dragMode is None:
            return
        snap = True if ev.modifiers() & self.snapModifier else None
        pos = roi.mapToParent(ev.pos())
        if self.dragMode == 'translate':
            newPos = pos + self.cursorOffset
            roi.translate(newPos - roi.pos(), snap=snap, finish=False)
        elif self.dragMode == 'rotate':
            diff = self.rotateSpeed * (ev.scenePos() - ev.buttonDownScenePos()).x()
            angle = self.startState['angle'] - diff
            roi.setAngle(angle, centerLocal=ev.buttonDownPos(), snap=snap, finish=False)
        elif self.dragMode == 'scale':
            diff = self.scaleSpeed ** (-(ev.scenePos() - ev.buttonDownScenePos()).y())
            roi.setSize(Point(self.startState['size']) * diff, centerLocal=ev.buttonDownPos(), snap=snap, finish=False)

class TestROI(ROI):

    def __init__(self, pos, size, **args):
        if False:
            i = 10
            return i + 15
        ROI.__init__(self, pos, size, **args)
        self.addTranslateHandle([0.5, 0.5])
        self.addScaleHandle([1, 1], [0, 0])
        self.addScaleHandle([0, 0], [1, 1])
        self.addScaleRotateHandle([1, 0.5], [0.5, 0.5])
        self.addScaleHandle([0.5, 1], [0.5, 0.5])
        self.addRotateHandle([1, 0], [0, 0])
        self.addRotateHandle([0, 1], [1, 1])

class RectROI(ROI):
    """
    Rectangular ROI subclass with a single scale handle at the top-right corner.

    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI origin.
                   See ROI().
    size           (length-2 sequence) The size of the ROI. See ROI().
    centered       (bool) If True, scale handles affect the ROI relative to its
                   center, rather than its origin.
    sideScalers    (bool) If True, extra scale handles are added at the top and 
                   right edges.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, pos, size, centered=False, sideScalers=False, **args):
        if False:
            i = 10
            return i + 15
        ROI.__init__(self, pos, size, **args)
        if centered:
            center = [0.5, 0.5]
        else:
            center = [0, 0]
        self.addScaleHandle([1, 1], center)
        if sideScalers:
            self.addScaleHandle([1, 0.5], [center[0], 0.5])
            self.addScaleHandle([0.5, 1], [0.5, center[1]])

class LineROI(ROI):
    """
    Rectangular ROI subclass with scale-rotate handles on either side. This
    allows the ROI to be positioned as if moving the ends of a line segment.
    A third handle controls the width of the ROI orthogonal to its "line" axis.
    
    ============== =============================================================
    **Arguments**
    pos1           (length-2 sequence) The position of the center of the ROI's
                   left edge.
    pos2           (length-2 sequence) The position of the center of the ROI's
                   right edge.
    width          (float) The width of the ROI.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, pos1, pos2, width, **args):
        if False:
            i = 10
            return i + 15
        pos1 = Point(pos1)
        pos2 = Point(pos2)
        d = pos2 - pos1
        l = d.length()
        ra = d.angle(Point(1, 0), units='radians')
        c = Point(width / 2.0 * sin(ra), -width / 2.0 * cos(ra))
        pos1 = pos1 + c
        ROI.__init__(self, pos1, size=Point(l, width), angle=degrees(ra), **args)
        self.addScaleRotateHandle([0, 0.5], [1, 0.5])
        self.addScaleRotateHandle([1, 0.5], [0, 0.5])
        self.addScaleHandle([0.5, 1], [0.5, 0.5])

class MultiRectROI(QtWidgets.QGraphicsObject):
    """
    Chain of rectangular ROIs connected by handles.

    This is generally used to mark a curved path through
    an image similarly to PolyLineROI. It differs in that each segment
    of the chain is rectangular instead of linear and thus has width.
    
    ============== =============================================================
    **Arguments**
    points         (list of length-2 sequences) The list of points in the path.
    width          (float) The width of the ROIs orthogonal to the path.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """
    sigRegionChangeFinished = QtCore.Signal(object)
    sigRegionChangeStarted = QtCore.Signal(object)
    sigRegionChanged = QtCore.Signal(object)

    def __init__(self, points, width, pen=None, **args):
        if False:
            while True:
                i = 10
        QtWidgets.QGraphicsObject.__init__(self)
        self.pen = pen
        self.roiArgs = args
        self.lines = []
        if len(points) < 2:
            raise Exception('Must start with at least 2 points')
        self.addSegment(points[1], connectTo=points[0], scaleHandle=True)
        for p in points[2:]:
            self.addSegment(p)

    def paint(self, *args):
        if False:
            return 10
        pass

    def boundingRect(self):
        if False:
            for i in range(10):
                print('nop')
        return QtCore.QRectF()

    def roiChangedEvent(self):
        if False:
            return 10
        w = self.lines[0].state['size'][1]
        for l in self.lines[1:]:
            w0 = l.state['size'][1]
            if w == w0:
                continue
            l.scale([1.0, w / w0], center=[0.5, 0.5])
        self.sigRegionChanged.emit(self)

    def roiChangeStartedEvent(self):
        if False:
            while True:
                i = 10
        self.sigRegionChangeStarted.emit(self)

    def roiChangeFinishedEvent(self):
        if False:
            for i in range(10):
                print('nop')
        self.sigRegionChangeFinished.emit(self)

    def getHandlePositions(self):
        if False:
            i = 10
            return i + 15
        'Return the positions of all handles in local coordinates.'
        pos = [self.mapFromScene(self.lines[0].getHandles()[0].scenePos())]
        for l in self.lines:
            pos.append(self.mapFromScene(l.getHandles()[1].scenePos()))
        return pos

    def getArrayRegion(self, arr, img=None, axes=(0, 1), **kwds):
        if False:
            while True:
                i = 10
        '\n        Return the result of :meth:`~pyqtgraph.ROI.getArrayRegion` for each rect\n        in the chain concatenated into a single ndarray.\n\n        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the\n        arguments.\n\n        Note: ``returnMappedCoords`` is not yet supported for this ROI type.\n        '
        rgns = []
        for l in self.lines:
            rgn = l.getArrayRegion(arr, img, axes=axes, **kwds)
            if rgn is None:
                continue
            rgns.append(rgn)
        if img.axisOrder == 'row-major':
            axes = axes[::-1]
        ms = min([r.shape[axes[1]] for r in rgns])
        sl = [slice(None)] * rgns[0].ndim
        sl[axes[1]] = slice(0, ms)
        rgns = [r[tuple(sl)] for r in rgns]
        return np.concatenate(rgns, axis=axes[0])

    def addSegment(self, pos=(0, 0), scaleHandle=False, connectTo=None):
        if False:
            print('Hello World!')
        '\n        Add a new segment to the ROI connecting from the previous endpoint to *pos*.\n        (pos is specified in the parent coordinate system of the MultiRectROI)\n        '
        if connectTo is None:
            connectTo = self.lines[-1].getHandles()[1]
        newRoi = ROI((0, 0), [1, 5], parent=self, pen=self.pen, **self.roiArgs)
        self.lines.append(newRoi)
        if isinstance(connectTo, Handle):
            self.lines[-1].addScaleRotateHandle([0, 0.5], [1, 0.5], item=connectTo)
            newRoi.movePoint(connectTo, connectTo.scenePos(), coords='scene')
        else:
            h = self.lines[-1].addScaleRotateHandle([0, 0.5], [1, 0.5])
            newRoi.movePoint(h, connectTo, coords='scene')
        h = self.lines[-1].addScaleRotateHandle([1, 0.5], [0, 0.5])
        newRoi.movePoint(h, pos)
        if scaleHandle:
            newRoi.addScaleHandle([0.5, 1], [0.5, 0.5])
        newRoi.translatable = False
        newRoi.sigRegionChanged.connect(self.roiChangedEvent)
        newRoi.sigRegionChangeStarted.connect(self.roiChangeStartedEvent)
        newRoi.sigRegionChangeFinished.connect(self.roiChangeFinishedEvent)
        self.sigRegionChanged.emit(self)

    def removeSegment(self, index=-1):
        if False:
            for i in range(10):
                print('nop')
        'Remove a segment from the ROI.'
        roi = self.lines[index]
        self.lines.pop(index)
        self.scene().removeItem(roi)
        roi.sigRegionChanged.disconnect(self.roiChangedEvent)
        roi.sigRegionChangeStarted.disconnect(self.roiChangeStartedEvent)
        roi.sigRegionChangeFinished.disconnect(self.roiChangeFinishedEvent)
        self.sigRegionChanged.emit(self)

class MultiLineROI(MultiRectROI):

    def __init__(self, *args, **kwds):
        if False:
            print('Hello World!')
        MultiRectROI.__init__(self, *args, **kwds)
        print('Warning: MultiLineROI has been renamed to MultiRectROI. (and MultiLineROI may be redefined in the future)')

class EllipseROI(ROI):
    """
    Elliptical ROI subclass with one scale handle and one rotation handle.


    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (length-2 sequence) The size of the ROI's bounding rectangle.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, pos, size, **args):
        if False:
            for i in range(10):
                print('nop')
        self.path = None
        ROI.__init__(self, pos, size, **args)
        self.sigRegionChanged.connect(self._clearPath)
        self._addHandles()

    def _addHandles(self):
        if False:
            print('Hello World!')
        self.addRotateHandle([1.0, 0.5], [0.5, 0.5])
        self.addScaleHandle([0.5 * 2.0 ** (-0.5) + 0.5, 0.5 * 2.0 ** (-0.5) + 0.5], [0.5, 0.5])

    def _clearPath(self):
        if False:
            print('Hello World!')
        self.path = None

    def paint(self, p, opt, widget):
        if False:
            return 10
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.scale(r.width(), r.height())
        r = QtCore.QRectF(r.x() / r.width(), r.y() / r.height(), 1, 1)
        p.drawEllipse(r)

    def getArrayRegion(self, arr, img=None, axes=(0, 1), returnMappedCoords=False, **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the result of :meth:`~pyqtgraph.ROI.getArrayRegion` masked by the\n        elliptical shape of the ROI. Regions outside the ellipse are set to 0.\n\n        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the\n        arguments.\n\n        Note: ``returnMappedCoords`` is not yet supported for this ROI type.\n        '
        if returnMappedCoords:
            (arr, mappedCoords) = ROI.getArrayRegion(self, arr, img, axes, returnMappedCoords, **kwds)
        else:
            arr = ROI.getArrayRegion(self, arr, img, axes, returnMappedCoords, **kwds)
        if arr is None or arr.shape[axes[0]] == 0 or arr.shape[axes[1]] == 0:
            if returnMappedCoords:
                return (arr, mappedCoords)
            else:
                return arr
        w = arr.shape[axes[0]]
        h = arr.shape[axes[1]]
        mask = np.fromfunction(lambda x, y: np.hypot((x + 0.5) / (w / 2.0) - 1, (y + 0.5) / (h / 2.0) - 1) < 1, (w, h))
        if axes[0] > axes[1]:
            mask = mask.T
        shape = [n if i in axes else 1 for (i, n) in enumerate(arr.shape)]
        mask = mask.reshape(shape)
        if returnMappedCoords:
            return (arr * mask, mappedCoords)
        else:
            return arr * mask

    def shape(self):
        if False:
            i = 10
            return i + 15
        if self.path is None:
            path = QtGui.QPainterPath()
            br = self.boundingRect()
            center = br.center()
            r1 = br.width() / 2.0
            r2 = br.height() / 2.0
            theta = np.linspace(0, 2 * np.pi, 24)
            x = center.x() + r1 * np.cos(theta)
            y = center.y() + r2 * np.sin(theta)
            path.moveTo(x[0], y[0])
            for i in range(1, len(x)):
                path.lineTo(x[i], y[i])
            self.path = path
        return self.path

class CircleROI(EllipseROI):
    """
    Circular ROI subclass. Behaves exactly as EllipseROI, but may only be scaled
    proportionally to maintain its aspect ratio.
    
    ============== =============================================================
    **Arguments**
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (length-2 sequence) The size of the ROI's bounding rectangle.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, pos, size=None, radius=None, **args):
        if False:
            for i in range(10):
                print('nop')
        if size is None:
            if radius is None:
                raise TypeError('Must provide either size or radius.')
            size = (radius * 2, radius * 2)
        EllipseROI.__init__(self, pos, size, aspectLocked=True, **args)

    def _addHandles(self):
        if False:
            while True:
                i = 10
        self.addScaleHandle([0.5 * 2.0 ** (-0.5) + 0.5, 0.5 * 2.0 ** (-0.5) + 0.5], [0.5, 0.5])

class PolyLineROI(ROI):
    """
    Container class for multiple connected LineSegmentROIs.

    This class allows the user to draw paths of multiple line segments.

    ============== =============================================================
    **Arguments**
    positions      (list of length-2 sequences) The list of points in the path.
                   Note that, unlike the handle positions specified in other
                   ROIs, these positions must be expressed in the normal
                   coordinate system of the ROI, rather than (0 to 1) relative
                   to the size of the ROI.
    closed         (bool) if True, an extra LineSegmentROI is added connecting 
                   the beginning and end points.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    
    """

    def __init__(self, positions, closed=False, pos=None, **args):
        if False:
            print('Hello World!')
        if pos is None:
            pos = [0, 0]
        self.closed = closed
        self.segments = []
        ROI.__init__(self, pos, size=[1, 1], **args)
        self.setPoints(positions)

    def setPoints(self, points, closed=None):
        if False:
            return 10
        '\n        Set the complete sequence of points displayed by this ROI.\n        \n        ============= =========================================================\n        **Arguments**\n        points        List of (x,y) tuples specifying handle locations to set.\n        closed        If bool, then this will set whether the ROI is closed \n                      (the last point is connected to the first point). If\n                      None, then the closed mode is left unchanged.\n        ============= =========================================================\n        \n        '
        if closed is not None:
            self.closed = closed
        self.clearPoints()
        for p in points:
            self.addFreeHandle(p)
        start = -1 if self.closed else 0
        for i in range(start, len(self.handles) - 1):
            self.addSegment(self.handles[i]['item'], self.handles[i + 1]['item'])

    def clearPoints(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Remove all handles and segments.\n        '
        while len(self.handles) > 0:
            self.removeHandle(self.handles[0]['item'])

    def getState(self):
        if False:
            while True:
                i = 10
        state = ROI.getState(self)
        state['closed'] = self.closed
        state['points'] = [Point(h.pos()) for h in self.getHandles()]
        return state

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        state = ROI.saveState(self)
        state['closed'] = self.closed
        state['points'] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self, state):
        if False:
            return 10
        ROI.setState(self, state)
        self.setPoints(state['points'], closed=state['closed'])

    def addSegment(self, h1, h2, index=None):
        if False:
            print('Hello World!')
        seg = _PolyLineSegment(handles=(h1, h2), pen=self.pen, hoverPen=self.hoverPen, parent=self, movable=False)
        if index is None:
            self.segments.append(seg)
        else:
            self.segments.insert(index, seg)
        seg.sigClicked.connect(self.segmentClicked)
        seg.setAcceptedMouseButtons(QtCore.Qt.MouseButton.LeftButton)
        seg.setZValue(self.zValue() + 1)
        for h in seg.handles:
            h['item'].setDeletable(True)
            h['item'].setAcceptedMouseButtons(h['item'].acceptedMouseButtons() | QtCore.Qt.MouseButton.LeftButton)

    def setMouseHover(self, hover):
        if False:
            for i in range(10):
                print('nop')
        ROI.setMouseHover(self, hover)
        for s in self.segments:
            s.setParentHover(hover)

    def addHandle(self, info, index=None):
        if False:
            for i in range(10):
                print('nop')
        h = ROI.addHandle(self, info, index=index)
        h.sigRemoveRequested.connect(self.removeHandle)
        self.stateChanged(finish=True)
        return h

    def segmentClicked(self, segment, ev=None, pos=None):
        if False:
            return 10
        if ev is not None:
            pos = segment.mapToParent(ev.pos())
        elif pos is None:
            raise Exception('Either an event or a position must be given.')
        h2 = segment.handles[1]['item']
        i = self.segments.index(segment)
        h3 = self.addFreeHandle(pos, index=self.indexOfHandle(h2))
        self.addSegment(h3, h2, index=i + 1)
        segment.replaceHandle(h2, h3)

    def removeHandle(self, handle, updateSegments=True):
        if False:
            while True:
                i = 10
        ROI.removeHandle(self, handle)
        handle.sigRemoveRequested.disconnect(self.removeHandle)
        if not updateSegments:
            return
        segments = handle.rois[:]
        if len(segments) == 1:
            self.removeSegment(segments[0])
        elif len(segments) > 1:
            handles = [h['item'] for h in segments[1].handles]
            handles.remove(handle)
            segments[0].replaceHandle(handle, handles[0])
            self.removeSegment(segments[1])
        self.stateChanged(finish=True)

    def removeSegment(self, seg):
        if False:
            return 10
        for handle in seg.handles[:]:
            seg.removeHandle(handle['item'])
        self.segments.remove(seg)
        seg.sigClicked.disconnect(self.segmentClicked)
        self.scene().removeItem(seg)

    def checkRemoveHandle(self, h):
        if False:
            while True:
                i = 10
        if self.closed:
            return len(self.handles) > 3
        else:
            return len(self.handles) > 2

    def paint(self, p, *args):
        if False:
            return 10
        pass

    def boundingRect(self):
        if False:
            while True:
                i = 10
        return self.shape().boundingRect()

    def shape(self):
        if False:
            return 10
        p = QtGui.QPainterPath()
        if len(self.handles) == 0:
            return p
        p.moveTo(self.handles[0]['item'].pos())
        for i in range(len(self.handles)):
            p.lineTo(self.handles[i]['item'].pos())
        p.lineTo(self.handles[0]['item'].pos())
        return p

    def getArrayRegion(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        return self._getArrayRegionForArbitraryShape(*args, **kwds)

    def setPen(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        ROI.setPen(self, *args, **kwds)
        for seg in self.segments:
            seg.setPen(*args, **kwds)

class LineSegmentROI(ROI):
    """
    ROI subclass with two freely-moving handles defining a line.

    ============== =============================================================
    **Arguments**
    positions      (list of two length-2 sequences) The endpoints of the line 
                   segment. Note that, unlike the handle positions specified in 
                   other ROIs, these positions must be expressed in the normal
                   coordinate system of the ROI, rather than (0 to 1) relative
                   to the size of the ROI.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """

    def __init__(self, positions=(None, None), pos=None, handles=(None, None), **args):
        if False:
            print('Hello World!')
        if pos is None:
            pos = [0, 0]
        ROI.__init__(self, pos, [1, 1], **args)
        if len(positions) > 2:
            raise Exception('LineSegmentROI must be defined by exactly 2 positions. For more points, use PolyLineROI.')
        for (i, p) in enumerate(positions):
            self.addFreeHandle(p, item=handles[i])

    @property
    def endpoints(self):
        if False:
            return 10
        return [h['item'] for h in self.handles]

    def listPoints(self):
        if False:
            i = 10
            return i + 15
        return [p['item'].pos() for p in self.handles]

    def getState(self):
        if False:
            print('Hello World!')
        state = ROI.getState(self)
        state['points'] = [Point(h.pos()) for h in self.getHandles()]
        return state

    def saveState(self):
        if False:
            while True:
                i = 10
        state = ROI.saveState(self)
        state['points'] = [tuple(h.pos()) for h in self.getHandles()]
        return state

    def setState(self, state):
        if False:
            while True:
                i = 10
        ROI.setState(self, state)
        p1 = [state['points'][0][0] + state['pos'][0], state['points'][0][1] + state['pos'][1]]
        p2 = [state['points'][1][0] + state['pos'][0], state['points'][1][1] + state['pos'][1]]
        self.movePoint(self.getHandles()[0], p1, finish=False)
        self.movePoint(self.getHandles()[1], p2)

    def paint(self, p, *args):
        if False:
            return 10
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()
        p.drawLine(h1, h2)

    def boundingRect(self):
        if False:
            print('Hello World!')
        return self.shape().boundingRect()

    def shape(self):
        if False:
            print('Hello World!')
        p = QtGui.QPainterPath()
        h1 = self.endpoints[0].pos()
        h2 = self.endpoints[1].pos()
        dh = h2 - h1
        if dh.length() == 0:
            return p
        pxv = self.pixelVectors(dh)[1]
        if pxv is None:
            return p
        pxv *= 4
        p.moveTo(h1 + pxv)
        p.lineTo(h2 + pxv)
        p.lineTo(h2 - pxv)
        p.lineTo(h1 - pxv)
        p.lineTo(h1 + pxv)
        return p

    def getArrayRegion(self, data, img, axes=(0, 1), order=1, returnMappedCoords=False, **kwds):
        if False:
            for i in range(10):
                print('nop')
        '\n        Use the position of this ROI relative to an imageItem to pull a slice \n        from an array.\n        \n        Since this pulls 1D data from a 2D coordinate system, the return value \n        will have ndim = data.ndim-1\n        \n        See :meth:`~pyqtgraph.ROI.getArrayRegion` for a description of the\n        arguments.\n        '
        imgPts = [self.mapToItem(img, h.pos()) for h in self.endpoints]
        d = Point(imgPts[1] - imgPts[0])
        o = Point(imgPts[0])
        rgn = fn.affineSlice(data, shape=(int(d.length()),), vectors=[Point(d.norm())], origin=o, axes=axes, order=order, returnCoords=returnMappedCoords, **kwds)
        return rgn

class _PolyLineSegment(LineSegmentROI):

    def __init__(self, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        self._parentHovering = False
        LineSegmentROI.__init__(self, *args, **kwds)

    def setParentHover(self, hover):
        if False:
            for i in range(10):
                print('nop')
        if self._parentHovering != hover:
            self._parentHovering = hover
            self._updateHoverColor()

    def _makePen(self):
        if False:
            while True:
                i = 10
        if self.mouseHovering or self._parentHovering:
            return self.hoverPen
        else:
            return self.pen

    def hoverEvent(self, ev):
        if False:
            print('Hello World!')
        if self.parentItem().translatable:
            ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)
        return LineSegmentROI.hoverEvent(self, ev)

class CrosshairROI(ROI):
    """A crosshair ROI whose position is at the center of the crosshairs. By default, it is scalable, rotatable and translatable."""

    def __init__(self, pos=None, size=None, **kargs):
        if False:
            print('Hello World!')
        if size is None:
            size = [1, 1]
        if pos is None:
            pos = [0, 0]
        self._shape = None
        ROI.__init__(self, pos, size, aspectLocked=True, **kargs)
        self.sigRegionChanged.connect(self.invalidate)
        self.addScaleRotateHandle(Point(1, 0), Point(0, 0))

    def invalidate(self):
        if False:
            for i in range(10):
                print('nop')
        self._shape = None
        self.prepareGeometryChange()

    def boundingRect(self):
        if False:
            return 10
        return self.shape().boundingRect()

    def shape(self):
        if False:
            print('Hello World!')
        if self._shape is None:
            radius = self.getState()['size'][1]
            p = QtGui.QPainterPath()
            p.moveTo(Point(0, -radius))
            p.lineTo(Point(0, radius))
            p.moveTo(Point(-radius, 0))
            p.lineTo(Point(radius, 0))
            p = self.mapToDevice(p)
            stroker = QtGui.QPainterPathStroker()
            stroker.setWidth(10)
            outline = stroker.createStroke(p)
            self._shape = self.mapFromDevice(outline)
        return self._shape

    def paint(self, p, *args):
        if False:
            for i in range(10):
                print('nop')
        radius = self.getState()['size'][1]
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.setPen(self.currentPen)
        p.drawLine(Point(0, -radius), Point(0, radius))
        p.drawLine(Point(-radius, 0), Point(radius, 0))

class RulerROI(LineSegmentROI):

    def paint(self, p, *args):
        if False:
            i = 10
            return i + 15
        LineSegmentROI.paint(self, p, *args)
        h1 = self.handles[0]['item'].pos()
        h2 = self.handles[1]['item'].pos()
        p1 = p.transform().map(h1)
        p2 = p.transform().map(h2)
        vec = Point(h2) - Point(h1)
        length = vec.length()
        angle = vec.angle(Point(1, 0))
        pvec = p2 - p1
        pvecT = Point(pvec.y(), -pvec.x())
        pos = 0.5 * (p1 + p2) + pvecT * 40 / pvecT.length()
        p.resetTransform()
        txt = fn.siFormat(length, suffix='m') + '\n%0.1f deg' % angle
        p.drawText(QtCore.QRectF(pos.x() - 50, pos.y() - 50, 100, 100), QtCore.Qt.AlignmentFlag.AlignCenter, txt)

    def boundingRect(self):
        if False:
            i = 10
            return i + 15
        r = LineSegmentROI.boundingRect(self)
        pxl = self.pixelLength(Point([1, 0]))
        if pxl is None:
            return r
        pxw = 50 * pxl
        return r.adjusted(-50, -50, 50, 50)

class TriangleROI(ROI):
    """
    Equilateral triangle ROI subclass with one scale handle and one rotation handle.
    Arguments
    pos            (length-2 sequence) The position of the ROI's origin.
    size           (float) The length of an edge of the triangle.
    \\**args        All extra keyword arguments are passed to ROI()
    ============== =============================================================
    """

    def __init__(self, pos, size, **args):
        if False:
            print('Hello World!')
        ROI.__init__(self, pos, [size, size], aspectLocked=True, **args)
        angles = np.linspace(0, np.pi * 4 / 3, 3)
        verticies = (np.array((np.sin(angles), np.cos(angles))).T + 1.0) / 2.0
        self.poly = QtGui.QPolygonF()
        for pt in verticies:
            self.poly.append(QtCore.QPointF(*pt))
        self.addRotateHandle(verticies[0], [0.5, 0.5])
        self.addScaleHandle(verticies[1], [0.5, 0.5])

    def paint(self, p, *args):
        if False:
            return 10
        r = self.boundingRect()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        p.scale(r.width(), r.height())
        p.setPen(self.currentPen)
        p.drawPolygon(self.poly)

    def shape(self):
        if False:
            i = 10
            return i + 15
        self.path = QtGui.QPainterPath()
        r = self.boundingRect()
        t = QtGui.QTransform()
        t.scale(r.width(), r.height())
        self.path.addPolygon(self.poly)
        return t.map(self.path)

    def getArrayRegion(self, *args, **kwds):
        if False:
            print('Hello World!')
        return self._getArrayRegionForArbitraryShape(*args, **kwds)