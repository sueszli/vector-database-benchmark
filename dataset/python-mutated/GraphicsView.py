"""
GraphicsView.py -   Extension of QGraphicsView
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from .. import functions as fn
from .. import getConfigOption
from ..GraphicsScene import GraphicsScene
from ..Point import Point
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
__all__ = ['GraphicsView']

class GraphicsView(QtWidgets.QGraphicsView):
    """Re-implementation of QGraphicsView that removes scrollbars and allows unambiguous control of the 
    viewed coordinate range. Also automatically creates a GraphicsScene and a central QGraphicsWidget
    that is automatically scaled to the full view geometry.
    
    This widget is the basis for :class:`PlotWidget <pyqtgraph.PlotWidget>`, 
    :class:`GraphicsLayoutWidget <pyqtgraph.GraphicsLayoutWidget>`, and the view widget in
    :class:`ImageView <pyqtgraph.ImageView>`.
    
    By default, the view coordinate system matches the widget's pixel coordinates and 
    automatically updates when the view is resized. This can be overridden by setting 
    autoPixelRange=False. The exact visible range can be set with setRange().
    
    The view can be panned using the middle mouse button and scaled using the right mouse button if
    enabled via enableMouse()  (but ordinarily, we use ViewBox for this functionality)."""
    sigDeviceRangeChanged = QtCore.Signal(object, object)
    sigDeviceTransformChanged = QtCore.Signal(object)
    sigMouseReleased = QtCore.Signal(object)
    sigSceneMouseMoved = QtCore.Signal(object)
    sigScaleChanged = QtCore.Signal(object)
    lastFileDir = None

    def __init__(self, parent=None, useOpenGL=None, background='default'):
        if False:
            i = 10
            return i + 15
        "\n        ==============  ============================================================\n        **Arguments:**\n        parent          Optional parent widget\n        useOpenGL       If True, the GraphicsView will use OpenGL to do all of its\n                        rendering. This can improve performance on some systems,\n                        but may also introduce bugs (the combination of \n                        QGraphicsView and QOpenGLWidget is still an 'experimental'\n                        feature of Qt)\n        background      Set the background color of the GraphicsView. Accepts any\n                        single argument accepted by \n                        :func:`mkColor <pyqtgraph.mkColor>`. By \n                        default, the background color is determined using the\n                        'backgroundColor' configuration option (see \n                        :func:`setConfigOptions <pyqtgraph.setConfigOptions>`).\n        ==============  ============================================================\n        "
        self.closed = False
        QtWidgets.QGraphicsView.__init__(self, parent)
        from .. import _connectCleanup
        _connectCleanup()
        if useOpenGL is None:
            useOpenGL = getConfigOption('useOpenGL')
        self.useOpenGL(useOpenGL)
        self.setCacheMode(self.CacheModeFlag.CacheBackground)
        self.setBackgroundRole(QtGui.QPalette.ColorRole.NoRole)
        self.setBackground(background)
        self.setFocusPolicy(QtCore.Qt.FocusPolicy.StrongFocus)
        self.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QtWidgets.QGraphicsView.ViewportAnchor.NoAnchor)
        self.setResizeAnchor(QtWidgets.QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setViewportUpdateMode(QtWidgets.QGraphicsView.ViewportUpdateMode.MinimalViewportUpdate)
        self.lockedViewports = []
        self.lastMousePos = None
        self.setMouseTracking(True)
        self.aspectLocked = False
        self.range = QtCore.QRectF(0, 0, 1, 1)
        self.autoPixelRange = True
        self.currentItem = None
        self.clearMouse()
        self.updateMatrix()
        self.sceneObj = GraphicsScene(parent=self)
        self.setScene(self.sceneObj)
        self.centralWidget = None
        self.setCentralItem(QtWidgets.QGraphicsWidget())
        self.centralLayout = QtWidgets.QGraphicsGridLayout()
        self.centralWidget.setLayout(self.centralLayout)
        self.mouseEnabled = False
        self.scaleCenter = False
        self.clickAccepted = False

    def setAntialiasing(self, aa):
        if False:
            while True:
                i = 10
        'Enable or disable default antialiasing.\n        Note that this will only affect items that do not specify their own antialiasing options.'
        if aa:
            self.setRenderHints(self.renderHints() | QtGui.QPainter.RenderHint.Antialiasing)
        else:
            self.setRenderHints(self.renderHints() & ~QtGui.QPainter.RenderHint.Antialiasing)

    def setBackground(self, background):
        if False:
            while True:
                i = 10
        "\n        Set the background color of the GraphicsView.\n        To use the defaults specified py pyqtgraph.setConfigOption, use background='default'.\n        To make the background transparent, use background=None.\n        "
        self._background = background
        if background == 'default':
            background = getConfigOption('background')
        brush = fn.mkBrush(background)
        self.setBackgroundBrush(brush)

    def paintEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        self.scene().prepareForPaint()
        return super().paintEvent(ev)

    def render(self, *args, **kwds):
        if False:
            i = 10
            return i + 15
        self.scene().prepareForPaint()
        return super().render(*args, **kwds)

    def close(self):
        if False:
            while True:
                i = 10
        self.centralWidget = None
        self.scene().clear()
        self.currentItem = None
        self.sceneObj = None
        self.closed = True
        self.setViewport(None)
        super(GraphicsView, self).close()

    def useOpenGL(self, b=True):
        if False:
            while True:
                i = 10
        if b:
            HAVE_OPENGL = hasattr(QtWidgets, 'QOpenGLWidget')
            if not HAVE_OPENGL:
                raise Exception('Requested to use OpenGL with QGraphicsView, but QOpenGLWidget is not available.')
            v = QtWidgets.QOpenGLWidget()
        else:
            v = QtWidgets.QWidget()
        self.setViewport(v)

    def keyPressEvent(self, ev):
        if False:
            i = 10
            return i + 15
        self.scene().keyPressEvent(ev)

    def setCentralItem(self, item):
        if False:
            i = 10
            return i + 15
        return self.setCentralWidget(item)

    def setCentralWidget(self, item):
        if False:
            return 10
        'Sets a QGraphicsWidget to automatically fill the entire view (the item will be automatically\n        resize whenever the GraphicsView is resized).'
        if self.centralWidget is not None:
            self.scene().removeItem(self.centralWidget)
        self.centralWidget = item
        if item is not None:
            self.sceneObj.addItem(item)
            self.resizeEvent(None)

    def addItem(self, *args):
        if False:
            i = 10
            return i + 15
        return self.scene().addItem(*args)

    def removeItem(self, *args):
        if False:
            return 10
        return self.scene().removeItem(*args)

    def enableMouse(self, b=True):
        if False:
            i = 10
            return i + 15
        self.mouseEnabled = b
        self.autoPixelRange = not b

    def clearMouse(self):
        if False:
            while True:
                i = 10
        self.mouseTrail = []
        self.lastButtonReleased = None

    def resizeEvent(self, ev):
        if False:
            print('Hello World!')
        if self.closed:
            return
        if self.autoPixelRange:
            self.range = QtCore.QRectF(0, 0, self.size().width(), self.size().height())
        GraphicsView.setRange(self, self.range, padding=0, disableAutoPixel=False)
        self.updateMatrix()

    def updateMatrix(self, propagate=True):
        if False:
            for i in range(10):
                print('nop')
        self.setSceneRect(self.range)
        if self.autoPixelRange:
            self.resetTransform()
        elif self.aspectLocked:
            self.fitInView(self.range, QtCore.Qt.AspectRatioMode.KeepAspectRatio)
        else:
            self.fitInView(self.range, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
        if propagate:
            for v in self.lockedViewports:
                v.setXRange(self.range, padding=0)
        self.sigDeviceRangeChanged.emit(self, self.range)
        self.sigDeviceTransformChanged.emit(self)

    def viewRect(self):
        if False:
            print('Hello World!')
        'Return the boundaries of the view in scene coordinates'
        r = QtCore.QRectF(self.rect())
        return self.viewportTransform().inverted()[0].mapRect(r)

    def visibleRange(self):
        if False:
            print('Hello World!')
        return self.viewRect()

    def translate(self, dx, dy):
        if False:
            for i in range(10):
                print('nop')
        self.range.adjust(dx, dy, dx, dy)
        self.updateMatrix()

    def scale(self, sx, sy, center=None):
        if False:
            for i in range(10):
                print('nop')
        scale = [sx, sy]
        if self.aspectLocked:
            scale[0] = scale[1]
        if self.scaleCenter:
            center = None
        if center is None:
            center = self.range.center()
        w = self.range.width() / scale[0]
        h = self.range.height() / scale[1]
        self.range = QtCore.QRectF(center.x() - (center.x() - self.range.left()) / scale[0], center.y() - (center.y() - self.range.top()) / scale[1], w, h)
        self.updateMatrix()
        self.sigScaleChanged.emit(self)

    def setRange(self, newRect=None, padding=0.05, lockAspect=None, propagate=True, disableAutoPixel=True):
        if False:
            return 10
        if disableAutoPixel:
            self.autoPixelRange = False
        if newRect is None:
            newRect = self.visibleRange()
            padding = 0
        padding = Point(padding)
        newRect = QtCore.QRectF(newRect)
        pw = newRect.width() * padding[0]
        ph = newRect.height() * padding[1]
        newRect = newRect.adjusted(-pw, -ph, pw, ph)
        scaleChanged = False
        if self.range.width() != newRect.width() or self.range.height() != newRect.height():
            scaleChanged = True
        self.range = newRect
        if self.centralWidget is not None:
            self.centralWidget.setGeometry(self.range)
        self.updateMatrix(propagate)
        if scaleChanged:
            self.sigScaleChanged.emit(self)

    def scaleToImage(self, image):
        if False:
            while True:
                i = 10
        'Scales such that pixels in image are the same size as screen pixels. This may result in a significant performance increase.'
        pxSize = image.pixelSize()
        image.setPxMode(True)
        try:
            self.sigScaleChanged.disconnect(image.setScaledMode)
        except (TypeError, RuntimeError):
            pass
        tl = image.sceneBoundingRect().topLeft()
        w = self.size().width() * pxSize[0]
        h = self.size().height() * pxSize[1]
        range = QtCore.QRectF(tl.x(), tl.y(), w, h)
        GraphicsView.setRange(self, range, padding=0)
        self.sigScaleChanged.connect(image.setScaledMode)

    def lockXRange(self, v1):
        if False:
            print('Hello World!')
        if not v1 in self.lockedViewports:
            self.lockedViewports.append(v1)

    def setXRange(self, r, padding=0.05):
        if False:
            print('Hello World!')
        r1 = QtCore.QRectF(self.range)
        r1.setLeft(r.left())
        r1.setRight(r.right())
        GraphicsView.setRange(self, r1, padding=[padding, 0], propagate=False)

    def setYRange(self, r, padding=0.05):
        if False:
            return 10
        r1 = QtCore.QRectF(self.range)
        r1.setTop(r.top())
        r1.setBottom(r.bottom())
        GraphicsView.setRange(self, r1, padding=[0, padding], propagate=False)

    def wheelEvent(self, ev):
        if False:
            i = 10
            return i + 15
        super().wheelEvent(ev)
        if not self.mouseEnabled:
            return
        delta = ev.angleDelta().x()
        if delta == 0:
            delta = ev.angleDelta().y()
        sc = 1.001 ** delta
        self.scale(sc, sc)

    def setAspectLocked(self, s):
        if False:
            return 10
        self.aspectLocked = s

    def leaveEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        self.scene().leaveEvent(ev)

    def mousePressEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        super().mousePressEvent(ev)
        if not self.mouseEnabled:
            return
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        self.lastMousePos = lpos
        self.mousePressPos = lpos
        self.clickAccepted = ev.isAccepted()
        if not self.clickAccepted:
            self.scene().clearSelection()
        return

    def mouseReleaseEvent(self, ev):
        if False:
            print('Hello World!')
        super().mouseReleaseEvent(ev)
        if not self.mouseEnabled:
            return
        self.sigMouseReleased.emit(ev)
        self.lastButtonReleased = ev.button()
        return

    def mouseMoveEvent(self, ev):
        if False:
            while True:
                i = 10
        lpos = ev.position() if hasattr(ev, 'position') else ev.localPos()
        if self.lastMousePos is None:
            self.lastMousePos = lpos
        delta = Point(lpos - self.lastMousePos)
        self.lastMousePos = lpos
        super().mouseMoveEvent(ev)
        if not self.mouseEnabled:
            return
        self.sigSceneMouseMoved.emit(self.mapToScene(lpos.toPoint()))
        if self.clickAccepted:
            return
        if ev.buttons() == QtCore.Qt.MouseButton.RightButton:
            delta = Point(fn.clip_scalar(delta[0], -50, 50), fn.clip_scalar(-delta[1], -50, 50))
            scale = 1.01 ** delta
            self.scale(scale[0], scale[1], center=self.mapToScene(self.mousePressPos.toPoint()))
            self.sigDeviceRangeChanged.emit(self, self.range)
        elif ev.buttons() in [QtCore.Qt.MouseButton.MiddleButton, QtCore.Qt.MouseButton.LeftButton]:
            px = self.pixelSize()
            tr = -delta * px
            self.translate(tr[0], tr[1])
            self.sigDeviceRangeChanged.emit(self, self.range)

    def pixelSize(self):
        if False:
            print('Hello World!')
        'Return vector with the length and width of one view pixel in scene coordinates'
        p0 = Point(0, 0)
        p1 = Point(1, 1)
        tr = self.transform().inverted()[0]
        p01 = tr.map(p0)
        p11 = tr.map(p1)
        return Point(p11 - p01)

    def dragEnterEvent(self, ev):
        if False:
            return 10
        ev.ignore()