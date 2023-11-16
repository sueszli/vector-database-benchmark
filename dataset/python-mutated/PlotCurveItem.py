from ..Qt import QtCore, QtGui, QtWidgets
HAVE_OPENGL = hasattr(QtWidgets, 'QOpenGLWidget')
import math
import sys
import warnings
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from .GraphicsObject import GraphicsObject
__all__ = ['PlotCurveItem']

def arrayToLineSegments(x, y, connect, finiteCheck, out=None):
    if False:
        while True:
            i = 10
    if out is None:
        out = Qt.internals.PrimitiveArray(QtCore.QLineF, 4)
    if len(x) < 2:
        out.resize(0)
        return out
    connect_array = None
    if isinstance(connect, np.ndarray):
        (connect_array, connect) = (np.asarray(connect[:-1], dtype=bool), 'array')
    all_finite = True
    if finiteCheck or connect == 'finite':
        mask = np.isfinite(x) & np.isfinite(y)
        all_finite = np.all(mask)
    if connect == 'all':
        if not all_finite:
            x = x[mask]
            y = y[mask]
    elif connect == 'finite':
        if all_finite:
            connect = 'all'
        else:
            connect_array = mask[:-1] & mask[1:]
    elif connect in ['pairs', 'array']:
        if not all_finite:
            backfill_idx = fn._compute_backfill_indices(mask)
            x = x[backfill_idx]
            y = y[backfill_idx]
    if connect == 'all':
        nsegs = len(x) - 1
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1]
            memory[:, 2] = x[1:]
            memory[:, 1] = y[:-1]
            memory[:, 3] = y[1:]
    elif connect == 'pairs':
        nsegs = len(x) // 2
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory = memory.reshape((-1, 2))
            memory[:, 0] = x[:nsegs * 2]
            memory[:, 1] = y[:nsegs * 2]
    elif connect_array is not None:
        nsegs = np.count_nonzero(connect_array)
        out.resize(nsegs)
        if nsegs:
            memory = out.ndarray()
            memory[:, 0] = x[:-1][connect_array]
            memory[:, 2] = x[1:][connect_array]
            memory[:, 1] = y[:-1][connect_array]
            memory[:, 3] = y[1:][connect_array]
    else:
        nsegs = 0
        out.resize(nsegs)
    return out

class PlotCurveItem(GraphicsObject):
    """
    Class representing a single plot curve. Instances of this class are created
    automatically as part of :class:`PlotDataItem <pyqtgraph.PlotDataItem>`; 
    these rarely need to be instantiated directly.

    Features:

      - Fast data update
      - Fill under curve
      - Mouse interaction

    =====================  ===============================================
    **Signals:**
    sigPlotChanged(self)   Emitted when the data being plotted has changed
    sigClicked(self, ev)   Emitted when the curve is clicked
    =====================  ===============================================
    """
    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)

    def __init__(self, *args, **kargs):
        if False:
            print('Hello World!')
        '\n        Forwards all arguments to :func:`setData <pyqtgraph.PlotCurveItem.setData>`.\n\n        Some extra arguments are accepted as well:\n\n        ==============  =======================================================\n        **Arguments:**\n        parent          The parent GraphicsObject (optional)\n        clickable       If `True`, the item will emit ``sigClicked`` when it is\n                        clicked on. Defaults to `False`.\n        ==============  =======================================================\n        '
        GraphicsObject.__init__(self, kargs.get('parent', None))
        self.clear()
        self.metaData = {}
        self.opts = {'shadowPen': None, 'fillLevel': None, 'fillOutline': False, 'brush': None, 'stepMode': None, 'name': None, 'antialias': getConfigOption('antialias'), 'connect': 'all', 'mouseWidth': 8, 'compositionMode': None, 'skipFiniteCheck': False, 'segmentedLineMode': getConfigOption('segmentedLineMode')}
        if 'pen' not in kargs:
            self.opts['pen'] = fn.mkPen('w')
        self.setClickable(kargs.get('clickable', False))
        self.setData(*args, **kargs)

    def implements(self, interface=None):
        if False:
            print('Hello World!')
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        if False:
            print('Hello World!')
        return self.opts.get('name', None)

    def setClickable(self, s, width=None):
        if False:
            i = 10
            return i + 15
        'Sets whether the item responds to mouse clicks.\n\n        The `width` argument specifies the width in pixels orthogonal to the\n        curve that will respond to a mouse click.\n        '
        self.clickable = s
        if width is not None:
            self.opts['mouseWidth'] = width
            self._mouseShape = None
            self._boundingRect = None

    def setCompositionMode(self, mode):
        if False:
            i = 10
            return i + 15
        '\n        Change the composition mode of the item. This is useful when overlaying\n        multiple items.\n        \n        Parameters\n        ----------\n        mode : ``QtGui.QPainter.CompositionMode``\n            Composition of the item, often used when overlaying items.  Common\n            options include:\n\n            ``QPainter.CompositionMode.CompositionMode_SourceOver`` (Default)\n            Image replaces the background if it is opaque. Otherwise, it uses\n            the alpha channel to blend the image with the background.\n\n            ``QPainter.CompositionMode.CompositionMode_Overlay`` Image color is\n            mixed with the background color to reflect the lightness or\n            darkness of the background\n\n            ``QPainter.CompositionMode.CompositionMode_Plus`` Both the alpha\n            and color of the image and background pixels are added together.\n\n            ``QPainter.CompositionMode.CompositionMode_Plus`` The output is the\n            image color multiplied by the background.\n\n            See ``QPainter::CompositionMode`` in the Qt Documentation for more\n            options and details\n        '
        self.opts['compositionMode'] = mode
        self.update()

    def getData(self):
        if False:
            print('Hello World!')
        return (self.xData, self.yData)

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if False:
            print('Hello World!')
        cache = self._boundsCache[ax]
        if cache is not None and cache[0] == (frac, orthoRange):
            return cache[1]
        (x, y) = self.getData()
        if x is None or len(x) == 0:
            return (None, None)
        if ax == 0:
            d = x
            d2 = y
        elif ax == 1:
            d = y
            d2 = x
        else:
            raise ValueError('Invalid axis value')
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            if self.opts.get('stepMode', None) == 'center':
                mask = mask[:-1]
            d = d[mask]
        if len(d) == 0:
            return (None, None)
        if frac >= 1.0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                b = (float(np.nanmin(d)), float(np.nanmax(d)))
            if math.isinf(b[0]) or math.isinf(b[1]):
                mask = np.isfinite(d)
                d = d[mask]
                if len(d) == 0:
                    return (None, None)
                b = (float(d.min()), float(d.max()))
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            mask = np.isfinite(d)
            d = d[mask]
            if len(d) == 0:
                return (None, None)
            b = np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])
        if ax == 1 and self.opts['fillLevel'] not in [None, 'enclosed']:
            b = (float(min(b[0], self.opts['fillLevel'])), float(max(b[1], self.opts['fillLevel'])))
        pen = self.opts['pen']
        spen = self.opts['shadowPen']
        if pen is not None and (not pen.isCosmetic()) and (pen.style() != QtCore.Qt.PenStyle.NoPen):
            b = (b[0] - pen.widthF() * 0.7072, b[1] + pen.widthF() * 0.7072)
        if spen is not None and (not spen.isCosmetic()) and (spen.style() != QtCore.Qt.PenStyle.NoPen):
            b = (b[0] - spen.widthF() * 0.7072, b[1] + spen.widthF() * 0.7072)
        self._boundsCache[ax] = [(frac, orthoRange), b]
        return b

    def pixelPadding(self):
        if False:
            while True:
                i = 10
        pen = self.opts['pen']
        spen = self.opts['shadowPen']
        w = 0
        if pen is not None and pen.isCosmetic() and (pen.style() != QtCore.Qt.PenStyle.NoPen):
            w += pen.widthF() * 0.7072
        if spen is not None and spen.isCosmetic() and (spen.style() != QtCore.Qt.PenStyle.NoPen):
            w = max(w, spen.widthF() * 0.7072)
        if self.clickable:
            w = max(w, self.opts['mouseWidth'] // 2 + 1)
        return w

    def boundingRect(self):
        if False:
            for i in range(10):
                print('nop')
        if self._boundingRect is None:
            (xmn, xmx) = self.dataBounds(ax=0)
            if xmn is None or xmx is None:
                return QtCore.QRectF()
            (ymn, ymx) = self.dataBounds(ax=1)
            if ymn is None or ymx is None:
                return QtCore.QRectF()
            px = py = 0.0
            pxPad = self.pixelPadding()
            if pxPad > 0:
                (px, py) = self.pixelVectors()
                try:
                    px = 0 if px is None else px.length()
                except OverflowError:
                    px = 0
                try:
                    py = 0 if py is None else py.length()
                except OverflowError:
                    py = 0
                px *= pxPad
                py *= pxPad
            self._boundingRect = QtCore.QRectF(xmn - px, ymn - py, 2 * px + xmx - xmn, 2 * py + ymx - ymn)
        return self._boundingRect

    def viewTransformChanged(self):
        if False:
            for i in range(10):
                print('nop')
        self.invalidateBounds()
        self.prepareGeometryChange()

    def invalidateBounds(self):
        if False:
            i = 10
            return i + 15
        self._boundingRect = None
        self._boundsCache = [None, None]

    def setPen(self, *args, **kargs):
        if False:
            print('Hello World!')
        'Set the pen used to draw the curve.'
        if args and args[0] is None:
            self.opts['pen'] = None
        else:
            self.opts['pen'] = fn.mkPen(*args, **kargs)
        self.invalidateBounds()
        self.update()

    def setShadowPen(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the shadow pen used to draw behind the primary pen.\n        This pen must have a larger width than the primary\n        pen to be visible. Arguments are passed to \n        :func:`mkPen <pyqtgraph.mkPen>`\n        '
        if args and args[0] is None:
            self.opts['shadowPen'] = None
        else:
            self.opts['shadowPen'] = fn.mkPen(*args, **kargs)
        self.invalidateBounds()
        self.update()

    def setBrush(self, *args, **kargs):
        if False:
            print('Hello World!')
        '\n        Sets the brush used when filling the area under the curve. All \n        arguments are passed to :func:`mkBrush <pyqtgraph.mkBrush>`.\n        '
        if args and args[0] is None:
            self.opts['brush'] = None
        else:
            self.opts['brush'] = fn.mkBrush(*args, **kargs)
        self.invalidateBounds()
        self.update()

    def setFillLevel(self, level):
        if False:
            for i in range(10):
                print('nop')
        'Sets the level filled to when filling under the curve'
        self.opts['fillLevel'] = level
        self.fillPath = None
        self._fillPathList = None
        self.invalidateBounds()
        self.update()

    def setSkipFiniteCheck(self, skipFiniteCheck):
        if False:
            print('Hello World!')
        '\n        When it is known that the plot data passed to ``PlotCurveItem`` contains only finite numerical values,\n        the `skipFiniteCheck` property can help speed up plotting. If this flag is set and the data contains \n        any non-finite values (such as `NaN` or `Inf`), unpredictable behavior will occur. The data might not\n        be plotted, or there migth be significant performance impact.\n        '
        self.opts['skipFiniteCheck'] = bool(skipFiniteCheck)

    def setData(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        =============== =================================================================\n        **Arguments:**\n        x, y            (numpy arrays) Data to display\n        pen             Pen to use when drawing. Any single argument accepted by\n                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.\n        shadowPen       Pen for drawing behind the primary pen. Usually this\n                        is used to emphasize the curve by providing a\n                        high-contrast border. Any single argument accepted by\n                        :func:`mkPen <pyqtgraph.mkPen>` is allowed.\n        fillLevel       (float or None) Fill the area under the curve to\n                        the specified value.\n        fillOutline     (bool) If True, an outline surrounding the `fillLevel`\n                        area is drawn.\n        brush           Brush to use when filling. Any single argument accepted\n                        by :func:`mkBrush <pyqtgraph.mkBrush>` is allowed.\n        antialias       (bool) Whether to use antialiasing when drawing. This\n                        is disabled by default because it decreases performance.\n        stepMode        (str or None) If 'center', a step is drawn using the `x`\n                        values as boundaries and the given `y` values are\n                        associated to the mid-points between the boundaries of\n                        each step. This is commonly used when drawing\n                        histograms. Note that in this case, ``len(x) == len(y) + 1``\n                        \n                        If 'left' or 'right', the step is drawn assuming that\n                        the `y` value is associated to the left or right boundary,\n                        respectively. In this case ``len(x) == len(y)``\n                        If not passed or an empty string or `None` is passed, the\n                        step mode is not enabled.\n        connect         Argument specifying how vertexes should be connected\n                        by line segments. \n                        \n                            | 'all' (default) indicates full connection. \n                            | 'pairs' draws one separate line segment for each two points given.\n                            | 'finite' omits segments attached to `NaN` or `Inf` values. \n                            | For any other connectivity, specify an array of boolean values.\n        compositionMode See :func:`setCompositionMode\n                        <pyqtgraph.PlotCurveItem.setCompositionMode>`.\n        skipFiniteCheck (bool, defaults to `False`) Optimization flag that can\n                        speed up plotting by not checking and compensating for\n                        `NaN` values.  If set to `True`, and `NaN` values exist, the\n                        data may not be displayed or the plot may take a\n                        significant performance hit.\n        =============== =================================================================\n\n        If non-keyword arguments are used, they will be interpreted as\n        ``setData(y)`` for a single argument and ``setData(x, y)`` for two\n        arguments.\n        \n        **Notes on performance:**\n        \n        Line widths greater than 1 pixel affect the performance as discussed in \n        the documentation of :class:`PlotDataItem <pyqtgraph.PlotDataItem>`.\n        "
        self.updateData(*args, **kargs)

    def updateData(self, *args, **kargs):
        if False:
            while True:
                i = 10
        profiler = debug.Profiler()
        if 'compositionMode' in kargs:
            self.setCompositionMode(kargs['compositionMode'])
        if len(args) == 1:
            kargs['y'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]
        if 'y' not in kargs or kargs['y'] is None:
            kargs['y'] = np.array([])
        if 'x' not in kargs or kargs['x'] is None:
            kargs['x'] = np.arange(len(kargs['y']))
        for k in ['x', 'y']:
            data = kargs[k]
            if isinstance(data, list):
                data = np.array(data)
                kargs[k] = data
            if not isinstance(data, np.ndarray) or data.ndim > 1:
                raise Exception('Plot data must be 1D ndarray.')
            if data.dtype.kind == 'c':
                raise Exception('Can not plot complex data types.')
        profiler('data checks')
        self.yData = kargs['y'].view(np.ndarray)
        self.xData = kargs['x'].view(np.ndarray)
        self.invalidateBounds()
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        profiler('copy')
        if 'stepMode' in kargs:
            self.opts['stepMode'] = kargs['stepMode']
        if self.opts['stepMode'] in ('center', True):
            if self.opts['stepMode'] is True:
                warnings.warn('stepMode=True is deprecated and will result in an error after October 2022. Use stepMode="center" instead.', DeprecationWarning, stacklevel=3)
            if len(self.xData) != len(self.yData) + 1:
                raise Exception('len(X) must be len(Y)+1 since stepMode=True (got %s and %s)' % (self.xData.shape, self.yData.shape))
        elif self.xData.shape != self.yData.shape:
            raise Exception('X and Y arrays must be the same shape--got %s and %s.' % (self.xData.shape, self.yData.shape))
        self.path = None
        self.fillPath = None
        self._fillPathList = None
        self._mouseShape = None
        self._lineSegmentsRendered = False
        if 'name' in kargs:
            self.opts['name'] = kargs['name']
        if 'connect' in kargs:
            self.opts['connect'] = kargs['connect']
        if 'pen' in kargs:
            self.setPen(kargs['pen'])
        if 'shadowPen' in kargs:
            self.setShadowPen(kargs['shadowPen'])
        if 'fillLevel' in kargs:
            self.setFillLevel(kargs['fillLevel'])
        if 'fillOutline' in kargs:
            self.opts['fillOutline'] = kargs['fillOutline']
        if 'brush' in kargs:
            self.setBrush(kargs['brush'])
        if 'antialias' in kargs:
            self.opts['antialias'] = kargs['antialias']
        if 'skipFiniteCheck' in kargs:
            self.opts['skipFiniteCheck'] = kargs['skipFiniteCheck']
        profiler('set')
        self.update()
        profiler('update')
        self.sigPlotChanged.emit(self)
        profiler('emit')

    @staticmethod
    def _generateStepModeData(stepMode, x, y, baseline):
        if False:
            for i in range(10):
                print('nop')
        if stepMode == 'right':
            x2 = np.empty((len(x) + 1, 2), dtype=x.dtype)
            x2[:-1] = x[:, np.newaxis]
            x2[-1] = x2[-2]
        elif stepMode == 'left':
            x2 = np.empty((len(x) + 1, 2), dtype=x.dtype)
            x2[1:] = x[:, np.newaxis]
            x2[0] = x2[1]
        elif stepMode in ('center', True):
            x2 = np.empty((len(x), 2), dtype=x.dtype)
            x2[:] = x[:, np.newaxis]
        else:
            raise ValueError('Unsupported stepMode %s' % stepMode)
        if baseline is None:
            x = x2.reshape(x2.size)[1:-1]
            y2 = np.empty((len(y), 2), dtype=y.dtype)
            y2[:] = y[:, np.newaxis]
            y = y2.reshape(y2.size)
        else:
            x = x2.reshape(x2.size)
            y2 = np.empty((len(y) + 2, 2), dtype=y.dtype)
            y2[1:-1] = y[:, np.newaxis]
            y = y2.reshape(y2.size)[1:-1]
            y[[0, -1]] = baseline
        return (x, y)

    def generatePath(self, x, y):
        if False:
            while True:
                i = 10
        if self.opts['stepMode']:
            (x, y) = self._generateStepModeData(self.opts['stepMode'], x, y, baseline=self.opts['fillLevel'])
        return fn.arrayToQPath(x, y, connect=self.opts['connect'], finiteCheck=not self.opts['skipFiniteCheck'])

    def getPath(self):
        if False:
            while True:
                i = 10
        if self.path is None:
            (x, y) = self.getData()
            if x is None or len(x) == 0 or y is None or (len(y) == 0):
                self.path = QtGui.QPainterPath()
            else:
                self.path = self.generatePath(*self.getData())
            self.fillPath = None
            self._fillPathList = None
            self._mouseShape = None
        return self.path

    def setSegmentedLineMode(self, mode):
        if False:
            print('Hello World!')
        "\n        Sets the mode that decides whether or not lines are drawn as segmented lines. Drawing lines\n        as segmented lines is more performant than the standard drawing method with continuous\n        lines.\n\n        Parameters\n        ----------\n        mode : str\n               ``'auto'`` (default) segmented lines are drawn if the pen's width > 1, pen style is a\n               solid line, the pen color is opaque and anti-aliasing is not enabled.\n\n               ``'on'`` lines are always drawn as segmented lines\n\n               ``'off'`` lines are never drawn as segmented lines, i.e. the drawing\n               method with continuous lines is used\n        "
        if mode not in ('auto', 'on', 'off'):
            raise ValueError(f'segmentedLineMode must be "auto", "on" or "off", got {mode} instead')
        self.opts['segmentedLineMode'] = mode
        self.invalidateBounds()
        self.update()

    def _shouldUseDrawLineSegments(self, pen):
        if False:
            for i in range(10):
                print('nop')
        mode = self.opts['segmentedLineMode']
        if mode in ('on',):
            return True
        if mode in ('off',):
            return False
        return pen.widthF() > 1.0 and pen.style() == QtCore.Qt.PenStyle.SolidLine and pen.isSolid() and (pen.color().alphaF() == 1.0) and (not self.opts['antialias'])

    def _getLineSegments(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._lineSegmentsRendered:
            (x, y) = self.getData()
            if self.opts['stepMode']:
                (x, y) = self._generateStepModeData(self.opts['stepMode'], x, y, baseline=self.opts['fillLevel'])
            self._lineSegments = arrayToLineSegments(x, y, connect=self.opts['connect'], finiteCheck=not self.opts['skipFiniteCheck'], out=self._lineSegments)
            self._lineSegmentsRendered = True
        return self._lineSegments.drawargs()

    def _getClosingSegments(self):
        if False:
            while True:
                i = 10
        segments = []
        if self.opts['fillLevel'] == 'enclosed':
            return segments
        baseline = self.opts['fillLevel']
        (x, y) = self.getData()
        (lx, rx) = x[[0, -1]]
        (ly, ry) = y[[0, -1]]
        if ry != baseline:
            segments.append(QtCore.QLineF(rx, ry, rx, baseline))
        segments.append(QtCore.QLineF(rx, baseline, lx, baseline))
        if ly != baseline:
            segments.append(QtCore.QLineF(lx, baseline, lx, ly))
        return segments

    def _getFillPath(self):
        if False:
            while True:
                i = 10
        if self.fillPath is not None:
            return self.fillPath
        path = QtGui.QPainterPath(self.getPath())
        self.fillPath = path
        if self.opts['fillLevel'] == 'enclosed':
            return path
        baseline = self.opts['fillLevel']
        (x, y) = self.getData()
        (lx, rx) = x[[0, -1]]
        (ly, ry) = y[[0, -1]]
        if ry != baseline:
            path.lineTo(rx, baseline)
        path.lineTo(lx, baseline)
        if ly != baseline:
            path.lineTo(lx, ly)
        return path

    def _shouldUseFillPathList(self):
        if False:
            while True:
                i = 10
        connect = self.opts['connect']
        return isinstance(connect, str) and connect == 'all' and isinstance(self.opts['fillLevel'], (int, float))

    def _getFillPathList(self, widget):
        if False:
            while True:
                i = 10
        if self._fillPathList is not None:
            return self._fillPathList
        (x, y) = self.getData()
        if self.opts['stepMode']:
            (x, y) = self._generateStepModeData(self.opts['stepMode'], x, y, baseline=None)
        if not self.opts['skipFiniteCheck']:
            mask = np.isfinite(x) & np.isfinite(y)
            if not mask.all():
                x = x[mask]
                y = y[mask]
        if len(x) < 2:
            return []
        chunksize = 50 if not isinstance(widget, QtWidgets.QOpenGLWidget) else 5000
        paths = self._fillPathList = []
        offset = 0
        xybuf = np.empty((chunksize + 3, 2))
        baseline = self.opts['fillLevel']
        while offset < len(x) - 1:
            subx = x[offset:offset + chunksize]
            suby = y[offset:offset + chunksize]
            size = len(subx)
            xyview = xybuf[:size + 3]
            xyview[:-3, 0] = subx
            xyview[:-3, 1] = suby
            xyview[-3:, 0] = subx[[-1, 0, 0]]
            xyview[-3:, 1] = [baseline, baseline, suby[0]]
            offset += size - 1
            path = fn._arrayToQPath_all(xyview[:, 0], xyview[:, 1], finiteCheck=False)
            paths.append(path)
        return paths

    @debug.warnOnException
    def paint(self, p, opt, widget):
        if False:
            while True:
                i = 10
        profiler = debug.Profiler()
        if self.xData is None or len(self.xData) == 0:
            return
        if getConfigOption('enableExperimental'):
            if HAVE_OPENGL and isinstance(widget, QtWidgets.QOpenGLWidget):
                self.paintGL(p, opt, widget)
                return
        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', True)
        else:
            aa = self.opts['antialias']
        p.setRenderHint(p.RenderHint.Antialiasing, aa)
        cmode = self.opts['compositionMode']
        if cmode is not None:
            p.setCompositionMode(cmode)
        do_fill = self.opts['brush'] is not None and self.opts['fillLevel'] is not None
        do_fill_outline = do_fill and self.opts['fillOutline']
        if do_fill:
            if self._shouldUseFillPathList():
                paths = self._getFillPathList(widget)
            else:
                paths = [self._getFillPath()]
            profiler('generate fill path')
            for path in paths:
                p.fillPath(path, self.opts['brush'])
            profiler('draw fill path')
        if self.opts.get('shadowPen') is not None:
            if isinstance(self.opts.get('shadowPen'), QtGui.QPen):
                sp = self.opts['shadowPen']
            else:
                sp = fn.mkPen(self.opts['shadowPen'])
            if sp.style() != QtCore.Qt.PenStyle.NoPen:
                p.setPen(sp)
                if self._shouldUseDrawLineSegments(sp):
                    p.drawLines(*self._getLineSegments())
                    if do_fill_outline:
                        p.drawLines(self._getClosingSegments())
                elif do_fill_outline:
                    p.drawPath(self._getFillPath())
                else:
                    p.drawPath(self.getPath())
        cp = self.opts['pen']
        if not isinstance(cp, QtGui.QPen):
            cp = fn.mkPen(cp)
        p.setPen(cp)
        if self._shouldUseDrawLineSegments(cp):
            p.drawLines(*self._getLineSegments())
            if do_fill_outline:
                p.drawLines(self._getClosingSegments())
        elif do_fill_outline:
            p.drawPath(self._getFillPath())
        else:
            p.drawPath(self.getPath())
        profiler('drawPath')

    def paintGL(self, p, opt, widget):
        if False:
            print('Hello World!')
        p.beginNativePainting()
        import OpenGL.GL as gl
        if sys.platform == 'win32':
            gl.glMatrixMode(gl.GL_PROJECTION)
            gl.glLoadIdentity()
            gl.glOrtho(0, widget.width(), widget.height(), 0, -999999, 999999)
            gl.glMatrixMode(gl.GL_MODELVIEW)
            mat = QtGui.QMatrix4x4(self.sceneTransform())
            gl.glLoadMatrixf(np.array(mat.data(), dtype=np.float32))
        view = self.getViewBox()
        if view is not None:
            rect = view.mapRectToItem(self, view.boundingRect())
            gl.glEnable(gl.GL_STENCIL_TEST)
            gl.glColorMask(gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE, gl.GL_FALSE)
            gl.glDepthMask(gl.GL_FALSE)
            gl.glStencilFunc(gl.GL_NEVER, 1, 255)
            gl.glStencilOp(gl.GL_REPLACE, gl.GL_KEEP, gl.GL_KEEP)
            gl.glStencilMask(255)
            gl.glClear(gl.GL_STENCIL_BUFFER_BIT)
            gl.glBegin(gl.GL_TRIANGLES)
            gl.glVertex2f(rect.x(), rect.y())
            gl.glVertex2f(rect.x() + rect.width(), rect.y())
            gl.glVertex2f(rect.x(), rect.y() + rect.height())
            gl.glVertex2f(rect.x() + rect.width(), rect.y() + rect.height())
            gl.glVertex2f(rect.x() + rect.width(), rect.y())
            gl.glVertex2f(rect.x(), rect.y() + rect.height())
            gl.glEnd()
            gl.glColorMask(gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE, gl.GL_TRUE)
            gl.glDepthMask(gl.GL_TRUE)
            gl.glStencilMask(0)
            gl.glStencilFunc(gl.GL_EQUAL, 1, 255)
        try:
            (x, y) = self.getData()
            pos = np.empty((len(x), 2), dtype=np.float32)
            pos[:, 0] = x
            pos[:, 1] = y
            gl.glEnableClientState(gl.GL_VERTEX_ARRAY)
            try:
                gl.glVertexPointerf(pos)
                pen = fn.mkPen(self.opts['pen'])
                gl.glColor4f(*pen.color().getRgbF())
                width = pen.width()
                if pen.isCosmetic() and width < 1:
                    width = 1
                gl.glPointSize(width)
                gl.glLineWidth(width)
                if self._exportOpts is not False:
                    aa = self._exportOpts.get('antialias', True)
                else:
                    aa = self.opts['antialias']
                if aa:
                    gl.glEnable(gl.GL_LINE_SMOOTH)
                    gl.glEnable(gl.GL_BLEND)
                    gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
                    gl.glHint(gl.GL_LINE_SMOOTH_HINT, gl.GL_NICEST)
                else:
                    gl.glDisable(gl.GL_LINE_SMOOTH)
                gl.glDrawArrays(gl.GL_LINE_STRIP, 0, pos.shape[0])
            finally:
                gl.glDisableClientState(gl.GL_VERTEX_ARRAY)
        finally:
            p.endNativePainting()

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.xData = None
        self.yData = None
        self._lineSegments = None
        self._lineSegmentsRendered = False
        self.path = None
        self.fillPath = None
        self._fillPathList = None
        self._mouseShape = None
        self._mouseBounds = None
        self._boundsCache = [None, None]

    def mouseShape(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a QPainterPath representing the clickable shape of the curve\n\n        '
        if self._mouseShape is None:
            view = self.getViewBox()
            if view is None:
                return QtGui.QPainterPath()
            stroker = QtGui.QPainterPathStroker()
            path = self.getPath()
            path = self.mapToItem(view, path)
            stroker.setWidth(self.opts['mouseWidth'])
            mousePath = stroker.createStroke(path)
            self._mouseShape = self.mapFromItem(view, mousePath)
        return self._mouseShape

    def mouseClickEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        if not self.clickable or ev.button() != QtCore.Qt.MouseButton.LeftButton:
            return
        if self.mouseShape().contains(ev.pos()):
            ev.accept()
            self.sigClicked.emit(self, ev)

class ROIPlotItem(PlotCurveItem):
    """Plot curve that monitors an ROI and image for changes to automatically replot."""

    def __init__(self, roi, data, img, axes=(0, 1), xVals=None, color=None):
        if False:
            while True:
                i = 10
        self.roi = roi
        self.roiData = data
        self.roiImg = img
        self.axes = axes
        self.xVals = xVals
        PlotCurveItem.__init__(self, self.getRoiData(), x=self.xVals, color=color)
        roi.sigRegionChanged.connect(self.roiChangedEvent)

    def getRoiData(self):
        if False:
            print('Hello World!')
        d = self.roi.getArrayRegion(self.roiData, self.roiImg, axes=self.axes)
        if d is None:
            return
        while d.ndim > 1:
            d = d.mean(axis=1)
        return d

    def roiChangedEvent(self):
        if False:
            return 10
        d = self.getRoiData()
        self.updateData(d, self.xVals)