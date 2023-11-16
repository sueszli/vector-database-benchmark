import itertools
import math
import weakref
from collections import OrderedDict
import numpy as np
from .. import Qt, debug
from .. import functions as fn
from .. import getConfigOption
from ..Point import Point
from ..Qt import QtCore, QtGui
from .GraphicsObject import GraphicsObject
__all__ = ['ScatterPlotItem', 'SpotItem']
name_list = ['o', 's', 't', 't1', 't2', 't3', 'd', '+', 'x', 'p', 'h', 'star', 'arrow_up', 'arrow_right', 'arrow_down', 'arrow_left', 'crosshair']
Symbols = OrderedDict([(name, QtGui.QPainterPath()) for name in name_list])
Symbols['o'].addEllipse(QtCore.QRectF(-0.5, -0.5, 1, 1))
Symbols['s'].addRect(QtCore.QRectF(-0.5, -0.5, 1, 1))

def makeCrosshair(r=0.5, w=1, h=1):
    if False:
        return 10
    path = QtGui.QPainterPath()
    rect = QtCore.QRectF(-r, -r, r * 2, r * 2)
    path.addEllipse(rect)
    path.moveTo(-w, 0)
    path.lineTo(w, 0)
    path.moveTo(0, -h)
    path.lineTo(0, h)
    return path
Symbols['crosshair'] = makeCrosshair()
coords = {'t': [(-0.5, -0.5), (0, 0.5), (0.5, -0.5)], 't1': [(-0.5, 0.5), (0, -0.5), (0.5, 0.5)], 't2': [(-0.5, -0.5), (-0.5, 0.5), (0.5, 0)], 't3': [(0.5, 0.5), (0.5, -0.5), (-0.5, 0)], 'd': [(0.0, -0.5), (-0.4, 0.0), (0, 0.5), (0.4, 0)], '+': [(-0.5, -0.1), (-0.5, 0.1), (-0.1, 0.1), (-0.1, 0.5), (0.1, 0.5), (0.1, 0.1), (0.5, 0.1), (0.5, -0.1), (0.1, -0.1), (0.1, -0.5), (-0.1, -0.5), (-0.1, -0.1)], 'p': [(0, -0.5), (-0.4755, -0.1545), (-0.2939, 0.4045), (0.2939, 0.4045), (0.4755, -0.1545)], 'h': [(0.433, 0.25), (0.0, 0.5), (-0.433, 0.25), (-0.433, -0.25), (0, -0.5), (0.433, -0.25)], 'star': [(0, -0.5), (-0.1123, -0.1545), (-0.4755, -0.1545), (-0.1816, 0.059), (-0.2939, 0.4045), (0, 0.191), (0.2939, 0.4045), (0.1816, 0.059), (0.4755, -0.1545), (0.1123, -0.1545)], 'arrow_up': [(-0.125, 0.125), (0, 0), (0.125, 0.125), (0.05, 0.125), (0.05, 0.5), (-0.05, 0.5), (-0.05, 0.125)]}
for (k, c) in coords.items():
    Symbols[k].moveTo(*c[0])
    for (x, y) in c[1:]:
        Symbols[k].lineTo(x, y)
    Symbols[k].closeSubpath()
tr = QtGui.QTransform()
tr.rotate(45)
Symbols['x'] = tr.map(Symbols['+'])
tr.rotate(45)
Symbols['arrow_right'] = tr.map(Symbols['arrow_up'])
Symbols['arrow_down'] = tr.map(Symbols['arrow_right'])
Symbols['arrow_left'] = tr.map(Symbols['arrow_down'])
_DEFAULT_STYLE = {'symbol': None, 'size': -1, 'pen': None, 'brush': None, 'visible': True}

def drawSymbol(painter, symbol, size, pen, brush):
    if False:
        while True:
            i = 10
    if symbol is None:
        return
    painter.scale(size, size)
    painter.setPen(pen)
    painter.setBrush(brush)
    if isinstance(symbol, str):
        symbol = Symbols[symbol]
    if np.isscalar(symbol):
        symbol = list(Symbols.values())[symbol % len(Symbols)]
    painter.drawPath(symbol)

def renderSymbol(symbol, size, pen, brush, device=None, dpr=1.0):
    if False:
        return 10
    '\n    Render a symbol specification to QImage.\n    Symbol may be either a QPainterPath or one of the keys in the Symbols dict.\n    If *device* is None, a new QPixmap will be returned. Otherwise,\n    the symbol will be rendered into the device specified (See QPainter documentation\n    for more information).\n    '
    penPxWidth = max(math.ceil(pen.widthF()), 1)
    if device is None:
        side = int(math.ceil(dpr * (size + penPxWidth)))
        device = QtGui.QImage(side, side, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
        device.setDevicePixelRatio(dpr)
        device.fill(QtCore.Qt.GlobalColor.transparent)
    p = QtGui.QPainter(device)
    try:
        p.setRenderHint(p.RenderHint.Antialiasing)
        p.translate(device.width() / dpr * 0.5, device.height() / dpr * 0.5)
        drawSymbol(p, symbol, size, pen, brush)
    finally:
        p.end()
    return device

def _mkPen(*args, **kwargs):
    if False:
        i = 10
        return i + 15
    '\n    Wrapper for fn.mkPen which avoids creating a new QPen object if passed one as its\n    sole argument. This is used to avoid unnecessary cache misses in SymbolAtlas which\n    uses the QPen object id in its key.\n    '
    if len(args) == 1 and isinstance(args[0], QtGui.QPen):
        return args[0]
    else:
        return fn.mkPen(*args, **kwargs)

def _mkBrush(*args, **kwargs):
    if False:
        while True:
            i = 10
    '\n    Wrapper for fn.mkBrush which avoids creating a new QBrush object if passed one as its\n    sole argument. This is used to avoid unnecessary cache misses in SymbolAtlas which\n    uses the QBrush object id in its key.\n    '
    if len(args) == 1 and isinstance(args[0], QtGui.QBrush):
        return args[0]
    else:
        return fn.mkBrush(*args, **kwargs)

class SymbolAtlas(object):
    """
    Used to efficiently construct a single QPixmap containing all rendered symbols
    for a ScatterPlotItem. This is required for fragment rendering.

    Use example:
        atlas = SymbolAtlas()
        sc1 = atlas[[('o', 5, QPen(..), QBrush(..))]]
        sc2 = atlas[[('t', 10, QPen(..), QBrush(..))]]
        pm = atlas.pixmap

    """
    _idGenerator = itertools.count()

    def __init__(self):
        if False:
            print('Hello World!')
        self._dpr = 1.0
        self.clear()

    def __getitem__(self, styles):
        if False:
            i = 10
            return i + 15
        '\n        Given a list of tuples, (symbol, size, pen, brush), return a list of coordinates of\n        corresponding symbols within the atlas. Note that these coordinates may change if the atlas is rebuilt.\n        '
        keys = self._keys(styles)
        new = {key: style for (key, style) in zip(keys, styles) if key not in self._coords}
        if new:
            self._extend(new)
        return list(map(self._coords.__getitem__, keys))

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self._coords)

    def devicePixelRatio(self):
        if False:
            for i in range(10):
                print('nop')
        return self._dpr

    def setDevicePixelRatio(self, dpr):
        if False:
            return 10
        self._dpr = dpr

    @property
    def pixmap(self):
        if False:
            i = 10
            return i + 15
        if self._pixmap is None:
            self._pixmap = self._createPixmap()
        return self._pixmap

    @property
    def maxWidth(self):
        if False:
            while True:
                i = 10
        return self._maxWidth / self._dpr

    def rebuild(self, styles=None):
        if False:
            return 10
        profiler = debug.Profiler()
        if styles is None:
            data = []
        else:
            keys = set(self._keys(styles))
            data = list(self._itemData(keys))
        self.clear()
        if data:
            self._extendFromData(data)

    def clear(self):
        if False:
            return 10
        self._data = np.zeros((0, 0, 4), dtype=np.ubyte)
        self._coords = {}
        self._pixmap = None
        self._maxWidth = 0
        self._totalWidth = 0
        self._totalArea = 0
        self._pos = (0, 0)
        self._rowShape = (0, 0)

    def diagnostics(self):
        if False:
            return 10
        n = len(self)
        (w, h, _) = self._data.shape
        a = self._totalArea
        return dict(count=n, width=w, height=h, area=w * h, area_used=1.0 if n == 0 else a / (w * h), squareness=1.0 if n == 0 else 2 * w * h / (w ** 2 + h ** 2))

    def _keys(self, styles):
        if False:
            while True:
                i = 10

        def getId(obj):
            if False:
                while True:
                    i = 10
            try:
                return obj._id
            except AttributeError:
                obj._id = next(SymbolAtlas._idGenerator)
                return obj._id
        return [(symbol if isinstance(symbol, (str, int)) else getId(symbol), size, getId(pen), getId(brush)) for (symbol, size, pen, brush) in styles]

    def _itemData(self, keys):
        if False:
            i = 10
            return i + 15
        for key in keys:
            (y, x, h, w) = self._coords[key]
            yield (key, self._data[x:x + w, y:y + h])

    def _extend(self, styles):
        if False:
            while True:
                i = 10
        profiler = debug.Profiler()
        images = []
        data = []
        for (key, style) in styles.items():
            img = renderSymbol(*style, dpr=self._dpr)
            arr = fn.ndarray_from_qimage(img)
            images.append(img)
            data.append((key, arr))
        profiler('render')
        self._extendFromData(data)
        profiler('insert')

    def _extendFromData(self, data):
        if False:
            i = 10
            return i + 15
        self._pack(data)
        (wNew, hNew) = self._minDataShape()
        (wOld, hOld, _) = self._data.shape
        if wNew > wOld or hNew > hOld:
            arr = np.zeros((wNew, hNew, 4), dtype=np.ubyte)
            arr[:wOld, :hOld] = self._data
            self._data = arr
        for (key, arr) in data:
            (y, x, h, w) = self._coords[key]
            self._data[x:x + w, y:y + h] = arr
        self._pixmap = None

    def _pack(self, data):
        if False:
            for i in range(10):
                print('nop')
        n = len(self)
        wMax = self._maxWidth
        wSum = self._totalWidth
        aSum = self._totalArea
        (x, y) = self._pos
        (wRow, hRow) = self._rowShape
        for (_, arr) in data:
            (w, h, _) = arr.shape
            wMax = max(w, wMax)
            wSum += w
            aSum += w * h
        n += len(data)
        wRowEst = int(wSum / n ** 0.5)
        if wRowEst > 2 * wRow:
            wRow = wRowEst
        wRow = max(wMax, wRow)
        for (key, arr) in sorted(data, key=lambda data: data[1].shape[1]):
            (w, h, _) = arr.shape
            if x + w > wRow:
                x = 0
                y += hRow
                hRow = h
            hRow = max(h, hRow)
            self._coords[key] = (y, x, h, w)
            x += w
        self._maxWidth = wMax
        self._totalWidth = wSum
        self._totalArea = aSum
        self._pos = (x, y)
        self._rowShape = (wRow, hRow)

    def _minDataShape(self):
        if False:
            i = 10
            return i + 15
        (x, y) = self._pos
        (w, h) = self._rowShape
        return (int(w), int(y + h))

    def _createPixmap(self):
        if False:
            print('Hello World!')
        profiler = debug.Profiler()
        if self._data.size == 0:
            pm = QtGui.QPixmap(0, 0)
        else:
            img = fn.ndarray_to_qimage(self._data, QtGui.QImage.Format.Format_ARGB32_Premultiplied)
            pm = QtGui.QPixmap(img)
        return pm

class ScatterPlotItem(GraphicsObject):
    """
    Displays a set of x/y points. Instances of this class are created
    automatically as part of PlotDataItem; these rarely need to be instantiated
    directly.

    The size, shape, pen, and fill brush may be set for each point individually
    or for all points.


    ============================  ===============================================
    **Signals:**
    sigPlotChanged(self)          Emitted when the data being plotted has changed
    sigClicked(self, points, ev)  Emitted when points are clicked. Sends a list
                                  of all the points under the mouse pointer.
    sigHovered(self, points, ev)  Emitted when the item is hovered. Sends a list
                                  of all the points under the mouse pointer.
    ============================  ===============================================

    """
    sigClicked = QtCore.Signal(object, object, object)
    sigHovered = QtCore.Signal(object, object, object)
    sigPlotChanged = QtCore.Signal(object)

    def __init__(self, *args, **kargs):
        if False:
            return 10
        '\n        Accepts the same arguments as setData()\n        '
        profiler = debug.Profiler()
        GraphicsObject.__init__(self)
        self.picture = None
        self.fragmentAtlas = SymbolAtlas()
        dtype = [('x', float), ('y', float), ('size', float), ('symbol', object), ('pen', object), ('brush', object), ('visible', bool), ('data', object), ('hovered', bool), ('item', object), ('sourceRect', [('x', int), ('y', int), ('w', int), ('h', int)])]
        self.data = np.empty(0, dtype=dtype)
        self.bounds = [None, None]
        self._maxSpotWidth = 0
        self._maxSpotPxWidth = 0
        self._pixmapFragments = Qt.internals.PrimitiveArray(QtGui.QPainter.PixmapFragment, 10)
        self.opts = {'pxMode': True, 'useCache': True, 'antialias': getConfigOption('antialias'), 'compositionMode': None, 'name': None, 'symbol': 'o', 'size': 7, 'pen': fn.mkPen(getConfigOption('foreground')), 'brush': fn.mkBrush(100, 100, 150), 'hoverable': False, 'tip': 'x: {x:.3g}\ny: {y:.3g}\ndata={data}'.format}
        self.opts.update({'hover' + opt.title(): _DEFAULT_STYLE[opt] for opt in ['symbol', 'size', 'pen', 'brush']})
        profiler()
        self.setData(*args, **kargs)
        profiler('setData')
        self._toolTipCleared = True

    def setData(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        "\n        **Ordered Arguments:**\n\n        * If there is only one unnamed argument, it will be interpreted like the 'spots' argument.\n        * If there are two unnamed arguments, they will be interpreted as sequences of x and y values.\n\n        ====================== ===============================================================================================\n        **Keyword Arguments:**\n        *spots*                Optional list of dicts. Each dict specifies parameters for a single spot:\n                               {'pos': (x,y), 'size', 'pen', 'brush', 'symbol'}. This is just an alternate method\n                               of passing in data for the corresponding arguments.\n        *x*,*y*                1D arrays of x,y values.\n        *pos*                  2D structure of x,y pairs (such as Nx2 array or list of tuples)\n        *pxMode*               If True, spots are always the same size regardless of scaling, and size is given in px.\n                               Otherwise, size is in scene coordinates and the spots scale with the view. To ensure\n                               effective caching, QPen and QBrush objects should be reused as much as possible.\n                               Default is True\n        *symbol*               can be one (or a list) of symbols. For a list of supported symbols, see \n                               :func:`~ScatterPlotItem.setSymbol`. QPainterPath is also supported to specify custom symbol\n                               shapes. To properly obey the position and size, custom symbols should be centered at (0,0) and\n                               width and height of 1.0. Note that it is also possible to 'install' custom shapes by setting \n                               ScatterPlotItem.Symbols[key] = shape.\n        *pen*                  The pen (or list of pens) to use for drawing spot outlines.\n        *brush*                The brush (or list of brushes) to use for filling spots.\n        *size*                 The size (or list of sizes) of spots. If *pxMode* is True, this value is in pixels. Otherwise,\n                               it is in the item's local coordinate system.\n        *data*                 a list of python objects used to uniquely identify each spot.\n        *hoverable*            If True, sigHovered is emitted with a list of hovered points, a tool tip is shown containing\n                               information about them, and an optional separate style for them is used. Default is False.\n        *tip*                  A string-valued function of a spot's (x, y, data) values. Set to None to prevent a tool tip\n                               from being shown.\n        *hoverSymbol*          A single symbol to use for hovered spots. Set to None to keep symbol unchanged. Default is None.\n        *hoverSize*            A single size to use for hovered spots. Set to -1 to keep size unchanged. Default is -1.\n        *hoverPen*             A single pen to use for hovered spots. Set to None to keep pen unchanged. Default is None.\n        *hoverBrush*           A single brush to use for hovered spots. Set to None to keep brush unchanged. Default is None.\n        *useCache*             (bool) By default, generated point graphics items are cached to\n                               improve performance. Setting this to False can improve image quality\n                               in certain situations.\n        *antialias*            Whether to draw symbols with antialiasing. Note that if pxMode is True, symbols are\n                               always rendered with antialiasing (since the rendered symbols can be cached, this\n                               incurs very little performance cost)\n        *compositionMode*      If specified, this sets the composition mode used when drawing the\n                               scatter plot (see QPainter::CompositionMode in the Qt documentation).\n        *name*                 The name of this item. Names are used for automatically\n                               generating LegendItem entries and by some exporters.\n        ====================== ===============================================================================================\n        "
        oldData = self.data
        self.clear()
        self.addPoints(*args, **kargs)

    def addPoints(self, *args, **kargs):
        if False:
            print('Hello World!')
        '\n        Add new points to the scatter plot.\n        Arguments are the same as setData()\n        '
        if len(args) == 1:
            kargs['spots'] = args[0]
        elif len(args) == 2:
            kargs['x'] = args[0]
            kargs['y'] = args[1]
        elif len(args) > 2:
            raise Exception('Only accepts up to two non-keyword arguments.')
        if 'pos' in kargs:
            pos = kargs['pos']
            if isinstance(pos, np.ndarray):
                kargs['x'] = pos[:, 0]
                kargs['y'] = pos[:, 1]
            else:
                x = []
                y = []
                for p in pos:
                    if isinstance(p, QtCore.QPointF):
                        x.append(p.x())
                        y.append(p.y())
                    else:
                        x.append(p[0])
                        y.append(p[1])
                kargs['x'] = x
                kargs['y'] = y
        if 'spots' in kargs:
            numPts = len(kargs['spots'])
        elif 'y' in kargs and kargs['y'] is not None:
            numPts = len(kargs['y'])
        else:
            kargs['x'] = []
            kargs['y'] = []
            numPts = 0
        self.data['item'][...] = None
        oldData = self.data
        self.data = np.empty(len(oldData) + numPts, dtype=self.data.dtype)
        self.data[:len(oldData)] = oldData
        newData = self.data[len(oldData):]
        newData['size'] = -1
        newData['visible'] = True
        if 'spots' in kargs:
            spots = kargs['spots']
            for i in range(len(spots)):
                spot = spots[i]
                for k in spot:
                    if k == 'pos':
                        pos = spot[k]
                        if isinstance(pos, QtCore.QPointF):
                            (x, y) = (pos.x(), pos.y())
                        else:
                            (x, y) = (pos[0], pos[1])
                        newData[i]['x'] = x
                        newData[i]['y'] = y
                    elif k == 'pen':
                        newData[i][k] = _mkPen(spot[k])
                    elif k == 'brush':
                        newData[i][k] = _mkBrush(spot[k])
                    elif k in ['x', 'y', 'size', 'symbol', 'data']:
                        newData[i][k] = spot[k]
                    else:
                        raise Exception('Unknown spot parameter: %s' % k)
        elif 'y' in kargs:
            newData['x'] = kargs['x']
            newData['y'] = kargs['y']
        if 'name' in kargs:
            self.opts['name'] = kargs['name']
        if 'pxMode' in kargs:
            self.setPxMode(kargs['pxMode'])
        if 'antialias' in kargs:
            self.opts['antialias'] = kargs['antialias']
        if 'hoverable' in kargs:
            self.opts['hoverable'] = bool(kargs['hoverable'])
        if 'tip' in kargs:
            self.opts['tip'] = kargs['tip']
        if 'useCache' in kargs:
            self.opts['useCache'] = kargs['useCache']
        for k in ['pen', 'brush', 'symbol', 'size']:
            if k in kargs:
                setMethod = getattr(self, 'set' + k[0].upper() + k[1:])
                setMethod(kargs[k], update=False, dataSet=newData, mask=kargs.get('mask', None))
            kh = 'hover' + k.title()
            if kh in kargs:
                vh = kargs[kh]
                if k == 'pen':
                    vh = _mkPen(vh)
                elif k == 'brush':
                    vh = _mkBrush(vh)
                self.opts[kh] = vh
        if 'data' in kargs:
            self.setPointData(kargs['data'], dataSet=newData)
        self.prepareGeometryChange()
        self.informViewBoundsChanged()
        self.bounds = [None, None]
        self.invalidate()
        self.updateSpots(newData)
        self.sigPlotChanged.emit(self)

    def invalidate(self):
        if False:
            return 10
        self.picture = None
        self.update()

    def getData(self):
        if False:
            while True:
                i = 10
        return (self.data['x'], self.data['y'])

    def implements(self, interface=None):
        if False:
            return 10
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        if False:
            i = 10
            return i + 15
        return self.opts.get('name', None)

    def setPen(self, *args, **kargs):
        if False:
            print('Hello World!')
        'Set the pen(s) used to draw the outline around each spot.\n        If a list or array is provided, then the pen for each spot will be set separately.\n        Otherwise, the arguments are passed to pg.mkPen and used as the default pen for\n        all spots which do not have a pen explicitly set.'
        update = kargs.pop('update', True)
        dataSet = kargs.pop('dataSet', self.data)
        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            pens = args[0]
            if 'mask' in kargs and kargs['mask'] is not None:
                pens = pens[kargs['mask']]
            if len(pens) != len(dataSet):
                raise Exception('Number of pens does not match number of points (%d != %d)' % (len(pens), len(dataSet)))
            dataSet['pen'] = list(map(_mkPen, pens))
        else:
            self.opts['pen'] = _mkPen(*args, **kargs)
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setBrush(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        'Set the brush(es) used to fill the interior of each spot.\n        If a list or array is provided, then the brush for each spot will be set separately.\n        Otherwise, the arguments are passed to pg.mkBrush and used as the default brush for\n        all spots which do not have a brush explicitly set.'
        update = kargs.pop('update', True)
        dataSet = kargs.pop('dataSet', self.data)
        if len(args) == 1 and (isinstance(args[0], np.ndarray) or isinstance(args[0], list)):
            brushes = args[0]
            if 'mask' in kargs and kargs['mask'] is not None:
                brushes = brushes[kargs['mask']]
            if len(brushes) != len(dataSet):
                raise Exception('Number of brushes does not match number of points (%d != %d)' % (len(brushes), len(dataSet)))
            dataSet['brush'] = list(map(_mkBrush, brushes))
        else:
            self.opts['brush'] = _mkBrush(*args, **kargs)
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setSymbol(self, symbol, update=True, dataSet=None, mask=None):
        if False:
            for i in range(10):
                print('nop')
        "Set the symbol(s) used to draw each spot.\n        If a list or array is provided, then the symbol for each spot will be set separately.\n        Otherwise, the argument will be used as the default symbol for\n        all spots which do not have a symbol explicitly set.\n\n        **Supported symbols:**\n\n        * 'o'  circle (default)\n        * 's'  square\n        * 't'  triangle\n        * 'd'  diamond\n        * '+'  plus\n        * 't1' triangle pointing upwards\n        * 't2'  triangle pointing right side\n        * 't3'  triangle pointing left side\n        * 'p'  pentagon\n        * 'h'  hexagon\n        * 'star'\n        * 'x'  cross\n        * 'arrow_up'\n        * 'arrow_right'\n        * 'arrow_down'\n        * 'arrow_left'\n        * 'crosshair'\n        * any QPainterPath to specify custom symbol shapes.\n\n        "
        if dataSet is None:
            dataSet = self.data
        if isinstance(symbol, np.ndarray) or isinstance(symbol, list):
            symbols = symbol
            if mask is not None:
                symbols = symbols[mask]
            if len(symbols) != len(dataSet):
                raise Exception('Number of symbols does not match number of points (%d != %d)' % (len(symbols), len(dataSet)))
            dataSet['symbol'] = symbols
        else:
            self.opts['symbol'] = symbol
            self._spotPixmap = None
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setSize(self, size, update=True, dataSet=None, mask=None):
        if False:
            return 10
        'Set the size(s) used to draw each spot.\n        If a list or array is provided, then the size for each spot will be set separately.\n        Otherwise, the argument will be used as the default size for\n        all spots which do not have a size explicitly set.'
        if dataSet is None:
            dataSet = self.data
        if isinstance(size, np.ndarray) or isinstance(size, list):
            sizes = size
            if mask is not None:
                sizes = sizes[mask]
            if len(sizes) != len(dataSet):
                raise Exception('Number of sizes does not match number of points (%d != %d)' % (len(sizes), len(dataSet)))
            dataSet['size'] = sizes
        else:
            self.opts['size'] = size
            self._spotPixmap = None
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setPointsVisible(self, visible, update=True, dataSet=None, mask=None):
        if False:
            for i in range(10):
                print('nop')
        'Set whether or not each spot is visible.\n        If a list or array is provided, then the visibility for each spot will be set separately.\n        Otherwise, the argument will be used for all spots.'
        if dataSet is None:
            dataSet = self.data
        if isinstance(visible, np.ndarray) or isinstance(visible, list):
            visibilities = visible
            if mask is not None:
                visibilities = visibilities[mask]
            if len(visibilities) != len(dataSet):
                raise Exception('Number of visibilities does not match number of points (%d != %d)' % (len(visibilities), len(dataSet)))
            dataSet['visible'] = visibilities
        else:
            dataSet['visible'] = visible
        dataSet['sourceRect'] = 0
        if update:
            self.updateSpots(dataSet)

    def setPointData(self, data, dataSet=None, mask=None):
        if False:
            return 10
        if dataSet is None:
            dataSet = self.data
        if isinstance(data, np.ndarray) or isinstance(data, list):
            if mask is not None:
                data = data[mask]
            if len(data) != len(dataSet):
                raise Exception('Length of meta data does not match number of points (%d != %d)' % (len(data), len(dataSet)))
        if isinstance(data, np.ndarray) and data.dtype.fields is not None and (len(data.dtype.fields) > 1):
            for (i, rec) in enumerate(data):
                dataSet['data'][i] = rec
        else:
            dataSet['data'] = data

    def setPxMode(self, mode):
        if False:
            for i in range(10):
                print('nop')
        if self.opts['pxMode'] == mode:
            return
        self.opts['pxMode'] = mode
        self.invalidate()

    def updateSpots(self, dataSet=None):
        if False:
            return 10
        profiler = debug.Profiler()
        if dataSet is None:
            dataSet = self.data
        invalidate = False
        if self.opts['pxMode'] and self.opts['useCache']:
            mask = dataSet['sourceRect']['w'] == 0
            if np.any(mask):
                invalidate = True
                coords = self.fragmentAtlas[list(zip(*self._style(['symbol', 'size', 'pen', 'brush'], data=dataSet, idx=mask)))]
                dataSet['sourceRect'][mask] = coords
            self._maybeRebuildAtlas()
        else:
            invalidate = True
        self._updateMaxSpotSizes(data=dataSet)
        if invalidate:
            self.invalidate()

    def _maybeRebuildAtlas(self, threshold=4, minlen=1000):
        if False:
            i = 10
            return i + 15
        n = len(self.fragmentAtlas)
        if n > minlen and n > threshold * len(self.data):
            self.fragmentAtlas.rebuild(list(zip(*self._style(['symbol', 'size', 'pen', 'brush']))))
            self.data['sourceRect'] = 0
            self.updateSpots()

    def _style(self, opts, data=None, idx=None, scale=None):
        if False:
            while True:
                i = 10
        if data is None:
            data = self.data
        if idx is None:
            idx = np.s_[:]
        for opt in opts:
            col = data[opt][idx]
            if col.base is not None:
                col = col.copy()
            if self.opts['hoverable']:
                val = self.opts['hover' + opt.title()]
                if val != _DEFAULT_STYLE[opt]:
                    col[data['hovered'][idx]] = val
            col[np.equal(col, _DEFAULT_STYLE[opt])] = self.opts[opt]
            if opt == 'size' and scale is not None:
                col *= scale
            yield col

    def _updateMaxSpotSizes(self, **kwargs):
        if False:
            return 10
        if self.opts['pxMode'] and self.opts['useCache']:
            (w, pw) = (0, self.fragmentAtlas.maxWidth)
        else:
            (w, pw) = max(itertools.chain([(self._maxSpotWidth, self._maxSpotPxWidth)], self._measureSpotSizes(**kwargs)))
        self._maxSpotWidth = w
        self._maxSpotPxWidth = pw
        self.bounds = [None, None]

    def _measureSpotSizes(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Generate pairs (width, pxWidth) for spots in data'
        styles = zip(*self._style(['size', 'pen'], **kwargs))
        if self.opts['pxMode']:
            for (size, pen) in styles:
                yield (0, size + pen.widthF())
        else:
            for (size, pen) in styles:
                if pen.isCosmetic():
                    yield (size, pen.widthF())
                else:
                    yield (size + pen.widthF(), 0)

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        'Remove all spots from the scatter plot'
        self._maxSpotWidth = 0
        self._maxSpotPxWidth = 0
        self.data = np.empty(0, dtype=self.data.dtype)
        self.bounds = [None, None]
        self.invalidate()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if False:
            return 10
        if frac >= 1.0 and orthoRange is None and (self.bounds[ax] is not None):
            return self.bounds[ax]
        if self.data is None or len(self.data) == 0:
            return (None, None)
        if ax == 0:
            d = self.data['x']
            d2 = self.data['y']
        elif ax == 1:
            d = self.data['y']
            d2 = self.data['x']
        else:
            raise ValueError('Invalid axis value')
        if orthoRange is not None:
            mask = (d2 >= orthoRange[0]) * (d2 <= orthoRange[1])
            d = d[mask]
            if d.size == 0:
                return (None, None)
        if frac >= 1.0:
            self.bounds[ax] = (np.nanmin(d) - self._maxSpotWidth * 0.7072, np.nanmax(d) + self._maxSpotWidth * 0.7072)
            return self.bounds[ax]
        elif frac <= 0.0:
            raise Exception("Value for parameter 'frac' must be > 0. (got %s)" % str(frac))
        else:
            mask = np.isfinite(d)
            d = d[mask]
            return np.percentile(d, [50 * (1 - frac), 50 * (1 + frac)])

    def pixelPadding(self):
        if False:
            return 10
        return self._maxSpotPxWidth * 0.7072

    def boundingRect(self):
        if False:
            return 10
        (xmn, xmx) = self.dataBounds(ax=0)
        (ymn, ymx) = self.dataBounds(ax=1)
        if xmn is None or xmx is None:
            xmn = 0
            xmx = 0
        if ymn is None or ymx is None:
            ymn = 0
            ymx = 0
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
        return QtCore.QRectF(xmn - px, ymn - py, 2 * px + xmx - xmn, 2 * py + ymx - ymn)

    def viewTransformChanged(self):
        if False:
            while True:
                i = 10
        self.prepareGeometryChange()
        GraphicsObject.viewTransformChanged(self)
        self.bounds = [None, None]

    def setExportMode(self, *args, **kwds):
        if False:
            for i in range(10):
                print('nop')
        GraphicsObject.setExportMode(self, *args, **kwds)
        self.invalidate()

    @debug.warnOnException
    def paint(self, p, option, widget):
        if False:
            return 10
        profiler = debug.Profiler()
        cmode = self.opts.get('compositionMode', None)
        if cmode is not None:
            p.setCompositionMode(cmode)
        if self._exportOpts is not False:
            aa = self._exportOpts.get('antialias', True)
            scale = self._exportOpts.get('resolutionScale', 1.0)
        else:
            aa = self.opts['antialias']
            scale = 1.0
        if self.opts['pxMode'] is True:
            viewMask = self._maskAt(self.viewRect())
            pts = np.vstack([self.data['x'], self.data['y']])
            pts = fn.transformCoordinates(p.transform(), pts)
            pts = fn.clip_array(pts, -2 ** 30, 2 ** 30)
            p.resetTransform()
            if self.opts['useCache'] and self._exportOpts is False:
                dpr = widget.devicePixelRatioF()
                if dpr != self.fragmentAtlas.devicePixelRatio():
                    self.fragmentAtlas.setDevicePixelRatio(dpr)
                    self.fragmentAtlas.clear()
                    self.data['sourceRect'] = 0
                    self.updateSpots()
                xy = pts[:, viewMask].T
                sr = self.data['sourceRect'][viewMask]
                self._pixmapFragments.resize(sr.size)
                frags = self._pixmapFragments.ndarray()
                frags[:, 0:2] = xy
                frags[:, 2:6] = np.frombuffer(sr, dtype=int).reshape((-1, 4))
                frags[:, 6:10] = [1 / dpr, 1 / dpr, 0.0, 1.0]
                profiler('prep')
                drawargs = self._pixmapFragments.drawargs()
                p.drawPixmapFragments(*drawargs, self.fragmentAtlas.pixmap)
                profiler('draw')
            else:
                p.setRenderHint(p.RenderHint.Antialiasing, aa)
                for (pt, style) in zip(pts[:, viewMask].T, zip(*self._style(['symbol', 'size', 'pen', 'brush'], idx=viewMask, scale=scale))):
                    p.resetTransform()
                    p.translate(*pt)
                    drawSymbol(p, *style)
        else:
            if self.picture is None:
                self.picture = QtGui.QPicture()
                p2 = QtGui.QPainter(self.picture)
                for (x, y, style) in zip(self.data['x'], self.data['y'], zip(*self._style(['symbol', 'size', 'pen', 'brush'], scale=scale))):
                    p2.resetTransform()
                    p2.translate(x, y)
                    drawSymbol(p2, *style)
                p2.end()
            p.setRenderHint(p.RenderHint.Antialiasing, aa)
            self.picture.play(p)

    def points(self):
        if False:
            print('Hello World!')
        m = np.equal(self.data['item'], None)
        for i in np.argwhere(m)[:, 0]:
            rec = self.data[i]
            if rec['item'] is None:
                rec['item'] = SpotItem(rec, self, i)
        return self.data['item']

    def pointsAt(self, pos):
        if False:
            print('Hello World!')
        return self.points()[self._maskAt(pos)][::-1]

    def _maskAt(self, obj):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a boolean mask indicating all points that overlap obj, a QPointF or QRectF.\n        '
        if isinstance(obj, QtCore.QPointF):
            l = r = obj.x()
            t = b = obj.y()
        elif isinstance(obj, QtCore.QRectF):
            l = obj.left()
            r = obj.right()
            t = obj.top()
            b = obj.bottom()
        else:
            raise TypeError
        if self.opts['pxMode'] and self.opts['useCache']:
            w = self.data['sourceRect']['w']
            h = self.data['sourceRect']['h']
        else:
            (s,) = self._style(['size'])
            w = h = s
        w = w / 2
        h = h / 2
        if self.opts['pxMode']:
            (px, py) = self.pixelVectors()
            try:
                px = 0 if px is None else px.length()
            except OverflowError:
                px = 0
            try:
                py = 0 if py is None else py.length()
            except OverflowError:
                py = 0
            w *= px
            h *= py
        return self.data['visible'] & (self.data['x'] + w > l) & (self.data['x'] - w < r) & (self.data['y'] + h > t) & (self.data['y'] - h < b)

    def mouseClickEvent(self, ev):
        if False:
            while True:
                i = 10
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            pts = self.pointsAt(ev.pos())
            if len(pts) > 0:
                self.ptsClicked = pts
                ev.accept()
                self.sigClicked.emit(self, self.ptsClicked, ev)
            else:
                ev.ignore()
        else:
            ev.ignore()

    def hoverEvent(self, ev):
        if False:
            i = 10
            return i + 15
        if self.opts['hoverable']:
            old = self.data['hovered']
            if ev.exit:
                new = np.zeros_like(self.data['hovered'])
            else:
                new = self._maskAt(ev.pos())
            if self._hasHoverStyle():
                self.data['sourceRect'][old ^ new] = 0
                self.data['hovered'] = new
                self.updateSpots()
            points = self.points()[new][::-1]
            vb = self.getViewBox()
            if vb is not None and self.opts['tip'] is not None:
                if len(points) > 0:
                    cutoff = 3
                    tip = [self.opts['tip'](x=pt.pos().x(), y=pt.pos().y(), data=pt.data()) for pt in points[:cutoff]]
                    if len(points) > cutoff:
                        tip.append('({} others...)'.format(len(points) - cutoff))
                    vb.setToolTip('\n\n'.join(tip))
                    self._toolTipCleared = False
                elif not self._toolTipCleared:
                    vb.setToolTip('')
                    self._toolTipCleared = True
            self.sigHovered.emit(self, points, ev)

    def _hasHoverStyle(self):
        if False:
            print('Hello World!')
        return any((self.opts['hover' + opt.title()] != _DEFAULT_STYLE[opt] for opt in ['symbol', 'size', 'pen', 'brush']))

class SpotItem(object):
    """
    Class referring to individual spots in a scatter plot.
    These can be retrieved by calling ScatterPlotItem.points() or
    by connecting to the ScatterPlotItem's click signals.
    """

    def __init__(self, data, plot, index):
        if False:
            print('Hello World!')
        self._data = data
        self._index = index
        self.__plot_ref = weakref.ref(plot)

    @property
    def _plot(self):
        if False:
            print('Hello World!')
        return self.__plot_ref()

    def data(self):
        if False:
            while True:
                i = 10
        'Return the user data associated with this spot.'
        return self._data['data']

    def index(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the index of this point as given in the scatter plot data.'
        return self._index

    def size(self):
        if False:
            i = 10
            return i + 15
        "Return the size of this spot.\n        If the spot has no explicit size set, then return the ScatterPlotItem's default size instead."
        if self._data['size'] == -1:
            return self._plot.opts['size']
        else:
            return self._data['size']

    def pos(self):
        if False:
            i = 10
            return i + 15
        return Point(self._data['x'], self._data['y'])

    def viewPos(self):
        if False:
            while True:
                i = 10
        return self._plot.mapToView(self.pos())

    def setSize(self, size):
        if False:
            print('Hello World!')
        "Set the size of this spot.\n        If the size is set to -1, then the ScatterPlotItem's default size\n        will be used instead."
        self._data['size'] = size
        self.updateItem()

    def symbol(self):
        if False:
            i = 10
            return i + 15
        "Return the symbol of this spot.\n        If the spot has no explicit symbol set, then return the ScatterPlotItem's default symbol instead.\n        "
        symbol = self._data['symbol']
        if symbol is None:
            symbol = self._plot.opts['symbol']
        try:
            n = int(symbol)
            symbol = list(Symbols.keys())[n % len(Symbols)]
        except:
            pass
        return symbol

    def setSymbol(self, symbol):
        if False:
            for i in range(10):
                print('nop')
        "Set the symbol for this spot.\n        If the symbol is set to '', then the ScatterPlotItem's default symbol will be used instead."
        self._data['symbol'] = symbol
        self.updateItem()

    def pen(self):
        if False:
            for i in range(10):
                print('nop')
        pen = self._data['pen']
        if pen is None:
            pen = self._plot.opts['pen']
        return fn.mkPen(pen)

    def setPen(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        'Set the outline pen for this spot'
        self._data['pen'] = _mkPen(*args, **kargs)
        self.updateItem()

    def resetPen(self):
        if False:
            i = 10
            return i + 15
        "Remove the pen set for this spot; the scatter plot's default pen will be used instead."
        self._data['pen'] = None
        self.updateItem()

    def brush(self):
        if False:
            i = 10
            return i + 15
        brush = self._data['brush']
        if brush is None:
            brush = self._plot.opts['brush']
        return fn.mkBrush(brush)

    def setBrush(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        'Set the fill brush for this spot'
        self._data['brush'] = _mkBrush(*args, **kargs)
        self.updateItem()

    def resetBrush(self):
        if False:
            while True:
                i = 10
        "Remove the brush set for this spot; the scatter plot's default brush will be used instead."
        self._data['brush'] = None
        self.updateItem()

    def isVisible(self):
        if False:
            i = 10
            return i + 15
        return self._data['visible']

    def setVisible(self, visible):
        if False:
            i = 10
            return i + 15
        'Set whether or not this spot is visible.'
        self._data['visible'] = visible
        self.updateItem()

    def setData(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Set the user-data associated with this spot'
        self._data['data'] = data

    def updateItem(self):
        if False:
            i = 10
            return i + 15
        self._data['sourceRect'] = (0, 0, 0, 0)
        self._plot.updateSpots(self._data.reshape(1))