import math
import warnings
import bisect
import numpy as np
from .. import debug as debug
from .. import functions as fn
from .. import getConfigOption
from ..Qt import QtCore
from .GraphicsObject import GraphicsObject
from .PlotCurveItem import PlotCurveItem
from .ScatterPlotItem import ScatterPlotItem
__all__ = ['PlotDataItem']

class PlotDataset(object):
    """
    :orphan:
    .. warning:: This class is intended for internal use. The interface may change without warning.

    Holds collected information for a plotable dataset. 
    Numpy arrays containing x and y coordinates are available as ``dataset.x`` and ``dataset.y``.
    
    After a search has been performed, typically during a call to :func:`dataRect() <pyqtgraph.PlotDataset.dataRect>`, 
    ``dataset.containsNonfinite`` is `True` if any coordinate values are nonfinite (e.g. NaN or inf) or `False` if all 
    values are finite. If no search has been performed yet, ``dataset.containsNonfinite`` is `None`.

    For internal use in :class:`PlotDataItem <pyqtgraph.PlotDataItem>`, this class should not be instantiated when no data is available. 
    """

    def __init__(self, x, y, xAllFinite=None, yAllFinite=None):
        if False:
            print('Hello World!')
        ' \n        Parameters\n        ----------\n        x: array\n            x coordinates of data points. \n        y: array\n            y coordinates of data points. \n        '
        super().__init__()
        self.x = x
        self.y = y
        self.xAllFinite = xAllFinite
        self.yAllFinite = yAllFinite
        self._dataRect = None
        if isinstance(x, np.ndarray) and x.dtype.kind in 'iu':
            self.xAllFinite = True
        if isinstance(y, np.ndarray) and y.dtype.kind in 'iu':
            self.yAllFinite = True

    @property
    def containsNonfinite(self):
        if False:
            print('Hello World!')
        if self.xAllFinite is None or self.yAllFinite is None:
            return None
        return not (self.xAllFinite and self.yAllFinite)

    def _updateDataRect(self):
        if False:
            i = 10
            return i + 15
        ' \n        Finds bounds of plotable data and stores them as ``dataset._dataRect``, \n        stores information about the presence of nonfinite data points.\n            '
        if self.y is None or self.x is None:
            return None
        (xmin, xmax, self.xAllFinite) = self._getArrayBounds(self.x, self.xAllFinite)
        (ymin, ymax, self.yAllFinite) = self._getArrayBounds(self.y, self.yAllFinite)
        self._dataRect = QtCore.QRectF(QtCore.QPointF(xmin, ymin), QtCore.QPointF(xmax, ymax))

    def _getArrayBounds(self, arr, all_finite):
        if False:
            print('Hello World!')
        if not all_finite:
            selection = np.isfinite(arr)
            all_finite = selection.all()
            if not all_finite:
                arr = arr[selection]
        try:
            amin = np.min(arr)
            amax = np.max(arr)
        except ValueError:
            amin = np.nan
            amax = np.nan
        return (amin, amax, all_finite)

    def dataRect(self):
        if False:
            return 10
        '\n        Returns a bounding rectangle (as :class:`QtCore.QRectF`) for the finite subset of data.\n        If there is an active mapping function, such as logarithmic scaling, then bounds represent the mapped data. \n        Will return `None` if there is no data or if all values (`x` or `y`) are NaN.\n        '
        if self._dataRect is None:
            self._updateDataRect()
        return self._dataRect

    def applyLogMapping(self, logMode):
        if False:
            i = 10
            return i + 15
        '\n        Applies a logarithmic mapping transformation (base 10) if requested for the respective axis.\n        This replaces the internal data. Values of ``-inf`` resulting from zeros in the original dataset are\n        replaced by ``np.NaN``.\n        \n        Parameters\n        ----------\n        logmode: tuple or list of two bool\n            A `True` value requests log-scale mapping for the x and y axis (in this order).\n        '
        if logMode[0]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.x = np.log10(self.x)
            nonfinites = ~np.isfinite(self.x)
            if nonfinites.any():
                self.x[nonfinites] = np.nan
                all_x_finite = False
            else:
                all_x_finite = True
            self.xAllFinite = all_x_finite
        if logMode[1]:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                self.y = np.log10(self.y)
            nonfinites = ~np.isfinite(self.y)
            if nonfinites.any():
                self.y[nonfinites] = np.nan
                all_y_finite = False
            else:
                all_y_finite = True
            self.yAllFinite = all_y_finite

class PlotDataItem(GraphicsObject):
    """
    **Bases:** :class:`GraphicsObject <pyqtgraph.GraphicsObject>`

    :class:`PlotDataItem` provides a unified interface for displaying plot curves, scatter plots, or both.
    It also contains methods to transform or decimate the original data before it is displayed. 

    As pyqtgraph's standard plotting object, ``plot()`` methods such as :func:`pyqtgraph.plot` and
    :func:`PlotItem.plot() <pyqtgraph.PlotItem.plot>` create instances of :class:`PlotDataItem`.

    While it is possible to use :class:`PlotCurveItem <pyqtgraph.PlotCurveItem>` or
    :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` individually, this is recommended only
    where performance is critical and the limited functionality of these classes is sufficient.

    ==================================  ==============================================
    **Signals:**
    sigPlotChanged(self)                Emitted when the data in this item is updated.
    sigClicked(self, ev)                Emitted when the item is clicked.
    sigPointsClicked(self, points, ev)  Emitted when a plot point is clicked
                                        Sends the list of points under the mouse.
    sigPointsHovered(self, points, ev)  Emitted when a plot point is hovered over.
                                        Sends the list of points under the mouse.
    ==================================  ==============================================
    """
    sigPlotChanged = QtCore.Signal(object)
    sigClicked = QtCore.Signal(object, object)
    sigPointsClicked = QtCore.Signal(object, object, object)
    sigPointsHovered = QtCore.Signal(object, object, object)

    def __init__(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        "\n        There are many different ways to create a PlotDataItem.\n\n        **Data initialization arguments:** (x,y data only)\n\n            ========================== =========================================\n            PlotDataItem(x, y)         x, y: array_like coordinate values\n            PlotDataItem(y)            y values only -- x will be\n                                       automatically set to ``range(len(y))``\n            PlotDataItem(x=x, y=y)     x and y given by keyword arguments\n            PlotDataItem(ndarray(N,2)) single numpy array with shape (N, 2),\n                                       where ``x=data[:,0]`` and ``y=data[:,1]``\n            ========================== =========================================\n\n        **Data initialization arguments:** (x,y data AND may include spot style)\n\n            ============================ ===============================================\n            PlotDataItem(recarray)       numpy record array with ``dtype=[('x', float),\n                                         ('y', float), ...]``\n            PlotDataItem(list-of-dicts)  ``[{'x': x, 'y': y, ...},   ...]``\n            PlotDataItem(dict-of-lists)  ``{'x': [...], 'y': [...],  ...}``\n            ============================ ===============================================\n        \n        **Line style keyword arguments:**\n\n            ============ ==============================================================================\n            connect      Specifies how / whether vertexes should be connected. See below for details.\n            pen          Pen to use for drawing the lines between points.\n                         Default is solid grey, 1px width. Use None to disable line drawing.\n                         May be a ``QPen`` or any single argument accepted by \n                         :func:`mkPen() <pyqtgraph.mkPen>`\n            shadowPen    Pen for secondary line to draw behind the primary line. Disabled by default.\n                         May be a ``QPen`` or any single argument accepted by \n                         :func:`mkPen() <pyqtgraph.mkPen>`\n            fillLevel    If specified, the area between the curve and fillLevel is filled.\n            fillOutline  (bool) If True, an outline surrounding the *fillLevel* area is drawn.\n            fillBrush    Fill to use in the *fillLevel* area. May be any single argument accepted by \n                         :func:`mkBrush() <pyqtgraph.mkBrush>`\n            stepMode     (str or None) If specified and not None, a stepped curve is drawn.\n                         For 'left' the specified points each describe the left edge of a step.\n                         For 'right', they describe the right edge. \n                         For 'center', the x coordinates specify the location of the step boundaries.\n                         This mode is commonly used for histograms. Note that it requires an additional\n                         x value, such that len(x) = len(y) + 1 .\n\n            ============ ==============================================================================\n        \n        ``connect`` supports the following arguments:\n        \n        - 'all' connects all points.  \n        - 'pairs' generates lines between every other point.\n        - 'finite' creates a break when a nonfinite points is encountered. \n        - If an ndarray is passed, it should contain `N` int32 values of 0 or 1.\n          Values of 1 indicate that the respective point will be connected to the next.\n        - In the default 'auto' mode, PlotDataItem will normally use 'all', but if any\n          nonfinite data points are detected, it will automatically switch to 'finite'.\n          \n        See :func:`arrayToQPath() <pyqtgraph.arrayToQPath>` for more details.\n        \n        **Point style keyword arguments:**  (see :func:`ScatterPlotItem.setData() <pyqtgraph.ScatterPlotItem.setData>` for more information)\n\n            ============ ======================================================\n            symbol       Symbol to use for drawing points, or a list of symbols\n                         for each. The default is no symbol.\n            symbolPen    Outline pen for drawing points, or a list of pens, one\n                         per point. May be any single argument accepted by\n                         :func:`mkPen() <pyqtgraph.mkPen>`.\n            symbolBrush  Brush for filling points, or a list of brushes, one \n                         per point. May be any single argument accepted by\n                         :func:`mkBrush() <pyqtgraph.mkBrush>`.\n            symbolSize   Diameter of symbols, or list of diameters.\n            pxMode       (bool) If True, then symbolSize is specified in\n                         pixels. If False, then symbolSize is\n                         specified in data coordinates.\n            ============ ======================================================\n            \n        Any symbol recognized by :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` can be specified,\n        including 'o' (circle), 's' (square), 't', 't1', 't2', 't3' (triangles of different orientation),\n        'd' (diamond), '+' (plus sign), 'x' (x mark), 'p' (pentagon), 'h' (hexagon) and 'star'.\n        \n        Symbols can also be directly given in the form of a :class:`QtGui.QPainterPath` instance.\n\n        **Optimization keyword arguments:**\n\n            ================= =======================================================================\n            useCache          (bool) By default, generated point graphics items are cached to\n                              improve performance. Setting this to False can improve image quality\n                              in certain situations.\n            antialias         (bool) By default, antialiasing is disabled to improve performance.\n                              Note that in some cases (in particular, when ``pxMode=True``), points\n                              will be rendered antialiased even if this is set to `False`.\n            downsample        (int) Reduce the number of samples displayed by the given factor.\n            downsampleMethod  'subsample': Downsample by taking the first of N samples.\n                              This method is fastest and least accurate.\n                              'mean': Downsample by taking the mean of N samples.\n                              'peak': Downsample by drawing a saw wave that follows the min\n                              and max of the original data. This method produces the best\n                              visual representation of the data but is slower.\n            autoDownsample    (bool) If `True`, resample the data before plotting to avoid plotting\n                              multiple line segments per pixel. This can improve performance when\n                              viewing very high-density data, but increases the initial overhead\n                              and memory usage.\n            clipToView        (bool) If `True`, only data visible within the X range of the containing\n                              :class:`ViewBox` is plotted. This can improve performance when plotting\n                              very large data sets where only a fraction of the data is visible\n                              at any time.\n            dynamicRangeLimit (float or `None`) Limit off-screen y positions of data points. \n                              `None` disables the limiting. This can increase performance but may\n                              cause plots to disappear at high levels of magnification.\n                              The default of 1e6 limits data to approximately 1,000,000 times the \n                              :class:`ViewBox` height.\n            dynamicRangeHyst  (float) Permits changes in vertical zoom up to the given hysteresis\n                              factor (the default is 3.0) before the limit calculation is repeated.\n            skipFiniteCheck   (bool, default `False`) Optimization flag that can speed up plotting by not \n                              checking and compensating for NaN values.  If set to `True`, and NaN \n                              values exist, unpredictable behavior will occur. The data may not be\n                              displayed or the plot may take a significant performance hit.\n                              \n                              In the default 'auto' connect mode, `PlotDataItem` will automatically\n                              override this setting.\n            ================= =======================================================================\n\n        **Meta-info keyword arguments:**\n\n            ==========   ================================================\n            name         (string) Name of item for use in the plot legend\n            ==========   ================================================\n\n        **Notes on performance:**\n        \n        Plotting lines with the default single-pixel width is the fastest available option. For such lines,\n        translucent colors (`alpha` < 1) do not result in a significant slowdown.\n        \n        Wider lines increase the complexity due to the overlap of individual line segments. Translucent colors\n        require merging the entire plot into a single entity before the alpha value can be applied. For plots with more\n        than a few hundred points, this can result in excessive slowdown.\n\n        Since version 0.12.4, this slowdown is automatically avoided by an algorithm that draws line segments\n        separately for fully opaque lines. Setting `alpha` < 1 reverts to the previous, slower drawing method.\n        \n        For lines with a width of more than 4 pixels, :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>` will automatically\n        create a ``QPen`` with `Qt.PenCapStyle.RoundCap` to ensure a smooth connection of line segments. This incurs a\n        small performance penalty.\n\n        "
        GraphicsObject.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemHasNoContents)
        self._dataset = None
        self._datasetMapped = None
        self._datasetDisplay = None
        self.curve = PlotCurveItem()
        self.scatter = ScatterPlotItem()
        self.curve.setParentItem(self)
        self.scatter.setParentItem(self)
        self.curve.sigClicked.connect(self.curveClicked)
        self.scatter.sigClicked.connect(self.scatterClicked)
        self.scatter.sigHovered.connect(self.scatterHovered)
        self.setProperty('xViewRangeWasChanged', False)
        self.setProperty('yViewRangeWasChanged', False)
        self.setProperty('styleWasChanged', True)
        self._drlLastClip = (0.0, 0.0)
        self.opts = {'connect': 'auto', 'skipFiniteCheck': False, 'fftMode': False, 'logMode': [False, False], 'derivativeMode': False, 'phasemapMode': False, 'alphaHint': 1.0, 'alphaMode': False, 'pen': (200, 200, 200), 'shadowPen': None, 'fillLevel': None, 'fillOutline': False, 'fillBrush': None, 'stepMode': None, 'symbol': None, 'symbolSize': 10, 'symbolPen': (200, 200, 200), 'symbolBrush': (50, 50, 150), 'pxMode': True, 'antialias': getConfigOption('antialias'), 'pointMode': None, 'useCache': True, 'downsample': 1, 'autoDownsample': False, 'downsampleMethod': 'peak', 'autoDownsampleFactor': 5.0, 'clipToView': False, 'dynamicRangeLimit': 1000000.0, 'dynamicRangeHyst': 3.0, 'data': None}
        self.setCurveClickable(kargs.get('clickable', False))
        self.setData(*args, **kargs)

    @property
    def xData(self):
        if False:
            for i in range(10):
                print('nop')
        if self._dataset is None:
            return None
        return self._dataset.x

    @property
    def yData(self):
        if False:
            for i in range(10):
                print('nop')
        if self._dataset is None:
            return None
        return self._dataset.y

    def implements(self, interface=None):
        if False:
            while True:
                i = 10
        ints = ['plotData']
        if interface is None:
            return ints
        return interface in ints

    def name(self):
        if False:
            return 10
        ' Returns the name that represents this item in the legend. '
        return self.opts.get('name', None)

    def setCurveClickable(self, state, width=None):
        if False:
            i = 10
            return i + 15
        ' ``state=True`` sets the curve to be clickable, with a tolerance margin represented by `width`. '
        self.curve.setClickable(state, width)

    def curveClickable(self):
        if False:
            print('Hello World!')
        ' Returns `True` if the curve is set to be clickable. '
        return self.curve.clickable

    def boundingRect(self):
        if False:
            print('Hello World!')
        return QtCore.QRectF()

    def setPos(self, x, y):
        if False:
            return 10
        GraphicsObject.setPos(self, x, y)
        self.viewTransformChanged()
        self.viewRangeChanged()

    def setAlpha(self, alpha, auto):
        if False:
            return 10
        if self.opts['alphaHint'] == alpha and self.opts['alphaMode'] == auto:
            return
        self.opts['alphaHint'] = alpha
        self.opts['alphaMode'] = auto
        self.setOpacity(alpha)

    def setFftMode(self, state):
        if False:
            for i in range(10):
                print('nop')
        '\n        ``state = True`` enables mapping the data by a fast Fourier transform.\n        If the `x` values are not equidistant, the data set is resampled at\n        equal intervals. \n        '
        if self.opts['fftMode'] == state:
            return
        self.opts['fftMode'] = state
        self._datasetMapped = None
        self._datasetDisplay = None
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setLogMode(self, xState, yState):
        if False:
            print('Hello World!')
        '\n        When log mode is enabled for the respective axis by setting ``xState`` or \n        ``yState`` to `True`, a mapping according to ``mapped = np.log10( value )``\n        is applied to the data. For negative or zero values, this results in a \n        `NaN` value.\n        '
        if self.opts['logMode'] == [xState, yState]:
            return
        self.opts['logMode'] = [xState, yState]
        self._datasetMapped = None
        self._datasetDisplay = None
        self._adsLastValue = 1
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setDerivativeMode(self, state):
        if False:
            i = 10
            return i + 15
        '\n        ``state = True`` enables derivative mode, where a mapping according to\n        ``y_mapped = dy / dx`` is applied, with `dx` and `dy` representing the \n        differences between adjacent `x` and `y` values.\n        '
        if self.opts['derivativeMode'] == state:
            return
        self.opts['derivativeMode'] = state
        self._datasetMapped = None
        self._datasetDisplay = None
        self._adsLastValue = 1
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setPhasemapMode(self, state):
        if False:
            print('Hello World!')
        '\n        ``state = True`` enables phase map mode, where a mapping \n        according to ``x_mappped = y`` and ``y_mapped = dy / dx``\n        is applied, plotting the numerical derivative of the data over the \n        original `y` values.\n        '
        if self.opts['phasemapMode'] == state:
            return
        self.opts['phasemapMode'] = state
        self._datasetMapped = None
        self._datasetDisplay = None
        self._adsLastValue = 1
        self.updateItems(styleUpdate=False)
        self.informViewBoundsChanged()

    def setPen(self, *args, **kargs):
        if False:
            print('Hello World!')
        '\n        Sets the pen used to draw lines between points.\n        The argument can be a :class:`QtGui.QPen` or any combination of arguments accepted by \n        :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`.\n        '
        pen = fn.mkPen(*args, **kargs)
        self.opts['pen'] = pen
        self.updateItems(styleUpdate=True)

    def setShadowPen(self, *args, **kargs):
        if False:
            print('Hello World!')
        '\n        Sets the shadow pen used to draw lines between points (this is for enhancing contrast or\n        emphasizing data). This line is drawn behind the primary pen and should generally be assigned \n        greater width than the primary pen.\n        The argument can be a :class:`QtGui.QPen` or any combination of arguments accepted by \n        :func:`pyqtgraph.mkPen() <pyqtgraph.mkPen>`.\n        '
        if args and args[0] is None:
            pen = None
        else:
            pen = fn.mkPen(*args, **kargs)
        self.opts['shadowPen'] = pen
        self.updateItems(styleUpdate=True)

    def setFillBrush(self, *args, **kargs):
        if False:
            while True:
                i = 10
        ' \n        Sets the :class:`QtGui.QBrush` used to fill the area under the curve.\n        See :func:`mkBrush() <pyqtgraph.mkBrush>`) for arguments.\n        '
        if args and args[0] is None:
            brush = None
        else:
            brush = fn.mkBrush(*args, **kargs)
        if self.opts['fillBrush'] == brush:
            return
        self.opts['fillBrush'] = brush
        self.updateItems(styleUpdate=True)

    def setBrush(self, *args, **kargs):
        if False:
            return 10
        '\n        See :func:`~pyqtgraph.PlotDataItem.setFillBrush`\n        '
        return self.setFillBrush(*args, **kargs)

    def setFillLevel(self, level):
        if False:
            while True:
                i = 10
        '\n        Enables filling the area under the curve towards the value specified by \n        `level`. `None` disables the filling. \n        '
        if self.opts['fillLevel'] == level:
            return
        self.opts['fillLevel'] = level
        self.updateItems(styleUpdate=True)

    def setSymbol(self, symbol):
        if False:
            return 10
        ' `symbol` can be any string recognized by \n        :class:`ScatterPlotItem <pyqtgraph.ScatterPlotItem>` or a list that\n        specifies a symbol for each point.\n        '
        if self.opts['symbol'] == symbol:
            return
        self.opts['symbol'] = symbol
        self.updateItems(styleUpdate=True)

    def setSymbolPen(self, *args, **kargs):
        if False:
            return 10
        ' \n        Sets the :class:`QtGui.QPen` used to draw symbol outlines.\n        See :func:`mkPen() <pyqtgraph.mkPen>`) for arguments.\n        '
        pen = fn.mkPen(*args, **kargs)
        if self.opts['symbolPen'] == pen:
            return
        self.opts['symbolPen'] = pen
        self.updateItems(styleUpdate=True)

    def setSymbolBrush(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the :class:`QtGui.QBrush` used to fill symbols.\n        See :func:`mkBrush() <pyqtgraph.mkBrush>`) for arguments.\n        '
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['symbolBrush'] == brush:
            return
        self.opts['symbolBrush'] = brush
        self.updateItems(styleUpdate=True)

    def setSymbolSize(self, size):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets the symbol size.\n        '
        if self.opts['symbolSize'] == size:
            return
        self.opts['symbolSize'] = size
        self.updateItems(styleUpdate=True)

    def setDownsampling(self, ds=None, auto=None, method=None):
        if False:
            while True:
                i = 10
        "\n        Sets the downsampling mode of this item. Downsampling reduces the number\n        of samples drawn to increase performance.\n\n        ==============  =================================================================\n        **Arguments:**\n        ds              (int) Reduce visible plot samples by this factor. To disable,\n                        set ds=1.\n        auto            (bool) If True, automatically pick *ds* based on visible range\n        mode            'subsample': Downsample by taking the first of N samples.\n                        This method is fastest and least accurate.\n                        'mean': Downsample by taking the mean of N samples.\n                        'peak': Downsample by drawing a saw wave that follows the min\n                        and max of the original data. This method produces the best\n                        visual representation of the data but is slower.\n        ==============  =================================================================\n        "
        changed = False
        if ds is not None:
            if self.opts['downsample'] != ds:
                changed = True
                self.opts['downsample'] = ds
        if auto is not None and self.opts['autoDownsample'] != auto:
            self.opts['autoDownsample'] = auto
            changed = True
        if method is not None:
            if self.opts['downsampleMethod'] != method:
                changed = True
                self.opts['downsampleMethod'] = method
        if changed:
            self._datasetMapped = None
            self._datasetDisplay = None
            self._adsLastValue = 1
            self.updateItems(styleUpdate=False)

    def setClipToView(self, state):
        if False:
            i = 10
            return i + 15
        '\n        ``state=True`` enables clipping the displayed data set to the\n        visible x-axis range.\n        '
        if self.opts['clipToView'] == state:
            return
        self.opts['clipToView'] = state
        self._datasetDisplay = None
        self.updateItems(styleUpdate=False)

    def setDynamicRangeLimit(self, limit=1000000.0, hysteresis=3.0):
        if False:
            while True:
                i = 10
        "\n        Limit the off-screen positions of data points at large magnification\n        This avoids errors with plots not displaying because their visibility is incorrectly determined. \n        The default setting repositions far-off points to be within ±10^6 times the viewport height.\n\n        =============== ================================================================\n        **Arguments:**\n        limit           (float or None) Any data outside the range of limit * hysteresis\n                        will be constrained to the limit value limit.\n                        All values are relative to the viewport height.\n                        'None' disables the check for a minimal increase in performance.\n                        Default is 1E+06.\n                        \n        hysteresis      (float) Hysteresis factor that controls how much change\n                        in zoom level (vertical height) is allowed before recalculating\n                        Default is 3.0\n        =============== ================================================================\n        "
        if hysteresis < 1.0:
            hysteresis = 1.0
        self.opts['dynamicRangeHyst'] = hysteresis
        if limit == self.opts['dynamicRangeLimit']:
            return
        self.opts['dynamicRangeLimit'] = limit
        self._datasetDisplay = None
        self.updateItems(styleUpdate=False)

    def setSkipFiniteCheck(self, skipFiniteCheck):
        if False:
            return 10
        "\n        When it is known that the plot data passed to ``PlotDataItem`` contains only finite numerical values,\n        the ``skipFiniteCheck`` property can help speed up plotting. If this flag is set and the data contains \n        any non-finite values (such as `NaN` or `Inf`), unpredictable behavior will occur. The data might not\n        be plotted, or there migth be significant performance impact.\n        \n        In the default 'auto' connect mode, ``PlotDataItem`` will apply this setting automatically.\n        "
        self.opts['skipFiniteCheck'] = bool(skipFiniteCheck)

    def setData(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        '\n        Clear any data displayed by this item and display new data.\n        See :func:`__init__() <pyqtgraph.PlotDataItem.__init__>` for details; it accepts the same arguments.\n        '
        if kargs.get('stepMode', None) is True:
            warnings.warn('stepMode=True is deprecated and will result in an error after October 2022. Use stepMode="center" instead.', DeprecationWarning, stacklevel=3)
        if 'decimate' in kargs.keys():
            warnings.warn('The decimate keyword has been deprecated. It has no effect and may result in an error in releases after October 2022. ', DeprecationWarning, stacklevel=2)
        if 'identical' in kargs.keys():
            warnings.warn('The identical keyword has been deprecated. It has no effect may result in an error in releases after October 2022. ', DeprecationWarning, stacklevel=2)
        profiler = debug.Profiler()
        y = None
        x = None
        if len(args) == 1:
            data = args[0]
            dt = dataType(data)
            if dt == 'empty':
                pass
            elif dt == 'listOfValues':
                y = np.array(data)
            elif dt == 'Nx2array':
                x = data[:, 0]
                y = data[:, 1]
            elif dt == 'recarray' or dt == 'dictOfLists':
                if 'x' in data:
                    x = np.array(data['x'])
                if 'y' in data:
                    y = np.array(data['y'])
            elif dt == 'listOfDicts':
                if 'x' in data[0]:
                    x = np.array([d.get('x', None) for d in data])
                if 'y' in data[0]:
                    y = np.array([d.get('y', None) for d in data])
                for k in ['data', 'symbolSize', 'symbolPen', 'symbolBrush', 'symbolShape']:
                    if k in data:
                        kargs[k] = [d.get(k, None) for d in data]
            elif dt == 'MetaArray':
                y = data.view(np.ndarray)
                x = data.xvals(0).view(np.ndarray)
            else:
                raise TypeError('Invalid data type %s' % type(data))
        elif len(args) == 2:
            seq = ('listOfValues', 'MetaArray', 'empty')
            dtyp = (dataType(args[0]), dataType(args[1]))
            if dtyp[0] not in seq or dtyp[1] not in seq:
                raise TypeError('When passing two unnamed arguments, both must be a list or array of values. (got %s, %s)' % (str(type(args[0])), str(type(args[1]))))
            if not isinstance(args[0], np.ndarray):
                if dtyp[0] == 'MetaArray':
                    x = args[0].asarray()
                else:
                    x = np.array(args[0])
            else:
                x = args[0].view(np.ndarray)
            if not isinstance(args[1], np.ndarray):
                if dtyp[1] == 'MetaArray':
                    y = args[1].asarray()
                else:
                    y = np.array(args[1])
            else:
                y = args[1].view(np.ndarray)
        if 'x' in kargs:
            x = kargs['x']
            if dataType(x) == 'MetaArray':
                x = x.asarray()
        if 'y' in kargs:
            y = kargs['y']
            if dataType(y) == 'MetaArray':
                y = y.asarray()
        profiler('interpret data')
        if 'name' in kargs:
            self.opts['name'] = kargs['name']
            self.setProperty('styleWasChanged', True)
        if 'connect' in kargs:
            self.opts['connect'] = kargs['connect']
            self.setProperty('styleWasChanged', True)
        if 'skipFiniteCheck' in kargs:
            self.opts['skipFiniteCheck'] = kargs['skipFiniteCheck']
        if 'symbol' not in kargs and ('symbolPen' in kargs or 'symbolBrush' in kargs or 'symbolSize' in kargs):
            if self.opts['symbol'] is None:
                kargs['symbol'] = 'o'
        if 'brush' in kargs:
            kargs['fillBrush'] = kargs['brush']
        for k in list(self.opts.keys()):
            if k in kargs:
                self.opts[k] = kargs[k]
                self.setProperty('styleWasChanged', True)
        if y is None or len(y) == 0:
            yData = None
        else:
            if not isinstance(y, np.ndarray):
                y = np.array(y)
            yData = y.view(np.ndarray)
            if x is None:
                x = np.arange(len(y))
        if x is None or len(x) == 0:
            xData = None
        else:
            if not isinstance(x, np.ndarray):
                x = np.array(x)
            xData = x.view(np.ndarray)
        if xData is None or yData is None:
            self._dataset = None
        else:
            self._dataset = PlotDataset(xData, yData)
        self._datasetMapped = None
        self._datasetDisplay = None
        self._adsLastValue = 1
        profiler('set data')
        self.updateItems(styleUpdate=self.property('styleWasChanged'))
        self.setProperty('styleWasChanged', False)
        profiler('update items')
        self.informViewBoundsChanged()
        self.sigPlotChanged.emit(self)
        profiler('emit')

    def updateItems(self, styleUpdate=True):
        if False:
            for i in range(10):
                print('nop')
        styleUpdate = True
        curveArgs = {}
        scatterArgs = {}
        if styleUpdate:
            for (k, v) in [('pen', 'pen'), ('shadowPen', 'shadowPen'), ('fillLevel', 'fillLevel'), ('fillOutline', 'fillOutline'), ('fillBrush', 'brush'), ('antialias', 'antialias'), ('connect', 'connect'), ('stepMode', 'stepMode'), ('skipFiniteCheck', 'skipFiniteCheck')]:
                if k in self.opts:
                    curveArgs[v] = self.opts[k]
            for (k, v) in [('symbolPen', 'pen'), ('symbolBrush', 'brush'), ('symbol', 'symbol'), ('symbolSize', 'size'), ('data', 'data'), ('pxMode', 'pxMode'), ('antialias', 'antialias'), ('useCache', 'useCache')]:
                if k in self.opts:
                    scatterArgs[v] = self.opts[k]
        dataset = self._getDisplayDataset()
        if dataset is None:
            self.curve.hide()
            self.scatter.hide()
            return
        x = dataset.x
        y = dataset.y
        if self.opts['pen'] is not None or (self.opts['fillBrush'] is not None and self.opts['fillLevel'] is not None):
            if isinstance(curveArgs['connect'], str) and curveArgs['connect'] == 'auto':
                if dataset.containsNonfinite is False:
                    curveArgs['connect'] = 'all'
                    curveArgs['skipFiniteCheck'] = True
                else:
                    curveArgs['connect'] = 'finite'
                    curveArgs['skipFiniteCheck'] = False
            self.curve.setData(x=x, y=y, **curveArgs)
            self.curve.show()
        else:
            self.curve.hide()
        if self.opts['symbol'] is not None:
            if self.opts.get('stepMode', False) in ('center', True):
                x = 0.5 * (x[:-1] + x[1:])
            self.scatter.setData(x=x, y=y, **scatterArgs)
            self.scatter.show()
        else:
            self.scatter.hide()

    def getOriginalDataset(self):
        if False:
            print('Hello World!')
        '\n            Returns the original, unmapped data as the tuple (`xData`, `yData`).\n            '
        dataset = self._dataset
        if dataset is None:
            return (None, None)
        return (dataset.x, dataset.y)

    def _getDisplayDataset(self):
        if False:
            while True:
                i = 10
        '\n        Returns a :class:`~.PlotDataset` object that contains data suitable for display \n        (after mapping and data reduction) as ``dataset.x`` and ``dataset.y``.\n        Intended for internal use.\n        '
        if self._dataset is None:
            return None
        if self._datasetDisplay is not None and (not (self.property('xViewRangeWasChanged') and self.opts['clipToView'])) and (not (self.property('xViewRangeWasChanged') and self.opts['autoDownsample'])) and (not (self.property('yViewRangeWasChanged') and self.opts['dynamicRangeLimit'] is not None)):
            return self._datasetDisplay
        if self._datasetMapped is None:
            x = self._dataset.x
            y = self._dataset.y
            if y.dtype == bool:
                y = y.astype(np.uint8)
            if x.dtype == bool:
                x = x.astype(np.uint8)
            if self.opts['fftMode']:
                (x, y) = self._fourierTransform(x, y)
                if self.opts['logMode'][0]:
                    x = x[1:]
                    y = y[1:]
            if self.opts['derivativeMode']:
                y = np.diff(self._dataset.y) / np.diff(self._dataset.x)
                x = x[:-1]
            if self.opts['phasemapMode']:
                x = self._dataset.y[:-1]
                y = np.diff(self._dataset.y) / np.diff(self._dataset.x)
            dataset = PlotDataset(x, y, self._dataset.xAllFinite, self._dataset.yAllFinite)
            if True in self.opts['logMode']:
                dataset.applyLogMapping(self.opts['logMode'])
            self._datasetMapped = dataset
        x = self._datasetMapped.x
        y = self._datasetMapped.y
        xAllFinite = self._datasetMapped.xAllFinite
        yAllFinite = self._datasetMapped.yAllFinite
        view = self.getViewBox()
        if view is None:
            view_range = None
        else:
            view_range = view.viewRect()
        if view_range is None:
            view_range = self.viewRect()
        ds = self.opts['downsample']
        if not isinstance(ds, int):
            ds = 1
        if self.opts['autoDownsample']:
            if xAllFinite:
                finite_x = x
            else:
                finite_x = x[np.isfinite(x)]
            if view_range is not None and len(finite_x) > 1:
                dx = float(finite_x[-1] - finite_x[0]) / (len(finite_x) - 1)
                if dx != 0.0:
                    width = self.getViewBox().width()
                    if width != 0.0:
                        ds_float = max(1.0, abs(view_range.width() / dx / (width * self.opts['autoDownsampleFactor'])))
                        if math.isfinite(ds_float):
                            ds = int(ds_float)
            if math.isclose(ds, self._adsLastValue, rel_tol=0.01):
                ds = self._adsLastValue
            self._adsLastValue = ds
        if self.opts['clipToView']:
            if view is None or view.autoRangeEnabled()[0]:
                pass
            elif view_range is not None and len(x) > 1:
                x0 = bisect.bisect_left(x, view_range.left()) - ds
                x0 = fn.clip_scalar(x0, 0, len(x))
                x1 = bisect.bisect_left(x, view_range.right()) + ds
                x1 = fn.clip_scalar(x1, x0, len(x))
                x = x[x0:x1]
                y = y[x0:x1]
        if ds > 1:
            if self.opts['downsampleMethod'] == 'subsample':
                x = x[::ds]
                y = y[::ds]
            elif self.opts['downsampleMethod'] == 'mean':
                n = len(x) // ds
                stx = ds // 2
                x = x[stx:stx + n * ds:ds]
                y = y[:n * ds].reshape(n, ds).mean(axis=1)
            elif self.opts['downsampleMethod'] == 'peak':
                n = len(x) // ds
                x1 = np.empty((n, 2))
                stx = ds // 2
                x1[:] = x[stx:stx + n * ds:ds, np.newaxis]
                x = x1.reshape(n * 2)
                y1 = np.empty((n, 2))
                y2 = y[:n * ds].reshape((n, ds))
                y1[:, 0] = y2.max(axis=1)
                y1[:, 1] = y2.min(axis=1)
                y = y1.reshape(n * 2)
        if self.opts['dynamicRangeLimit'] is not None:
            if view_range is not None:
                data_range = self._datasetMapped.dataRect()
                if data_range is not None:
                    view_height = view_range.height()
                    limit = self.opts['dynamicRangeLimit']
                    hyst = self.opts['dynamicRangeHyst']
                    if view_height > 0 and (not data_range.bottom() < view_range.top()) and (not data_range.top() > view_range.bottom()) and (data_range.height() > 2 * hyst * limit * view_height):
                        cache_is_good = False
                        if self._datasetDisplay is not None:
                            top_exc = -(self._drlLastClip[0] - view_range.bottom()) / view_height
                            bot_exc = (self._drlLastClip[1] - view_range.top()) / view_height
                            if top_exc >= limit / hyst and top_exc <= limit * hyst and (bot_exc >= limit / hyst) and (bot_exc <= limit * hyst):
                                x = self._datasetDisplay.x
                                y = self._datasetDisplay.y
                                cache_is_good = True
                        if not cache_is_good:
                            min_val = view_range.bottom() - limit * view_height
                            max_val = view_range.top() + limit * view_height
                            y = fn.clip_array(y, min_val, max_val)
                            self._drlLastClip = (min_val, max_val)
        self._datasetDisplay = PlotDataset(x, y, xAllFinite, yAllFinite)
        self.setProperty('xViewRangeWasChanged', False)
        self.setProperty('yViewRangeWasChanged', False)
        return self._datasetDisplay

    def getData(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns the displayed data as the tuple (`xData`, `yData`) after mapping and data reduction.\n        '
        dataset = self._getDisplayDataset()
        if dataset is None:
            return (None, None)
        return (dataset.x, dataset.y)

    def dataRect(self):
        if False:
            return 10
        '\n        Returns a bounding rectangle (as :class:`QtCore.QRectF`) for the full set of data.\n        Will return `None` if there is no data or if all values (x or y) are NaN.\n        '
        if self._dataset is None:
            return None
        return self._dataset.dataRect()

    def dataBounds(self, ax, frac=1.0, orthoRange=None):
        if False:
            for i in range(10):
                print('nop')
        "\n        Returns the range occupied by the data (along a specific axis) in this item.\n        This method is called by :class:`ViewBox` when auto-scaling.\n\n        =============== ====================================================================\n        **Arguments:**\n        ax              (0 or 1) the axis for which to return this item's data range\n        frac            (float 0.0-1.0) Specifies what fraction of the total data\n                        range to return. By default, the entire range is returned.\n                        This allows the :class:`ViewBox` to ignore large spikes in the data\n                        when auto-scaling.\n        orthoRange      ([min,max] or None) Specifies that only the data within the\n                        given range (orthogonal to *ax*) should me measured when\n                        returning the data range. (For example, a ViewBox might ask\n                        what is the y-range of all data with x-values between min\n                        and max)\n        =============== ====================================================================\n        "
        range = [None, None]
        if self.curve.isVisible():
            range = self.curve.dataBounds(ax, frac, orthoRange)
        elif self.scatter.isVisible():
            r2 = self.scatter.dataBounds(ax, frac, orthoRange)
            range = [r2[0] if range[0] is None else range[0] if r2[0] is None else min(r2[0], range[0]), r2[1] if range[1] is None else range[1] if r2[1] is None else min(r2[1], range[1])]
        return range

    def pixelPadding(self):
        if False:
            return 10
        '\n        Returns the size in pixels that this item may draw beyond the values returned by dataBounds().\n        This method is called by :class:`ViewBox` when auto-scaling.\n        '
        pad = 0
        if self.curve.isVisible():
            pad = max(pad, self.curve.pixelPadding())
        elif self.scatter.isVisible():
            pad = max(pad, self.scatter.pixelPadding())
        return pad

    def clear(self):
        if False:
            while True:
                i = 10
        self._dataset = None
        self._datasetMapped = None
        self._datasetDisplay = None
        self.curve.clear()
        self.scatter.clear()

    def appendData(self, *args, **kargs):
        if False:
            i = 10
            return i + 15
        pass

    def curveClicked(self, curve, ev):
        if False:
            i = 10
            return i + 15
        self.sigClicked.emit(self, ev)

    def scatterClicked(self, plt, points, ev):
        if False:
            while True:
                i = 10
        self.sigClicked.emit(self, ev)
        self.sigPointsClicked.emit(self, points, ev)

    def scatterHovered(self, plt, points, ev):
        if False:
            return 10
        self.sigPointsHovered.emit(self, points, ev)

    def viewRangeChanged(self, vb=None, ranges=None, changed=None):
        if False:
            i = 10
            return i + 15
        update_needed = False
        if changed is None or changed[0]:
            self.setProperty('xViewRangeWasChanged', True)
            if self.opts['clipToView'] or self.opts['autoDownsample']:
                self._datasetDisplay = None
                update_needed = True
        if changed is None or changed[1]:
            self.setProperty('yViewRangeWasChanged', True)
            if self.opts['dynamicRangeLimit'] is not None:
                update_needed = True
        if update_needed:
            self.updateItems(styleUpdate=False)

    def _fourierTransform(self, x, y):
        if False:
            while True:
                i = 10
        dx = np.diff(x)
        uniform = not np.any(np.abs(dx - dx[0]) > abs(dx[0]) / 1000.0)
        if not uniform:
            x2 = np.linspace(x[0], x[-1], len(x))
            y = np.interp(x2, x, y)
            x = x2
        n = y.size
        f = np.fft.rfft(y) / n
        d = float(x[-1] - x[0]) / (len(x) - 1)
        x = np.fft.rfftfreq(n, d)
        y = np.abs(f)
        return (x, y)

def dataType(obj):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(obj, '__len__') and len(obj) == 0:
        return 'empty'
    if isinstance(obj, dict):
        return 'dictOfLists'
    elif isSequence(obj):
        first = obj[0]
        if hasattr(obj, 'implements') and obj.implements('MetaArray'):
            return 'MetaArray'
        elif isinstance(obj, np.ndarray):
            if obj.ndim == 1:
                if obj.dtype.names is None:
                    return 'listOfValues'
                else:
                    return 'recarray'
            elif obj.ndim == 2 and obj.dtype.names is None and (obj.shape[1] == 2):
                return 'Nx2array'
            else:
                raise ValueError('array shape must be (N,) or (N,2); got %s instead' % str(obj.shape))
        elif isinstance(first, dict):
            return 'listOfDicts'
        else:
            return 'listOfValues'

def isSequence(obj):
    if False:
        i = 10
        return i + 15
    return hasattr(obj, '__iter__') or isinstance(obj, np.ndarray) or (hasattr(obj, 'implements') and obj.implements('MetaArray'))