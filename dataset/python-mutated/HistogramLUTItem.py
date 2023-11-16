"""
GraphicsWidget displaying an image histogram along with gradient editor. Can be used to
adjust the appearance of images.
"""
import weakref
import numpy as np
from .. import debug as debug
from .. import functions as fn
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .AxisItem import AxisItem
from .GradientEditorItem import GradientEditorItem
from .GraphicsWidget import GraphicsWidget
from .LinearRegionItem import LinearRegionItem
from .PlotCurveItem import PlotCurveItem
from .ViewBox import ViewBox
__all__ = ['HistogramLUTItem']

class HistogramLUTItem(GraphicsWidget):
    """
    :class:`~pyqtgraph.GraphicsWidget` with controls for adjusting the display of an
    :class:`~pyqtgraph.ImageItem`.

    Includes:

      - Image histogram
      - Movable region over the histogram to select black/white levels
      - Gradient editor to define color lookup table for single-channel images

    Parameters
    ----------
    image : pyqtgraph.ImageItem, optional
        If provided, control will be automatically linked to the image and changes to
        the control will be reflected in the image's appearance. This may also be set
        via :meth:`setImageItem`.
    fillHistogram : bool, optional
        By default, the histogram is rendered with a fill. Performance may be improved
        by disabling the fill. Additional control over the fill is provided by
        :meth:`fillHistogram`.
    levelMode : str, optional
        'mono' (default)
            One histogram with a :class:`~pyqtgraph.LinearRegionItem` is displayed to
            control the black/white levels of the image. This option may be used for
            color images, in which case the histogram and levels correspond to all
            channels of the image.
        'rgba'
            A histogram and level control pair is provided for each image channel. The
            alpha channel histogram and level control are only shown if the image
            contains an alpha channel.
    gradientPosition : str, optional
        Position of the gradient editor relative to the histogram. Must be one of
        {'right', 'left', 'top', 'bottom'}. 'right' and 'left' options should be used
        with a 'vertical' orientation; 'top' and 'bottom' options are for 'horizontal'
        orientation.
    orientation : str, optional
        The orientation of the axis along which the histogram is displayed. Either
        'vertical' (default) or 'horizontal'.

    Attributes
    ----------
    sigLookupTableChanged : QtCore.Signal
        Emits the HistogramLUTItem itself when the gradient changes
    sigLevelsChanged : QtCore.Signal
        Emits the HistogramLUTItem itself while the movable region is changing
    sigLevelChangeFinished : QtCore.Signal
        Emits the HistogramLUTItem itself when the movable region is finished changing

    See Also
    --------
    :class:`~pyqtgraph.ImageItem`
        HistogramLUTItem is most useful when paired with an ImageItem.
    :class:`~pyqtgraph.ImageView`
        Widget containing a paired ImageItem and HistogramLUTItem.
    :class:`~pyqtgraph.HistogramLUTWidget`
        QWidget containing a HistogramLUTItem for widget-based layouts.
    """
    sigLookupTableChanged = QtCore.Signal(object)
    sigLevelsChanged = QtCore.Signal(object)
    sigLevelChangeFinished = QtCore.Signal(object)

    def __init__(self, image=None, fillHistogram=True, levelMode='mono', gradientPosition='right', orientation='vertical'):
        if False:
            print('Hello World!')
        GraphicsWidget.__init__(self)
        self.lut = None
        self.imageItem = lambda : None
        self.levelMode = levelMode
        self.orientation = orientation
        self.gradientPosition = gradientPosition
        if orientation == 'vertical' and gradientPosition not in {'right', 'left'}:
            self.gradientPosition = 'right'
        elif orientation == 'horizontal' and gradientPosition not in {'top', 'bottom'}:
            self.gradientPosition = 'bottom'
        self.layout = QtWidgets.QGraphicsGridLayout()
        self.setLayout(self.layout)
        self.layout.setContentsMargins(1, 1, 1, 1)
        self.layout.setSpacing(0)
        self.vb = ViewBox(parent=self)
        if self.orientation == 'vertical':
            self.vb.setMaximumWidth(152)
            self.vb.setMinimumWidth(45)
            self.vb.setMouseEnabled(x=False, y=True)
        else:
            self.vb.setMaximumHeight(152)
            self.vb.setMinimumHeight(45)
            self.vb.setMouseEnabled(x=True, y=False)
        self.gradient = GradientEditorItem(orientation=self.gradientPosition)
        self.gradient.loadPreset('grey')
        regionOrientation = 'horizontal' if self.orientation == 'vertical' else 'vertical'
        self.regions = [LinearRegionItem([0, 1], regionOrientation, swapMode='block'), LinearRegionItem([0, 1], regionOrientation, swapMode='block', pen='r', brush=fn.mkBrush((255, 50, 50, 50)), span=(0.0, 1 / 3.0)), LinearRegionItem([0, 1], regionOrientation, swapMode='block', pen='g', brush=fn.mkBrush((50, 255, 50, 50)), span=(1 / 3.0, 2 / 3.0)), LinearRegionItem([0, 1], regionOrientation, swapMode='block', pen='b', brush=fn.mkBrush((50, 50, 255, 80)), span=(2 / 3.0, 1.0)), LinearRegionItem([0, 1], regionOrientation, swapMode='block', pen='w', brush=fn.mkBrush((255, 255, 255, 50)), span=(2 / 3.0, 1.0))]
        self.region = self.regions[0]
        for region in self.regions:
            region.setZValue(1000)
            self.vb.addItem(region)
            region.lines[0].addMarker('<|', 0.5)
            region.lines[1].addMarker('|>', 0.5)
            region.sigRegionChanged.connect(self.regionChanging)
            region.sigRegionChangeFinished.connect(self.regionChanged)
        ax = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}[self.gradientPosition]
        self.axis = AxisItem(ax, linkView=self.vb, maxTickLength=-10, parent=self)
        avg = (0, 1, 2) if self.gradientPosition in {'right', 'bottom'} else (2, 1, 0)
        if self.orientation == 'vertical':
            self.layout.addItem(self.axis, 0, avg[0])
            self.layout.addItem(self.vb, 0, avg[1])
            self.layout.addItem(self.gradient, 0, avg[2])
        else:
            self.layout.addItem(self.axis, avg[0], 0)
            self.layout.addItem(self.vb, avg[1], 0)
            self.layout.addItem(self.gradient, avg[2], 0)
        self.gradient.setFlag(self.gradient.GraphicsItemFlag.ItemStacksBehindParent)
        self.vb.setFlag(self.gradient.GraphicsItemFlag.ItemStacksBehindParent)
        self.gradient.sigGradientChanged.connect(self.gradientChanged)
        self.vb.sigRangeChanged.connect(self.viewRangeChanged)
        comp = QtGui.QPainter.CompositionMode.CompositionMode_Plus
        self.plots = [PlotCurveItem(pen=(200, 200, 200, 100)), PlotCurveItem(pen=(255, 0, 0, 100), compositionMode=comp), PlotCurveItem(pen=(0, 255, 0, 100), compositionMode=comp), PlotCurveItem(pen=(0, 0, 255, 100), compositionMode=comp), PlotCurveItem(pen=(200, 200, 200, 100), compositionMode=comp)]
        self.plot = self.plots[0]
        for plot in self.plots:
            if self.orientation == 'vertical':
                plot.setRotation(90)
            self.vb.addItem(plot)
        self.fillHistogram(fillHistogram)
        self._showRegions()
        self.autoHistogramRange()
        if image is not None:
            self.setImageItem(image)

    def fillHistogram(self, fill=True, level=0.0, color=(100, 100, 200)):
        if False:
            for i in range(10):
                print('nop')
        'Control fill of the histogram curve(s).\n\n        Parameters\n        ----------\n        fill : bool, optional\n            Set whether or not the histogram should be filled.\n        level : float, optional\n            Set the fill level. See :meth:`PlotCurveItem.setFillLevel\n            <pyqtgraph.PlotCurveItem.setFillLevel>`. Only used if ``fill`` is True.\n        color : color_like, optional\n            Color to use for the fill when the histogram ``levelMode == "mono"``. See\n            :meth:`PlotCurveItem.setBrush <pyqtgraph.PlotCurveItem.setBrush>`.\n        '
        colors = [color, (255, 0, 0, 50), (0, 255, 0, 50), (0, 0, 255, 50), (255, 255, 255, 50)]
        for (color, plot) in zip(colors, self.plots):
            if fill:
                plot.setFillLevel(level)
                plot.setBrush(color)
            else:
                plot.setFillLevel(None)

    def paint(self, p, *args):
        if False:
            while True:
                i = 10
        if self.levelMode != 'mono' or not self.region.isVisible():
            return
        pen = self.region.lines[0].pen
        (mn, mx) = self.getLevels()
        vbc = self.vb.viewRect().center()
        gradRect = self.gradient.mapRectToParent(self.gradient.gradRect.rect())
        if self.orientation == 'vertical':
            p1mn = self.vb.mapFromViewToItem(self, Point(vbc.x(), mn)) + Point(0, 5)
            p1mx = self.vb.mapFromViewToItem(self, Point(vbc.x(), mx)) - Point(0, 5)
            if self.gradientPosition == 'right':
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.topLeft()
            else:
                p2mn = gradRect.bottomRight()
                p2mx = gradRect.topRight()
        else:
            p1mn = self.vb.mapFromViewToItem(self, Point(mn, vbc.y())) - Point(5, 0)
            p1mx = self.vb.mapFromViewToItem(self, Point(mx, vbc.y())) + Point(5, 0)
            if self.gradientPosition == 'bottom':
                p2mn = gradRect.topLeft()
                p2mx = gradRect.topRight()
            else:
                p2mn = gradRect.bottomLeft()
                p2mx = gradRect.bottomRight()
        p.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
        for pen in [fn.mkPen((0, 0, 0, 100), width=3), pen]:
            p.setPen(pen)
            p.drawLine(p1mn, p2mn)
            p.drawLine(p1mx, p2mx)
            if self.orientation == 'vertical':
                p.drawLine(gradRect.topLeft(), gradRect.topRight())
                p.drawLine(gradRect.bottomLeft(), gradRect.bottomRight())
            else:
                p.drawLine(gradRect.topLeft(), gradRect.bottomLeft())
                p.drawLine(gradRect.topRight(), gradRect.bottomRight())

    def setHistogramRange(self, mn, mx, padding=0.1):
        if False:
            print('Hello World!')
        'Set the X/Y range on the histogram plot, depending on the orientation. This disables auto-scaling.'
        if self.orientation == 'vertical':
            self.vb.enableAutoRange(self.vb.YAxis, False)
            self.vb.setYRange(mn, mx, padding)
        else:
            self.vb.enableAutoRange(self.vb.XAxis, False)
            self.vb.setXRange(mn, mx, padding)

    def getHistogramRange(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns range on the histogram plot.'
        if self.orientation == 'vertical':
            return self.vb.viewRange()[1]
        else:
            return self.vb.viewRange()[0]

    def autoHistogramRange(self):
        if False:
            i = 10
            return i + 15
        'Enable auto-scaling on the histogram plot.'
        self.vb.enableAutoRange(self.vb.XYAxes)

    def disableAutoHistogramRange(self):
        if False:
            while True:
                i = 10
        'Disable auto-scaling on the histogram plot.'
        self.vb.disableAutoRange(self.vb.XYAxes)

    def setImageItem(self, img):
        if False:
            print('Hello World!')
        'Set an ImageItem to have its levels and LUT automatically controlled by this\n        HistogramLUTItem.\n        '
        self.imageItem = weakref.ref(img)
        if hasattr(img, 'sigImageChanged'):
            img.sigImageChanged.connect(self.imageChanged)
        self._setImageLookupTable()
        self.regionChanged()
        self.imageChanged(autoLevel=True)

    def viewRangeChanged(self):
        if False:
            print('Hello World!')
        self.update()

    def gradientChanged(self):
        if False:
            return 10
        if self.imageItem() is not None:
            self._setImageLookupTable()
        self.lut = None
        self.sigLookupTableChanged.emit(self)

    def _setImageLookupTable(self):
        if False:
            print('Hello World!')
        if self.gradient.isLookupTrivial():
            self.imageItem().setLookupTable(None)
        else:
            self.imageItem().setLookupTable(self.getLookupTable)

    def getLookupTable(self, img=None, n=None, alpha=None):
        if False:
            print('Hello World!')
        'Return a lookup table from the color gradient defined by this\n        HistogramLUTItem.\n        '
        if self.levelMode != 'mono':
            return None
        if n is None:
            if img.dtype == np.uint8:
                n = 256
            else:
                n = 512
        if self.lut is None:
            self.lut = self.gradient.getLookupTable(n, alpha=alpha)
        return self.lut

    def regionChanged(self):
        if False:
            i = 10
            return i + 15
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())
        self.sigLevelChangeFinished.emit(self)

    def regionChanging(self):
        if False:
            for i in range(10):
                print('nop')
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())
        self.update()
        self.sigLevelsChanged.emit(self)

    def imageChanged(self, autoLevel=False, autoRange=False):
        if False:
            for i in range(10):
                print('nop')
        if self.imageItem() is None:
            return
        if self.levelMode == 'mono':
            for plt in self.plots[1:]:
                plt.setVisible(False)
            self.plots[0].setVisible(True)
            profiler = debug.Profiler()
            h = self.imageItem().getHistogram()
            profiler('get histogram')
            if h[0] is None:
                return
            self.plot.setData(*h)
            profiler('set plot')
            if autoLevel:
                mn = h[0][0]
                mx = h[0][-1]
                self.region.setRegion([mn, mx])
                profiler('set region')
            else:
                (mn, mx) = self.imageItem().getLevels()
                self.region.setRegion([mn, mx])
        else:
            self.plots[0].setVisible(False)
            ch = self.imageItem().getHistogram(perChannel=True)
            if ch[0] is None:
                return
            for i in range(1, 5):
                if len(ch) >= i:
                    h = ch[i - 1]
                    self.plots[i].setVisible(True)
                    self.plots[i].setData(*h)
                    if autoLevel:
                        mn = h[0][0]
                        mx = h[0][-1]
                        self.regions[i].setRegion([mn, mx])
                else:
                    self.plots[i].setVisible(False)
            self._showRegions()

    def getLevels(self):
        if False:
            return 10
        'Return the min and max levels.\n\n        For rgba mode, this returns a list of the levels for each channel.\n        '
        if self.levelMode == 'mono':
            return self.region.getRegion()
        else:
            nch = self.imageItem().channels()
            if nch is None:
                nch = 3
            return [r.getRegion() for r in self.regions[1:nch + 1]]

    def setLevels(self, min=None, max=None, rgba=None):
        if False:
            i = 10
            return i + 15
        "Set the min/max (bright and dark) levels.\n\n        Parameters\n        ----------\n        min : float, optional\n            Minimum level.\n        max : float, optional\n            Maximum level.\n        rgba : list, optional\n            Sequence of (min, max) pairs for each channel for 'rgba' mode.\n        "
        if None in {min, max} and (rgba is None or None in rgba[0]):
            raise ValueError('Must specify min and max levels')
        if self.levelMode == 'mono':
            if min is None:
                (min, max) = rgba[0]
            self.region.setRegion((min, max))
        else:
            if rgba is None:
                rgba = 4 * [(min, max)]
            for (levels, region) in zip(rgba, self.regions[1:]):
                region.setRegion(levels)

    def setLevelMode(self, mode):
        if False:
            return 10
        "Set the method of controlling the image levels offered to the user.\n\n        Options are 'mono' or 'rgba'.\n        "
        if mode not in {'mono', 'rgba'}:
            raise ValueError(f"Level mode must be one of {{'mono', 'rgba'}}, got {mode}")
        if mode == self.levelMode:
            return
        oldLevels = self.getLevels()
        self.levelMode = mode
        self._showRegions()
        if mode == 'mono':
            levels = np.array(oldLevels).mean(axis=0)
            self.setLevels(*levels)
        else:
            levels = [oldLevels] * 4
            self.setLevels(rgba=levels)
        if self.imageItem() is not None:
            self.imageItem().setLevels(self.getLevels())
        self.imageChanged()
        self.update()

    def _showRegions(self):
        if False:
            for i in range(10):
                print('nop')
        for i in range(len(self.regions)):
            self.regions[i].setVisible(False)
        if self.levelMode == 'rgba':
            nch = 4
            if self.imageItem() is not None:
                nch = self.imageItem().channels()
                if nch is None:
                    nch = 3
            xdif = 1.0 / nch
            for i in range(1, nch + 1):
                self.regions[i].setVisible(True)
                self.regions[i].setSpan((i - 1) * xdif, i * xdif)
            self.gradient.hide()
        elif self.levelMode == 'mono':
            self.regions[0].setVisible(True)
            self.gradient.show()
        else:
            raise ValueError(f'Unknown level mode {self.levelMode}')

    def saveState(self):
        if False:
            for i in range(10):
                print('nop')
        return {'gradient': self.gradient.saveState(), 'levels': self.getLevels(), 'mode': self.levelMode}

    def restoreState(self, state):
        if False:
            return 10
        if 'mode' in state:
            self.setLevelMode(state['mode'])
        self.gradient.restoreState(state['gradient'])
        self.setLevels(*state['levels'])