import math
from .. import functions as fn
from ..icons import invisibleEye
from ..Point import Point
from ..Qt import QtCore, QtGui, QtWidgets
from .BarGraphItem import BarGraphItem
from .GraphicsWidget import GraphicsWidget
from .GraphicsWidgetAnchor import GraphicsWidgetAnchor
from .LabelItem import LabelItem
from .PlotDataItem import PlotDataItem
from .ScatterPlotItem import ScatterPlotItem, drawSymbol
__all__ = ['LegendItem', 'ItemSample']

class LegendItem(GraphicsWidgetAnchor, GraphicsWidget):
    """
    Displays a legend used for describing the contents of a plot.

    LegendItems are most commonly created by calling :meth:`PlotItem.addLegend
    <pyqtgraph.PlotItem.addLegend>`.

    Note that this item should *not* be added directly to a PlotItem (via
    :meth:`PlotItem.addItem <pyqtgraph.PlotItem.addItem>`). Instead, make it a
    direct descendant of the PlotItem::

        legend.setParentItem(plotItem)

    """

    def __init__(self, size=None, offset=None, horSpacing=25, verSpacing=0, pen=None, brush=None, labelTextColor=None, frame=True, labelTextSize='9pt', colCount=1, sampleType=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        ==============  ===============================================================\n        **Arguments:**\n        size            Specifies the fixed size (width, height) of the legend. If\n                        this argument is omitted, the legend will automatically resize\n                        to fit its contents.\n        offset          Specifies the offset position relative to the legend's parent.\n                        Positive values offset from the left or top; negative values\n                        offset from the right or bottom. If offset is None, the\n                        legend must be anchored manually by calling anchor() or\n                        positioned by calling setPos().\n        horSpacing      Specifies the spacing between the line symbol and the label.\n        verSpacing      Specifies the spacing between individual entries of the legend\n                        vertically. (Can also be negative to have them really close)\n        pen             Pen to use when drawing legend border. Any single argument\n                        accepted by :func:`mkPen <pyqtgraph.mkPen>` is allowed.\n        brush           QBrush to use as legend background filling. Any single argument\n                        accepted by :func:`mkBrush <pyqtgraph.mkBrush>` is allowed.\n        labelTextColor  Pen to use when drawing legend text. Any single argument\n                        accepted by :func:`mkPen <pyqtgraph.mkPen>` is allowed.\n        labelTextSize   Size to use when drawing legend text. Accepts CSS style\n                        string arguments, e.g. '9pt'.\n        colCount        Specifies the integer number of columns that the legend should\n                        be divided into. The number of rows will be calculated\n                        based on this argument. This is useful for plots with many\n                        curves displayed simultaneously. Default: 1 column.\n        sampleType      Customizes the item sample class of the `LegendItem`.\n        ==============  ===============================================================\n\n        "
        GraphicsWidget.__init__(self)
        GraphicsWidgetAnchor.__init__(self)
        self.setFlag(self.GraphicsItemFlag.ItemIgnoresTransformations)
        self.layout = QtWidgets.QGraphicsGridLayout()
        self.layout.setVerticalSpacing(verSpacing)
        self.layout.setHorizontalSpacing(horSpacing)
        self.setLayout(self.layout)
        self.items = []
        self.size = size
        self.offset = offset
        self.frame = frame
        self.columnCount = colCount
        self.rowCount = 1
        if size is not None:
            self.setGeometry(QtCore.QRectF(0, 0, self.size[0], self.size[1]))
        if sampleType is not None:
            if not issubclass(sampleType, GraphicsWidget):
                raise RuntimeError('Only classes of type `GraphicsWidgets` are allowed as `sampleType`')
            self.sampleType = sampleType
        else:
            self.sampleType = ItemSample
        self.opts = {'pen': fn.mkPen(pen), 'brush': fn.mkBrush(brush), 'labelTextColor': labelTextColor, 'labelTextSize': labelTextSize, 'offset': offset}
        self.opts.update(kwargs)

    def setSampleType(self, sample):
        if False:
            return 10
        'Set the new sample item claspes'
        if sample is self.sampleType:
            return
        items = list(self.items)
        self.sampleType = sample
        self.clear()
        for (sample, label) in items:
            plot_item = sample.item
            plot_name = label.text
            self.addItem(plot_item, plot_name)
        self.updateSize()

    def offset(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the offset position relative to the parent.'
        return self.opts['offset']

    def setOffset(self, offset):
        if False:
            for i in range(10):
                print('nop')
        'Set the offset position relative to the parent.'
        self.opts['offset'] = offset
        offset = Point(self.opts['offset'])
        anchorx = 1 if offset[0] <= 0 else 0
        anchory = 1 if offset[1] <= 0 else 0
        anchor = (anchorx, anchory)
        self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)

    def pen(self):
        if False:
            for i in range(10):
                print('nop')
        'Get the QPen used to draw the border around the legend.'
        return self.opts['pen']

    def setPen(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        'Set the pen used to draw a border around the legend.\n\n        Accepts the same arguments as :func:`~pyqtgraph.mkPen`.\n        '
        pen = fn.mkPen(*args, **kargs)
        self.opts['pen'] = pen
        self.update()

    def brush(self):
        if False:
            print('Hello World!')
        'Get the QBrush used to draw the legend background.'
        return self.opts['brush']

    def setBrush(self, *args, **kargs):
        if False:
            for i in range(10):
                print('nop')
        'Set the brush used to draw the legend background.\n\n        Accepts the same arguments as :func:`~pyqtgraph.mkBrush`.\n        '
        brush = fn.mkBrush(*args, **kargs)
        if self.opts['brush'] == brush:
            return
        self.opts['brush'] = brush
        self.update()

    def labelTextColor(self):
        if False:
            i = 10
            return i + 15
        'Get the QColor used for the item labels.'
        return self.opts['labelTextColor']

    def setLabelTextColor(self, *args, **kargs):
        if False:
            while True:
                i = 10
        'Set the color of the item labels.\n\n        Accepts the same arguments as :func:`~pyqtgraph.mkColor`.\n        '
        self.opts['labelTextColor'] = fn.mkColor(*args, **kargs)
        for (sample, label) in self.items:
            label.setAttr('color', self.opts['labelTextColor'])
        self.update()

    def labelTextSize(self):
        if False:
            return 10
        'Get the `labelTextSize` used for the item labels.'
        return self.opts['labelTextSize']

    def setLabelTextSize(self, size):
        if False:
            i = 10
            return i + 15
        "Set the `size` of the item labels.\n\n        Accepts the CSS style string arguments, e.g. '8pt'.\n        "
        self.opts['labelTextSize'] = size
        for (_, label) in self.items:
            label.setAttr('size', self.opts['labelTextSize'])
        self.update()

    def setParentItem(self, p):
        if False:
            i = 10
            return i + 15
        'Set the parent.'
        ret = GraphicsWidget.setParentItem(self, p)
        if self.opts['offset'] is not None:
            offset = Point(self.opts['offset'])
            anchorx = 1 if offset[0] <= 0 else 0
            anchory = 1 if offset[1] <= 0 else 0
            anchor = (anchorx, anchory)
            self.anchor(itemPos=anchor, parentPos=anchor, offset=offset)
        return ret

    def addItem(self, item, name):
        if False:
            print('Hello World!')
        '\n        Add a new entry to the legend.\n\n        ==============  ========================================================\n        **Arguments:**\n        item            A :class:`~pyqtgraph.PlotDataItem` from which the line\n                        and point style of the item will be determined or an\n                        instance of ItemSample (or a subclass), allowing the\n                        item display to be customized.\n        title           The title to display for this item. Simple HTML allowed.\n        ==============  ========================================================\n        '
        label = LabelItem(name, color=self.opts['labelTextColor'], justify='left', size=self.opts['labelTextSize'])
        if isinstance(item, self.sampleType):
            sample = item
        else:
            sample = self.sampleType(item)
        self.items.append((sample, label))
        self._addItemToLayout(sample, label)
        self.updateSize()

    def _addItemToLayout(self, sample, label):
        if False:
            i = 10
            return i + 15
        col = self.layout.columnCount()
        row = self.layout.rowCount()
        if row:
            row -= 1
        nCol = self.columnCount * 2
        if col == nCol:
            for col in range(0, nCol, 2):
                if not self.layout.itemAt(row, col):
                    break
            else:
                if col + 2 == nCol:
                    col = 0
                    row += 1
        self.layout.addItem(sample, row, col)
        self.layout.addItem(label, row, col + 1)
        self.rowCount = max(self.rowCount, row + 1)

    def setColumnCount(self, columnCount):
        if False:
            print('Hello World!')
        'change the orientation of all items of the legend\n        '
        if columnCount != self.columnCount:
            self.columnCount = columnCount
            self.rowCount = math.ceil(len(self.items) / columnCount)
            for i in range(self.layout.count() - 1, -1, -1):
                self.layout.removeAt(i)
            for (sample, label) in self.items:
                self._addItemToLayout(sample, label)
            self.updateSize()

    def getLabel(self, plotItem):
        if False:
            return 10
        'Return the labelItem inside the legend for a given plotItem\n\n        The label-text can be changed via labelItem.setText\n        '
        out = [(it, lab) for (it, lab) in self.items if it.item == plotItem]
        try:
            return out[0][1]
        except IndexError:
            return None

    def _removeItemFromLayout(self, *args):
        if False:
            for i in range(10):
                print('nop')
        for item in args:
            self.layout.removeItem(item)
            item.close()
            scene = item.scene()
            if scene:
                scene.removeItem(item)

    def removeItem(self, item):
        if False:
            return 10
        'Removes one item from the legend.\n\n        ==============  ========================================================\n        **Arguments:**\n        item            The item to remove or its name.\n        ==============  ========================================================\n        '
        for (sample, label) in self.items:
            if sample.item is item or label.text == item:
                self.items.remove((sample, label))
                self._removeItemFromLayout(sample, label)
                self.updateSize()
                return

    def clear(self):
        if False:
            while True:
                i = 10
        'Remove all items from the legend.'
        for (sample, label) in self.items:
            self._removeItemFromLayout(sample, label)
        self.items = []
        self.updateSize()

    def updateSize(self):
        if False:
            while True:
                i = 10
        if self.size is not None:
            return
        height = 0
        width = 0
        for row in range(self.layout.rowCount()):
            row_height = 0
            col_width = 0
            for col in range(self.layout.columnCount()):
                item = self.layout.itemAt(row, col)
                if item:
                    col_width += item.width() + 3
                    row_height = max(row_height, item.height())
            width = max(width, col_width)
            height += row_height
        self.setGeometry(0, 0, width, height)
        return

    def boundingRect(self):
        if False:
            return 10
        return QtCore.QRectF(0, 0, self.width(), self.height())

    def paint(self, p, *args):
        if False:
            while True:
                i = 10
        if self.frame:
            p.setPen(self.opts['pen'])
            p.setBrush(self.opts['brush'])
            p.drawRect(self.boundingRect())

    def hoverEvent(self, ev):
        if False:
            while True:
                i = 10
        ev.acceptDrags(QtCore.Qt.MouseButton.LeftButton)

    def mouseDragEvent(self, ev):
        if False:
            while True:
                i = 10
        if ev.button() == QtCore.Qt.MouseButton.LeftButton:
            ev.accept()
            dpos = ev.pos() - ev.lastPos()
            self.autoAnchor(self.pos() + dpos)

class ItemSample(GraphicsWidget):
    """Class responsible for drawing a single item in a LegendItem (sans label)
    """

    def __init__(self, item):
        if False:
            i = 10
            return i + 15
        GraphicsWidget.__init__(self)
        self.item = item

    def boundingRect(self):
        if False:
            print('Hello World!')
        return QtCore.QRectF(0, 0, 20, 20)

    def paint(self, p, *args):
        if False:
            print('Hello World!')
        opts = self.item.opts
        if opts.get('antialias'):
            p.setRenderHint(p.RenderHint.Antialiasing)
        visible = self.item.isVisible()
        if not visible:
            icon = invisibleEye.qicon
            p.drawPixmap(QtCore.QPoint(1, 1), icon.pixmap(18, 18))
            return
        if not isinstance(self.item, ScatterPlotItem):
            p.setPen(fn.mkPen(opts['pen']))
            p.drawLine(0, 11, 20, 11)
            if opts.get('fillLevel', None) is not None and opts.get('fillBrush', None) is not None:
                p.setBrush(fn.mkBrush(opts['fillBrush']))
                p.setPen(fn.mkPen(opts['pen']))
                p.drawPolygon(QtGui.QPolygonF([QtCore.QPointF(2, 18), QtCore.QPointF(18, 2), QtCore.QPointF(18, 18)]))
        symbol = opts.get('symbol', None)
        if symbol is not None:
            if isinstance(self.item, PlotDataItem):
                opts = self.item.scatter.opts
            p.translate(10, 10)
            drawSymbol(p, symbol, opts['size'], fn.mkPen(opts['pen']), fn.mkBrush(opts['brush']))
        if isinstance(self.item, BarGraphItem):
            p.setBrush(fn.mkBrush(opts['brush']))
            p.drawRect(QtCore.QRectF(2, 2, 18, 18))

    def mouseClickEvent(self, event):
        if False:
            print('Hello World!')
        'Use the mouseClick event to toggle the visibility of the plotItem\n        '
        if event.button() == QtCore.Qt.MouseButton.LeftButton:
            visible = self.item.isVisible()
            self.item.setVisible(not visible)
        event.accept()
        self.update()