"""
MultiPlotItem.py -  Graphics item used for displaying an array of PlotItems
Copyright 2010  Luke Campagnola
Distributed under MIT/X11 license. See license.txt for more information.
"""
from . import GraphicsLayout
__all__ = ['MultiPlotItem']

class MultiPlotItem(GraphicsLayout.GraphicsLayout):
    """
    :class:`~pyqtgraph.GraphicsLayout` that automatically generates a grid of
    plots from a MetaArray.

    .. seealso:: :class:`~pyqtgraph.MultiPlotWidget`: Widget containing a MultiPlotItem
    """

    def __init__(self, *args, **kwds):
        if False:
            print('Hello World!')
        GraphicsLayout.GraphicsLayout.__init__(self, *args, **kwds)
        self.plots = []

    def plot(self, data, **plotArgs):
        if False:
            print('Hello World!')
        'Plot the data from a MetaArray with each array column as a separate\n        :class:`~pyqtgraph.PlotItem`.\n\n        Axis labels are automatically extracted from the array info.\n\n        ``plotArgs`` are passed to :meth:`PlotItem.plot\n        <pyqtgraph.PlotItem.plot>`.\n        '
        if hasattr(data, 'implements') and data.implements('MetaArray'):
            if data.ndim != 2:
                raise Exception('MultiPlot currently only accepts 2D MetaArray.')
            ic = data.infoCopy()
            ax = 0
            for i in [0, 1]:
                if 'cols' in ic[i]:
                    ax = i
                    break
            for i in range(data.shape[ax]):
                pi = self.addPlot()
                self.nextRow()
                sl = [slice(None)] * 2
                sl[ax] = i
                pi.plot(data[tuple(sl)], **plotArgs)
                self.plots.append((pi, i, 0))
                info = ic[ax]['cols'][i]
                title = info.get('title', info.get('name', None))
                units = info.get('units', None)
                pi.setLabel('left', text=title, units=units)
            info = ic[1 - ax]
            title = info.get('title', info.get('name', None))
            units = info.get('units', None)
            pi.setLabel('bottom', text=title, units=units)
        else:
            raise Exception('Data type %s not (yet?) supported for MultiPlot.' % type(data))

    def close(self):
        if False:
            i = 10
            return i + 15
        for p in self.plots:
            p[0].close()
        self.plots = None
        self.clear()