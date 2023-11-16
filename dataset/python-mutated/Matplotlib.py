from .. import PlotItem
from .. import functions as fn
from ..Qt import QtCore, QtWidgets
from .Exporter import Exporter
__all__ = ['MatplotlibExporter']
"\nIt is helpful when using the matplotlib Exporter if your\n.matplotlib/matplotlibrc file is configured appropriately.\nThe following are suggested for getting usable PDF output that\ncan be edited in Illustrator, etc.\n\nbackend      : Qt4Agg\ntext.usetex : True  # Assumes you have a findable LaTeX installation\ninteractive : False\nfont.family : sans-serif\nfont.sans-serif : 'Arial'  # (make first in list)\nmathtext.default : sf\nfigure.facecolor : white  # personal preference\n# next setting allows pdf font to be readable in Adobe Illustrator\npdf.fonttype : 42   # set fonts to TrueType (otherwise it will be 3\n                    # and the text will be vectorized.\ntext.dvipnghack : True  # primarily to clean up font appearance on Mac\n\nThe advantage is that there is less to do to get an exported file cleaned and ready for\npublication. Fonts are not vectorized (outlined), and window colors are white.\n\n"
_symbol_pg_to_mpl = {'o': 'o', 's': 's', 't': 'v', 't1': '^', 't2': '>', 't3': '<', 'd': 'd', '+': 'P', 'x': 'X', 'p': 'p', 'h': 'h', 'star': '*', 'arrow_up': 6, 'arrow_right': 5, 'arrow_down': 7, 'arrow_left': 4, 'crosshair': 'o'}

class MatplotlibExporter(Exporter):
    Name = 'Matplotlib Window'
    windows = []

    def __init__(self, item):
        if False:
            return 10
        Exporter.__init__(self, item)

    def parameters(self):
        if False:
            i = 10
            return i + 15
        return None

    def cleanAxes(self, axl):
        if False:
            i = 10
            return i + 15
        if type(axl) is not list:
            axl = [axl]
        for ax in axl:
            if ax is None:
                continue
            for (loc, spine) in ax.spines.items():
                if loc in ['left', 'bottom']:
                    pass
                elif loc in ['right', 'top']:
                    spine.set_color('none')
                else:
                    raise ValueError('Unknown spine location: %s' % loc)
                ax.xaxis.set_ticks_position('bottom')

    def export(self, fileName=None):
        if False:
            print('Hello World!')
        if not isinstance(self.item, PlotItem):
            raise Exception('MatplotlibExporter currently only works with PlotItem')
        mpw = MatplotlibWindow()
        MatplotlibExporter.windows.append(mpw)
        fig = mpw.getFigure()
        xax = self.item.getAxis('bottom')
        yax = self.item.getAxis('left')
        xlabel = xax.label.toPlainText()
        ylabel = yax.label.toPlainText()
        title = self.item.titleLabel.text
        xscale = yscale = 1.0
        if xax.autoSIPrefix:
            xscale = xax.autoSIPrefixScale
        if yax.autoSIPrefix:
            yscale = yax.autoSIPrefixScale
        ax = fig.add_subplot(111, title=title)
        ax.clear()
        self.cleanAxes(ax)
        for item in self.item.curves:
            (x, y) = item.getData()
            x = x * xscale
            y = y * yscale
            opts = item.opts
            pen = fn.mkPen(opts['pen'])
            if pen.style() == QtCore.Qt.PenStyle.NoPen:
                linestyle = ''
            else:
                linestyle = '-'
            color = pen.color().getRgbF()
            symbol = opts['symbol']
            symbol = _symbol_pg_to_mpl.get(symbol, '')
            symbolPen = fn.mkPen(opts['symbolPen'])
            symbolBrush = fn.mkBrush(opts['symbolBrush'])
            markeredgecolor = symbolPen.color().getRgbF()
            markerfacecolor = symbolBrush.color().getRgbF()
            markersize = opts['symbolSize']
            if opts['fillLevel'] is not None and opts['fillBrush'] is not None:
                fillBrush = fn.mkBrush(opts['fillBrush'])
                fillcolor = fillBrush.color().getRgbF()
                ax.fill_between(x=x, y1=y, y2=opts['fillLevel'], facecolor=fillcolor)
            ax.plot(x, y, marker=symbol, color=color, linewidth=pen.width(), linestyle=linestyle, markeredgecolor=markeredgecolor, markerfacecolor=markerfacecolor, markersize=markersize)
            (xr, yr) = self.item.viewRange()
            ax.set_xbound(xr[0] * xscale, xr[1] * xscale)
            ax.set_ybound(yr[0] * yscale, yr[1] * yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        mpw.draw()
MatplotlibExporter.register()

class MatplotlibWindow(QtWidgets.QMainWindow):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        from ..widgets import MatplotlibWidget
        QtWidgets.QMainWindow.__init__(self)
        self.mpl = MatplotlibWidget.MatplotlibWidget()
        self.setCentralWidget(self.mpl)
        self.show()

    def __getattr__(self, attr):
        if False:
            i = 10
            return i + 15
        return getattr(self.mpl, attr)

    def closeEvent(self, ev):
        if False:
            for i in range(10):
                print('nop')
        MatplotlibExporter.windows.remove(self)
        self.deleteLater()