"""This file contains code for use with "Think Stats",
by Allen B. Downey, available from greenteapress.com

Copyright 2014 Allen B. Downey
License: GNU GPLv3 http://www.gnu.org/licenses/gpl.html
"""
from __future__ import print_function
import math
import matplotlib
import matplotlib.pyplot as pyplot
import numpy as np
import pandas
import warnings

class _Brewer(object):
    """Encapsulates a nice sequence of colors.

    Shades of blue that look good in color and can be distinguished
    in grayscale (up to a point).
    
    Borrowed from http://colorbrewer2.org/
    """
    color_iter = None
    colors = ['#081D58', '#253494', '#225EA8', '#1D91C0', '#41B6C4', '#7FCDBB', '#C7E9B4', '#EDF8B1', '#FFFFD9']
    which_colors = [[], [1], [1, 3], [0, 2, 4], [0, 2, 4, 6], [0, 2, 3, 5, 6], [0, 2, 3, 4, 5, 6], [0, 1, 2, 3, 4, 5, 6]]

    @classmethod
    def Colors(cls):
        if False:
            while True:
                i = 10
        'Returns the list of colors.\n        '
        return cls.colors

    @classmethod
    def ColorGenerator(cls, n):
        if False:
            i = 10
            return i + 15
        'Returns an iterator of color strings.\n\n        n: how many colors will be used\n        '
        for i in cls.which_colors[n]:
            yield cls.colors[i]
        raise StopIteration('Ran out of colors in _Brewer.ColorGenerator')

    @classmethod
    def InitializeIter(cls, num):
        if False:
            i = 10
            return i + 15
        'Initializes the color iterator with the given number of colors.'
        cls.color_iter = cls.ColorGenerator(num)

    @classmethod
    def ClearIter(cls):
        if False:
            for i in range(10):
                print('nop')
        'Sets the color iterator to None.'
        cls.color_iter = None

    @classmethod
    def GetIter(cls):
        if False:
            return 10
        'Gets the color iterator.'
        if cls.color_iter is None:
            cls.InitializeIter(7)
        return cls.color_iter

def PrePlot(num=None, rows=None, cols=None):
    if False:
        print('Hello World!')
    "Takes hints about what's coming.\n\n    num: number of lines that will be plotted\n    rows: number of rows of subplots\n    cols: number of columns of subplots\n    "
    if num:
        _Brewer.InitializeIter(num)
    if rows is None and cols is None:
        return
    if rows is not None and cols is None:
        cols = 1
    if cols is not None and rows is None:
        rows = 1
    size_map = {(1, 1): (8, 6), (1, 2): (14, 6), (1, 3): (14, 6), (2, 2): (10, 10), (2, 3): (16, 10), (3, 1): (8, 10)}
    if (rows, cols) in size_map:
        fig = pyplot.gcf()
        fig.set_size_inches(*size_map[rows, cols])
    if rows > 1 or cols > 1:
        pyplot.subplot(rows, cols, 1)
        global SUBPLOT_ROWS, SUBPLOT_COLS
        SUBPLOT_ROWS = rows
        SUBPLOT_COLS = cols

def SubPlot(plot_number, rows=None, cols=None):
    if False:
        while True:
            i = 10
    'Configures the number of subplots and changes the current plot.\n\n    rows: int\n    cols: int\n    plot_number: int\n    '
    rows = rows or SUBPLOT_ROWS
    cols = cols or SUBPLOT_COLS
    pyplot.subplot(rows, cols, plot_number)

def _Underride(d, **options):
    if False:
        print('Hello World!')
    'Add key-value pairs to d only if key is not in d.\n\n    If d is None, create a new dictionary.\n\n    d: dictionary\n    options: keyword args to add to d\n    '
    if d is None:
        d = {}
    for (key, val) in options.items():
        d.setdefault(key, val)
    return d

def Clf():
    if False:
        for i in range(10):
            print('nop')
    'Clears the figure and any hints that have been set.'
    global LOC
    LOC = None
    _Brewer.ClearIter()
    pyplot.clf()
    fig = pyplot.gcf()
    fig.set_size_inches(8, 6)

def Figure(**options):
    if False:
        for i in range(10):
            print('nop')
    'Sets options for the current figure.'
    _Underride(options, figsize=(6, 8))
    pyplot.figure(**options)

def _UnderrideColor(options):
    if False:
        i = 10
        return i + 15
    if 'color' in options:
        return options
    color_iter = _Brewer.GetIter()
    if color_iter:
        try:
            options['color'] = next(color_iter)
        except StopIteration:
            _Brewer.ClearIter()
    return options

def Plot(obj, ys=None, style='', **options):
    if False:
        return 10
    'Plots a line.\n\n    Args:\n      obj: sequence of x values, or Series, or anything with Render()\n      ys: sequence of y values\n      style: style string passed along to pyplot.plot\n      options: keyword args passed to pyplot.plot\n    '
    options = _UnderrideColor(options)
    label = getattr(obj, 'label', '_nolegend_')
    options = _Underride(options, linewidth=3, alpha=0.8, label=label)
    xs = obj
    if ys is None:
        if hasattr(obj, 'Render'):
            (xs, ys) = obj.Render()
        if isinstance(obj, pandas.Series):
            ys = obj.values
            xs = obj.index
    if ys is None:
        pyplot.plot(xs, style, **options)
    else:
        pyplot.plot(xs, ys, style, **options)

def FillBetween(xs, y1, y2=None, where=None, **options):
    if False:
        while True:
            i = 10
    'Plots a line.\n\n    Args:\n      xs: sequence of x values\n      y1: sequence of y values\n      y2: sequence of y values\n      where: sequence of boolean\n      options: keyword args passed to pyplot.fill_between\n    '
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.5)
    pyplot.fill_between(xs, y1, y2, where, **options)

def Bar(xs, ys, **options):
    if False:
        for i in range(10):
            print('nop')
    'Plots a line.\n\n    Args:\n      xs: sequence of x values\n      ys: sequence of y values\n      options: keyword args passed to pyplot.bar\n    '
    options = _UnderrideColor(options)
    options = _Underride(options, linewidth=0, alpha=0.6)
    pyplot.bar(xs, ys, **options)

def Scatter(xs, ys=None, **options):
    if False:
        for i in range(10):
            print('nop')
    'Makes a scatter plot.\n\n    xs: x values\n    ys: y values\n    options: options passed to pyplot.scatter\n    '
    options = _Underride(options, color='blue', alpha=0.2, s=30, edgecolors='none')
    if ys is None and isinstance(xs, pandas.Series):
        ys = xs.values
        xs = xs.index
    pyplot.scatter(xs, ys, **options)

def HexBin(xs, ys, **options):
    if False:
        for i in range(10):
            print('nop')
    'Makes a scatter plot.\n\n    xs: x values\n    ys: y values\n    options: options passed to pyplot.scatter\n    '
    options = _Underride(options, cmap=matplotlib.cm.Blues)
    pyplot.hexbin(xs, ys, **options)

def Pdf(pdf, **options):
    if False:
        print('Hello World!')
    'Plots a Pdf, Pmf, or Hist as a line.\n\n    Args:\n      pdf: Pdf, Pmf, or Hist object\n      options: keyword args passed to pyplot.plot\n    '
    (low, high) = (options.pop('low', None), options.pop('high', None))
    n = options.pop('n', 101)
    (xs, ps) = pdf.Render(low=low, high=high, n=n)
    options = _Underride(options, label=pdf.label)
    Plot(xs, ps, **options)

def Pdfs(pdfs, **options):
    if False:
        for i in range(10):
            print('nop')
    'Plots a sequence of PDFs.\n\n    Options are passed along for all PDFs.  If you want different\n    options for each pdf, make multiple calls to Pdf.\n    \n    Args:\n      pdfs: sequence of PDF objects\n      options: keyword args passed to pyplot.plot\n    '
    for pdf in pdfs:
        Pdf(pdf, **options)

def Hist(hist, **options):
    if False:
        i = 10
        return i + 15
    "Plots a Pmf or Hist with a bar plot.\n\n    The default width of the bars is based on the minimum difference\n    between values in the Hist.  If that's too small, you can override\n    it by providing a width keyword argument, in the same units\n    as the values.\n\n    Args:\n      hist: Hist or Pmf object\n      options: keyword args passed to pyplot.bar\n    "
    (xs, ys) = hist.Render()
    if 'width' not in options:
        try:
            options['width'] = 0.9 * np.diff(xs).min()
        except TypeError:
            warnings.warn("Hist: Can't compute bar width automatically.Check for non-numeric types in Hist.Or try providing width option.")
    options = _Underride(options, label=hist.label)
    options = _Underride(options, align='center')
    if options['align'] == 'left':
        options['align'] = 'edge'
    elif options['align'] == 'right':
        options['align'] = 'edge'
        options['width'] *= -1
    Bar(xs, ys, **options)

def Hists(hists, **options):
    if False:
        while True:
            i = 10
    'Plots two histograms as interleaved bar plots.\n\n    Options are passed along for all PMFs.  If you want different\n    options for each pmf, make multiple calls to Pmf.\n\n    Args:\n      hists: list of two Hist or Pmf objects\n      options: keyword args passed to pyplot.plot\n    '
    for hist in hists:
        Hist(hist, **options)

def Pmf(pmf, **options):
    if False:
        print('Hello World!')
    'Plots a Pmf or Hist as a line.\n\n    Args:\n      pmf: Hist or Pmf object\n      options: keyword args passed to pyplot.plot\n    '
    (xs, ys) = pmf.Render()
    (low, high) = (min(xs), max(xs))
    width = options.pop('width', None)
    if width is None:
        try:
            width = np.diff(xs).min()
        except TypeError:
            warnings.warn("Pmf: Can't compute bar width automatically.Check for non-numeric types in Pmf.Or try providing width option.")
    points = []
    lastx = np.nan
    lasty = 0
    for (x, y) in zip(xs, ys):
        if x - lastx > 1e-05:
            points.append((lastx, 0))
            points.append((x, 0))
        points.append((x, lasty))
        points.append((x, y))
        points.append((x + width, y))
        lastx = x + width
        lasty = y
    points.append((lastx, 0))
    (pxs, pys) = zip(*points)
    align = options.pop('align', 'center')
    if align == 'center':
        pxs = np.array(pxs) - width / 2.0
    if align == 'right':
        pxs = np.array(pxs) - width
    options = _Underride(options, label=pmf.label)
    Plot(pxs, pys, **options)

def Pmfs(pmfs, **options):
    if False:
        print('Hello World!')
    'Plots a sequence of PMFs.\n\n    Options are passed along for all PMFs.  If you want different\n    options for each pmf, make multiple calls to Pmf.\n    \n    Args:\n      pmfs: sequence of PMF objects\n      options: keyword args passed to pyplot.plot\n    '
    for pmf in pmfs:
        Pmf(pmf, **options)

def Diff(t):
    if False:
        return 10
    'Compute the differences between adjacent elements in a sequence.\n\n    Args:\n        t: sequence of number\n\n    Returns:\n        sequence of differences (length one less than t)\n    '
    diffs = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    return diffs

def Cdf(cdf, complement=False, transform=None, **options):
    if False:
        while True:
            i = 10
    "Plots a CDF as a line.\n\n    Args:\n      cdf: Cdf object\n      complement: boolean, whether to plot the complementary CDF\n      transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'\n      options: keyword args passed to pyplot.plot\n\n    Returns:\n      dictionary with the scale options that should be passed to\n      Config, Show or Save.\n    "
    (xs, ps) = cdf.Render()
    xs = np.asarray(xs)
    ps = np.asarray(ps)
    scale = dict(xscale='linear', yscale='linear')
    for s in ['xscale', 'yscale']:
        if s in options:
            scale[s] = options.pop(s)
    if transform == 'exponential':
        complement = True
        scale['yscale'] = 'log'
    if transform == 'pareto':
        complement = True
        scale['yscale'] = 'log'
        scale['xscale'] = 'log'
    if complement:
        ps = [1.0 - p for p in ps]
    if transform == 'weibull':
        xs = np.delete(xs, -1)
        ps = np.delete(ps, -1)
        ps = [-math.log(1.0 - p) for p in ps]
        scale['xscale'] = 'log'
        scale['yscale'] = 'log'
    if transform == 'gumbel':
        xs = xp.delete(xs, 0)
        ps = np.delete(ps, 0)
        ps = [-math.log(p) for p in ps]
        scale['yscale'] = 'log'
    options = _Underride(options, label=cdf.label)
    Plot(xs, ps, **options)
    return scale

def Cdfs(cdfs, complement=False, transform=None, **options):
    if False:
        while True:
            i = 10
    "Plots a sequence of CDFs.\n    \n    cdfs: sequence of CDF objects\n    complement: boolean, whether to plot the complementary CDF\n    transform: string, one of 'exponential', 'pareto', 'weibull', 'gumbel'\n    options: keyword args passed to pyplot.plot\n    "
    for cdf in cdfs:
        Cdf(cdf, complement, transform, **options)

def Contour(obj, pcolor=False, contour=True, imshow=False, **options):
    if False:
        return 10
    'Makes a contour plot.\n    \n    d: map from (x, y) to z, or object that provides GetDict\n    pcolor: boolean, whether to make a pseudocolor plot\n    contour: boolean, whether to make a contour plot\n    imshow: boolean, whether to use pyplot.imshow\n    options: keyword args passed to pyplot.pcolor and/or pyplot.contour\n    '
    try:
        d = obj.GetDict()
    except AttributeError:
        d = obj
    _Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)
    (xs, ys) = zip(*d.keys())
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    (X, Y) = np.meshgrid(xs, ys)
    func = lambda x, y: d.get((x, y), 0)
    func = np.vectorize(func)
    Z = func(X, Y)
    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)
    if pcolor:
        pyplot.pcolormesh(X, Y, Z, **options)
    if contour:
        cs = pyplot.contour(X, Y, Z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)
    if imshow:
        extent = (xs[0], xs[-1], ys[0], ys[-1])
        pyplot.imshow(Z, extent=extent, **options)

def Pcolor(xs, ys, zs, pcolor=True, contour=False, **options):
    if False:
        i = 10
        return i + 15
    'Makes a pseudocolor plot.\n    \n    xs:\n    ys:\n    zs:\n    pcolor: boolean, whether to make a pseudocolor plot\n    contour: boolean, whether to make a contour plot\n    options: keyword args passed to pyplot.pcolor and/or pyplot.contour\n    '
    _Underride(options, linewidth=3, cmap=matplotlib.cm.Blues)
    (X, Y) = np.meshgrid(xs, ys)
    Z = zs
    x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
    axes = pyplot.gca()
    axes.xaxis.set_major_formatter(x_formatter)
    if pcolor:
        pyplot.pcolormesh(X, Y, Z, **options)
    if contour:
        cs = pyplot.contour(X, Y, Z, **options)
        pyplot.clabel(cs, inline=1, fontsize=10)

def Text(x, y, s, **options):
    if False:
        for i in range(10):
            print('nop')
    'Puts text in a figure.\n\n    x: number\n    y: number\n    s: string\n    options: keyword args passed to pyplot.text\n    '
    options = _Underride(options, fontsize=16, verticalalignment='top', horizontalalignment='left')
    pyplot.text(x, y, s, **options)
LEGEND = True
LOC = None

def Config(**options):
    if False:
        for i in range(10):
            print('nop')
    'Configures the plot.\n\n    Pulls options out of the option dictionary and passes them to\n    the corresponding pyplot functions.\n    '
    names = ['title', 'xlabel', 'ylabel', 'xscale', 'yscale', 'xticks', 'yticks', 'axis', 'xlim', 'ylim']
    for name in names:
        if name in options:
            getattr(pyplot, name)(options[name])
    loc_dict = {'upper right': 1, 'upper left': 2, 'lower left': 3, 'lower right': 4, 'right': 5, 'center left': 6, 'center right': 7, 'lower center': 8, 'upper center': 9, 'center': 10}
    global LEGEND
    LEGEND = options.get('legend', LEGEND)
    if LEGEND:
        global LOC
        LOC = options.get('loc', LOC)
        pyplot.legend(loc=LOC)

def Show(**options):
    if False:
        return 10
    'Shows the plot.\n\n    For options, see Config.\n\n    options: keyword args used to invoke various pyplot functions\n    '
    clf = options.pop('clf', True)
    Config(**options)
    pyplot.show()
    if clf:
        Clf()

def Plotly(**options):
    if False:
        for i in range(10):
            print('nop')
    'Shows the plot.\n\n    For options, see Config.\n\n    options: keyword args used to invoke various pyplot functions\n    '
    clf = options.pop('clf', True)
    Config(**options)
    import plotly.plotly as plotly
    url = plotly.plot_mpl(pyplot.gcf())
    if clf:
        Clf()
    return url

def Save(root=None, formats=None, **options):
    if False:
        print('Hello World!')
    'Saves the plot in the given formats and clears the figure.\n\n    For options, see Config.\n\n    Args:\n      root: string filename root\n      formats: list of string formats\n      options: keyword args used to invoke various pyplot functions\n    '
    clf = options.pop('clf', True)
    Config(**options)
    if formats is None:
        formats = ['pdf', 'eps']
    try:
        formats.remove('plotly')
        Plotly(clf=False)
    except ValueError:
        pass
    if root:
        for fmt in formats:
            SaveFormat(root, fmt)
    if clf:
        Clf()

def SaveFormat(root, fmt='eps'):
    if False:
        i = 10
        return i + 15
    'Writes the current figure to a file in the given format.\n\n    Args:\n      root: string filename root\n      fmt: string format\n    '
    filename = '%s.%s' % (root, fmt)
    print('Writing', filename)
    pyplot.savefig(filename, format=fmt, dpi=300)
preplot = PrePlot
subplot = SubPlot
clf = Clf
figure = Figure
plot = Plot
text = Text
scatter = Scatter
pmf = Pmf
pmfs = Pmfs
hist = Hist
hists = Hists
diff = Diff
cdf = Cdf
cdfs = Cdfs
contour = Contour
pcolor = Pcolor
config = Config
show = Show
save = Save

def main():
    if False:
        print('Hello World!')
    color_iter = _Brewer.ColorGenerator(7)
    for color in color_iter:
        print(color)
if __name__ == '__main__':
    main()