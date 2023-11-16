import PySimpleGUI as sg
import matplotlib
import inspect
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
'\nDemonstrates one way of embedding Matplotlib figures into a PySimpleGUI window.\n\nBasic steps are:\n * Create a Canvas Element\n * Layout form\n * Display form (NON BLOCKING)\n * Draw plots onto convas\n * Display form (BLOCKING)\n \nEach plotting function, complete with imports, was copied directly from Matplot examples page \n'
import numpy as np
import matplotlib.pyplot as plt

def PyplotSimple():
    if False:
        while True:
            i = 10
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.arange(0.0, 5.0, 0.2)
    plt.plot(t, t, 'r--', t, t ** 2, 'bs', t, t ** 3, 'g^')
    fig = plt.gcf()
    return fig

def PyplotHistogram():
    if False:
        while True:
            i = 10
    '\n    =============================================================\n    Demo of the histogram (hist) function with multiple data sets\n    =============================================================\n\n    Plot histogram with multiple sample sets and demonstrate:\n\n        * Use of legend with multiple sample sets\n        * Stacked bars\n        * Step curve with no fill\n        * Data sets of different sample sizes\n\n    Selecting different bin counts and sizes can significantly affect the\n    shape of a histogram. The Astropy docs have a great section on how to\n    select these parameters:\n    http://docs.astropy.org/en/stable/visualization/histogram.html\n    '
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(0)
    n_bins = 10
    x = np.random.randn(1000, 3)
    (fig, axes) = plt.subplots(nrows=2, ncols=2)
    (ax0, ax1, ax2, ax3) = axes.flatten()
    colors = ['red', 'tan', 'lime']
    ax0.hist(x, n_bins, normed=1, histtype='bar', color=colors, label=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('bars with legend')
    ax1.hist(x, n_bins, normed=1, histtype='bar', stacked=True)
    ax1.set_title('stacked bar')
    ax2.hist(x, n_bins, histtype='step', stacked=True, fill=False)
    ax2.set_title('stack step (unfilled)')
    x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]
    ax3.hist(x_multi, n_bins, histtype='bar')
    ax3.set_title('different sample sizes')
    fig.tight_layout()
    return fig

def PyplotArtistBoxPlots():
    if False:
        print('Hello World!')
    '\n    =========================================\n    Demo of artist customization in box plots\n    =========================================\n\n    This example demonstrates how to use the various kwargs\n    to fully customize box plots. The first figure demonstrates\n    how to remove and add individual components (note that the\n    mean is the only value not shown by default). The second\n    figure demonstrates how the styles of the artists can\n    be customized. It also demonstrates how to set the limit\n    of the whiskers to specific percentiles (lower right axes)\n\n    A good general reference on boxplots and their history can be found\n    here: http://vita.had.co.nz/papers/boxplots.pdf\n\n    '
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(937)
    data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
    labels = list('ABCD')
    fs = 10
    (fig, axes) = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
    axes[0, 0].boxplot(data, labels=labels)
    axes[0, 0].set_title('Default', fontsize=fs)
    axes[0, 1].boxplot(data, labels=labels, showmeans=True)
    axes[0, 1].set_title('showmeans=True', fontsize=fs)
    axes[0, 2].boxplot(data, labels=labels, showmeans=True, meanline=True)
    axes[0, 2].set_title('showmeans=True,\nmeanline=True', fontsize=fs)
    axes[1, 0].boxplot(data, labels=labels, showbox=False, showcaps=False)
    tufte_title = 'Tufte Style \n(showbox=False,\nshowcaps=False)'
    axes[1, 0].set_title(tufte_title, fontsize=fs)
    axes[1, 1].boxplot(data, labels=labels, notch=True, bootstrap=10000)
    axes[1, 1].set_title('notch=True,\nbootstrap=10000', fontsize=fs)
    axes[1, 2].boxplot(data, labels=labels, showfliers=False)
    axes[1, 2].set_title('showfliers=False', fontsize=fs)
    for ax in axes.flatten():
        ax.set_yscale('log')
        ax.set_yticklabels([])
    fig.subplots_adjust(hspace=0.4)
    return fig

def ArtistBoxplot2():
    if False:
        i = 10
        return i + 15
    np.random.seed(937)
    data = np.random.lognormal(size=(37, 4), mean=1.5, sigma=1.75)
    labels = list('ABCD')
    fs = 10
    boxprops = dict(linestyle='--', linewidth=3, color='darkgoldenrod')
    flierprops = dict(marker='o', markerfacecolor='green', markersize=12, linestyle='none')
    medianprops = dict(linestyle='-.', linewidth=2.5, color='firebrick')
    meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
    meanlineprops = dict(linestyle='--', linewidth=2.5, color='purple')
    (fig, axes) = plt.subplots(nrows=2, ncols=3, figsize=(6, 6), sharey=True)
    axes[0, 0].boxplot(data, boxprops=boxprops)
    axes[0, 0].set_title('Custom boxprops', fontsize=fs)
    axes[0, 1].boxplot(data, flierprops=flierprops, medianprops=medianprops)
    axes[0, 1].set_title('Custom medianprops\nand flierprops', fontsize=fs)
    axes[0, 2].boxplot(data, whis='range')
    axes[0, 2].set_title('whis="range"', fontsize=fs)
    axes[1, 0].boxplot(data, meanprops=meanpointprops, meanline=False, showmeans=True)
    axes[1, 0].set_title('Custom mean\nas point', fontsize=fs)
    axes[1, 1].boxplot(data, meanprops=meanlineprops, meanline=True, showmeans=True)
    axes[1, 1].set_title('Custom mean\nas line', fontsize=fs)
    axes[1, 2].boxplot(data, whis=[15, 85])
    axes[1, 2].set_title('whis=[15, 85]\n#percentiles', fontsize=fs)
    for ax in axes.flatten():
        ax.set_yscale('log')
        ax.set_yticklabels([])
    fig.suptitle("I never said they'd be pretty")
    fig.subplots_adjust(hspace=0.4)
    return fig

def PyplotScatterWithLegend():
    if False:
        print('Hello World!')
    import matplotlib.pyplot as plt
    from numpy.random import rand
    (fig, ax) = plt.subplots()
    for color in ['red', 'green', 'blue']:
        n = 750
        (x, y) = rand(2, n)
        scale = 200.0 * rand(n)
        ax.scatter(x, y, c=color, s=scale, label=color, alpha=0.3, edgecolors='none')
    ax.legend()
    ax.grid(True)
    return fig

def PyplotLineStyles():
    if False:
        for i in range(10):
            print('nop')
    '\n    ==========\n    Linestyles\n    ==========\n\n    This examples showcases different linestyles copying those of Tikz/PGF.\n    '
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import OrderedDict
    from matplotlib.transforms import blended_transform_factory
    linestyles = OrderedDict([('solid', (0, ())), ('loosely dotted', (0, (1, 10))), ('dotted', (0, (1, 5))), ('densely dotted', (0, (1, 1))), ('loosely dashed', (0, (5, 10))), ('dashed', (0, (5, 5))), ('densely dashed', (0, (5, 1))), ('loosely dashdotted', (0, (3, 10, 1, 10))), ('dashdotted', (0, (3, 5, 1, 5))), ('densely dashdotted', (0, (3, 1, 1, 1))), ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))), ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))), ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))])
    plt.figure(figsize=(10, 6))
    ax = plt.subplot(1, 1, 1)
    (X, Y) = (np.linspace(0, 100, 10), np.zeros(10))
    for (i, (name, linestyle)) in enumerate(linestyles.items()):
        ax.plot(X, Y + i, linestyle=linestyle, linewidth=1.5, color='black')
    ax.set_ylim(-0.5, len(linestyles) - 0.5)
    plt.yticks(np.arange(len(linestyles)), linestyles.keys())
    plt.xticks([])
    reference_transform = blended_transform_factory(ax.transAxes, ax.transData)
    for (i, (name, linestyle)) in enumerate(linestyles.items()):
        ax.annotate(str(linestyle), xy=(0.0, i), xycoords=reference_transform, xytext=(-6, -12), textcoords='offset points', color='blue', fontsize=8, ha='right', family='monospace')
    plt.tight_layout()
    return plt.gcf()

def PyplotLinePolyCollection():
    if False:
        return 10
    import matplotlib.pyplot as plt
    from matplotlib import collections, colors, transforms
    import numpy as np
    nverts = 50
    npts = 100
    r = np.arange(nverts)
    theta = np.linspace(0, 2 * np.pi, nverts)
    xx = r * np.sin(theta)
    yy = r * np.cos(theta)
    spiral = np.column_stack([xx, yy])
    rs = np.random.RandomState(19680801)
    xyo = rs.randn(npts, 2)
    colors = [colors.to_rgba(c) for c in plt.rcParams['axes.prop_cycle'].by_key()['color']]
    (fig, axes) = plt.subplots(2, 2)
    fig.subplots_adjust(top=0.92, left=0.07, right=0.97, hspace=0.3, wspace=0.3)
    ((ax1, ax2), (ax3, ax4)) = axes
    col = collections.LineCollection([spiral], offsets=xyo, transOffset=ax1.transData)
    trans = fig.dpi_scale_trans + transforms.Affine2D().scale(1.0 / 72.0)
    col.set_transform(trans)
    ax1.add_collection(col, autolim=True)
    col.set_color(colors)
    ax1.autoscale_view()
    ax1.set_title('LineCollection using offsets')
    col = collections.PolyCollection([spiral], offsets=xyo, transOffset=ax2.transData)
    trans = transforms.Affine2D().scale(fig.dpi / 72.0)
    col.set_transform(trans)
    ax2.add_collection(col, autolim=True)
    col.set_color(colors)
    ax2.autoscale_view()
    ax2.set_title('PolyCollection using offsets')
    col = collections.RegularPolyCollection(7, sizes=np.abs(xx) * 10.0, offsets=xyo, transOffset=ax3.transData)
    trans = transforms.Affine2D().scale(fig.dpi / 72.0)
    col.set_transform(trans)
    ax3.add_collection(col, autolim=True)
    col.set_color(colors)
    ax3.autoscale_view()
    ax3.set_title('RegularPolyCollection using offsets')
    nverts = 60
    ncurves = 20
    offs = (0.1, 0.0)
    yy = np.linspace(0, 2 * np.pi, nverts)
    ym = np.max(yy)
    xx = (0.2 + (ym - yy) / ym) ** 2 * np.cos(yy - 0.4) * 0.5
    segs = []
    for i in range(ncurves):
        xxx = xx + 0.02 * rs.randn(nverts)
        curve = np.column_stack([xxx, yy * 100])
        segs.append(curve)
    col = collections.LineCollection(segs, offsets=offs)
    ax4.add_collection(col, autolim=True)
    col.set_color(colors)
    ax4.autoscale_view()
    ax4.set_title('Successive data offsets')
    ax4.set_xlabel('Zonal velocity component (m/s)')
    ax4.set_ylabel('Depth (m)')
    ax4.set_ylim(ax4.get_ylim()[::-1])
    return fig

def PyplotGGPlotSytleSheet():
    if False:
        print('Hello World!')
    import numpy as np
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    np.random.seed(19680801)
    (fig, axes) = plt.subplots(ncols=2, nrows=2)
    (ax1, ax2, ax3, ax4) = axes.ravel()
    (x, y) = np.random.normal(size=(2, 200))
    ax1.plot(x, y, 'o')
    L = 2 * np.pi
    x = np.linspace(0, L)
    ncolors = len(plt.rcParams['axes.prop_cycle'])
    shift = np.linspace(0, L, ncolors, endpoint=False)
    for s in shift:
        ax2.plot(x, np.sin(x + s), '-')
    ax2.margins(0)
    x = np.arange(5)
    (y1, y2) = np.random.randint(1, 25, size=(2, 5))
    width = 0.25
    ax3.bar(x, y1, width)
    ax3.bar(x + width, y2, width, color=list(plt.rcParams['axes.prop_cycle'])[2]['color'])
    ax3.set_xticks(x + width)
    ax3.set_xticklabels(['a', 'b', 'c', 'd', 'e'])
    for (i, color) in enumerate(plt.rcParams['axes.prop_cycle']):
        xy = np.random.normal(size=2)
        ax4.add_patch(plt.Circle(xy, radius=0.3, color=color['color']))
    ax4.axis('equal')
    ax4.margins(0)
    fig = plt.gcf()
    return fig

def PyplotBoxPlot():
    if False:
        i = 10
        return i + 15
    import numpy as np
    import matplotlib.pyplot as plt
    np.random.seed(19680801)
    spread = np.random.rand(50) * 100
    center = np.ones(25) * 50
    flier_high = np.random.rand(10) * 100 + 100
    flier_low = np.random.rand(10) * -100
    data = np.concatenate((spread, center, flier_high, flier_low), 0)
    (fig1, ax1) = plt.subplots()
    ax1.set_title('Basic Plot')
    ax1.boxplot(data)
    return fig1

def PyplotRadarChart():
    if False:
        return 10
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from matplotlib.spines import Spine
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection

    def radar_factory(num_vars, frame='circle'):
        if False:
            while True:
                i = 10
        "Create a radar chart with `num_vars` axes.\n\n        This function creates a RadarAxes projection and registers it.\n\n        Parameters\n        ----------\n        num_vars : int\n            Number of variables for radar chart.\n        frame : {'circle' | 'polygon'}\n            Shape of frame surrounding axes.\n\n        "
        theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

        def draw_poly_patch(self):
            if False:
                i = 10
                return i + 15
            verts = unit_poly_verts(theta + np.pi / 2)
            return plt.Polygon(verts, closed=True, edgecolor='k')

        def draw_circle_patch(self):
            if False:
                i = 10
                return i + 15
            return plt.Circle((0.5, 0.5), 0.5)
        patch_dict = {'polygon': draw_poly_patch, 'circle': draw_circle_patch}
        if frame not in patch_dict:
            raise ValueError('unknown value for `frame`: %s' % frame)

        class RadarAxes(PolarAxes):
            name = 'radar'
            RESOLUTION = 1
            draw_patch = patch_dict[frame]

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                super(RadarAxes, self).__init__(*args, **kwargs)
                self.set_theta_zero_location('N')

            def fill(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                'Override fill so that line is closed by default'
                closed = kwargs.pop('closed', True)
                return super(RadarAxes, self).fill(*args, closed=closed, **kwargs)

            def plot(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                'Override plot so that line is closed by default'
                lines = super(RadarAxes, self).plot(*args, **kwargs)
                for line in lines:
                    self._close_line(line)

            def _close_line(self, line):
                if False:
                    for i in range(10):
                        print('nop')
                (x, y) = line.get_data()
                if x[0] != x[-1]:
                    x = np.concatenate((x, [x[0]]))
                    y = np.concatenate((y, [y[0]]))
                    line.set_data(x, y)

            def set_varlabels(self, labels):
                if False:
                    while True:
                        i = 10
                self.set_thetagrids(np.degrees(theta), labels)

            def _gen_axes_patch(self):
                if False:
                    while True:
                        i = 10
                return self.draw_patch()

            def _gen_axes_spines(self):
                if False:
                    return 10
                if frame == 'circle':
                    return PolarAxes._gen_axes_spines(self)
                spine_type = 'circle'
                verts = unit_poly_verts(theta + np.pi / 2)
                verts.append(verts[0])
                path = Path(verts)
                spine = Spine(self, spine_type, path)
                spine.set_transform(self.transAxes)
                return {'polar': spine}
        register_projection(RadarAxes)
        return theta

    def unit_poly_verts(theta):
        if False:
            i = 10
            return i + 15
        'Return vertices of polygon for subplot axes.\n\n        This polygon is circumscribed by a unit circle centered at (0.5, 0.5)\n        '
        (x0, y0, r) = [0.5] * 3
        verts = [(r * np.cos(t) + x0, r * np.sin(t) + y0) for t in theta]
        return verts

    def example_data():
        if False:
            print('Hello World!')
        data = [['Sulfate', 'Nitrate', 'EC', 'OC1', 'OC2', 'OC3', 'OP', 'CO', 'O3'], ('Basecase', [[0.88, 0.01, 0.03, 0.03, 0.0, 0.06, 0.01, 0.0, 0.0], [0.07, 0.95, 0.04, 0.05, 0.0, 0.02, 0.01, 0.0, 0.0], [0.01, 0.02, 0.85, 0.19, 0.05, 0.1, 0.0, 0.0, 0.0], [0.02, 0.01, 0.07, 0.01, 0.21, 0.12, 0.98, 0.0, 0.0], [0.01, 0.01, 0.02, 0.71, 0.74, 0.7, 0.0, 0.0, 0.0]]), ('With CO', [[0.88, 0.02, 0.02, 0.02, 0.0, 0.05, 0.0, 0.05, 0.0], [0.08, 0.94, 0.04, 0.02, 0.0, 0.01, 0.12, 0.04, 0.0], [0.01, 0.01, 0.79, 0.1, 0.0, 0.05, 0.0, 0.31, 0.0], [0.0, 0.02, 0.03, 0.38, 0.31, 0.31, 0.0, 0.59, 0.0], [0.02, 0.02, 0.11, 0.47, 0.69, 0.58, 0.88, 0.0, 0.0]]), ('With O3', [[0.89, 0.01, 0.07, 0.0, 0.0, 0.05, 0.0, 0.0, 0.03], [0.07, 0.95, 0.05, 0.04, 0.0, 0.02, 0.12, 0.0, 0.0], [0.01, 0.02, 0.86, 0.27, 0.16, 0.19, 0.0, 0.0, 0.0], [0.01, 0.03, 0.0, 0.32, 0.29, 0.27, 0.0, 0.0, 0.95], [0.02, 0.0, 0.03, 0.37, 0.56, 0.47, 0.87, 0.0, 0.0]]), ('CO & O3', [[0.87, 0.01, 0.08, 0.0, 0.0, 0.04, 0.0, 0.0, 0.01], [0.09, 0.95, 0.02, 0.03, 0.0, 0.01, 0.13, 0.06, 0.0], [0.01, 0.02, 0.71, 0.24, 0.13, 0.16, 0.0, 0.5, 0.0], [0.01, 0.03, 0.0, 0.28, 0.24, 0.23, 0.0, 0.44, 0.88], [0.02, 0.0, 0.18, 0.45, 0.64, 0.55, 0.86, 0.0, 0.16]])]
        return data
    N = 9
    theta = radar_factory(N, frame='polygon')
    data = example_data()
    spoke_labels = data.pop(0)
    (fig, axes) = plt.subplots(figsize=(9, 9), nrows=2, ncols=2, subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.2, top=0.85, bottom=0.05)
    colors = ['b', 'r', 'g', 'm', 'y']
    for (ax, (title, case_data)) in zip(axes.flatten(), data):
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1), horizontalalignment='center', verticalalignment='center')
        for (d, color) in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25)
        ax.set_varlabels(spoke_labels)
    ax = axes[0, 0]
    labels = ('Factor 1', 'Factor 2', 'Factor 3', 'Factor 4', 'Factor 5')
    legend = ax.legend(labels, loc=(0.9, 0.95), labelspacing=0.1, fontsize='small')
    fig.text(0.5, 0.965, '5-Factor Solution Profiles Across Four Scenarios', horizontalalignment='center', color='black', weight='bold', size='large')
    return fig

def DifferentScales():
    if False:
        while True:
            i = 10
    import numpy as np
    import matplotlib.pyplot as plt
    t = np.arange(0.01, 10.0, 0.01)
    data1 = np.exp(t)
    data2 = np.sin(2 * np.pi * t)
    (fig, ax1) = plt.subplots()
    color = 'tab:red'
    ax1.set_xlabel('time (s)')
    ax1.set_ylabel('exp', color=color)
    ax1.plot(t, data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('sin', color=color)
    ax2.plot(t, data2, color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    fig.tight_layout()
    return fig

def ExploringNormalizations():
    if False:
        return 10
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    from numpy.random import multivariate_normal
    data = np.vstack([multivariate_normal([10, 10], [[3, 2], [2, 3]], size=100000), multivariate_normal([30, 20], [[2, 3], [1, 3]], size=1000)])
    gammas = [0.8, 0.5, 0.3]
    (fig, axes) = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].set_title('Linear normalization')
    axes[0, 0].hist2d(data[:, 0], data[:, 1], bins=100)
    for (ax, gamma) in zip(axes.flat[1:], gammas):
        ax.set_title('Power law $(\\gamma=%1.1f)$' % gamma)
        ax.hist2d(data[:, 0], data[:, 1], bins=100, norm=mcolors.PowerNorm(gamma))
    fig.tight_layout()
    return fig

def PyplotFormatstr():
    if False:
        print('Hello World!')

    def f(t):
        if False:
            i = 10
            return i + 15
        return np.exp(-t) * np.cos(2 * np.pi * t)
    t1 = np.arange(0.0, 5.0, 0.1)
    t2 = np.arange(0.0, 5.0, 0.02)
    plt.figure(1)
    plt.subplot(211)
    plt.plot(t1, f(t1), 'bo', t2, f(t2), 'k')
    plt.subplot(212)
    plt.plot(t2, np.cos(2 * np.pi * t2), 'r--')
    fig = plt.gcf()
    return fig

def UnicodeMinus():
    if False:
        return 10
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    np.random.seed(19680801)
    matplotlib.rcParams['axes.unicode_minus'] = False
    (fig, ax) = plt.subplots()
    ax.plot(10 * np.random.randn(100), 10 * np.random.randn(100), 'o')
    ax.set_title('Using hyphen instead of Unicode minus')
    return fig

def Subplot3d():
    if False:
        for i in range(10):
            print('nop')
    from mpl_toolkits.mplot3d.axes3d import Axes3D
    from matplotlib import cm
    import matplotlib.pyplot as plt
    import numpy as np
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    X = np.arange(-5, 5, 0.25)
    Y = np.arange(-5, 5, 0.25)
    (X, Y) = np.meshgrid(X, Y)
    R = np.sqrt(X ** 2 + Y ** 2)
    Z = np.sin(R)
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.jet, linewidth=0, antialiased=False)
    ax.set_zlim3d(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    from mpl_toolkits.mplot3d.axes3d import get_test_data
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    (X, Y, Z) = get_test_data(0.05)
    ax.plot_wireframe(X, Y, Z, rstride=10, cstride=10)
    return fig

def PyplotScales():
    if False:
        i = 10
        return i + 15
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    np.random.seed(19680801)
    y = np.random.normal(loc=0.5, scale=0.4, size=1000)
    y = y[(y > 0) & (y < 1)]
    y.sort()
    x = np.arange(len(y))
    plt.figure(1)
    plt.subplot(221)
    plt.plot(x, y)
    plt.yscale('linear')
    plt.title('linear')
    plt.grid(True)
    plt.subplot(222)
    plt.plot(x, y)
    plt.yscale('log')
    plt.title('log')
    plt.grid(True)
    plt.subplot(223)
    plt.plot(x, y - y.mean())
    plt.yscale('symlog', linthreshy=0.01)
    plt.title('symlog')
    plt.grid(True)
    plt.subplot(224)
    plt.plot(x, y)
    plt.yscale('logit')
    plt.title('logit')
    plt.grid(True)
    plt.gca().yaxis.set_minor_formatter(NullFormatter())
    plt.subplots_adjust(top=0.92, bottom=0.08, left=0.1, right=0.95, hspace=0.25, wspace=0.35)
    return plt.gcf()

def AxesGrid():
    if False:
        print('Hello World!')
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.axes_rgb import RGBAxes

    def get_demo_image():
        if False:
            for i in range(10):
                print('nop')
        delta = 0.5
        extent = (-3, 4, -4, 3)
        x = np.arange(-3.0, 4.001, delta)
        y = np.arange(-4.0, 3.001, delta)
        (X, Y) = np.meshgrid(x, y)
        Z1 = np.exp(-X ** 2 - Y ** 2)
        Z2 = np.exp(-(X - 1) ** 2 - (Y - 1) ** 2)
        Z = (Z1 - Z2) * 2
        return (Z, extent)

    def get_rgb():
        if False:
            for i in range(10):
                print('nop')
        (Z, extent) = get_demo_image()
        Z[Z < 0] = 0.0
        Z = Z / Z.max()
        R = Z[:13, :13]
        G = Z[2:, 2:]
        B = Z[:13, 2:]
        return (R, G, B)
    fig = plt.figure(1)
    ax = RGBAxes(fig, [0.1, 0.1, 0.8, 0.8])
    (r, g, b) = get_rgb()
    kwargs = dict(origin='lower', interpolation='nearest')
    ax.imshow_rgb(r, g, b, **kwargs)
    ax.RGB.set_xlim(0.0, 9.5)
    ax.RGB.set_ylim(0.9, 10.6)
    plt.draw()
    return plt.gcf()

def draw_figure(canvas, figure):
    if False:
        i = 10
        return i + 15
    if not hasattr(draw_figure, 'canvas_packed'):
        draw_figure.canvas_packed = {}
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    widget = figure_canvas_agg.get_tk_widget()
    if widget not in draw_figure.canvas_packed:
        draw_figure.canvas_packed[widget] = figure
        widget.pack(side='top', fill='both', expand=1)
    return figure_canvas_agg

def delete_figure_agg(figure_agg):
    if False:
        for i in range(10):
            print('nop')
    figure_agg.get_tk_widget().forget()
    try:
        draw_figure.canvas_packed.pop(figure_agg.get_tk_widget())
    except Exception as e:
        print(f'Error removing {figure_agg} from list', e)
    plt.close('all')
fig_dict = {'Pyplot Simple': PyplotSimple, 'Pyplot Formatstr': PyplotFormatstr, 'PyPlot Three': Subplot3d, 'Unicode Minus': UnicodeMinus, 'Pyplot Scales': PyplotScales, 'Axes Grid': AxesGrid, 'Exploring Normalizations': ExploringNormalizations, 'Different Scales': DifferentScales, 'Pyplot Box Plot': PyplotBoxPlot, 'Pyplot ggplot Style Sheet': PyplotGGPlotSytleSheet, 'Pyplot Line Poly Collection': PyplotLinePolyCollection, 'Pyplot Line Styles': PyplotLineStyles, 'Pyplot Scatter With Legend': PyplotScatterWithLegend, 'Artist Customized Box Plots': PyplotArtistBoxPlots, 'Artist Customized Box Plots 2': ArtistBoxplot2, 'Pyplot Histogram': PyplotHistogram}
sg.theme('LightGreen')
(figure_w, figure_h) = (650, 650)
listbox_values = list(fig_dict)
col_listbox = [[sg.Listbox(values=listbox_values, enable_events=True, size=(28, len(listbox_values)), key='-LISTBOX-')], [sg.Text(' ' * 12), sg.Exit(size=(5, 2))]]
layout = [[sg.Text('Matplotlib Plot Test', font='current 18')], [sg.Col(col_listbox, pad=(5, (3, 330))), sg.Canvas(size=(figure_w, figure_h), key='-CANVAS-'), sg.MLine(size=(70, 35), pad=(5, (3, 90)), key='-MULTILINE-')]]
window = sg.Window('Demo Application - Embedding Matplotlib In PySimpleGUI', layout, grab_anywhere=False, finalize=True)
figure_agg = None
while True:
    (event, values) = window.read()
    if event in (sg.WIN_CLOSED, 'Exit'):
        break
    if figure_agg:
        delete_figure_agg(figure_agg)
    choice = values['-LISTBOX-'][0]
    func = fig_dict[choice]
    window['-MULTILINE-'].update(inspect.getsource(func))
    try:
        fig = func()
        figure_agg = draw_figure(window['-CANVAS-'].TKCanvas, fig)
    except Exception as e:
        print('Exception in fucntion', e)
window.close()