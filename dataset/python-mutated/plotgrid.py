from sympy.external import import_module
import sympy.plotting.backends.base_backend as base_backend

class PlotGrid:
    """This class helps to plot subplots from already created SymPy plots
    in a single figure.

    Examples
    ========

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> from sympy import symbols
        >>> from sympy.plotting import plot, plot3d, PlotGrid
        >>> x, y = symbols('x, y')
        >>> p1 = plot(x, x**2, x**3, (x, -5, 5))
        >>> p2 = plot((x**2, (x, -6, 6)), (x, (x, -5, 5)))
        >>> p3 = plot(x**3, (x, -5, 5))
        >>> p4 = plot3d(x*y, (x, -5, 5), (y, -5, 5))

    Plotting vertically in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 1, p1, p2)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)

    Plotting horizontally in a single line:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(1, 3, p2, p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    Plotting in a grid form:

    .. plot::
       :context: close-figs
       :format: doctest
       :include-source: True

        >>> PlotGrid(2, 2, p1, p2, p3, p4)
        PlotGrid object containing:
        Plot[0]:Plot object containing:
        [0]: cartesian line: x for x over (-5.0, 5.0)
        [1]: cartesian line: x**2 for x over (-5.0, 5.0)
        [2]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[1]:Plot object containing:
        [0]: cartesian line: x**2 for x over (-6.0, 6.0)
        [1]: cartesian line: x for x over (-5.0, 5.0)
        Plot[2]:Plot object containing:
        [0]: cartesian line: x**3 for x over (-5.0, 5.0)
        Plot[3]:Plot object containing:
        [0]: cartesian surface: x*y for x over (-5.0, 5.0) and y over (-5.0, 5.0)

    """

    def __init__(self, nrows, ncolumns, *args, show=True, size=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ==========\n\n        nrows :\n            The number of rows that should be in the grid of the\n            required subplot.\n        ncolumns :\n            The number of columns that should be in the grid\n            of the required subplot.\n\n        nrows and ncolumns together define the required grid.\n\n        Arguments\n        =========\n\n        A list of predefined plot objects entered in a row-wise sequence\n        i.e. plot objects which are to be in the top row of the required\n        grid are written first, then the second row objects and so on\n\n        Keyword arguments\n        =================\n\n        show : Boolean\n            The default value is set to ``True``. Set show to ``False`` and\n            the function will not display the subplot. The returned instance\n            of the ``PlotGrid`` class can then be used to save or display the\n            plot by calling the ``save()`` and ``show()`` methods\n            respectively.\n        size : (float, float), optional\n            A tuple in the form (width, height) in inches to specify the size of\n            the overall figure. The default value is set to ``None``, meaning\n            the size will be set by the default backend.\n        '
        self.matplotlib = import_module('matplotlib', import_kwargs={'fromlist': ['pyplot', 'cm', 'collections']}, min_module_version='1.1.0', catch=(RuntimeError,))
        self.nrows = nrows
        self.ncolumns = ncolumns
        self._series = []
        self._fig = None
        self.args = args
        for arg in args:
            self._series.append(arg._series)
        self.size = size
        if show and self.matplotlib:
            self.show()

    def _create_figure(self):
        if False:
            return 10
        gs = self.matplotlib.gridspec.GridSpec(self.nrows, self.ncolumns)
        mapping = {}
        c = 0
        for i in range(self.nrows):
            for j in range(self.ncolumns):
                if c < len(self.args):
                    mapping[gs[i, j]] = self.args[c]
                c += 1
        kw = {} if not self.size else {'figsize': self.size}
        self._fig = self.matplotlib.pyplot.figure(**kw)
        for (spec, p) in mapping.items():
            kw = {'projection': '3d'} if len(p._series) > 0 and p._series[0].is_3D else {}
            cur_ax = self._fig.add_subplot(spec, **kw)
            p._plotgrid_fig = self._fig
            p._plotgrid_ax = cur_ax
            p.process_series()

    @property
    def fig(self):
        if False:
            print('Hello World!')
        if not self._fig:
            self._create_figure()
        return self._fig

    @property
    def _backend(self):
        if False:
            while True:
                i = 10
        return self

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.matplotlib.pyplot.close(self.fig)

    def show(self):
        if False:
            return 10
        if base_backend._show:
            self.fig.tight_layout()
            self.matplotlib.pyplot.show()
        else:
            self.close()

    def save(self, path):
        if False:
            print('Hello World!')
        self.fig.savefig(path)

    def __str__(self):
        if False:
            return 10
        plot_strs = ['Plot[%d]:' % i + str(plot) for (i, plot) in enumerate(self.args)]
        return 'PlotGrid object containing:\n' + '\n'.join(plot_strs)