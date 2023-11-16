"""
:mod:`~matplotlib.gridspec` contains classes that help to layout multiple
`~.axes.Axes` in a grid-like pattern within a figure.

The `GridSpec` specifies the overall grid structure. Individual cells within
the grid are referenced by `SubplotSpec`\\s.

Often, users need not access this module directly, and can use higher-level
methods like `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` and
`~.Figure.subfigures`. See the tutorial :ref:`arranging_axes` for a guide.
"""
import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
_log = logging.getLogger(__name__)

class GridSpecBase:
    """
    A base class of GridSpec that specifies the geometry of the grid
    that a subplot will be placed.
    """

    def __init__(self, nrows, ncols, height_ratios=None, width_ratios=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Parameters\n        ----------\n        nrows, ncols : int\n            The number of rows and columns of the grid.\n        width_ratios : array-like of length *ncols*, optional\n            Defines the relative widths of the columns. Each column gets a\n            relative width of ``width_ratios[i] / sum(width_ratios)``.\n            If not given, all columns will have the same width.\n        height_ratios : array-like of length *nrows*, optional\n            Defines the relative heights of the rows. Each row gets a\n            relative height of ``height_ratios[i] / sum(height_ratios)``.\n            If not given, all rows will have the same height.\n        '
        if not isinstance(nrows, Integral) or nrows <= 0:
            raise ValueError(f'Number of rows must be a positive integer, not {nrows!r}')
        if not isinstance(ncols, Integral) or ncols <= 0:
            raise ValueError(f'Number of columns must be a positive integer, not {ncols!r}')
        (self._nrows, self._ncols) = (nrows, ncols)
        self.set_height_ratios(height_ratios)
        self.set_width_ratios(width_ratios)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        height_arg = f', height_ratios={self._row_height_ratios!r}' if len(set(self._row_height_ratios)) != 1 else ''
        width_arg = f', width_ratios={self._col_width_ratios!r}' if len(set(self._col_width_ratios)) != 1 else ''
        return '{clsname}({nrows}, {ncols}{optionals})'.format(clsname=self.__class__.__name__, nrows=self._nrows, ncols=self._ncols, optionals=height_arg + width_arg)
    nrows = property(lambda self: self._nrows, doc='The number of rows in the grid.')
    ncols = property(lambda self: self._ncols, doc='The number of columns in the grid.')

    def get_geometry(self):
        if False:
            print('Hello World!')
        '\n        Return a tuple containing the number of rows and columns in the grid.\n        '
        return (self._nrows, self._ncols)

    def get_subplot_params(self, figure=None):
        if False:
            return 10
        pass

    def new_subplotspec(self, loc, rowspan=1, colspan=1):
        if False:
            while True:
                i = 10
        '\n        Create and return a `.SubplotSpec` instance.\n\n        Parameters\n        ----------\n        loc : (int, int)\n            The position of the subplot in the grid as\n            ``(row_index, column_index)``.\n        rowspan, colspan : int, default: 1\n            The number of rows and columns the subplot should span in the grid.\n        '
        (loc1, loc2) = loc
        subplotspec = self[loc1:loc1 + rowspan, loc2:loc2 + colspan]
        return subplotspec

    def set_width_ratios(self, width_ratios):
        if False:
            return 10
        '\n        Set the relative widths of the columns.\n\n        *width_ratios* must be of length *ncols*. Each column gets a relative\n        width of ``width_ratios[i] / sum(width_ratios)``.\n        '
        if width_ratios is None:
            width_ratios = [1] * self._ncols
        elif len(width_ratios) != self._ncols:
            raise ValueError('Expected the given number of width ratios to match the number of columns of the grid')
        self._col_width_ratios = width_ratios

    def get_width_ratios(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the width ratios.\n\n        This is *None* if no width ratios have been set explicitly.\n        '
        return self._col_width_ratios

    def set_height_ratios(self, height_ratios):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the relative heights of the rows.\n\n        *height_ratios* must be of length *nrows*. Each row gets a relative\n        height of ``height_ratios[i] / sum(height_ratios)``.\n        '
        if height_ratios is None:
            height_ratios = [1] * self._nrows
        elif len(height_ratios) != self._nrows:
            raise ValueError('Expected the given number of height ratios to match the number of rows of the grid')
        self._row_height_ratios = height_ratios

    def get_height_ratios(self):
        if False:
            while True:
                i = 10
        '\n        Return the height ratios.\n\n        This is *None* if no height ratios have been set explicitly.\n        '
        return self._row_height_ratios

    def get_grid_positions(self, fig):
        if False:
            while True:
                i = 10
        '\n        Return the positions of the grid cells in figure coordinates.\n\n        Parameters\n        ----------\n        fig : `~matplotlib.figure.Figure`\n            The figure the grid should be applied to. The subplot parameters\n            (margins and spacing between subplots) are taken from *fig*.\n\n        Returns\n        -------\n        bottoms, tops, lefts, rights : array\n            The bottom, top, left, right positions of the grid cells in\n            figure coordinates.\n        '
        (nrows, ncols) = self.get_geometry()
        subplot_params = self.get_subplot_params(fig)
        left = subplot_params.left
        right = subplot_params.right
        bottom = subplot_params.bottom
        top = subplot_params.top
        wspace = subplot_params.wspace
        hspace = subplot_params.hspace
        tot_width = right - left
        tot_height = top - bottom
        cell_h = tot_height / (nrows + hspace * (nrows - 1))
        sep_h = hspace * cell_h
        norm = cell_h * nrows / sum(self._row_height_ratios)
        cell_heights = [r * norm for r in self._row_height_ratios]
        sep_heights = [0] + [sep_h] * (nrows - 1)
        cell_hs = np.cumsum(np.column_stack([sep_heights, cell_heights]).flat)
        cell_w = tot_width / (ncols + wspace * (ncols - 1))
        sep_w = wspace * cell_w
        norm = cell_w * ncols / sum(self._col_width_ratios)
        cell_widths = [r * norm for r in self._col_width_ratios]
        sep_widths = [0] + [sep_w] * (ncols - 1)
        cell_ws = np.cumsum(np.column_stack([sep_widths, cell_widths]).flat)
        (fig_tops, fig_bottoms) = (top - cell_hs).reshape((-1, 2)).T
        (fig_lefts, fig_rights) = (left + cell_ws).reshape((-1, 2)).T
        return (fig_bottoms, fig_tops, fig_lefts, fig_rights)

    @staticmethod
    def _check_gridspec_exists(figure, nrows, ncols):
        if False:
            i = 10
            return i + 15
        '\n        Check if the figure already has a gridspec with these dimensions,\n        or create a new one\n        '
        for ax in figure.get_axes():
            gs = ax.get_gridspec()
            if gs is not None:
                if hasattr(gs, 'get_topmost_subplotspec'):
                    gs = gs.get_topmost_subplotspec().get_gridspec()
                if gs.get_geometry() == (nrows, ncols):
                    return gs
        return GridSpec(nrows, ncols, figure=figure)

    def __getitem__(self, key):
        if False:
            while True:
                i = 10
        'Create and return a `.SubplotSpec` instance.'
        (nrows, ncols) = self.get_geometry()

        def _normalize(key, size, axis):
            if False:
                return 10
            orig_key = key
            if isinstance(key, slice):
                (start, stop, _) = key.indices(size)
                if stop > start:
                    return (start, stop - 1)
                raise IndexError('GridSpec slice would result in no space allocated for subplot')
            else:
                if key < 0:
                    key = key + size
                if 0 <= key < size:
                    return (key, key)
                elif axis is not None:
                    raise IndexError(f'index {orig_key} is out of bounds for axis {axis} with size {size}')
                else:
                    raise IndexError(f'index {orig_key} is out of bounds for GridSpec with size {size}')
        if isinstance(key, tuple):
            try:
                (k1, k2) = key
            except ValueError as err:
                raise ValueError('Unrecognized subplot spec') from err
            (num1, num2) = np.ravel_multi_index([_normalize(k1, nrows, 0), _normalize(k2, ncols, 1)], (nrows, ncols))
        else:
            (num1, num2) = _normalize(key, nrows * ncols, None)
        return SubplotSpec(self, num1, num2)

    def subplots(self, *, sharex=False, sharey=False, squeeze=True, subplot_kw=None):
        if False:
            return 10
        '\n        Add all subplots specified by this `GridSpec` to its parent figure.\n\n        See `.Figure.subplots` for detailed documentation.\n        '
        figure = self.figure
        if figure is None:
            raise ValueError('GridSpec.subplots() only works for GridSpecs created with a parent figure')
        if not isinstance(sharex, str):
            sharex = 'all' if sharex else 'none'
        if not isinstance(sharey, str):
            sharey = 'all' if sharey else 'none'
        _api.check_in_list(['all', 'row', 'col', 'none', False, True], sharex=sharex, sharey=sharey)
        if subplot_kw is None:
            subplot_kw = {}
        subplot_kw = subplot_kw.copy()
        axarr = np.empty((self._nrows, self._ncols), dtype=object)
        for row in range(self._nrows):
            for col in range(self._ncols):
                shared_with = {'none': None, 'all': axarr[0, 0], 'row': axarr[row, 0], 'col': axarr[0, col]}
                subplot_kw['sharex'] = shared_with[sharex]
                subplot_kw['sharey'] = shared_with[sharey]
                axarr[row, col] = figure.add_subplot(self[row, col], **subplot_kw)
        if sharex in ['col', 'all']:
            for ax in axarr.flat:
                ax._label_outer_xaxis(skip_non_rectangular_axes=True)
        if sharey in ['row', 'all']:
            for ax in axarr.flat:
                ax._label_outer_yaxis(skip_non_rectangular_axes=True)
        if squeeze:
            return axarr.item() if axarr.size == 1 else axarr.squeeze()
        else:
            return axarr

class GridSpec(GridSpecBase):
    """
    A grid layout to place subplots within a figure.

    The location of the grid cells is determined in a similar way to
    `.SubplotParams` using *left*, *right*, *top*, *bottom*, *wspace*
    and *hspace*.

    Indexing a GridSpec instance returns a `.SubplotSpec`.
    """

    def __init__(self, nrows, ncols, figure=None, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None, width_ratios=None, height_ratios=None):
        if False:
            print('Hello World!')
        '\n        Parameters\n        ----------\n        nrows, ncols : int\n            The number of rows and columns of the grid.\n\n        figure : `.Figure`, optional\n            Only used for constrained layout to create a proper layoutgrid.\n\n        left, right, top, bottom : float, optional\n            Extent of the subplots as a fraction of figure width or height.\n            Left cannot be larger than right, and bottom cannot be larger than\n            top. If not given, the values will be inferred from a figure or\n            rcParams at draw time. See also `GridSpec.get_subplot_params`.\n\n        wspace : float, optional\n            The amount of width reserved for space between subplots,\n            expressed as a fraction of the average axis width.\n            If not given, the values will be inferred from a figure or\n            rcParams when necessary. See also `GridSpec.get_subplot_params`.\n\n        hspace : float, optional\n            The amount of height reserved for space between subplots,\n            expressed as a fraction of the average axis height.\n            If not given, the values will be inferred from a figure or\n            rcParams when necessary. See also `GridSpec.get_subplot_params`.\n\n        width_ratios : array-like of length *ncols*, optional\n            Defines the relative widths of the columns. Each column gets a\n            relative width of ``width_ratios[i] / sum(width_ratios)``.\n            If not given, all columns will have the same width.\n\n        height_ratios : array-like of length *nrows*, optional\n            Defines the relative heights of the rows. Each row gets a\n            relative height of ``height_ratios[i] / sum(height_ratios)``.\n            If not given, all rows will have the same height.\n\n        '
        self.left = left
        self.bottom = bottom
        self.right = right
        self.top = top
        self.wspace = wspace
        self.hspace = hspace
        self.figure = figure
        super().__init__(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)
    _AllowedKeys = ['left', 'bottom', 'right', 'top', 'wspace', 'hspace']

    def update(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the subplot parameters of the grid.\n\n        Parameters that are not explicitly given are not changed. Setting a\n        parameter to *None* resets it to :rc:`figure.subplot.*`.\n\n        Parameters\n        ----------\n        left, right, top, bottom : float or None, optional\n            Extent of the subplots as a fraction of figure width or height.\n        wspace, hspace : float, optional\n            Spacing between the subplots as a fraction of the average subplot\n            width / height.\n        '
        for (k, v) in kwargs.items():
            if k in self._AllowedKeys:
                setattr(self, k, v)
            else:
                raise AttributeError(f'{k} is an unknown keyword')
        for figmanager in _pylab_helpers.Gcf.figs.values():
            for ax in figmanager.canvas.figure.axes:
                if ax.get_subplotspec() is not None:
                    ss = ax.get_subplotspec().get_topmost_subplotspec()
                    if ss.get_gridspec() == self:
                        ax._set_position(ax.get_subplotspec().get_position(ax.figure))

    def get_subplot_params(self, figure=None):
        if False:
            return 10
        '\n        Return the `.SubplotParams` for the GridSpec.\n\n        In order of precedence the values are taken from\n\n        - non-*None* attributes of the GridSpec\n        - the provided *figure*\n        - :rc:`figure.subplot.*`\n\n        Note that the ``figure`` attribute of the GridSpec is always ignored.\n        '
        if figure is None:
            kw = {k: mpl.rcParams['figure.subplot.' + k] for k in self._AllowedKeys}
            subplotpars = SubplotParams(**kw)
        else:
            subplotpars = copy.copy(figure.subplotpars)
        subplotpars.update(**{k: getattr(self, k) for k in self._AllowedKeys})
        return subplotpars

    def locally_modified_subplot_params(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of the names of the subplot parameters explicitly set\n        in the GridSpec.\n\n        This is a subset of the attributes of `.SubplotParams`.\n        '
        return [k for k in self._AllowedKeys if getattr(self, k)]

    def tight_layout(self, figure, renderer=None, pad=1.08, h_pad=None, w_pad=None, rect=None):
        if False:
            return 10
        '\n        Adjust subplot parameters to give specified padding.\n\n        Parameters\n        ----------\n        figure : `.Figure`\n            The figure.\n        renderer :  `.RendererBase` subclass, optional\n            The renderer to be used.\n        pad : float\n            Padding between the figure edge and the edges of subplots, as a\n            fraction of the font-size.\n        h_pad, w_pad : float, optional\n            Padding (height/width) between edges of adjacent subplots.\n            Defaults to *pad*.\n        rect : tuple (left, bottom, right, top), default: None\n            (left, bottom, right, top) rectangle in normalized figure\n            coordinates that the whole subplots area (including labels) will\n            fit into. Default (None) is the whole figure.\n        '
        if renderer is None:
            renderer = figure._get_renderer()
        kwargs = _tight_layout.get_tight_layout_figure(figure, figure.axes, _tight_layout.get_subplotspec_list(figure.axes, grid_spec=self), renderer, pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)
        if kwargs:
            self.update(**kwargs)

class GridSpecFromSubplotSpec(GridSpecBase):
    """
    GridSpec whose subplot layout parameters are inherited from the
    location specified by a given SubplotSpec.
    """

    def __init__(self, nrows, ncols, subplot_spec, wspace=None, hspace=None, height_ratios=None, width_ratios=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        nrows, ncols : int\n            Number of rows and number of columns of the grid.\n        subplot_spec : SubplotSpec\n            Spec from which the layout parameters are inherited.\n        wspace, hspace : float, optional\n            See `GridSpec` for more details. If not specified default values\n            (from the figure or rcParams) are used.\n        height_ratios : array-like of length *nrows*, optional\n            See `GridSpecBase` for details.\n        width_ratios : array-like of length *ncols*, optional\n            See `GridSpecBase` for details.\n        '
        self._wspace = wspace
        self._hspace = hspace
        self._subplot_spec = subplot_spec
        self.figure = self._subplot_spec.get_gridspec().figure
        super().__init__(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)

    def get_subplot_params(self, figure=None):
        if False:
            i = 10
            return i + 15
        'Return a dictionary of subplot layout parameters.'
        hspace = self._hspace if self._hspace is not None else figure.subplotpars.hspace if figure is not None else mpl.rcParams['figure.subplot.hspace']
        wspace = self._wspace if self._wspace is not None else figure.subplotpars.wspace if figure is not None else mpl.rcParams['figure.subplot.wspace']
        figbox = self._subplot_spec.get_position(figure)
        (left, bottom, right, top) = figbox.extents
        return SubplotParams(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)

    def get_topmost_subplotspec(self):
        if False:
            print('Hello World!')
        '\n        Return the topmost `.SubplotSpec` instance associated with the subplot.\n        '
        return self._subplot_spec.get_topmost_subplotspec()

class SubplotSpec:
    """
    The location of a subplot in a `GridSpec`.

    .. note::

        Likely, you will never instantiate a `SubplotSpec` yourself. Instead,
        you will typically obtain one from a `GridSpec` using item-access.

    Parameters
    ----------
    gridspec : `~matplotlib.gridspec.GridSpec`
        The GridSpec, which the subplot is referencing.
    num1, num2 : int
        The subplot will occupy the *num1*-th cell of the given
        *gridspec*.  If *num2* is provided, the subplot will span between
        *num1*-th cell and *num2*-th cell **inclusive**.

        The index starts from 0.
    """

    def __init__(self, gridspec, num1, num2=None):
        if False:
            return 10
        self._gridspec = gridspec
        self.num1 = num1
        self.num2 = num2

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return f'{self.get_gridspec()}[{self.rowspan.start}:{self.rowspan.stop}, {self.colspan.start}:{self.colspan.stop}]'

    @staticmethod
    def _from_subplot_args(figure, args):
        if False:
            print('Hello World!')
        '\n        Construct a `.SubplotSpec` from a parent `.Figure` and either\n\n        - a `.SubplotSpec` -- returned as is;\n        - one or three numbers -- a MATLAB-style subplot specifier.\n        '
        if len(args) == 1:
            (arg,) = args
            if isinstance(arg, SubplotSpec):
                return arg
            elif not isinstance(arg, Integral):
                raise ValueError(f'Single argument to subplot must be a three-digit integer, not {arg!r}')
            try:
                (rows, cols, num) = map(int, str(arg))
            except ValueError:
                raise ValueError(f'Single argument to subplot must be a three-digit integer, not {arg!r}') from None
        elif len(args) == 3:
            (rows, cols, num) = args
        else:
            raise _api.nargs_error('subplot', takes='1 or 3', given=len(args))
        gs = GridSpec._check_gridspec_exists(figure, rows, cols)
        if gs is None:
            gs = GridSpec(rows, cols, figure=figure)
        if isinstance(num, tuple) and len(num) == 2:
            if not all((isinstance(n, Integral) for n in num)):
                raise ValueError(f'Subplot specifier tuple must contain integers, not {num}')
            (i, j) = num
        else:
            if not isinstance(num, Integral) or num < 1 or num > rows * cols:
                raise ValueError(f'num must be an integer with 1 <= num <= {rows * cols}, not {num!r}')
            i = j = num
        return gs[i - 1:j]

    @property
    def num2(self):
        if False:
            return 10
        return self.num1 if self._num2 is None else self._num2

    @num2.setter
    def num2(self, value):
        if False:
            print('Hello World!')
        self._num2 = value

    def get_gridspec(self):
        if False:
            print('Hello World!')
        return self._gridspec

    def get_geometry(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the subplot geometry as tuple ``(n_rows, n_cols, start, stop)``.\n\n        The indices *start* and *stop* define the range of the subplot within\n        the `GridSpec`. *stop* is inclusive (i.e. for a single cell\n        ``start == stop``).\n        '
        (rows, cols) = self.get_gridspec().get_geometry()
        return (rows, cols, self.num1, self.num2)

    @property
    def rowspan(self):
        if False:
            print('Hello World!')
        'The rows spanned by this subplot, as a `range` object.'
        ncols = self.get_gridspec().ncols
        return range(self.num1 // ncols, self.num2 // ncols + 1)

    @property
    def colspan(self):
        if False:
            i = 10
            return i + 15
        'The columns spanned by this subplot, as a `range` object.'
        ncols = self.get_gridspec().ncols
        (c1, c2) = sorted([self.num1 % ncols, self.num2 % ncols])
        return range(c1, c2 + 1)

    def is_first_row(self):
        if False:
            while True:
                i = 10
        return self.rowspan.start == 0

    def is_last_row(self):
        if False:
            for i in range(10):
                print('nop')
        return self.rowspan.stop == self.get_gridspec().nrows

    def is_first_col(self):
        if False:
            i = 10
            return i + 15
        return self.colspan.start == 0

    def is_last_col(self):
        if False:
            print('Hello World!')
        return self.colspan.stop == self.get_gridspec().ncols

    def get_position(self, figure):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update the subplot position from ``figure.subplotpars``.\n        '
        gridspec = self.get_gridspec()
        (nrows, ncols) = gridspec.get_geometry()
        (rows, cols) = np.unravel_index([self.num1, self.num2], (nrows, ncols))
        (fig_bottoms, fig_tops, fig_lefts, fig_rights) = gridspec.get_grid_positions(figure)
        fig_bottom = fig_bottoms[rows].min()
        fig_top = fig_tops[rows].max()
        fig_left = fig_lefts[cols].min()
        fig_right = fig_rights[cols].max()
        return Bbox.from_extents(fig_left, fig_bottom, fig_right, fig_top)

    def get_topmost_subplotspec(self):
        if False:
            print('Hello World!')
        '\n        Return the topmost `SubplotSpec` instance associated with the subplot.\n        '
        gridspec = self.get_gridspec()
        if hasattr(gridspec, 'get_topmost_subplotspec'):
            return gridspec.get_topmost_subplotspec()
        else:
            return self

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        '\n        Two SubplotSpecs are considered equal if they refer to the same\n        position(s) in the same `GridSpec`.\n        '
        return (self._gridspec, self.num1, self.num2) == (getattr(other, '_gridspec', object()), getattr(other, 'num1', object()), getattr(other, 'num2', object()))

    def __hash__(self):
        if False:
            print('Hello World!')
        return hash((self._gridspec, self.num1, self.num2))

    def subgridspec(self, nrows, ncols, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Create a GridSpec within this subplot.\n\n        The created `.GridSpecFromSubplotSpec` will have this `SubplotSpec` as\n        a parent.\n\n        Parameters\n        ----------\n        nrows : int\n            Number of rows in grid.\n\n        ncols : int\n            Number of columns in grid.\n\n        Returns\n        -------\n        `.GridSpecFromSubplotSpec`\n\n        Other Parameters\n        ----------------\n        **kwargs\n            All other parameters are passed to `.GridSpecFromSubplotSpec`.\n\n        See Also\n        --------\n        matplotlib.pyplot.subplots\n\n        Examples\n        --------\n        Adding three subplots in the space occupied by a single subplot::\n\n            fig = plt.figure()\n            gs0 = fig.add_gridspec(3, 1)\n            ax1 = fig.add_subplot(gs0[0])\n            ax2 = fig.add_subplot(gs0[1])\n            gssub = gs0[2].subgridspec(1, 3)\n            for i in range(3):\n                fig.add_subplot(gssub[0, i])\n        '
        return GridSpecFromSubplotSpec(nrows, ncols, self, **kwargs)

class SubplotParams:
    """
    Parameters defining the positioning of a subplots grid in a figure.
    """

    def __init__(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        if False:
            return 10
        '\n        Defaults are given by :rc:`figure.subplot.[name]`.\n\n        Parameters\n        ----------\n        left : float\n            The position of the left edge of the subplots,\n            as a fraction of the figure width.\n        right : float\n            The position of the right edge of the subplots,\n            as a fraction of the figure width.\n        bottom : float\n            The position of the bottom edge of the subplots,\n            as a fraction of the figure height.\n        top : float\n            The position of the top edge of the subplots,\n            as a fraction of the figure height.\n        wspace : float\n            The width of the padding between subplots,\n            as a fraction of the average Axes width.\n        hspace : float\n            The height of the padding between subplots,\n            as a fraction of the average Axes height.\n        '
        for key in ['left', 'bottom', 'right', 'top', 'wspace', 'hspace']:
            setattr(self, key, mpl.rcParams[f'figure.subplot.{key}'])
        self.update(left, bottom, right, top, wspace, hspace)

    def update(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        if False:
            while True:
                i = 10
        '\n        Update the dimensions of the passed parameters. *None* means unchanged.\n        '
        if (left if left is not None else self.left) >= (right if right is not None else self.right):
            raise ValueError('left cannot be >= right')
        if (bottom if bottom is not None else self.bottom) >= (top if top is not None else self.top):
            raise ValueError('bottom cannot be >= top')
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if bottom is not None:
            self.bottom = bottom
        if top is not None:
            self.top = top
        if wspace is not None:
            self.wspace = wspace
        if hspace is not None:
            self.hspace = hspace