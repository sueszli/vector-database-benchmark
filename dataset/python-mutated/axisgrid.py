from __future__ import annotations
from itertools import product
from inspect import signature
import warnings
from textwrap import dedent
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from ._base import VectorPlotter, variable_type, categorical_order
from ._core.data import handle_data_source
from ._compat import share_axis, get_legend_handles
from . import utils
from .utils import adjust_legend_subtitles, set_hls_values, _check_argument, _draw_figure, _disable_autolayout
from .palettes import color_palette, blend_palette
from ._docstrings import DocstringComponents, _core_docs
__all__ = ['FacetGrid', 'PairGrid', 'JointGrid', 'pairplot', 'jointplot']
_param_docs = DocstringComponents.from_nested_components(core=_core_docs['params'])

class _BaseGrid:
    """Base class for grids of subplots."""

    def set(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Set attributes on each subplot Axes.'
        for ax in self.axes.flat:
            if ax is not None:
                ax.set(**kwargs)
        return self

    @property
    def fig(self):
        if False:
            return 10
        'DEPRECATED: prefer the `figure` property.'
        return self._figure

    @property
    def figure(self):
        if False:
            return 10
        'Access the :class:`matplotlib.figure.Figure` object underlying the grid.'
        return self._figure

    def apply(self, func, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Pass the grid to a user-supplied function and return self.\n\n        The `func` must accept an object of this type for its first\n        positional argument. Additional arguments are passed through.\n        The return value of `func` is ignored; this method returns self.\n        See the `pipe` method if you want the return value.\n\n        Added in v0.12.0.\n\n        '
        func(self, *args, **kwargs)
        return self

    def pipe(self, func, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Pass the grid to a user-supplied function and return its value.\n\n        The `func` must accept an object of this type for its first\n        positional argument. Additional arguments are passed through.\n        The return value of `func` becomes the return value of this method.\n        See the `apply` method if you want to return self instead.\n\n        Added in v0.12.0.\n\n        '
        return func(self, *args, **kwargs)

    def savefig(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Save an image of the plot.\n\n        This wraps :meth:`matplotlib.figure.Figure.savefig`, using bbox_inches="tight"\n        by default. Parameters are passed through to the matplotlib function.\n\n        '
        kwargs = kwargs.copy()
        kwargs.setdefault('bbox_inches', 'tight')
        self.figure.savefig(*args, **kwargs)

class Grid(_BaseGrid):
    """A grid that can have multiple subplots and an external legend."""
    _margin_titles = False
    _legend_out = True

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._tight_layout_rect = [0, 0, 1, 1]
        self._tight_layout_pad = None
        self._extract_legend_handles = False

    def tight_layout(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Call fig.tight_layout within rect that exclude the legend.'
        kwargs = kwargs.copy()
        kwargs.setdefault('rect', self._tight_layout_rect)
        if self._tight_layout_pad is not None:
            kwargs.setdefault('pad', self._tight_layout_pad)
        self._figure.tight_layout(*args, **kwargs)
        return self

    def add_legend(self, legend_data=None, title=None, label_order=None, adjust_subtitles=False, **kwargs):
        if False:
            return 10
        'Draw a legend, maybe placing it outside axes and resizing the figure.\n\n        Parameters\n        ----------\n        legend_data : dict\n            Dictionary mapping label names (or two-element tuples where the\n            second element is a label name) to matplotlib artist handles. The\n            default reads from ``self._legend_data``.\n        title : string\n            Title for the legend. The default reads from ``self._hue_var``.\n        label_order : list of labels\n            The order that the legend entries should appear in. The default\n            reads from ``self.hue_names``.\n        adjust_subtitles : bool\n            If True, modify entries with invisible artists to left-align\n            the labels and set the font size to that of a title.\n        kwargs : key, value pairings\n            Other keyword arguments are passed to the underlying legend methods\n            on the Figure or Axes object.\n\n        Returns\n        -------\n        self : Grid instance\n            Returns self for easy chaining.\n\n        '
        if legend_data is None:
            legend_data = self._legend_data
        if label_order is None:
            if self.hue_names is None:
                label_order = list(legend_data.keys())
            else:
                label_order = list(map(utils.to_utf8, self.hue_names))
        blank_handle = mpl.patches.Patch(alpha=0, linewidth=0)
        handles = [legend_data.get(lab, blank_handle) for lab in label_order]
        title = self._hue_var if title is None else title
        title_size = mpl.rcParams['legend.title_fontsize']
        labels = []
        for entry in label_order:
            if isinstance(entry, tuple):
                (_, label) = entry
            else:
                label = entry
            labels.append(label)
        kwargs.setdefault('scatterpoints', 1)
        if self._legend_out:
            kwargs.setdefault('frameon', False)
            kwargs.setdefault('loc', 'center right')
            figlegend = self._figure.legend(handles, labels, **kwargs)
            self._legend = figlegend
            figlegend.set_title(title, prop={'size': title_size})
            if adjust_subtitles:
                adjust_legend_subtitles(figlegend)
            _draw_figure(self._figure)
            legend_width = figlegend.get_window_extent().width / self._figure.dpi
            (fig_width, fig_height) = self._figure.get_size_inches()
            self._figure.set_size_inches(fig_width + legend_width, fig_height)
            _draw_figure(self._figure)
            legend_width = figlegend.get_window_extent().width / self._figure.dpi
            space_needed = legend_width / (fig_width + legend_width)
            margin = 0.04 if self._margin_titles else 0.01
            self._space_needed = margin + space_needed
            right = 1 - self._space_needed
            self._figure.subplots_adjust(right=right)
            self._tight_layout_rect[2] = right
        else:
            ax = self.axes.flat[0]
            kwargs.setdefault('loc', 'best')
            leg = ax.legend(handles, labels, **kwargs)
            leg.set_title(title, prop={'size': title_size})
            self._legend = leg
            if adjust_subtitles:
                adjust_legend_subtitles(leg)
        return self

    def _update_legend_data(self, ax):
        if False:
            while True:
                i = 10
        'Extract the legend data from an axes object and save it.'
        data = {}
        if ax.legend_ is not None and self._extract_legend_handles:
            handles = get_legend_handles(ax.legend_)
            labels = [t.get_text() for t in ax.legend_.texts]
            data.update({label: handle for (handle, label) in zip(handles, labels)})
        (handles, labels) = ax.get_legend_handles_labels()
        data.update({label: handle for (handle, label) in zip(handles, labels)})
        self._legend_data.update(data)
        ax.legend_ = None

    def _get_palette(self, data, hue, hue_order, palette):
        if False:
            i = 10
            return i + 15
        'Get a list of colors for the hue variable.'
        if hue is None:
            palette = color_palette(n_colors=1)
        else:
            hue_names = categorical_order(data[hue], hue_order)
            n_colors = len(hue_names)
            if palette is None:
                current_palette = utils.get_color_cycle()
                if n_colors > len(current_palette):
                    colors = color_palette('husl', n_colors)
                else:
                    colors = color_palette(n_colors=n_colors)
            elif isinstance(palette, dict):
                color_names = [palette[h] for h in hue_names]
                colors = color_palette(color_names, n_colors)
            else:
                colors = color_palette(palette, n_colors)
            palette = color_palette(colors, n_colors)
        return palette

    @property
    def legend(self):
        if False:
            print('Hello World!')
        'The :class:`matplotlib.legend.Legend` object, if present.'
        try:
            return self._legend
        except AttributeError:
            return None

    def tick_params(self, axis='both', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "Modify the ticks, tick labels, and gridlines.\n\n        Parameters\n        ----------\n        axis : {'x', 'y', 'both'}\n            The axis on which to apply the formatting.\n        kwargs : keyword arguments\n            Additional keyword arguments to pass to\n            :meth:`matplotlib.axes.Axes.tick_params`.\n\n        Returns\n        -------\n        self : Grid instance\n            Returns self for easy chaining.\n\n        "
        for ax in self.figure.axes:
            ax.tick_params(axis=axis, **kwargs)
        return self
_facet_docs = dict(data=dedent('    data : DataFrame\n        Tidy ("long-form") dataframe where each column is a variable and each\n        row is an observation.    '), rowcol=dedent('    row, col : vectors or keys in ``data``\n        Variables that define subsets to plot on different facets.    '), rowcol_order=dedent('    {row,col}_order : vector of strings\n        Specify the order in which levels of the ``row`` and/or ``col`` variables\n        appear in the grid of subplots.    '), col_wrap=dedent('    col_wrap : int\n        "Wrap" the column variable at this width, so that the column facets\n        span multiple rows. Incompatible with a ``row`` facet.    '), share_xy=dedent("    share{x,y} : bool, 'col', or 'row' optional\n        If true, the facets will share y axes across columns and/or x axes\n        across rows.    "), height=dedent('    height : scalar\n        Height (in inches) of each facet. See also: ``aspect``.    '), aspect=dedent('    aspect : scalar\n        Aspect ratio of each facet, so that ``aspect * height`` gives the width\n        of each facet in inches.    '), palette=dedent('    palette : palette name, list, or dict\n        Colors to use for the different levels of the ``hue`` variable. Should\n        be something that can be interpreted by :func:`color_palette`, or a\n        dictionary mapping hue levels to matplotlib colors.    '), legend_out=dedent('    legend_out : bool\n        If ``True``, the figure size will be extended, and the legend will be\n        drawn outside the plot on the center right.    '), margin_titles=dedent('    margin_titles : bool\n        If ``True``, the titles for the row variable are drawn to the right of\n        the last column. This option is experimental and may not work in all\n        cases.    '), facet_kws=dedent('    facet_kws : dict\n        Additional parameters passed to :class:`FacetGrid`.\n    '))

class FacetGrid(Grid):
    """Multi-plot grid for plotting conditional relationships."""

    def __init__(self, data, *, row=None, col=None, hue=None, col_wrap=None, sharex=True, sharey=True, height=3, aspect=1, palette=None, row_order=None, col_order=None, hue_order=None, hue_kws=None, dropna=False, legend_out=True, despine=True, margin_titles=False, xlim=None, ylim=None, subplot_kws=None, gridspec_kws=None):
        if False:
            return 10
        super().__init__()
        data = handle_data_source(data)
        hue_var = hue
        if hue is None:
            hue_names = None
        else:
            hue_names = categorical_order(data[hue], hue_order)
        colors = self._get_palette(data, hue, hue_order, palette)
        if row is None:
            row_names = []
        else:
            row_names = categorical_order(data[row], row_order)
        if col is None:
            col_names = []
        else:
            col_names = categorical_order(data[col], col_order)
        hue_kws = hue_kws if hue_kws is not None else {}
        none_na = np.zeros(len(data), bool)
        if dropna:
            row_na = none_na if row is None else data[row].isnull()
            col_na = none_na if col is None else data[col].isnull()
            hue_na = none_na if hue is None else data[hue].isnull()
            not_na = ~(row_na | col_na | hue_na)
        else:
            not_na = ~none_na
        ncol = 1 if col is None else len(col_names)
        nrow = 1 if row is None else len(row_names)
        self._n_facets = ncol * nrow
        self._col_wrap = col_wrap
        if col_wrap is not None:
            if row is not None:
                err = 'Cannot use `row` and `col_wrap` together.'
                raise ValueError(err)
            ncol = col_wrap
            nrow = int(np.ceil(len(col_names) / col_wrap))
        self._ncol = ncol
        self._nrow = nrow
        figsize = (ncol * height * aspect, nrow * height)
        if col_wrap is not None:
            margin_titles = False
        subplot_kws = {} if subplot_kws is None else subplot_kws.copy()
        gridspec_kws = {} if gridspec_kws is None else gridspec_kws.copy()
        if xlim is not None:
            subplot_kws['xlim'] = xlim
        if ylim is not None:
            subplot_kws['ylim'] = ylim
        with _disable_autolayout():
            fig = plt.figure(figsize=figsize)
        if col_wrap is None:
            kwargs = dict(squeeze=False, sharex=sharex, sharey=sharey, subplot_kw=subplot_kws, gridspec_kw=gridspec_kws)
            axes = fig.subplots(nrow, ncol, **kwargs)
            if col is None and row is None:
                axes_dict = {}
            elif col is None:
                axes_dict = dict(zip(row_names, axes.flat))
            elif row is None:
                axes_dict = dict(zip(col_names, axes.flat))
            else:
                facet_product = product(row_names, col_names)
                axes_dict = dict(zip(facet_product, axes.flat))
        else:
            if gridspec_kws:
                warnings.warn('`gridspec_kws` ignored when using `col_wrap`')
            n_axes = len(col_names)
            axes = np.empty(n_axes, object)
            axes[0] = fig.add_subplot(nrow, ncol, 1, **subplot_kws)
            if sharex:
                subplot_kws['sharex'] = axes[0]
            if sharey:
                subplot_kws['sharey'] = axes[0]
            for i in range(1, n_axes):
                axes[i] = fig.add_subplot(nrow, ncol, i + 1, **subplot_kws)
            axes_dict = dict(zip(col_names, axes))
        self._figure = fig
        self._axes = axes
        self._axes_dict = axes_dict
        self._legend = None
        self.data = data
        self.row_names = row_names
        self.col_names = col_names
        self.hue_names = hue_names
        self.hue_kws = hue_kws
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col
        self._margin_titles = margin_titles
        self._margin_titles_texts = []
        self._col_wrap = col_wrap
        self._hue_var = hue_var
        self._colors = colors
        self._legend_out = legend_out
        self._legend_data = {}
        self._x_var = None
        self._y_var = None
        self._sharex = sharex
        self._sharey = sharey
        self._dropna = dropna
        self._not_na = not_na
        self.set_titles()
        self.tight_layout()
        if despine:
            self.despine()
        if sharex in [True, 'col']:
            for ax in self._not_bottom_axes:
                for label in ax.get_xticklabels():
                    label.set_visible(False)
                ax.xaxis.offsetText.set_visible(False)
                ax.xaxis.label.set_visible(False)
        if sharey in [True, 'row']:
            for ax in self._not_left_axes:
                for label in ax.get_yticklabels():
                    label.set_visible(False)
                ax.yaxis.offsetText.set_visible(False)
                ax.yaxis.label.set_visible(False)
    __init__.__doc__ = dedent('        Initialize the matplotlib figure and FacetGrid object.\n\n        This class maps a dataset onto multiple axes arrayed in a grid of rows\n        and columns that correspond to *levels* of variables in the dataset.\n        The plots it produces are often called "lattice", "trellis", or\n        "small-multiple" graphics.\n\n        It can also represent levels of a third variable with the ``hue``\n        parameter, which plots different subsets of data in different colors.\n        This uses color to resolve elements on a third dimension, but only\n        draws subsets on top of each other and will not tailor the ``hue``\n        parameter for the specific visualization the way that axes-level\n        functions that accept ``hue`` will.\n\n        The basic workflow is to initialize the :class:`FacetGrid` object with\n        the dataset and the variables that are used to structure the grid. Then\n        one or more plotting functions can be applied to each subset by calling\n        :meth:`FacetGrid.map` or :meth:`FacetGrid.map_dataframe`. Finally, the\n        plot can be tweaked with other methods to do things like change the\n        axis labels, use different ticks, or add a legend. See the detailed\n        code examples below for more information.\n\n        .. warning::\n\n            When using seaborn functions that infer semantic mappings from a\n            dataset, care must be taken to synchronize those mappings across\n            facets (e.g., by defining the ``hue`` mapping with a palette dict or\n            setting the data type of the variables to ``category``). In most cases,\n            it will be better to use a figure-level function (e.g. :func:`relplot`\n            or :func:`catplot`) than to use :class:`FacetGrid` directly.\n\n        See the :ref:`tutorial <grid_tutorial>` for more information.\n\n        Parameters\n        ----------\n        {data}\n        row, col, hue : strings\n            Variables that define subsets of the data, which will be drawn on\n            separate facets in the grid. See the ``{{var}}_order`` parameters to\n            control the order of levels of this variable.\n        {col_wrap}\n        {share_xy}\n        {height}\n        {aspect}\n        {palette}\n        {{row,col,hue}}_order : lists\n            Order for the levels of the faceting variables. By default, this\n            will be the order that the levels appear in ``data`` or, if the\n            variables are pandas categoricals, the category order.\n        hue_kws : dictionary of param -> list of values mapping\n            Other keyword arguments to insert into the plotting call to let\n            other plot attributes vary across levels of the hue variable (e.g.\n            the markers in a scatterplot).\n        {legend_out}\n        despine : boolean\n            Remove the top and right spines from the plots.\n        {margin_titles}\n        {{x, y}}lim: tuples\n            Limits for each of the axes on each facet (only relevant when\n            share{{x, y}} is True).\n        subplot_kws : dict\n            Dictionary of keyword arguments passed to matplotlib subplot(s)\n            methods.\n        gridspec_kws : dict\n            Dictionary of keyword arguments passed to\n            :class:`matplotlib.gridspec.GridSpec`\n            (via :meth:`matplotlib.figure.Figure.subplots`).\n            Ignored if ``col_wrap`` is not ``None``.\n\n        See Also\n        --------\n        PairGrid : Subplot grid for plotting pairwise relationships\n        relplot : Combine a relational plot and a :class:`FacetGrid`\n        displot : Combine a distribution plot and a :class:`FacetGrid`\n        catplot : Combine a categorical plot and a :class:`FacetGrid`\n        lmplot : Combine a regression plot and a :class:`FacetGrid`\n\n        Examples\n        --------\n\n        .. note::\n\n            These examples use seaborn functions to demonstrate some of the\n            advanced features of the class, but in most cases you will want\n            to use figue-level functions (e.g. :func:`displot`, :func:`relplot`)\n            to make the plots shown here.\n\n        .. include:: ../docstrings/FacetGrid.rst\n\n        ').format(**_facet_docs)

    def facet_data(self):
        if False:
            print('Hello World!')
        'Generator for name indices and data subsets for each facet.\n\n        Yields\n        ------\n        (i, j, k), data_ijk : tuple of ints, DataFrame\n            The ints provide an index into the {row, col, hue}_names attribute,\n            and the dataframe contains a subset of the full data corresponding\n            to each facet. The generator yields subsets that correspond with\n            the self.axes.flat iterator, or self.axes[i, j] when `col_wrap`\n            is None.\n\n        '
        data = self.data
        if self.row_names:
            row_masks = [data[self._row_var] == n for n in self.row_names]
        else:
            row_masks = [np.repeat(True, len(self.data))]
        if self.col_names:
            col_masks = [data[self._col_var] == n for n in self.col_names]
        else:
            col_masks = [np.repeat(True, len(self.data))]
        if self.hue_names:
            hue_masks = [data[self._hue_var] == n for n in self.hue_names]
        else:
            hue_masks = [np.repeat(True, len(self.data))]
        for ((i, row), (j, col), (k, hue)) in product(enumerate(row_masks), enumerate(col_masks), enumerate(hue_masks)):
            data_ijk = data[row & col & hue & self._not_na]
            yield ((i, j, k), data_ijk)

    def map(self, func, *args, **kwargs):
        if False:
            print('Hello World!')
        "Apply a plotting function to each facet's subset of the data.\n\n        Parameters\n        ----------\n        func : callable\n            A plotting function that takes data and keyword arguments. It\n            must plot to the currently active matplotlib Axes and take a\n            `color` keyword argument. If faceting on the `hue` dimension,\n            it must also take a `label` keyword argument.\n        args : strings\n            Column names in self.data that identify variables with data to\n            plot. The data for each variable is passed to `func` in the\n            order the variables are specified in the call.\n        kwargs : keyword arguments\n            All keyword arguments are passed to the plotting function.\n\n        Returns\n        -------\n        self : object\n            Returns self.\n\n        "
        kw_color = kwargs.pop('color', None)
        func_module = str(getattr(func, '__module__', ''))
        if func_module == 'seaborn.categorical':
            if 'order' not in kwargs:
                warning = 'Using the {} function without specifying `order` is likely to produce an incorrect plot.'.format(func.__name__)
                warnings.warn(warning)
            if len(args) == 3 and 'hue_order' not in kwargs:
                warning = 'Using the {} function without specifying `hue_order` is likely to produce an incorrect plot.'.format(func.__name__)
                warnings.warn(warning)
        for ((row_i, col_j, hue_k), data_ijk) in self.facet_data():
            if not data_ijk.values.size:
                continue
            modify_state = not func_module.startswith('seaborn')
            ax = self.facet_axis(row_i, col_j, modify_state)
            kwargs['color'] = self._facet_color(hue_k, kw_color)
            for (kw, val_list) in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]
            if self._hue_var is not None:
                kwargs['label'] = utils.to_utf8(self.hue_names[hue_k])
            plot_data = data_ijk[list(args)]
            if self._dropna:
                plot_data = plot_data.dropna()
            plot_args = [v for (k, v) in plot_data.items()]
            if func_module.startswith('matplotlib'):
                plot_args = [v.values for v in plot_args]
            self._facet_plot(func, ax, plot_args, kwargs)
        self._finalize_grid(args[:2])
        return self

    def map_dataframe(self, func, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Like ``.map`` but passes args as strings and inserts data in kwargs.\n\n        This method is suitable for plotting with functions that accept a\n        long-form DataFrame as a `data` keyword argument and access the\n        data in that DataFrame using string variable names.\n\n        Parameters\n        ----------\n        func : callable\n            A plotting function that takes data and keyword arguments. Unlike\n            the `map` method, a function used here must "understand" Pandas\n            objects. It also must plot to the currently active matplotlib Axes\n            and take a `color` keyword argument. If faceting on the `hue`\n            dimension, it must also take a `label` keyword argument.\n        args : strings\n            Column names in self.data that identify variables with data to\n            plot. The data for each variable is passed to `func` in the\n            order the variables are specified in the call.\n        kwargs : keyword arguments\n            All keyword arguments are passed to the plotting function.\n\n        Returns\n        -------\n        self : object\n            Returns self.\n\n        '
        kw_color = kwargs.pop('color', None)
        for ((row_i, col_j, hue_k), data_ijk) in self.facet_data():
            if not data_ijk.values.size:
                continue
            modify_state = not str(func.__module__).startswith('seaborn')
            ax = self.facet_axis(row_i, col_j, modify_state)
            kwargs['color'] = self._facet_color(hue_k, kw_color)
            for (kw, val_list) in self.hue_kws.items():
                kwargs[kw] = val_list[hue_k]
            if self._hue_var is not None:
                kwargs['label'] = self.hue_names[hue_k]
            if self._dropna:
                data_ijk = data_ijk.dropna()
            kwargs['data'] = data_ijk
            self._facet_plot(func, ax, args, kwargs)
        axis_labels = [kwargs.get('x', None), kwargs.get('y', None)]
        for (i, val) in enumerate(args[:2]):
            axis_labels[i] = val
        self._finalize_grid(axis_labels)
        return self

    def _facet_color(self, hue_index, kw_color):
        if False:
            print('Hello World!')
        color = self._colors[hue_index]
        if kw_color is not None:
            return kw_color
        elif color is not None:
            return color

    def _facet_plot(self, func, ax, plot_args, plot_kwargs):
        if False:
            while True:
                i = 10
        if str(func.__module__).startswith('seaborn'):
            plot_kwargs = plot_kwargs.copy()
            semantics = ['x', 'y', 'hue', 'size', 'style']
            for (key, val) in zip(semantics, plot_args):
                plot_kwargs[key] = val
            plot_args = []
            plot_kwargs['ax'] = ax
        func(*plot_args, **plot_kwargs)
        self._update_legend_data(ax)

    def _finalize_grid(self, axlabels):
        if False:
            return 10
        'Finalize the annotations and layout.'
        self.set_axis_labels(*axlabels)
        self.tight_layout()

    def facet_axis(self, row_i, col_j, modify_state=True):
        if False:
            i = 10
            return i + 15
        'Make the axis identified by these indices active and return it.'
        if self._col_wrap is not None:
            ax = self.axes.flat[col_j]
        else:
            ax = self.axes[row_i, col_j]
        if modify_state:
            plt.sca(ax)
        return ax

    def despine(self, **kwargs):
        if False:
            return 10
        'Remove axis spines from the facets.'
        utils.despine(self._figure, **kwargs)
        return self

    def set_axis_labels(self, x_var=None, y_var=None, clear_inner=True, **kwargs):
        if False:
            return 10
        'Set axis labels on the left column and bottom row of the grid.'
        if x_var is not None:
            self._x_var = x_var
            self.set_xlabels(x_var, clear_inner=clear_inner, **kwargs)
        if y_var is not None:
            self._y_var = y_var
            self.set_ylabels(y_var, clear_inner=clear_inner, **kwargs)
        return self

    def set_xlabels(self, label=None, clear_inner=True, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Label the x axis on the bottom row of the grid.'
        if label is None:
            label = self._x_var
        for ax in self._bottom_axes:
            ax.set_xlabel(label, **kwargs)
        if clear_inner:
            for ax in self._not_bottom_axes:
                ax.set_xlabel('')
        return self

    def set_ylabels(self, label=None, clear_inner=True, **kwargs):
        if False:
            print('Hello World!')
        'Label the y axis on the left column of the grid.'
        if label is None:
            label = self._y_var
        for ax in self._left_axes:
            ax.set_ylabel(label, **kwargs)
        if clear_inner:
            for ax in self._not_left_axes:
                ax.set_ylabel('')
        return self

    def set_xticklabels(self, labels=None, step=None, **kwargs):
        if False:
            return 10
        'Set x axis tick labels of the grid.'
        for ax in self.axes.flat:
            curr_ticks = ax.get_xticks()
            ax.set_xticks(curr_ticks)
            if labels is None:
                curr_labels = [label.get_text() for label in ax.get_xticklabels()]
                if step is not None:
                    xticks = ax.get_xticks()[::step]
                    curr_labels = curr_labels[::step]
                    ax.set_xticks(xticks)
                ax.set_xticklabels(curr_labels, **kwargs)
            else:
                ax.set_xticklabels(labels, **kwargs)
        return self

    def set_yticklabels(self, labels=None, **kwargs):
        if False:
            while True:
                i = 10
        'Set y axis tick labels on the left column of the grid.'
        for ax in self.axes.flat:
            curr_ticks = ax.get_yticks()
            ax.set_yticks(curr_ticks)
            if labels is None:
                curr_labels = [label.get_text() for label in ax.get_yticklabels()]
                ax.set_yticklabels(curr_labels, **kwargs)
            else:
                ax.set_yticklabels(labels, **kwargs)
        return self

    def set_titles(self, template=None, row_template=None, col_template=None, **kwargs):
        if False:
            print('Hello World!')
        'Draw titles either above each facet or on the grid margins.\n\n        Parameters\n        ----------\n        template : string\n            Template for all titles with the formatting keys {col_var} and\n            {col_name} (if using a `col` faceting variable) and/or {row_var}\n            and {row_name} (if using a `row` faceting variable).\n        row_template:\n            Template for the row variable when titles are drawn on the grid\n            margins. Must have {row_var} and {row_name} formatting keys.\n        col_template:\n            Template for the column variable when titles are drawn on the grid\n            margins. Must have {col_var} and {col_name} formatting keys.\n\n        Returns\n        -------\n        self: object\n            Returns self.\n\n        '
        args = dict(row_var=self._row_var, col_var=self._col_var)
        kwargs['size'] = kwargs.pop('size', mpl.rcParams['axes.labelsize'])
        if row_template is None:
            row_template = '{row_var} = {row_name}'
        if col_template is None:
            col_template = '{col_var} = {col_name}'
        if template is None:
            if self._row_var is None:
                template = col_template
            elif self._col_var is None:
                template = row_template
            else:
                template = ' | '.join([row_template, col_template])
        row_template = utils.to_utf8(row_template)
        col_template = utils.to_utf8(col_template)
        template = utils.to_utf8(template)
        if self._margin_titles:
            for text in self._margin_titles_texts:
                text.remove()
            self._margin_titles_texts = []
            if self.row_names is not None:
                for (i, row_name) in enumerate(self.row_names):
                    ax = self.axes[i, -1]
                    args.update(dict(row_name=row_name))
                    title = row_template.format(**args)
                    text = ax.annotate(title, xy=(1.02, 0.5), xycoords='axes fraction', rotation=270, ha='left', va='center', **kwargs)
                    self._margin_titles_texts.append(text)
            if self.col_names is not None:
                for (j, col_name) in enumerate(self.col_names):
                    args.update(dict(col_name=col_name))
                    title = col_template.format(**args)
                    self.axes[0, j].set_title(title, **kwargs)
            return self
        if self._row_var is not None and self._col_var is not None:
            for (i, row_name) in enumerate(self.row_names):
                for (j, col_name) in enumerate(self.col_names):
                    args.update(dict(row_name=row_name, col_name=col_name))
                    title = template.format(**args)
                    self.axes[i, j].set_title(title, **kwargs)
        elif self.row_names is not None and len(self.row_names):
            for (i, row_name) in enumerate(self.row_names):
                args.update(dict(row_name=row_name))
                title = template.format(**args)
                self.axes[i, 0].set_title(title, **kwargs)
        elif self.col_names is not None and len(self.col_names):
            for (i, col_name) in enumerate(self.col_names):
                args.update(dict(col_name=col_name))
                title = template.format(**args)
                self.axes.flat[i].set_title(title, **kwargs)
        return self

    def refline(self, *, x=None, y=None, color='.5', linestyle='--', **line_kws):
        if False:
            for i in range(10):
                print('nop')
        'Add a reference line(s) to each facet.\n\n        Parameters\n        ----------\n        x, y : numeric\n            Value(s) to draw the line(s) at.\n        color : :mod:`matplotlib color <matplotlib.colors>`\n            Specifies the color of the reference line(s). Pass ``color=None`` to\n            use ``hue`` mapping.\n        linestyle : str\n            Specifies the style of the reference line(s).\n        line_kws : key, value mappings\n            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`\n            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``\n            is not None.\n\n        Returns\n        -------\n        :class:`FacetGrid` instance\n            Returns ``self`` for easy method chaining.\n\n        '
        line_kws['color'] = color
        line_kws['linestyle'] = linestyle
        if x is not None:
            self.map(plt.axvline, x=x, **line_kws)
        if y is not None:
            self.map(plt.axhline, y=y, **line_kws)
        return self

    @property
    def axes(self):
        if False:
            for i in range(10):
                print('nop')
        'An array of the :class:`matplotlib.axes.Axes` objects in the grid.'
        return self._axes

    @property
    def ax(self):
        if False:
            while True:
                i = 10
        'The :class:`matplotlib.axes.Axes` when no faceting variables are assigned.'
        if self.axes.shape == (1, 1):
            return self.axes[0, 0]
        else:
            err = 'Use the `.axes` attribute when facet variables are assigned.'
            raise AttributeError(err)

    @property
    def axes_dict(self):
        if False:
            print('Hello World!')
        'A mapping of facet names to corresponding :class:`matplotlib.axes.Axes`.\n\n        If only one of ``row`` or ``col`` is assigned, each key is a string\n        representing a level of that variable. If both facet dimensions are\n        assigned, each key is a ``({row_level}, {col_level})`` tuple.\n\n        '
        return self._axes_dict

    @property
    def _inner_axes(self):
        if False:
            i = 10
            return i + 15
        'Return a flat array of the inner axes.'
        if self._col_wrap is None:
            return self.axes[:-1, 1:].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for (i, ax) in enumerate(self.axes):
                append = i % self._ncol and i < self._ncol * (self._nrow - 1) and (i < self._ncol * (self._nrow - 1) - n_empty)
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _left_axes(self):
        if False:
            return 10
        'Return a flat array of the left column of axes.'
        if self._col_wrap is None:
            return self.axes[:, 0].flat
        else:
            axes = []
            for (i, ax) in enumerate(self.axes):
                if not i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _not_left_axes(self):
        if False:
            print('Hello World!')
        "Return a flat array of axes that aren't on the left column."
        if self._col_wrap is None:
            return self.axes[:, 1:].flat
        else:
            axes = []
            for (i, ax) in enumerate(self.axes):
                if i % self._ncol:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _bottom_axes(self):
        if False:
            while True:
                i = 10
        'Return a flat array of the bottom row of axes.'
        if self._col_wrap is None:
            return self.axes[-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for (i, ax) in enumerate(self.axes):
                append = i >= self._ncol * (self._nrow - 1) or i >= self._ncol * (self._nrow - 1) - n_empty
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

    @property
    def _not_bottom_axes(self):
        if False:
            while True:
                i = 10
        "Return a flat array of axes that aren't on the bottom row."
        if self._col_wrap is None:
            return self.axes[:-1, :].flat
        else:
            axes = []
            n_empty = self._nrow * self._ncol - self._n_facets
            for (i, ax) in enumerate(self.axes):
                append = i < self._ncol * (self._nrow - 1) and i < self._ncol * (self._nrow - 1) - n_empty
                if append:
                    axes.append(ax)
            return np.array(axes, object).flat

class PairGrid(Grid):
    """Subplot grid for plotting pairwise relationships in a dataset.

    This object maps each variable in a dataset onto a column and row in a
    grid of multiple axes. Different axes-level plotting functions can be
    used to draw bivariate plots in the upper and lower triangles, and the
    marginal distribution of each variable can be shown on the diagonal.

    Several different common plots can be generated in a single line using
    :func:`pairplot`. Use :class:`PairGrid` when you need more flexibility.

    See the :ref:`tutorial <grid_tutorial>` for more information.

    """

    def __init__(self, data, *, hue=None, vars=None, x_vars=None, y_vars=None, hue_order=None, palette=None, hue_kws=None, corner=False, diag_sharey=True, height=2.5, aspect=1, layout_pad=0.5, despine=True, dropna=False):
        if False:
            for i in range(10):
                print('nop')
        'Initialize the plot figure and PairGrid object.\n\n        Parameters\n        ----------\n        data : DataFrame\n            Tidy (long-form) dataframe where each column is a variable and\n            each row is an observation.\n        hue : string (variable name)\n            Variable in ``data`` to map plot aspects to different colors. This\n            variable will be excluded from the default x and y variables.\n        vars : list of variable names\n            Variables within ``data`` to use, otherwise use every column with\n            a numeric datatype.\n        {x, y}_vars : lists of variable names\n            Variables within ``data`` to use separately for the rows and\n            columns of the figure; i.e. to make a non-square plot.\n        hue_order : list of strings\n            Order for the levels of the hue variable in the palette\n        palette : dict or seaborn color palette\n            Set of colors for mapping the ``hue`` variable. If a dict, keys\n            should be values  in the ``hue`` variable.\n        hue_kws : dictionary of param -> list of values mapping\n            Other keyword arguments to insert into the plotting call to let\n            other plot attributes vary across levels of the hue variable (e.g.\n            the markers in a scatterplot).\n        corner : bool\n            If True, don\'t add axes to the upper (off-diagonal) triangle of the\n            grid, making this a "corner" plot.\n        height : scalar\n            Height (in inches) of each facet.\n        aspect : scalar\n            Aspect * height gives the width (in inches) of each facet.\n        layout_pad : scalar\n            Padding between axes; passed to ``fig.tight_layout``.\n        despine : boolean\n            Remove the top and right spines from the plots.\n        dropna : boolean\n            Drop missing values from the data before plotting.\n\n        See Also\n        --------\n        pairplot : Easily drawing common uses of :class:`PairGrid`.\n        FacetGrid : Subplot grid for plotting conditional relationships.\n\n        Examples\n        --------\n\n        .. include:: ../docstrings/PairGrid.rst\n\n        '
        super().__init__()
        data = handle_data_source(data)
        numeric_cols = self._find_numeric_cols(data)
        if hue in numeric_cols:
            numeric_cols.remove(hue)
        if vars is not None:
            x_vars = list(vars)
            y_vars = list(vars)
        if x_vars is None:
            x_vars = numeric_cols
        if y_vars is None:
            y_vars = numeric_cols
        if np.isscalar(x_vars):
            x_vars = [x_vars]
        if np.isscalar(y_vars):
            y_vars = [y_vars]
        self.x_vars = x_vars = list(x_vars)
        self.y_vars = y_vars = list(y_vars)
        self.square_grid = self.x_vars == self.y_vars
        if not x_vars:
            raise ValueError('No variables found for grid columns.')
        if not y_vars:
            raise ValueError('No variables found for grid rows.')
        figsize = (len(x_vars) * height * aspect, len(y_vars) * height)
        with _disable_autolayout():
            fig = plt.figure(figsize=figsize)
        axes = fig.subplots(len(y_vars), len(x_vars), sharex='col', sharey='row', squeeze=False)
        self._corner = corner
        if corner:
            hide_indices = np.triu_indices_from(axes, 1)
            for (i, j) in zip(*hide_indices):
                axes[i, j].remove()
                axes[i, j] = None
        self._figure = fig
        self.axes = axes
        self.data = data
        self.diag_sharey = diag_sharey
        self.diag_vars = None
        self.diag_axes = None
        self._dropna = dropna
        self._add_axis_labels()
        self._hue_var = hue
        if hue is None:
            self.hue_names = hue_order = ['_nolegend_']
            self.hue_vals = pd.Series(['_nolegend_'] * len(data), index=data.index)
        else:
            hue_names = hue_order = categorical_order(data[hue], hue_order)
            if dropna:
                hue_names = list(filter(pd.notnull, hue_names))
            self.hue_names = hue_names
            self.hue_vals = data[hue]
        self.hue_kws = hue_kws if hue_kws is not None else {}
        self._orig_palette = palette
        self._hue_order = hue_order
        self.palette = self._get_palette(data, hue, hue_order, palette)
        self._legend_data = {}
        for ax in axes[:-1, :].flat:
            if ax is None:
                continue
            for label in ax.get_xticklabels():
                label.set_visible(False)
            ax.xaxis.offsetText.set_visible(False)
            ax.xaxis.label.set_visible(False)
        for ax in axes[:, 1:].flat:
            if ax is None:
                continue
            for label in ax.get_yticklabels():
                label.set_visible(False)
            ax.yaxis.offsetText.set_visible(False)
            ax.yaxis.label.set_visible(False)
        self._tight_layout_rect = [0.01, 0.01, 0.99, 0.99]
        self._tight_layout_pad = layout_pad
        self._despine = despine
        if despine:
            utils.despine(fig=fig)
        self.tight_layout(pad=layout_pad)

    def map(self, func, **kwargs):
        if False:
            return 10
        'Plot with the same function in every subplot.\n\n        Parameters\n        ----------\n        func : callable plotting function\n            Must take x, y arrays as positional arguments and draw onto the\n            "currently active" matplotlib Axes. Also needs to accept kwargs\n            called ``color`` and  ``label``.\n\n        '
        (row_indices, col_indices) = np.indices(self.axes.shape)
        indices = zip(row_indices.flat, col_indices.flat)
        self._map_bivariate(func, indices, **kwargs)
        return self

    def map_lower(self, func, **kwargs):
        if False:
            print('Hello World!')
        'Plot with a bivariate function on the lower diagonal subplots.\n\n        Parameters\n        ----------\n        func : callable plotting function\n            Must take x, y arrays as positional arguments and draw onto the\n            "currently active" matplotlib Axes. Also needs to accept kwargs\n            called ``color`` and  ``label``.\n\n        '
        indices = zip(*np.tril_indices_from(self.axes, -1))
        self._map_bivariate(func, indices, **kwargs)
        return self

    def map_upper(self, func, **kwargs):
        if False:
            i = 10
            return i + 15
        'Plot with a bivariate function on the upper diagonal subplots.\n\n        Parameters\n        ----------\n        func : callable plotting function\n            Must take x, y arrays as positional arguments and draw onto the\n            "currently active" matplotlib Axes. Also needs to accept kwargs\n            called ``color`` and  ``label``.\n\n        '
        indices = zip(*np.triu_indices_from(self.axes, 1))
        self._map_bivariate(func, indices, **kwargs)
        return self

    def map_offdiag(self, func, **kwargs):
        if False:
            i = 10
            return i + 15
        'Plot with a bivariate function on the off-diagonal subplots.\n\n        Parameters\n        ----------\n        func : callable plotting function\n            Must take x, y arrays as positional arguments and draw onto the\n            "currently active" matplotlib Axes. Also needs to accept kwargs\n            called ``color`` and  ``label``.\n\n        '
        if self.square_grid:
            self.map_lower(func, **kwargs)
            if not self._corner:
                self.map_upper(func, **kwargs)
        else:
            indices = []
            for (i, y_var) in enumerate(self.y_vars):
                for (j, x_var) in enumerate(self.x_vars):
                    if x_var != y_var:
                        indices.append((i, j))
            self._map_bivariate(func, indices, **kwargs)
        return self

    def map_diag(self, func, **kwargs):
        if False:
            print('Hello World!')
        'Plot with a univariate function on each diagonal subplot.\n\n        Parameters\n        ----------\n        func : callable plotting function\n            Must take an x array as a positional argument and draw onto the\n            "currently active" matplotlib Axes. Also needs to accept kwargs\n            called ``color`` and  ``label``.\n\n        '
        if self.diag_axes is None:
            diag_vars = []
            diag_axes = []
            for (i, y_var) in enumerate(self.y_vars):
                for (j, x_var) in enumerate(self.x_vars):
                    if x_var == y_var:
                        diag_vars.append(x_var)
                        ax = self.axes[i, j]
                        diag_ax = ax.twinx()
                        diag_ax.set_axis_off()
                        diag_axes.append(diag_ax)
                        if not plt.rcParams.get('ytick.left', True):
                            for tick in ax.yaxis.majorTicks:
                                tick.tick1line.set_visible(False)
                        if self._corner:
                            ax.yaxis.set_visible(False)
                            if self._despine:
                                utils.despine(ax=ax, left=True)
            if self.diag_sharey and diag_axes:
                for ax in diag_axes[1:]:
                    share_axis(diag_axes[0], ax, 'y')
            self.diag_vars = diag_vars
            self.diag_axes = diag_axes
        if 'hue' not in signature(func).parameters:
            return self._map_diag_iter_hue(func, **kwargs)
        for (var, ax) in zip(self.diag_vars, self.diag_axes):
            plot_kwargs = kwargs.copy()
            if str(func.__module__).startswith('seaborn'):
                plot_kwargs['ax'] = ax
            else:
                plt.sca(ax)
            vector = self.data[var]
            if self._hue_var is not None:
                hue = self.data[self._hue_var]
            else:
                hue = None
            if self._dropna:
                not_na = vector.notna()
                if hue is not None:
                    not_na &= hue.notna()
                vector = vector[not_na]
                if hue is not None:
                    hue = hue[not_na]
            plot_kwargs.setdefault('hue', hue)
            plot_kwargs.setdefault('hue_order', self._hue_order)
            plot_kwargs.setdefault('palette', self._orig_palette)
            func(x=vector, **plot_kwargs)
            ax.legend_ = None
        self._add_axis_labels()
        return self

    def _map_diag_iter_hue(self, func, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Put marginal plot on each diagonal axes, iterating over hue.'
        fixed_color = kwargs.pop('color', None)
        for (var, ax) in zip(self.diag_vars, self.diag_axes):
            hue_grouped = self.data[var].groupby(self.hue_vals, observed=True)
            plot_kwargs = kwargs.copy()
            if str(func.__module__).startswith('seaborn'):
                plot_kwargs['ax'] = ax
            else:
                plt.sca(ax)
            for (k, label_k) in enumerate(self._hue_order):
                try:
                    data_k = hue_grouped.get_group(label_k)
                except KeyError:
                    data_k = pd.Series([], dtype=float)
                if fixed_color is None:
                    color = self.palette[k]
                else:
                    color = fixed_color
                if self._dropna:
                    data_k = utils.remove_na(data_k)
                if str(func.__module__).startswith('seaborn'):
                    func(x=data_k, label=label_k, color=color, **plot_kwargs)
                else:
                    func(data_k, label=label_k, color=color, **plot_kwargs)
        self._add_axis_labels()
        return self

    def _map_bivariate(self, func, indices, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Draw a bivariate plot on the indicated axes.'
        from .distributions import histplot, kdeplot
        if func is histplot or func is kdeplot:
            self._extract_legend_handles = True
        kws = kwargs.copy()
        for (i, j) in indices:
            x_var = self.x_vars[j]
            y_var = self.y_vars[i]
            ax = self.axes[i, j]
            if ax is None:
                continue
            self._plot_bivariate(x_var, y_var, ax, func, **kws)
        self._add_axis_labels()
        if 'hue' in signature(func).parameters:
            self.hue_names = list(self._legend_data)

    def _plot_bivariate(self, x_var, y_var, ax, func, **kwargs):
        if False:
            i = 10
            return i + 15
        'Draw a bivariate plot on the specified axes.'
        if 'hue' not in signature(func).parameters:
            self._plot_bivariate_iter_hue(x_var, y_var, ax, func, **kwargs)
            return
        kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            kwargs['ax'] = ax
        else:
            plt.sca(ax)
        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]
        if self._hue_var is not None and self._hue_var not in axes_vars:
            axes_vars.append(self._hue_var)
        data = self.data[axes_vars]
        if self._dropna:
            data = data.dropna()
        x = data[x_var]
        y = data[y_var]
        if self._hue_var is None:
            hue = None
        else:
            hue = data.get(self._hue_var)
        if 'hue' not in kwargs:
            kwargs.update({'hue': hue, 'hue_order': self._hue_order, 'palette': self._orig_palette})
        func(x=x, y=y, **kwargs)
        self._update_legend_data(ax)

    def _plot_bivariate_iter_hue(self, x_var, y_var, ax, func, **kwargs):
        if False:
            while True:
                i = 10
        'Draw a bivariate plot while iterating over hue subsets.'
        kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            kwargs['ax'] = ax
        else:
            plt.sca(ax)
        if x_var == y_var:
            axes_vars = [x_var]
        else:
            axes_vars = [x_var, y_var]
        hue_grouped = self.data.groupby(self.hue_vals, observed=True)
        for (k, label_k) in enumerate(self._hue_order):
            kws = kwargs.copy()
            try:
                data_k = hue_grouped.get_group(label_k)
            except KeyError:
                data_k = pd.DataFrame(columns=axes_vars, dtype=float)
            if self._dropna:
                data_k = data_k[axes_vars].dropna()
            x = data_k[x_var]
            y = data_k[y_var]
            for (kw, val_list) in self.hue_kws.items():
                kws[kw] = val_list[k]
            kws.setdefault('color', self.palette[k])
            if self._hue_var is not None:
                kws['label'] = label_k
            if str(func.__module__).startswith('seaborn'):
                func(x=x, y=y, **kws)
            else:
                func(x, y, **kws)
        self._update_legend_data(ax)

    def _add_axis_labels(self):
        if False:
            for i in range(10):
                print('nop')
        'Add labels to the left and bottom Axes.'
        for (ax, label) in zip(self.axes[-1, :], self.x_vars):
            ax.set_xlabel(label)
        for (ax, label) in zip(self.axes[:, 0], self.y_vars):
            ax.set_ylabel(label)

    def _find_numeric_cols(self, data):
        if False:
            i = 10
            return i + 15
        'Find which variables in a DataFrame are numeric.'
        numeric_cols = []
        for col in data:
            if variable_type(data[col]) == 'numeric':
                numeric_cols.append(col)
        return numeric_cols

class JointGrid(_BaseGrid):
    """Grid for drawing a bivariate plot with marginal univariate plots.

    Many plots can be drawn by using the figure-level interface :func:`jointplot`.
    Use this class directly when you need more flexibility.

    """

    def __init__(self, data=None, *, x=None, y=None, hue=None, height=6, ratio=5, space=0.2, palette=None, hue_order=None, hue_norm=None, dropna=False, xlim=None, ylim=None, marginal_ticks=False):
        if False:
            return 10
        f = plt.figure(figsize=(height, height))
        gs = plt.GridSpec(ratio + 1, ratio + 1)
        ax_joint = f.add_subplot(gs[1:, :-1])
        ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
        ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
        self._figure = f
        self.ax_joint = ax_joint
        self.ax_marg_x = ax_marg_x
        self.ax_marg_y = ax_marg_y
        plt.setp(ax_marg_x.get_xticklabels(), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(), visible=False)
        plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
        plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)
        if not marginal_ticks:
            plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
            plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)
            plt.setp(ax_marg_x.get_yticklabels(), visible=False)
            plt.setp(ax_marg_y.get_xticklabels(), visible=False)
            plt.setp(ax_marg_x.get_yticklabels(minor=True), visible=False)
            plt.setp(ax_marg_y.get_xticklabels(minor=True), visible=False)
            ax_marg_x.yaxis.grid(False)
            ax_marg_y.xaxis.grid(False)
        p = VectorPlotter(data=data, variables=dict(x=x, y=y, hue=hue))
        plot_data = p.plot_data.loc[:, p.plot_data.notna().any()]
        if dropna:
            plot_data = plot_data.dropna()

        def get_var(var):
            if False:
                i = 10
                return i + 15
            vector = plot_data.get(var, None)
            if vector is not None:
                vector = vector.rename(p.variables.get(var, None))
            return vector
        self.x = get_var('x')
        self.y = get_var('y')
        self.hue = get_var('hue')
        for axis in 'xy':
            name = p.variables.get(axis, None)
            if name is not None:
                getattr(ax_joint, f'set_{axis}label')(name)
        if xlim is not None:
            ax_joint.set_xlim(xlim)
        if ylim is not None:
            ax_joint.set_ylim(ylim)
        self._hue_params = dict(palette=palette, hue_order=hue_order, hue_norm=hue_norm)
        utils.despine(f)
        if not marginal_ticks:
            utils.despine(ax=ax_marg_x, left=True)
            utils.despine(ax=ax_marg_y, bottom=True)
        for axes in [ax_marg_x, ax_marg_y]:
            for axis in [axes.xaxis, axes.yaxis]:
                axis.label.set_visible(False)
        f.tight_layout()
        f.subplots_adjust(hspace=space, wspace=space)

    def _inject_kwargs(self, func, kws, params):
        if False:
            i = 10
            return i + 15
        'Add params to kws if they are accepted by func.'
        func_params = signature(func).parameters
        for (key, val) in params.items():
            if key in func_params:
                kws.setdefault(key, val)

    def plot(self, joint_func, marginal_func, **kwargs):
        if False:
            while True:
                i = 10
        'Draw the plot by passing functions for joint and marginal axes.\n\n        This method passes the ``kwargs`` dictionary to both functions. If you\n        need more control, call :meth:`JointGrid.plot_joint` and\n        :meth:`JointGrid.plot_marginals` directly with specific parameters.\n\n        Parameters\n        ----------\n        joint_func, marginal_func : callables\n            Functions to draw the bivariate and univariate plots. See methods\n            referenced above for information about the required characteristics\n            of these functions.\n        kwargs\n            Additional keyword arguments are passed to both functions.\n\n        Returns\n        -------\n        :class:`JointGrid` instance\n            Returns ``self`` for easy method chaining.\n\n        '
        self.plot_marginals(marginal_func, **kwargs)
        self.plot_joint(joint_func, **kwargs)
        return self

    def plot_joint(self, func, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Draw a bivariate plot on the joint axes of the grid.\n\n        Parameters\n        ----------\n        func : plotting callable\n            If a seaborn function, it should accept ``x`` and ``y``. Otherwise,\n            it must accept ``x`` and ``y`` vectors of data as the first two\n            positional arguments, and it must plot on the "current" axes.\n            If ``hue`` was defined in the class constructor, the function must\n            accept ``hue`` as a parameter.\n        kwargs\n            Keyword argument are passed to the plotting function.\n\n        Returns\n        -------\n        :class:`JointGrid` instance\n            Returns ``self`` for easy method chaining.\n\n        '
        kwargs = kwargs.copy()
        if str(func.__module__).startswith('seaborn'):
            kwargs['ax'] = self.ax_joint
        else:
            plt.sca(self.ax_joint)
        if self.hue is not None:
            kwargs['hue'] = self.hue
            self._inject_kwargs(func, kwargs, self._hue_params)
        if str(func.__module__).startswith('seaborn'):
            func(x=self.x, y=self.y, **kwargs)
        else:
            func(self.x, self.y, **kwargs)
        return self

    def plot_marginals(self, func, **kwargs):
        if False:
            while True:
                i = 10
        'Draw univariate plots on each marginal axes.\n\n        Parameters\n        ----------\n        func : plotting callable\n            If a seaborn function, it should  accept ``x`` and ``y`` and plot\n            when only one of them is defined. Otherwise, it must accept a vector\n            of data as the first positional argument and determine its orientation\n            using the ``vertical`` parameter, and it must plot on the "current" axes.\n            If ``hue`` was defined in the class constructor, it must accept ``hue``\n            as a parameter.\n        kwargs\n            Keyword argument are passed to the plotting function.\n\n        Returns\n        -------\n        :class:`JointGrid` instance\n            Returns ``self`` for easy method chaining.\n\n        '
        seaborn_func = str(func.__module__).startswith('seaborn') and (not func.__name__ == 'distplot')
        func_params = signature(func).parameters
        kwargs = kwargs.copy()
        if self.hue is not None:
            kwargs['hue'] = self.hue
            self._inject_kwargs(func, kwargs, self._hue_params)
        if 'legend' in func_params:
            kwargs.setdefault('legend', False)
        if 'orientation' in func_params:
            orient_kw_x = {'orientation': 'vertical'}
            orient_kw_y = {'orientation': 'horizontal'}
        elif 'vertical' in func_params:
            orient_kw_x = {'vertical': False}
            orient_kw_y = {'vertical': True}
        if seaborn_func:
            func(x=self.x, ax=self.ax_marg_x, **kwargs)
        else:
            plt.sca(self.ax_marg_x)
            func(self.x, **orient_kw_x, **kwargs)
        if seaborn_func:
            func(y=self.y, ax=self.ax_marg_y, **kwargs)
        else:
            plt.sca(self.ax_marg_y)
            func(self.y, **orient_kw_y, **kwargs)
        self.ax_marg_x.yaxis.get_label().set_visible(False)
        self.ax_marg_y.xaxis.get_label().set_visible(False)
        return self

    def refline(self, *, x=None, y=None, joint=True, marginal=True, color='.5', linestyle='--', **line_kws):
        if False:
            i = 10
            return i + 15
        'Add a reference line(s) to joint and/or marginal axes.\n\n        Parameters\n        ----------\n        x, y : numeric\n            Value(s) to draw the line(s) at.\n        joint, marginal : bools\n            Whether to add the reference line(s) to the joint/marginal axes.\n        color : :mod:`matplotlib color <matplotlib.colors>`\n            Specifies the color of the reference line(s).\n        linestyle : str\n            Specifies the style of the reference line(s).\n        line_kws : key, value mappings\n            Other keyword arguments are passed to :meth:`matplotlib.axes.Axes.axvline`\n            when ``x`` is not None and :meth:`matplotlib.axes.Axes.axhline` when ``y``\n            is not None.\n\n        Returns\n        -------\n        :class:`JointGrid` instance\n            Returns ``self`` for easy method chaining.\n\n        '
        line_kws['color'] = color
        line_kws['linestyle'] = linestyle
        if x is not None:
            if joint:
                self.ax_joint.axvline(x, **line_kws)
            if marginal:
                self.ax_marg_x.axvline(x, **line_kws)
        if y is not None:
            if joint:
                self.ax_joint.axhline(y, **line_kws)
            if marginal:
                self.ax_marg_y.axhline(y, **line_kws)
        return self

    def set_axis_labels(self, xlabel='', ylabel='', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Set axis labels on the bivariate axes.\n\n        Parameters\n        ----------\n        xlabel, ylabel : strings\n            Label names for the x and y variables.\n        kwargs : key, value mappings\n            Other keyword arguments are passed to the following functions:\n\n            - :meth:`matplotlib.axes.Axes.set_xlabel`\n            - :meth:`matplotlib.axes.Axes.set_ylabel`\n\n        Returns\n        -------\n        :class:`JointGrid` instance\n            Returns ``self`` for easy method chaining.\n\n        '
        self.ax_joint.set_xlabel(xlabel, **kwargs)
        self.ax_joint.set_ylabel(ylabel, **kwargs)
        return self
JointGrid.__init__.__doc__ = 'Set up the grid of subplots and store data internally for easy plotting.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\nheight : number\n    Size of each side of the figure in inches (it will be square).\nratio : number\n    Ratio of joint axes height to marginal axes height.\nspace : number\n    Space between the joint and marginal axes\ndropna : bool\n    If True, remove missing observations before plotting.\n{{x, y}}lim : pairs of numbers\n    Set axis limits to these values before plotting.\nmarginal_ticks : bool\n    If False, suppress ticks on the count/density axis of the marginal plots.\n{params.core.hue}\n    Note: unlike in :class:`FacetGrid` or :class:`PairGrid`, the axes-level\n    functions must support ``hue`` to use it in :class:`JointGrid`.\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\n\nSee Also\n--------\n{seealso.jointplot}\n{seealso.pairgrid}\n{seealso.pairplot}\n\nExamples\n--------\n\n.. include:: ../docstrings/JointGrid.rst\n\n'.format(params=_param_docs, seealso=_core_docs['seealso'])

def pairplot(data, *, hue=None, hue_order=None, palette=None, vars=None, x_vars=None, y_vars=None, kind='scatter', diag_kind='auto', markers=None, height=2.5, aspect=1, corner=False, dropna=False, plot_kws=None, diag_kws=None, grid_kws=None, size=None):
    if False:
        i = 10
        return i + 15
    'Plot pairwise relationships in a dataset.\n\n    By default, this function will create a grid of Axes such that each numeric\n    variable in ``data`` will by shared across the y-axes across a single row and\n    the x-axes across a single column. The diagonal plots are treated\n    differently: a univariate distribution plot is drawn to show the marginal\n    distribution of the data in each column.\n\n    It is also possible to show a subset of variables or plot different\n    variables on the rows and columns.\n\n    This is a high-level interface for :class:`PairGrid` that is intended to\n    make it easy to draw a few common styles. You should use :class:`PairGrid`\n    directly if you need more flexibility.\n\n    Parameters\n    ----------\n    data : `pandas.DataFrame`\n        Tidy (long-form) dataframe where each column is a variable and\n        each row is an observation.\n    hue : name of variable in ``data``\n        Variable in ``data`` to map plot aspects to different colors.\n    hue_order : list of strings\n        Order for the levels of the hue variable in the palette\n    palette : dict or seaborn color palette\n        Set of colors for mapping the ``hue`` variable. If a dict, keys\n        should be values  in the ``hue`` variable.\n    vars : list of variable names\n        Variables within ``data`` to use, otherwise use every column with\n        a numeric datatype.\n    {x, y}_vars : lists of variable names\n        Variables within ``data`` to use separately for the rows and\n        columns of the figure; i.e. to make a non-square plot.\n    kind : {\'scatter\', \'kde\', \'hist\', \'reg\'}\n        Kind of plot to make.\n    diag_kind : {\'auto\', \'hist\', \'kde\', None}\n        Kind of plot for the diagonal subplots. If \'auto\', choose based on\n        whether or not ``hue`` is used.\n    markers : single matplotlib marker code or list\n        Either the marker to use for all scatterplot points or a list of markers\n        with a length the same as the number of levels in the hue variable so that\n        differently colored points will also have different scatterplot\n        markers.\n    height : scalar\n        Height (in inches) of each facet.\n    aspect : scalar\n        Aspect * height gives the width (in inches) of each facet.\n    corner : bool\n        If True, don\'t add axes to the upper (off-diagonal) triangle of the\n        grid, making this a "corner" plot.\n    dropna : boolean\n        Drop missing values from the data before plotting.\n    {plot, diag, grid}_kws : dicts\n        Dictionaries of keyword arguments. ``plot_kws`` are passed to the\n        bivariate plotting function, ``diag_kws`` are passed to the univariate\n        plotting function, and ``grid_kws`` are passed to the :class:`PairGrid`\n        constructor.\n\n    Returns\n    -------\n    grid : :class:`PairGrid`\n        Returns the underlying :class:`PairGrid` instance for further tweaking.\n\n    See Also\n    --------\n    PairGrid : Subplot grid for more flexible plotting of pairwise relationships.\n    JointGrid : Grid for plotting joint and marginal distributions of two variables.\n\n    Examples\n    --------\n\n    .. include:: ../docstrings/pairplot.rst\n\n    '
    from .distributions import histplot, kdeplot
    if size is not None:
        height = size
        msg = 'The `size` parameter has been renamed to `height`; please update your code.'
        warnings.warn(msg, UserWarning)
    if not isinstance(data, pd.DataFrame):
        raise TypeError(f"'data' must be pandas DataFrame object, not: {type(data)}")
    plot_kws = {} if plot_kws is None else plot_kws.copy()
    diag_kws = {} if diag_kws is None else diag_kws.copy()
    grid_kws = {} if grid_kws is None else grid_kws.copy()
    if diag_kind == 'auto':
        if hue is None:
            diag_kind = 'kde' if kind == 'kde' else 'hist'
        else:
            diag_kind = 'hist' if kind == 'hist' else 'kde'
    grid_kws.setdefault('diag_sharey', diag_kind == 'hist')
    grid = PairGrid(data, vars=vars, x_vars=x_vars, y_vars=y_vars, hue=hue, hue_order=hue_order, palette=palette, corner=corner, height=height, aspect=aspect, dropna=dropna, **grid_kws)
    if markers is not None:
        if kind == 'reg':
            if grid.hue_names is None:
                n_markers = 1
            else:
                n_markers = len(grid.hue_names)
            if not isinstance(markers, list):
                markers = [markers] * n_markers
            if len(markers) != n_markers:
                raise ValueError('markers must be a singleton or a list of markers for each level of the hue variable')
            grid.hue_kws = {'marker': markers}
        elif kind == 'scatter':
            if isinstance(markers, str):
                plot_kws['marker'] = markers
            elif hue is not None:
                plot_kws['style'] = data[hue]
                plot_kws['markers'] = markers
    diag_kws = diag_kws.copy()
    diag_kws.setdefault('legend', False)
    if diag_kind == 'hist':
        grid.map_diag(histplot, **diag_kws)
    elif diag_kind == 'kde':
        diag_kws.setdefault('fill', True)
        diag_kws.setdefault('warn_singular', False)
        grid.map_diag(kdeplot, **diag_kws)
    if diag_kind is not None:
        plotter = grid.map_offdiag
    else:
        plotter = grid.map
    if kind == 'scatter':
        from .relational import scatterplot
        plotter(scatterplot, **plot_kws)
    elif kind == 'reg':
        from .regression import regplot
        plotter(regplot, **plot_kws)
    elif kind == 'kde':
        from .distributions import kdeplot
        plot_kws.setdefault('warn_singular', False)
        plotter(kdeplot, **plot_kws)
    elif kind == 'hist':
        from .distributions import histplot
        plotter(histplot, **plot_kws)
    if hue is not None:
        grid.add_legend()
    grid.tight_layout()
    return grid

def jointplot(data=None, *, x=None, y=None, hue=None, kind='scatter', height=6, ratio=5, space=0.2, dropna=False, xlim=None, ylim=None, color=None, palette=None, hue_order=None, hue_norm=None, marginal_ticks=False, joint_kws=None, marginal_kws=None, **kwargs):
    if False:
        while True:
            i = 10
    from .relational import scatterplot
    from .regression import regplot, residplot
    from .distributions import histplot, kdeplot, _freedman_diaconis_bins
    if kwargs.pop('ax', None) is not None:
        msg = 'Ignoring `ax`; jointplot is a figure-level function.'
        warnings.warn(msg, UserWarning, stacklevel=2)
    joint_kws = {} if joint_kws is None else joint_kws.copy()
    joint_kws.update(kwargs)
    marginal_kws = {} if marginal_kws is None else marginal_kws.copy()
    distplot_keys = ['rug', 'fit', 'hist_kws', 'norm_histhist_kws', 'rug_kws']
    unused_keys = []
    for key in distplot_keys:
        if key in marginal_kws:
            unused_keys.append(key)
            marginal_kws.pop(key)
    if unused_keys and kind != 'kde':
        msg = 'The marginal plotting function has changed to `histplot`, which does not accept the following argument(s): {}.'.format(', '.join(unused_keys))
        warnings.warn(msg, UserWarning)
    plot_kinds = ['scatter', 'hist', 'hex', 'kde', 'reg', 'resid']
    _check_argument('kind', plot_kinds, kind)
    if hue is not None and kind in ['hex', 'reg', 'resid']:
        msg = f"Use of `hue` with `kind='{kind}'` is not currently supported."
        raise ValueError(msg)
    if color is None:
        color = 'C0'
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [set_hls_values(color_rgb, l=val) for val in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)
    if kind == 'hex':
        dropna = True
    grid = JointGrid(data=data, x=x, y=y, hue=hue, palette=palette, hue_order=hue_order, hue_norm=hue_norm, dropna=dropna, height=height, ratio=ratio, space=space, xlim=xlim, ylim=ylim, marginal_ticks=marginal_ticks)
    if grid.hue is not None:
        marginal_kws.setdefault('legend', False)
    if kind.startswith('scatter'):
        joint_kws.setdefault('color', color)
        grid.plot_joint(scatterplot, **joint_kws)
        if grid.hue is None:
            marg_func = histplot
        else:
            marg_func = kdeplot
            marginal_kws.setdefault('warn_singular', False)
            marginal_kws.setdefault('fill', True)
        marginal_kws.setdefault('color', color)
        grid.plot_marginals(marg_func, **marginal_kws)
    elif kind.startswith('hist'):
        joint_kws.setdefault('color', color)
        grid.plot_joint(histplot, **joint_kws)
        marginal_kws.setdefault('kde', False)
        marginal_kws.setdefault('color', color)
        marg_x_kws = marginal_kws.copy()
        marg_y_kws = marginal_kws.copy()
        pair_keys = ('bins', 'binwidth', 'binrange')
        for key in pair_keys:
            if isinstance(joint_kws.get(key), tuple):
                (x_val, y_val) = joint_kws[key]
                marg_x_kws.setdefault(key, x_val)
                marg_y_kws.setdefault(key, y_val)
        histplot(data=data, x=x, hue=hue, **marg_x_kws, ax=grid.ax_marg_x)
        histplot(data=data, y=y, hue=hue, **marg_y_kws, ax=grid.ax_marg_y)
    elif kind.startswith('kde'):
        joint_kws.setdefault('color', color)
        joint_kws.setdefault('warn_singular', False)
        grid.plot_joint(kdeplot, **joint_kws)
        marginal_kws.setdefault('color', color)
        if 'fill' in joint_kws:
            marginal_kws.setdefault('fill', joint_kws['fill'])
        grid.plot_marginals(kdeplot, **marginal_kws)
    elif kind.startswith('hex'):
        x_bins = min(_freedman_diaconis_bins(grid.x), 50)
        y_bins = min(_freedman_diaconis_bins(grid.y), 50)
        gridsize = int(np.mean([x_bins, y_bins]))
        joint_kws.setdefault('gridsize', gridsize)
        joint_kws.setdefault('cmap', cmap)
        grid.plot_joint(plt.hexbin, **joint_kws)
        marginal_kws.setdefault('kde', False)
        marginal_kws.setdefault('color', color)
        grid.plot_marginals(histplot, **marginal_kws)
    elif kind.startswith('reg'):
        marginal_kws.setdefault('color', color)
        marginal_kws.setdefault('kde', True)
        grid.plot_marginals(histplot, **marginal_kws)
        joint_kws.setdefault('color', color)
        grid.plot_joint(regplot, **joint_kws)
    elif kind.startswith('resid'):
        joint_kws.setdefault('color', color)
        grid.plot_joint(residplot, **joint_kws)
        (x, y) = grid.ax_joint.collections[0].get_offsets().T
        marginal_kws.setdefault('color', color)
        histplot(x=x, hue=hue, ax=grid.ax_marg_x, **marginal_kws)
        histplot(y=y, hue=hue, ax=grid.ax_marg_y, **marginal_kws)
    plt.sca(grid.ax_joint)
    return grid
jointplot.__doc__ = 'Draw a plot of two variables with bivariate and univariate graphs.\n\nThis function provides a convenient interface to the :class:`JointGrid`\nclass, with several canned plot kinds. This is intended to be a fairly\nlightweight wrapper; if you need more flexibility, you should use\n:class:`JointGrid` directly.\n\nParameters\n----------\n{params.core.data}\n{params.core.xy}\n{params.core.hue}\nkind : {{ "scatter" | "kde" | "hist" | "hex" | "reg" | "resid" }}\n    Kind of plot to draw. See the examples for references to the underlying functions.\nheight : numeric\n    Size of the figure (it will be square).\nratio : numeric\n    Ratio of joint axes height to marginal axes height.\nspace : numeric\n    Space between the joint and marginal axes\ndropna : bool\n    If True, remove observations that are missing from ``x`` and ``y``.\n{{x, y}}lim : pairs of numbers\n    Axis limits to set before plotting.\n{params.core.color}\n{params.core.palette}\n{params.core.hue_order}\n{params.core.hue_norm}\nmarginal_ticks : bool\n    If False, suppress ticks on the count/density axis of the marginal plots.\n{{joint, marginal}}_kws : dicts\n    Additional keyword arguments for the plot components.\nkwargs\n    Additional keyword arguments are passed to the function used to\n    draw the plot on the joint Axes, superseding items in the\n    ``joint_kws`` dictionary.\n\nReturns\n-------\n{returns.jointgrid}\n\nSee Also\n--------\n{seealso.jointgrid}\n{seealso.pairgrid}\n{seealso.pairplot}\n\nExamples\n--------\n\n.. include:: ../docstrings/jointplot.rst\n\n'.format(params=_param_docs, returns=_core_docs['returns'], seealso=_core_docs['seealso'])