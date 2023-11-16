from __future__ import annotations
import functools
import itertools
import warnings
from collections.abc import Hashable, Iterable, MutableMapping
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, TypeVar, cast
import numpy as np
from xarray.core.formatting import format_item
from xarray.core.types import HueStyleOptions, T_DataArrayOrSet
from xarray.plot.utils import _LINEWIDTH_RANGE, _MARKERSIZE_RANGE, _add_legend, _determine_guide, _get_nice_quiver_magnitude, _guess_coords_to_plot, _infer_xy_labels, _Normalize, _parse_size, _process_cmap_cbar_kwargs, label_from_attrs
if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.cm import ScalarMappable
    from matplotlib.colorbar import Colorbar
    from matplotlib.figure import Figure
    from matplotlib.legend import Legend
    from matplotlib.quiver import QuiverKey
    from matplotlib.text import Annotation
    from xarray.core.dataarray import DataArray
_FONTSIZE = 'small'
_NTICKS = 5

def _nicetitle(coord, value, maxchar, template):
    if False:
        return 10
    '\n    Put coord, value in template and truncate at maxchar\n    '
    prettyvalue = format_item(value, quote_strings=False)
    title = template.format(coord=coord, value=prettyvalue)
    if len(title) > maxchar:
        title = title[:maxchar - 3] + '...'
    return title
T_FacetGrid = TypeVar('T_FacetGrid', bound='FacetGrid')

class FacetGrid(Generic[T_DataArrayOrSet]):
    """
    Initialize the Matplotlib figure and FacetGrid object.

    The :class:`FacetGrid` is an object that links a xarray DataArray to
    a Matplotlib figure with a particular structure.

    In particular, :class:`FacetGrid` is used to draw plots with multiple
    axes, where each axes shows the same relationship conditioned on
    different levels of some dimension. It's possible to condition on up to
    two variables by assigning variables to the rows and columns of the
    grid.

    The general approach to plotting here is called "small multiples",
    where the same kind of plot is repeated multiple times, and the
    specific use of small multiples to display the same relationship
    conditioned on one or more other variables is often called a "trellis
    plot".

    The basic workflow is to initialize the :class:`FacetGrid` object with
    the DataArray and the variable names that are used to structure the grid.
    Then plotting functions can be applied to each subset by calling
    :meth:`FacetGrid.map_dataarray` or :meth:`FacetGrid.map`.

    Attributes
    ----------
    axs : ndarray of matplotlib.axes.Axes
        Array containing axes in corresponding position, as returned from
        :py:func:`matplotlib.pyplot.subplots`.
    col_labels : list of matplotlib.text.Annotation
        Column titles.
    row_labels : list of matplotlib.text.Annotation
        Row titles.
    fig : matplotlib.figure.Figure
        The figure containing all the axes.
    name_dicts : ndarray of dict
        Array containing dictionaries mapping coordinate names to values. ``None`` is
        used as a sentinel value for axes that should remain empty, i.e.,
        sometimes the rightmost grid positions in the bottom row.
    """
    data: T_DataArrayOrSet
    name_dicts: np.ndarray
    fig: Figure
    axs: np.ndarray
    row_names: list[np.ndarray]
    col_names: list[np.ndarray]
    figlegend: Legend | None
    quiverkey: QuiverKey | None
    cbar: Colorbar | None
    _single_group: bool | Hashable
    _nrow: int
    _row_var: Hashable | None
    _ncol: int
    _col_var: Hashable | None
    _col_wrap: int | None
    row_labels: list[Annotation | None]
    col_labels: list[Annotation | None]
    _x_var: None
    _y_var: None
    _cmap_extend: Any | None
    _mappables: list[ScalarMappable]
    _finalized: bool

    def __init__(self, data: T_DataArrayOrSet, col: Hashable | None=None, row: Hashable | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, figsize: Iterable[float] | None=None, aspect: float=1, size: float=3, subplot_kws: dict[str, Any] | None=None) -> None:
        if False:
            return 10
        '\n        Parameters\n        ----------\n        data : DataArray or Dataset\n            DataArray or Dataset to be plotted.\n        row, col : str\n            Dimension names that define subsets of the data, which will be drawn\n            on separate facets in the grid.\n        col_wrap : int, optional\n            "Wrap" the grid the for the column variable after this number of columns,\n            adding rows if ``col_wrap`` is less than the number of facets.\n        sharex : bool, optional\n            If true, the facets will share *x* axes.\n        sharey : bool, optional\n            If true, the facets will share *y* axes.\n        figsize : Iterable of float or None, optional\n            A tuple (width, height) of the figure in inches.\n            If set, overrides ``size`` and ``aspect``.\n        aspect : scalar, default: 1\n            Aspect ratio of each facet, so that ``aspect * size`` gives the\n            width of each facet in inches.\n        size : scalar, default: 3\n            Height (in inches) of each facet. See also: ``aspect``.\n        subplot_kws : dict, optional\n            Dictionary of keyword arguments for Matplotlib subplots\n            (:py:func:`matplotlib.pyplot.subplots`).\n\n        '
        import matplotlib.pyplot as plt
        rep_col = col is not None and (not data[col].to_index().is_unique)
        rep_row = row is not None and (not data[row].to_index().is_unique)
        if rep_col or rep_row:
            raise ValueError('Coordinates used for faceting cannot contain repeated (nonunique) values.')
        single_group: bool | Hashable
        if col and row:
            single_group = False
            nrow = len(data[row])
            ncol = len(data[col])
            nfacet = nrow * ncol
            if col_wrap is not None:
                warnings.warn('Ignoring col_wrap since both col and row were passed')
        elif row and (not col):
            single_group = row
        elif not row and col:
            single_group = col
        else:
            raise ValueError('Pass a coordinate name as an argument for row or col')
        if single_group:
            nfacet = len(data[single_group])
            if col:
                ncol = nfacet
            if row:
                ncol = 1
            if col_wrap is not None:
                ncol = col_wrap
            nrow = int(np.ceil(nfacet / ncol))
        subplot_kws = {} if subplot_kws is None else subplot_kws
        if figsize is None:
            cbar_space = 1
            figsize = (ncol * size * aspect + cbar_space, nrow * size)
        (fig, axs) = plt.subplots(nrow, ncol, sharex=sharex, sharey=sharey, squeeze=False, figsize=figsize, subplot_kw=subplot_kws)
        col_names = list(data[col].to_numpy()) if col else []
        row_names = list(data[row].to_numpy()) if row else []
        if single_group:
            full: list[dict[Hashable, Any] | None] = [{single_group: x} for x in data[single_group].to_numpy()]
            empty: list[dict[Hashable, Any] | None] = [None for x in range(nrow * ncol - len(full))]
            name_dict_list = full + empty
        else:
            rowcols = itertools.product(row_names, col_names)
            name_dict_list = [{row: r, col: c} for (r, c) in rowcols]
        name_dicts = np.array(name_dict_list).reshape(nrow, ncol)
        self.data = data
        self.name_dicts = name_dicts
        self.fig = fig
        self.axs = axs
        self.row_names = row_names
        self.col_names = col_names
        self.figlegend = None
        self.quiverkey = None
        self.cbar = None
        self._single_group = single_group
        self._nrow = nrow
        self._row_var = row
        self._ncol = ncol
        self._col_var = col
        self._col_wrap = col_wrap
        self.row_labels = [None] * nrow
        self.col_labels = [None] * ncol
        self._x_var = None
        self._y_var = None
        self._cmap_extend = None
        self._mappables = []
        self._finalized = False

    @property
    def axes(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        warnings.warn('self.axes is deprecated since 2022.11 in order to align with matplotlibs plt.subplots, use self.axs instead.', DeprecationWarning, stacklevel=2)
        return self.axs

    @axes.setter
    def axes(self, axs: np.ndarray) -> None:
        if False:
            return 10
        warnings.warn('self.axes is deprecated since 2022.11 in order to align with matplotlibs plt.subplots, use self.axs instead.', DeprecationWarning, stacklevel=2)
        self.axs = axs

    @property
    def _left_axes(self) -> np.ndarray:
        if False:
            print('Hello World!')
        return self.axs[:, 0]

    @property
    def _bottom_axes(self) -> np.ndarray:
        if False:
            while True:
                i = 10
        return self.axs[-1, :]

    def map_dataarray(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, **kwargs: Any) -> T_FacetGrid:
        if False:
            while True:
                i = 10
        "\n        Apply a plotting function to a 2d facet's subset of the data.\n\n        This is more convenient and less general than ``FacetGrid.map``\n\n        Parameters\n        ----------\n        func : callable\n            A plotting function with the same signature as a 2d xarray\n            plotting method such as `xarray.plot.imshow`\n        x, y : string\n            Names of the coordinates to plot on x, y axes\n        **kwargs\n            additional keyword arguments to func\n\n        Returns\n        -------\n        self : FacetGrid object\n\n        "
        if kwargs.get('cbar_ax', None) is not None:
            raise ValueError('cbar_ax not supported by FacetGrid.')
        (cmap_params, cbar_kwargs) = _process_cmap_cbar_kwargs(func, self.data.to_numpy(), **kwargs)
        self._cmap_extend = cmap_params.get('extend')
        func_kwargs = {k: v for (k, v) in kwargs.items() if k not in {'cmap', 'colors', 'cbar_kwargs', 'levels'}}
        func_kwargs.update(cmap_params)
        func_kwargs['add_colorbar'] = False
        if func.__name__ != 'surface':
            func_kwargs['add_labels'] = False
        (x, y) = _infer_xy_labels(darray=self.data.loc[self.name_dicts.flat[0]], x=x, y=y, imshow=func.__name__ == 'imshow', rgb=kwargs.get('rgb', None))
        for (d, ax) in zip(self.name_dicts.flat, self.axs.flat):
            if d is not None:
                subset = self.data.loc[d]
                mappable = func(subset, x=x, y=y, ax=ax, **func_kwargs, _is_facetgrid=True)
                self._mappables.append(mappable)
        self._finalize_grid(x, y)
        if kwargs.get('add_colorbar', True):
            self.add_colorbar(**cbar_kwargs)
        return self

    def map_plot1d(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, *, z: Hashable | None=None, hue: Hashable | None=None, markersize: Hashable | None=None, linewidth: Hashable | None=None, **kwargs: Any) -> T_FacetGrid:
        if False:
            print('Hello World!')
        "\n        Apply a plotting function to a 1d facet's subset of the data.\n\n        This is more convenient and less general than ``FacetGrid.map``\n\n        Parameters\n        ----------\n        func :\n            A plotting function with the same signature as a 1d xarray\n            plotting method such as `xarray.plot.scatter`\n        x, y :\n            Names of the coordinates to plot on x, y axes\n        **kwargs\n            additional keyword arguments to func\n\n        Returns\n        -------\n        self : FacetGrid object\n\n        "
        self.data = self.data.copy()
        if kwargs.get('cbar_ax', None) is not None:
            raise ValueError('cbar_ax not supported by FacetGrid.')
        if func.__name__ == 'scatter':
            size_ = kwargs.pop('_size', markersize)
            size_r = _MARKERSIZE_RANGE
        else:
            size_ = kwargs.pop('_size', linewidth)
            size_r = _LINEWIDTH_RANGE
        coords_to_plot: MutableMapping[str, Hashable | None] = dict(x=x, z=z, hue=hue, size=size_)
        coords_to_plot = _guess_coords_to_plot(self.data, coords_to_plot, kwargs)
        hue = coords_to_plot['hue']
        hueplt = self.data.coords[hue] if hue else None
        hueplt_norm = _Normalize(hueplt)
        self._hue_var = hueplt
        cbar_kwargs = kwargs.pop('cbar_kwargs', {})
        if hueplt_norm.data is not None:
            if not hueplt_norm.data_is_numeric:
                cbar_kwargs.update(format=hueplt_norm.format, ticks=hueplt_norm.ticks)
                kwargs.update(levels=hueplt_norm.levels)
            (cmap_params, cbar_kwargs) = _process_cmap_cbar_kwargs(func, cast('DataArray', hueplt_norm.values).data, cbar_kwargs=cbar_kwargs, **kwargs)
            self._cmap_extend = cmap_params.get('extend')
        else:
            cmap_params = {}
        size_ = coords_to_plot['size']
        sizeplt = self.data.coords[size_] if size_ else None
        sizeplt_norm = _Normalize(data=sizeplt, width=size_r)
        if sizeplt_norm.data is not None:
            self.data[size_] = sizeplt_norm.values
        func_kwargs = {k: v for (k, v) in kwargs.items() if k not in {'cmap', 'colors', 'cbar_kwargs', 'levels'}}
        func_kwargs.update(cmap_params)
        func_kwargs['add_colorbar'] = False
        func_kwargs['add_legend'] = False
        func_kwargs['add_title'] = False
        add_labels_ = np.zeros(self.axs.shape + (3,), dtype=bool)
        if kwargs.get('z') is not None:
            add_labels_[:] = True
        else:
            add_labels_[-1, :, 0] = True
            add_labels_[:, 0, 1] = True
        if self._single_group:
            full = tuple(({self._single_group: x} for x in range(0, self.data[self._single_group].size)))
            empty = tuple((None for x in range(self._nrow * self._ncol - len(full))))
            name_d = full + empty
        else:
            rowcols = itertools.product(range(0, self.data[self._row_var].size), range(0, self.data[self._col_var].size))
            name_d = tuple(({self._row_var: r, self._col_var: c} for (r, c) in rowcols))
        name_dicts = np.array(name_d).reshape(self._nrow, self._ncol)
        for (add_lbls, d, ax) in zip(add_labels_.reshape((self.axs.size, -1)), name_dicts.flat, self.axs.flat):
            func_kwargs['add_labels'] = add_lbls
            if d is not None:
                subset = self.data.isel(d)
                mappable = func(subset, x=x, y=y, ax=ax, hue=hue, _size=size_, **func_kwargs, _is_facetgrid=True)
                self._mappables.append(mappable)
        self._finalize_grid()
        self._set_lims()
        (add_colorbar, add_legend) = _determine_guide(hueplt_norm, sizeplt_norm, kwargs.get('add_colorbar', None), kwargs.get('add_legend', None))
        if add_legend:
            use_legend_elements = False if func.__name__ == 'hist' else True
            if use_legend_elements:
                self.add_legend(use_legend_elements=use_legend_elements, hueplt_norm=hueplt_norm if not add_colorbar else _Normalize(None), sizeplt_norm=sizeplt_norm, primitive=self._mappables, legend_ax=self.fig, plotfunc=func.__name__)
            else:
                self.add_legend(use_legend_elements=use_legend_elements)
        if add_colorbar:
            if 'label' not in cbar_kwargs:
                cbar_kwargs['label'] = label_from_attrs(hueplt_norm.data)
            self.add_colorbar(**cbar_kwargs)
        return self

    def map_dataarray_line(self: T_FacetGrid, func: Callable, x: Hashable | None, y: Hashable | None, hue: Hashable | None, add_legend: bool=True, _labels=None, **kwargs: Any) -> T_FacetGrid:
        if False:
            print('Hello World!')
        from xarray.plot.dataarray_plot import _infer_line_data
        for (d, ax) in zip(self.name_dicts.flat, self.axs.flat):
            if d is not None:
                subset = self.data.loc[d]
                mappable = func(subset, x=x, y=y, ax=ax, hue=hue, add_legend=False, _labels=False, **kwargs)
                self._mappables.append(mappable)
        (xplt, yplt, hueplt, huelabel) = _infer_line_data(darray=self.data.loc[self.name_dicts.flat[0]], x=x, y=y, hue=hue)
        xlabel = label_from_attrs(xplt)
        ylabel = label_from_attrs(yplt)
        self._hue_var = hueplt
        self._finalize_grid(xlabel, ylabel)
        if add_legend and hueplt is not None and (huelabel is not None):
            self.add_legend(label=huelabel)
        return self

    def map_dataset(self: T_FacetGrid, func: Callable, x: Hashable | None=None, y: Hashable | None=None, hue: Hashable | None=None, hue_style: HueStyleOptions=None, add_guide: bool | None=None, **kwargs: Any) -> T_FacetGrid:
        if False:
            return 10
        from xarray.plot.dataset_plot import _infer_meta_data
        kwargs['add_guide'] = False
        if kwargs.get('markersize', None):
            kwargs['size_mapping'] = _parse_size(self.data[kwargs['markersize']], kwargs.pop('size_norm', None))
        meta_data = _infer_meta_data(self.data, x, y, hue, hue_style, add_guide, funcname=func.__name__)
        kwargs['meta_data'] = meta_data
        if hue and meta_data['hue_style'] == 'continuous':
            (cmap_params, cbar_kwargs) = _process_cmap_cbar_kwargs(func, self.data[hue].to_numpy(), **kwargs)
            kwargs['meta_data']['cmap_params'] = cmap_params
            kwargs['meta_data']['cbar_kwargs'] = cbar_kwargs
        kwargs['_is_facetgrid'] = True
        if func.__name__ == 'quiver' and 'scale' not in kwargs:
            raise ValueError('Please provide scale.')
        for (d, ax) in zip(self.name_dicts.flat, self.axs.flat):
            if d is not None:
                subset = self.data.loc[d]
                maybe_mappable = func(ds=subset, x=x, y=y, hue=hue, hue_style=hue_style, ax=ax, **kwargs)
                self._mappables.append(maybe_mappable)
        self._finalize_grid(meta_data['xlabel'], meta_data['ylabel'])
        if hue:
            hue_label = meta_data.pop('hue_label', None)
            self._hue_label = hue_label
            if meta_data['add_legend']:
                self._hue_var = meta_data['hue']
                self.add_legend(label=hue_label)
            elif meta_data['add_colorbar']:
                self.add_colorbar(label=hue_label, **cbar_kwargs)
        if meta_data['add_quiverkey']:
            self.add_quiverkey(kwargs['u'], kwargs['v'])
        return self

    def _finalize_grid(self, *axlabels: Hashable) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Finalize the annotations and layout.'
        if not self._finalized:
            self.set_axis_labels(*axlabels)
            self.set_titles()
            self.fig.tight_layout()
            for (ax, namedict) in zip(self.axs.flat, self.name_dicts.flat):
                if namedict is None:
                    ax.set_visible(False)
            self._finalized = True

    def _adjust_fig_for_guide(self, guide) -> None:
        if False:
            while True:
                i = 10
        if hasattr(self.fig.canvas, 'get_renderer'):
            renderer = self.fig.canvas.get_renderer()
        else:
            raise RuntimeError('MPL backend has no renderer')
        self.fig.draw(renderer)
        guide_width = guide.get_window_extent(renderer).width / self.fig.dpi
        figure_width = self.fig.get_figwidth()
        total_width = figure_width + guide_width
        self.fig.set_figwidth(total_width)
        self.fig.draw(renderer)
        guide_width = guide.get_window_extent(renderer).width / self.fig.dpi
        space_needed = guide_width / total_width + 0.02
        right = 1 - space_needed
        self.fig.subplots_adjust(right=right)

    def add_legend(self, *, label: str | None=None, use_legend_elements: bool=False, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        if use_legend_elements:
            self.figlegend = _add_legend(**kwargs)
        else:
            self.figlegend = self.fig.legend(handles=self._mappables[-1], labels=list(self._hue_var.to_numpy()), title=label if label is not None else label_from_attrs(self._hue_var), loc=kwargs.pop('loc', 'center right'), **kwargs)
        self._adjust_fig_for_guide(self.figlegend)

    def add_colorbar(self, **kwargs: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Draw a colorbar.'
        kwargs = kwargs.copy()
        if self._cmap_extend is not None:
            kwargs.setdefault('extend', self._cmap_extend)
        if hasattr(self._mappables[-1], 'extend'):
            kwargs.pop('extend', None)
        if 'label' not in kwargs:
            from xarray import DataArray
            assert isinstance(self.data, DataArray)
            kwargs.setdefault('label', label_from_attrs(self.data))
        self.cbar = self.fig.colorbar(self._mappables[-1], ax=list(self.axs.flat), **kwargs)

    def add_quiverkey(self, u: Hashable, v: Hashable, **kwargs: Any) -> None:
        if False:
            return 10
        kwargs = kwargs.copy()
        magnitude = _get_nice_quiver_magnitude(self.data[u], self.data[v])
        units = self.data[u].attrs.get('units', '')
        self.quiverkey = self.axs.flat[-1].quiverkey(self._mappables[-1], X=0.8, Y=0.9, U=magnitude, label=f'{magnitude}\n{units}', labelpos='E', coordinates='figure')

    def _get_largest_lims(self) -> dict[str, tuple[float, float]]:
        if False:
            print('Hello World!')
        '\n        Get largest limits in the facetgrid.\n\n        Returns\n        -------\n        lims_largest : dict[str, tuple[float, float]]\n            Dictionary with the largest limits along each axis.\n\n        Examples\n        --------\n        >>> ds = xr.tutorial.scatter_example_dataset(seed=42)\n        >>> fg = ds.plot.scatter(x="A", y="B", hue="y", row="x", col="w")\n        >>> round(fg._get_largest_lims()["x"][0], 3)\n        -0.334\n        '
        lims_largest: dict[str, tuple[float, float]] = dict(x=(np.inf, -np.inf), y=(np.inf, -np.inf), z=(np.inf, -np.inf))
        for axis in ('x', 'y', 'z'):
            (lower, upper) = lims_largest[axis]
            for ax in self.axs.flat:
                get_lim: None | Callable[[], tuple[float, float]] = getattr(ax, f'get_{axis}lim', None)
                if get_lim:
                    (lower_new, upper_new) = get_lim()
                    (lower, upper) = (min(lower, lower_new), max(upper, upper_new))
            lims_largest[axis] = (lower, upper)
        return lims_largest

    def _set_lims(self, x: tuple[float, float] | None=None, y: tuple[float, float] | None=None, z: tuple[float, float] | None=None) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Set the same limits for all the subplots in the facetgrid.\n\n        Parameters\n        ----------\n        x : tuple[float, float] or None, optional\n            x axis limits.\n        y : tuple[float, float] or None, optional\n            y axis limits.\n        z : tuple[float, float] or None, optional\n            z axis limits.\n\n        Examples\n        --------\n        >>> ds = xr.tutorial.scatter_example_dataset(seed=42)\n        >>> fg = ds.plot.scatter(x="A", y="B", hue="y", row="x", col="w")\n        >>> fg._set_lims(x=(-0.3, 0.3), y=(0, 2), z=(0, 4))\n        >>> fg.axs[0, 0].get_xlim(), fg.axs[0, 0].get_ylim()\n        ((-0.3, 0.3), (0.0, 2.0))\n        '
        lims_largest = self._get_largest_lims()
        for ax in self.axs.flat:
            for ((axis, data_limit), parameter_limit) in zip(lims_largest.items(), (x, y, z)):
                set_lim = getattr(ax, f'set_{axis}lim', None)
                if set_lim:
                    set_lim(data_limit if parameter_limit is None else parameter_limit)

    def set_axis_labels(self, *axlabels: Hashable) -> None:
        if False:
            print('Hello World!')
        'Set axis labels on the left column and bottom row of the grid.'
        from xarray.core.dataarray import DataArray
        for (var, axis) in zip(axlabels, ['x', 'y', 'z']):
            if var is not None:
                if isinstance(var, DataArray):
                    getattr(self, f'set_{axis}labels')(label_from_attrs(var))
                else:
                    getattr(self, f'set_{axis}labels')(str(var))

    def _set_labels(self, axis: str, axes: Iterable, label: str | None=None, **kwargs) -> None:
        if False:
            return 10
        if label is None:
            label = label_from_attrs(self.data[getattr(self, f'_{axis}_var')])
        for ax in axes:
            getattr(ax, f'set_{axis}label')(label, **kwargs)

    def set_xlabels(self, label: None | str=None, **kwargs: Any) -> None:
        if False:
            while True:
                i = 10
        'Label the x axis on the bottom row of the grid.'
        self._set_labels('x', self._bottom_axes, label, **kwargs)

    def set_ylabels(self, label: None | str=None, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Label the y axis on the left column of the grid.'
        self._set_labels('y', self._left_axes, label, **kwargs)

    def set_zlabels(self, label: None | str=None, **kwargs: Any) -> None:
        if False:
            print('Hello World!')
        'Label the z axis.'
        self._set_labels('z', self._left_axes, label, **kwargs)

    def set_titles(self, template: str='{coord} = {value}', maxchar: int=30, size=None, **kwargs) -> None:
        if False:
            while True:
                i = 10
        '\n        Draw titles either above each facet or on the grid margins.\n\n        Parameters\n        ----------\n        template : str, default: "{coord} = {value}"\n            Template for plot titles containing {coord} and {value}\n        maxchar : int, default: 30\n            Truncate titles at maxchar\n        **kwargs : keyword args\n            additional arguments to matplotlib.text\n\n        Returns\n        -------\n        self: FacetGrid object\n\n        '
        import matplotlib as mpl
        if size is None:
            size = mpl.rcParams['axes.labelsize']
        nicetitle = functools.partial(_nicetitle, maxchar=maxchar, template=template)
        if self._single_group:
            for (d, ax) in zip(self.name_dicts.flat, self.axs.flat):
                if d is not None:
                    (coord, value) = list(d.items()).pop()
                    title = nicetitle(coord, value, maxchar=maxchar)
                    ax.set_title(title, size=size, **kwargs)
        else:
            for (index, (ax, row_name, handle)) in enumerate(zip(self.axs[:, -1], self.row_names, self.row_labels)):
                title = nicetitle(coord=self._row_var, value=row_name, maxchar=maxchar)
                if not handle:
                    self.row_labels[index] = ax.annotate(title, xy=(1.02, 0.5), xycoords='axes fraction', rotation=270, ha='left', va='center', **kwargs)
                else:
                    handle.set_text(title)
                    handle.update(kwargs)
            for (index, (ax, col_name, handle)) in enumerate(zip(self.axs[0, :], self.col_names, self.col_labels)):
                title = nicetitle(coord=self._col_var, value=col_name, maxchar=maxchar)
                if not handle:
                    self.col_labels[index] = ax.set_title(title, size=size, **kwargs)
                else:
                    handle.set_text(title)
                    handle.update(kwargs)

    def set_ticks(self, max_xticks: int=_NTICKS, max_yticks: int=_NTICKS, fontsize: str | int=_FONTSIZE) -> None:
        if False:
            while True:
                i = 10
        '\n        Set and control tick behavior.\n\n        Parameters\n        ----------\n        max_xticks, max_yticks : int, optional\n            Maximum number of labeled ticks to plot on x, y axes\n        fontsize : string or int\n            Font size as used by matplotlib text\n\n        Returns\n        -------\n        self : FacetGrid object\n\n        '
        from matplotlib.ticker import MaxNLocator
        x_major_locator = MaxNLocator(nbins=max_xticks)
        y_major_locator = MaxNLocator(nbins=max_yticks)
        for ax in self.axs.flat:
            ax.xaxis.set_major_locator(x_major_locator)
            ax.yaxis.set_major_locator(y_major_locator)
            for tick in itertools.chain(ax.xaxis.get_major_ticks(), ax.yaxis.get_major_ticks()):
                tick.label1.set_fontsize(fontsize)

    def map(self: T_FacetGrid, func: Callable, *args: Hashable, **kwargs: Any) -> T_FacetGrid:
        if False:
            while True:
                i = 10
        "\n        Apply a plotting function to each facet's subset of the data.\n\n        Parameters\n        ----------\n        func : callable\n            A plotting function that takes data and keyword arguments. It\n            must plot to the currently active matplotlib Axes and take a\n            `color` keyword argument. If faceting on the `hue` dimension,\n            it must also take a `label` keyword argument.\n        *args : Hashable\n            Column names in self.data that identify variables with data to\n            plot. The data for each variable is passed to `func` in the\n            order the variables are specified in the call.\n        **kwargs : keyword arguments\n            All keyword arguments are passed to the plotting function.\n\n        Returns\n        -------\n        self : FacetGrid object\n\n        "
        import matplotlib.pyplot as plt
        for (ax, namedict) in zip(self.axs.flat, self.name_dicts.flat):
            if namedict is not None:
                data = self.data.loc[namedict]
                plt.sca(ax)
                innerargs = [data[a].to_numpy() for a in args]
                maybe_mappable = func(*innerargs, **kwargs)
                if maybe_mappable and hasattr(maybe_mappable, 'autoscale_None'):
                    self._mappables.append(maybe_mappable)
        self._finalize_grid(*args[:2])
        return self

def _easy_facetgrid(data: T_DataArrayOrSet, plotfunc: Callable, kind: Literal['line', 'dataarray', 'dataset', 'plot1d'], x: Hashable | None=None, y: Hashable | None=None, row: Hashable | None=None, col: Hashable | None=None, col_wrap: int | None=None, sharex: bool=True, sharey: bool=True, aspect: float | None=None, size: float | None=None, subplot_kws: dict[str, Any] | None=None, ax: Axes | None=None, figsize: Iterable[float] | None=None, **kwargs: Any) -> FacetGrid[T_DataArrayOrSet]:
    if False:
        i = 10
        return i + 15
    '\n    Convenience method to call xarray.plot.FacetGrid from 2d plotting methods\n\n    kwargs are the arguments to 2d plotting method\n    '
    if ax is not None:
        raise ValueError("Can't use axes when making faceted plots.")
    if aspect is None:
        aspect = 1
    if size is None:
        size = 3
    elif figsize is not None:
        raise ValueError('cannot provide both `figsize` and `size` arguments')
    if kwargs.get('z') is not None:
        sharex = False
        sharey = False
    g = FacetGrid(data=data, col=col, row=row, col_wrap=col_wrap, sharex=sharex, sharey=sharey, figsize=figsize, aspect=aspect, size=size, subplot_kws=subplot_kws)
    if kind == 'line':
        return g.map_dataarray_line(plotfunc, x, y, **kwargs)
    if kind == 'dataarray':
        return g.map_dataarray(plotfunc, x, y, **kwargs)
    if kind == 'plot1d':
        return g.map_plot1d(plotfunc, x, y, **kwargs)
    if kind == 'dataset':
        return g.map_dataset(plotfunc, x, y, **kwargs)
    raise ValueError(f'kind must be one of `line`, `dataarray`, `dataset` or `plot1d`, got {kind}')