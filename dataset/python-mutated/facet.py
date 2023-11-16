from __future__ import annotations
import itertools
import types
import typing
from copy import copy, deepcopy
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
from ..exceptions import PlotnineError
from ..scales.scales import Scales
from ..utils import cross_join, match
from .strips import Strips
if typing.TYPE_CHECKING:
    from typing import Any, Literal, Optional
    import numpy.typing as npt
    from matplotlib.gridspec import GridSpec
    from plotnine.iapi import layout_details, panel_view
    from plotnine.typing import Axes, Coord, EvalEnvironment, Figure, Ggplot, Layers, Layout, Scale, Theme

class facet:
    """
    Base class for all facets

    Parameters
    ----------
    scales : str in ``['fixed', 'free', 'free_x', 'free_y']``
        Whether ``x`` or ``y`` scales should be allowed (free)
        to vary according to the data along the rows or the
        columns. Default is ``'fixed'``.
    shrink : bool
        Whether to shrink the scales to the output of the
        statistics instead of the raw data. Default is ``True``.
    labeller : str | function
        How to label the facets. If it is a ``str``, it should
        be one of ``'label_value'`` ``'label_both'`` or
        ``'label_context'``. Default is ``'label_value'``
    as_table : bool
        If ``True``, the facets are laid out like a table with
        the highest values at the bottom-right. If ``False``
        the facets are laid out like a plot with the highest
        value a the top-right. Default it ``True``.
    drop : bool
        If ``True``, all factor levels not used in the data
        will automatically be dropped. If ``False``, all
        factor levels will be shown, regardless of whether
        or not they appear in the data. Default is ``True``.
    dir : str in ``['h', 'v']``
        Direction in which to layout the panels. ``h`` for
        horizontal and ``v`` for vertical.
    """
    ncol: int
    nrow: int
    as_table = True
    drop = True
    shrink = True
    free: dict[Literal['x', 'y'], bool]
    params: dict[str, Any]
    theme: Theme
    figure: Figure
    coordinates: Coord
    layout: Layout
    axs: list[Axes]
    first_ax: Axes
    last_ax: Axes
    num_vars_x = 0
    num_vars_y = 0
    plot: Ggplot
    strips: Strips
    space: Literal['fixed', 'free', 'free_x', 'free_y'] | dict[Literal['x', 'y'], list[int]] = 'fixed'
    grid_spec: GridSpec

    def __init__(self, scales: Literal['fixed', 'free', 'free_x', 'free_y']='fixed', shrink: bool=True, labeller: Literal['label_value', 'label_both', 'label_context']='label_value', as_table: bool=True, drop: bool=True, dir: Literal['h', 'v']='h'):
        if False:
            while True:
                i = 10
        from .labelling import as_labeller
        self.shrink = shrink
        self.labeller = as_labeller(labeller)
        self.as_table = as_table
        self.drop = drop
        self.dir = dir
        self.free = {'x': scales in ('free_x', 'free'), 'y': scales in ('free_y', 'free')}

    def __radd__(self, gg: Ggplot) -> Ggplot:
        if False:
            while True:
                i = 10
        '\n        Add facet to ggplot object\n        '
        gg.facet = copy(self)
        gg.facet.plot = gg
        return gg

    def set_properties(self, gg: Ggplot):
        if False:
            while True:
                i = 10
        '\n        Copy required properties from ggplot object\n        '
        self.axs = gg.axs
        self.coordinates = gg.coordinates
        self.figure = gg.figure
        self.layout = gg.layout
        self.theme = gg.theme
        self.strips = Strips.from_facet(self)

    def setup_data(self, data: list[pd.DataFrame]) -> list[pd.DataFrame]:
        if False:
            print('Hello World!')
        '\n        Allow the facet to manipulate the data\n\n        Parameters\n        ----------\n        data : list of dataframes\n            Data for each of the layers\n\n        Returns\n        -------\n        data : list of dataframes\n            Data for each of the layers\n\n        Notes\n        -----\n        This method will be called after :meth:`setup_params`,\n        therefore the `params` property will be set.\n        '
        return data

    def setup_params(self, data: list[pd.DataFrame]):
        if False:
            while True:
                i = 10
        '\n        Create facet parameters\n\n        Parameters\n        ----------\n        data : list of dataframes\n            Plot data and data for the layers\n        '
        self.params = {}

    def init_scales(self, layout: pd.DataFrame, x_scale: Optional[Scale]=None, y_scale: Optional[Scale]=None) -> types.SimpleNamespace:
        if False:
            return 10
        scales = types.SimpleNamespace()
        if x_scale is not None:
            n = layout['SCALE_X'].max()
            scales.x = Scales([x_scale.clone() for i in range(n)])
        if y_scale is not None:
            n = layout['SCALE_Y'].max()
            scales.y = Scales([y_scale.clone() for i in range(n)])
        return scales

    def map(self, data: pd.DataFrame, layout: pd.DataFrame) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Assign a data points to panels\n\n        Parameters\n        ----------\n        data : DataFrame\n            Data for a layer\n        layout : DataFrame\n            As returned by self.compute_layout\n\n        Returns\n        -------\n        data : DataFrame\n            Data with all points mapped to the panels\n            on which they will be plotted.\n        '
        msg = '{} should implement this method.'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    def compute_layout(self, data: list[pd.DataFrame]) -> pd.DataFrame:
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute layout\n\n        Parameters\n        ----------\n        data : Dataframes\n            Dataframe for a each layer\n        '
        msg = '{} should implement this method.'
        raise NotImplementedError(msg.format(self.__class__.__name__))

    def finish_data(self, data: pd.DataFrame, layout: Layout) -> pd.DataFrame:
        if False:
            i = 10
            return i + 15
        "\n        Modify data before it is drawn out by the geom\n\n        The default is to return the data without modification.\n        Subclasses should override this method as the require.\n\n        Parameters\n        ----------\n        data : DataFrame\n            A single layer's data.\n        layout : Layout\n            Layout\n\n        Returns\n        -------\n        data : DataFrame\n            Modified layer data\n        "
        return data

    def train_position_scales(self, layout: Layout, layers: Layers) -> facet:
        if False:
            while True:
                i = 10
        '\n        Compute ranges for the x and y scales\n        '
        _layout = layout.layout
        panel_scales_x = layout.panel_scales_x
        panel_scales_y = layout.panel_scales_y
        for layer in layers:
            data = layer.data
            match_id = match(data['PANEL'], _layout['PANEL'])
            if panel_scales_x:
                x_vars = list(set(panel_scales_x[0].aesthetics) & set(data.columns))
                SCALE_X = _layout['SCALE_X'].iloc[match_id].tolist()
                panel_scales_x.train(data, x_vars, SCALE_X)
            if panel_scales_y:
                y_vars = list(set(panel_scales_y[0].aesthetics) & set(data.columns))
                SCALE_Y = _layout['SCALE_Y'].iloc[match_id].tolist()
                panel_scales_y.train(data, y_vars, SCALE_Y)
        return self

    def make_ax_strips(self, layout_info: layout_details, ax: Axes) -> Strips:
        if False:
            print('Hello World!')
        '\n        Create strips for the facet\n\n        Parameters\n        ----------\n        layout_info : dict-like\n            Layout information. Row from the layout table\n\n        ax : axes\n            Axes to label\n        '
        return Strips()

    def set_limits_breaks_and_labels(self, panel_params: panel_view, ax: Axes):
        if False:
            i = 10
            return i + 15
        '\n        Add limits, breaks and labels to the axes\n\n        Parameters\n        ----------\n        ranges : dict-like\n            range information for the axes\n        ax : Axes\n            Axes\n        '
        from .._mpl.ticker import MyFixedFormatter

        def _inf_to_none(t: tuple[float, float]) -> tuple[float | None, float | None]:
            if False:
                i = 10
                return i + 15
            '\n            Replace infinities with None\n            '
            a = t[0] if np.isfinite(t[0]) else None
            b = t[1] if np.isfinite(t[1]) else None
            return (a, b)
        ax.set_xlim(_inf_to_none(panel_params.x.range))
        ax.set_ylim(_inf_to_none(panel_params.y.range))
        if typing.TYPE_CHECKING:
            assert callable(ax.set_xticks)
            assert callable(ax.set_yticks)
        ax.set_xticks(panel_params.x.breaks, panel_params.x.labels)
        ax.set_yticks(panel_params.y.breaks, panel_params.y.labels)
        ax.set_xticks(panel_params.x.minor_breaks, minor=True)
        ax.set_yticks(panel_params.y.minor_breaks, minor=True)
        ax.xaxis.set_major_formatter(MyFixedFormatter(panel_params.x.labels))
        ax.yaxis.set_major_formatter(MyFixedFormatter(panel_params.y.labels))
        _property = self.theme.themeables.property
        margin = _property('axis_text_x', 'margin')
        pad_x = margin.get_as('t', 'pt')
        margin = _property('axis_text_y', 'margin')
        pad_y = margin.get_as('r', 'pt')
        ax.tick_params(axis='x', which='major', pad=pad_x)
        ax.tick_params(axis='y', which='major', pad=pad_y)

    def __deepcopy__(self, memo: dict[Any, Any]) -> facet:
        if False:
            for i in range(10):
                print('nop')
        '\n        Deep copy without copying the dataframe and environment\n        '
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        old = self.__dict__
        new = result.__dict__
        shallow = {'figure', 'axs', 'first_ax', 'last_ax'}
        for (key, item) in old.items():
            if key in shallow:
                new[key] = old[key]
                memo[id(new[key])] = new[key]
            else:
                new[key] = deepcopy(old[key], memo)
        return result

    def _create_subplots(self, fig: Figure, layout: pd.DataFrame) -> list[Axes]:
        if False:
            i = 10
            return i + 15
        '\n        Create suplots and return axs\n        '
        from matplotlib.gridspec import GridSpec
        num_panels = len(layout)
        axsarr = np.empty((self.nrow, self.ncol), dtype=object)
        space = self.space
        default_space: dict[Literal['x', 'y'], list[int]] = {'x': [1 for x in range(self.ncol)], 'y': [1 for x in range(self.nrow)]}
        if isinstance(space, str):
            if space == 'fixed':
                space = default_space
            else:
                space = default_space
        elif isinstance(space, dict):
            if 'x' not in space:
                space['x'] = default_space['x']
            if 'y' not in space:
                space['y'] = default_space['y']
        if len(space['x']) != self.ncol:
            raise ValueError('The number of x-ratios for the facet space sizes should match the number of columns.')
        if len(space['y']) != self.nrow:
            raise ValueError('The number of y-ratios for the facet space sizes should match the number of rows.')
        gs = GridSpec(self.nrow, self.ncol, height_ratios=space['y'], width_ratios=space['x'])
        self.grid_spec = gs
        i = 1
        for row in range(self.nrow):
            for col in range(self.ncol):
                axsarr[row, col] = fig.add_subplot(gs[i - 1])
                i += 1
        if self.dir == 'h':
            order: Literal['C', 'F'] = 'C'
            if not self.as_table:
                axsarr = axsarr[::-1]
        elif self.dir == 'v':
            order = 'F'
            if not self.as_table:
                axsarr = np.array([row[::-1] for row in axsarr])
        else:
            raise ValueError(f"Bad value `dir='{self.dir}'` for direction")
        axs = axsarr.ravel(order)
        for ax in axs[num_panels:]:
            fig.delaxes(ax)
        axs = axs[:num_panels]
        return list(axs)

    def make_axes(self, figure: Figure, layout: pd.DataFrame, coordinates: Coord) -> list[Axes]:
        if False:
            print('Hello World!')
        '\n        Create and return Matplotlib axes\n        '
        axs = self._create_subplots(figure, layout)
        self.first_ax = figure.axes[0]
        self.last_ax = figure.axes[-1]
        self.figure = figure
        self.axs = axs
        return axs

    def _aspect_ratio(self) -> Optional[float]:
        if False:
            i = 10
            return i + 15
        '\n        Return the aspect_ratio\n        '
        aspect_ratio = self.theme.themeables.property('aspect_ratio')
        if aspect_ratio == 'auto':
            if not self.free['x'] and (not self.free['y']):
                aspect_ratio = self.coordinates.aspect(self.layout.panel_params[0])
            else:
                aspect_ratio = None
        return aspect_ratio

def combine_vars(data: list[pd.DataFrame], environment: EvalEnvironment, vars: list[str], drop: bool=True) -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    '\n    Generate all combinations of data needed for facetting\n\n    The first data frame in the list should be the default data\n    for the plot. Other data frames in the list are ones that are\n    added to the layers.\n    '
    if len(vars) == 0:
        return pd.DataFrame()
    values = [eval_facet_vars(df, vars, environment) for df in data if df is not None]
    has_all = [x.shape[1] == len(vars) for x in values]
    if not any(has_all):
        raise PlotnineError('At least one layer must contain all variables used for facetting')
    base = pd.concat([x for (i, x) in enumerate(values) if has_all[i]], axis=0)
    base = base.drop_duplicates()
    if not drop:
        base = unique_combs(base)
    base = base.sort_values(base.columns.tolist())
    for (i, value) in enumerate(values):
        if has_all[i] or len(value.columns) == 0:
            continue
        old = base.loc[:, list(base.columns.difference(value.columns))]
        new = value.loc[:, list(base.columns.intersection(value.columns))].drop_duplicates()
        if not drop:
            new = unique_combs(new)
        base = pd.concat([base, cross_join(old, new)], ignore_index=True)
    if len(base) == 0:
        raise PlotnineError('Faceting variables must have at least one value')
    base = base.reset_index(drop=True)
    return base

def unique_combs(df: pd.DataFrame) -> pd.DataFrame:
    if False:
        return 10
    '\n    Generate all possible combinations of the values in the columns\n    '

    def _unique(s: pd.Series[Any]) -> npt.NDArray[Any] | pd.Index:
        if False:
            print('Hello World!')
        if isinstance(s.dtype, pdtypes.CategoricalDtype):
            return s.cat.categories
        return s.unique()
    lst = (_unique(x) for (_, x) in df.items())
    rows = list(itertools.product(*lst))
    _df = pd.DataFrame(rows, columns=df.columns)
    for col in df:
        t = df[col].dtype
        _df[col] = _df[col].astype(t, copy=False)
    return _df

def layout_null() -> pd.DataFrame:
    if False:
        i = 10
        return i + 15
    '\n    Layout Null\n    '
    layout = pd.DataFrame({'PANEL': [1], 'ROW': 1, 'COL': 1, 'SCALE_X': 1, 'SCALE_Y': 1, 'AXIS_X': True, 'AXIS_Y': True})
    return layout

def add_missing_facets(data: pd.DataFrame, layout: pd.DataFrame, vars: list[str], facet_vals: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if False:
        return 10
    '\n    Add missing facets\n    '
    missing_facets = list(set(vars) - set(facet_vals.columns.tolist()))
    if missing_facets:
        to_add = layout.loc[:, missing_facets].drop_duplicates()
        to_add.reset_index(drop=True, inplace=True)
        data_rep = np.tile(np.arange(len(data)), len(to_add))
        facet_rep = np.repeat(np.arange(len(to_add)), len(data))
        data = data.iloc[data_rep, :].reset_index(drop=True)
        facet_vals = facet_vals.iloc[data_rep, :].reset_index(drop=True)
        to_add = to_add.iloc[facet_rep, :].reset_index(drop=True)
        facet_vals = pd.concat([facet_vals, to_add], axis=1, ignore_index=False)
    return (data, facet_vals)

def eval_facet_vars(data: pd.DataFrame, vars: list[str], env: EvalEnvironment) -> pd.DataFrame:
    if False:
        while True:
            i = 10
    '\n    Evaluate facet variables\n\n    Parameters\n    ----------\n    data : DataFrame\n        Factet dataframe\n    vars : list\n        Facet variables\n    env : environment\n        Plot environment\n\n    Returns\n    -------\n    facet_vals : DataFrame\n        Facet values that correspond to the specified\n        variables.\n    '

    def I(value: Any) -> Any:
        if False:
            return 10
        return value
    env = env.with_outer_namespace({'I': I})
    facet_vals = pd.DataFrame(index=data.index)
    for name in vars:
        if name in data:
            res = data[name]
        elif str.isidentifier(name):
            continue
        else:
            try:
                res = env.eval(name, inner_namespace=data)
            except NameError:
                continue
        facet_vals[name] = res
    return facet_vals