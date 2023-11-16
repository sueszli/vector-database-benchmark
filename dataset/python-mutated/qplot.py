from __future__ import annotations
import typing
from contextlib import suppress
from warnings import warn
import numpy as np
import pandas as pd
from .exceptions import PlotnineError, PlotnineWarning
from .facets import facet_grid, facet_null, facet_wrap
from .facets.facet_grid import parse_grid_facets
from .facets.facet_wrap import parse_wrap_facets
from .ggplot import ggplot
from .labels import labs
from .mapping.aes import ALL_AESTHETICS, SCALED_AESTHETICS, aes
from .scales import lims, scale_x_log10, scale_y_log10
from .themes import theme
from .utils import Registry, array_kind
if typing.TYPE_CHECKING:
    from typing import Any, Iterable, Literal
    from plotnine.typing import DataLike, Ggplot, TupleFloat2
__all__ = ('qplot',)

def qplot(x: str | Iterable[Any] | range | None=None, y: str | Iterable[Any] | range | None=None, data: DataLike | None=None, facets: str='', margins: bool | list[str]=False, geom: str | list[str] | tuple[str]='auto', xlim: TupleFloat2 | None=None, ylim: TupleFloat2 | None=None, log: Literal['x', 'y', 'xy'] | None=None, main: str | None=None, xlab: str | None=None, ylab: str | None=None, asp: float | None=None, **kwargs: Any) -> Ggplot:
    if False:
        return 10
    "\n    Quick plot\n\n    Parameters\n    ----------\n    x : str | array_like\n        x aesthetic\n    y : str | array_like\n        y aesthetic\n    data : dataframe\n        Data frame to use (optional). If not specified,\n        will create one, extracting arrays from the\n        current environment.\n    geom : str | list\n        *geom(s)* to do the drawing. If ``auto``, defaults\n        to 'point' if ``x`` and ``y`` are specified or\n        'histogram' if only ``x`` is specified.\n    facets : str\n        Facets\n    margins : bool | list[str]\n        variable names to compute margins for. True will compute\n        all possible margins. Depends on the facetting.\n    xlim : tuple\n        x-axis limits\n    ylim : tuple\n        y-axis limits\n    log : str in ``{'x', 'y', 'xy'}``\n        Which (if any) variables to log transform.\n    main : str\n        Plot title\n    xlab : str\n        x-axis label\n    ylab : str\n        y-axis label\n    asp : str | float\n        The y/x aspect ratio.\n    **kwargs : dict\n        Arguments passed on to the geom.\n\n    Returns\n    -------\n    p : ggplot\n        ggplot object\n    "
    from patsy.eval import EvalEnvironment
    environment = EvalEnvironment.capture(1)
    aesthetics = {} if x is None else {'x': x}
    if y is not None:
        aesthetics['y'] = y

    def is_mapping(value: Any) -> bool:
        if False:
            return 10
        '\n        Return True if value is not enclosed in I() function\n        '
        with suppress(AttributeError):
            return not (value.startswith('I(') and value.endswith(')'))
        return True

    def I(value: Any) -> Any:
        if False:
            i = 10
            return i + 15
        return value
    I_env = EvalEnvironment([{'I': I}])
    for ae in kwargs.keys() & ALL_AESTHETICS:
        value = kwargs[ae]
        if is_mapping(value):
            aesthetics[ae] = value
        else:
            kwargs[ae] = I_env.eval(value)
    if isinstance(geom, str):
        geom = [geom]
    elif isinstance(geom, tuple):
        geom = list(geom)
    if data is None:
        data = pd.DataFrame()

    def replace_auto(lst: list[str], str2: str) -> list[str]:
        if False:
            return 10
        "\n        Replace all occurences of 'auto' in with str2\n        "
        for (i, value) in enumerate(lst):
            if value == 'auto':
                lst[i] = str2
        return lst
    if 'auto' in geom:
        if 'sample' in aesthetics:
            replace_auto(geom, 'qq')
        elif y is None:
            env = environment.with_outer_namespace({'factor': pd.Categorical})
            if isinstance(aesthetics['x'], str):
                try:
                    x = env.eval(aesthetics['x'], inner_namespace=data)
                except Exception:
                    msg = "Could not evaluate aesthetic 'x={}'"
                    raise PlotnineError(msg.format(aesthetics['x']))
            elif not hasattr(aesthetics['x'], 'dtype'):
                x = np.asarray(aesthetics['x'])
            if array_kind.discrete(x):
                replace_auto(geom, 'bar')
            else:
                replace_auto(geom, 'histogram')
        else:
            if x is None:
                if isinstance(aesthetics['y'], typing.Sized):
                    aesthetics['x'] = range(len(aesthetics['y']))
                    xlab = 'range(len(y))'
                    ylab = 'y'
                else:
                    raise PlotnineError('Cannot infer how long x should be.')
            replace_auto(geom, 'point')
    p: Ggplot = ggplot(data, aes(**aesthetics), environment=environment)

    def get_facet_type(facets: str) -> Literal['grid', 'wrap', 'null']:
        if False:
            while True:
                i = 10
        with suppress(PlotnineError):
            parse_grid_facets(facets)
            return 'grid'
        with suppress(PlotnineError):
            parse_wrap_facets(facets)
            return 'wrap'
        warn('Could not determine the type of faceting, therefore no faceting.', PlotnineWarning)
        return 'null'
    if facets:
        facet_type = get_facet_type(facets)
        if facet_type == 'grid':
            p += facet_grid(facets, margins=margins)
        elif facet_type == 'wrap':
            p += facet_wrap(facets)
        else:
            p += facet_null()
    for g in geom:
        geom_name = f'geom_{g}'
        geom_klass = Registry[geom_name]
        stat_name = 'stat_{}'.format(geom_klass.DEFAULT_PARAMS['stat'])
        stat_klass = Registry[stat_name]
        recognized = kwargs.keys() & (geom_klass.DEFAULT_PARAMS.keys() | geom_klass.aesthetics() | stat_klass.DEFAULT_PARAMS.keys() | stat_klass.aesthetics())
        recognized = recognized - aesthetics.keys()
        params = {ae: kwargs[ae] for ae in recognized}
        p += geom_klass(**params)
    labels = {}
    for ae in SCALED_AESTHETICS & kwargs.keys():
        with suppress(AttributeError):
            labels[ae] = kwargs[ae].name
    with suppress(AttributeError):
        labels['x'] = xlab if xlab is not None else x.name
    with suppress(AttributeError):
        labels['y'] = ylab if ylab is not None else y.name
    if main is not None:
        labels['title'] = main
    if log:
        if 'x' in log:
            p += scale_x_log10()
        if 'y' in log:
            p += scale_y_log10()
    if labels:
        p += labs(**labels)
    if asp:
        p += theme(aspect_ratio=asp)
    if xlim:
        p += lims(x=xlim)
    if ylim:
        p += lims(y=ylim)
    return p