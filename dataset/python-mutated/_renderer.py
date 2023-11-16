from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import sys
from collections.abc import Iterable
import numpy as np
from ..core.properties import ColorSpec
from ..models import ColumnarDataSource, ColumnDataSource, GlyphRenderer
from ..util.strings import nice_join
from ._legends import pop_legend_kwarg, update_legend
__all__ = ('create_renderer', 'make_glyph', 'pop_visuals')
RENDERER_ARGS = ['name', 'coordinates', 'x_range_name', 'y_range_name', 'level', 'view', 'visible', 'muted']

def get_default_color(plot=None):
    if False:
        for i in range(10):
            print('nop')
    colors = ['#1f77b4', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']
    if plot:
        renderers = plot.renderers
        renderers = [x for x in renderers if x.__view_model__ == 'GlyphRenderer']
        num_renderers = len(renderers)
        return colors[num_renderers]
    else:
        return colors[0]

def create_renderer(glyphclass, plot, **kwargs):
    if False:
        print('Hello World!')
    is_user_source = _convert_data_source(kwargs)
    legend_kwarg = pop_legend_kwarg(kwargs)
    renderer_kws = _pop_renderer_args(kwargs)
    source = renderer_kws['data_source']
    glyph_visuals = pop_visuals(glyphclass, kwargs)
    incompatible_literal_spec_values = []
    incompatible_literal_spec_values += _process_sequence_literals(glyphclass, kwargs, source, is_user_source)
    incompatible_literal_spec_values += _process_sequence_literals(glyphclass, glyph_visuals, source, is_user_source)
    if incompatible_literal_spec_values:
        raise RuntimeError(_GLYPH_SOURCE_MSG % nice_join(incompatible_literal_spec_values, conjunction='and'))
    nonselection_visuals = pop_visuals(glyphclass, kwargs, prefix='nonselection_', defaults=glyph_visuals, override_defaults={'alpha': 0.1})
    if any((x.startswith('selection_') for x in kwargs)):
        selection_visuals = pop_visuals(glyphclass, kwargs, prefix='selection_', defaults=glyph_visuals)
    else:
        selection_visuals = None
    if any((x.startswith('hover_') for x in kwargs)):
        hover_visuals = pop_visuals(glyphclass, kwargs, prefix='hover_', defaults=glyph_visuals)
    else:
        hover_visuals = None
    muted_visuals = pop_visuals(glyphclass, kwargs, prefix='muted_', defaults=glyph_visuals, override_defaults={'alpha': 0.2})
    glyph = make_glyph(glyphclass, kwargs, glyph_visuals)
    nonselection_glyph = make_glyph(glyphclass, kwargs, nonselection_visuals)
    selection_glyph = make_glyph(glyphclass, kwargs, selection_visuals)
    hover_glyph = make_glyph(glyphclass, kwargs, hover_visuals)
    muted_glyph = make_glyph(glyphclass, kwargs, muted_visuals)
    glyph_renderer = GlyphRenderer(glyph=glyph, nonselection_glyph=nonselection_glyph or 'auto', selection_glyph=selection_glyph or 'auto', hover_glyph=hover_glyph, muted_glyph=muted_glyph or 'auto', **renderer_kws)
    plot.renderers.append(glyph_renderer)
    if legend_kwarg:
        update_legend(plot, legend_kwarg, glyph_renderer)
    return glyph_renderer

def make_glyph(glyphclass, kws, extra):
    if False:
        while True:
            i = 10
    if extra is None:
        return None
    kws = kws.copy()
    kws.update(extra)
    return glyphclass(**kws)

def pop_visuals(glyphclass, props, prefix='', defaults={}, override_defaults={}):
    if False:
        i = 10
        return i + 15
    '\n    Applies basic cascading logic to deduce properties for a glyph.\n\n    Args:\n        glyphclass :\n            the type of glyph being handled\n\n        props (dict) :\n            Maps properties and prefixed properties to their values.\n            Keys in `props` matching `glyphclass` visual properties (those of\n            \'line_\', \'fill_\', \'hatch_\' or \'text_\') with added `prefix` will get\n            popped, other keys will be ignored.\n            Keys take the form \'[{prefix}][{feature}_]{trait}\'. Only {feature}\n              must not contain underscores.\n            Keys of the form \'{prefix}{trait}\' work as lower precedence aliases\n              for {trait} for all {features}, as long as the glyph has no\n              property called {trait}. I.e. this won\'t apply to "width" in a\n              `rect` glyph.\n            Ex: {\'fill_color\': \'blue\', \'selection_line_width\': 0.5}\n\n        prefix (str) :\n            Prefix used when accessing `props`. Ex: \'selection_\'\n\n        override_defaults (dict) :\n            Explicitly provided fallback based on \'{trait}\', in case property\n            not set in `props`.\n            Ex. \'width\' here may be used for \'selection_line_width\'.\n\n        defaults (dict) :\n            Property fallback, in case prefixed property not in `props` or\n            `override_defaults`.\n            Ex. \'line_width\' here may be used for \'selection_line_width\'.\n\n    Returns:\n        result (dict) :\n            Resulting properties for the instance (no prefixes).\n\n    Notes:\n        Feature trait \'text_color\', as well as traits \'color\' and \'alpha\', have\n        ultimate defaults in case those can\'t be deduced.\n    '
    defaults = defaults.copy()
    defaults.setdefault('text_color', 'black')
    defaults.setdefault('hatch_color', 'black')
    trait_defaults = {}
    trait_defaults.setdefault('color', get_default_color())
    trait_defaults.setdefault('alpha', 1.0)
    (result, traits) = (dict(), set())
    prop_names = set(glyphclass.properties())
    for name in filter(_is_visual, prop_names):
        (_, trait) = _split_feature_trait(name)
        if prefix + name in props:
            result[name] = props.pop(prefix + name)
        elif trait not in prop_names and prefix + trait in props:
            result[name] = props[prefix + trait]
        elif trait in override_defaults:
            result[name] = override_defaults[trait]
        elif name in defaults:
            result[name] = defaults[name]
        elif trait in trait_defaults:
            result[name] = trait_defaults[trait]
        if trait not in prop_names:
            traits.add(trait)
    for trait in traits:
        props.pop(prefix + trait, None)
    return result

def _convert_data_source(kwargs):
    if False:
        while True:
            i = 10
    is_user_source = kwargs.get('source', None) is not None
    if is_user_source:
        source = kwargs['source']
        if not isinstance(source, ColumnarDataSource):
            try:
                source = ColumnDataSource(source)
            except ValueError as err:
                msg = f'Failed to auto-convert {type(source)} to ColumnDataSource.\n Original error: {err}'
                raise ValueError(msg).with_traceback(sys.exc_info()[2])
            kwargs['source'] = source
    return is_user_source

def _pop_renderer_args(kwargs):
    if False:
        return 10
    result = {attr: kwargs.pop(attr) for attr in RENDERER_ARGS if attr in kwargs}
    result['data_source'] = kwargs.pop('source', ColumnDataSource())
    return result

def _process_sequence_literals(glyphclass, kwargs, source, is_user_source):
    if False:
        i = 10
        return i + 15
    incompatible_literal_spec_values = []
    dataspecs = glyphclass.dataspecs()
    for (var, val) in kwargs.items():
        if not isinstance(val, Iterable):
            continue
        if isinstance(val, dict):
            continue
        if var not in dataspecs:
            continue
        if isinstance(val, str):
            continue
        if isinstance(dataspecs[var], ColorSpec) and dataspecs[var].is_color_tuple_shape(val):
            continue
        if isinstance(val, np.ndarray):
            if isinstance(dataspecs[var], ColorSpec):
                if val.dtype == 'uint32' and val.ndim == 1:
                    pass
                elif val.dtype == 'uint8' and val.ndim == 1:
                    pass
                elif val.dtype.kind == 'U' and val.ndim == 1:
                    pass
                elif (val.dtype == 'uint8' or val.dtype.kind == 'f') and val.ndim == 2 and (val.shape[1] in (3, 4)):
                    pass
                else:
                    raise RuntimeError(f"Color columns need to be of type uint32[N], uint8[N] or uint8/float[N, {{3, 4}}] ({var} is {val.dtype}[{', '.join(map(str, val.shape))}]")
            elif val.ndim != 1:
                raise RuntimeError(f'Columns need to be 1D ({var} is not)')
        if is_user_source:
            incompatible_literal_spec_values.append(var)
        else:
            source.add(val, name=var)
            kwargs[var] = var
    return incompatible_literal_spec_values

def _split_feature_trait(ft):
    if False:
        i = 10
        return i + 15
    "Feature is up to first '_'. Ex. 'line_color' => ['line', 'color']"
    ft = ft.split('_', 1)
    return ft if len(ft) == 2 else [*ft, None]

def _is_visual(ft):
    if False:
        while True:
            i = 10
    'Whether a feature trait name is visual'
    (feature, trait) = _split_feature_trait(ft)
    return feature in ('line', 'fill', 'hatch', 'text', 'global') and trait is not None
_GLYPH_SOURCE_MSG = "\n\nExpected %s to reference fields in the supplied data source.\n\nWhen a 'source' argument is passed to a glyph method, values that are sequences\n(like lists or arrays) must come from references to data columns in the source.\n\nFor instance, as an example:\n\n    source = ColumnDataSource(data=dict(x=a_list, y=an_array))\n\n    p.circle(x='x', y='y', source=source, ...) # pass column names and a source\n\nAlternatively, *all* data sequences may be provided as literals as long as a\nsource is *not* provided:\n\n    p.circle(x=a_list, y=an_array, ...)  # pass actual sequences and no source\n\n"