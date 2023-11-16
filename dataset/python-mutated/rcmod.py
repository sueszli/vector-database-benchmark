"""Control plot style and scaling using the matplotlib rcParams interface."""
import functools
import matplotlib as mpl
from cycler import cycler
from . import palettes
__all__ = ['set_theme', 'set', 'reset_defaults', 'reset_orig', 'axes_style', 'set_style', 'plotting_context', 'set_context', 'set_palette']
_style_keys = ['axes.facecolor', 'axes.edgecolor', 'axes.grid', 'axes.axisbelow', 'axes.labelcolor', 'figure.facecolor', 'grid.color', 'grid.linestyle', 'text.color', 'xtick.color', 'ytick.color', 'xtick.direction', 'ytick.direction', 'lines.solid_capstyle', 'patch.edgecolor', 'patch.force_edgecolor', 'image.cmap', 'font.family', 'font.sans-serif', 'xtick.bottom', 'xtick.top', 'ytick.left', 'ytick.right', 'axes.spines.left', 'axes.spines.bottom', 'axes.spines.right', 'axes.spines.top']
_context_keys = ['font.size', 'axes.labelsize', 'axes.titlesize', 'xtick.labelsize', 'ytick.labelsize', 'legend.fontsize', 'legend.title_fontsize', 'axes.linewidth', 'grid.linewidth', 'lines.linewidth', 'lines.markersize', 'patch.linewidth', 'xtick.major.width', 'ytick.major.width', 'xtick.minor.width', 'ytick.minor.width', 'xtick.major.size', 'ytick.major.size', 'xtick.minor.size', 'ytick.minor.size']

def set_theme(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=True, rc=None):
    if False:
        return 10
    '\n    Set aspects of the visual theme for all matplotlib and seaborn plots.\n\n    This function changes the global defaults for all plots using the\n    matplotlib rcParams system. The themeing is decomposed into several distinct\n    sets of parameter values.\n\n    The options are illustrated in the :doc:`aesthetics <../tutorial/aesthetics>`\n    and :doc:`color palette <../tutorial/color_palettes>` tutorials.\n\n    Parameters\n    ----------\n    context : string or dict\n        Scaling parameters, see :func:`plotting_context`.\n    style : string or dict\n        Axes style parameters, see :func:`axes_style`.\n    palette : string or sequence\n        Color palette, see :func:`color_palette`.\n    font : string\n        Font family, see matplotlib font manager.\n    font_scale : float, optional\n        Separate scaling factor to independently scale the size of the\n        font elements.\n    color_codes : bool\n        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand\n        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.\n    rc : dict or None\n        Dictionary of rc parameter mappings to override the above.\n\n    Examples\n    --------\n\n    .. include:: ../docstrings/set_theme.rst\n\n    '
    set_context(context, font_scale)
    set_style(style, rc={'font.family': font})
    set_palette(palette, color_codes=color_codes)
    if rc is not None:
        mpl.rcParams.update(rc)

def set(*args, **kwargs):
    if False:
        return 10
    '\n    Alias for :func:`set_theme`, which is the preferred interface.\n\n    This function may be removed in the future.\n    '
    set_theme(*args, **kwargs)

def reset_defaults():
    if False:
        return 10
    'Restore all RC params to default settings.'
    mpl.rcParams.update(mpl.rcParamsDefault)

def reset_orig():
    if False:
        for i in range(10):
            print('nop')
    'Restore all RC params to original settings (respects custom rc).'
    from . import _orig_rc_params
    mpl.rcParams.update(_orig_rc_params)

def axes_style(style=None, rc=None):
    if False:
        i = 10
        return i + 15
    '\n    Get the parameters that control the general style of the plots.\n\n    The style parameters control properties like the color of the background and\n    whether a grid is enabled by default. This is accomplished using the\n    matplotlib rcParams system.\n\n    The options are illustrated in the\n    :doc:`aesthetics tutorial <../tutorial/aesthetics>`.\n\n    This function can also be used as a context manager to temporarily\n    alter the global defaults. See :func:`set_theme` or :func:`set_style`\n    to modify the global defaults for all plots.\n\n    Parameters\n    ----------\n    style : None, dict, or one of {darkgrid, whitegrid, dark, white, ticks}\n        A dictionary of parameters or the name of a preconfigured style.\n    rc : dict, optional\n        Parameter mappings to override the values in the preset seaborn\n        style dictionaries. This only updates parameters that are\n        considered part of the style definition.\n\n    Examples\n    --------\n\n    .. include:: ../docstrings/axes_style.rst\n\n    '
    if style is None:
        style_dict = {k: mpl.rcParams[k] for k in _style_keys}
    elif isinstance(style, dict):
        style_dict = style
    else:
        styles = ['white', 'dark', 'whitegrid', 'darkgrid', 'ticks']
        if style not in styles:
            raise ValueError(f"style must be one of {', '.join(styles)}")
        dark_gray = '.15'
        light_gray = '.8'
        style_dict = {'figure.facecolor': 'white', 'axes.labelcolor': dark_gray, 'xtick.direction': 'out', 'ytick.direction': 'out', 'xtick.color': dark_gray, 'ytick.color': dark_gray, 'axes.axisbelow': True, 'grid.linestyle': '-', 'text.color': dark_gray, 'font.family': ['sans-serif'], 'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif'], 'lines.solid_capstyle': 'round', 'patch.edgecolor': 'w', 'patch.force_edgecolor': True, 'image.cmap': 'rocket', 'xtick.top': False, 'ytick.right': False}
        if 'grid' in style:
            style_dict.update({'axes.grid': True})
        else:
            style_dict.update({'axes.grid': False})
        if style.startswith('dark'):
            style_dict.update({'axes.facecolor': '#EAEAF2', 'axes.edgecolor': 'white', 'grid.color': 'white', 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True})
        elif style == 'whitegrid':
            style_dict.update({'axes.facecolor': 'white', 'axes.edgecolor': light_gray, 'grid.color': light_gray, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True})
        elif style in ['white', 'ticks']:
            style_dict.update({'axes.facecolor': 'white', 'axes.edgecolor': dark_gray, 'grid.color': light_gray, 'axes.spines.left': True, 'axes.spines.bottom': True, 'axes.spines.right': True, 'axes.spines.top': True})
        if style == 'ticks':
            style_dict.update({'xtick.bottom': True, 'ytick.left': True})
        else:
            style_dict.update({'xtick.bottom': False, 'ytick.left': False})
    style_dict = {k: v for (k, v) in style_dict.items() if k in _style_keys}
    if rc is not None:
        rc = {k: v for (k, v) in rc.items() if k in _style_keys}
        style_dict.update(rc)
    style_object = _AxesStyle(style_dict)
    return style_object

def set_style(style=None, rc=None):
    if False:
        return 10
    '\n    Set the parameters that control the general style of the plots.\n\n    The style parameters control properties like the color of the background and\n    whether a grid is enabled by default. This is accomplished using the\n    matplotlib rcParams system.\n\n    The options are illustrated in the\n    :doc:`aesthetics tutorial <../tutorial/aesthetics>`.\n\n    See :func:`axes_style` to get the parameter values.\n\n    Parameters\n    ----------\n    style : dict, or one of {darkgrid, whitegrid, dark, white, ticks}\n        A dictionary of parameters or the name of a preconfigured style.\n    rc : dict, optional\n        Parameter mappings to override the values in the preset seaborn\n        style dictionaries. This only updates parameters that are\n        considered part of the style definition.\n\n    Examples\n    --------\n\n    .. include:: ../docstrings/set_style.rst\n\n    '
    style_object = axes_style(style, rc)
    mpl.rcParams.update(style_object)

def plotting_context(context=None, font_scale=1, rc=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Get the parameters that control the scaling of plot elements.\n\n    This affects things like the size of the labels, lines, and other elements\n    of the plot, but not the overall style. This is accomplished using the\n    matplotlib rcParams system.\n\n    The base context is "notebook", and the other contexts are "paper", "talk",\n    and "poster", which are version of the notebook parameters scaled by different\n    values. Font elements can also be scaled independently of (but relative to)\n    the other values.\n\n    This function can also be used as a context manager to temporarily\n    alter the global defaults. See :func:`set_theme` or :func:`set_context`\n    to modify the global defaults for all plots.\n\n    Parameters\n    ----------\n    context : None, dict, or one of {paper, notebook, talk, poster}\n        A dictionary of parameters or the name of a preconfigured set.\n    font_scale : float, optional\n        Separate scaling factor to independently scale the size of the\n        font elements.\n    rc : dict, optional\n        Parameter mappings to override the values in the preset seaborn\n        context dictionaries. This only updates parameters that are\n        considered part of the context definition.\n\n    Examples\n    --------\n\n    .. include:: ../docstrings/plotting_context.rst\n\n    '
    if context is None:
        context_dict = {k: mpl.rcParams[k] for k in _context_keys}
    elif isinstance(context, dict):
        context_dict = context
    else:
        contexts = ['paper', 'notebook', 'talk', 'poster']
        if context not in contexts:
            raise ValueError(f"context must be in {', '.join(contexts)}")
        texts_base_context = {'font.size': 12, 'axes.labelsize': 12, 'axes.titlesize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11, 'legend.fontsize': 11, 'legend.title_fontsize': 12}
        base_context = {'axes.linewidth': 1.25, 'grid.linewidth': 1, 'lines.linewidth': 1.5, 'lines.markersize': 6, 'patch.linewidth': 1, 'xtick.major.width': 1.25, 'ytick.major.width': 1.25, 'xtick.minor.width': 1, 'ytick.minor.width': 1, 'xtick.major.size': 6, 'ytick.major.size': 6, 'xtick.minor.size': 4, 'ytick.minor.size': 4}
        base_context.update(texts_base_context)
        scaling = dict(paper=0.8, notebook=1, talk=1.5, poster=2)[context]
        context_dict = {k: v * scaling for (k, v) in base_context.items()}
        font_keys = texts_base_context.keys()
        font_dict = {k: context_dict[k] * font_scale for k in font_keys}
        context_dict.update(font_dict)
    if rc is not None:
        rc = {k: v for (k, v) in rc.items() if k in _context_keys}
        context_dict.update(rc)
    context_object = _PlottingContext(context_dict)
    return context_object

def set_context(context=None, font_scale=1, rc=None):
    if False:
        print('Hello World!')
    '\n    Set the parameters that control the scaling of plot elements.\n\n    This affects things like the size of the labels, lines, and other elements\n    of the plot, but not the overall style. This is accomplished using the\n    matplotlib rcParams system.\n\n    The base context is "notebook", and the other contexts are "paper", "talk",\n    and "poster", which are version of the notebook parameters scaled by different\n    values. Font elements can also be scaled independently of (but relative to)\n    the other values.\n\n    See :func:`plotting_context` to get the parameter values.\n\n    Parameters\n    ----------\n    context : dict, or one of {paper, notebook, talk, poster}\n        A dictionary of parameters or the name of a preconfigured set.\n    font_scale : float, optional\n        Separate scaling factor to independently scale the size of the\n        font elements.\n    rc : dict, optional\n        Parameter mappings to override the values in the preset seaborn\n        context dictionaries. This only updates parameters that are\n        considered part of the context definition.\n\n    Examples\n    --------\n\n    .. include:: ../docstrings/set_context.rst\n\n    '
    context_object = plotting_context(context, font_scale, rc)
    mpl.rcParams.update(context_object)

class _RCAesthetics(dict):

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        rc = mpl.rcParams
        self._orig = {k: rc[k] for k in self._keys}
        self._set(self)

    def __exit__(self, exc_type, exc_value, exc_tb):
        if False:
            return 10
        self._set(self._orig)

    def __call__(self, func):
        if False:
            i = 10
            return i + 15

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            with self:
                return func(*args, **kwargs)
        return wrapper

class _AxesStyle(_RCAesthetics):
    """Light wrapper on a dict to set style temporarily."""
    _keys = _style_keys
    _set = staticmethod(set_style)

class _PlottingContext(_RCAesthetics):
    """Light wrapper on a dict to set context temporarily."""
    _keys = _context_keys
    _set = staticmethod(set_context)

def set_palette(palette, n_colors=None, desat=None, color_codes=False):
    if False:
        for i in range(10):
            print('nop')
    'Set the matplotlib color cycle using a seaborn palette.\n\n    Parameters\n    ----------\n    palette : seaborn color palette | matplotlib colormap | hls | husl\n        Palette definition. Should be something :func:`color_palette` can process.\n    n_colors : int\n        Number of colors in the cycle. The default number of colors will depend\n        on the format of ``palette``, see the :func:`color_palette`\n        documentation for more information.\n    desat : float\n        Proportion to desaturate each color by.\n    color_codes : bool\n        If ``True`` and ``palette`` is a seaborn palette, remap the shorthand\n        color codes (e.g. "b", "g", "r", etc.) to the colors from this palette.\n\n    See Also\n    --------\n    color_palette : build a color palette or set the color cycle temporarily\n                    in a ``with`` statement.\n    set_context : set parameters to scale plot elements\n    set_style : set the default parameters for figure style\n\n    '
    colors = palettes.color_palette(palette, n_colors, desat)
    cyl = cycler('color', colors)
    mpl.rcParams['axes.prop_cycle'] = cyl
    if color_codes:
        try:
            palettes.set_color_codes(palette)
        except (ValueError, TypeError):
            pass