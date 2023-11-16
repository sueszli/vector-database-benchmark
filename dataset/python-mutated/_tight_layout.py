"""
Routines to adjust subplot params so that subplots are
nicely fit in the figure. In doing so, only axis labels, tick labels, axes
titles and offsetboxes that are anchored to axes are currently considered.

Internally, this module assumes that the margins (left margin, etc.) which are
differences between ``Axes.get_tightbbox`` and ``Axes.bbox`` are independent of
Axes position. This may fail if ``Axes.adjustable`` is ``datalim`` as well as
such cases as when left or right margin are affected by xlabel.
"""
import numpy as np
import matplotlib as mpl
from matplotlib import _api, artist as martist
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

def _auto_adjust_subplotpars(fig, renderer, shape, span_pairs, subplot_list, ax_bbox_list=None, pad=1.08, h_pad=None, w_pad=None, rect=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a dict of subplot parameters to adjust spacing between subplots\n    or ``None`` if resulting axes would have zero height or width.\n\n    Note that this function ignores geometry information of subplot itself, but\n    uses what is given by the *shape* and *subplot_list* parameters.  Also, the\n    results could be incorrect if some subplots have ``adjustable=datalim``.\n\n    Parameters\n    ----------\n    shape : tuple[int, int]\n        Number of rows and columns of the grid.\n    span_pairs : list[tuple[slice, slice]]\n        List of rowspans and colspans occupied by each subplot.\n    subplot_list : list of subplots\n        List of subplots that will be used to calculate optimal subplot_params.\n    pad : float\n        Padding between the figure edge and the edges of subplots, as a\n        fraction of the font size.\n    h_pad, w_pad : float\n        Padding (height/width) between edges of adjacent subplots, as a\n        fraction of the font size.  Defaults to *pad*.\n    rect : tuple\n        (left, bottom, right, top), default: None.\n    '
    (rows, cols) = shape
    font_size_inch = FontProperties(size=mpl.rcParams['font.size']).get_size_in_points() / 72
    pad_inch = pad * font_size_inch
    vpad_inch = h_pad * font_size_inch if h_pad is not None else pad_inch
    hpad_inch = w_pad * font_size_inch if w_pad is not None else pad_inch
    if len(span_pairs) != len(subplot_list) or len(subplot_list) == 0:
        raise ValueError
    if rect is None:
        margin_left = margin_bottom = margin_right = margin_top = None
    else:
        (margin_left, margin_bottom, _right, _top) = rect
        margin_right = 1 - _right if _right else None
        margin_top = 1 - _top if _top else None
    vspaces = np.zeros((rows + 1, cols))
    hspaces = np.zeros((rows, cols + 1))
    if ax_bbox_list is None:
        ax_bbox_list = [Bbox.union([ax.get_position(original=True) for ax in subplots]) for subplots in subplot_list]
    for (subplots, ax_bbox, (rowspan, colspan)) in zip(subplot_list, ax_bbox_list, span_pairs):
        if all((not ax.get_visible() for ax in subplots)):
            continue
        bb = []
        for ax in subplots:
            if ax.get_visible():
                bb += [martist._get_tightbbox_for_layout_only(ax, renderer)]
        tight_bbox_raw = Bbox.union(bb)
        tight_bbox = fig.transFigure.inverted().transform_bbox(tight_bbox_raw)
        hspaces[rowspan, colspan.start] += ax_bbox.xmin - tight_bbox.xmin
        hspaces[rowspan, colspan.stop] += tight_bbox.xmax - ax_bbox.xmax
        vspaces[rowspan.start, colspan] += tight_bbox.ymax - ax_bbox.ymax
        vspaces[rowspan.stop, colspan] += ax_bbox.ymin - tight_bbox.ymin
    (fig_width_inch, fig_height_inch) = fig.get_size_inches()
    if not margin_left:
        margin_left = max(hspaces[:, 0].max(), 0) + pad_inch / fig_width_inch
        suplabel = fig._supylabel
        if suplabel and suplabel.get_in_layout():
            rel_width = fig.transFigure.inverted().transform_bbox(suplabel.get_window_extent(renderer)).width
            margin_left += rel_width + pad_inch / fig_width_inch
    if not margin_right:
        margin_right = max(hspaces[:, -1].max(), 0) + pad_inch / fig_width_inch
    if not margin_top:
        margin_top = max(vspaces[0, :].max(), 0) + pad_inch / fig_height_inch
        if fig._suptitle and fig._suptitle.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(fig._suptitle.get_window_extent(renderer)).height
            margin_top += rel_height + pad_inch / fig_height_inch
    if not margin_bottom:
        margin_bottom = max(vspaces[-1, :].max(), 0) + pad_inch / fig_height_inch
        suplabel = fig._supxlabel
        if suplabel and suplabel.get_in_layout():
            rel_height = fig.transFigure.inverted().transform_bbox(suplabel.get_window_extent(renderer)).height
            margin_bottom += rel_height + pad_inch / fig_height_inch
    if margin_left + margin_right >= 1:
        _api.warn_external('Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations.')
        return None
    if margin_bottom + margin_top >= 1:
        _api.warn_external('Tight layout not applied. The bottom and top margins cannot be made large enough to accommodate all axes decorations.')
        return None
    kwargs = dict(left=margin_left, right=1 - margin_right, bottom=margin_bottom, top=1 - margin_top)
    if cols > 1:
        hspace = hspaces[:, 1:-1].max() + hpad_inch / fig_width_inch
        h_axes = (1 - margin_right - margin_left - hspace * (cols - 1)) / cols
        if h_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout cannot make axes width small enough to accommodate all axes decorations')
            return None
        else:
            kwargs['wspace'] = hspace / h_axes
    if rows > 1:
        vspace = vspaces[1:-1, :].max() + vpad_inch / fig_height_inch
        v_axes = (1 - margin_top - margin_bottom - vspace * (rows - 1)) / rows
        if v_axes < 0:
            _api.warn_external('Tight layout not applied. tight_layout cannot make axes height small enough to accommodate all axes decorations.')
            return None
        else:
            kwargs['hspace'] = vspace / v_axes
    return kwargs

def get_subplotspec_list(axes_list, grid_spec=None):
    if False:
        i = 10
        return i + 15
    '\n    Return a list of subplotspec from the given list of axes.\n\n    For an instance of axes that does not support subplotspec, None is inserted\n    in the list.\n\n    If grid_spec is given, None is inserted for those not from the given\n    grid_spec.\n    '
    subplotspec_list = []
    for ax in axes_list:
        axes_or_locator = ax.get_axes_locator()
        if axes_or_locator is None:
            axes_or_locator = ax
        if hasattr(axes_or_locator, 'get_subplotspec'):
            subplotspec = axes_or_locator.get_subplotspec()
            if subplotspec is not None:
                subplotspec = subplotspec.get_topmost_subplotspec()
                gs = subplotspec.get_gridspec()
                if grid_spec is not None:
                    if gs != grid_spec:
                        subplotspec = None
                elif gs.locally_modified_subplot_params():
                    subplotspec = None
        else:
            subplotspec = None
        subplotspec_list.append(subplotspec)
    return subplotspec_list

def get_tight_layout_figure(fig, axes_list, subplotspec_list, renderer, pad=1.08, h_pad=None, w_pad=None, rect=None):
    if False:
        print('Hello World!')
    '\n    Return subplot parameters for tight-layouted-figure with specified padding.\n\n    Parameters\n    ----------\n    fig : Figure\n    axes_list : list of Axes\n    subplotspec_list : list of `.SubplotSpec`\n        The subplotspecs of each axes.\n    renderer : renderer\n    pad : float\n        Padding between the figure edge and the edges of subplots, as a\n        fraction of the font size.\n    h_pad, w_pad : float\n        Padding (height/width) between edges of adjacent subplots.  Defaults to\n        *pad*.\n    rect : tuple (left, bottom, right, top), default: None.\n        rectangle in normalized figure coordinates\n        that the whole subplots area (including labels) will fit into.\n        Defaults to using the entire figure.\n\n    Returns\n    -------\n    subplotspec or None\n        subplotspec kwargs to be passed to `.Figure.subplots_adjust` or\n        None if tight_layout could not be accomplished.\n    '
    ss_to_subplots = {ss: [] for ss in subplotspec_list}
    for (ax, ss) in zip(axes_list, subplotspec_list):
        ss_to_subplots[ss].append(ax)
    if ss_to_subplots.pop(None, None):
        _api.warn_external('This figure includes Axes that are not compatible with tight_layout, so results might be incorrect.')
    if not ss_to_subplots:
        return {}
    subplot_list = list(ss_to_subplots.values())
    ax_bbox_list = [ss.get_position(fig) for ss in ss_to_subplots]
    max_nrows = max((ss.get_gridspec().nrows for ss in ss_to_subplots))
    max_ncols = max((ss.get_gridspec().ncols for ss in ss_to_subplots))
    span_pairs = []
    for ss in ss_to_subplots:
        (rows, cols) = ss.get_gridspec().get_geometry()
        (div_row, mod_row) = divmod(max_nrows, rows)
        (div_col, mod_col) = divmod(max_ncols, cols)
        if mod_row != 0:
            _api.warn_external('tight_layout not applied: number of rows in subplot specifications must be multiples of one another.')
            return {}
        if mod_col != 0:
            _api.warn_external('tight_layout not applied: number of columns in subplot specifications must be multiples of one another.')
            return {}
        span_pairs.append((slice(ss.rowspan.start * div_row, ss.rowspan.stop * div_row), slice(ss.colspan.start * div_col, ss.colspan.stop * div_col)))
    kwargs = _auto_adjust_subplotpars(fig, renderer, shape=(max_nrows, max_ncols), span_pairs=span_pairs, subplot_list=subplot_list, ax_bbox_list=ax_bbox_list, pad=pad, h_pad=h_pad, w_pad=w_pad)
    if rect is not None and kwargs is not None:
        (left, bottom, right, top) = rect
        if left is not None:
            left += kwargs['left']
        if bottom is not None:
            bottom += kwargs['bottom']
        if right is not None:
            right -= 1 - kwargs['right']
        if top is not None:
            top -= 1 - kwargs['top']
        kwargs = _auto_adjust_subplotpars(fig, renderer, shape=(max_nrows, max_ncols), span_pairs=span_pairs, subplot_list=subplot_list, ax_bbox_list=ax_bbox_list, pad=pad, h_pad=h_pad, w_pad=w_pad, rect=(left, bottom, right, top))
    return kwargs