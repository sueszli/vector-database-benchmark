import numpy as np
from . import utils

def dot_plot(points, intervals=None, lines=None, sections=None, styles=None, marker_props=None, line_props=None, split_names=None, section_order=None, line_order=None, stacked=False, styles_order=None, striped=False, horizontal=True, show_names='both', fmt_left_name=None, fmt_right_name=None, show_section_titles=None, ax=None):
    if False:
        i = 10
        return i + 15
    '\n    Dot plotting (also known as forest and blobbogram).\n\n    Produce a dotplot similar in style to those in Cleveland\'s\n    "Visualizing Data" book ([1]_).  These are also known as "forest plots".\n\n    Parameters\n    ----------\n    points : array_like\n        The quantitative values to be plotted as markers.\n    intervals : array_like\n        The intervals to be plotted around the points.  The elements\n        of `intervals` are either scalars or sequences of length 2.  A\n        scalar indicates the half width of a symmetric interval.  A\n        sequence of length 2 contains the left and right half-widths\n        (respectively) of a nonsymmetric interval.  If None, no\n        intervals are drawn.\n    lines : array_like\n        A grouping variable indicating which points/intervals are\n        drawn on a common line.  If None, each point/interval appears\n        on its own line.\n    sections : array_like\n        A grouping variable indicating which lines are grouped into\n        sections.  If None, everything is drawn in a single section.\n    styles : array_like\n        A grouping label defining the plotting style of the markers\n        and intervals.\n    marker_props : dict\n        A dictionary mapping style codes (the values in `styles`) to\n        dictionaries defining key/value pairs to be passed as keyword\n        arguments to `plot` when plotting markers.  Useful keyword\n        arguments are "color", "marker", and "ms" (marker size).\n    line_props : dict\n        A dictionary mapping style codes (the values in `styles`) to\n        dictionaries defining key/value pairs to be passed as keyword\n        arguments to `plot` when plotting interval lines.  Useful\n        keyword arguments are "color", "linestyle", "solid_capstyle",\n        and "linewidth".\n    split_names : str\n        If not None, this is used to split the values of `lines` into\n        substrings that are drawn in the left and right margins,\n        respectively.  If None, the values of `lines` are drawn in the\n        left margin.\n    section_order : array_like\n        The section labels in the order in which they appear in the\n        dotplot.\n    line_order : array_like\n        The line labels in the order in which they appear in the\n        dotplot.\n    stacked : bool\n        If True, when multiple points or intervals are drawn on the\n        same line, they are offset from each other.\n    styles_order : array_like\n        If stacked=True, this is the order in which the point styles\n        on a given line are drawn from top to bottom (if horizontal\n        is True) or from left to right (if horizontal is False).  If\n        None (default), the order is lexical.\n    striped : bool\n        If True, every other line is enclosed in a shaded box.\n    horizontal : bool\n        If True (default), the lines are drawn horizontally, otherwise\n        they are drawn vertically.\n    show_names : str\n        Determines whether labels (names) are shown in the left and/or\n        right margins (top/bottom margins if `horizontal` is True).\n        If `both`, labels are drawn in both margins, if \'left\', labels\n        are drawn in the left or top margin.  If `right`, labels are\n        drawn in the right or bottom margin.\n    fmt_left_name : callable\n        The left/top margin names are passed through this function\n        before drawing on the plot.\n    fmt_right_name : callable\n        The right/bottom marginnames are passed through this function\n        before drawing on the plot.\n    show_section_titles : bool or None\n        If None, section titles are drawn only if there is more than\n        one section.  If False/True, section titles are never/always\n        drawn, respectively.\n    ax : matplotlib.axes\n        The axes on which the dotplot is drawn.  If None, a new axes\n        is created.\n\n    Returns\n    -------\n    fig : Figure\n        The figure given by `ax.figure` or a new instance.\n\n    Notes\n    -----\n    `points`, `intervals`, `lines`, `sections`, `styles` must all have\n    the same length whenever present.\n\n    References\n    ----------\n    .. [1] Cleveland, William S. (1993). "Visualizing Data". Hobart Press.\n    .. [2] Jacoby, William G. (2006) "The Dot Plot: A Graphical Display\n       for Labeled Quantitative Values." The Political Methodologist\n       14(1): 6-14.\n\n    Examples\n    --------\n    This is a simple dotplot with one point per line:\n\n    >>> dot_plot(points=point_values)\n\n    This dotplot has labels on the lines (if elements in\n    `label_values` are repeated, the corresponding points appear on\n    the same line):\n\n    >>> dot_plot(points=point_values, lines=label_values)\n    '
    import matplotlib.transforms as transforms
    (fig, ax) = utils.create_mpl_ax(ax)
    points = np.asarray(points)
    asarray_or_none = lambda x: None if x is None else np.asarray(x)
    intervals = asarray_or_none(intervals)
    lines = asarray_or_none(lines)
    sections = asarray_or_none(sections)
    styles = asarray_or_none(styles)
    npoint = len(points)
    if lines is None:
        lines = np.arange(npoint)
    if sections is None:
        sections = np.zeros(npoint)
    if styles is None:
        styles = np.zeros(npoint)
    section_title_space = 0.5
    nsect = len(set(sections))
    if section_order is not None:
        nsect = len(set(section_order))
    if show_section_titles is False:
        draw_section_titles = False
        nsect_title = 0
    elif show_section_titles is True:
        draw_section_titles = True
        nsect_title = nsect
    else:
        draw_section_titles = nsect > 1
        nsect_title = nsect if nsect > 1 else 0
    section_space_total = section_title_space * nsect_title
    ax.set_xmargin(0.02)
    ax.set_ymargin(0.02)
    if section_order is None:
        lines0 = list(set(sections))
        lines0.sort()
    else:
        lines0 = section_order
    if line_order is None:
        lines1 = list(set(lines))
        lines1.sort()
    else:
        lines1 = line_order
    lines_map = {}
    for i in range(npoint):
        if section_order is not None and sections[i] not in section_order:
            continue
        if line_order is not None and lines[i] not in line_order:
            continue
        ky = (sections[i], lines[i])
        if ky not in lines_map:
            lines_map[ky] = []
        lines_map[ky].append(i)
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    (awidth, aheight) = (bbox.width, bbox.height)
    nrows = len(lines_map)
    (bottom, top) = (0, 1)
    if horizontal:
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    else:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    title_space_axes = section_title_space / aheight
    if horizontal:
        dpos = (top - bottom - nsect_title * title_space_axes) / float(nrows)
    else:
        dpos = (top - bottom) / float(nrows)
    if styles_order is not None:
        style_codes = styles_order
    else:
        style_codes = list(set(styles))
        style_codes.sort()
    if horizontal:
        style_codes = style_codes[::-1]
    nval = len(style_codes)
    if nval > 1:
        stackd = dpos / (2.5 * (float(nval) - 1))
    else:
        stackd = 0.0
    style_codes_map = {x: style_codes.index(x) for x in style_codes}
    colors = ['r', 'g', 'b', 'y', 'k', 'purple', 'orange']
    if marker_props is None:
        marker_props = {x: {} for x in style_codes}
    for j in range(nval):
        sc = style_codes[j]
        if 'color' not in marker_props[sc]:
            marker_props[sc]['color'] = colors[j % len(colors)]
        if 'marker' not in marker_props[sc]:
            marker_props[sc]['marker'] = 'o'
        if 'ms' not in marker_props[sc]:
            marker_props[sc]['ms'] = 10 if stackd == 0 else 6
    if line_props is None:
        line_props = {x: {} for x in style_codes}
    for j in range(nval):
        sc = style_codes[j]
        if 'color' not in line_props[sc]:
            line_props[sc]['color'] = 'grey'
        if 'linewidth' not in line_props[sc]:
            line_props[sc]['linewidth'] = 2 if stackd > 0 else 8
    if horizontal:
        pos = top - dpos / 2 if nsect == 1 else top
    else:
        pos = bottom + dpos / 2
    labeled = set()
    ticks = []
    for k0 in lines0:
        if draw_section_titles:
            if horizontal:
                y0 = pos + dpos / 2 if k0 == lines0[0] else pos
                ax.fill_between((0, 1), (y0, y0), (pos - 0.7 * title_space_axes, pos - 0.7 * title_space_axes), color='darkgrey', transform=ax.transAxes, zorder=1)
                txt = ax.text(0.5, pos - 0.35 * title_space_axes, k0, horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
                txt.set_fontweight('bold')
                pos -= title_space_axes
            else:
                m = len([k for k in lines_map if k[0] == k0])
                ax.fill_between((pos - dpos / 2 + 0.01, pos + (m - 1) * dpos + dpos / 2 - 0.01), (1.01, 1.01), (1.06, 1.06), color='darkgrey', transform=ax.transAxes, zorder=1, clip_on=False)
                txt = ax.text(pos + (m - 1) * dpos / 2, 1.02, k0, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes)
                txt.set_fontweight('bold')
        jrow = 0
        for k1 in lines1:
            if (k0, k1) not in lines_map:
                continue
            if horizontal:
                ax.axhline(pos, color='grey')
            else:
                ax.axvline(pos, color='grey')
            if split_names is not None:
                us = k1.split(split_names)
                if len(us) >= 2:
                    (left_label, right_label) = (us[0], us[1])
                else:
                    (left_label, right_label) = (k1, None)
            else:
                (left_label, right_label) = (k1, None)
            if fmt_left_name is not None:
                left_label = fmt_left_name(left_label)
            if fmt_right_name is not None:
                right_label = fmt_right_name(right_label)
            if striped and jrow % 2 == 0:
                if horizontal:
                    ax.fill_between((0, 1), (pos - dpos / 2, pos - dpos / 2), (pos + dpos / 2, pos + dpos / 2), color='lightgrey', transform=ax.transAxes, zorder=0)
                else:
                    ax.fill_between((pos - dpos / 2, pos + dpos / 2), (0, 0), (1, 1), color='lightgrey', transform=ax.transAxes, zorder=0)
            jrow += 1
            if show_names.lower() in ('left', 'both'):
                if horizontal:
                    ax.text(-0.1 / awidth, pos, left_label, horizontalalignment='right', verticalalignment='center', transform=ax.transAxes, family='monospace')
                else:
                    ax.text(pos, -0.1 / aheight, left_label, horizontalalignment='center', verticalalignment='top', transform=ax.transAxes, family='monospace')
            if show_names.lower() in ('right', 'both'):
                if right_label is not None:
                    if horizontal:
                        ax.text(1 + 0.1 / awidth, pos, right_label, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes, family='monospace')
                    else:
                        ax.text(pos, 1 + 0.1 / aheight, right_label, horizontalalignment='center', verticalalignment='bottom', transform=ax.transAxes, family='monospace')
            ticks.append(pos)
            for (ji, jp) in enumerate(lines_map[k0, k1]):
                yo = 0
                if stacked:
                    yo = -dpos / 5 + style_codes_map[styles[jp]] * stackd
                pt = points[jp]
                if intervals is not None:
                    if np.isscalar(intervals[jp]):
                        (lcb, ucb) = (pt - intervals[jp], pt + intervals[jp])
                    else:
                        (lcb, ucb) = (pt - intervals[jp][0], pt + intervals[jp][1])
                    if horizontal:
                        ax.plot([lcb, ucb], [pos + yo, pos + yo], '-', transform=trans, **line_props[styles[jp]])
                    else:
                        ax.plot([pos + yo, pos + yo], [lcb, ucb], '-', transform=trans, **line_props[styles[jp]])
                sl = styles[jp]
                sll = sl if sl not in labeled else None
                labeled.add(sl)
                if horizontal:
                    ax.plot([pt], [pos + yo], ls='None', transform=trans, label=sll, **marker_props[sl])
                else:
                    ax.plot([pos + yo], [pt], ls='None', transform=trans, label=sll, **marker_props[sl])
            if horizontal:
                pos -= dpos
            else:
                pos += dpos
    if horizontal:
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('none')
        ax.set_yticklabels([])
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position(('axes', -0.1 / aheight))
        ax.set_ylim(0, 1)
        ax.yaxis.set_ticks(ticks)
        ax.autoscale_view(scaley=False, tight=True)
    else:
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('none')
        ax.set_xticklabels([])
        ax.spines['bottom'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_position(('axes', -0.1 / awidth))
        ax.set_xlim(0, 1)
        ax.xaxis.set_ticks(ticks)
        ax.autoscale_view(scalex=False, tight=True)
    return fig