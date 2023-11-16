"""
The :mod:`.axis_artist` module implements custom artists to draw axis elements
(axis lines and labels, tick lines and labels, grid lines).

Axis lines and labels and tick lines and labels are managed by the `AxisArtist`
class; grid lines are managed by the `GridlinesCollection` class.

There is one `AxisArtist` per Axis; it can be accessed through
the ``axis`` dictionary of the parent Axes (which should be a
`mpl_toolkits.axislines.Axes`), e.g. ``ax.axis["bottom"]``.

Children of the AxisArtist are accessed as attributes: ``.line`` and ``.label``
for the axis line and label, ``.major_ticks``, ``.major_ticklabels``,
``.minor_ticks``, ``.minor_ticklabels`` for the tick lines and labels (e.g.
``ax.axis["bottom"].line``).

Children properties (colors, fonts, line widths, etc.) can be set using
setters, e.g. ::

  # Make the major ticks of the bottom axis red.
  ax.axis["bottom"].major_ticks.set_color("red")

However, things like the locations of ticks, and their ticklabels need to be
changed from the side of the grid_helper.

axis_direction
--------------

`AxisArtist`, `AxisLabel`, `TickLabels` have an *axis_direction* attribute,
which adjusts the location, angle, etc. The *axis_direction* must be one of
"left", "right", "bottom", "top", and follows the Matplotlib convention for
rectangular axis.

For example, for the *bottom* axis (the left and right is relative to the
direction of the increasing coordinate),

* ticklabels and axislabel are on the right
* ticklabels and axislabel have text angle of 0
* ticklabels are baseline, center-aligned
* axislabel is top, center-aligned

The text angles are actually relative to (90 + angle of the direction to the
ticklabel), which gives 0 for bottom axis.

=================== ====== ======== ====== ========
Property            left   bottom   right  top
=================== ====== ======== ====== ========
ticklabel location  left   right    right  left
axislabel location  left   right    right  left
ticklabel angle     90     0        -90    180
axislabel angle     180    0        0      180
ticklabel va        center baseline center baseline
axislabel va        center top      center bottom
ticklabel ha        right  center   right  center
axislabel ha        right  center   right  center
=================== ====== ======== ====== ========

Ticks are by default direct opposite side of the ticklabels. To make ticks to
the same side of the ticklabels, ::

  ax.axis["bottom"].major_ticks.set_tick_out(True)

The following attributes can be customized (use the ``set_xxx`` methods):

* `Ticks`: ticksize, tick_out
* `TickLabels`: pad
* `AxisLabel`: pad
"""
from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import Affine2D, Bbox, IdentityTransform, ScaledTranslation
from .axisline_style import AxislineStyle

class AttributeCopier:

    def get_ref_artist(self):
        if False:
            while True:
                i = 10
        '\n        Return the underlying artist that actually defines some properties\n        (e.g., color) of this artist.\n        '
        raise RuntimeError('get_ref_artist must overridden')

    def get_attribute_from_ref_artist(self, attr_name):
        if False:
            while True:
                i = 10
        getter = methodcaller('get_' + attr_name)
        prop = getter(super())
        return getter(self.get_ref_artist()) if prop == 'auto' else prop

class Ticks(AttributeCopier, Line2D):
    """
    Ticks are derived from `.Line2D`, and note that ticks themselves
    are markers. Thus, you should use set_mec, set_mew, etc.

    To change the tick size (length), you need to use
    `set_ticksize`. To change the direction of the ticks (ticks are
    in opposite direction of ticklabels by default), use
    ``set_tick_out(False)``
    """

    def __init__(self, ticksize, tick_out=False, *, axis=None, **kwargs):
        if False:
            print('Hello World!')
        self._ticksize = ticksize
        self.locs_angles_labels = []
        self.set_tick_out(tick_out)
        self._axis = axis
        if self._axis is not None:
            if 'color' not in kwargs:
                kwargs['color'] = 'auto'
            if 'mew' not in kwargs and 'markeredgewidth' not in kwargs:
                kwargs['markeredgewidth'] = 'auto'
        Line2D.__init__(self, [0.0], [0.0], **kwargs)
        self.set_snap(True)

    def get_ref_artist(self):
        if False:
            while True:
                i = 10
        return self._axis.majorTicks[0].tick1line

    def set_color(self, color):
        if False:
            i = 10
            return i + 15
        if not cbook._str_equal(color, 'auto'):
            mcolors._check_color_like(color=color)
        self._color = color
        self.stale = True

    def get_color(self):
        if False:
            return 10
        return self.get_attribute_from_ref_artist('color')

    def get_markeredgecolor(self):
        if False:
            print('Hello World!')
        return self.get_attribute_from_ref_artist('markeredgecolor')

    def get_markeredgewidth(self):
        if False:
            print('Hello World!')
        return self.get_attribute_from_ref_artist('markeredgewidth')

    def set_tick_out(self, b):
        if False:
            print('Hello World!')
        'Set whether ticks are drawn inside or outside the axes.'
        self._tick_out = b

    def get_tick_out(self):
        if False:
            i = 10
            return i + 15
        'Return whether ticks are drawn inside or outside the axes.'
        return self._tick_out

    def set_ticksize(self, ticksize):
        if False:
            print('Hello World!')
        'Set length of the ticks in points.'
        self._ticksize = ticksize

    def get_ticksize(self):
        if False:
            i = 10
            return i + 15
        'Return length of the ticks in points.'
        return self._ticksize

    def set_locs_angles(self, locs_angles):
        if False:
            print('Hello World!')
        self.locs_angles = locs_angles
    _tickvert_path = Path([[0.0, 0.0], [1.0, 0.0]])

    def draw(self, renderer):
        if False:
            print('Hello World!')
        if not self.get_visible():
            return
        gc = renderer.new_gc()
        gc.set_foreground(self.get_markeredgecolor())
        gc.set_linewidth(self.get_markeredgewidth())
        gc.set_alpha(self._alpha)
        path_trans = self.get_transform()
        marker_transform = Affine2D().scale(renderer.points_to_pixels(self._ticksize))
        if self.get_tick_out():
            marker_transform.rotate_deg(180)
        for (loc, angle) in self.locs_angles:
            locs = path_trans.transform_non_affine(np.array([loc]))
            if self.axes and (not self.axes.viewLim.contains(*locs[0])):
                continue
            renderer.draw_markers(gc, self._tickvert_path, marker_transform + Affine2D().rotate_deg(angle), Path(locs), path_trans.get_affine())
        gc.restore()

class LabelBase(mtext.Text):
    """
    A base class for `.AxisLabel` and `.TickLabels`. The position and
    angle of the text are calculated by the offset_ref_angle,
    text_ref_angle, and offset_radius attributes.
    """

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        self.locs_angles_labels = []
        self._ref_angle = 0
        self._offset_radius = 0.0
        super().__init__(*args, **kwargs)
        self.set_rotation_mode('anchor')
        self._text_follow_ref_angle = True

    @property
    def _text_ref_angle(self):
        if False:
            return 10
        if self._text_follow_ref_angle:
            return self._ref_angle + 90
        else:
            return 0

    @property
    def _offset_ref_angle(self):
        if False:
            for i in range(10):
                print('nop')
        return self._ref_angle
    _get_opposite_direction = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}.__getitem__

    def draw(self, renderer):
        if False:
            print('Hello World!')
        if not self.get_visible():
            return
        tr = self.get_transform()
        angle_orig = self.get_rotation()
        theta = np.deg2rad(self._offset_ref_angle)
        dd = self._offset_radius
        (dx, dy) = (dd * np.cos(theta), dd * np.sin(theta))
        self.set_transform(tr + Affine2D().translate(dx, dy))
        self.set_rotation(self._text_ref_angle + angle_orig)
        super().draw(renderer)
        self.set_transform(tr)
        self.set_rotation(angle_orig)

    def get_window_extent(self, renderer=None):
        if False:
            for i in range(10):
                print('nop')
        if renderer is None:
            renderer = self.figure._get_renderer()
        tr = self.get_transform()
        angle_orig = self.get_rotation()
        theta = np.deg2rad(self._offset_ref_angle)
        dd = self._offset_radius
        (dx, dy) = (dd * np.cos(theta), dd * np.sin(theta))
        self.set_transform(tr + Affine2D().translate(dx, dy))
        self.set_rotation(self._text_ref_angle + angle_orig)
        bbox = super().get_window_extent(renderer).frozen()
        self.set_transform(tr)
        self.set_rotation(angle_orig)
        return bbox

class AxisLabel(AttributeCopier, LabelBase):
    """
    Axis label. Derived from `.Text`. The position of the text is updated
    in the fly, so changing text position has no effect. Otherwise, the
    properties can be changed as a normal `.Text`.

    To change the pad between tick labels and axis label, use `set_pad`.
    """

    def __init__(self, *args, axis_direction='bottom', axis=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._axis = axis
        self._pad = 5
        self._external_pad = 0
        LabelBase.__init__(self, *args, **kwargs)
        self.set_axis_direction(axis_direction)

    def set_pad(self, pad):
        if False:
            while True:
                i = 10
        '\n        Set the internal pad in points.\n\n        The actual pad will be the sum of the internal pad and the\n        external pad (the latter is set automatically by the `.AxisArtist`).\n\n        Parameters\n        ----------\n        pad : float\n            The internal pad in points.\n        '
        self._pad = pad

    def get_pad(self):
        if False:
            return 10
        '\n        Return the internal pad in points.\n\n        See `.set_pad` for more details.\n        '
        return self._pad

    def get_ref_artist(self):
        if False:
            print('Hello World!')
        return self._axis.get_label()

    def get_text(self):
        if False:
            for i in range(10):
                print('nop')
        t = super().get_text()
        if t == '__from_axes__':
            return self._axis.get_label().get_text()
        return self._text
    _default_alignments = dict(left=('bottom', 'center'), right=('top', 'center'), bottom=('top', 'center'), top=('bottom', 'center'))

    def set_default_alignment(self, d):
        if False:
            print('Hello World!')
        '\n        Set the default alignment. See `set_axis_direction` for details.\n\n        Parameters\n        ----------\n        d : {"left", "bottom", "right", "top"}\n        '
        (va, ha) = _api.check_getitem(self._default_alignments, d=d)
        self.set_va(va)
        self.set_ha(ha)
    _default_angles = dict(left=180, right=0, bottom=0, top=180)

    def set_default_angle(self, d):
        if False:
            print('Hello World!')
        '\n        Set the default angle. See `set_axis_direction` for details.\n\n        Parameters\n        ----------\n        d : {"left", "bottom", "right", "top"}\n        '
        self.set_rotation(_api.check_getitem(self._default_angles, d=d))

    def set_axis_direction(self, d):
        if False:
            print('Hello World!')
        '\n        Adjust the text angle and text alignment of axis label\n        according to the matplotlib convention.\n\n        =====================    ========== ========= ========== ==========\n        Property                 left       bottom    right      top\n        =====================    ========== ========= ========== ==========\n        axislabel angle          180        0         0          180\n        axislabel va             center     top       center     bottom\n        axislabel ha             right      center    right      center\n        =====================    ========== ========= ========== ==========\n\n        Note that the text angles are actually relative to (90 + angle\n        of the direction to the ticklabel), which gives 0 for bottom\n        axis.\n\n        Parameters\n        ----------\n        d : {"left", "bottom", "right", "top"}\n        '
        self.set_default_alignment(d)
        self.set_default_angle(d)

    def get_color(self):
        if False:
            i = 10
            return i + 15
        return self.get_attribute_from_ref_artist('color')

    def draw(self, renderer):
        if False:
            print('Hello World!')
        if not self.get_visible():
            return
        self._offset_radius = self._external_pad + renderer.points_to_pixels(self.get_pad())
        super().draw(renderer)

    def get_window_extent(self, renderer=None):
        if False:
            for i in range(10):
                print('nop')
        if renderer is None:
            renderer = self.figure._get_renderer()
        if not self.get_visible():
            return
        r = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r
        bb = super().get_window_extent(renderer)
        return bb

class TickLabels(AxisLabel):
    """
    Tick labels. While derived from `.Text`, this single artist draws all
    ticklabels. As in `.AxisLabel`, the position of the text is updated
    in the fly, so changing text position has no effect. Otherwise,
    the properties can be changed as a normal `.Text`. Unlike the
    ticklabels of the mainline Matplotlib, properties of a single
    ticklabel alone cannot be modified.

    To change the pad between ticks and ticklabels, use `~.AxisLabel.set_pad`.
    """

    def __init__(self, *, axis_direction='bottom', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(**kwargs)
        self.set_axis_direction(axis_direction)
        self._axislabel_pad = 0

    def get_ref_artist(self):
        if False:
            print('Hello World!')
        return self._axis.get_ticklabels()[0]

    def set_axis_direction(self, label_direction):
        if False:
            print('Hello World!')
        '\n        Adjust the text angle and text alignment of ticklabels\n        according to the Matplotlib convention.\n\n        The *label_direction* must be one of [left, right, bottom, top].\n\n        =====================    ========== ========= ========== ==========\n        Property                 left       bottom    right      top\n        =====================    ========== ========= ========== ==========\n        ticklabel angle          90         0         -90        180\n        ticklabel va             center     baseline  center     baseline\n        ticklabel ha             right      center    right      center\n        =====================    ========== ========= ========== ==========\n\n        Note that the text angles are actually relative to (90 + angle\n        of the direction to the ticklabel), which gives 0 for bottom\n        axis.\n\n        Parameters\n        ----------\n        label_direction : {"left", "bottom", "right", "top"}\n\n        '
        self.set_default_alignment(label_direction)
        self.set_default_angle(label_direction)
        self._axis_direction = label_direction

    def invert_axis_direction(self):
        if False:
            print('Hello World!')
        label_direction = self._get_opposite_direction(self._axis_direction)
        self.set_axis_direction(label_direction)

    def _get_ticklabels_offsets(self, renderer, label_direction):
        if False:
            while True:
                i = 10
        "\n        Calculate the ticklabel offsets from the tick and their total heights.\n\n        The offset only takes account the offset due to the vertical alignment\n        of the ticklabels: if axis direction is bottom and va is 'top', it will\n        return 0; if va is 'baseline', it will return (height-descent).\n        "
        whd_list = self.get_texts_widths_heights_descents(renderer)
        if not whd_list:
            return (0, 0)
        r = 0
        (va, ha) = (self.get_va(), self.get_ha())
        if label_direction == 'left':
            pad = max((w for (w, h, d) in whd_list))
            if ha == 'left':
                r = pad
            elif ha == 'center':
                r = 0.5 * pad
        elif label_direction == 'right':
            pad = max((w for (w, h, d) in whd_list))
            if ha == 'right':
                r = pad
            elif ha == 'center':
                r = 0.5 * pad
        elif label_direction == 'bottom':
            pad = max((h for (w, h, d) in whd_list))
            if va == 'bottom':
                r = pad
            elif va == 'center':
                r = 0.5 * pad
            elif va == 'baseline':
                max_ascent = max((h - d for (w, h, d) in whd_list))
                max_descent = max((d for (w, h, d) in whd_list))
                r = max_ascent
                pad = max_ascent + max_descent
        elif label_direction == 'top':
            pad = max((h for (w, h, d) in whd_list))
            if va == 'top':
                r = pad
            elif va == 'center':
                r = 0.5 * pad
            elif va == 'baseline':
                max_ascent = max((h - d for (w, h, d) in whd_list))
                max_descent = max((d for (w, h, d) in whd_list))
                r = max_descent
                pad = max_ascent + max_descent
        return (r, pad)
    _default_alignments = dict(left=('center', 'right'), right=('center', 'left'), bottom=('baseline', 'center'), top=('baseline', 'center'))
    _default_angles = dict(left=90, right=-90, bottom=0, top=180)

    def draw(self, renderer):
        if False:
            print('Hello World!')
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return
        (r, total_width) = self._get_ticklabels_offsets(renderer, self._axis_direction)
        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad
        for ((x, y), a, l) in self._locs_angles_labels:
            if not l.strip():
                continue
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            LabelBase.draw(self, renderer)
        self._axislabel_pad = total_width + pad

    def set_locs_angles_labels(self, locs_angles_labels):
        if False:
            print('Hello World!')
        self._locs_angles_labels = locs_angles_labels

    def get_window_extents(self, renderer=None):
        if False:
            return 10
        if renderer is None:
            renderer = self.figure._get_renderer()
        if not self.get_visible():
            self._axislabel_pad = self._external_pad
            return []
        bboxes = []
        (r, total_width) = self._get_ticklabels_offsets(renderer, self._axis_direction)
        pad = self._external_pad + renderer.points_to_pixels(self.get_pad())
        self._offset_radius = r + pad
        for ((x, y), a, l) in self._locs_angles_labels:
            self._ref_angle = a
            self.set_x(x)
            self.set_y(y)
            self.set_text(l)
            bb = LabelBase.get_window_extent(self, renderer)
            bboxes.append(bb)
        self._axislabel_pad = total_width + pad
        return bboxes

    def get_texts_widths_heights_descents(self, renderer):
        if False:
            while True:
                i = 10
        '\n        Return a list of ``(width, height, descent)`` tuples for ticklabels.\n\n        Empty labels are left out.\n        '
        whd_list = []
        for (_loc, _angle, label) in self._locs_angles_labels:
            if not label.strip():
                continue
            (clean_line, ismath) = self._preprocess_math(label)
            whd = renderer.get_text_width_height_descent(clean_line, self._fontproperties, ismath=ismath)
            whd_list.append(whd)
        return whd_list

class GridlinesCollection(LineCollection):

    def __init__(self, *args, which='major', axis='both', **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Collection of grid lines.\n\n        Parameters\n        ----------\n        which : {"major", "minor"}\n            Which grid to consider.\n        axis : {"both", "x", "y"}\n            Which axis to consider.\n        *args, **kwargs\n            Passed to `.LineCollection`.\n        '
        self._which = which
        self._axis = axis
        super().__init__(*args, **kwargs)
        self.set_grid_helper(None)

    def set_which(self, which):
        if False:
            while True:
                i = 10
        '\n        Select major or minor grid lines.\n\n        Parameters\n        ----------\n        which : {"major", "minor"}\n        '
        self._which = which

    def set_axis(self, axis):
        if False:
            for i in range(10):
                print('nop')
        '\n        Select axis.\n\n        Parameters\n        ----------\n        axis : {"both", "x", "y"}\n        '
        self._axis = axis

    def set_grid_helper(self, grid_helper):
        if False:
            print('Hello World!')
        '\n        Set grid helper.\n\n        Parameters\n        ----------\n        grid_helper : `.GridHelperBase` subclass\n        '
        self._grid_helper = grid_helper

    def draw(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        if self._grid_helper is not None:
            self._grid_helper.update_lim(self.axes)
            gl = self._grid_helper.get_gridlines(self._which, self._axis)
            self.set_segments([np.transpose(l) for l in gl])
        super().draw(renderer)

class AxisArtist(martist.Artist):
    """
    An artist which draws axis (a line along which the n-th axes coord
    is constant) line, ticks, tick labels, and axis label.
    """
    zorder = 2.5

    @property
    def LABELPAD(self):
        if False:
            print('Hello World!')
        return self.label.get_pad()

    @LABELPAD.setter
    def LABELPAD(self, v):
        if False:
            print('Hello World!')
        self.label.set_pad(v)

    def __init__(self, axes, helper, offset=None, axis_direction='bottom', **kwargs):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        axes : `mpl_toolkits.axisartist.axislines.Axes`\n        helper : `~mpl_toolkits.axisartist.axislines.AxisArtistHelper`\n        '
        super().__init__(**kwargs)
        self.axes = axes
        self._axis_artist_helper = helper
        if offset is None:
            offset = (0, 0)
        self.offset_transform = ScaledTranslation(*offset, Affine2D().scale(1 / 72) + self.axes.figure.dpi_scale_trans)
        if axis_direction in ['left', 'right']:
            self.axis = axes.yaxis
        else:
            self.axis = axes.xaxis
        self._axisline_style = None
        self._axis_direction = axis_direction
        self._init_line()
        self._init_ticks(**kwargs)
        self._init_offsetText(axis_direction)
        self._init_label()
        self._ticklabel_add_angle = 0.0
        self._axislabel_add_angle = 0.0
        self.set_axis_direction(axis_direction)

    def set_axis_direction(self, axis_direction):
        if False:
            while True:
                i = 10
        '\n        Adjust the direction, text angle, and text alignment of tick labels\n        and axis labels following the Matplotlib convention for the rectangle\n        axes.\n\n        The *axis_direction* must be one of [left, right, bottom, top].\n\n        =====================    ========== ========= ========== ==========\n        Property                 left       bottom    right      top\n        =====================    ========== ========= ========== ==========\n        ticklabel direction      "-"        "+"       "+"        "-"\n        axislabel direction      "-"        "+"       "+"        "-"\n        ticklabel angle          90         0         -90        180\n        ticklabel va             center     baseline  center     baseline\n        ticklabel ha             right      center    right      center\n        axislabel angle          180        0         0          180\n        axislabel va             center     top       center     bottom\n        axislabel ha             right      center    right      center\n        =====================    ========== ========= ========== ==========\n\n        Note that the direction "+" and "-" are relative to the direction of\n        the increasing coordinate. Also, the text angles are actually\n        relative to (90 + angle of the direction to the ticklabel),\n        which gives 0 for bottom axis.\n\n        Parameters\n        ----------\n        axis_direction : {"left", "bottom", "right", "top"}\n        '
        self.major_ticklabels.set_axis_direction(axis_direction)
        self.label.set_axis_direction(axis_direction)
        self._axis_direction = axis_direction
        if axis_direction in ['left', 'top']:
            self.set_ticklabel_direction('-')
            self.set_axislabel_direction('-')
        else:
            self.set_ticklabel_direction('+')
            self.set_axislabel_direction('+')

    def set_ticklabel_direction(self, tick_direction):
        if False:
            i = 10
            return i + 15
        '\n        Adjust the direction of the tick labels.\n\n        Note that the *tick_direction*\\s \'+\' and \'-\' are relative to the\n        direction of the increasing coordinate.\n\n        Parameters\n        ----------\n        tick_direction : {"+", "-"}\n        '
        self._ticklabel_add_angle = _api.check_getitem({'+': 0, '-': 180}, tick_direction=tick_direction)

    def invert_ticklabel_direction(self):
        if False:
            return 10
        self._ticklabel_add_angle = (self._ticklabel_add_angle + 180) % 360
        self.major_ticklabels.invert_axis_direction()
        self.minor_ticklabels.invert_axis_direction()

    def set_axislabel_direction(self, label_direction):
        if False:
            for i in range(10):
                print('nop')
        '\n        Adjust the direction of the axis label.\n\n        Note that the *label_direction*\\s \'+\' and \'-\' are relative to the\n        direction of the increasing coordinate.\n\n        Parameters\n        ----------\n        label_direction : {"+", "-"}\n        '
        self._axislabel_add_angle = _api.check_getitem({'+': 0, '-': 180}, label_direction=label_direction)

    def get_transform(self):
        if False:
            print('Hello World!')
        return self.axes.transAxes + self.offset_transform

    def get_helper(self):
        if False:
            i = 10
            return i + 15
        '\n        Return axis artist helper instance.\n        '
        return self._axis_artist_helper

    def set_axisline_style(self, axisline_style=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Set the axisline style.\n\n        The new style is completely defined by the passed attributes. Existing\n        style attributes are forgotten.\n\n        Parameters\n        ----------\n        axisline_style : str or None\n            The line style, e.g. \'->\', optionally followed by a comma-separated\n            list of attributes. Alternatively, the attributes can be provided\n            as keywords.\n\n            If *None* this returns a string containing the available styles.\n\n        Examples\n        --------\n        The following two commands are equal:\n\n        >>> set_axisline_style("->,size=1.5")\n        >>> set_axisline_style("->", size=1.5)\n        '
        if axisline_style is None:
            return AxislineStyle.pprint_styles()
        if isinstance(axisline_style, AxislineStyle._Base):
            self._axisline_style = axisline_style
        else:
            self._axisline_style = AxislineStyle(axisline_style, **kwargs)
        self._init_line()

    def get_axisline_style(self):
        if False:
            return 10
        'Return the current axisline style.'
        return self._axisline_style

    def _init_line(self):
        if False:
            return 10
        '\n        Initialize the *line* artist that is responsible to draw the axis line.\n        '
        tran = self._axis_artist_helper.get_line_transform(self.axes) + self.offset_transform
        axisline_style = self.get_axisline_style()
        if axisline_style is None:
            self.line = PathPatch(self._axis_artist_helper.get_line(self.axes), color=mpl.rcParams['axes.edgecolor'], fill=False, linewidth=mpl.rcParams['axes.linewidth'], capstyle=mpl.rcParams['lines.solid_capstyle'], joinstyle=mpl.rcParams['lines.solid_joinstyle'], transform=tran)
        else:
            self.line = axisline_style(self, transform=tran)

    def _draw_line(self, renderer):
        if False:
            while True:
                i = 10
        self.line.set_path(self._axis_artist_helper.get_line(self.axes))
        if self.get_axisline_style() is not None:
            self.line.set_line_mutation_scale(self.major_ticklabels.get_size())
        self.line.draw(renderer)

    def _init_ticks(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        axis_name = self.axis.axis_name
        trans = self._axis_artist_helper.get_tick_transform(self.axes) + self.offset_transform
        self.major_ticks = Ticks(kwargs.get('major_tick_size', mpl.rcParams[f'{axis_name}tick.major.size']), axis=self.axis, transform=trans)
        self.minor_ticks = Ticks(kwargs.get('minor_tick_size', mpl.rcParams[f'{axis_name}tick.minor.size']), axis=self.axis, transform=trans)
        size = mpl.rcParams[f'{axis_name}tick.labelsize']
        self.major_ticklabels = TickLabels(axis=self.axis, axis_direction=self._axis_direction, figure=self.axes.figure, transform=trans, fontsize=size, pad=kwargs.get('major_tick_pad', mpl.rcParams[f'{axis_name}tick.major.pad']))
        self.minor_ticklabels = TickLabels(axis=self.axis, axis_direction=self._axis_direction, figure=self.axes.figure, transform=trans, fontsize=size, pad=kwargs.get('minor_tick_pad', mpl.rcParams[f'{axis_name}tick.minor.pad']))

    def _get_tick_info(self, tick_iter):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a pair of:\n\n        - list of locs and angles for ticks\n        - list of locs, angles and labels for ticklabels.\n        '
        ticks_loc_angle = []
        ticklabels_loc_angle_label = []
        ticklabel_add_angle = self._ticklabel_add_angle
        for (loc, angle_normal, angle_tangent, label) in tick_iter:
            angle_label = angle_tangent - 90 + ticklabel_add_angle
            angle_tick = angle_normal if 90 <= (angle_label - angle_normal) % 360 <= 270 else angle_normal + 180
            ticks_loc_angle.append([loc, angle_tick])
            ticklabels_loc_angle_label.append([loc, angle_label, label])
        return (ticks_loc_angle, ticklabels_loc_angle_label)

    def _update_ticks(self, renderer=None):
        if False:
            return 10
        if renderer is None:
            renderer = self.figure._get_renderer()
        dpi_cor = renderer.points_to_pixels(1.0)
        if self.major_ticks.get_visible() and self.major_ticks.get_tick_out():
            ticklabel_pad = self.major_ticks._ticksize * dpi_cor
            self.major_ticklabels._external_pad = ticklabel_pad
            self.minor_ticklabels._external_pad = ticklabel_pad
        else:
            self.major_ticklabels._external_pad = 0
            self.minor_ticklabels._external_pad = 0
        (majortick_iter, minortick_iter) = self._axis_artist_helper.get_tick_iterators(self.axes)
        (tick_loc_angle, ticklabel_loc_angle_label) = self._get_tick_info(majortick_iter)
        self.major_ticks.set_locs_angles(tick_loc_angle)
        self.major_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)
        (tick_loc_angle, ticklabel_loc_angle_label) = self._get_tick_info(minortick_iter)
        self.minor_ticks.set_locs_angles(tick_loc_angle)
        self.minor_ticklabels.set_locs_angles_labels(ticklabel_loc_angle_label)

    def _draw_ticks(self, renderer):
        if False:
            return 10
        self._update_ticks(renderer)
        self.major_ticks.draw(renderer)
        self.major_ticklabels.draw(renderer)
        self.minor_ticks.draw(renderer)
        self.minor_ticklabels.draw(renderer)
        if self.major_ticklabels.get_visible() or self.minor_ticklabels.get_visible():
            self._draw_offsetText(renderer)
    _offsetText_pos = dict(left=(0, 1, 'bottom', 'right'), right=(1, 1, 'bottom', 'left'), bottom=(1, 0, 'top', 'right'), top=(1, 1, 'bottom', 'right'))

    def _init_offsetText(self, direction):
        if False:
            return 10
        (x, y, va, ha) = self._offsetText_pos[direction]
        self.offsetText = mtext.Annotation('', xy=(x, y), xycoords='axes fraction', xytext=(0, 0), textcoords='offset points', color=mpl.rcParams['xtick.color'], horizontalalignment=ha, verticalalignment=va)
        self.offsetText.set_transform(IdentityTransform())
        self.axes._set_artist_props(self.offsetText)

    def _update_offsetText(self):
        if False:
            print('Hello World!')
        self.offsetText.set_text(self.axis.major.formatter.get_offset())
        self.offsetText.set_size(self.major_ticklabels.get_size())
        offset = self.major_ticklabels.get_pad() + self.major_ticklabels.get_size() + 2
        self.offsetText.xyann = (0, offset)

    def _draw_offsetText(self, renderer):
        if False:
            print('Hello World!')
        self._update_offsetText()
        self.offsetText.draw(renderer)

    def _init_label(self, **kwargs):
        if False:
            while True:
                i = 10
        tr = self._axis_artist_helper.get_axislabel_transform(self.axes) + self.offset_transform
        self.label = AxisLabel(0, 0, '__from_axes__', color='auto', fontsize=kwargs.get('labelsize', mpl.rcParams['axes.labelsize']), fontweight=mpl.rcParams['axes.labelweight'], axis=self.axis, transform=tr, axis_direction=self._axis_direction)
        self.label.set_figure(self.axes.figure)
        labelpad = kwargs.get('labelpad', 5)
        self.label.set_pad(labelpad)

    def _update_label(self, renderer):
        if False:
            print('Hello World!')
        if not self.label.get_visible():
            return
        if self._ticklabel_add_angle != self._axislabel_add_angle:
            if self.major_ticks.get_visible() and (not self.major_ticks.get_tick_out()) or (self.minor_ticks.get_visible() and (not self.major_ticks.get_tick_out())):
                axislabel_pad = self.major_ticks._ticksize
            else:
                axislabel_pad = 0
        else:
            axislabel_pad = max(self.major_ticklabels._axislabel_pad, self.minor_ticklabels._axislabel_pad)
        self.label._external_pad = axislabel_pad
        (xy, angle_tangent) = self._axis_artist_helper.get_axislabel_pos_angle(self.axes)
        if xy is None:
            return
        angle_label = angle_tangent - 90
        (x, y) = xy
        self.label._ref_angle = angle_label + self._axislabel_add_angle
        self.label.set(x=x, y=y)

    def _draw_label(self, renderer):
        if False:
            for i in range(10):
                print('nop')
        self._update_label(renderer)
        self.label.draw(renderer)

    def set_label(self, s):
        if False:
            while True:
                i = 10
        self.label.set_text(s)

    def get_tightbbox(self, renderer=None):
        if False:
            while True:
                i = 10
        if not self.get_visible():
            return
        self._axis_artist_helper.update_lim(self.axes)
        self._update_ticks(renderer)
        self._update_label(renderer)
        self.line.set_path(self._axis_artist_helper.get_line(self.axes))
        if self.get_axisline_style() is not None:
            self.line.set_line_mutation_scale(self.major_ticklabels.get_size())
        bb = [*self.major_ticklabels.get_window_extents(renderer), *self.minor_ticklabels.get_window_extents(renderer), self.label.get_window_extent(renderer), self.offsetText.get_window_extent(renderer), self.line.get_window_extent(renderer)]
        bb = [b for b in bb if b and (b.width != 0 or b.height != 0)]
        if bb:
            _bbox = Bbox.union(bb)
            return _bbox
        else:
            return None

    @martist.allow_rasterization
    def draw(self, renderer):
        if False:
            print('Hello World!')
        if not self.get_visible():
            return
        renderer.open_group(__name__, gid=self.get_gid())
        self._axis_artist_helper.update_lim(self.axes)
        self._draw_ticks(renderer)
        self._draw_line(renderer)
        self._draw_label(renderer)
        renderer.close_group(__name__)

    def toggle(self, all=None, ticks=None, ticklabels=None, label=None):
        if False:
            return 10
        '\n        Toggle visibility of ticks, ticklabels, and (axis) label.\n        To turn all off, ::\n\n          axis.toggle(all=False)\n\n        To turn all off but ticks on ::\n\n          axis.toggle(all=False, ticks=True)\n\n        To turn all on but (axis) label off ::\n\n          axis.toggle(all=True, label=False)\n\n        '
        if all:
            (_ticks, _ticklabels, _label) = (True, True, True)
        elif all is not None:
            (_ticks, _ticklabels, _label) = (False, False, False)
        else:
            (_ticks, _ticklabels, _label) = (None, None, None)
        if ticks is not None:
            _ticks = ticks
        if ticklabels is not None:
            _ticklabels = ticklabels
        if label is not None:
            _label = label
        if _ticks is not None:
            self.major_ticks.set_visible(_ticks)
            self.minor_ticks.set_visible(_ticks)
        if _ticklabels is not None:
            self.major_ticklabels.set_visible(_ticklabels)
            self.minor_ticklabels.set_visible(_ticklabels)
        if _label is not None:
            self.label.set_visible(_label)