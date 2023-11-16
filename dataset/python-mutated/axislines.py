"""
Axislines includes modified implementation of the Axes class. The
biggest difference is that the artists responsible for drawing the axis spine,
ticks, ticklabels and axis labels are separated out from Matplotlib's Axis
class. Originally, this change was motivated to support curvilinear
grid. Here are a few reasons that I came up with a new axes class:

* "top" and "bottom" x-axis (or "left" and "right" y-axis) can have
  different ticks (tick locations and labels). This is not possible
  with the current Matplotlib, although some twin axes trick can help.

* Curvilinear grid.

* angled ticks.

In the new axes class, xaxis and yaxis is set to not visible by
default, and new set of artist (AxisArtist) are defined to draw axis
line, ticks, ticklabels and axis label. Axes.axis attribute serves as
a dictionary of these artists, i.e., ax.axis["left"] is a AxisArtist
instance responsible to draw left y-axis. The default Axes.axis contains
"bottom", "left", "top" and "right".

AxisArtist can be considered as a container artist and has the following
children artists which will draw ticks, labels, etc.

* line
* major_ticks, major_ticklabels
* minor_ticks, minor_ticklabels
* offsetText
* label

Note that these are separate artists from `matplotlib.axis.Axis`, thus most
tick-related functions in Matplotlib won't work. For example, color and
markerwidth of the ``ax.axis["bottom"].major_ticks`` will follow those of
Axes.xaxis unless explicitly specified.

In addition to AxisArtist, the Axes will have *gridlines* attribute,
which obviously draws grid lines. The gridlines needs to be separated
from the axis as some gridlines can never pass any axis.
"""
import numpy as np
import matplotlib as mpl
from matplotlib import _api
import matplotlib.axes as maxes
from matplotlib.path import Path
from mpl_toolkits.axes_grid1 import mpl_axes
from .axisline_style import AxislineStyle
from .axis_artist import AxisArtist, GridlinesCollection

class _AxisArtistHelperBase:
    """
    Base class for axis helper.

    Subclasses should define the methods listed below.  The *axes*
    argument will be the ``.axes`` attribute of the caller artist. ::

        # Construct the spine.

        def get_line_transform(self, axes):
            return transform

        def get_line(self, axes):
            return path

        # Construct the label.

        def get_axislabel_transform(self, axes):
            return transform

        def get_axislabel_pos_angle(self, axes):
            return (x, y), angle

        # Construct the ticks.

        def get_tick_transform(self, axes):
            return transform

        def get_tick_iterators(self, axes):
            # A pair of iterables (one for major ticks, one for minor ticks)
            # that yield (tick_position, tick_angle, tick_label).
            return iter_major, iter_minor
    """

    def update_lim(self, axes):
        if False:
            return 10
        pass

    def _to_xy(self, values, const):
        if False:
            while True:
                i = 10
        '\n        Create a (*values.shape, 2)-shape array representing (x, y) pairs.\n\n        The other coordinate is filled with the constant *const*.\n\n        Example::\n\n            >>> self.nth_coord = 0\n            >>> self._to_xy([1, 2, 3], const=0)\n            array([[1, 0],\n                   [2, 0],\n                   [3, 0]])\n        '
        if self.nth_coord == 0:
            return np.stack(np.broadcast_arrays(values, const), axis=-1)
        elif self.nth_coord == 1:
            return np.stack(np.broadcast_arrays(const, values), axis=-1)
        else:
            raise ValueError('Unexpected nth_coord')

class _FixedAxisArtistHelperBase(_AxisArtistHelperBase):
    """Helper class for a fixed (in the axes coordinate) axis."""

    @_api.delete_parameter('3.9', 'nth_coord')
    def __init__(self, loc, nth_coord=None):
        if False:
            i = 10
            return i + 15
        '``nth_coord = 0``: x-axis; ``nth_coord = 1``: y-axis.'
        self.nth_coord = _api.check_getitem({'bottom': 0, 'top': 0, 'left': 1, 'right': 1}, loc=loc)
        self._loc = loc
        self._pos = {'bottom': 0, 'top': 1, 'left': 0, 'right': 1}[loc]
        super().__init__()
        self._path = Path(self._to_xy((0, 1), const=self._pos))

    def get_nth_coord(self):
        if False:
            print('Hello World!')
        return self.nth_coord

    def get_line(self, axes):
        if False:
            while True:
                i = 10
        return self._path

    def get_line_transform(self, axes):
        if False:
            for i in range(10):
                print('nop')
        return axes.transAxes

    def get_axislabel_transform(self, axes):
        if False:
            i = 10
            return i + 15
        return axes.transAxes

    def get_axislabel_pos_angle(self, axes):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the label reference position in transAxes.\n\n        get_label_transform() returns a transform of (transAxes+offset)\n        '
        return dict(left=((0.0, 0.5), 90), right=((1.0, 0.5), 90), bottom=((0.5, 0.0), 0), top=((0.5, 1.0), 0))[self._loc]

    def get_tick_transform(self, axes):
        if False:
            for i in range(10):
                print('nop')
        return [axes.get_xaxis_transform(), axes.get_yaxis_transform()][self.nth_coord]

class _FloatingAxisArtistHelperBase(_AxisArtistHelperBase):

    def __init__(self, nth_coord, value):
        if False:
            while True:
                i = 10
        self.nth_coord = nth_coord
        self._value = value
        super().__init__()

    def get_nth_coord(self):
        if False:
            for i in range(10):
                print('nop')
        return self.nth_coord

    def get_line(self, axes):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError('get_line method should be defined by the derived class')

class FixedAxisArtistHelperRectilinear(_FixedAxisArtistHelperBase):

    @_api.delete_parameter('3.9', 'nth_coord')
    def __init__(self, axes, loc, nth_coord=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        nth_coord = along which coordinate value varies\n        in 2D, nth_coord = 0 ->  x axis, nth_coord = 1 -> y axis\n        '
        super().__init__(loc)
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

    def get_tick_iterators(self, axes):
        if False:
            print('Hello World!')
        'tick_loc, tick_angle, tick_label'
        (angle_normal, angle_tangent) = {0: (90, 0), 1: (0, 90)}[self.nth_coord]
        major = self.axis.major
        major_locs = major.locator()
        major_labels = major.formatter.format_ticks(major_locs)
        minor = self.axis.minor
        minor_locs = minor.locator()
        minor_labels = minor.formatter.format_ticks(minor_locs)
        tick_to_axes = self.get_tick_transform(axes) - axes.transAxes

        def _f(locs, labels):
            if False:
                for i in range(10):
                    print('nop')
            for (loc, label) in zip(locs, labels):
                c = self._to_xy(loc, const=self._pos)
                c2 = tick_to_axes.transform(c)
                if mpl.transforms._interval_contains_close((0, 1), c2[self.nth_coord]):
                    yield (c, angle_normal, angle_tangent, label)
        return (_f(major_locs, major_labels), _f(minor_locs, minor_labels))

class FloatingAxisArtistHelperRectilinear(_FloatingAxisArtistHelperBase):

    def __init__(self, axes, nth_coord, passingthrough_point, axis_direction='bottom'):
        if False:
            return 10
        super().__init__(nth_coord, passingthrough_point)
        self._axis_direction = axis_direction
        self.axis = [axes.xaxis, axes.yaxis][self.nth_coord]

    def get_line(self, axes):
        if False:
            i = 10
            return i + 15
        fixed_coord = 1 - self.nth_coord
        data_to_axes = axes.transData - axes.transAxes
        p = data_to_axes.transform([self._value, self._value])
        return Path(self._to_xy((0, 1), const=p[fixed_coord]))

    def get_line_transform(self, axes):
        if False:
            i = 10
            return i + 15
        return axes.transAxes

    def get_axislabel_transform(self, axes):
        if False:
            for i in range(10):
                print('nop')
        return axes.transAxes

    def get_axislabel_pos_angle(self, axes):
        if False:
            return 10
        '\n        Return the label reference position in transAxes.\n\n        get_label_transform() returns a transform of (transAxes+offset)\n        '
        angle = [0, 90][self.nth_coord]
        fixed_coord = 1 - self.nth_coord
        data_to_axes = axes.transData - axes.transAxes
        p = data_to_axes.transform([self._value, self._value])
        verts = self._to_xy(0.5, const=p[fixed_coord])
        return (verts, angle) if 0 <= verts[fixed_coord] <= 1 else (None, None)

    def get_tick_transform(self, axes):
        if False:
            for i in range(10):
                print('nop')
        return axes.transData

    def get_tick_iterators(self, axes):
        if False:
            return 10
        'tick_loc, tick_angle, tick_label'
        (angle_normal, angle_tangent) = {0: (90, 0), 1: (0, 90)}[self.nth_coord]
        major = self.axis.major
        major_locs = major.locator()
        major_labels = major.formatter.format_ticks(major_locs)
        minor = self.axis.minor
        minor_locs = minor.locator()
        minor_labels = minor.formatter.format_ticks(minor_locs)
        data_to_axes = axes.transData - axes.transAxes

        def _f(locs, labels):
            if False:
                print('Hello World!')
            for (loc, label) in zip(locs, labels):
                c = self._to_xy(loc, const=self._value)
                (c1, c2) = data_to_axes.transform(c)
                if 0 <= c1 <= 1 and 0 <= c2 <= 1:
                    yield (c, angle_normal, angle_tangent, label)
        return (_f(major_locs, major_labels), _f(minor_locs, minor_labels))

class AxisArtistHelper:
    Fixed = _FixedAxisArtistHelperBase
    Floating = _FloatingAxisArtistHelperBase

class AxisArtistHelperRectlinear:
    Fixed = FixedAxisArtistHelperRectilinear
    Floating = FloatingAxisArtistHelperRectilinear

class GridHelperBase:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._old_limits = None
        super().__init__()

    def update_lim(self, axes):
        if False:
            print('Hello World!')
        (x1, x2) = axes.get_xlim()
        (y1, y2) = axes.get_ylim()
        if self._old_limits != (x1, x2, y1, y2):
            self._update_grid(x1, y1, x2, y2)
            self._old_limits = (x1, x2, y1, y2)

    def _update_grid(self, x1, y1, x2, y2):
        if False:
            for i in range(10):
                print('nop')
        'Cache relevant computations when the axes limits have changed.'

    def get_gridlines(self, which, axis):
        if False:
            print('Hello World!')
        '\n        Return list of grid lines as a list of paths (list of points).\n\n        Parameters\n        ----------\n        which : {"both", "major", "minor"}\n        axis : {"both", "x", "y"}\n        '
        return []

class GridHelperRectlinear(GridHelperBase):

    def __init__(self, axes):
        if False:
            print('Hello World!')
        super().__init__()
        self.axes = axes

    @_api.delete_parameter('3.9', 'nth_coord', addendum="'nth_coord' is now inferred from 'loc'.")
    def new_fixed_axis(self, loc, nth_coord=None, axis_direction=None, offset=None, axes=None):
        if False:
            return 10
        if axes is None:
            _api.warn_external("'new_fixed_axis' explicitly requires the axes keyword.")
            axes = self.axes
        if axis_direction is None:
            axis_direction = loc
        return AxisArtist(axes, FixedAxisArtistHelperRectilinear(axes, loc), offset=offset, axis_direction=axis_direction)

    def new_floating_axis(self, nth_coord, value, axis_direction='bottom', axes=None):
        if False:
            for i in range(10):
                print('nop')
        if axes is None:
            _api.warn_external("'new_floating_axis' explicitly requires the axes keyword.")
            axes = self.axes
        helper = FloatingAxisArtistHelperRectilinear(axes, nth_coord, value, axis_direction)
        axisline = AxisArtist(axes, helper, axis_direction=axis_direction)
        axisline.line.set_clip_on(True)
        axisline.line.set_clip_box(axisline.axes.bbox)
        return axisline

    def get_gridlines(self, which='major', axis='both'):
        if False:
            i = 10
            return i + 15
        '\n        Return list of gridline coordinates in data coordinates.\n\n        Parameters\n        ----------\n        which : {"both", "major", "minor"}\n        axis : {"both", "x", "y"}\n        '
        _api.check_in_list(['both', 'major', 'minor'], which=which)
        _api.check_in_list(['both', 'x', 'y'], axis=axis)
        gridlines = []
        if axis in ('both', 'x'):
            locs = []
            (y1, y2) = self.axes.get_ylim()
            if which in ('both', 'major'):
                locs.extend(self.axes.xaxis.major.locator())
            if which in ('both', 'minor'):
                locs.extend(self.axes.xaxis.minor.locator())
            for x in locs:
                gridlines.append([[x, x], [y1, y2]])
        if axis in ('both', 'y'):
            (x1, x2) = self.axes.get_xlim()
            locs = []
            if self.axes.yaxis._major_tick_kw['gridOn']:
                locs.extend(self.axes.yaxis.major.locator())
            if self.axes.yaxis._minor_tick_kw['gridOn']:
                locs.extend(self.axes.yaxis.minor.locator())
            for y in locs:
                gridlines.append([[x1, x2], [y, y]])
        return gridlines

class Axes(maxes.Axes):

    @_api.deprecated('3.8', alternative='ax.axis')
    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return maxes.Axes.axis(self.axes, *args, **kwargs)

    def __init__(self, *args, grid_helper=None, **kwargs):
        if False:
            print('Hello World!')
        self._axisline_on = True
        self._grid_helper = grid_helper if grid_helper else GridHelperRectlinear(self)
        super().__init__(*args, **kwargs)
        self.toggle_axisline(True)

    def toggle_axisline(self, b=None):
        if False:
            i = 10
            return i + 15
        if b is None:
            b = not self._axisline_on
        if b:
            self._axisline_on = True
            self.spines[:].set_visible(False)
            self.xaxis.set_visible(False)
            self.yaxis.set_visible(False)
        else:
            self._axisline_on = False
            self.spines[:].set_visible(True)
            self.xaxis.set_visible(True)
            self.yaxis.set_visible(True)

    @property
    def axis(self):
        if False:
            while True:
                i = 10
        return self._axislines

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.gridlines = gridlines = GridlinesCollection([], colors=mpl.rcParams['grid.color'], linestyles=mpl.rcParams['grid.linestyle'], linewidths=mpl.rcParams['grid.linewidth'])
        self._set_artist_props(gridlines)
        gridlines.set_grid_helper(self.get_grid_helper())
        super().clear()
        gridlines.set_clip_path(self.axes.patch)
        self._axislines = mpl_axes.Axes.AxisDict(self)
        new_fixed_axis = self.get_grid_helper().new_fixed_axis
        self._axislines.update({loc: new_fixed_axis(loc=loc, axes=self, axis_direction=loc) for loc in ['bottom', 'top', 'left', 'right']})
        for axisline in [self._axislines['top'], self._axislines['right']]:
            axisline.label.set_visible(False)
            axisline.major_ticklabels.set_visible(False)
            axisline.minor_ticklabels.set_visible(False)

    def get_grid_helper(self):
        if False:
            i = 10
            return i + 15
        return self._grid_helper

    def grid(self, visible=None, which='major', axis='both', **kwargs):
        if False:
            print('Hello World!')
        '\n        Toggle the gridlines, and optionally set the properties of the lines.\n        '
        super().grid(visible, which=which, axis=axis, **kwargs)
        if not self._axisline_on:
            return
        if visible is None:
            visible = self.axes.xaxis._minor_tick_kw['gridOn'] or self.axes.xaxis._major_tick_kw['gridOn'] or self.axes.yaxis._minor_tick_kw['gridOn'] or self.axes.yaxis._major_tick_kw['gridOn']
        self.gridlines.set(which=which, axis=axis, visible=visible)
        self.gridlines.set(**kwargs)

    def get_children(self):
        if False:
            i = 10
            return i + 15
        if self._axisline_on:
            children = [*self._axislines.values(), self.gridlines]
        else:
            children = []
        children.extend(super().get_children())
        return children

    def new_fixed_axis(self, loc, offset=None):
        if False:
            while True:
                i = 10
        return self.get_grid_helper().new_fixed_axis(loc, offset=offset, axes=self)

    def new_floating_axis(self, nth_coord, value, axis_direction='bottom'):
        if False:
            i = 10
            return i + 15
        return self.get_grid_helper().new_floating_axis(nth_coord, value, axis_direction=axis_direction, axes=self)

class AxesZero(Axes):

    def clear(self):
        if False:
            i = 10
            return i + 15
        super().clear()
        new_floating_axis = self.get_grid_helper().new_floating_axis
        self._axislines.update(xzero=new_floating_axis(nth_coord=0, value=0.0, axis_direction='bottom', axes=self), yzero=new_floating_axis(nth_coord=1, value=0.0, axis_direction='left', axes=self))
        for k in ['xzero', 'yzero']:
            self._axislines[k].line.set_clip_path(self.patch)
            self._axislines[k].set_visible(False)
Subplot = Axes
SubplotZero = AxesZero