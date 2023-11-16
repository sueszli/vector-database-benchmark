"""
This file defines the classes used to represent a 'coordinate', which includes
axes, ticks, tick labels, and grid lines.
"""
import warnings
import numpy as np
from matplotlib import rcParams
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.ticker import Formatter
from matplotlib.transforms import Affine2D, ScaledTranslation
from astropy import units as u
from astropy.utils.exceptions import AstropyDeprecationWarning
from .axislabels import AxisLabels
from .formatter_locator import AngleFormatterLocator, ScalarFormatterLocator
from .frame import EllipticalFrame, RectangularFrame1D
from .grid_paths import get_gridline_path, get_lon_lat_path
from .ticklabels import TickLabels
from .ticks import Ticks
from .utils import MATPLOTLIB_LT_3_8
__all__ = ['CoordinateHelper']
LINES_TO_PATCHES_LINESTYLE = {'-': 'solid', '--': 'dashed', '-.': 'dashdot', ':': 'dotted', 'none': 'none', 'None': 'none', ' ': 'none', '': 'none'}

def wrap_angle_at(values, coord_wrap):
    if False:
        print('Hello World!')
    with np.errstate(invalid='ignore'):
        return np.mod(values - coord_wrap, 360.0) - (360.0 - coord_wrap)

class CoordinateHelper:
    """
    Helper class to control one of the coordinates in the
    :class:`~astropy.visualization.wcsaxes.WCSAxes`.

    Parameters
    ----------
    parent_axes : :class:`~astropy.visualization.wcsaxes.WCSAxes`
        The axes the coordinate helper belongs to.
    parent_map : :class:`~astropy.visualization.wcsaxes.CoordinatesMap`
        The :class:`~astropy.visualization.wcsaxes.CoordinatesMap` object this
        coordinate belongs to.
    transform : `~matplotlib.transforms.Transform`
        The transform corresponding to this coordinate system.
    coord_index : int
        The index of this coordinate in the
        :class:`~astropy.visualization.wcsaxes.CoordinatesMap`.
    coord_type : {'longitude', 'latitude', 'scalar'}
        The type of this coordinate, which is used to determine the wrapping and
        boundary behavior of coordinates. Longitudes wrap at ``coord_wrap``,
        latitudes have to be in the range -90 to 90, and scalars are unbounded
        and do not wrap.
    coord_unit : `~astropy.units.Unit`
        The unit that this coordinate is in given the output of transform.
    format_unit : `~astropy.units.Unit`, optional
        The unit to use to display the coordinates.
    coord_wrap : `astropy.units.Quantity`
        The angle at which the longitude wraps (defaults to 360 degrees).
    frame : `~astropy.visualization.wcsaxes.frame.BaseFrame`
        The frame of the :class:`~astropy.visualization.wcsaxes.WCSAxes`.
    """

    def __init__(self, parent_axes=None, parent_map=None, transform=None, coord_index=None, coord_type='scalar', coord_unit=None, coord_wrap=None, frame=None, format_unit=None, default_label=None):
        if False:
            print('Hello World!')
        self.parent_axes = parent_axes
        self.parent_map = parent_map
        self.transform = transform
        self.coord_index = coord_index
        self.coord_unit = coord_unit
        self._format_unit = format_unit
        self.frame = frame
        self.default_label = default_label or ''
        self._auto_axislabel = True
        if issubclass(self.parent_axes.frame_class, EllipticalFrame):
            self._auto_axislabel = False
        self.set_coord_type(coord_type, coord_wrap)
        self.dpi_transform = Affine2D()
        self.offset_transform = ScaledTranslation(0, 0, self.dpi_transform)
        self.ticks = Ticks(transform=parent_axes.transData + self.offset_transform)
        self.ticklabels = TickLabels(self.frame, transform=None, figure=parent_axes.get_figure())
        self.ticks.display_minor_ticks(rcParams['xtick.minor.visible'])
        self.minor_frequency = 5
        self.axislabels = AxisLabels(self.frame, transform=None, figure=parent_axes.get_figure())
        self.grid_lines = []
        self.grid_lines_kwargs = {'visible': False, 'facecolor': 'none', 'edgecolor': rcParams['grid.color'], 'linestyle': LINES_TO_PATCHES_LINESTYLE[rcParams['grid.linestyle']], 'linewidth': rcParams['grid.linewidth'], 'alpha': rcParams['grid.alpha'], 'transform': self.parent_axes.transData}

    def grid(self, draw_grid=True, grid_type=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Plot grid lines for this coordinate.\n\n        Standard matplotlib appearance options (color, alpha, etc.) can be\n        passed as keyword arguments.\n\n        Parameters\n        ----------\n        draw_grid : bool\n            Whether to show the gridlines\n        grid_type : {'lines', 'contours'}\n            Whether to plot the contours by determining the grid lines in\n            world coordinates and then plotting them in world coordinates\n            (``'lines'``) or by determining the world coordinates at many\n            positions in the image and then drawing contours\n            (``'contours'``). The first is recommended for 2-d images, while\n            for 3-d (or higher dimensional) cubes, the ``'contours'`` option\n            is recommended. By default, 'lines' is used if the transform has\n            an inverse, otherwise 'contours' is used.\n        "
        if grid_type == 'lines' and (not self.transform.has_inverse):
            raise ValueError("The specified transform has no inverse, so the grid cannot be drawn using grid_type='lines'")
        if grid_type is None:
            grid_type = 'lines' if self.transform.has_inverse else 'contours'
        if grid_type in ('lines', 'contours'):
            self._grid_type = grid_type
        else:
            raise ValueError("grid_type should be 'lines' or 'contours'")
        if 'color' in kwargs:
            kwargs['edgecolor'] = kwargs.pop('color')
        self.grid_lines_kwargs.update(kwargs)
        if self.grid_lines_kwargs['visible']:
            if not draw_grid:
                self.grid_lines_kwargs['visible'] = False
        else:
            self.grid_lines_kwargs['visible'] = True

    def set_coord_type(self, coord_type, coord_wrap=None):
        if False:
            print('Hello World!')
        "\n        Set the coordinate type for the axis.\n\n        Parameters\n        ----------\n        coord_type : str\n            One of 'longitude', 'latitude' or 'scalar'\n        coord_wrap : `~astropy.units.Quantity`, optional\n            The value to wrap at for angular coordinates.\n        "
        self.coord_type = coord_type
        if coord_wrap is not None and (not isinstance(coord_wrap, u.Quantity)):
            warnings.warn("Passing 'coord_wrap' as a number is deprecated. Use a Quantity with units convertible to angular degrees instead.", AstropyDeprecationWarning)
            coord_wrap = coord_wrap * u.deg
        if coord_type == 'longitude' and coord_wrap is None:
            self.coord_wrap = 360 * u.deg
        elif coord_type != 'longitude' and coord_wrap is not None:
            raise NotImplementedError('coord_wrap is not yet supported for non-longitude coordinates')
        else:
            self.coord_wrap = coord_wrap
        if coord_type == 'scalar':
            self._coord_scale_to_deg = None
            self._formatter_locator = ScalarFormatterLocator(unit=self.coord_unit)
        elif coord_type in ['longitude', 'latitude']:
            if self.coord_unit is u.deg:
                self._coord_scale_to_deg = None
            else:
                self._coord_scale_to_deg = self.coord_unit.to(u.deg)
            self._formatter_locator = AngleFormatterLocator(unit=self.coord_unit, format_unit=self._format_unit)
        else:
            raise ValueError("coord_type should be one of 'scalar', 'longitude', or 'latitude'")

    def set_major_formatter(self, formatter):
        if False:
            print('Hello World!')
        '\n        Set the formatter to use for the major tick labels.\n\n        Parameters\n        ----------\n        formatter : str or `~matplotlib.ticker.Formatter`\n            The format or formatter to use.\n        '
        if isinstance(formatter, Formatter):
            raise NotImplementedError()
        elif isinstance(formatter, str):
            self._formatter_locator.format = formatter
        else:
            raise TypeError('formatter should be a string or a Formatter instance')

    def format_coord(self, value, format='auto'):
        if False:
            i = 10
            return i + 15
        "\n        Given the value of a coordinate, will format it according to the\n        format of the formatter_locator.\n\n        Parameters\n        ----------\n        value : float\n            The value to format\n        format : {'auto', 'ascii', 'latex'}, optional\n            The format to use - by default the formatting will be adjusted\n            depending on whether Matplotlib is using LaTeX or MathTex. To\n            get plain ASCII strings, use format='ascii'.\n        "
        if not hasattr(self, '_fl_spacing'):
            return ''
        fl = self._formatter_locator
        if isinstance(fl, AngleFormatterLocator):
            if self._coord_scale_to_deg is not None:
                value *= self._coord_scale_to_deg
            if self.coord_type == 'longitude':
                value = wrap_angle_at(value, self.coord_wrap.to_value(u.deg))
            value = value * u.degree
            value = value.to_value(fl._unit)
        spacing = self._fl_spacing
        string = fl.formatter(values=[value] * fl._unit, spacing=spacing, format=format)
        return string[0]

    def set_separator(self, separator):
        if False:
            while True:
                i = 10
        '\n        Set the separator to use for the angle major tick labels.\n\n        Parameters\n        ----------\n        separator : str or tuple or None\n            The separator between numbers in sexagesimal representation. Can be\n            either a string or a tuple (or `None` for default).\n        '
        if not self._formatter_locator.__class__ == AngleFormatterLocator:
            raise TypeError('Separator can only be specified for angle coordinates')
        if isinstance(separator, (str, tuple)) or separator is None:
            self._formatter_locator.sep = separator
        else:
            raise TypeError('separator should be a string, a tuple, or None')

    def set_format_unit(self, unit, decimal=None, show_decimal_unit=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the unit for the major tick labels.\n\n        Parameters\n        ----------\n        unit : class:`~astropy.units.Unit`\n            The unit to which the tick labels should be converted to.\n        decimal : bool, optional\n            Whether to use decimal formatting. By default this is `False`\n            for degrees or hours (which therefore use sexagesimal formatting)\n            and `True` for all other units.\n        show_decimal_unit : bool, optional\n            Whether to include units when in decimal mode.\n        '
        self._formatter_locator.format_unit = u.Unit(unit)
        self._formatter_locator.decimal = decimal
        self._formatter_locator.show_decimal_unit = show_decimal_unit

    def get_format_unit(self):
        if False:
            return 10
        '\n        Get the unit for the major tick labels.\n        '
        return self._formatter_locator.format_unit

    def set_ticks(self, values=None, spacing=None, number=None, size=None, width=None, color=None, alpha=None, direction=None, exclude_overlapping=None):
        if False:
            return 10
        "\n        Set the location and properties of the ticks.\n\n        At most one of the options from ``values``, ``spacing``, or\n        ``number`` can be specified.\n\n        Parameters\n        ----------\n        values : iterable, optional\n            The coordinate values at which to show the ticks.\n        spacing : float, optional\n            The spacing between ticks.\n        number : float, optional\n            The approximate number of ticks shown.\n        size : float, optional\n            The length of the ticks in points\n        color : str or tuple, optional\n            A valid Matplotlib color for the ticks\n        alpha : float, optional\n            The alpha value (transparency) for the ticks.\n        direction : {'in','out'}, optional\n            Whether the ticks should point inwards or outwards.\n        "
        if sum([values is None, spacing is None, number is None]) < 2:
            raise ValueError('At most one of values, spacing, or number should be specified')
        if values is not None:
            self._formatter_locator.values = values
        elif spacing is not None:
            self._formatter_locator.spacing = spacing
        elif number is not None:
            self._formatter_locator.number = number
        if size is not None:
            self.ticks.set_ticksize(size)
        if width is not None:
            self.ticks.set_linewidth(width)
        if color is not None:
            self.ticks.set_color(color)
        if alpha is not None:
            self.ticks.set_alpha(alpha)
        if direction is not None:
            if direction in ('in', 'out'):
                self.ticks.set_tick_out(direction == 'out')
            else:
                raise ValueError("direction should be 'in' or 'out'")
        if exclude_overlapping is not None:
            warnings.warn('exclude_overlapping= should be passed to set_ticklabel instead of set_ticks', AstropyDeprecationWarning)
            self.ticklabels.set_exclude_overlapping(exclude_overlapping)

    def set_ticks_position(self, position):
        if False:
            i = 10
            return i + 15
        "\n        Set where ticks should appear.\n\n        Parameters\n        ----------\n        position : str\n            The axes on which the ticks for this coordinate should appear.\n            Should be a string containing zero or more of ``'b'``, ``'t'``,\n            ``'l'``, ``'r'``. For example, ``'lb'`` will lead the ticks to be\n            shown on the left and bottom axis.\n        "
        self.ticks.set_visible_axes(position)

    def set_ticks_visible(self, visible):
        if False:
            print('Hello World!')
        '\n        Set whether ticks are visible or not.\n\n        Parameters\n        ----------\n        visible : bool\n            The visibility of ticks. Setting as ``False`` will hide ticks\n            along this coordinate.\n        '
        self.ticks.set_visible(visible)

    def set_ticklabel(self, color=None, size=None, pad=None, exclude_overlapping=None, **kwargs):
        if False:
            return 10
        '\n        Set the visual properties for the tick labels.\n\n        Parameters\n        ----------\n        size : float, optional\n            The size of the ticks labels in points\n        color : str or tuple, optional\n            A valid Matplotlib color for the tick labels\n        pad : float, optional\n            Distance in points between tick and label.\n        exclude_overlapping : bool, optional\n            Whether to exclude tick labels that overlap over each other.\n        **kwargs\n            Other keyword arguments are passed to :class:`matplotlib.text.Text`.\n        '
        if size is not None:
            self.ticklabels.set_size(size)
        if color is not None:
            self.ticklabels.set_color(color)
        if pad is not None:
            self.ticklabels.set_pad(pad)
        if exclude_overlapping is not None:
            self.ticklabels.set_exclude_overlapping(exclude_overlapping)
        self.ticklabels.set(**kwargs)

    def set_ticklabel_position(self, position):
        if False:
            print('Hello World!')
        "\n        Set where tick labels should appear.\n\n        Parameters\n        ----------\n        position : str\n            The axes on which the tick labels for this coordinate should\n            appear. Should be a string containing zero or more of ``'b'``,\n            ``'t'``, ``'l'``, ``'r'``. For example, ``'lb'`` will lead the\n            tick labels to be shown on the left and bottom axis.\n        "
        self.ticklabels.set_visible_axes(position)

    def set_ticklabel_visible(self, visible):
        if False:
            return 10
        "\n        Set whether the tick labels are visible or not.\n\n        Parameters\n        ----------\n        visible : bool\n            The visibility of ticks. Setting as ``False`` will hide this\n            coordinate's tick labels.\n        "
        self.ticklabels.set_visible(visible)

    def set_axislabel(self, text, minpad=1, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Set the text and optionally visual properties for the axis label.\n\n        Parameters\n        ----------\n        text : str\n            The axis label text.\n        minpad : float, optional\n            The padding for the label in terms of axis label font size.\n        **kwargs\n            Keywords are passed to :class:`matplotlib.text.Text`. These\n            can include keywords to set the ``color``, ``size``, ``weight``, and\n            other text properties.\n        '
        fontdict = kwargs.pop('fontdict', None)
        if minpad is None:
            minpad = 1
        self.axislabels.set_text(text)
        self.axislabels.set_minpad(minpad)
        self.axislabels.set(**kwargs)
        if fontdict is not None:
            self.axislabels.update(fontdict)

    def get_axislabel(self):
        if False:
            print('Hello World!')
        '\n        Get the text for the axis label.\n\n        Returns\n        -------\n        label : str\n            The axis label\n        '
        return self.axislabels.get_text()

    def set_auto_axislabel(self, auto_label):
        if False:
            while True:
                i = 10
        '\n        Render default axis labels if no explicit label is provided.\n\n        Parameters\n        ----------\n        auto_label : `bool`\n            `True` if default labels will be rendered.\n        '
        self._auto_axislabel = bool(auto_label)

    def get_auto_axislabel(self):
        if False:
            i = 10
            return i + 15
        '\n        Render default axis labels if no explicit label is provided.\n\n        Returns\n        -------\n        auto_axislabel : `bool`\n            `True` if default labels will be rendered.\n        '
        return self._auto_axislabel

    def _get_default_axislabel(self):
        if False:
            while True:
                i = 10
        unit = self.get_format_unit() or self.coord_unit
        if not unit or unit is u.one or self.coord_type in ('longitude', 'latitude'):
            return f'{self.default_label}'
        else:
            return f'{self.default_label} [{unit:latex}]'

    def set_axislabel_position(self, position):
        if False:
            print('Hello World!')
        "\n        Set where axis labels should appear.\n\n        Parameters\n        ----------\n        position : str\n            The axes on which the axis label for this coordinate should\n            appear. Should be a string containing zero or more of ``'b'``,\n            ``'t'``, ``'l'``, ``'r'``. For example, ``'lb'`` will lead the\n            axis label to be shown on the left and bottom axis.\n        "
        self.axislabels.set_visible_axes(position)

    def set_axislabel_visibility_rule(self, rule):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set the rule used to determine when the axis label is drawn.\n\n        Parameters\n        ----------\n        rule : str\n            If the rule is 'always' axis labels will always be drawn on the\n            axis. If the rule is 'ticks' the label will only be drawn if ticks\n            were drawn on that axis. If the rule is 'labels' the axis label\n            will only be drawn if tick labels were drawn on that axis.\n        "
        self.axislabels.set_visibility_rule(rule)

    def get_axislabel_visibility_rule(self, rule):
        if False:
            return 10
        '\n        Get the rule used to determine when the axis label is drawn.\n        '
        return self.axislabels.get_visibility_rule()

    @property
    def locator(self):
        if False:
            return 10
        return self._formatter_locator.locator

    @property
    def formatter(self):
        if False:
            i = 10
            return i + 15
        return self._formatter_locator.formatter

    def _draw_grid(self, renderer):
        if False:
            i = 10
            return i + 15
        renderer.open_group('grid lines')
        self._update_ticks()
        if self.grid_lines_kwargs['visible']:
            if isinstance(self.frame, RectangularFrame1D):
                self._update_grid_lines_1d()
            elif self._grid_type == 'lines':
                self._update_grid_lines()
            else:
                self._update_grid_contour()
            if self._grid_type == 'lines':
                frame_patch = self.frame.patch
                for path in self.grid_lines:
                    p = PathPatch(path, **self.grid_lines_kwargs)
                    p.set_clip_path(frame_patch)
                    p.draw(renderer)
            elif self._grid is not None:
                if MATPLOTLIB_LT_3_8:
                    for line in self._grid.collections:
                        line.set(**self.grid_lines_kwargs)
                        line.draw(renderer)
                else:
                    self._grid.set(**self.grid_lines_kwargs)
                    self._grid.draw(renderer)
        renderer.close_group('grid lines')

    def _draw_ticks(self, renderer, existing_bboxes):
        if False:
            return 10
        '\n        Draw all ticks and ticklabels.\n\n        Parameters\n        ----------\n        existing_bboxes : list[Bbox]\n            All bboxes for ticks that have already been drawn by other\n            coordinates.\n        '
        renderer.open_group('ticks')
        self.ticks.draw(renderer)
        self.ticklabels._tick_out_size = self.ticks.out_size
        self.ticklabels._set_existing_bboxes(existing_bboxes)
        self.ticklabels.draw(renderer)
        renderer.close_group('ticks')

    def _draw_axislabels(self, renderer, bboxes, ticklabels_bbox, visible_ticks):
        if False:
            i = 10
            return i + 15
        if self._auto_axislabel and (not self.get_axislabel()):
            self.set_axislabel(self._get_default_axislabel())
        renderer.open_group('axis labels')
        self.axislabels.draw(renderer, bboxes=bboxes, ticklabels_bbox=ticklabels_bbox, coord_ticklabels_bbox=ticklabels_bbox[self], ticks_locs=self.ticks.ticks_locs, visible_ticks=visible_ticks)
        renderer.close_group('axis labels')

    def _update_ticks(self):
        if False:
            print('Hello World!')
        if self.coord_index is None:
            return
        coord_range = self.parent_map.get_coord_range()
        (tick_world_coordinates, self._fl_spacing) = self.locator(*coord_range[self.coord_index])
        if self.ticks.get_display_minor_ticks():
            minor_ticks_w_coordinates = self._formatter_locator.minor_locator(self._fl_spacing, self.get_minor_frequency(), *coord_range[self.coord_index])
        from . import conf
        frame = self.frame.sample(conf.frame_boundary_samples)
        self.ticks.clear()
        self.ticklabels.clear()
        self.lblinfo = []
        self.lbl_world = []
        transData = self.parent_axes.transData
        invertedTransLimits = transData.inverted()
        for (axis, spine) in frame.items():
            if spine.data.size == 0:
                continue
            if not isinstance(self.frame, RectangularFrame1D):
                pixel0 = spine.data
                world0 = spine.world[:, self.coord_index]
                if np.isnan(world0).all():
                    continue
                axes0 = transData.transform(pixel0)
                pixel1 = axes0.copy()
                pixel1[:, 0] += 2.0
                pixel1 = invertedTransLimits.transform(pixel1)
                with np.errstate(invalid='ignore'):
                    world1 = self.transform.transform(pixel1)[:, self.coord_index]
                pixel2 = axes0.copy()
                pixel2[:, 1] += 2.0 if self.frame.origin == 'lower' else -2.0
                pixel2 = invertedTransLimits.transform(pixel2)
                with np.errstate(invalid='ignore'):
                    world2 = self.transform.transform(pixel2)[:, self.coord_index]
                dx = world1 - world0
                dy = world2 - world0
                (dx, dy) = (-dy, dx)
                if self.coord_type == 'longitude':
                    if self._coord_scale_to_deg is not None:
                        dx *= self._coord_scale_to_deg
                        dy *= self._coord_scale_to_deg
                    dx = wrap_angle_at(dx, 180.0)
                    dy = wrap_angle_at(dy, 180.0)
                tick_angle = np.degrees(np.arctan2(dy, dx))
                normal_angle_full = np.hstack([spine.normal_angle, spine.normal_angle[-1]])
                with np.errstate(invalid='ignore'):
                    reset = ((normal_angle_full - tick_angle) % 360 > 90.0) & ((tick_angle - normal_angle_full) % 360 > 90.0)
                tick_angle[reset] -= 180.0
            else:
                rotation = 90 if axis == 'b' else -90
                tick_angle = np.zeros((conf.frame_boundary_samples,)) + rotation
            w1 = spine.world[:-1, self.coord_index]
            w2 = spine.world[1:, self.coord_index]
            if self.coord_type == 'longitude':
                if self._coord_scale_to_deg is not None:
                    w1 = w1 * self._coord_scale_to_deg
                    w2 = w2 * self._coord_scale_to_deg
                w1 = wrap_angle_at(w1, self.coord_wrap.to_value(u.deg))
                w2 = wrap_angle_at(w2, self.coord_wrap.to_value(u.deg))
                with np.errstate(invalid='ignore'):
                    w1[w2 - w1 > 180.0] += 360
                    w2[w1 - w2 > 180.0] += 360
                if self._coord_scale_to_deg is not None:
                    w1 = w1 / self._coord_scale_to_deg
                    w2 = w2 / self._coord_scale_to_deg
            self._compute_ticks(tick_world_coordinates, spine, axis, w1, w2, tick_angle)
            if self.ticks.get_display_minor_ticks():
                self._compute_ticks(minor_ticks_w_coordinates, spine, axis, w1, w2, tick_angle, ticks='minor')
        text = self.formatter(self.lbl_world * tick_world_coordinates.unit, spacing=self._fl_spacing)
        for (kwargs, txt) in zip(self.lblinfo, text):
            self.ticklabels.add(text=txt, **kwargs)

    def _compute_ticks(self, tick_world_coordinates, spine, axis, w1, w2, tick_angle, ticks='major'):
        if False:
            while True:
                i = 10
        if self.coord_type == 'longitude':
            tick_world_coordinates_values = tick_world_coordinates.to_value(u.deg)
            tick_world_coordinates_values = np.hstack([tick_world_coordinates_values, tick_world_coordinates_values + 360])
            tick_world_coordinates_values *= u.deg.to(self.coord_unit)
        else:
            tick_world_coordinates_values = tick_world_coordinates.to_value(self.coord_unit)
        for t in tick_world_coordinates_values:
            with np.errstate(invalid='ignore'):
                intersections = np.hstack([np.nonzero(t - w1 == 0)[0], np.nonzero((t - w1) * (t - w2) < 0)[0]])
            if t - w2[-1] == 0:
                intersections = np.append(intersections, len(w2) - 1)
            for imin in intersections:
                imax = imin + 1
                if np.allclose(w1[imin], w2[imin], rtol=1e-13, atol=1e-13):
                    continue
                else:
                    frac = (t - w1[imin]) / (w2[imin] - w1[imin])
                    x_data_i = spine.data[imin, 0] + frac * (spine.data[imax, 0] - spine.data[imin, 0])
                    y_data_i = spine.data[imin, 1] + frac * (spine.data[imax, 1] - spine.data[imin, 1])
                    delta_angle = tick_angle[imax] - tick_angle[imin]
                    if delta_angle > 180.0:
                        delta_angle -= 360.0
                    elif delta_angle < -180.0:
                        delta_angle += 360.0
                    angle_i = tick_angle[imin] + frac * delta_angle
                if self.coord_type == 'longitude':
                    if self._coord_scale_to_deg is not None:
                        t *= self._coord_scale_to_deg
                    world = wrap_angle_at(t, self.coord_wrap.to_value(u.deg))
                    if self._coord_scale_to_deg is not None:
                        world /= self._coord_scale_to_deg
                else:
                    world = t
                if ticks == 'major':
                    self.ticks.add(axis=axis, pixel=(x_data_i, y_data_i), world=world, angle=angle_i, axis_displacement=imin + frac)
                    self.lblinfo.append(dict(axis=axis, data=(x_data_i, y_data_i), world=world, angle=spine.normal_angle[imin], axis_displacement=imin + frac))
                    self.lbl_world.append(world)
                else:
                    self.ticks.add_minor(minor_axis=axis, minor_pixel=(x_data_i, y_data_i), minor_world=world, minor_angle=angle_i, minor_axis_displacement=imin + frac)

    def display_minor_ticks(self, display_minor_ticks):
        if False:
            return 10
        '\n        Display minor ticks for this coordinate.\n\n        Parameters\n        ----------\n        display_minor_ticks : bool\n            Whether or not to display minor ticks.\n        '
        self.ticks.display_minor_ticks(display_minor_ticks)

    def get_minor_frequency(self):
        if False:
            i = 10
            return i + 15
        return self.minor_frequency

    def set_minor_frequency(self, frequency):
        if False:
            return 10
        '\n        Set the frequency of minor ticks per major ticks.\n\n        Parameters\n        ----------\n        frequency : int\n            The number of minor ticks per major ticks.\n        '
        self.minor_frequency = frequency

    def _update_grid_lines_1d(self):
        if False:
            print('Hello World!')
        if self.coord_index is None:
            return
        x_ticks_pos = [a[0] for a in self.ticks.pixel['b']]
        (ymin, ymax) = self.parent_axes.get_ylim()
        self.grid_lines = []
        for x_coord in x_ticks_pos:
            pixel = [[x_coord, ymin], [x_coord, ymax]]
            self.grid_lines.append(Path(pixel))

    def _update_grid_lines(self):
        if False:
            print('Hello World!')
        if self.coord_index is None:
            return
        coord_range = self.parent_map.get_coord_range()
        (tick_world_coordinates, spacing) = self.locator(*coord_range[self.coord_index])
        tick_world_coordinates_values = tick_world_coordinates.to_value(self.coord_unit)
        n_coord = len(tick_world_coordinates_values)
        if n_coord == 0:
            return
        from . import conf
        n_samples = conf.grid_samples
        xy_world = np.zeros((n_samples * n_coord, 2))
        self.grid_lines = []
        for (iw, w) in enumerate(tick_world_coordinates_values):
            subset = slice(iw * n_samples, (iw + 1) * n_samples)
            if self.coord_index == 0:
                xy_world[subset, 0] = np.repeat(w, n_samples)
                xy_world[subset, 1] = np.linspace(coord_range[1][0], coord_range[1][1], n_samples)
            else:
                xy_world[subset, 0] = np.linspace(coord_range[0][0], coord_range[0][1], n_samples)
                xy_world[subset, 1] = np.repeat(w, n_samples)
        pixel = self.transform.inverted().transform(xy_world)
        xy_world_round = self.transform.transform(pixel)
        for iw in range(n_coord):
            subset = slice(iw * n_samples, (iw + 1) * n_samples)
            self.grid_lines.append(self._get_gridline(xy_world[subset], pixel[subset], xy_world_round[subset]))

    def add_tickable_gridline(self, name, constant):
        if False:
            i = 10
            return i + 15
        '\n        Define a gridline that can be used for ticks and labels.\n\n        This gridline is not itself drawn, but instead can be specified in calls to\n        methods such as\n        :meth:`~astropy.visualization.wcsaxes.coordinate_helpers.CoordinateHelper.set_ticklabel_position`\n        for drawing ticks and labels.  Since the gridline has a constant value in this\n        coordinate, and thus would not have any ticks or labels for the same coordinate,\n        the call to\n        :meth:`~astropy.visualization.wcsaxes.coordinate_helpers.CoordinateHelper.set_ticklabel_position`\n        would typically be made on the complementary coordinate.\n\n        Parameters\n        ----------\n        name : str\n            The name for the gridline, usually a single character, but can be longer\n        constant : `~astropy.units.Quantity`\n            The constant coordinate value of the gridline\n\n        Notes\n        -----\n        A limitation is that the tickable part of the gridline must be contiguous.  If\n        the gridline consists of more than one disconnected segment within the plot\n        extent, only one of those segments will be made tickable.\n        '
        if self.coord_index is None:
            return
        if name in self.frame:
            raise ValueError(f"The frame already has a spine with the name '{name}'")
        coord_range = self.parent_map.get_coord_range()
        constant = constant.to_value(self.coord_unit)
        from . import conf
        n_samples = conf.grid_samples
        xy_world = np.zeros((n_samples, 2))
        xy_world[:, self.coord_index] = np.repeat(constant, n_samples)
        if self.parent_map[1 - self.coord_index].coord_type == 'longitude':
            xy_world[:-1, 1 - self.coord_index] = np.linspace(coord_range[1 - self.coord_index][0], coord_range[1 - self.coord_index][1], n_samples - 1)
            xy_world[-1, 1 - self.coord_index] = coord_range[1 - self.coord_index][0]
        else:
            xy_world[:, 1 - self.coord_index] = np.linspace(coord_range[1 - self.coord_index][0], coord_range[1 - self.coord_index][1], n_samples)
        pixel = self.transform.inverted().transform(xy_world)
        xy_world_round = self.transform.transform(pixel)
        gridline = self._get_gridline(xy_world, pixel, xy_world_round)

        def data_for_spine(spine):
            if False:
                print('Hello World!')
            vertices = gridline.vertices.copy()
            codes = gridline.codes.copy()
            (xmin, xmax) = spine.parent_axes.get_xlim()
            (ymin, ymax) = spine.parent_axes.get_ylim()
            keep = (vertices[:, 0] >= xmin) & (vertices[:, 0] <= xmax) & (vertices[:, 1] >= ymin) & (vertices[:, 1] <= ymax)
            codes[~keep] = Path.MOVETO
            codes[1:][~keep[:-1]] = Path.MOVETO
            lineto = np.flatnonzero(codes == Path.LINETO)
            if np.size(lineto) == 0:
                return np.zeros((0, 2))
            last_segment = np.flatnonzero(codes[:lineto[-1]] == Path.MOVETO)[-1]
            if vertices[0, 0] == vertices[-1, 0] and vertices[0, 1] == vertices[-1, 1]:
                codes = np.concatenate([codes, codes[1:]])
                vertices = np.vstack([vertices, vertices[1:, :]])
            moveto = np.flatnonzero(codes[last_segment + 1:] == Path.MOVETO)
            if np.size(moveto) > 0:
                return vertices[last_segment:last_segment + moveto[0] + 1, :]
            else:
                return vertices[last_segment:n_samples, :]
        self.frame[name] = self.frame.spine_class(self.frame.parent_axes, self.frame.transform, data_func=data_for_spine)

    def _get_gridline(self, xy_world, pixel, xy_world_round):
        if False:
            for i in range(10):
                print('nop')
        if self.coord_type == 'scalar':
            return get_gridline_path(xy_world, pixel)
        else:
            return get_lon_lat_path(xy_world, pixel, xy_world_round)

    def _clear_grid_contour(self):
        if False:
            i = 10
            return i + 15
        if hasattr(self, '_grid') and self._grid:
            if MATPLOTLIB_LT_3_8:
                for line in self._grid.collections:
                    line.remove()
            else:
                self._grid.remove()

    def _update_grid_contour(self):
        if False:
            for i in range(10):
                print('nop')
        if self.coord_index is None:
            return
        (xmin, xmax) = self.parent_axes.get_xlim()
        (ymin, ymax) = self.parent_axes.get_ylim()
        from . import conf
        res = conf.contour_grid_samples
        (x, y) = np.meshgrid(np.linspace(xmin, xmax, res), np.linspace(ymin, ymax, res))
        pixel = np.array([x.ravel(), y.ravel()]).T
        world = self.transform.transform(pixel)
        field = world[:, self.coord_index].reshape(res, res).T
        coord_range = self.parent_map.get_coord_range()
        (tick_world_coordinates, spacing) = self.locator(*coord_range[self.coord_index])
        tick_world_coordinates_values = tick_world_coordinates.value
        if self.coord_type == 'longitude':
            mid = 0.5 * (tick_world_coordinates_values[0] + tick_world_coordinates_values[1])
            field = wrap_angle_at(field, mid)
            tick_world_coordinates_values = wrap_angle_at(tick_world_coordinates_values, mid)
            with np.errstate(invalid='ignore'):
                reset = (np.abs(np.diff(field[:, :-1], axis=0)) > 180) | (np.abs(np.diff(field[:-1, :], axis=1)) > 180)
            field[:-1, :-1][reset] = np.nan
            field[1:, :-1][reset] = np.nan
            field[:-1, 1:][reset] = np.nan
            field[1:, 1:][reset] = np.nan
        if len(tick_world_coordinates_values) > 0:
            with np.errstate(invalid='ignore'):
                self._grid = self.parent_axes.contour(x, y, field.transpose(), levels=np.sort(tick_world_coordinates_values))
        else:
            self._grid = None

    def tick_params(self, which='both', **kwargs):
        if False:
            while True:
                i = 10
        "\n        Method to set the tick and tick label parameters in the same way as the\n        :meth:`~matplotlib.axes.Axes.tick_params` method in Matplotlib.\n\n        This is provided for convenience, but the recommended API is to use\n        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticks`,\n        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticklabel`,\n        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticks_position`,\n        :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.set_ticklabel_position`,\n        and :meth:`~astropy.visualization.wcsaxes.CoordinateHelper.grid`.\n\n        Parameters\n        ----------\n        which : {'both', 'major', 'minor'}, optional\n            Which ticks to apply the settings to. By default, setting are\n            applied to both major and minor ticks. Note that if ``'minor'`` is\n            specified, only the length of the ticks can be set currently.\n        direction : {'in', 'out'}, optional\n            Puts ticks inside the axes, or outside the axes.\n        length : float, optional\n            Tick length in points.\n        width : float, optional\n            Tick width in points.\n        color : color, optional\n            Tick color (accepts any valid Matplotlib color)\n        pad : float, optional\n            Distance in points between tick and label.\n        labelsize : float or str, optional\n            Tick label font size in points or as a string (e.g., 'large').\n        labelcolor : color, optional\n            Tick label color (accepts any valid Matplotlib color)\n        colors : color, optional\n            Changes the tick color and the label color to the same value\n             (accepts any valid Matplotlib color).\n        bottom, top, left, right : bool, optional\n            Where to draw the ticks. Note that this will not work correctly if\n            the frame is not rectangular.\n        labelbottom, labeltop, labelleft, labelright : bool, optional\n            Where to draw the tick labels. Note that this will not work\n            correctly if the frame is not rectangular.\n        grid_color : color, optional\n            The color of the grid lines (accepts any valid Matplotlib color).\n        grid_alpha : float, optional\n            Transparency of grid lines: 0 (transparent) to 1 (opaque).\n        grid_linewidth : float, optional\n            Width of grid lines in points.\n        grid_linestyle : str, optional\n            The style of the grid lines (accepts any valid Matplotlib line\n            style).\n        "
        if 'colors' in kwargs:
            if 'color' not in kwargs:
                kwargs['color'] = kwargs['colors']
            if 'labelcolor' not in kwargs:
                kwargs['labelcolor'] = kwargs['colors']
        if which == 'minor':
            if len(set(kwargs) - {'length'}) > 0:
                raise ValueError("When setting which='minor', the only property that can be set at the moment is 'length' (the minor tick length)")
            elif 'length' in kwargs:
                self.ticks.set_minor_ticksize(kwargs['length'])
            return
        self.set_ticks(size=kwargs.get('length'), width=kwargs.get('width'), color=kwargs.get('color'), direction=kwargs.get('direction'))
        position = None
        for arg in ('bottom', 'left', 'top', 'right'):
            if arg in kwargs and position is None:
                position = ''
            if kwargs.get(arg):
                position += arg[0]
        if position is not None:
            self.set_ticks_position(position)
        self.set_ticklabel(color=kwargs.get('labelcolor'), size=kwargs.get('labelsize'), pad=kwargs.get('pad'))
        position = None
        for arg in ('bottom', 'left', 'top', 'right'):
            if 'label' + arg in kwargs and position is None:
                position = ''
            if kwargs.get('label' + arg):
                position += arg[0]
        if position is not None:
            self.set_ticklabel_position(position)
        if 'grid_color' in kwargs:
            self.grid_lines_kwargs['edgecolor'] = kwargs['grid_color']
        if 'grid_alpha' in kwargs:
            self.grid_lines_kwargs['alpha'] = kwargs['grid_alpha']
        if 'grid_linewidth' in kwargs:
            self.grid_lines_kwargs['linewidth'] = kwargs['grid_linewidth']
        if 'grid_linestyle' in kwargs:
            if kwargs['grid_linestyle'] in LINES_TO_PATCHES_LINESTYLE:
                self.grid_lines_kwargs['linestyle'] = LINES_TO_PATCHES_LINESTYLE[kwargs['grid_linestyle']]
            else:
                self.grid_lines_kwargs['linestyle'] = kwargs['grid_linestyle']