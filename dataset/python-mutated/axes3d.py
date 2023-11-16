"""
axes3d.py, original mplot3d version by John Porter
Created: 23 Sep 2005

Parts fixed by Reinier Heeres <reinier@heeres.eu>
Minor additions by Ben Axelrod <baxelrod@coroware.com>
Significant updates and revisions by Ben Root <ben.v.root@gmail.com>

Module containing Axes3D, an object which can plot 3D objects on a
2D matplotlib figure.
"""
from collections import defaultdict
import itertools
import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, _docstring, _preprocess_data
import matplotlib.artist as martist
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.image as mimage
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.container as mcontainer
import matplotlib.transforms as mtransforms
from matplotlib.axes import Axes
from matplotlib.axes._base import _axis_method_wrapper, _process_plot_format
from matplotlib.transforms import Bbox
from matplotlib.tri._triangulation import Triangulation
from . import art3d
from . import proj3d
from . import axis3d

@_docstring.interpd
@_api.define_aliases({'xlim': ['xlim3d'], 'ylim': ['ylim3d'], 'zlim': ['zlim3d']})
class Axes3D(Axes):
    """
    3D Axes object.

    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.
    """
    name = '3d'
    _axis_names = ('x', 'y', 'z')
    Axes._shared_axes['z'] = cbook.Grouper()
    Axes._shared_axes['view'] = cbook.Grouper()

    def __init__(self, fig, rect=None, *args, elev=30, azim=-60, roll=0, sharez=None, proj_type='persp', box_aspect=None, computed_zorder=True, focal_length=None, shareview=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Parameters\n        ----------\n        fig : Figure\n            The parent figure.\n        rect : tuple (left, bottom, width, height), default: None.\n            The ``(left, bottom, width, height)`` axes position.\n        elev : float, default: 30\n            The elevation angle in degrees rotates the camera above and below\n            the x-y plane, with a positive angle corresponding to a location\n            above the plane.\n        azim : float, default: -60\n            The azimuthal angle in degrees rotates the camera about the z axis,\n            with a positive angle corresponding to a right-handed rotation. In\n            other words, a positive azimuth rotates the camera about the origin\n            from its location along the +x axis towards the +y axis.\n        roll : float, default: 0\n            The roll angle in degrees rotates the camera about the viewing\n            axis. A positive angle spins the camera clockwise, causing the\n            scene to rotate counter-clockwise.\n        sharez : Axes3D, optional\n            Other Axes to share z-limits with.\n        proj_type : {'persp', 'ortho'}\n            The projection type, default 'persp'.\n        box_aspect : 3-tuple of floats, default: None\n            Changes the physical dimensions of the Axes3D, such that the ratio\n            of the axis lengths in display units is x:y:z.\n            If None, defaults to 4:4:3\n        computed_zorder : bool, default: True\n            If True, the draw order is computed based on the average position\n            of the `.Artist`\\s along the view direction.\n            Set to False if you want to manually control the order in which\n            Artists are drawn on top of each other using their *zorder*\n            attribute. This can be used for fine-tuning if the automatic order\n            does not produce the desired result. Note however, that a manual\n            zorder will only be correct for a limited view angle. If the figure\n            is rotated by the user, it will look wrong from certain angles.\n        focal_length : float, default: None\n            For a projection type of 'persp', the focal length of the virtual\n            camera. Must be > 0. If None, defaults to 1.\n            For a projection type of 'ortho', must be set to either None\n            or infinity (numpy.inf). If None, defaults to infinity.\n            The focal length can be computed from a desired Field Of View via\n            the equation: focal_length = 1/tan(FOV/2)\n        shareview : Axes3D, optional\n            Other Axes to share view angles with.\n\n        **kwargs\n            Other optional keyword arguments:\n\n            %(Axes3D:kwdoc)s\n        "
        if rect is None:
            rect = [0.0, 0.0, 1.0, 1.0]
        self.initial_azim = azim
        self.initial_elev = elev
        self.initial_roll = roll
        self.set_proj_type(proj_type, focal_length)
        self.computed_zorder = computed_zorder
        self.xy_viewLim = Bbox.unit()
        self.zz_viewLim = Bbox.unit()
        xymargin = 0.05 * 10 / 11
        self.xy_dataLim = Bbox([[xymargin, xymargin], [1 - xymargin, 1 - xymargin]])
        self.zz_dataLim = Bbox.unit()
        self.view_init(self.initial_elev, self.initial_azim, self.initial_roll)
        self._sharez = sharez
        if sharez is not None:
            self._shared_axes['z'].join(self, sharez)
            self._adjustable = 'datalim'
        self._shareview = shareview
        if shareview is not None:
            self._shared_axes['view'].join(self, shareview)
        if kwargs.pop('auto_add_to_figure', False):
            raise AttributeError('auto_add_to_figure is no longer supported for Axes3D. Use fig.add_axes(ax) instead.')
        super().__init__(fig, rect, *args, frameon=True, box_aspect=box_aspect, **kwargs)
        super().set_axis_off()
        self.set_axis_on()
        self.M = None
        self.invM = None
        self._view_margin = 1 / 48
        self.autoscale_view()
        self.fmt_zdata = None
        self.mouse_init()
        self.figure.canvas.callbacks._connect_picklable('motion_notify_event', self._on_move)
        self.figure.canvas.callbacks._connect_picklable('button_press_event', self._button_press)
        self.figure.canvas.callbacks._connect_picklable('button_release_event', self._button_release)
        self.set_top_view()
        self.patch.set_linewidth(0)
        pseudo_bbox = self.transLimits.inverted().transform([(0, 0), (1, 1)])
        (self._pseudo_w, self._pseudo_h) = pseudo_bbox[1] - pseudo_bbox[0]
        self.spines[:].set_visible(False)

    def set_axis_off(self):
        if False:
            return 10
        self._axis3don = False
        self.stale = True

    def set_axis_on(self):
        if False:
            print('Hello World!')
        self._axis3don = True
        self.stale = True

    def convert_zunits(self, z):
        if False:
            print('Hello World!')
        '\n        For artists in an Axes, if the zaxis has units support,\n        convert *z* using zaxis unit type\n        '
        return self.zaxis.convert_units(z)

    def set_top_view(self):
        if False:
            for i in range(10):
                print('nop')
        xdwl = 0.95 / self._dist
        xdw = 0.9 / self._dist
        ydwl = 0.95 / self._dist
        ydw = 0.9 / self._dist
        self.viewLim.intervalx = (-xdwl, xdw)
        self.viewLim.intervaly = (-ydwl, ydw)
        self.stale = True

    def _init_axis(self):
        if False:
            i = 10
            return i + 15
        'Init 3D axes; overrides creation of regular X/Y axes.'
        self.xaxis = axis3d.XAxis(self)
        self.yaxis = axis3d.YAxis(self)
        self.zaxis = axis3d.ZAxis(self)

    def get_zaxis(self):
        if False:
            print('Hello World!')
        'Return the ``ZAxis`` (`~.axis3d.Axis`) instance.'
        return self.zaxis
    get_zgridlines = _axis_method_wrapper('zaxis', 'get_gridlines')
    get_zticklines = _axis_method_wrapper('zaxis', 'get_ticklines')

    def _unit_cube(self, vals=None):
        if False:
            print('Hello World!')
        (minx, maxx, miny, maxy, minz, maxz) = vals or self.get_w_lims()
        return [(minx, miny, minz), (maxx, miny, minz), (maxx, maxy, minz), (minx, maxy, minz), (minx, miny, maxz), (maxx, miny, maxz), (maxx, maxy, maxz), (minx, maxy, maxz)]

    def _tunit_cube(self, vals=None, M=None):
        if False:
            i = 10
            return i + 15
        if M is None:
            M = self.M
        xyzs = self._unit_cube(vals)
        tcube = proj3d._proj_points(xyzs, M)
        return tcube

    def _tunit_edges(self, vals=None, M=None):
        if False:
            i = 10
            return i + 15
        tc = self._tunit_cube(vals, M)
        edges = [(tc[0], tc[1]), (tc[1], tc[2]), (tc[2], tc[3]), (tc[3], tc[0]), (tc[0], tc[4]), (tc[1], tc[5]), (tc[2], tc[6]), (tc[3], tc[7]), (tc[4], tc[5]), (tc[5], tc[6]), (tc[6], tc[7]), (tc[7], tc[4])]
        return edges

    def set_aspect(self, aspect, adjustable=None, anchor=None, share=False):
        if False:
            while True:
                i = 10
        "\n        Set the aspect ratios.\n\n        Parameters\n        ----------\n        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}\n            Possible values:\n\n            =========   ==================================================\n            value       description\n            =========   ==================================================\n            'auto'      automatic; fill the position rectangle with data.\n            'equal'     adapt all the axes to have equal aspect ratios.\n            'equalxy'   adapt the x and y axes to have equal aspect ratios.\n            'equalxz'   adapt the x and z axes to have equal aspect ratios.\n            'equalyz'   adapt the y and z axes to have equal aspect ratios.\n            =========   ==================================================\n\n        adjustable : None or {'box', 'datalim'}, optional\n            If not *None*, this defines which parameter will be adjusted to\n            meet the required aspect. See `.set_adjustable` for further\n            details.\n\n        anchor : None or str or 2-tuple of float, optional\n            If not *None*, this defines where the Axes will be drawn if there\n            is extra space due to aspect constraints. The most common way to\n            specify the anchor are abbreviations of cardinal directions:\n\n            =====   =====================\n            value   description\n            =====   =====================\n            'C'     centered\n            'SW'    lower left corner\n            'S'     middle of bottom edge\n            'SE'    lower right corner\n            etc.\n            =====   =====================\n\n            See `~.Axes.set_anchor` for further details.\n\n        share : bool, default: False\n            If ``True``, apply the settings to all shared Axes.\n\n        See Also\n        --------\n        mpl_toolkits.mplot3d.axes3d.Axes3D.set_box_aspect\n        "
        _api.check_in_list(('auto', 'equal', 'equalxy', 'equalyz', 'equalxz'), aspect=aspect)
        super().set_aspect(aspect='auto', adjustable=adjustable, anchor=anchor, share=share)
        self._aspect = aspect
        if aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
            ax_indices = self._equal_aspect_axis_indices(aspect)
            view_intervals = np.array([self.xaxis.get_view_interval(), self.yaxis.get_view_interval(), self.zaxis.get_view_interval()])
            ptp = np.ptp(view_intervals, axis=1)
            if self._adjustable == 'datalim':
                mean = np.mean(view_intervals, axis=1)
                scale = max(ptp[ax_indices] / self._box_aspect[ax_indices])
                deltas = scale * self._box_aspect
                for (i, set_lim) in enumerate((self.set_xlim3d, self.set_ylim3d, self.set_zlim3d)):
                    if i in ax_indices:
                        set_lim(mean[i] - deltas[i] / 2.0, mean[i] + deltas[i] / 2.0, auto=True, view_margin=None)
            else:
                box_aspect = np.array(self._box_aspect)
                box_aspect[ax_indices] = ptp[ax_indices]
                remaining_ax_indices = {0, 1, 2}.difference(ax_indices)
                if remaining_ax_indices:
                    remaining = remaining_ax_indices.pop()
                    old_diag = np.linalg.norm(self._box_aspect[ax_indices])
                    new_diag = np.linalg.norm(box_aspect[ax_indices])
                    box_aspect[remaining] *= new_diag / old_diag
                self.set_box_aspect(box_aspect)

    def _equal_aspect_axis_indices(self, aspect):
        if False:
            while True:
                i = 10
        "\n        Get the indices for which of the x, y, z axes are constrained to have\n        equal aspect ratios.\n\n        Parameters\n        ----------\n        aspect : {'auto', 'equal', 'equalxy', 'equalxz', 'equalyz'}\n            See descriptions in docstring for `.set_aspect()`.\n        "
        ax_indices = []
        if aspect == 'equal':
            ax_indices = [0, 1, 2]
        elif aspect == 'equalxy':
            ax_indices = [0, 1]
        elif aspect == 'equalxz':
            ax_indices = [0, 2]
        elif aspect == 'equalyz':
            ax_indices = [1, 2]
        return ax_indices

    def set_box_aspect(self, aspect, *, zoom=1):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the Axes box aspect.\n\n        The box aspect is the ratio of height to width in display\n        units for each face of the box when viewed perpendicular to\n        that face.  This is not to be confused with the data aspect (see\n        `~.Axes3D.set_aspect`). The default ratios are 4:4:3 (x:y:z).\n\n        To simulate having equal aspect in data space, set the box\n        aspect to match your data range in each dimension.\n\n        *zoom* controls the overall size of the Axes3D in the figure.\n\n        Parameters\n        ----------\n        aspect : 3-tuple of floats or None\n            Changes the physical dimensions of the Axes3D, such that the ratio\n            of the axis lengths in display units is x:y:z.\n            If None, defaults to (4, 4, 3).\n\n        zoom : float, default: 1\n            Control overall size of the Axes3D in the figure. Must be > 0.\n        '
        if zoom <= 0:
            raise ValueError(f'Argument zoom = {zoom} must be > 0')
        if aspect is None:
            aspect = np.asarray((4, 4, 3), dtype=float)
        else:
            aspect = np.asarray(aspect, dtype=float)
            _api.check_shape((3,), aspect=aspect)
        aspect *= 1.8294640721620434 * 25 / 24 * zoom / np.linalg.norm(aspect)
        self._box_aspect = aspect
        self.stale = True

    def apply_aspect(self, position=None):
        if False:
            i = 10
            return i + 15
        if position is None:
            position = self.get_position(original=True)
        trans = self.get_figure().transSubfigure
        bb = mtransforms.Bbox.unit().transformed(trans)
        fig_aspect = bb.height / bb.width
        box_aspect = 1
        pb = position.frozen()
        pb1 = pb.shrunk_to_aspect(box_aspect, pb, fig_aspect)
        self._set_position(pb1.anchored(self.get_anchor(), pb), 'active')

    @martist.allow_rasterization
    def draw(self, renderer):
        if False:
            return 10
        if not self.get_visible():
            return
        self._unstale_viewLim()
        self.patch.draw(renderer)
        self._frameon = False
        locator = self.get_axes_locator()
        self.apply_aspect(locator(self, renderer) if locator else None)
        self.M = self.get_proj()
        self.invM = np.linalg.inv(self.M)
        collections_and_patches = (artist for artist in self._children if isinstance(artist, (mcoll.Collection, mpatches.Patch)) and artist.get_visible())
        if self.computed_zorder:
            zorder_offset = max((axis.get_zorder() for axis in self._axis_map.values())) + 1
            collection_zorder = patch_zorder = zorder_offset
            for artist in sorted(collections_and_patches, key=lambda artist: artist.do_3d_projection(), reverse=True):
                if isinstance(artist, mcoll.Collection):
                    artist.zorder = collection_zorder
                    collection_zorder += 1
                elif isinstance(artist, mpatches.Patch):
                    artist.zorder = patch_zorder
                    patch_zorder += 1
        else:
            for artist in collections_and_patches:
                artist.do_3d_projection()
        if self._axis3don:
            for axis in self._axis_map.values():
                axis.draw_pane(renderer)
            for axis in self._axis_map.values():
                axis.draw_grid(renderer)
            for axis in self._axis_map.values():
                axis.draw(renderer)
        super().draw(renderer)

    def get_axis_position(self):
        if False:
            print('Hello World!')
        vals = self.get_w_lims()
        tc = self._tunit_cube(vals, self.M)
        xhigh = tc[1][2] > tc[2][2]
        yhigh = tc[3][2] > tc[2][2]
        zhigh = tc[0][2] > tc[2][2]
        return (xhigh, yhigh, zhigh)

    def update_datalim(self, xys, **kwargs):
        if False:
            print('Hello World!')
        '\n        Not implemented in `~mpl_toolkits.mplot3d.axes3d.Axes3D`.\n        '
        pass
    get_autoscalez_on = _axis_method_wrapper('zaxis', '_get_autoscale_on')
    set_autoscalez_on = _axis_method_wrapper('zaxis', '_set_autoscale_on')

    def get_zmargin(self):
        if False:
            while True:
                i = 10
        '\n        Retrieve autoscaling margin of the z-axis.\n\n        .. versionadded:: 3.9\n\n        Returns\n        -------\n        zmargin : float\n\n        See Also\n        --------\n        mpl_toolkits.mplot3d.axes3d.Axes3D.set_zmargin\n        '
        return self._zmargin

    def set_zmargin(self, m):
        if False:
            while True:
                i = 10
        '\n        Set padding of Z data limits prior to autoscaling.\n\n        *m* times the data interval will be added to each end of that interval\n        before it is used in autoscaling.  If *m* is negative, this will clip\n        the data range instead of expanding it.\n\n        For example, if your data is in the range [0, 2], a margin of 0.1 will\n        result in a range [-0.2, 2.2]; a margin of -0.1 will result in a range\n        of [0.2, 1.8].\n\n        Parameters\n        ----------\n        m : float greater than -0.5\n        '
        if m <= -0.5:
            raise ValueError('margin must be greater than -0.5')
        self._zmargin = m
        self._request_autoscale_view('z')
        self.stale = True

    def margins(self, *margins, x=None, y=None, z=None, tight=True):
        if False:
            i = 10
            return i + 15
        '\n        Set or retrieve autoscaling margins.\n\n        See `.Axes.margins` for full documentation.  Because this function\n        applies to 3D Axes, it also takes a *z* argument, and returns\n        ``(xmargin, ymargin, zmargin)``.\n        '
        if margins and (x is not None or y is not None or z is not None):
            raise TypeError('Cannot pass both positional and keyword arguments for x, y, and/or z.')
        elif len(margins) == 1:
            x = y = z = margins[0]
        elif len(margins) == 3:
            (x, y, z) = margins
        elif margins:
            raise TypeError('Must pass a single positional argument for all margins, or one for each margin (x, y, z).')
        if x is None and y is None and (z is None):
            if tight is not True:
                _api.warn_external(f'ignoring tight={tight!r} in get mode')
            return (self._xmargin, self._ymargin, self._zmargin)
        if x is not None:
            self.set_xmargin(x)
        if y is not None:
            self.set_ymargin(y)
        if z is not None:
            self.set_zmargin(z)
        self.autoscale_view(tight=tight, scalex=x is not None, scaley=y is not None, scalez=z is not None)

    def autoscale(self, enable=True, axis='both', tight=None):
        if False:
            return 10
        "\n        Convenience method for simple axis view autoscaling.\n\n        See `.Axes.autoscale` for full documentation.  Because this function\n        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*\n        to 'both' autoscales all three axes.\n        "
        if enable is None:
            scalex = True
            scaley = True
            scalez = True
        else:
            if axis in ['x', 'both']:
                self.set_autoscalex_on(enable)
                scalex = self.get_autoscalex_on()
            else:
                scalex = False
            if axis in ['y', 'both']:
                self.set_autoscaley_on(enable)
                scaley = self.get_autoscaley_on()
            else:
                scaley = False
            if axis in ['z', 'both']:
                self.set_autoscalez_on(enable)
                scalez = self.get_autoscalez_on()
            else:
                scalez = False
        if scalex:
            self._request_autoscale_view('x', tight=tight)
        if scaley:
            self._request_autoscale_view('y', tight=tight)
        if scalez:
            self._request_autoscale_view('z', tight=tight)

    def auto_scale_xyz(self, X, Y, Z=None, had_data=None):
        if False:
            print('Hello World!')
        if np.shape(X) == np.shape(Y):
            self.xy_dataLim.update_from_data_xy(np.column_stack([np.ravel(X), np.ravel(Y)]), not had_data)
        else:
            self.xy_dataLim.update_from_data_x(X, not had_data)
            self.xy_dataLim.update_from_data_y(Y, not had_data)
        if Z is not None:
            self.zz_dataLim.update_from_data_x(Z, not had_data)
        self.autoscale_view()

    def autoscale_view(self, tight=None, scalex=True, scaley=True, scalez=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Autoscale the view limits using the data limits.\n\n        See `.Axes.autoscale_view` for full documentation.  Because this\n        function applies to 3D Axes, it also takes a *scalez* argument.\n        '
        if tight is None:
            _tight = self._tight
            if not _tight:
                for artist in self._children:
                    if isinstance(artist, mimage.AxesImage):
                        _tight = True
                    elif isinstance(artist, (mlines.Line2D, mpatches.Patch)):
                        _tight = False
                        break
        else:
            _tight = self._tight = bool(tight)
        if scalex and self.get_autoscalex_on():
            (x0, x1) = self.xy_dataLim.intervalx
            xlocator = self.xaxis.get_major_locator()
            (x0, x1) = xlocator.nonsingular(x0, x1)
            if self._xmargin > 0:
                delta = (x1 - x0) * self._xmargin
                x0 -= delta
                x1 += delta
            if not _tight:
                (x0, x1) = xlocator.view_limits(x0, x1)
            self.set_xbound(x0, x1, self._view_margin)
        if scaley and self.get_autoscaley_on():
            (y0, y1) = self.xy_dataLim.intervaly
            ylocator = self.yaxis.get_major_locator()
            (y0, y1) = ylocator.nonsingular(y0, y1)
            if self._ymargin > 0:
                delta = (y1 - y0) * self._ymargin
                y0 -= delta
                y1 += delta
            if not _tight:
                (y0, y1) = ylocator.view_limits(y0, y1)
            self.set_ybound(y0, y1, self._view_margin)
        if scalez and self.get_autoscalez_on():
            (z0, z1) = self.zz_dataLim.intervalx
            zlocator = self.zaxis.get_major_locator()
            (z0, z1) = zlocator.nonsingular(z0, z1)
            if self._zmargin > 0:
                delta = (z1 - z0) * self._zmargin
                z0 -= delta
                z1 += delta
            if not _tight:
                (z0, z1) = zlocator.view_limits(z0, z1)
            self.set_zbound(z0, z1, self._view_margin)

    def get_w_lims(self):
        if False:
            return 10
        'Get 3D world limits.'
        (minx, maxx) = self.get_xlim3d()
        (miny, maxy) = self.get_ylim3d()
        (minz, maxz) = self.get_zlim3d()
        return (minx, maxx, miny, maxy, minz, maxz)

    def _set_bound3d(self, get_bound, set_lim, axis_inverted, lower=None, upper=None, view_margin=None):
        if False:
            i = 10
            return i + 15
        '\n        Set 3D axis bounds.\n        '
        if upper is None and np.iterable(lower):
            (lower, upper) = lower
        (old_lower, old_upper) = get_bound()
        if lower is None:
            lower = old_lower
        if upper is None:
            upper = old_upper
        set_lim(sorted((lower, upper), reverse=bool(axis_inverted())), auto=None, view_margin=view_margin)

    def set_xbound(self, lower=None, upper=None, view_margin=None):
        if False:
            while True:
                i = 10
        '\n        Set the lower and upper numerical bounds of the x-axis.\n\n        This method will honor axis inversion regardless of parameter order.\n        It will not change the autoscaling setting (`.get_autoscalex_on()`).\n\n        Parameters\n        ----------\n        lower, upper : float or None\n            The lower and upper bounds. If *None*, the respective axis bound\n            is not modified.\n        view_margin : float or None\n            The margin to apply to the bounds. If *None*, the margin is handled\n            by `.set_xlim`.\n\n        See Also\n        --------\n        get_xbound\n        get_xlim, set_xlim\n        invert_xaxis, xaxis_inverted\n        '
        self._set_bound3d(self.get_xbound, self.set_xlim, self.xaxis_inverted, lower, upper, view_margin)

    def set_ybound(self, lower=None, upper=None, view_margin=None):
        if False:
            print('Hello World!')
        '\n        Set the lower and upper numerical bounds of the y-axis.\n\n        This method will honor axis inversion regardless of parameter order.\n        It will not change the autoscaling setting (`.get_autoscaley_on()`).\n\n        Parameters\n        ----------\n        lower, upper : float or None\n            The lower and upper bounds. If *None*, the respective axis bound\n            is not modified.\n        view_margin : float or None\n            The margin to apply to the bounds. If *None*, the margin is handled\n            by `.set_ylim`.\n\n        See Also\n        --------\n        get_ybound\n        get_ylim, set_ylim\n        invert_yaxis, yaxis_inverted\n        '
        self._set_bound3d(self.get_ybound, self.set_ylim, self.yaxis_inverted, lower, upper, view_margin)

    def set_zbound(self, lower=None, upper=None, view_margin=None):
        if False:
            print('Hello World!')
        '\n        Set the lower and upper numerical bounds of the z-axis.\n        This method will honor axis inversion regardless of parameter order.\n        It will not change the autoscaling setting (`.get_autoscaley_on()`).\n\n        Parameters\n        ----------\n        lower, upper : float or None\n            The lower and upper bounds. If *None*, the respective axis bound\n            is not modified.\n        view_margin : float or None\n            The margin to apply to the bounds. If *None*, the margin is handled\n            by `.set_zlim`.\n\n        See Also\n        --------\n        get_zbound\n        get_zlim, set_zlim\n        invert_zaxis, zaxis_inverted\n        '
        self._set_bound3d(self.get_zbound, self.set_zlim, self.zaxis_inverted, lower, upper, view_margin)

    def _set_lim3d(self, axis, lower=None, upper=None, *, emit=True, auto=False, view_margin=None, axmin=None, axmax=None):
        if False:
            print('Hello World!')
        '\n        Set 3D axis limits.\n        '
        if upper is None:
            if np.iterable(lower):
                (lower, upper) = lower
            elif axmax is None:
                upper = axis.get_view_interval()[1]
        if lower is None and axmin is None:
            lower = axis.get_view_interval()[0]
        if axmin is not None:
            if lower is not None:
                raise TypeError("Cannot pass both 'lower' and 'min'")
            lower = axmin
        if axmax is not None:
            if upper is not None:
                raise TypeError("Cannot pass both 'upper' and 'max'")
            upper = axmax
        if np.isinf(lower) or np.isinf(upper):
            raise ValueError(f'Axis limits {lower}, {upper} cannot be infinite')
        if view_margin is None:
            if mpl.rcParams['axes3d.automargin']:
                view_margin = self._view_margin
            else:
                view_margin = 0
        delta = (upper - lower) * view_margin
        lower -= delta
        upper += delta
        return axis._set_lim(lower, upper, emit=emit, auto=auto)

    def set_xlim(self, left=None, right=None, *, emit=True, auto=False, view_margin=None, xmin=None, xmax=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the 3D x-axis view limits.\n\n        Parameters\n        ----------\n        left : float, optional\n            The left xlim in data coordinates. Passing *None* leaves the\n            limit unchanged.\n\n            The left and right xlims may also be passed as the tuple\n            (*left*, *right*) as the first positional argument (or as\n            the *left* keyword argument).\n\n            .. ACCEPTS: (left: float, right: float)\n\n        right : float, optional\n            The right xlim in data coordinates. Passing *None* leaves the\n            limit unchanged.\n\n        emit : bool, default: True\n            Whether to notify observers of limit change.\n\n        auto : bool or None, default: False\n            Whether to turn on autoscaling of the x-axis. *True* turns on,\n            *False* turns off, *None* leaves unchanged.\n\n        view_margin : float, optional\n            The additional margin to apply to the limits.\n\n        xmin, xmax : float, optional\n            They are equivalent to left and right respectively, and it is an\n            error to pass both *xmin* and *left* or *xmax* and *right*.\n\n        Returns\n        -------\n        left, right : (float, float)\n            The new x-axis limits in data coordinates.\n\n        See Also\n        --------\n        get_xlim\n        set_xbound, get_xbound\n        invert_xaxis, xaxis_inverted\n\n        Notes\n        -----\n        The *left* value may be greater than the *right* value, in which\n        case the x-axis values will decrease from *left* to *right*.\n\n        Examples\n        --------\n        >>> set_xlim(left, right)\n        >>> set_xlim((left, right))\n        >>> left, right = set_xlim(left, right)\n\n        One limit may be left unchanged.\n\n        >>> set_xlim(right=right_lim)\n\n        Limits may be passed in reverse order to flip the direction of\n        the x-axis. For example, suppose ``x`` represents depth of the\n        ocean in m. The x-axis limits might be set like the following\n        so 5000 m depth is at the left of the plot and the surface,\n        0 m, is at the right.\n\n        >>> set_xlim(5000, 0)\n        '
        return self._set_lim3d(self.xaxis, left, right, emit=emit, auto=auto, view_margin=view_margin, axmin=xmin, axmax=xmax)

    def set_ylim(self, bottom=None, top=None, *, emit=True, auto=False, view_margin=None, ymin=None, ymax=None):
        if False:
            print('Hello World!')
        '\n        Set the 3D y-axis view limits.\n\n        Parameters\n        ----------\n        bottom : float, optional\n            The bottom ylim in data coordinates. Passing *None* leaves the\n            limit unchanged.\n\n            The bottom and top ylims may also be passed as the tuple\n            (*bottom*, *top*) as the first positional argument (or as\n            the *bottom* keyword argument).\n\n            .. ACCEPTS: (bottom: float, top: float)\n\n        top : float, optional\n            The top ylim in data coordinates. Passing *None* leaves the\n            limit unchanged.\n\n        emit : bool, default: True\n            Whether to notify observers of limit change.\n\n        auto : bool or None, default: False\n            Whether to turn on autoscaling of the y-axis. *True* turns on,\n            *False* turns off, *None* leaves unchanged.\n\n        view_margin : float, optional\n            The additional margin to apply to the limits.\n\n        ymin, ymax : float, optional\n            They are equivalent to bottom and top respectively, and it is an\n            error to pass both *ymin* and *bottom* or *ymax* and *top*.\n\n        Returns\n        -------\n        bottom, top : (float, float)\n            The new y-axis limits in data coordinates.\n\n        See Also\n        --------\n        get_ylim\n        set_ybound, get_ybound\n        invert_yaxis, yaxis_inverted\n\n        Notes\n        -----\n        The *bottom* value may be greater than the *top* value, in which\n        case the y-axis values will decrease from *bottom* to *top*.\n\n        Examples\n        --------\n        >>> set_ylim(bottom, top)\n        >>> set_ylim((bottom, top))\n        >>> bottom, top = set_ylim(bottom, top)\n\n        One limit may be left unchanged.\n\n        >>> set_ylim(top=top_lim)\n\n        Limits may be passed in reverse order to flip the direction of\n        the y-axis. For example, suppose ``y`` represents depth of the\n        ocean in m. The y-axis limits might be set like the following\n        so 5000 m depth is at the bottom of the plot and the surface,\n        0 m, is at the top.\n\n        >>> set_ylim(5000, 0)\n        '
        return self._set_lim3d(self.yaxis, bottom, top, emit=emit, auto=auto, view_margin=view_margin, axmin=ymin, axmax=ymax)

    def set_zlim(self, bottom=None, top=None, *, emit=True, auto=False, view_margin=None, zmin=None, zmax=None):
        if False:
            print('Hello World!')
        '\n        Set the 3D z-axis view limits.\n\n        Parameters\n        ----------\n        bottom : float, optional\n            The bottom zlim in data coordinates. Passing *None* leaves the\n            limit unchanged.\n\n            The bottom and top zlims may also be passed as the tuple\n            (*bottom*, *top*) as the first positional argument (or as\n            the *bottom* keyword argument).\n\n            .. ACCEPTS: (bottom: float, top: float)\n\n        top : float, optional\n            The top zlim in data coordinates. Passing *None* leaves the\n            limit unchanged.\n\n        emit : bool, default: True\n            Whether to notify observers of limit change.\n\n        auto : bool or None, default: False\n            Whether to turn on autoscaling of the z-axis. *True* turns on,\n            *False* turns off, *None* leaves unchanged.\n\n        view_margin : float, optional\n            The additional margin to apply to the limits.\n\n        zmin, zmax : float, optional\n            They are equivalent to bottom and top respectively, and it is an\n            error to pass both *zmin* and *bottom* or *zmax* and *top*.\n\n        Returns\n        -------\n        bottom, top : (float, float)\n            The new z-axis limits in data coordinates.\n\n        See Also\n        --------\n        get_zlim\n        set_zbound, get_zbound\n        invert_zaxis, zaxis_inverted\n\n        Notes\n        -----\n        The *bottom* value may be greater than the *top* value, in which\n        case the z-axis values will decrease from *bottom* to *top*.\n\n        Examples\n        --------\n        >>> set_zlim(bottom, top)\n        >>> set_zlim((bottom, top))\n        >>> bottom, top = set_zlim(bottom, top)\n\n        One limit may be left unchanged.\n\n        >>> set_zlim(top=top_lim)\n\n        Limits may be passed in reverse order to flip the direction of\n        the z-axis. For example, suppose ``z`` represents depth of the\n        ocean in m. The z-axis limits might be set like the following\n        so 5000 m depth is at the bottom of the plot and the surface,\n        0 m, is at the top.\n\n        >>> set_zlim(5000, 0)\n        '
        return self._set_lim3d(self.zaxis, bottom, top, emit=emit, auto=auto, view_margin=view_margin, axmin=zmin, axmax=zmax)
    set_xlim3d = set_xlim
    set_ylim3d = set_ylim
    set_zlim3d = set_zlim

    def get_xlim(self):
        if False:
            while True:
                i = 10
        return tuple(self.xy_viewLim.intervalx)

    def get_ylim(self):
        if False:
            i = 10
            return i + 15
        return tuple(self.xy_viewLim.intervaly)

    def get_zlim(self):
        if False:
            while True:
                i = 10
        '\n        Return the 3D z-axis view limits.\n\n        Returns\n        -------\n        left, right : (float, float)\n            The current z-axis limits in data coordinates.\n\n        See Also\n        --------\n        set_zlim\n        set_zbound, get_zbound\n        invert_zaxis, zaxis_inverted\n\n        Notes\n        -----\n        The z-axis may be inverted, in which case the *left* value will\n        be greater than the *right* value.\n        '
        return tuple(self.zz_viewLim.intervalx)
    get_zscale = _axis_method_wrapper('zaxis', 'get_scale')
    set_xscale = _axis_method_wrapper('xaxis', '_set_axes_scale')
    set_yscale = _axis_method_wrapper('yaxis', '_set_axes_scale')
    set_zscale = _axis_method_wrapper('zaxis', '_set_axes_scale')
    (set_xscale.__doc__, set_yscale.__doc__, set_zscale.__doc__) = map('\n        Set the {}-axis scale.\n\n        Parameters\n        ----------\n        value : {{"linear"}}\n            The axis scale type to apply.  3D axes currently only support\n            linear scales; other scales yield nonsensical results.\n\n        **kwargs\n            Keyword arguments are nominally forwarded to the scale class, but\n            none of them is applicable for linear scales.\n        '.format, ['x', 'y', 'z'])
    get_zticks = _axis_method_wrapper('zaxis', 'get_ticklocs')
    set_zticks = _axis_method_wrapper('zaxis', 'set_ticks')
    get_zmajorticklabels = _axis_method_wrapper('zaxis', 'get_majorticklabels')
    get_zminorticklabels = _axis_method_wrapper('zaxis', 'get_minorticklabels')
    get_zticklabels = _axis_method_wrapper('zaxis', 'get_ticklabels')
    set_zticklabels = _axis_method_wrapper('zaxis', 'set_ticklabels', doc_sub={'Axis.set_ticks': 'Axes3D.set_zticks'})
    zaxis_date = _axis_method_wrapper('zaxis', 'axis_date')
    if zaxis_date.__doc__:
        zaxis_date.__doc__ += textwrap.dedent('\n\n        Notes\n        -----\n        This function is merely provided for completeness, but 3D axes do not\n        support dates for ticks, and so this may not work as expected.\n        ')

    def clabel(self, *args, **kwargs):
        if False:
            print('Hello World!')
        'Currently not implemented for 3D axes, and returns *None*.'
        return None

    def view_init(self, elev=None, azim=None, roll=None, vertical_axis='z', share=False):
        if False:
            i = 10
            return i + 15
        '\n        Set the elevation and azimuth of the axes in degrees (not radians).\n\n        This can be used to rotate the axes programmatically.\n\n        To look normal to the primary planes, the following elevation and\n        azimuth angles can be used. A roll angle of 0, 90, 180, or 270 deg\n        will rotate these views while keeping the axes at right angles.\n\n        ==========   ====  ====\n        view plane   elev  azim\n        ==========   ====  ====\n        XY           90    -90\n        XZ           0     -90\n        YZ           0     0\n        -XY          -90   90\n        -XZ          0     90\n        -YZ          0     180\n        ==========   ====  ====\n\n        Parameters\n        ----------\n        elev : float, default: None\n            The elevation angle in degrees rotates the camera above the plane\n            pierced by the vertical axis, with a positive angle corresponding\n            to a location above that plane. For example, with the default\n            vertical axis of \'z\', the elevation defines the angle of the camera\n            location above the x-y plane.\n            If None, then the initial value as specified in the `Axes3D`\n            constructor is used.\n        azim : float, default: None\n            The azimuthal angle in degrees rotates the camera about the\n            vertical axis, with a positive angle corresponding to a\n            right-handed rotation. For example, with the default vertical axis\n            of \'z\', a positive azimuth rotates the camera about the origin from\n            its location along the +x axis towards the +y axis.\n            If None, then the initial value as specified in the `Axes3D`\n            constructor is used.\n        roll : float, default: None\n            The roll angle in degrees rotates the camera about the viewing\n            axis. A positive angle spins the camera clockwise, causing the\n            scene to rotate counter-clockwise.\n            If None, then the initial value as specified in the `Axes3D`\n            constructor is used.\n        vertical_axis : {"z", "x", "y"}, default: "z"\n            The axis to align vertically. *azim* rotates about this axis.\n        share : bool, default: False\n            If ``True``, apply the settings to all Axes with shared views.\n        '
        self._dist = 10
        if elev is None:
            elev = self.initial_elev
        if azim is None:
            azim = self.initial_azim
        if roll is None:
            roll = self.initial_roll
        vertical_axis = _api.check_getitem(dict(x=0, y=1, z=2), vertical_axis=vertical_axis)
        if share:
            axes = {sibling for sibling in self._shared_axes['view'].get_siblings(self)}
        else:
            axes = [self]
        for ax in axes:
            ax.elev = elev
            ax.azim = azim
            ax.roll = roll
            ax._vertical_axis = vertical_axis

    def set_proj_type(self, proj_type, focal_length=None):
        if False:
            print('Hello World!')
        "\n        Set the projection type.\n\n        Parameters\n        ----------\n        proj_type : {'persp', 'ortho'}\n            The projection type.\n        focal_length : float, default: None\n            For a projection type of 'persp', the focal length of the virtual\n            camera. Must be > 0. If None, defaults to 1.\n            The focal length can be computed from a desired Field Of View via\n            the equation: focal_length = 1/tan(FOV/2)\n        "
        _api.check_in_list(['persp', 'ortho'], proj_type=proj_type)
        if proj_type == 'persp':
            if focal_length is None:
                focal_length = 1
            elif focal_length <= 0:
                raise ValueError(f'focal_length = {focal_length} must be greater than 0')
            self._focal_length = focal_length
        else:
            if focal_length not in (None, np.inf):
                raise ValueError(f'focal_length = {focal_length} must be None for proj_type = {proj_type}')
            self._focal_length = np.inf

    def _roll_to_vertical(self, arr):
        if False:
            while True:
                i = 10
        'Roll arrays to match the different vertical axis.'
        return np.roll(arr, self._vertical_axis - 2)

    def get_proj(self):
        if False:
            while True:
                i = 10
        'Create the projection matrix from the current viewing position.'
        box_aspect = self._roll_to_vertical(self._box_aspect)
        worldM = proj3d.world_transformation(*self.get_xlim3d(), *self.get_ylim3d(), *self.get_zlim3d(), pb_aspect=box_aspect)
        R = 0.5 * box_aspect
        elev_rad = np.deg2rad(self.elev)
        azim_rad = np.deg2rad(self.azim)
        p0 = np.cos(elev_rad) * np.cos(azim_rad)
        p1 = np.cos(elev_rad) * np.sin(azim_rad)
        p2 = np.sin(elev_rad)
        ps = self._roll_to_vertical([p0, p1, p2])
        eye = R + self._dist * ps
        (u, v, w) = self._calc_view_axes(eye)
        self._view_u = u
        self._view_v = v
        self._view_w = w
        if self._focal_length == np.inf:
            viewM = proj3d._view_transformation_uvw(u, v, w, eye)
            projM = proj3d._ortho_transformation(-self._dist, self._dist)
        else:
            eye_focal = R + self._dist * ps * self._focal_length
            viewM = proj3d._view_transformation_uvw(u, v, w, eye_focal)
            projM = proj3d._persp_transformation(-self._dist, self._dist, self._focal_length)
        M0 = np.dot(viewM, worldM)
        M = np.dot(projM, M0)
        return M

    def mouse_init(self, rotate_btn=1, pan_btn=2, zoom_btn=3):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the mouse buttons for 3D rotation and zooming.\n\n        Parameters\n        ----------\n        rotate_btn : int or list of int, default: 1\n            The mouse button or buttons to use for 3D rotation of the axes.\n        pan_btn : int or list of int, default: 2\n            The mouse button or buttons to use to pan the 3D axes.\n        zoom_btn : int or list of int, default: 3\n            The mouse button or buttons to use to zoom the 3D axes.\n        '
        self.button_pressed = None
        self._rotate_btn = np.atleast_1d(rotate_btn).tolist()
        self._pan_btn = np.atleast_1d(pan_btn).tolist()
        self._zoom_btn = np.atleast_1d(zoom_btn).tolist()

    def disable_mouse_rotation(self):
        if False:
            i = 10
            return i + 15
        'Disable mouse buttons for 3D rotation, panning, and zooming.'
        self.mouse_init(rotate_btn=[], pan_btn=[], zoom_btn=[])

    def can_zoom(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def can_pan(self):
        if False:
            i = 10
            return i + 15
        return True

    def sharez(self, other):
        if False:
            for i in range(10):
                print('nop')
        '\n        Share the z-axis with *other*.\n\n        This is equivalent to passing ``sharez=other`` when constructing the\n        Axes, and cannot be used if the z-axis is already being shared with\n        another Axes.\n        '
        _api.check_isinstance(Axes3D, other=other)
        if self._sharez is not None and other is not self._sharez:
            raise ValueError('z-axis is already shared')
        self._shared_axes['z'].join(self, other)
        self._sharez = other
        self.zaxis.major = other.zaxis.major
        self.zaxis.minor = other.zaxis.minor
        (z0, z1) = other.get_zlim()
        self.set_zlim(z0, z1, emit=False, auto=other.get_autoscalez_on())
        self.zaxis._scale = other.zaxis._scale

    def shareview(self, other):
        if False:
            return 10
        '\n        Share the view angles with *other*.\n\n        This is equivalent to passing ``shareview=other`` when\n        constructing the Axes, and cannot be used if the view angles are\n        already being shared with another Axes.\n        '
        _api.check_isinstance(Axes3D, other=other)
        if self._shareview is not None and other is not self._shareview:
            raise ValueError('view angles are already shared')
        self._shared_axes['view'].join(self, other)
        self._shareview = other
        vertical_axis = {0: 'x', 1: 'y', 2: 'z'}[other._vertical_axis]
        self.view_init(elev=other.elev, azim=other.azim, roll=other.roll, vertical_axis=vertical_axis, share=True)

    def clear(self):
        if False:
            return 10
        super().clear()
        if self._focal_length == np.inf:
            self._zmargin = mpl.rcParams['axes.zmargin']
        else:
            self._zmargin = 0.0
        xymargin = 0.05 * 10 / 11
        self.xy_dataLim = Bbox([[xymargin, xymargin], [1 - xymargin, 1 - xymargin]])
        self.zz_dataLim = Bbox.unit()
        self._view_margin = 1 / 48
        self.autoscale_view()
        self.grid(mpl.rcParams['axes3d.grid'])

    def _button_press(self, event):
        if False:
            print('Hello World!')
        if event.inaxes == self:
            self.button_pressed = event.button
            (self._sx, self._sy) = (event.xdata, event.ydata)
            toolbar = self.figure.canvas.toolbar
            if toolbar and toolbar._nav_stack() is None:
                toolbar.push_current()

    def _button_release(self, event):
        if False:
            i = 10
            return i + 15
        self.button_pressed = None
        toolbar = self.figure.canvas.toolbar
        if toolbar and self.get_navigate_mode() is None:
            toolbar.push_current()

    def _get_view(self):
        if False:
            i = 10
            return i + 15
        return ({'xlim': self.get_xlim(), 'autoscalex_on': self.get_autoscalex_on(), 'ylim': self.get_ylim(), 'autoscaley_on': self.get_autoscaley_on(), 'zlim': self.get_zlim(), 'autoscalez_on': self.get_autoscalez_on()}, (self.elev, self.azim, self.roll))

    def _set_view(self, view):
        if False:
            return 10
        (props, (elev, azim, roll)) = view
        self.set(**props)
        self.elev = elev
        self.azim = azim
        self.roll = roll

    def format_zdata(self, z):
        if False:
            print('Hello World!')
        '\n        Return *z* string formatted.  This function will use the\n        :attr:`fmt_zdata` attribute if it is callable, else will fall\n        back on the zaxis major formatter\n        '
        try:
            return self.fmt_zdata(z)
        except (AttributeError, TypeError):
            func = self.zaxis.get_major_formatter().format_data_short
            val = func(z)
            return val

    def format_coord(self, xv, yv, renderer=None):
        if False:
            while True:
                i = 10
        '\n        Return a string giving the current view rotation angles, or the x, y, z\n        coordinates of the point on the nearest axis pane underneath the mouse\n        cursor, depending on the mouse button pressed.\n        '
        coords = ''
        if self.button_pressed in self._rotate_btn:
            coords = self._rotation_coords()
        elif self.M is not None:
            coords = self._location_coords(xv, yv, renderer)
        return coords

    def _rotation_coords(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the rotation angles as a string.\n        '
        norm_elev = art3d._norm_angle(self.elev)
        norm_azim = art3d._norm_angle(self.azim)
        norm_roll = art3d._norm_angle(self.roll)
        coords = f'elevation={norm_elev:.0f}°, azimuth={norm_azim:.0f}°, roll={norm_roll:.0f}°'.replace('-', '−')
        return coords

    def _location_coords(self, xv, yv, renderer):
        if False:
            i = 10
            return i + 15
        '\n        Return the location on the axis pane underneath the cursor as a string.\n        '
        (p1, pane_idx) = self._calc_coord(xv, yv, renderer)
        xs = self.format_xdata(p1[0])
        ys = self.format_ydata(p1[1])
        zs = self.format_zdata(p1[2])
        if pane_idx == 0:
            coords = f'x pane={xs}, y={ys}, z={zs}'
        elif pane_idx == 1:
            coords = f'x={xs}, y pane={ys}, z={zs}'
        elif pane_idx == 2:
            coords = f'x={xs}, y={ys}, z pane={zs}'
        return coords

    def _get_camera_loc(self):
        if False:
            return 10
        '\n        Returns the current camera location in data coordinates.\n        '
        (cx, cy, cz, dx, dy, dz) = self._get_w_centers_ranges()
        c = np.array([cx, cy, cz])
        r = np.array([dx, dy, dz])
        if self._focal_length == np.inf:
            focal_length = 1000000000.0
        else:
            focal_length = self._focal_length
        eye = c + self._view_w * self._dist * r / self._box_aspect * focal_length
        return eye

    def _calc_coord(self, xv, yv, renderer=None):
        if False:
            print('Hello World!')
        '\n        Given the 2D view coordinates, find the point on the nearest axis pane\n        that lies directly below those coordinates. Returns a 3D point in data\n        coordinates.\n        '
        if self._focal_length == np.inf:
            zv = 1
        else:
            zv = -1 / self._focal_length
        p1 = np.array(proj3d.inv_transform(xv, yv, zv, self.invM)).ravel()
        vec = self._get_camera_loc() - p1
        pane_locs = []
        for axis in self._axis_map.values():
            (xys, loc) = axis.active_pane()
            pane_locs.append(loc)
        scales = np.zeros(3)
        for i in range(3):
            if vec[i] == 0:
                scales[i] = np.inf
            else:
                scales[i] = (p1[i] - pane_locs[i]) / vec[i]
        pane_idx = np.argmin(abs(scales))
        scale = scales[pane_idx]
        p2 = p1 - scale * vec
        return (p2, pane_idx)

    def _on_move(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Mouse moving.\n\n        By default, button-1 rotates, button-2 pans, and button-3 zooms;\n        these buttons can be modified via `mouse_init`.\n        '
        if not self.button_pressed:
            return
        if self.get_navigate_mode() is not None:
            return
        if self.M is None:
            return
        (x, y) = (event.xdata, event.ydata)
        if x is None or event.inaxes != self:
            return
        (dx, dy) = (x - self._sx, y - self._sy)
        w = self._pseudo_w
        h = self._pseudo_h
        if self.button_pressed in self._rotate_btn:
            if dx == 0 and dy == 0:
                return
            roll = np.deg2rad(self.roll)
            delev = -(dy / h) * 180 * np.cos(roll) + dx / w * 180 * np.sin(roll)
            dazim = -(dy / h) * 180 * np.sin(roll) - dx / w * 180 * np.cos(roll)
            elev = self.elev + delev
            azim = self.azim + dazim
            self.view_init(elev=elev, azim=azim, roll=roll, share=True)
            self.stale = True
        elif self.button_pressed in self._pan_btn:
            (px, py) = self.transData.transform([self._sx, self._sy])
            self.start_pan(px, py, 2)
            self.drag_pan(2, None, event.x, event.y)
            self.end_pan()
        elif self.button_pressed in self._zoom_btn:
            scale = h / (h - dy)
            self._scale_axis_limits(scale, scale, scale)
        (self._sx, self._sy) = (x, y)
        self.figure.canvas.draw_idle()

    def drag_pan(self, button, key, x, y):
        if False:
            while True:
                i = 10
        p = self._pan_start
        ((xdata, ydata), (xdata_start, ydata_start)) = p.trans_inverse.transform([(x, y), (p.x, p.y)])
        (self._sx, self._sy) = (xdata, ydata)
        self.start_pan(x, y, button)
        (du, dv) = (xdata - xdata_start, ydata - ydata_start)
        dw = 0
        if key == 'x':
            dv = 0
        elif key == 'y':
            du = 0
        if du == 0 and dv == 0:
            return
        R = np.array([self._view_u, self._view_v, self._view_w])
        R = -R / self._box_aspect * self._dist
        duvw_projected = R.T @ np.array([du, dv, dw])
        (minx, maxx, miny, maxy, minz, maxz) = self.get_w_lims()
        dx = (maxx - minx) * duvw_projected[0]
        dy = (maxy - miny) * duvw_projected[1]
        dz = (maxz - minz) * duvw_projected[2]
        self.set_xlim3d(minx + dx, maxx + dx, auto=None)
        self.set_ylim3d(miny + dy, maxy + dy, auto=None)
        self.set_zlim3d(minz + dz, maxz + dz, auto=None)

    def _calc_view_axes(self, eye):
        if False:
            print('Hello World!')
        '\n        Get the unit vectors for the viewing axes in data coordinates.\n        `u` is towards the right of the screen\n        `v` is towards the top of the screen\n        `w` is out of the screen\n        '
        elev_rad = np.deg2rad(art3d._norm_angle(self.elev))
        roll_rad = np.deg2rad(art3d._norm_angle(self.roll))
        R = 0.5 * self._roll_to_vertical(self._box_aspect)
        V = np.zeros(3)
        V[self._vertical_axis] = -1 if abs(elev_rad) > np.pi / 2 else 1
        (u, v, w) = proj3d._view_axes(eye, R, V, roll_rad)
        return (u, v, w)

    def _set_view_from_bbox(self, bbox, direction='in', mode=None, twinx=False, twiny=False):
        if False:
            print('Hello World!')
        '\n        Zoom in or out of the bounding box.\n\n        Will center the view in the center of the bounding box, and zoom by\n        the ratio of the size of the bounding box to the size of the Axes3D.\n        '
        (start_x, start_y, stop_x, stop_y) = bbox
        if mode == 'x':
            start_y = self.bbox.min[1]
            stop_y = self.bbox.max[1]
        elif mode == 'y':
            start_x = self.bbox.min[0]
            stop_x = self.bbox.max[0]
        (start_x, stop_x) = np.clip(sorted([start_x, stop_x]), self.bbox.min[0], self.bbox.max[0])
        (start_y, stop_y) = np.clip(sorted([start_y, stop_y]), self.bbox.min[1], self.bbox.max[1])
        zoom_center_x = (start_x + stop_x) / 2
        zoom_center_y = (start_y + stop_y) / 2
        ax_center_x = (self.bbox.max[0] + self.bbox.min[0]) / 2
        ax_center_y = (self.bbox.max[1] + self.bbox.min[1]) / 2
        self.start_pan(zoom_center_x, zoom_center_y, 2)
        self.drag_pan(2, None, ax_center_x, ax_center_y)
        self.end_pan()
        dx = abs(start_x - stop_x)
        dy = abs(start_y - stop_y)
        scale_u = dx / (self.bbox.max[0] - self.bbox.min[0])
        scale_v = dy / (self.bbox.max[1] - self.bbox.min[1])
        scale = max(scale_u, scale_v)
        if direction == 'out':
            scale = 1 / scale
        self._zoom_data_limits(scale, scale, scale)

    def _zoom_data_limits(self, scale_u, scale_v, scale_w):
        if False:
            return 10
        "\n        Zoom in or out of a 3D plot.\n\n        Will scale the data limits by the scale factors. These will be\n        transformed to the x, y, z data axes based on the current view angles.\n        A scale factor > 1 zooms out and a scale factor < 1 zooms in.\n\n        For an axes that has had its aspect ratio set to 'equal', 'equalxy',\n        'equalyz', or 'equalxz', the relevant axes are constrained to zoom\n        equally.\n\n        Parameters\n        ----------\n        scale_u : float\n            Scale factor for the u view axis (view screen horizontal).\n        scale_v : float\n            Scale factor for the v view axis (view screen vertical).\n        scale_w : float\n            Scale factor for the w view axis (view screen depth).\n        "
        scale = np.array([scale_u, scale_v, scale_w])
        if not np.allclose(scale, scale_u):
            R = np.array([self._view_u, self._view_v, self._view_w])
            S = scale * np.eye(3)
            scale = np.linalg.norm(R.T @ S, axis=1)
            if self._aspect in ('equal', 'equalxy', 'equalxz', 'equalyz'):
                ax_idxs = self._equal_aspect_axis_indices(self._aspect)
                min_ax_idxs = np.argmin(np.abs(scale[ax_idxs] - 1))
                scale[ax_idxs] = scale[ax_idxs][min_ax_idxs]
        self._scale_axis_limits(scale[0], scale[1], scale[2])

    def _scale_axis_limits(self, scale_x, scale_y, scale_z):
        if False:
            print('Hello World!')
        '\n        Keeping the center of the x, y, and z data axes fixed, scale their\n        limits by scale factors. A scale factor > 1 zooms out and a scale\n        factor < 1 zooms in.\n\n        Parameters\n        ----------\n        scale_x : float\n            Scale factor for the x data axis.\n        scale_y : float\n            Scale factor for the y data axis.\n        scale_z : float\n            Scale factor for the z data axis.\n        '
        (cx, cy, cz, dx, dy, dz) = self._get_w_centers_ranges()
        self.set_xlim3d(cx - dx * scale_x / 2, cx + dx * scale_x / 2, auto=None)
        self.set_ylim3d(cy - dy * scale_y / 2, cy + dy * scale_y / 2, auto=None)
        self.set_zlim3d(cz - dz * scale_z / 2, cz + dz * scale_z / 2, auto=None)

    def _get_w_centers_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        'Get 3D world centers and axis ranges.'
        (minx, maxx, miny, maxy, minz, maxz) = self.get_w_lims()
        cx = (maxx + minx) / 2
        cy = (maxy + miny) / 2
        cz = (maxz + minz) / 2
        dx = maxx - minx
        dy = maxy - miny
        dz = maxz - minz
        return (cx, cy, cz, dx, dy, dz)

    def set_zlabel(self, zlabel, fontdict=None, labelpad=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Set zlabel.  See doc for `.set_ylabel` for description.\n        '
        if labelpad is not None:
            self.zaxis.labelpad = labelpad
        return self.zaxis.set_label_text(zlabel, fontdict, **kwargs)

    def get_zlabel(self):
        if False:
            while True:
                i = 10
        '\n        Get the z-label text string.\n        '
        label = self.zaxis.get_label()
        return label.get_text()
    get_frame_on = None
    set_frame_on = None

    def grid(self, visible=True, **kwargs):
        if False:
            return 10
        '\n        Set / unset 3D grid.\n\n        .. note::\n\n            Currently, this function does not behave the same as\n            `.axes.Axes.grid`, but it is intended to eventually support that\n            behavior.\n        '
        if len(kwargs):
            visible = True
        self._draw_grid = visible
        self.stale = True

    def tick_params(self, axis='both', **kwargs):
        if False:
            while True:
                i = 10
        "\n        Convenience method for changing the appearance of ticks and\n        tick labels.\n\n        See `.Axes.tick_params` for full documentation.  Because this function\n        applies to 3D Axes, *axis* can also be set to 'z', and setting *axis*\n        to 'both' autoscales all three axes.\n\n        Also, because of how Axes3D objects are drawn very differently\n        from regular 2D axes, some of these settings may have\n        ambiguous meaning.  For simplicity, the 'z' axis will\n        accept settings as if it was like the 'y' axis.\n\n        .. note::\n           Axes3D currently ignores some of these settings.\n        "
        _api.check_in_list(['x', 'y', 'z', 'both'], axis=axis)
        if axis in ['x', 'y', 'both']:
            super().tick_params(axis, **kwargs)
        if axis in ['z', 'both']:
            zkw = dict(kwargs)
            zkw.pop('top', None)
            zkw.pop('bottom', None)
            zkw.pop('labeltop', None)
            zkw.pop('labelbottom', None)
            self.zaxis.set_tick_params(**zkw)

    def invert_zaxis(self):
        if False:
            i = 10
            return i + 15
        '\n        Invert the z-axis.\n\n        See Also\n        --------\n        zaxis_inverted\n        get_zlim, set_zlim\n        get_zbound, set_zbound\n        '
        (bottom, top) = self.get_zlim()
        self.set_zlim(top, bottom, auto=None)
    zaxis_inverted = _axis_method_wrapper('zaxis', 'get_inverted')

    def get_zbound(self):
        if False:
            while True:
                i = 10
        '\n        Return the lower and upper z-axis bounds, in increasing order.\n\n        See Also\n        --------\n        set_zbound\n        get_zlim, set_zlim\n        invert_zaxis, zaxis_inverted\n        '
        (lower, upper) = self.get_zlim()
        if lower < upper:
            return (lower, upper)
        else:
            return (upper, lower)

    def text(self, x, y, z, s, zdir=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Add the text *s* to the 3D Axes at location *x*, *y*, *z* in data coordinates.\n\n        Parameters\n        ----------\n        x, y, z : float\n            The position to place the text.\n        s : str\n            The text.\n        zdir : {'x', 'y', 'z', 3-tuple}, optional\n            The direction to be used as the z-direction. Default: 'z'.\n            See `.get_dir_vector` for a description of the values.\n        **kwargs\n            Other arguments are forwarded to `matplotlib.axes.Axes.text`.\n\n        Returns\n        -------\n        `.Text3D`\n            The created `.Text3D` instance.\n        "
        text = super().text(x, y, s, **kwargs)
        art3d.text_2d_to_3d(text, z, zdir)
        return text
    text3D = text
    text2D = Axes.text

    def plot(self, xs, ys, *args, zdir='z', **kwargs):
        if False:
            print('Hello World!')
        "\n        Plot 2D or 3D data.\n\n        Parameters\n        ----------\n        xs : 1D array-like\n            x coordinates of vertices.\n        ys : 1D array-like\n            y coordinates of vertices.\n        zs : float or 1D array-like\n            z coordinates of vertices; either one for all points or one for\n            each point.\n        zdir : {'x', 'y', 'z'}, default: 'z'\n            When plotting 2D data, the direction to use as z.\n        **kwargs\n            Other arguments are forwarded to `matplotlib.axes.Axes.plot`.\n        "
        had_data = self.has_data()
        if args and (not isinstance(args[0], str)):
            (zs, *args) = args
            if 'zs' in kwargs:
                raise TypeError("plot() for multiple values for argument 'z'")
        else:
            zs = kwargs.pop('zs', 0)
        (xs, ys, zs) = cbook._broadcast_with_masks(xs, ys, zs)
        lines = super().plot(xs, ys, *args, **kwargs)
        for line in lines:
            art3d.line_2d_to_3d(line, zs=zs, zdir=zdir)
        (xs, ys, zs) = art3d.juggle_axes(xs, ys, zs, zdir)
        self.auto_scale_xyz(xs, ys, zs, had_data)
        return lines
    plot3D = plot

    def plot_surface(self, X, Y, Z, *, norm=None, vmin=None, vmax=None, lightsource=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Create a surface plot.\n\n        By default, it will be colored in shades of a solid color, but it also\n        supports colormapping by supplying the *cmap* argument.\n\n        .. note::\n\n           The *rcount* and *ccount* kwargs, which both default to 50,\n           determine the maximum number of samples used in each direction.  If\n           the input data is larger, it will be downsampled (by slicing) to\n           these numbers of points.\n\n        .. note::\n\n           To maximize rendering speed consider setting *rstride* and *cstride*\n           to divisors of the number of rows minus 1 and columns minus 1\n           respectively. For example, given 51 rows rstride can be any of the\n           divisors of 50.\n\n           Similarly, a setting of *rstride* and *cstride* equal to 1 (or\n           *rcount* and *ccount* equal the number of rows and columns) can use\n           the optimized path.\n\n        Parameters\n        ----------\n        X, Y, Z : 2D arrays\n            Data values.\n\n        rcount, ccount : int\n            Maximum number of samples used in each direction.  If the input\n            data is larger, it will be downsampled (by slicing) to these\n            numbers of points.  Defaults to 50.\n\n        rstride, cstride : int\n            Downsampling stride in each direction.  These arguments are\n            mutually exclusive with *rcount* and *ccount*.  If only one of\n            *rstride* or *cstride* is set, the other defaults to 10.\n\n            'classic' mode uses a default of ``rstride = cstride = 10`` instead\n            of the new default of ``rcount = ccount = 50``.\n\n        color : color-like\n            Color of the surface patches.\n\n        cmap : Colormap\n            Colormap of the surface patches.\n\n        facecolors : array-like of colors.\n            Colors of each individual patch.\n\n        norm : Normalize\n            Normalization for the colormap.\n\n        vmin, vmax : float\n            Bounds for the normalization.\n\n        shade : bool, default: True\n            Whether to shade the facecolors.  Shading is always disabled when\n            *cmap* is specified.\n\n        lightsource : `~matplotlib.colors.LightSource`\n            The lightsource to use when *shade* is True.\n\n        **kwargs\n            Other keyword arguments are forwarded to `.Poly3DCollection`.\n        "
        had_data = self.has_data()
        if Z.ndim != 2:
            raise ValueError('Argument Z must be 2-dimensional.')
        Z = cbook._to_unmasked_float_array(Z)
        (X, Y, Z) = np.broadcast_arrays(X, Y, Z)
        (rows, cols) = Z.shape
        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs
        if has_stride and has_count:
            raise ValueError('Cannot specify both stride and count arguments')
        rstride = kwargs.pop('rstride', 10)
        cstride = kwargs.pop('cstride', 10)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)
        if mpl.rcParams['_internal.classic_mode']:
            compute_strides = has_count
        else:
            compute_strides = not has_stride
        if compute_strides:
            rstride = int(max(np.ceil(rows / rcount), 1))
            cstride = int(max(np.ceil(cols / ccount), 1))
        fcolors = kwargs.pop('facecolors', None)
        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)
        if shade is None:
            raise ValueError('shade cannot be None.')
        colset = []
        if (rows - 1) % rstride == 0 and (cols - 1) % cstride == 0 and (fcolors is None):
            polys = np.stack([cbook._array_patch_perimeters(a, rstride, cstride) for a in (X, Y, Z)], axis=-1)
        else:
            row_inds = list(range(0, rows - 1, rstride)) + [rows - 1]
            col_inds = list(range(0, cols - 1, cstride)) + [cols - 1]
            polys = []
            for (rs, rs_next) in zip(row_inds[:-1], row_inds[1:]):
                for (cs, cs_next) in zip(col_inds[:-1], col_inds[1:]):
                    ps = [cbook._array_perimeter(a[rs:rs_next + 1, cs:cs_next + 1]) for a in (X, Y, Z)]
                    ps = np.array(ps).T
                    polys.append(ps)
                    if fcolors is not None:
                        colset.append(fcolors[rs][cs])
        if not isinstance(polys, np.ndarray) or not np.isfinite(polys).all():
            new_polys = []
            new_colset = []
            for (p, col) in itertools.zip_longest(polys, colset):
                new_poly = np.array(p)[np.isfinite(p).all(axis=1)]
                if len(new_poly):
                    new_polys.append(new_poly)
                    new_colset.append(col)
            polys = new_polys
            if fcolors is not None:
                colset = new_colset
        if fcolors is not None:
            polyc = art3d.Poly3DCollection(polys, edgecolors=colset, facecolors=colset, shade=shade, lightsource=lightsource, **kwargs)
        elif cmap:
            polyc = art3d.Poly3DCollection(polys, **kwargs)
            if isinstance(polys, np.ndarray):
                avg_z = polys[..., 2].mean(axis=-1)
            else:
                avg_z = np.array([ps[:, 2].mean() for ps in polys])
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            color = kwargs.pop('color', None)
            if color is None:
                color = self._get_lines.get_next_color()
            color = np.array(mcolors.to_rgba(color))
            polyc = art3d.Poly3DCollection(polys, facecolors=color, shade=shade, lightsource=lightsource, **kwargs)
        self.add_collection(polyc)
        self.auto_scale_xyz(X, Y, Z, had_data)
        return polyc

    def plot_wireframe(self, X, Y, Z, **kwargs):
        if False:
            return 10
        "\n        Plot a 3D wireframe.\n\n        .. note::\n\n           The *rcount* and *ccount* kwargs, which both default to 50,\n           determine the maximum number of samples used in each direction.  If\n           the input data is larger, it will be downsampled (by slicing) to\n           these numbers of points.\n\n        Parameters\n        ----------\n        X, Y, Z : 2D arrays\n            Data values.\n\n        rcount, ccount : int\n            Maximum number of samples used in each direction.  If the input\n            data is larger, it will be downsampled (by slicing) to these\n            numbers of points.  Setting a count to zero causes the data to be\n            not sampled in the corresponding direction, producing a 3D line\n            plot rather than a wireframe plot.  Defaults to 50.\n\n        rstride, cstride : int\n            Downsampling stride in each direction.  These arguments are\n            mutually exclusive with *rcount* and *ccount*.  If only one of\n            *rstride* or *cstride* is set, the other defaults to 1.  Setting a\n            stride to zero causes the data to be not sampled in the\n            corresponding direction, producing a 3D line plot rather than a\n            wireframe plot.\n\n            'classic' mode uses a default of ``rstride = cstride = 1`` instead\n            of the new default of ``rcount = ccount = 50``.\n\n        **kwargs\n            Other keyword arguments are forwarded to `.Line3DCollection`.\n        "
        had_data = self.has_data()
        if Z.ndim != 2:
            raise ValueError('Argument Z must be 2-dimensional.')
        (X, Y, Z) = np.broadcast_arrays(X, Y, Z)
        (rows, cols) = Z.shape
        has_stride = 'rstride' in kwargs or 'cstride' in kwargs
        has_count = 'rcount' in kwargs or 'ccount' in kwargs
        if has_stride and has_count:
            raise ValueError('Cannot specify both stride and count arguments')
        rstride = kwargs.pop('rstride', 1)
        cstride = kwargs.pop('cstride', 1)
        rcount = kwargs.pop('rcount', 50)
        ccount = kwargs.pop('ccount', 50)
        if mpl.rcParams['_internal.classic_mode']:
            if has_count:
                rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
                cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0
        elif not has_stride:
            rstride = int(max(np.ceil(rows / rcount), 1)) if rcount else 0
            cstride = int(max(np.ceil(cols / ccount), 1)) if ccount else 0
        (tX, tY, tZ) = (np.transpose(X), np.transpose(Y), np.transpose(Z))
        if rstride:
            rii = list(range(0, rows, rstride))
            if rows > 0 and rii[-1] != rows - 1:
                rii += [rows - 1]
        else:
            rii = []
        if cstride:
            cii = list(range(0, cols, cstride))
            if cols > 0 and cii[-1] != cols - 1:
                cii += [cols - 1]
        else:
            cii = []
        if rstride == 0 and cstride == 0:
            raise ValueError('Either rstride or cstride must be non zero')
        if Z.size == 0:
            rii = []
            cii = []
        xlines = [X[i] for i in rii]
        ylines = [Y[i] for i in rii]
        zlines = [Z[i] for i in rii]
        txlines = [tX[i] for i in cii]
        tylines = [tY[i] for i in cii]
        tzlines = [tZ[i] for i in cii]
        lines = [list(zip(xl, yl, zl)) for (xl, yl, zl) in zip(xlines, ylines, zlines)] + [list(zip(xl, yl, zl)) for (xl, yl, zl) in zip(txlines, tylines, tzlines)]
        linec = art3d.Line3DCollection(lines, **kwargs)
        self.add_collection(linec)
        self.auto_scale_xyz(X, Y, Z, had_data)
        return linec

    def plot_trisurf(self, *args, color=None, norm=None, vmin=None, vmax=None, lightsource=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Plot a triangulated surface.\n\n        The (optional) triangulation can be specified in one of two ways;\n        either::\n\n          plot_trisurf(triangulation, ...)\n\n        where triangulation is a `~matplotlib.tri.Triangulation` object, or::\n\n          plot_trisurf(X, Y, ...)\n          plot_trisurf(X, Y, triangles, ...)\n          plot_trisurf(X, Y, triangles=triangles, ...)\n\n        in which case a Triangulation object will be created.  See\n        `.Triangulation` for an explanation of these possibilities.\n\n        The remaining arguments are::\n\n          plot_trisurf(..., Z)\n\n        where *Z* is the array of values to contour, one per point\n        in the triangulation.\n\n        Parameters\n        ----------\n        X, Y, Z : array-like\n            Data values as 1D arrays.\n        color\n            Color of the surface patches.\n        cmap\n            A colormap for the surface patches.\n        norm : Normalize\n            An instance of Normalize to map values to colors.\n        vmin, vmax : float, default: None\n            Minimum and maximum value to map.\n        shade : bool, default: True\n            Whether to shade the facecolors.  Shading is always disabled when\n            *cmap* is specified.\n        lightsource : `~matplotlib.colors.LightSource`\n            The lightsource to use when *shade* is True.\n        **kwargs\n            All other keyword arguments are passed on to\n            :class:`~mpl_toolkits.mplot3d.art3d.Poly3DCollection`\n\n        Examples\n        --------\n        .. plot:: gallery/mplot3d/trisurf3d.py\n        .. plot:: gallery/mplot3d/trisurf3d_2.py\n        '
        had_data = self.has_data()
        if color is None:
            color = self._get_lines.get_next_color()
        color = np.array(mcolors.to_rgba(color))
        cmap = kwargs.get('cmap', None)
        shade = kwargs.pop('shade', cmap is None)
        (tri, args, kwargs) = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
        try:
            z = kwargs.pop('Z')
        except KeyError:
            (z, *args) = args
        z = np.asarray(z)
        triangles = tri.get_masked_triangles()
        xt = tri.x[triangles]
        yt = tri.y[triangles]
        zt = z[triangles]
        verts = np.stack((xt, yt, zt), axis=-1)
        if cmap:
            polyc = art3d.Poly3DCollection(verts, *args, **kwargs)
            avg_z = verts[:, :, 2].mean(axis=1)
            polyc.set_array(avg_z)
            if vmin is not None or vmax is not None:
                polyc.set_clim(vmin, vmax)
            if norm is not None:
                polyc.set_norm(norm)
        else:
            polyc = art3d.Poly3DCollection(verts, *args, shade=shade, lightsource=lightsource, facecolors=color, **kwargs)
        self.add_collection(polyc)
        self.auto_scale_xyz(tri.x, tri.y, z, had_data)
        return polyc

    def _3d_extend_contour(self, cset, stride=5):
        if False:
            print('Hello World!')
        '\n        Extend a contour in 3D by creating\n        '
        dz = (cset.levels[1] - cset.levels[0]) / 2
        polyverts = []
        colors = []
        for (idx, level) in enumerate(cset.levels):
            path = cset.get_paths()[idx]
            subpaths = [*path._iter_connected_components()]
            color = cset.get_edgecolor()[idx]
            top = art3d._paths_to_3d_segments(subpaths, level - dz)
            bot = art3d._paths_to_3d_segments(subpaths, level + dz)
            if not len(top[0]):
                continue
            nsteps = max(round(len(top[0]) / stride), 2)
            stepsize = (len(top[0]) - 1) / (nsteps - 1)
            polyverts.extend([(top[0][round(i * stepsize)], top[0][round((i + 1) * stepsize)], bot[0][round((i + 1) * stepsize)], bot[0][round(i * stepsize)]) for i in range(round(nsteps) - 1)])
            colors.extend([color] * (round(nsteps) - 1))
        self.add_collection3d(art3d.Poly3DCollection(np.array(polyverts), facecolors=colors, edgecolors=colors, shade=True))
        cset.remove()

    def add_contour_set(self, cset, extend3d=False, stride=5, zdir='z', offset=None):
        if False:
            i = 10
            return i + 15
        zdir = '-' + zdir
        if extend3d:
            self._3d_extend_contour(cset, stride)
        else:
            art3d.collection_2d_to_3d(cset, zs=offset if offset is not None else cset.levels, zdir=zdir)

    def add_contourf_set(self, cset, zdir='z', offset=None):
        if False:
            i = 10
            return i + 15
        self._add_contourf_set(cset, zdir=zdir, offset=offset)

    def _add_contourf_set(self, cset, zdir='z', offset=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns\n        -------\n        levels : `numpy.ndarray`\n            Levels at which the filled contours are added.\n        '
        zdir = '-' + zdir
        midpoints = cset.levels[:-1] + np.diff(cset.levels) / 2
        if cset._extend_min:
            min_level = cset.levels[0] - np.diff(cset.levels[:2]) / 2
            midpoints = np.insert(midpoints, 0, min_level)
        if cset._extend_max:
            max_level = cset.levels[-1] + np.diff(cset.levels[-2:]) / 2
            midpoints = np.append(midpoints, max_level)
        art3d.collection_2d_to_3d(cset, zs=offset if offset is not None else midpoints, zdir=zdir)
        return midpoints

    @_preprocess_data()
    def contour(self, X, Y, Z, *args, extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a 3D contour plot.\n\n        Parameters\n        ----------\n        X, Y, Z : array-like,\n            Input data. See `.Axes.contour` for supported data shapes.\n        extend3d : bool, default: False\n            Whether to extend contour in 3D.\n        stride : int\n            Step size for extending contour.\n        zdir : {'x', 'y', 'z'}, default: 'z'\n            The direction to use.\n        offset : float, optional\n            If specified, plot a projection of the contour lines at this\n            position in a plane normal to *zdir*.\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        *args, **kwargs\n            Other arguments are forwarded to `matplotlib.axes.Axes.contour`.\n\n        Returns\n        -------\n        matplotlib.contour.QuadContourSet\n        "
        had_data = self.has_data()
        (jX, jY, jZ) = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contour(jX, jY, jZ, *args, **kwargs)
        self.add_contour_set(cset, extend3d, stride, zdir, offset)
        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset
    contour3D = contour

    @_preprocess_data()
    def tricontour(self, *args, extend3d=False, stride=5, zdir='z', offset=None, **kwargs):
        if False:
            return 10
        "\n        Create a 3D contour plot.\n\n        .. note::\n            This method currently produces incorrect output due to a\n            longstanding bug in 3D PolyCollection rendering.\n\n        Parameters\n        ----------\n        X, Y, Z : array-like\n            Input data. See `.Axes.tricontour` for supported data shapes.\n        extend3d : bool, default: False\n            Whether to extend contour in 3D.\n        stride : int\n            Step size for extending contour.\n        zdir : {'x', 'y', 'z'}, default: 'z'\n            The direction to use.\n        offset : float, optional\n            If specified, plot a projection of the contour lines at this\n            position in a plane normal to *zdir*.\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        *args, **kwargs\n            Other arguments are forwarded to `matplotlib.axes.Axes.tricontour`.\n\n        Returns\n        -------\n        matplotlib.tri._tricontour.TriContourSet\n        "
        had_data = self.has_data()
        (tri, args, kwargs) = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
        X = tri.x
        Y = tri.y
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            (Z, *args) = args
        (jX, jY, jZ) = art3d.rotate_axes(X, Y, Z, zdir)
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)
        cset = super().tricontour(tri, jZ, *args, **kwargs)
        self.add_contour_set(cset, extend3d, stride, zdir, offset)
        self.auto_scale_xyz(X, Y, Z, had_data)
        return cset

    def _auto_scale_contourf(self, X, Y, Z, zdir, levels, had_data):
        if False:
            i = 10
            return i + 15
        dim_vals = {'x': X, 'y': Y, 'z': Z, zdir: levels}
        limits = [(np.nanmin(dim_vals[dim]), np.nanmax(dim_vals[dim])) for dim in ['x', 'y', 'z']]
        self.auto_scale_xyz(*limits, had_data)

    @_preprocess_data()
    def contourf(self, X, Y, Z, *args, zdir='z', offset=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Create a 3D filled contour plot.\n\n        Parameters\n        ----------\n        X, Y, Z : array-like\n            Input data. See `.Axes.contourf` for supported data shapes.\n        zdir : {'x', 'y', 'z'}, default: 'z'\n            The direction to use.\n        offset : float, optional\n            If specified, plot a projection of the contour lines at this\n            position in a plane normal to *zdir*.\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        *args, **kwargs\n            Other arguments are forwarded to `matplotlib.axes.Axes.contourf`.\n\n        Returns\n        -------\n        matplotlib.contour.QuadContourSet\n        "
        had_data = self.has_data()
        (jX, jY, jZ) = art3d.rotate_axes(X, Y, Z, zdir)
        cset = super().contourf(jX, jY, jZ, *args, **kwargs)
        levels = self._add_contourf_set(cset, zdir, offset)
        self._auto_scale_contourf(X, Y, Z, zdir, levels, had_data)
        return cset
    contourf3D = contourf

    @_preprocess_data()
    def tricontourf(self, *args, zdir='z', offset=None, **kwargs):
        if False:
            return 10
        "\n        Create a 3D filled contour plot.\n\n        .. note::\n            This method currently produces incorrect output due to a\n            longstanding bug in 3D PolyCollection rendering.\n\n        Parameters\n        ----------\n        X, Y, Z : array-like\n            Input data. See `.Axes.tricontourf` for supported data shapes.\n        zdir : {'x', 'y', 'z'}, default: 'z'\n            The direction to use.\n        offset : float, optional\n            If specified, plot a projection of the contour lines at this\n            position in a plane normal to zdir.\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        *args, **kwargs\n            Other arguments are forwarded to\n            `matplotlib.axes.Axes.tricontourf`.\n\n        Returns\n        -------\n        matplotlib.tri._tricontour.TriContourSet\n        "
        had_data = self.has_data()
        (tri, args, kwargs) = Triangulation.get_from_args_and_kwargs(*args, **kwargs)
        X = tri.x
        Y = tri.y
        if 'Z' in kwargs:
            Z = kwargs.pop('Z')
        else:
            (Z, *args) = args
        (jX, jY, jZ) = art3d.rotate_axes(X, Y, Z, zdir)
        tri = Triangulation(jX, jY, tri.triangles, tri.mask)
        cset = super().tricontourf(tri, jZ, *args, **kwargs)
        levels = self._add_contourf_set(cset, zdir, offset)
        self._auto_scale_contourf(X, Y, Z, zdir, levels, had_data)
        return cset

    def add_collection3d(self, col, zs=0, zdir='z'):
        if False:
            i = 10
            return i + 15
        '\n        Add a 3D collection object to the plot.\n\n        2D collection types are converted to a 3D version by\n        modifying the object and adding z coordinate information.\n\n        Supported are:\n\n        - PolyCollection\n        - LineCollection\n        - PatchCollection\n        '
        zvals = np.atleast_1d(zs)
        zsortval = np.min(zvals) if zvals.size else 0
        if type(col) is mcoll.PolyCollection:
            art3d.poly_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.LineCollection:
            art3d.line_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)
        elif type(col) is mcoll.PatchCollection:
            art3d.patch_collection_2d_to_3d(col, zs=zs, zdir=zdir)
            col.set_sort_zpos(zsortval)
        collection = super().add_collection(col)
        return collection

    @_preprocess_data(replace_names=['xs', 'ys', 'zs', 's', 'edgecolors', 'c', 'facecolor', 'facecolors', 'color'])
    def scatter(self, xs, ys, zs=0, zdir='z', s=20, c=None, depthshade=True, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Create a scatter plot.\n\n        Parameters\n        ----------\n        xs, ys : array-like\n            The data positions.\n        zs : float or array-like, default: 0\n            The z-positions. Either an array of the same length as *xs* and\n            *ys* or a single value to place all points in the same plane.\n        zdir : {'x', 'y', 'z', '-x', '-y', '-z'}, default: 'z'\n            The axis direction for the *zs*. This is useful when plotting 2D\n            data on a 3D Axes. The data must be passed as *xs*, *ys*. Setting\n            *zdir* to 'y' then plots the data to the x-z-plane.\n\n            See also :doc:`/gallery/mplot3d/2dcollections3d`.\n\n        s : float or array-like, default: 20\n            The marker size in points**2. Either an array of the same length\n            as *xs* and *ys* or a single value to make all markers the same\n            size.\n        c : color, sequence, or sequence of colors, optional\n            The marker color. Possible values:\n\n            - A single color format string.\n            - A sequence of colors of length n.\n            - A sequence of n numbers to be mapped to colors using *cmap* and\n              *norm*.\n            - A 2D array in which the rows are RGB or RGBA.\n\n            For more details see the *c* argument of `~.axes.Axes.scatter`.\n        depthshade : bool, default: True\n            Whether to shade the scatter markers to give the appearance of\n            depth. Each call to ``scatter()`` will perform its depthshading\n            independently.\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        **kwargs\n            All other keyword arguments are passed on to `~.axes.Axes.scatter`.\n\n        Returns\n        -------\n        paths : `~matplotlib.collections.PathCollection`\n        "
        had_data = self.has_data()
        zs_orig = zs
        (xs, ys, zs) = cbook._broadcast_with_masks(xs, ys, zs)
        s = np.ma.ravel(s)
        (xs, ys, zs, s, c, color) = cbook.delete_masked_points(xs, ys, zs, s, c, kwargs.get('color', None))
        if kwargs.get('color') is not None:
            kwargs['color'] = color
        if np.may_share_memory(zs_orig, zs):
            zs = zs.copy()
        patches = super().scatter(xs, ys, *args, s=s, c=c, **kwargs)
        art3d.patch_collection_2d_to_3d(patches, zs=zs, zdir=zdir, depthshade=depthshade)
        if self._zmargin < 0.05 and xs.size > 0:
            self.set_zmargin(0.05)
        self.auto_scale_xyz(xs, ys, zs, had_data)
        return patches
    scatter3D = scatter

    @_preprocess_data()
    def bar(self, left, height, zs=0, zdir='z', *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        Add 2D bar(s).\n\n        Parameters\n        ----------\n        left : 1D array-like\n            The x coordinates of the left sides of the bars.\n        height : 1D array-like\n            The height of the bars.\n        zs : float or 1D array-like\n            Z coordinate of bars; if a single value is specified, it will be\n            used for all bars.\n        zdir : {'x', 'y', 'z'}, default: 'z'\n            When plotting 2D data, the direction to use as z ('x', 'y' or 'z').\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        **kwargs\n            Other keyword arguments are forwarded to\n            `matplotlib.axes.Axes.bar`.\n\n        Returns\n        -------\n        mpl_toolkits.mplot3d.art3d.Patch3DCollection\n        "
        had_data = self.has_data()
        patches = super().bar(left, height, *args, **kwargs)
        zs = np.broadcast_to(zs, len(left), subok=True)
        verts = []
        verts_zs = []
        for (p, z) in zip(patches, zs):
            vs = art3d._get_patch_verts(p)
            verts += vs.tolist()
            verts_zs += [z] * len(vs)
            art3d.patch_2d_to_3d(p, z, zdir)
            if 'alpha' in kwargs:
                p.set_alpha(kwargs['alpha'])
        if len(verts) > 0:
            (xs, ys) = zip(*verts)
        else:
            (xs, ys) = ([], [])
        (xs, ys, verts_zs) = art3d.juggle_axes(xs, ys, verts_zs, zdir)
        self.auto_scale_xyz(xs, ys, verts_zs, had_data)
        return patches

    @_preprocess_data()
    def bar3d(self, x, y, z, dx, dy, dz, color=None, zsort='average', shade=True, lightsource=None, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Generate a 3D barplot.\n\n        This method creates three-dimensional barplot where the width,\n        depth, height, and color of the bars can all be uniquely set.\n\n        Parameters\n        ----------\n        x, y, z : array-like\n            The coordinates of the anchor point of the bars.\n\n        dx, dy, dz : float or array-like\n            The width, depth, and height of the bars, respectively.\n\n        color : sequence of colors, optional\n            The color of the bars can be specified globally or\n            individually. This parameter can be:\n\n            - A single color, to color all bars the same color.\n            - An array of colors of length N bars, to color each bar\n              independently.\n            - An array of colors of length 6, to color the faces of the\n              bars similarly.\n            - An array of colors of length 6 * N bars, to color each face\n              independently.\n\n            When coloring the faces of the boxes specifically, this is\n            the order of the coloring:\n\n            1. -Z (bottom of box)\n            2. +Z (top of box)\n            3. -Y\n            4. +Y\n            5. -X\n            6. +X\n\n        zsort : str, optional\n            The z-axis sorting scheme passed onto `~.art3d.Poly3DCollection`\n\n        shade : bool, default: True\n            When true, this shades the dark sides of the bars (relative\n            to the plot's source of light).\n\n        lightsource : `~matplotlib.colors.LightSource`\n            The lightsource to use when *shade* is True.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Any additional keyword arguments are passed onto\n            `~.art3d.Poly3DCollection`.\n\n        Returns\n        -------\n        collection : `~.art3d.Poly3DCollection`\n            A collection of three-dimensional polygons representing the bars.\n        "
        had_data = self.has_data()
        (x, y, z, dx, dy, dz) = np.broadcast_arrays(np.atleast_1d(x), y, z, dx, dy, dz)
        minx = np.min(x)
        maxx = np.max(x + dx)
        miny = np.min(y)
        maxy = np.max(y + dy)
        minz = np.min(z)
        maxz = np.max(z + dz)
        cuboid = np.array([((0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)), ((0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)), ((0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)), ((0, 1, 0), (0, 1, 1), (1, 1, 1), (1, 1, 0)), ((0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)), ((1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1))])
        polys = np.empty(x.shape + cuboid.shape)
        for (i, p, dp) in [(0, x, dx), (1, y, dy), (2, z, dz)]:
            p = p[..., np.newaxis, np.newaxis]
            dp = dp[..., np.newaxis, np.newaxis]
            polys[..., i] = p + dp * cuboid[..., i]
        polys = polys.reshape((-1,) + polys.shape[2:])
        facecolors = []
        if color is None:
            color = [self._get_patches_for_fill.get_next_color()]
        color = list(mcolors.to_rgba_array(color))
        if len(color) == len(x):
            for c in color:
                facecolors.extend([c] * 6)
        else:
            facecolors = color
            if len(facecolors) < len(x):
                facecolors *= 6 * len(x)
        col = art3d.Poly3DCollection(polys, *args, zsort=zsort, facecolors=facecolors, shade=shade, lightsource=lightsource, **kwargs)
        self.add_collection(col)
        self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)
        return col

    def set_title(self, label, fontdict=None, loc='center', **kwargs):
        if False:
            i = 10
            return i + 15
        ret = super().set_title(label, fontdict=fontdict, loc=loc, **kwargs)
        (x, y) = self.title.get_position()
        self.title.set_y(0.92 * y)
        return ret

    @_preprocess_data()
    def quiver(self, X, Y, Z, U, V, W, *, length=1, arrow_length_ratio=0.3, pivot='tail', normalize=False, **kwargs):
        if False:
            print('Hello World!')
        "\n        Plot a 3D field of arrows.\n\n        The arguments can be array-like or scalars, so long as they can be\n        broadcast together. The arguments can also be masked arrays. If an\n        element in any of argument is masked, then that corresponding quiver\n        element will not be plotted.\n\n        Parameters\n        ----------\n        X, Y, Z : array-like\n            The x, y and z coordinates of the arrow locations (default is\n            tail of arrow; see *pivot* kwarg).\n\n        U, V, W : array-like\n            The x, y and z components of the arrow vectors.\n\n        length : float, default: 1\n            The length of each quiver.\n\n        arrow_length_ratio : float, default: 0.3\n            The ratio of the arrow head with respect to the quiver.\n\n        pivot : {'tail', 'middle', 'tip'}, default: 'tail'\n            The part of the arrow that is at the grid point; the arrow\n            rotates about this point, hence the name *pivot*.\n\n        normalize : bool, default: False\n            Whether all arrows are normalized to have the same length, or keep\n            the lengths defined by *u*, *v*, and *w*.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Any additional keyword arguments are delegated to\n            :class:`.Line3DCollection`\n        "

        def calc_arrows(UVW):
            if False:
                print('Hello World!')
            x = UVW[:, 0]
            y = UVW[:, 1]
            norm = np.linalg.norm(UVW[:, :2], axis=1)
            x_p = np.divide(y, norm, where=norm != 0, out=np.zeros_like(x))
            y_p = np.divide(-x, norm, where=norm != 0, out=np.ones_like(x))
            rangle = math.radians(15)
            c = math.cos(rangle)
            s = math.sin(rangle)
            r13 = y_p * s
            r32 = x_p * s
            r12 = x_p * y_p * (1 - c)
            Rpos = np.array([[c + x_p ** 2 * (1 - c), r12, r13], [r12, c + y_p ** 2 * (1 - c), -r32], [-r13, r32, np.full_like(x_p, c)]])
            Rneg = Rpos.copy()
            Rneg[[0, 1, 2, 2], [2, 2, 0, 1]] *= -1
            Rpos_vecs = np.einsum('ij...,...j->...i', Rpos, UVW)
            Rneg_vecs = np.einsum('ij...,...j->...i', Rneg, UVW)
            return np.stack([Rpos_vecs, Rneg_vecs], axis=1)
        had_data = self.has_data()
        input_args = cbook._broadcast_with_masks(X, Y, Z, U, V, W, compress=True)
        if any((len(v) == 0 for v in input_args)):
            linec = art3d.Line3DCollection([], **kwargs)
            self.add_collection(linec)
            return linec
        shaft_dt = np.array([0.0, length], dtype=float)
        arrow_dt = shaft_dt * arrow_length_ratio
        _api.check_in_list(['tail', 'middle', 'tip'], pivot=pivot)
        if pivot == 'tail':
            shaft_dt -= length
        elif pivot == 'middle':
            shaft_dt -= length / 2
        XYZ = np.column_stack(input_args[:3])
        UVW = np.column_stack(input_args[3:]).astype(float)
        norm = np.linalg.norm(UVW, axis=1)
        mask = norm > 0
        XYZ = XYZ[mask]
        if normalize:
            UVW = UVW[mask] / norm[mask].reshape((-1, 1))
        else:
            UVW = UVW[mask]
        if len(XYZ) > 0:
            shafts = (XYZ - np.multiply.outer(shaft_dt, UVW)).swapaxes(0, 1)
            head_dirs = calc_arrows(UVW)
            heads = shafts[:, :1] - np.multiply.outer(arrow_dt, head_dirs)
            heads = heads.reshape((len(arrow_dt), -1, 3))
            heads = heads.swapaxes(0, 1)
            lines = [*shafts, *heads]
        else:
            lines = []
        linec = art3d.Line3DCollection(lines, **kwargs)
        self.add_collection(linec)
        self.auto_scale_xyz(XYZ[:, 0], XYZ[:, 1], XYZ[:, 2], had_data)
        return linec
    quiver3D = quiver

    def voxels(self, *args, facecolors=None, edgecolors=None, shade=True, lightsource=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        ax.voxels([x, y, z,] /, filled, facecolors=None, edgecolors=None, **kwargs)\n\n        Plot a set of filled voxels\n\n        All voxels are plotted as 1x1x1 cubes on the axis, with\n        ``filled[0, 0, 0]`` placed with its lower corner at the origin.\n        Occluded faces are not plotted.\n\n        Parameters\n        ----------\n        filled : 3D np.array of bool\n            A 3D array of values, with truthy values indicating which voxels\n            to fill\n\n        x, y, z : 3D np.array, optional\n            The coordinates of the corners of the voxels. This should broadcast\n            to a shape one larger in every dimension than the shape of\n            *filled*.  These can be used to plot non-cubic voxels.\n\n            If not specified, defaults to increasing integers along each axis,\n            like those returned by :func:`~numpy.indices`.\n            As indicated by the ``/`` in the function signature, these\n            arguments can only be passed positionally.\n\n        facecolors, edgecolors : array-like, optional\n            The color to draw the faces and edges of the voxels. Can only be\n            passed as keyword arguments.\n            These parameters can be:\n\n            - A single color value, to color all voxels the same color. This\n              can be either a string, or a 1D RGB/RGBA array\n            - ``None``, the default, to use a single color for the faces, and\n              the style default for the edges.\n            - A 3D `~numpy.ndarray` of color names, with each item the color\n              for the corresponding voxel. The size must match the voxels.\n            - A 4D `~numpy.ndarray` of RGB/RGBA data, with the components\n              along the last axis.\n\n        shade : bool, default: True\n            Whether to shade the facecolors.\n\n        lightsource : `~matplotlib.colors.LightSource`\n            The lightsource to use when *shade* is True.\n\n        **kwargs\n            Additional keyword arguments to pass onto\n            `~mpl_toolkits.mplot3d.art3d.Poly3DCollection`.\n\n        Returns\n        -------\n        faces : dict\n            A dictionary indexed by coordinate, where ``faces[i, j, k]`` is a\n            `.Poly3DCollection` of the faces drawn for the voxel\n            ``filled[i, j, k]``. If no faces were drawn for a given voxel,\n            either because it was not asked to be drawn, or it is fully\n            occluded, then ``(i, j, k) not in faces``.\n\n        Examples\n        --------\n        .. plot:: gallery/mplot3d/voxels.py\n        .. plot:: gallery/mplot3d/voxels_rgb.py\n        .. plot:: gallery/mplot3d/voxels_torus.py\n        .. plot:: gallery/mplot3d/voxels_numpy_logo.py\n        '
        if len(args) >= 3:

            def voxels(__x, __y, __z, filled, **kwargs):
                if False:
                    return 10
                return ((__x, __y, __z), filled, kwargs)
        else:

            def voxels(filled, **kwargs):
                if False:
                    print('Hello World!')
                return (None, filled, kwargs)
        (xyz, filled, kwargs) = voxels(*args, **kwargs)
        if filled.ndim != 3:
            raise ValueError('Argument filled must be 3-dimensional')
        size = np.array(filled.shape, dtype=np.intp)
        coord_shape = tuple(size + 1)
        if xyz is None:
            (x, y, z) = np.indices(coord_shape)
        else:
            (x, y, z) = (np.broadcast_to(c, coord_shape) for c in xyz)

        def _broadcast_color_arg(color, name):
            if False:
                while True:
                    i = 10
            if np.ndim(color) in (0, 1):
                return np.broadcast_to(color, filled.shape + np.shape(color))
            elif np.ndim(color) in (3, 4):
                if np.shape(color)[:3] != filled.shape:
                    raise ValueError(f'When multidimensional, {name} must match the shape of filled')
                return color
            else:
                raise ValueError(f'Invalid {name} argument')
        if facecolors is None:
            facecolors = self._get_patches_for_fill.get_next_color()
        facecolors = _broadcast_color_arg(facecolors, 'facecolors')
        edgecolors = _broadcast_color_arg(edgecolors, 'edgecolors')
        self.auto_scale_xyz(x, y, z)
        square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=np.intp)
        voxel_faces = defaultdict(list)

        def permutation_matrices(n):
            if False:
                while True:
                    i = 10
            'Generate cyclic permutation matrices.'
            mat = np.eye(n, dtype=np.intp)
            for i in range(n):
                yield mat
                mat = np.roll(mat, 1, axis=0)
        for permute in permutation_matrices(3):
            (pc, qc, rc) = permute.T.dot(size)
            pinds = np.arange(pc)
            qinds = np.arange(qc)
            rinds = np.arange(rc)
            square_rot_pos = square.dot(permute.T)
            square_rot_neg = square_rot_pos[::-1]
            for p in pinds:
                for q in qinds:
                    p0 = permute.dot([p, q, 0])
                    i0 = tuple(p0)
                    if filled[i0]:
                        voxel_faces[i0].append(p0 + square_rot_neg)
                    for (r1, r2) in zip(rinds[:-1], rinds[1:]):
                        p1 = permute.dot([p, q, r1])
                        p2 = permute.dot([p, q, r2])
                        i1 = tuple(p1)
                        i2 = tuple(p2)
                        if filled[i1] and (not filled[i2]):
                            voxel_faces[i1].append(p2 + square_rot_pos)
                        elif not filled[i1] and filled[i2]:
                            voxel_faces[i2].append(p2 + square_rot_neg)
                    pk = permute.dot([p, q, rc - 1])
                    pk2 = permute.dot([p, q, rc])
                    ik = tuple(pk)
                    if filled[ik]:
                        voxel_faces[ik].append(pk2 + square_rot_pos)
        polygons = {}
        for (coord, faces_inds) in voxel_faces.items():
            if xyz is None:
                faces = faces_inds
            else:
                faces = []
                for face_inds in faces_inds:
                    ind = (face_inds[:, 0], face_inds[:, 1], face_inds[:, 2])
                    face = np.empty(face_inds.shape)
                    face[:, 0] = x[ind]
                    face[:, 1] = y[ind]
                    face[:, 2] = z[ind]
                    faces.append(face)
            facecolor = facecolors[coord]
            edgecolor = edgecolors[coord]
            poly = art3d.Poly3DCollection(faces, facecolors=facecolor, edgecolors=edgecolor, shade=shade, lightsource=lightsource, **kwargs)
            self.add_collection3d(poly)
            polygons[coord] = poly
        return polygons

    @_preprocess_data(replace_names=['x', 'y', 'z', 'xerr', 'yerr', 'zerr'])
    def errorbar(self, x, y, z, zerr=None, yerr=None, xerr=None, fmt='', barsabove=False, errorevery=1, ecolor=None, elinewidth=None, capsize=None, capthick=None, xlolims=False, xuplims=False, ylolims=False, yuplims=False, zlolims=False, zuplims=False, **kwargs):
        if False:
            return 10
        "\n        Plot lines and/or markers with errorbars around them.\n\n        *x*/*y*/*z* define the data locations, and *xerr*/*yerr*/*zerr* define\n        the errorbar sizes. By default, this draws the data markers/lines as\n        well the errorbars. Use fmt='none' to draw errorbars only.\n\n        Parameters\n        ----------\n        x, y, z : float or array-like\n            The data positions.\n\n        xerr, yerr, zerr : float or array-like, shape (N,) or (2, N), optional\n            The errorbar sizes:\n\n            - scalar: Symmetric +/- values for all data points.\n            - shape(N,): Symmetric +/-values for each data point.\n            - shape(2, N): Separate - and + values for each bar. First row\n              contains the lower errors, the second row contains the upper\n              errors.\n            - *None*: No errorbar.\n\n            Note that all error arrays should have *positive* values.\n\n        fmt : str, default: ''\n            The format for the data points / data lines. See `.plot` for\n            details.\n\n            Use 'none' (case-insensitive) to plot errorbars without any data\n            markers.\n\n        ecolor : color, default: None\n            The color of the errorbar lines.  If None, use the color of the\n            line connecting the markers.\n\n        elinewidth : float, default: None\n            The linewidth of the errorbar lines. If None, the linewidth of\n            the current style is used.\n\n        capsize : float, default: :rc:`errorbar.capsize`\n            The length of the error bar caps in points.\n\n        capthick : float, default: None\n            An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).\n            This setting is a more sensible name for the property that\n            controls the thickness of the error bar cap in points. For\n            backwards compatibility, if *mew* or *markeredgewidth* are given,\n            then they will over-ride *capthick*. This may change in future\n            releases.\n\n        barsabove : bool, default: False\n            If True, will plot the errorbars above the plot\n            symbols. Default is below.\n\n        xlolims, ylolims, zlolims : bool, default: False\n            These arguments can be used to indicate that a value gives only\n            lower limits. In that case a caret symbol is used to indicate\n            this. *lims*-arguments may be scalars, or array-likes of the same\n            length as the errors. To use limits with inverted axes,\n            `~.set_xlim`, `~.set_ylim`, or `~.set_zlim` must be\n            called before `errorbar`. Note the tricky parameter names: setting\n            e.g. *ylolims* to True means that the y-value is a *lower* limit of\n            the True value, so, only an *upward*-pointing arrow will be drawn!\n\n        xuplims, yuplims, zuplims : bool, default: False\n            Same as above, but for controlling the upper limits.\n\n        errorevery : int or (int, int), default: 1\n            draws error bars on a subset of the data. *errorevery* =N draws\n            error bars on the points (x[::N], y[::N], z[::N]).\n            *errorevery* =(start, N) draws error bars on the points\n            (x[start::N], y[start::N], z[start::N]). e.g. *errorevery* =(6, 3)\n            adds error bars to the data at (x[6], x[9], x[12], x[15], ...).\n            Used to avoid overlapping error bars when two series share x-axis\n            values.\n\n        Returns\n        -------\n        errlines : list\n            List of `~mpl_toolkits.mplot3d.art3d.Line3DCollection` instances\n            each containing an errorbar line.\n        caplines : list\n            List of `~mpl_toolkits.mplot3d.art3d.Line3D` instances each\n            containing a capline object.\n        limmarks : list\n            List of `~mpl_toolkits.mplot3d.art3d.Line3D` instances each\n            containing a marker with an upper or lower limit.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            All other keyword arguments for styling errorbar lines are passed\n            `~mpl_toolkits.mplot3d.art3d.Line3DCollection`.\n\n        Examples\n        --------\n        .. plot:: gallery/mplot3d/errorbar3d.py\n        "
        had_data = self.has_data()
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
        kwargs.setdefault('zorder', 2)
        self._process_unit_info([('x', x), ('y', y), ('z', z)], kwargs, convert=False)
        x = x if np.iterable(x) else [x]
        y = y if np.iterable(y) else [y]
        z = z if np.iterable(z) else [z]
        if not len(x) == len(y) == len(z):
            raise ValueError("'x', 'y', and 'z' must have the same size")
        everymask = self._errorevery_to_mask(x, errorevery)
        label = kwargs.pop('label', None)
        kwargs['label'] = '_nolegend_'
        ((data_line, base_style),) = self._get_lines._plot_args(self, (x, y) if fmt == '' else (x, y, fmt), kwargs, return_kwargs=True)
        art3d.line_2d_to_3d(data_line, zs=z)
        if barsabove:
            data_line.set_zorder(kwargs['zorder'] - 0.1)
        else:
            data_line.set_zorder(kwargs['zorder'] + 0.1)
        if fmt.lower() != 'none':
            self.add_line(data_line)
        else:
            data_line = None
            base_style.pop('color')
        if 'color' not in base_style:
            base_style['color'] = 'C0'
        if ecolor is None:
            ecolor = base_style['color']
        for key in ['marker', 'markersize', 'markerfacecolor', 'markeredgewidth', 'markeredgecolor', 'markevery', 'linestyle', 'fillstyle', 'drawstyle', 'dash_capstyle', 'dash_joinstyle', 'solid_capstyle', 'solid_joinstyle']:
            base_style.pop(key, None)
        eb_lines_style = {**base_style, 'color': ecolor}
        if elinewidth:
            eb_lines_style['linewidth'] = elinewidth
        elif 'linewidth' in kwargs:
            eb_lines_style['linewidth'] = kwargs['linewidth']
        for key in ('transform', 'alpha', 'zorder', 'rasterized'):
            if key in kwargs:
                eb_lines_style[key] = kwargs[key]
        eb_cap_style = {**base_style, 'linestyle': 'None'}
        if capsize is None:
            capsize = mpl.rcParams['errorbar.capsize']
        if capsize > 0:
            eb_cap_style['markersize'] = 2.0 * capsize
        if capthick is not None:
            eb_cap_style['markeredgewidth'] = capthick
        eb_cap_style['color'] = ecolor

        def _apply_mask(arrays, mask):
            if False:
                while True:
                    i = 10
            return [[*itertools.compress(array, mask)] for array in arrays]

        def _extract_errs(err, data, lomask, himask):
            if False:
                i = 10
                return i + 15
            if len(err.shape) == 2:
                (low_err, high_err) = err
            else:
                (low_err, high_err) = (err, err)
            lows = np.where(lomask | ~everymask, data, data - low_err)
            highs = np.where(himask | ~everymask, data, data + high_err)
            return (lows, highs)
        (errlines, caplines, limmarks) = ([], [], [])
        coorderrs = []
        capmarker = {0: '|', 1: '|', 2: '_'}
        i_xyz = {'x': 0, 'y': 1, 'z': 2}
        quiversize = eb_cap_style.get('markersize', mpl.rcParams['lines.markersize']) ** 2
        quiversize *= self.figure.dpi / 72
        quiversize = self.transAxes.inverted().transform([(0, 0), (quiversize, quiversize)])
        quiversize = np.mean(np.diff(quiversize, axis=0))
        with cbook._setattr_cm(self, elev=0, azim=0, roll=0):
            invM = np.linalg.inv(self.get_proj())
        quiversize = np.dot(invM, [quiversize, 0, 0, 0])[1]
        quiversize *= 1.8660254037844388
        eb_quiver_style = {**eb_cap_style, 'length': quiversize, 'arrow_length_ratio': 1}
        eb_quiver_style.pop('markersize', None)
        for (zdir, data, err, lolims, uplims) in zip(['x', 'y', 'z'], [x, y, z], [xerr, yerr, zerr], [xlolims, ylolims, zlolims], [xuplims, yuplims, zuplims]):
            dir_vector = art3d.get_dir_vector(zdir)
            i_zdir = i_xyz[zdir]
            if err is None:
                continue
            if not np.iterable(err):
                err = [err] * len(data)
            err = np.atleast_1d(err)
            lolims = np.broadcast_to(lolims, len(data)).astype(bool)
            uplims = np.broadcast_to(uplims, len(data)).astype(bool)
            coorderr = [_extract_errs(err * dir_vector[i], coord, lolims, uplims) for (i, coord) in enumerate([x, y, z])]
            ((xl, xh), (yl, yh), (zl, zh)) = coorderr
            nolims = ~(lolims | uplims)
            if nolims.any() and capsize > 0:
                lo_caps_xyz = _apply_mask([xl, yl, zl], nolims & everymask)
                hi_caps_xyz = _apply_mask([xh, yh, zh], nolims & everymask)
                cap_lo = art3d.Line3D(*lo_caps_xyz, ls='', marker=capmarker[i_zdir], **eb_cap_style)
                cap_hi = art3d.Line3D(*hi_caps_xyz, ls='', marker=capmarker[i_zdir], **eb_cap_style)
                self.add_line(cap_lo)
                self.add_line(cap_hi)
                caplines.append(cap_lo)
                caplines.append(cap_hi)
            if lolims.any():
                (xh0, yh0, zh0) = _apply_mask([xh, yh, zh], lolims & everymask)
                self.quiver(xh0, yh0, zh0, *dir_vector, **eb_quiver_style)
            if uplims.any():
                (xl0, yl0, zl0) = _apply_mask([xl, yl, zl], uplims & everymask)
                self.quiver(xl0, yl0, zl0, *-dir_vector, **eb_quiver_style)
            errline = art3d.Line3DCollection(np.array(coorderr).T, **eb_lines_style)
            self.add_collection(errline)
            errlines.append(errline)
            coorderrs.append(coorderr)
        coorderrs = np.array(coorderrs)

        def _digout_minmax(err_arr, coord_label):
            if False:
                for i in range(10):
                    print('nop')
            return (np.nanmin(err_arr[:, i_xyz[coord_label], :, :]), np.nanmax(err_arr[:, i_xyz[coord_label], :, :]))
        (minx, maxx) = _digout_minmax(coorderrs, 'x')
        (miny, maxy) = _digout_minmax(coorderrs, 'y')
        (minz, maxz) = _digout_minmax(coorderrs, 'z')
        self.auto_scale_xyz((minx, maxx), (miny, maxy), (minz, maxz), had_data)
        errorbar_container = mcontainer.ErrorbarContainer((data_line, tuple(caplines), tuple(errlines)), has_xerr=xerr is not None or yerr is not None, has_yerr=zerr is not None, label=label)
        self.containers.append(errorbar_container)
        return (errlines, caplines, limmarks)

    @_api.make_keyword_only('3.8', 'call_axes_locator')
    def get_tightbbox(self, renderer=None, call_axes_locator=True, bbox_extra_artists=None, *, for_layout_only=False):
        if False:
            print('Hello World!')
        ret = super().get_tightbbox(renderer, call_axes_locator=call_axes_locator, bbox_extra_artists=bbox_extra_artists, for_layout_only=for_layout_only)
        batch = [ret]
        if self._axis3don:
            for axis in self._axis_map.values():
                if axis.get_visible():
                    axis_bb = martist._get_tightbbox_for_layout_only(axis, renderer)
                    if axis_bb:
                        batch.append(axis_bb)
        return mtransforms.Bbox.union(batch)

    @_preprocess_data()
    def stem(self, x, y, z, *, linefmt='C0-', markerfmt='C0o', basefmt='C3-', bottom=0, label=None, orientation='z'):
        if False:
            print('Hello World!')
        "\n        Create a 3D stem plot.\n\n        A stem plot draws lines perpendicular to a baseline, and places markers\n        at the heads. By default, the baseline is defined by *x* and *y*, and\n        stems are drawn vertically from *bottom* to *z*.\n\n        Parameters\n        ----------\n        x, y, z : array-like\n            The positions of the heads of the stems. The stems are drawn along\n            the *orientation*-direction from the baseline at *bottom* (in the\n            *orientation*-coordinate) to the heads. By default, the *x* and *y*\n            positions are used for the baseline and *z* for the head position,\n            but this can be changed by *orientation*.\n\n        linefmt : str, default: 'C0-'\n            A string defining the properties of the vertical lines. Usually,\n            this will be a color or a color and a linestyle:\n\n            =========  =============\n            Character  Line Style\n            =========  =============\n            ``'-'``    solid line\n            ``'--'``   dashed line\n            ``'-.'``   dash-dot line\n            ``':'``    dotted line\n            =========  =============\n\n            Note: While it is technically possible to specify valid formats\n            other than color or color and linestyle (e.g. 'rx' or '-.'), this\n            is beyond the intention of the method and will most likely not\n            result in a reasonable plot.\n\n        markerfmt : str, default: 'C0o'\n            A string defining the properties of the markers at the stem heads.\n\n        basefmt : str, default: 'C3-'\n            A format string defining the properties of the baseline.\n\n        bottom : float, default: 0\n            The position of the baseline, in *orientation*-coordinates.\n\n        label : str, default: None\n            The label to use for the stems in legends.\n\n        orientation : {'x', 'y', 'z'}, default: 'z'\n            The direction along which stems are drawn.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        Returns\n        -------\n        `.StemContainer`\n            The container may be treated like a tuple\n            (*markerline*, *stemlines*, *baseline*)\n\n        Examples\n        --------\n        .. plot:: gallery/mplot3d/stem3d_demo.py\n        "
        from matplotlib.container import StemContainer
        had_data = self.has_data()
        _api.check_in_list(['x', 'y', 'z'], orientation=orientation)
        xlim = (np.min(x), np.max(x))
        ylim = (np.min(y), np.max(y))
        zlim = (np.min(z), np.max(z))
        if orientation == 'x':
            (basex, basexlim) = (y, ylim)
            (basey, baseylim) = (z, zlim)
            lines = [[(bottom, thisy, thisz), (thisx, thisy, thisz)] for (thisx, thisy, thisz) in zip(x, y, z)]
        elif orientation == 'y':
            (basex, basexlim) = (x, xlim)
            (basey, baseylim) = (z, zlim)
            lines = [[(thisx, bottom, thisz), (thisx, thisy, thisz)] for (thisx, thisy, thisz) in zip(x, y, z)]
        else:
            (basex, basexlim) = (x, xlim)
            (basey, baseylim) = (y, ylim)
            lines = [[(thisx, thisy, bottom), (thisx, thisy, thisz)] for (thisx, thisy, thisz) in zip(x, y, z)]
        (linestyle, linemarker, linecolor) = _process_plot_format(linefmt)
        if linestyle is None:
            linestyle = mpl.rcParams['lines.linestyle']
        (baseline,) = self.plot(basex, basey, basefmt, zs=bottom, zdir=orientation, label='_nolegend_')
        stemlines = art3d.Line3DCollection(lines, linestyles=linestyle, colors=linecolor, label='_nolegend_')
        self.add_collection(stemlines)
        (markerline,) = self.plot(x, y, z, markerfmt, label='_nolegend_')
        stem_container = StemContainer((markerline, stemlines, baseline), label=label)
        self.add_container(stem_container)
        (jx, jy, jz) = art3d.juggle_axes(basexlim, baseylim, [bottom, bottom], orientation)
        self.auto_scale_xyz([*jx, *xlim], [*jy, *ylim], [*jz, *zlim], had_data)
        return stem_container
    stem3D = stem

def get_test_data(delta=0.05):
    if False:
        print('Hello World!')
    'Return a tuple X, Y, Z with a test data set.'
    x = y = np.arange(-3.0, 3.0, delta)
    (X, Y) = np.meshgrid(x, y)
    Z1 = np.exp(-(X ** 2 + Y ** 2) / 2) / (2 * np.pi)
    Z2 = np.exp(-(((X - 1) / 1.5) ** 2 + ((Y - 1) / 0.5) ** 2) / 2) / (2 * np.pi * 0.5 * 1.5)
    Z = Z2 - Z1
    X = X * 10
    Y = Y * 10
    Z = Z * 500
    return (X, Y, Z)