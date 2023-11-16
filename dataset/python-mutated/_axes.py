import functools
import itertools
import logging
import math
from numbers import Integral, Number, Real
import numpy as np
from numpy import ma
import matplotlib as mpl
import matplotlib.category
import matplotlib.cbook as cbook
import matplotlib.collections as mcoll
import matplotlib.colors as mcolors
import matplotlib.contour as mcontour
import matplotlib.dates
import matplotlib.image as mimage
import matplotlib.legend as mlegend
import matplotlib.lines as mlines
import matplotlib.markers as mmarkers
import matplotlib.mlab as mlab
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.quiver as mquiver
import matplotlib.stackplot as mstack
import matplotlib.streamplot as mstream
import matplotlib.table as mtable
import matplotlib.text as mtext
import matplotlib.ticker as mticker
import matplotlib.transforms as mtransforms
import matplotlib.tri as mtri
import matplotlib.units as munits
from matplotlib import _api, _docstring, _preprocess_data
from matplotlib.axes._base import _AxesBase, _TransformedBoundsLocator, _process_plot_format
from matplotlib.axes._secondary_axes import SecondaryAxis
from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer
_log = logging.getLogger(__name__)

@_docstring.interpd
class Axes(_AxesBase):
    """
    An Axes object encapsulates all the elements of an individual (sub-)plot in
    a figure.

    It contains most of the (sub-)plot elements: `~.axis.Axis`,
    `~.axis.Tick`, `~.lines.Line2D`, `~.text.Text`, `~.patches.Polygon`, etc.,
    and sets the coordinate system.

    Like all visible elements in a figure, Axes is an `.Artist` subclass.

    The `Axes` instance supports callbacks through a callbacks attribute which
    is a `~.cbook.CallbackRegistry` instance.  The events you can connect to
    are 'xlim_changed' and 'ylim_changed' and the callback will be called with
    func(*ax*) where *ax* is the `Axes` instance.

    .. note::

        As a user, you do not instantiate Axes directly, but use Axes creation
        methods instead; e.g. from `.pyplot` or `.Figure`:
        `~.pyplot.subplots`, `~.pyplot.subplot_mosaic` or `.Figure.add_axes`.

    Attributes
    ----------
    dataLim : `.Bbox`
        The bounding box enclosing all data displayed in the Axes.
    viewLim : `.Bbox`
        The view limits in data coordinates.

    """

    def get_title(self, loc='center'):
        if False:
            while True:
                i = 10
        "\n        Get an Axes title.\n\n        Get one of the three available Axes titles. The available titles\n        are positioned above the Axes in the center, flush with the left\n        edge, and flush with the right edge.\n\n        Parameters\n        ----------\n        loc : {'center', 'left', 'right'}, str, default: 'center'\n            Which title to return.\n\n        Returns\n        -------\n        str\n            The title text string.\n\n        "
        titles = {'left': self._left_title, 'center': self.title, 'right': self._right_title}
        title = _api.check_getitem(titles, loc=loc.lower())
        return title.get_text()

    def set_title(self, label, fontdict=None, loc=None, pad=None, *, y=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Set a title for the Axes.\n\n        Set one of the three available Axes titles. The available titles\n        are positioned above the Axes in the center, flush with the left\n        edge, and flush with the right edge.\n\n        Parameters\n        ----------\n        label : str\n            Text to use for the title\n\n        fontdict : dict\n\n            .. admonition:: Discouraged\n\n               The use of *fontdict* is discouraged. Parameters should be passed as\n               individual keyword arguments or using dictionary-unpacking\n               ``set_title(..., **fontdict)``.\n\n            A dictionary controlling the appearance of the title text,\n            the default *fontdict* is::\n\n               {'fontsize': rcParams['axes.titlesize'],\n                'fontweight': rcParams['axes.titleweight'],\n                'color': rcParams['axes.titlecolor'],\n                'verticalalignment': 'baseline',\n                'horizontalalignment': loc}\n\n        loc : {'center', 'left', 'right'}, default: :rc:`axes.titlelocation`\n            Which title to set.\n\n        y : float, default: :rc:`axes.titley`\n            Vertical Axes location for the title (1.0 is the top).  If\n            None (the default) and :rc:`axes.titley` is also None, y is\n            determined automatically to avoid decorators on the Axes.\n\n        pad : float, default: :rc:`axes.titlepad`\n            The offset of the title from the top of the Axes, in points.\n\n        Returns\n        -------\n        `.Text`\n            The matplotlib text instance representing the title\n\n        Other Parameters\n        ----------------\n        **kwargs : `~matplotlib.text.Text` properties\n            Other keyword arguments are text properties, see `.Text` for a list\n            of valid text properties.\n        "
        if loc is None:
            loc = mpl.rcParams['axes.titlelocation']
        if y is None:
            y = mpl.rcParams['axes.titley']
        if y is None:
            y = 1.0
        else:
            self._autotitlepos = False
        kwargs['y'] = y
        titles = {'left': self._left_title, 'center': self.title, 'right': self._right_title}
        title = _api.check_getitem(titles, loc=loc.lower())
        default = {'fontsize': mpl.rcParams['axes.titlesize'], 'fontweight': mpl.rcParams['axes.titleweight'], 'verticalalignment': 'baseline', 'horizontalalignment': loc.lower()}
        titlecolor = mpl.rcParams['axes.titlecolor']
        if not cbook._str_lower_equal(titlecolor, 'auto'):
            default['color'] = titlecolor
        if pad is None:
            pad = mpl.rcParams['axes.titlepad']
        self._set_title_offset_trans(float(pad))
        title.set_text(label)
        title.update(default)
        if fontdict is not None:
            title.update(fontdict)
        title._internal_update(kwargs)
        return title

    def get_legend_handles_labels(self, legend_handler_map=None):
        if False:
            while True:
                i = 10
        '\n        Return handles and labels for legend\n\n        ``ax.legend()`` is equivalent to ::\n\n          h, l = ax.get_legend_handles_labels()\n          ax.legend(h, l)\n        '
        (handles, labels) = mlegend._get_legend_handles_labels([self], legend_handler_map)
        return (handles, labels)

    @_docstring.dedent_interpd
    def legend(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Place a legend on the Axes.\n\n        Call signatures::\n\n            legend()\n            legend(handles, labels)\n            legend(handles=handles)\n            legend(labels)\n\n        The call signatures correspond to the following different ways to use\n        this method:\n\n        **1. Automatic detection of elements to be shown in the legend**\n\n        The elements to be added to the legend are automatically determined,\n        when you do not pass in any extra arguments.\n\n        In this case, the labels are taken from the artist. You can specify\n        them either at artist creation or by calling the\n        :meth:`~.Artist.set_label` method on the artist::\n\n            ax.plot([1, 2, 3], label=\'Inline label\')\n            ax.legend()\n\n        or::\n\n            line, = ax.plot([1, 2, 3])\n            line.set_label(\'Label via method\')\n            ax.legend()\n\n        .. note::\n            Specific artists can be excluded from the automatic legend element\n            selection by using a label starting with an underscore, "_".\n            A string starting with an underscore is the default label for all\n            artists, so calling `.Axes.legend` without any arguments and\n            without setting the labels manually will result in no legend being\n            drawn.\n\n\n        **2. Explicitly listing the artists and labels in the legend**\n\n        For full control of which artists have a legend entry, it is possible\n        to pass an iterable of legend artists followed by an iterable of\n        legend labels respectively::\n\n            ax.legend([line1, line2, line3], [\'label1\', \'label2\', \'label3\'])\n\n\n        **3. Explicitly listing the artists in the legend**\n\n        This is similar to 2, but the labels are taken from the artists\'\n        label properties. Example::\n\n            line1, = ax.plot([1, 2, 3], label=\'label1\')\n            line2, = ax.plot([1, 2, 3], label=\'label2\')\n            ax.legend(handles=[line1, line2])\n\n\n        **4. Labeling existing plot elements**\n\n        .. admonition:: Discouraged\n\n            This call signature is discouraged, because the relation between\n            plot elements and labels is only implicit by their order and can\n            easily be mixed up.\n\n        To make a legend for all artists on an Axes, call this function with\n        an iterable of strings, one for each legend item. For example::\n\n            ax.plot([1, 2, 3])\n            ax.plot([5, 6, 7])\n            ax.legend([\'First line\', \'Second line\'])\n\n\n        Parameters\n        ----------\n        handles : sequence of (`.Artist` or tuple of `.Artist`), optional\n            A list of Artists (lines, patches) to be added to the legend.\n            Use this together with *labels*, if you need full control on what\n            is shown in the legend and the automatic mechanism described above\n            is not sufficient.\n\n            The length of handles and labels should be the same in this\n            case. If they are not, they are truncated to the smaller length.\n\n            If an entry contains a tuple, then the legend handler for all Artists in the\n            tuple will be placed alongside a single label.\n\n        labels : list of str, optional\n            A list of labels to show next to the artists.\n            Use this together with *handles*, if you need full control on what\n            is shown in the legend and the automatic mechanism described above\n            is not sufficient.\n\n        Returns\n        -------\n        `~matplotlib.legend.Legend`\n\n        Other Parameters\n        ----------------\n        %(_legend_kw_axes)s\n\n        See Also\n        --------\n        .Figure.legend\n\n        Notes\n        -----\n        Some artists are not supported by this function.  See\n        :ref:`legend_guide` for details.\n\n        Examples\n        --------\n        .. plot:: gallery/text_labels_and_annotations/legend.py\n        '
        (handles, labels, kwargs) = mlegend._parse_legend_args([self], *args, **kwargs)
        self.legend_ = mlegend.Legend(self, handles, labels, **kwargs)
        self.legend_._remove_method = self._remove_legend
        return self.legend_

    def _remove_legend(self, legend):
        if False:
            return 10
        self.legend_ = None

    def inset_axes(self, bounds, *, transform=None, zorder=5, **kwargs):
        if False:
            return 10
        "\n        Add a child inset Axes to this existing Axes.\n\n        Warnings\n        --------\n        This method is experimental as of 3.0, and the API may change.\n\n        Parameters\n        ----------\n        bounds : [x0, y0, width, height]\n            Lower-left corner of inset Axes, and its width and height.\n\n        transform : `.Transform`\n            Defaults to `ax.transAxes`, i.e. the units of *rect* are in\n            Axes-relative coordinates.\n\n        projection : {None, 'aitoff', 'hammer', 'lambert', 'mollweide', 'polar', 'rectilinear', str}, optional\n            The projection type of the inset `~.axes.Axes`. *str* is the name\n            of a custom projection, see `~matplotlib.projections`. The default\n            None results in a 'rectilinear' projection.\n\n        polar : bool, default: False\n            If True, equivalent to projection='polar'.\n\n        axes_class : subclass type of `~.axes.Axes`, optional\n            The `.axes.Axes` subclass that is instantiated.  This parameter\n            is incompatible with *projection* and *polar*.  See\n            :ref:`axisartist_users-guide-index` for examples.\n\n        zorder : number\n            Defaults to 5 (same as `.Axes.legend`).  Adjust higher or lower\n            to change whether it is above or below data plotted on the\n            parent Axes.\n\n        **kwargs\n            Other keyword arguments are passed on to the inset Axes class.\n\n        Returns\n        -------\n        ax\n            The created `~.axes.Axes` instance.\n\n        Examples\n        --------\n        This example makes two inset Axes, the first is in Axes-relative\n        coordinates, and the second in data-coordinates::\n\n            fig, ax = plt.subplots()\n            ax.plot(range(10))\n            axin1 = ax.inset_axes([0.8, 0.1, 0.15, 0.15])\n            axin2 = ax.inset_axes(\n                    [5, 7, 2.3, 2.3], transform=ax.transData)\n\n        "
        if transform is None:
            transform = self.transAxes
        kwargs.setdefault('label', 'inset_axes')
        inset_locator = _TransformedBoundsLocator(bounds, transform)
        bounds = inset_locator(self, None).bounds
        (projection_class, pkw) = self.figure._process_projection_requirements(**kwargs)
        inset_ax = projection_class(self.figure, bounds, zorder=zorder, **pkw)
        inset_ax.set_axes_locator(inset_locator)
        self.add_child_axes(inset_ax)
        return inset_ax

    @_docstring.dedent_interpd
    def indicate_inset(self, bounds, inset_ax=None, *, transform=None, facecolor='none', edgecolor='0.5', alpha=0.5, zorder=4.99, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Add an inset indicator to the Axes.  This is a rectangle on the plot\n        at the position indicated by *bounds* that optionally has lines that\n        connect the rectangle to an inset Axes (`.Axes.inset_axes`).\n\n        Warnings\n        --------\n        This method is experimental as of 3.0, and the API may change.\n\n        Parameters\n        ----------\n        bounds : [x0, y0, width, height]\n            Lower-left corner of rectangle to be marked, and its width\n            and height.\n\n        inset_ax : `.Axes`\n            An optional inset Axes to draw connecting lines to.  Two lines are\n            drawn connecting the indicator box to the inset Axes on corners\n            chosen so as to not overlap with the indicator box.\n\n        transform : `.Transform`\n            Transform for the rectangle coordinates. Defaults to\n            `ax.transAxes`, i.e. the units of *rect* are in Axes-relative\n            coordinates.\n\n        facecolor : color, default: 'none'\n            Facecolor of the rectangle.\n\n        edgecolor : color, default: '0.5'\n            Color of the rectangle and color of the connecting lines.\n\n        alpha : float, default: 0.5\n            Transparency of the rectangle and connector lines.\n\n        zorder : float, default: 4.99\n            Drawing order of the rectangle and connector lines.  The default,\n            4.99, is just below the default level of inset Axes.\n\n        **kwargs\n            Other keyword arguments are passed on to the `.Rectangle` patch:\n\n            %(Rectangle:kwdoc)s\n\n        Returns\n        -------\n        rectangle_patch : `.patches.Rectangle`\n             The indicator frame.\n\n        connector_lines : 4-tuple of `.patches.ConnectionPatch`\n            The four connector lines connecting to (lower_left, upper_left,\n            lower_right upper_right) corners of *inset_ax*. Two lines are\n            set with visibility to *False*,  but the user can set the\n            visibility to True if the automatic choice is not deemed correct.\n\n        "
        self.apply_aspect()
        if transform is None:
            transform = self.transData
        kwargs.setdefault('label', '_indicate_inset')
        (x, y, width, height) = bounds
        rectangle_patch = mpatches.Rectangle((x, y), width, height, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha, zorder=zorder, transform=transform, **kwargs)
        self.add_patch(rectangle_patch)
        connects = []
        if inset_ax is not None:
            for xy_inset_ax in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                (ex, ey) = xy_inset_ax
                if self.xaxis.get_inverted():
                    ex = 1 - ex
                if self.yaxis.get_inverted():
                    ey = 1 - ey
                xy_data = (x + ex * width, y + ey * height)
                p = mpatches.ConnectionPatch(xyA=xy_inset_ax, coordsA=inset_ax.transAxes, xyB=xy_data, coordsB=self.transData, arrowstyle='-', zorder=zorder, edgecolor=edgecolor, alpha=alpha)
                connects.append(p)
                self.add_patch(p)
            pos = inset_ax.get_position()
            bboxins = pos.transformed(self.figure.transSubfigure)
            rectbbox = mtransforms.Bbox.from_bounds(*bounds).transformed(transform)
            x0 = rectbbox.x0 < bboxins.x0
            x1 = rectbbox.x1 < bboxins.x1
            y0 = rectbbox.y0 < bboxins.y0
            y1 = rectbbox.y1 < bboxins.y1
            connects[0].set_visible(x0 ^ y0)
            connects[1].set_visible(x0 == y1)
            connects[2].set_visible(x1 == y0)
            connects[3].set_visible(x1 ^ y1)
        return (rectangle_patch, tuple(connects) if connects else None)

    def indicate_inset_zoom(self, inset_ax, **kwargs):
        if False:
            print('Hello World!')
        '\n        Add an inset indicator rectangle to the Axes based on the axis\n        limits for an *inset_ax* and draw connectors between *inset_ax*\n        and the rectangle.\n\n        Warnings\n        --------\n        This method is experimental as of 3.0, and the API may change.\n\n        Parameters\n        ----------\n        inset_ax : `.Axes`\n            Inset Axes to draw connecting lines to.  Two lines are\n            drawn connecting the indicator box to the inset Axes on corners\n            chosen so as to not overlap with the indicator box.\n\n        **kwargs\n            Other keyword arguments are passed on to `.Axes.indicate_inset`\n\n        Returns\n        -------\n        rectangle_patch : `.patches.Rectangle`\n             Rectangle artist.\n\n        connector_lines : 4-tuple of `.patches.ConnectionPatch`\n            Each of four connector lines coming from the rectangle drawn on\n            this axis, in the order lower left, upper left, lower right,\n            upper right.\n            Two are set with visibility to *False*,  but the user can\n            set the visibility to *True* if the automatic choice is not deemed\n            correct.\n        '
        xlim = inset_ax.get_xlim()
        ylim = inset_ax.get_ylim()
        rect = (xlim[0], ylim[0], xlim[1] - xlim[0], ylim[1] - ylim[0])
        return self.indicate_inset(rect, inset_ax, **kwargs)

    @_docstring.dedent_interpd
    def secondary_xaxis(self, location, *, functions=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Add a second x-axis to this `~.axes.Axes`.\n\n        For example if we want to have a second scale for the data plotted on\n        the xaxis.\n\n        %(_secax_docstring)s\n\n        Examples\n        --------\n        The main axis shows frequency, and the secondary axis shows period.\n\n        .. plot::\n\n            fig, ax = plt.subplots()\n            ax.loglog(range(1, 360, 5), range(1, 360, 5))\n            ax.set_xlabel('frequency [Hz]')\n\n            def invert(x):\n                # 1/x with special treatment of x == 0\n                x = np.array(x).astype(float)\n                near_zero = np.isclose(x, 0)\n                x[near_zero] = np.inf\n                x[~near_zero] = 1 / x[~near_zero]\n                return x\n\n            # the inverse of 1/x is itself\n            secax = ax.secondary_xaxis('top', functions=(invert, invert))\n            secax.set_xlabel('Period [s]')\n            plt.show()\n        "
        if location in ['top', 'bottom'] or isinstance(location, Real):
            secondary_ax = SecondaryAxis(self, 'x', location, functions, **kwargs)
            self.add_child_axes(secondary_ax)
            return secondary_ax
        else:
            raise ValueError('secondary_xaxis location must be either a float or "top"/"bottom"')

    @_docstring.dedent_interpd
    def secondary_yaxis(self, location, *, functions=None, **kwargs):
        if False:
            return 10
        "\n        Add a second y-axis to this `~.axes.Axes`.\n\n        For example if we want to have a second scale for the data plotted on\n        the yaxis.\n\n        %(_secax_docstring)s\n\n        Examples\n        --------\n        Add a secondary Axes that converts from radians to degrees\n\n        .. plot::\n\n            fig, ax = plt.subplots()\n            ax.plot(range(1, 360, 5), range(1, 360, 5))\n            ax.set_ylabel('degrees')\n            secax = ax.secondary_yaxis('right', functions=(np.deg2rad,\n                                                           np.rad2deg))\n            secax.set_ylabel('radians')\n        "
        if location in ['left', 'right'] or isinstance(location, Real):
            secondary_ax = SecondaryAxis(self, 'y', location, functions, **kwargs)
            self.add_child_axes(secondary_ax)
            return secondary_ax
        else:
            raise ValueError('secondary_yaxis location must be either a float or "left"/"right"')

    @_docstring.dedent_interpd
    def text(self, x, y, s, fontdict=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Add text to the Axes.\n\n        Add the text *s* to the Axes at location *x*, *y* in data coordinates.\n\n        Parameters\n        ----------\n        x, y : float\n            The position to place the text. By default, this is in data\n            coordinates. The coordinate system can be changed using the\n            *transform* parameter.\n\n        s : str\n            The text.\n\n        fontdict : dict, default: None\n\n            .. admonition:: Discouraged\n\n               The use of *fontdict* is discouraged. Parameters should be passed as\n               individual keyword arguments or using dictionary-unpacking\n               ``text(..., **fontdict)``.\n\n            A dictionary to override the default text properties. If fontdict\n            is None, the defaults are determined by `.rcParams`.\n\n        Returns\n        -------\n        `.Text`\n            The created `.Text` instance.\n\n        Other Parameters\n        ----------------\n        **kwargs : `~matplotlib.text.Text` properties.\n            Other miscellaneous text parameters.\n\n            %(Text:kwdoc)s\n\n        Examples\n        --------\n        Individual keyword arguments can be used to override any given\n        parameter::\n\n            >>> text(x, y, s, fontsize=12)\n\n        The default transform specifies that text is in data coords,\n        alternatively, you can specify text in axis coords ((0, 0) is\n        lower-left and (1, 1) is upper-right).  The example below places\n        text in the center of the Axes::\n\n            >>> text(0.5, 0.5, 'matplotlib', horizontalalignment='center',\n            ...      verticalalignment='center', transform=ax.transAxes)\n\n        You can put a rectangular box around the text instance (e.g., to\n        set a background color) by using the keyword *bbox*.  *bbox* is\n        a dictionary of `~matplotlib.patches.Rectangle`\n        properties.  For example::\n\n            >>> text(x, y, s, bbox=dict(facecolor='red', alpha=0.5))\n        "
        effective_kwargs = {'verticalalignment': 'baseline', 'horizontalalignment': 'left', 'transform': self.transData, 'clip_on': False, **(fontdict if fontdict is not None else {}), **kwargs}
        t = mtext.Text(x, y, text=s, **effective_kwargs)
        if t.get_clip_path() is None:
            t.set_clip_path(self.patch)
        self._add_text(t)
        return t

    @_docstring.dedent_interpd
    def annotate(self, text, xy, xytext=None, xycoords='data', textcoords=None, arrowprops=None, annotation_clip=None, **kwargs):
        if False:
            i = 10
            return i + 15
        a = mtext.Annotation(text, xy, xytext=xytext, xycoords=xycoords, textcoords=textcoords, arrowprops=arrowprops, annotation_clip=annotation_clip, **kwargs)
        a.set_transform(mtransforms.IdentityTransform())
        if kwargs.get('clip_on', False) and a.get_clip_path() is None:
            a.set_clip_path(self.patch)
        self._add_text(a)
        return a
    annotate.__doc__ = mtext.Annotation.__init__.__doc__

    @_docstring.dedent_interpd
    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Add a horizontal line across the Axes.\n\n        Parameters\n        ----------\n        y : float, default: 0\n            y position in data coordinates of the horizontal line.\n\n        xmin : float, default: 0\n            Should be between 0 and 1, 0 being the far left of the plot, 1 the\n            far right of the plot.\n\n        xmax : float, default: 1\n            Should be between 0 and 1, 0 being the far left of the plot, 1 the\n            far right of the plot.\n\n        Returns\n        -------\n        `~matplotlib.lines.Line2D`\n\n        Other Parameters\n        ----------------\n        **kwargs\n            Valid keyword arguments are `.Line2D` properties, except for\n            'transform':\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        hlines : Add horizontal lines in data coordinates.\n        axhspan : Add a horizontal span (rectangle) across the axis.\n        axline : Add a line with an arbitrary slope.\n\n        Examples\n        --------\n        * draw a thick red hline at 'y' = 0 that spans the xrange::\n\n            >>> axhline(linewidth=4, color='r')\n\n        * draw a default hline at 'y' = 1 that spans the xrange::\n\n            >>> axhline(y=1)\n\n        * draw a default hline at 'y' = .5 that spans the middle half of\n          the xrange::\n\n            >>> axhline(y=.5, xmin=0.25, xmax=0.75)\n        "
        self._check_no_units([xmin, xmax], ['xmin', 'xmax'])
        if 'transform' in kwargs:
            raise ValueError("'transform' is not allowed as a keyword argument; axhline generates its own transform.")
        (ymin, ymax) = self.get_ybound()
        (yy,) = self._process_unit_info([('y', y)], kwargs)
        scaley = yy < ymin or yy > ymax
        trans = self.get_yaxis_transform(which='grid')
        l = mlines.Line2D([xmin, xmax], [y, y], transform=trans, **kwargs)
        self.add_line(l)
        l.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS
        if scaley:
            self._request_autoscale_view('y')
        return l

    @_docstring.dedent_interpd
    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Add a vertical line across the Axes.\n\n        Parameters\n        ----------\n        x : float, default: 0\n            x position in data coordinates of the vertical line.\n\n        ymin : float, default: 0\n            Should be between 0 and 1, 0 being the bottom of the plot, 1 the\n            top of the plot.\n\n        ymax : float, default: 1\n            Should be between 0 and 1, 0 being the bottom of the plot, 1 the\n            top of the plot.\n\n        Returns\n        -------\n        `~matplotlib.lines.Line2D`\n\n        Other Parameters\n        ----------------\n        **kwargs\n            Valid keyword arguments are `.Line2D` properties, except for\n            'transform':\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        vlines : Add vertical lines in data coordinates.\n        axvspan : Add a vertical span (rectangle) across the axis.\n        axline : Add a line with an arbitrary slope.\n\n        Examples\n        --------\n        * draw a thick red vline at *x* = 0 that spans the yrange::\n\n            >>> axvline(linewidth=4, color='r')\n\n        * draw a default vline at *x* = 1 that spans the yrange::\n\n            >>> axvline(x=1)\n\n        * draw a default vline at *x* = .5 that spans the middle half of\n          the yrange::\n\n            >>> axvline(x=.5, ymin=0.25, ymax=0.75)\n        "
        self._check_no_units([ymin, ymax], ['ymin', 'ymax'])
        if 'transform' in kwargs:
            raise ValueError("'transform' is not allowed as a keyword argument; axvline generates its own transform.")
        (xmin, xmax) = self.get_xbound()
        (xx,) = self._process_unit_info([('x', x)], kwargs)
        scalex = xx < xmin or xx > xmax
        trans = self.get_xaxis_transform(which='grid')
        l = mlines.Line2D([x, x], [ymin, ymax], transform=trans, **kwargs)
        self.add_line(l)
        l.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS
        if scalex:
            self._request_autoscale_view('x')
        return l

    @staticmethod
    def _check_no_units(vals, names):
        if False:
            while True:
                i = 10
        for (val, name) in zip(vals, names):
            if not munits._is_natively_supported(val):
                raise ValueError(f'{name} must be a single scalar value, but got {val}')

    @_docstring.dedent_interpd
    def axline(self, xy1, xy2=None, *, slope=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add an infinitely long straight line.\n\n        The line can be defined either by two points *xy1* and *xy2*, or\n        by one point *xy1* and a *slope*.\n\n        This draws a straight line "on the screen", regardless of the x and y\n        scales, and is thus also suitable for drawing exponential decays in\n        semilog plots, power laws in loglog plots, etc. However, *slope*\n        should only be used with linear scales; It has no clear meaning for\n        all other scales, and thus the behavior is undefined. Please specify\n        the line using the points *xy1*, *xy2* for non-linear scales.\n\n        The *transform* keyword argument only applies to the points *xy1*,\n        *xy2*. The *slope* (if given) is always in data coordinates. This can\n        be used e.g. with ``ax.transAxes`` for drawing grid lines with a fixed\n        slope.\n\n        Parameters\n        ----------\n        xy1, xy2 : (float, float)\n            Points for the line to pass through.\n            Either *xy2* or *slope* has to be given.\n        slope : float, optional\n            The slope of the line. Either *xy2* or *slope* has to be given.\n\n        Returns\n        -------\n        `.Line2D`\n\n        Other Parameters\n        ----------------\n        **kwargs\n            Valid kwargs are `.Line2D` properties\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        axhline : for horizontal lines\n        axvline : for vertical lines\n\n        Examples\n        --------\n        Draw a thick red line passing through (0, 0) and (1, 1)::\n\n            >>> axline((0, 0), (1, 1), linewidth=4, color=\'r\')\n        '
        if slope is not None and (self.get_xscale() != 'linear' or self.get_yscale() != 'linear'):
            raise TypeError("'slope' cannot be used with non-linear scales")
        datalim = [xy1] if xy2 is None else [xy1, xy2]
        if 'transform' in kwargs:
            datalim = []
        line = mlines.AxLine(xy1, xy2, slope, **kwargs)
        self._set_artist_props(line)
        if line.get_clip_path() is None:
            line.set_clip_path(self.patch)
        if not line.get_label():
            line.set_label(f'_child{len(self._children)}')
        self._children.append(line)
        line._remove_method = self._children.remove
        self.update_datalim(datalim)
        self._request_autoscale_view()
        return line

    @_docstring.dedent_interpd
    def axhspan(self, ymin, ymax, xmin=0, xmax=1, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add a horizontal span (rectangle) across the Axes.\n\n        The rectangle spans from *ymin* to *ymax* vertically, and, by default,\n        the whole x-axis horizontally.  The x-span can be set using *xmin*\n        (default: 0) and *xmax* (default: 1) which are in axis units; e.g.\n        ``xmin = 0.5`` always refers to the middle of the x-axis regardless of\n        the limits set by `~.Axes.set_xlim`.\n\n        Parameters\n        ----------\n        ymin : float\n            Lower y-coordinate of the span, in data units.\n        ymax : float\n            Upper y-coordinate of the span, in data units.\n        xmin : float, default: 0\n            Lower x-coordinate of the span, in x-axis (0-1) units.\n        xmax : float, default: 1\n            Upper x-coordinate of the span, in x-axis (0-1) units.\n\n        Returns\n        -------\n        `~matplotlib.patches.Polygon`\n            Horizontal span (rectangle) from (xmin, ymin) to (xmax, ymax).\n\n        Other Parameters\n        ----------------\n        **kwargs : `~matplotlib.patches.Polygon` properties\n\n        %(Polygon:kwdoc)s\n\n        See Also\n        --------\n        axvspan : Add a vertical span across the Axes.\n        '
        self._check_no_units([xmin, xmax], ['xmin', 'xmax'])
        ((ymin, ymax),) = self._process_unit_info([('y', [ymin, ymax])], kwargs)
        p = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
        p.set_transform(self.get_yaxis_transform(which='grid'))
        ix = self.dataLim.intervalx
        mx = self.dataLim.minposx
        self.add_patch(p)
        self.dataLim.intervalx = ix
        self.dataLim.minposx = mx
        p.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS
        self._request_autoscale_view('y')
        return p

    @_docstring.dedent_interpd
    def axvspan(self, xmin, xmax, ymin=0, ymax=1, **kwargs):
        if False:
            print('Hello World!')
        "\n        Add a vertical span (rectangle) across the Axes.\n\n        The rectangle spans from *xmin* to *xmax* horizontally, and, by\n        default, the whole y-axis vertically.  The y-span can be set using\n        *ymin* (default: 0) and *ymax* (default: 1) which are in axis units;\n        e.g. ``ymin = 0.5`` always refers to the middle of the y-axis\n        regardless of the limits set by `~.Axes.set_ylim`.\n\n        Parameters\n        ----------\n        xmin : float\n            Lower x-coordinate of the span, in data units.\n        xmax : float\n            Upper x-coordinate of the span, in data units.\n        ymin : float, default: 0\n            Lower y-coordinate of the span, in y-axis units (0-1).\n        ymax : float, default: 1\n            Upper y-coordinate of the span, in y-axis units (0-1).\n\n        Returns\n        -------\n        `~matplotlib.patches.Polygon`\n            Vertical span (rectangle) from (xmin, ymin) to (xmax, ymax).\n\n        Other Parameters\n        ----------------\n        **kwargs : `~matplotlib.patches.Polygon` properties\n\n        %(Polygon:kwdoc)s\n\n        See Also\n        --------\n        axhspan : Add a horizontal span across the Axes.\n\n        Examples\n        --------\n        Draw a vertical, green, translucent rectangle from x = 1.25 to\n        x = 1.55 that spans the yrange of the Axes.\n\n        >>> axvspan(1.25, 1.55, facecolor='g', alpha=0.5)\n\n        "
        self._check_no_units([ymin, ymax], ['ymin', 'ymax'])
        ((xmin, xmax),) = self._process_unit_info([('x', [xmin, xmax])], kwargs)
        p = mpatches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
        p.set_transform(self.get_xaxis_transform(which='grid'))
        iy = self.dataLim.intervaly.copy()
        my = self.dataLim.minposy
        self.add_patch(p)
        self.dataLim.intervaly = iy
        self.dataLim.minposy = my
        p.get_path()._interpolation_steps = mpl.axis.GRIDLINE_INTERPOLATION_STEPS
        self._request_autoscale_view('x')
        return p

    @_preprocess_data(replace_names=['y', 'xmin', 'xmax', 'colors'], label_namer='y')
    def hlines(self, y, xmin, xmax, colors=None, linestyles='solid', label='', **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Plot horizontal lines at each *y* from *xmin* to *xmax*.\n\n        Parameters\n        ----------\n        y : float or array-like\n            y-indexes where to plot the lines.\n\n        xmin, xmax : float or array-like\n            Respective beginning and end of each line. If scalars are\n            provided, all lines will have the same length.\n\n        colors : color or list of colors, default: :rc:`lines.color`\n\n        linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, default: 'solid'\n\n        label : str, default: ''\n\n        Returns\n        -------\n        `~matplotlib.collections.LineCollection`\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        **kwargs :  `~matplotlib.collections.LineCollection` properties.\n\n        See Also\n        --------\n        vlines : vertical lines\n        axhline : horizontal line across the Axes\n        "
        (xmin, xmax, y) = self._process_unit_info([('x', xmin), ('x', xmax), ('y', y)], kwargs)
        if not np.iterable(y):
            y = [y]
        if not np.iterable(xmin):
            xmin = [xmin]
        if not np.iterable(xmax):
            xmax = [xmax]
        (y, xmin, xmax) = cbook._combine_masks(y, xmin, xmax)
        y = np.ravel(y)
        xmin = np.ravel(xmin)
        xmax = np.ravel(xmax)
        masked_verts = np.ma.empty((len(y), 2, 2))
        masked_verts[:, 0, 0] = xmin
        masked_verts[:, 0, 1] = y
        masked_verts[:, 1, 0] = xmax
        masked_verts[:, 1, 1] = y
        lines = mcoll.LineCollection(masked_verts, colors=colors, linestyles=linestyles, label=label)
        self.add_collection(lines, autolim=False)
        lines._internal_update(kwargs)
        if len(y) > 0:
            updatex = True
            updatey = True
            if self.name == 'rectilinear':
                datalim = lines.get_datalim(self.transData)
                t = lines.get_transform()
                (updatex, updatey) = t.contains_branch_seperately(self.transData)
                minx = np.nanmin(datalim.xmin)
                maxx = np.nanmax(datalim.xmax)
                miny = np.nanmin(datalim.ymin)
                maxy = np.nanmax(datalim.ymax)
            else:
                minx = np.nanmin(masked_verts[..., 0])
                maxx = np.nanmax(masked_verts[..., 0])
                miny = np.nanmin(masked_verts[..., 1])
                maxy = np.nanmax(masked_verts[..., 1])
            corners = ((minx, miny), (maxx, maxy))
            self.update_datalim(corners, updatex, updatey)
            self._request_autoscale_view()
        return lines

    @_preprocess_data(replace_names=['x', 'ymin', 'ymax', 'colors'], label_namer='x')
    def vlines(self, x, ymin, ymax, colors=None, linestyles='solid', label='', **kwargs):
        if False:
            return 10
        "\n        Plot vertical lines at each *x* from *ymin* to *ymax*.\n\n        Parameters\n        ----------\n        x : float or array-like\n            x-indexes where to plot the lines.\n\n        ymin, ymax : float or array-like\n            Respective beginning and end of each line. If scalars are\n            provided, all lines will have the same length.\n\n        colors : color or list of colors, default: :rc:`lines.color`\n\n        linestyles : {'solid', 'dashed', 'dashdot', 'dotted'}, default: 'solid'\n\n        label : str, default: ''\n\n        Returns\n        -------\n        `~matplotlib.collections.LineCollection`\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        **kwargs : `~matplotlib.collections.LineCollection` properties.\n\n        See Also\n        --------\n        hlines : horizontal lines\n        axvline : vertical line across the Axes\n        "
        (x, ymin, ymax) = self._process_unit_info([('x', x), ('y', ymin), ('y', ymax)], kwargs)
        if not np.iterable(x):
            x = [x]
        if not np.iterable(ymin):
            ymin = [ymin]
        if not np.iterable(ymax):
            ymax = [ymax]
        (x, ymin, ymax) = cbook._combine_masks(x, ymin, ymax)
        x = np.ravel(x)
        ymin = np.ravel(ymin)
        ymax = np.ravel(ymax)
        masked_verts = np.ma.empty((len(x), 2, 2))
        masked_verts[:, 0, 0] = x
        masked_verts[:, 0, 1] = ymin
        masked_verts[:, 1, 0] = x
        masked_verts[:, 1, 1] = ymax
        lines = mcoll.LineCollection(masked_verts, colors=colors, linestyles=linestyles, label=label)
        self.add_collection(lines, autolim=False)
        lines._internal_update(kwargs)
        if len(x) > 0:
            updatex = True
            updatey = True
            if self.name == 'rectilinear':
                datalim = lines.get_datalim(self.transData)
                t = lines.get_transform()
                (updatex, updatey) = t.contains_branch_seperately(self.transData)
                minx = np.nanmin(datalim.xmin)
                maxx = np.nanmax(datalim.xmax)
                miny = np.nanmin(datalim.ymin)
                maxy = np.nanmax(datalim.ymax)
            else:
                minx = np.nanmin(masked_verts[..., 0])
                maxx = np.nanmax(masked_verts[..., 0])
                miny = np.nanmin(masked_verts[..., 1])
                maxy = np.nanmax(masked_verts[..., 1])
            corners = ((minx, miny), (maxx, maxy))
            self.update_datalim(corners, updatex, updatey)
            self._request_autoscale_view()
        return lines

    @_preprocess_data(replace_names=['positions', 'lineoffsets', 'linelengths', 'linewidths', 'colors', 'linestyles'])
    @_docstring.dedent_interpd
    def eventplot(self, positions, orientation='horizontal', lineoffsets=1, linelengths=1, linewidths=None, colors=None, alpha=None, linestyles='solid', **kwargs):
        if False:
            while True:
                i = 10
        "\n        Plot identical parallel lines at the given positions.\n\n        This type of plot is commonly used in neuroscience for representing\n        neural events, where it is usually called a spike raster, dot raster,\n        or raster plot.\n\n        However, it is useful in any situation where you wish to show the\n        timing or position of multiple sets of discrete events, such as the\n        arrival times of people to a business on each day of the month or the\n        date of hurricanes each year of the last century.\n\n        Parameters\n        ----------\n        positions : array-like or list of array-like\n            A 1D array-like defines the positions of one sequence of events.\n\n            Multiple groups of events may be passed as a list of array-likes.\n            Each group can be styled independently by passing lists of values\n            to *lineoffsets*, *linelengths*, *linewidths*, *colors* and\n            *linestyles*.\n\n            Note that *positions* can be a 2D array, but in practice different\n            event groups usually have different counts so that one will use a\n            list of different-length arrays rather than a 2D array.\n\n        orientation : {'horizontal', 'vertical'}, default: 'horizontal'\n            The direction of the event sequence:\n\n            - 'horizontal': the events are arranged horizontally.\n              The indicator lines are vertical.\n            - 'vertical': the events are arranged vertically.\n              The indicator lines are horizontal.\n\n        lineoffsets : float or array-like, default: 1\n            The offset of the center of the lines from the origin, in the\n            direction orthogonal to *orientation*.\n\n            If *positions* is 2D, this can be a sequence with length matching\n            the length of *positions*.\n\n        linelengths : float or array-like, default: 1\n            The total height of the lines (i.e. the lines stretches from\n            ``lineoffset - linelength/2`` to ``lineoffset + linelength/2``).\n\n            If *positions* is 2D, this can be a sequence with length matching\n            the length of *positions*.\n\n        linewidths : float or array-like, default: :rc:`lines.linewidth`\n            The line width(s) of the event lines, in points.\n\n            If *positions* is 2D, this can be a sequence with length matching\n            the length of *positions*.\n\n        colors : color or list of colors, default: :rc:`lines.color`\n            The color(s) of the event lines.\n\n            If *positions* is 2D, this can be a sequence with length matching\n            the length of *positions*.\n\n        alpha : float or array-like, default: 1\n            The alpha blending value(s), between 0 (transparent) and 1\n            (opaque).\n\n            If *positions* is 2D, this can be a sequence with length matching\n            the length of *positions*.\n\n        linestyles : str or tuple or list of such values, default: 'solid'\n            Default is 'solid'. Valid strings are ['solid', 'dashed',\n            'dashdot', 'dotted', '-', '--', '-.', ':']. Dash tuples\n            should be of the form::\n\n                (offset, onoffseq),\n\n            where *onoffseq* is an even length tuple of on and off ink\n            in points.\n\n            If *positions* is 2D, this can be a sequence with length matching\n            the length of *positions*.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Other keyword arguments are line collection properties.  See\n            `.LineCollection` for a list of the valid properties.\n\n        Returns\n        -------\n        list of `.EventCollection`\n            The `.EventCollection` that were added.\n\n        Notes\n        -----\n        For *linelengths*, *linewidths*, *colors*, *alpha* and *linestyles*, if\n        only a single value is given, that value is applied to all lines. If an\n        array-like is given, it must have the same length as *positions*, and\n        each value will be applied to the corresponding row of the array.\n\n        Examples\n        --------\n        .. plot:: gallery/lines_bars_and_markers/eventplot_demo.py\n        "
        (lineoffsets, linelengths) = self._process_unit_info([('y', lineoffsets), ('y', linelengths)], kwargs)
        if not np.iterable(positions):
            positions = [positions]
        elif any((np.iterable(position) for position in positions)):
            positions = [np.asanyarray(position) for position in positions]
        else:
            positions = [np.asanyarray(positions)]
        poss = []
        for position in positions:
            poss += self._process_unit_info([('x', position)], kwargs)
        positions = poss
        colors = cbook._local_over_kwdict(colors, kwargs, 'color')
        linewidths = cbook._local_over_kwdict(linewidths, kwargs, 'linewidth')
        linestyles = cbook._local_over_kwdict(linestyles, kwargs, 'linestyle')
        if not np.iterable(lineoffsets):
            lineoffsets = [lineoffsets]
        if not np.iterable(linelengths):
            linelengths = [linelengths]
        if not np.iterable(linewidths):
            linewidths = [linewidths]
        if not np.iterable(colors):
            colors = [colors]
        if not np.iterable(alpha):
            alpha = [alpha]
        if hasattr(linestyles, 'lower') or not np.iterable(linestyles):
            linestyles = [linestyles]
        lineoffsets = np.asarray(lineoffsets)
        linelengths = np.asarray(linelengths)
        linewidths = np.asarray(linewidths)
        if len(lineoffsets) == 0:
            raise ValueError('lineoffsets cannot be empty')
        if len(linelengths) == 0:
            raise ValueError('linelengths cannot be empty')
        if len(linestyles) == 0:
            raise ValueError('linestyles cannot be empty')
        if len(linewidths) == 0:
            raise ValueError('linewidths cannot be empty')
        if len(alpha) == 0:
            raise ValueError('alpha cannot be empty')
        if len(colors) == 0:
            colors = [None]
        try:
            colors = mcolors.to_rgba_array(colors)
        except ValueError:
            pass
        if len(lineoffsets) == 1 and len(positions) != 1:
            lineoffsets = np.tile(lineoffsets, len(positions))
            lineoffsets[0] = 0
            lineoffsets = np.cumsum(lineoffsets)
        if len(linelengths) == 1:
            linelengths = np.tile(linelengths, len(positions))
        if len(linewidths) == 1:
            linewidths = np.tile(linewidths, len(positions))
        if len(colors) == 1:
            colors = list(colors) * len(positions)
        if len(alpha) == 1:
            alpha = list(alpha) * len(positions)
        if len(linestyles) == 1:
            linestyles = [linestyles] * len(positions)
        if len(lineoffsets) != len(positions):
            raise ValueError('lineoffsets and positions are unequal sized sequences')
        if len(linelengths) != len(positions):
            raise ValueError('linelengths and positions are unequal sized sequences')
        if len(linewidths) != len(positions):
            raise ValueError('linewidths and positions are unequal sized sequences')
        if len(colors) != len(positions):
            raise ValueError('colors and positions are unequal sized sequences')
        if len(alpha) != len(positions):
            raise ValueError('alpha and positions are unequal sized sequences')
        if len(linestyles) != len(positions):
            raise ValueError('linestyles and positions are unequal sized sequences')
        colls = []
        for (position, lineoffset, linelength, linewidth, color, alpha_, linestyle) in zip(positions, lineoffsets, linelengths, linewidths, colors, alpha, linestyles):
            coll = mcoll.EventCollection(position, orientation=orientation, lineoffset=lineoffset, linelength=linelength, linewidth=linewidth, color=color, alpha=alpha_, linestyle=linestyle)
            self.add_collection(coll, autolim=False)
            coll._internal_update(kwargs)
            colls.append(coll)
        if len(positions) > 0:
            min_max = [(np.min(_p), np.max(_p)) for _p in positions if len(_p) > 0]
            if len(min_max) > 0:
                (mins, maxes) = zip(*min_max)
                minpos = np.min(mins)
                maxpos = np.max(maxes)
                minline = (lineoffsets - linelengths).min()
                maxline = (lineoffsets + linelengths).max()
                if orientation == 'vertical':
                    corners = ((minline, minpos), (maxline, maxpos))
                else:
                    corners = ((minpos, minline), (maxpos, maxline))
                self.update_datalim(corners)
                self._request_autoscale_view()
        return colls

    @_docstring.dedent_interpd
    def plot(self, *args, scalex=True, scaley=True, data=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Plot y versus x as lines and/or markers.\n\n        Call signatures::\n\n            plot([x], y, [fmt], *, data=None, **kwargs)\n            plot([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)\n\n        The coordinates of the points or line nodes are given by *x*, *y*.\n\n        The optional parameter *fmt* is a convenient way for defining basic\n        formatting like color, marker and linestyle. It's a shortcut string\n        notation described in the *Notes* section below.\n\n        >>> plot(x, y)        # plot x and y using default line style and color\n        >>> plot(x, y, 'bo')  # plot x and y using blue circle markers\n        >>> plot(y)           # plot y using x as index array 0..N-1\n        >>> plot(y, 'r+')     # ditto, but with red plusses\n\n        You can use `.Line2D` properties as keyword arguments for more\n        control on the appearance. Line properties and *fmt* can be mixed.\n        The following two calls yield identical results:\n\n        >>> plot(x, y, 'go--', linewidth=2, markersize=12)\n        >>> plot(x, y, color='green', marker='o', linestyle='dashed',\n        ...      linewidth=2, markersize=12)\n\n        When conflicting with *fmt*, keyword arguments take precedence.\n\n\n        **Plotting labelled data**\n\n        There's a convenient way for plotting objects with labelled data (i.e.\n        data that can be accessed by index ``obj['y']``). Instead of giving\n        the data in *x* and *y*, you can provide the object in the *data*\n        parameter and just give the labels for *x* and *y*::\n\n        >>> plot('xlabel', 'ylabel', data=obj)\n\n        All indexable objects are supported. This could e.g. be a `dict`, a\n        `pandas.DataFrame` or a structured numpy array.\n\n\n        **Plotting multiple sets of data**\n\n        There are various ways to plot multiple sets of data.\n\n        - The most straight forward way is just to call `plot` multiple times.\n          Example:\n\n          >>> plot(x1, y1, 'bo')\n          >>> plot(x2, y2, 'go')\n\n        - If *x* and/or *y* are 2D arrays a separate data set will be drawn\n          for every column. If both *x* and *y* are 2D, they must have the\n          same shape. If only one of them is 2D with shape (N, m) the other\n          must have length N and will be used for every data set m.\n\n          Example:\n\n          >>> x = [1, 2, 3]\n          >>> y = np.array([[1, 2], [3, 4], [5, 6]])\n          >>> plot(x, y)\n\n          is equivalent to:\n\n          >>> for col in range(y.shape[1]):\n          ...     plot(x, y[:, col])\n\n        - The third way is to specify multiple sets of *[x]*, *y*, *[fmt]*\n          groups::\n\n          >>> plot(x1, y1, 'g^', x2, y2, 'g-')\n\n          In this case, any additional keyword argument applies to all\n          datasets. Also, this syntax cannot be combined with the *data*\n          parameter.\n\n        By default, each line is assigned a different style specified by a\n        'style cycle'. The *fmt* and line property parameters are only\n        necessary if you want explicit deviations from these defaults.\n        Alternatively, you can also change the style cycle using\n        :rc:`axes.prop_cycle`.\n\n\n        Parameters\n        ----------\n        x, y : array-like or scalar\n            The horizontal / vertical coordinates of the data points.\n            *x* values are optional and default to ``range(len(y))``.\n\n            Commonly, these parameters are 1D arrays.\n\n            They can also be scalars, or two-dimensional (in that case, the\n            columns represent separate data sets).\n\n            These arguments cannot be passed as keywords.\n\n        fmt : str, optional\n            A format string, e.g. 'ro' for red circles. See the *Notes*\n            section for a full description of the format strings.\n\n            Format strings are just an abbreviation for quickly setting\n            basic line properties. All of these and more can also be\n            controlled by keyword arguments.\n\n            This argument cannot be passed as keyword.\n\n        data : indexable object, optional\n            An object with labelled data. If given, provide the label names to\n            plot in *x* and *y*.\n\n            .. note::\n                Technically there's a slight ambiguity in calls where the\n                second label is a valid *fmt*. ``plot('n', 'o', data=obj)``\n                could be ``plt(x, y)`` or ``plt(y, fmt)``. In such cases,\n                the former interpretation is chosen, but a warning is issued.\n                You may suppress the warning by adding an empty format string\n                ``plot('n', 'o', '', data=obj)``.\n\n        Returns\n        -------\n        list of `.Line2D`\n            A list of lines representing the plotted data.\n\n        Other Parameters\n        ----------------\n        scalex, scaley : bool, default: True\n            These parameters determine if the view limits are adapted to the\n            data limits. The values are passed on to\n            `~.axes.Axes.autoscale_view`.\n\n        **kwargs : `~matplotlib.lines.Line2D` properties, optional\n            *kwargs* are used to specify properties like a line label (for\n            auto legends), linewidth, antialiasing, marker face color.\n            Example::\n\n            >>> plot([1, 2, 3], [1, 2, 3], 'go-', label='line 1', linewidth=2)\n            >>> plot([1, 2, 3], [1, 4, 9], 'rs', label='line 2')\n\n            If you specify multiple lines with one plot call, the kwargs apply\n            to all those lines. In case the label object is iterable, each\n            element is used as labels for each set of data.\n\n            Here is a list of available `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        scatter : XY scatter plot with markers of varying size and/or color (\n            sometimes also called bubble chart).\n\n        Notes\n        -----\n        **Format Strings**\n\n        A format string consists of a part for color, marker and line::\n\n            fmt = '[marker][line][color]'\n\n        Each of them is optional. If not provided, the value from the style\n        cycle is used. Exception: If ``line`` is given, but no ``marker``,\n        the data will be a line without markers.\n\n        Other combinations such as ``[color][marker][line]`` are also\n        supported, but note that their parsing may be ambiguous.\n\n        **Markers**\n\n        =============   ===============================\n        character       description\n        =============   ===============================\n        ``'.'``         point marker\n        ``','``         pixel marker\n        ``'o'``         circle marker\n        ``'v'``         triangle_down marker\n        ``'^'``         triangle_up marker\n        ``'<'``         triangle_left marker\n        ``'>'``         triangle_right marker\n        ``'1'``         tri_down marker\n        ``'2'``         tri_up marker\n        ``'3'``         tri_left marker\n        ``'4'``         tri_right marker\n        ``'8'``         octagon marker\n        ``'s'``         square marker\n        ``'p'``         pentagon marker\n        ``'P'``         plus (filled) marker\n        ``'*'``         star marker\n        ``'h'``         hexagon1 marker\n        ``'H'``         hexagon2 marker\n        ``'+'``         plus marker\n        ``'x'``         x marker\n        ``'X'``         x (filled) marker\n        ``'D'``         diamond marker\n        ``'d'``         thin_diamond marker\n        ``'|'``         vline marker\n        ``'_'``         hline marker\n        =============   ===============================\n\n        **Line Styles**\n\n        =============    ===============================\n        character        description\n        =============    ===============================\n        ``'-'``          solid line style\n        ``'--'``         dashed line style\n        ``'-.'``         dash-dot line style\n        ``':'``          dotted line style\n        =============    ===============================\n\n        Example format strings::\n\n            'b'    # blue markers with default shape\n            'or'   # red circles\n            '-g'   # green solid line\n            '--'   # dashed line with default color\n            '^k:'  # black triangle_up markers connected by a dotted line\n\n        **Colors**\n\n        The supported color abbreviations are the single letter codes\n\n        =============    ===============================\n        character        color\n        =============    ===============================\n        ``'b'``          blue\n        ``'g'``          green\n        ``'r'``          red\n        ``'c'``          cyan\n        ``'m'``          magenta\n        ``'y'``          yellow\n        ``'k'``          black\n        ``'w'``          white\n        =============    ===============================\n\n        and the ``'CN'`` colors that index into the default property cycle.\n\n        If the color is the only part of the format string, you can\n        additionally use any  `matplotlib.colors` spec, e.g. full names\n        (``'green'``) or hex strings (``'#008000'``).\n        "
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        lines = [*self._get_lines(self, *args, data=data, **kwargs)]
        for line in lines:
            self.add_line(line)
        if scalex:
            self._request_autoscale_view('x')
        if scaley:
            self._request_autoscale_view('y')
        return lines

    @_preprocess_data(replace_names=['x', 'y'], label_namer='y')
    @_docstring.dedent_interpd
    def plot_date(self, x, y, fmt='o', tz=None, xdate=True, ydate=False, **kwargs):
        if False:
            return 10
        '\n        [*Discouraged*] Plot coercing the axis to treat floats as dates.\n\n        .. admonition:: Discouraged\n\n            This method exists for historic reasons and will be deprecated in\n            the future.\n\n            - ``datetime``-like data should directly be plotted using\n              `~.Axes.plot`.\n            -  If you need to plot plain numeric data as :ref:`date-format` or\n               need to set a timezone, call ``ax.xaxis.axis_date`` /\n               ``ax.yaxis.axis_date`` before `~.Axes.plot`. See\n               `.Axis.axis_date`.\n\n        Similar to `.plot`, this plots *y* vs. *x* as lines or markers.\n        However, the axis labels are formatted as dates depending on *xdate*\n        and *ydate*.  Note that `.plot` will work with `datetime` and\n        `numpy.datetime64` objects without resorting to this method.\n\n        Parameters\n        ----------\n        x, y : array-like\n            The coordinates of the data points. If *xdate* or *ydate* is\n            *True*, the respective values *x* or *y* are interpreted as\n            :ref:`Matplotlib dates <date-format>`.\n\n        fmt : str, optional\n            The plot format string. For details, see the corresponding\n            parameter in `.plot`.\n\n        tz : timezone string or `datetime.tzinfo`, default: :rc:`timezone`\n            The time zone to use in labeling dates.\n\n        xdate : bool, default: True\n            If *True*, the *x*-axis will be interpreted as Matplotlib dates.\n\n        ydate : bool, default: False\n            If *True*, the *y*-axis will be interpreted as Matplotlib dates.\n\n        Returns\n        -------\n        list of `.Line2D`\n            Objects representing the plotted data.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        matplotlib.dates : Helper functions on dates.\n        matplotlib.dates.date2num : Convert dates to num.\n        matplotlib.dates.num2date : Convert num to dates.\n        matplotlib.dates.drange : Create an equally spaced sequence of dates.\n\n        Notes\n        -----\n        If you are using custom date tickers and formatters, it may be\n        necessary to set the formatters/locators after the call to\n        `.plot_date`. `.plot_date` will set the default tick locator to\n        `.AutoDateLocator` (if the tick locator is not already set to a\n        `.DateLocator` instance) and the default tick formatter to\n        `.AutoDateFormatter` (if the tick formatter is not already set to a\n        `.DateFormatter` instance).\n        '
        if xdate:
            self.xaxis_date(tz)
        if ydate:
            self.yaxis_date(tz)
        return self.plot(x, y, fmt, **kwargs)

    @_docstring.dedent_interpd
    def loglog(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Make a plot with log scaling on both the x- and y-axis.\n\n        Call signatures::\n\n            loglog([x], y, [fmt], data=None, **kwargs)\n            loglog([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)\n\n        This is just a thin wrapper around `.plot` which additionally changes\n        both the x-axis and the y-axis to log scaling. All the concepts and\n        parameters of plot can be used here as well.\n\n        The additional parameters *base*, *subs* and *nonpositive* control the\n        x/y-axis properties. They are just forwarded to `.Axes.set_xscale` and\n        `.Axes.set_yscale`. To use different properties on the x-axis and the\n        y-axis, use e.g.\n        ``ax.set_xscale("log", base=10); ax.set_yscale("log", base=2)``.\n\n        Parameters\n        ----------\n        base : float, default: 10\n            Base of the logarithm.\n\n        subs : sequence, optional\n            The location of the minor ticks. If *None*, reasonable locations\n            are automatically chosen depending on the number of decades in the\n            plot. See `.Axes.set_xscale`/`.Axes.set_yscale` for details.\n\n        nonpositive : {\'mask\', \'clip\'}, default: \'clip\'\n            Non-positive values can be masked as invalid, or clipped to a very\n            small positive number.\n\n        **kwargs\n            All parameters supported by `.plot`.\n\n        Returns\n        -------\n        list of `.Line2D`\n            Objects representing the plotted data.\n        '
        dx = {k: v for (k, v) in kwargs.items() if k in ['base', 'subs', 'nonpositive', 'basex', 'subsx', 'nonposx']}
        self.set_xscale('log', **dx)
        dy = {k: v for (k, v) in kwargs.items() if k in ['base', 'subs', 'nonpositive', 'basey', 'subsy', 'nonposy']}
        self.set_yscale('log', **dy)
        return self.plot(*args, **{k: v for (k, v) in kwargs.items() if k not in {*dx, *dy}})

    @_docstring.dedent_interpd
    def semilogx(self, *args, **kwargs):
        if False:
            print('Hello World!')
        "\n        Make a plot with log scaling on the x-axis.\n\n        Call signatures::\n\n            semilogx([x], y, [fmt], data=None, **kwargs)\n            semilogx([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)\n\n        This is just a thin wrapper around `.plot` which additionally changes\n        the x-axis to log scaling. All the concepts and parameters of plot can\n        be used here as well.\n\n        The additional parameters *base*, *subs*, and *nonpositive* control the\n        x-axis properties. They are just forwarded to `.Axes.set_xscale`.\n\n        Parameters\n        ----------\n        base : float, default: 10\n            Base of the x logarithm.\n\n        subs : array-like, optional\n            The location of the minor xticks. If *None*, reasonable locations\n            are automatically chosen depending on the number of decades in the\n            plot. See `.Axes.set_xscale` for details.\n\n        nonpositive : {'mask', 'clip'}, default: 'clip'\n            Non-positive values in x can be masked as invalid, or clipped to a\n            very small positive number.\n\n        **kwargs\n            All parameters supported by `.plot`.\n\n        Returns\n        -------\n        list of `.Line2D`\n            Objects representing the plotted data.\n        "
        d = {k: v for (k, v) in kwargs.items() if k in ['base', 'subs', 'nonpositive', 'basex', 'subsx', 'nonposx']}
        self.set_xscale('log', **d)
        return self.plot(*args, **{k: v for (k, v) in kwargs.items() if k not in d})

    @_docstring.dedent_interpd
    def semilogy(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Make a plot with log scaling on the y-axis.\n\n        Call signatures::\n\n            semilogy([x], y, [fmt], data=None, **kwargs)\n            semilogy([x], y, [fmt], [x2], y2, [fmt2], ..., **kwargs)\n\n        This is just a thin wrapper around `.plot` which additionally changes\n        the y-axis to log scaling. All the concepts and parameters of plot can\n        be used here as well.\n\n        The additional parameters *base*, *subs*, and *nonpositive* control the\n        y-axis properties. They are just forwarded to `.Axes.set_yscale`.\n\n        Parameters\n        ----------\n        base : float, default: 10\n            Base of the y logarithm.\n\n        subs : array-like, optional\n            The location of the minor yticks. If *None*, reasonable locations\n            are automatically chosen depending on the number of decades in the\n            plot. See `.Axes.set_yscale` for details.\n\n        nonpositive : {'mask', 'clip'}, default: 'clip'\n            Non-positive values in y can be masked as invalid, or clipped to a\n            very small positive number.\n\n        **kwargs\n            All parameters supported by `.plot`.\n\n        Returns\n        -------\n        list of `.Line2D`\n            Objects representing the plotted data.\n        "
        d = {k: v for (k, v) in kwargs.items() if k in ['base', 'subs', 'nonpositive', 'basey', 'subsy', 'nonposy']}
        self.set_yscale('log', **d)
        return self.plot(*args, **{k: v for (k, v) in kwargs.items() if k not in d})

    @_preprocess_data(replace_names=['x'], label_namer='x')
    def acorr(self, x, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Plot the autocorrelation of *x*.\n\n        Parameters\n        ----------\n        x : array-like\n\n        detrend : callable, default: `.mlab.detrend_none` (no detrending)\n            A detrending function applied to *x*.  It must have the\n            signature ::\n\n                detrend(x: np.ndarray) -> np.ndarray\n\n        normed : bool, default: True\n            If ``True``, input vectors are normalised to unit length.\n\n        usevlines : bool, default: True\n            Determines the plot style.\n\n            If ``True``, vertical lines are plotted from 0 to the acorr value\n            using `.Axes.vlines`. Additionally, a horizontal line is plotted\n            at y=0 using `.Axes.axhline`.\n\n            If ``False``, markers are plotted at the acorr values using\n            `.Axes.plot`.\n\n        maxlags : int, default: 10\n            Number of lags to show. If ``None``, will return all\n            ``2 * len(x) - 1`` lags.\n\n        Returns\n        -------\n        lags : array (length ``2*maxlags+1``)\n            The lag vector.\n        c : array  (length ``2*maxlags+1``)\n            The auto correlation vector.\n        line : `.LineCollection` or `.Line2D`\n            `.Artist` added to the Axes of the correlation:\n\n            - `.LineCollection` if *usevlines* is True.\n            - `.Line2D` if *usevlines* is False.\n        b : `~matplotlib.lines.Line2D` or None\n            Horizontal line at 0 if *usevlines* is True\n            None *usevlines* is False.\n\n        Other Parameters\n        ----------------\n        linestyle : `~matplotlib.lines.Line2D` property, optional\n            The linestyle for plotting the data points.\n            Only used if *usevlines* is ``False``.\n\n        marker : str, default: \'o\'\n            The marker for plotting the data points.\n            Only used if *usevlines* is ``False``.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Additional parameters are passed to `.Axes.vlines` and\n            `.Axes.axhline` if *usevlines* is ``True``; otherwise they are\n            passed to `.Axes.plot`.\n\n        Notes\n        -----\n        The cross correlation is performed with `numpy.correlate` with\n        ``mode = "full"``.\n        '
        return self.xcorr(x, x, **kwargs)

    @_preprocess_data(replace_names=['x', 'y'], label_namer='y')
    def xcorr(self, x, y, normed=True, detrend=mlab.detrend_none, usevlines=True, maxlags=10, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Plot the cross correlation between *x* and *y*.\n\n        The correlation with lag k is defined as\n        :math:`\\sum_n x[n+k] \\cdot y^*[n]`, where :math:`y^*` is the complex\n        conjugate of :math:`y`.\n\n        Parameters\n        ----------\n        x, y : array-like of length n\n\n        detrend : callable, default: `.mlab.detrend_none` (no detrending)\n            A detrending function applied to *x* and *y*.  It must have the\n            signature ::\n\n                detrend(x: np.ndarray) -> np.ndarray\n\n        normed : bool, default: True\n            If ``True``, input vectors are normalised to unit length.\n\n        usevlines : bool, default: True\n            Determines the plot style.\n\n            If ``True``, vertical lines are plotted from 0 to the xcorr value\n            using `.Axes.vlines`. Additionally, a horizontal line is plotted\n            at y=0 using `.Axes.axhline`.\n\n            If ``False``, markers are plotted at the xcorr values using\n            `.Axes.plot`.\n\n        maxlags : int, default: 10\n            Number of lags to show. If None, will return all ``2 * len(x) - 1``\n            lags.\n\n        Returns\n        -------\n        lags : array (length ``2*maxlags+1``)\n            The lag vector.\n        c : array  (length ``2*maxlags+1``)\n            The auto correlation vector.\n        line : `.LineCollection` or `.Line2D`\n            `.Artist` added to the Axes of the correlation:\n\n            - `.LineCollection` if *usevlines* is True.\n            - `.Line2D` if *usevlines* is False.\n        b : `~matplotlib.lines.Line2D` or None\n            Horizontal line at 0 if *usevlines* is True\n            None *usevlines* is False.\n\n        Other Parameters\n        ----------------\n        linestyle : `~matplotlib.lines.Line2D` property, optional\n            The linestyle for plotting the data points.\n            Only used if *usevlines* is ``False``.\n\n        marker : str, default: \'o\'\n            The marker for plotting the data points.\n            Only used if *usevlines* is ``False``.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Additional parameters are passed to `.Axes.vlines` and\n            `.Axes.axhline` if *usevlines* is ``True``; otherwise they are\n            passed to `.Axes.plot`.\n\n        Notes\n        -----\n        The cross correlation is performed with `numpy.correlate` with\n        ``mode = "full"``.\n        '
        Nx = len(x)
        if Nx != len(y):
            raise ValueError('x and y must be equal length')
        x = detrend(np.asarray(x))
        y = detrend(np.asarray(y))
        correls = np.correlate(x, y, mode='full')
        if normed:
            correls = correls / np.sqrt(np.dot(x, x) * np.dot(y, y))
        if maxlags is None:
            maxlags = Nx - 1
        if maxlags >= Nx or maxlags < 1:
            raise ValueError('maxlags must be None or strictly positive < %d' % Nx)
        lags = np.arange(-maxlags, maxlags + 1)
        correls = correls[Nx - 1 - maxlags:Nx + maxlags]
        if usevlines:
            a = self.vlines(lags, [0], correls, **kwargs)
            kwargs.pop('label', '')
            b = self.axhline(**kwargs)
        else:
            kwargs.setdefault('marker', 'o')
            kwargs.setdefault('linestyle', 'None')
            (a,) = self.plot(lags, correls, **kwargs)
            b = None
        return (lags, correls, a, b)

    def step(self, x, y, *args, where='pre', data=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Make a step plot.\n\n        Call signatures::\n\n            step(x, y, [fmt], *, data=None, where='pre', **kwargs)\n            step(x, y, [fmt], x2, y2, [fmt2], ..., *, where='pre', **kwargs)\n\n        This is just a thin wrapper around `.plot` which changes some\n        formatting options. Most of the concepts and parameters of plot can be\n        used here as well.\n\n        .. note::\n\n            This method uses a standard plot with a step drawstyle: The *x*\n            values are the reference positions and steps extend left/right/both\n            directions depending on *where*.\n\n            For the common case where you know the values and edges of the\n            steps, use `~.Axes.stairs` instead.\n\n        Parameters\n        ----------\n        x : array-like\n            1D sequence of x positions. It is assumed, but not checked, that\n            it is uniformly increasing.\n\n        y : array-like\n            1D sequence of y levels.\n\n        fmt : str, optional\n            A format string, e.g. 'g' for a green line. See `.plot` for a more\n            detailed description.\n\n            Note: While full format strings are accepted, it is recommended to\n            only specify the color. Line styles are currently ignored (use\n            the keyword argument *linestyle* instead). Markers are accepted\n            and plotted on the given positions, however, this is a rarely\n            needed feature for step plots.\n\n        where : {'pre', 'post', 'mid'}, default: 'pre'\n            Define where the steps should be placed:\n\n            - 'pre': The y value is continued constantly to the left from\n              every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the\n              value ``y[i]``.\n            - 'post': The y value is continued constantly to the right from\n              every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the\n              value ``y[i]``.\n            - 'mid': Steps occur half-way between the *x* positions.\n\n        data : indexable object, optional\n            An object with labelled data. If given, provide the label names to\n            plot in *x* and *y*.\n\n        **kwargs\n            Additional parameters are the same as those for `.plot`.\n\n        Returns\n        -------\n        list of `.Line2D`\n            Objects representing the plotted data.\n        "
        _api.check_in_list(('pre', 'post', 'mid'), where=where)
        kwargs['drawstyle'] = 'steps-' + where
        return self.plot(x, y, *args, data=data, **kwargs)

    @staticmethod
    def _convert_dx(dx, x0, xconv, convert):
        if False:
            i = 10
            return i + 15
        '\n        Small helper to do logic of width conversion flexibly.\n\n        *dx* and *x0* have units, but *xconv* has already been converted\n        to unitless (and is an ndarray).  This allows the *dx* to have units\n        that are different from *x0*, but are still accepted by the\n        ``__add__`` operator of *x0*.\n        '
        assert type(xconv) is np.ndarray
        if xconv.size == 0:
            return convert(dx)
        try:
            try:
                x0 = cbook._safe_first_finite(x0)
            except (TypeError, IndexError, KeyError):
                pass
            try:
                x = cbook._safe_first_finite(xconv)
            except (TypeError, IndexError, KeyError):
                x = xconv
            delist = False
            if not np.iterable(dx):
                dx = [dx]
                delist = True
            dx = [convert(x0 + ddx) - x for ddx in dx]
            if delist:
                dx = dx[0]
        except (ValueError, TypeError, AttributeError):
            dx = convert(dx)
        return dx

    @_preprocess_data()
    @_docstring.dedent_interpd
    def bar(self, x, height, width=0.8, bottom=None, *, align='center', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Make a bar plot.\n\n        The bars are positioned at *x* with the given *align*\\ment. Their\n        dimensions are given by *height* and *width*. The vertical baseline\n        is *bottom* (default 0).\n\n        Many parameters can take either a single value applying to all bars\n        or a sequence of values, one for each bar.\n\n        Parameters\n        ----------\n        x : float or array-like\n            The x coordinates of the bars. See also *align* for the\n            alignment of the bars to the coordinates.\n\n        height : float or array-like\n            The height(s) of the bars.\n\n            Note that if *bottom* has units (e.g. datetime), *height* should be in\n            units that are a difference from the value of *bottom* (e.g. timedelta).\n\n        width : float or array-like, default: 0.8\n            The width(s) of the bars.\n\n            Note that if *x* has units (e.g. datetime), then *width* should be in\n            units that are a difference (e.g. timedelta) around the *x* values.\n\n        bottom : float or array-like, default: 0\n            The y coordinate(s) of the bottom side(s) of the bars.\n\n            Note that if *bottom* has units, then the y-axis will get a Locator and\n            Formatter appropriate for the units (e.g. dates, or categorical).\n\n        align : {'center', 'edge'}, default: 'center'\n            Alignment of the bars to the *x* coordinates:\n\n            - 'center': Center the base on the *x* positions.\n            - 'edge': Align the left edges of the bars with the *x* positions.\n\n            To align the bars on the right edge pass a negative *width* and\n            ``align='edge'``.\n\n        Returns\n        -------\n        `.BarContainer`\n            Container with all the bars and optionally errorbars.\n\n        Other Parameters\n        ----------------\n        color : color or list of color, optional\n            The colors of the bar faces.\n\n        edgecolor : color or list of color, optional\n            The colors of the bar edges.\n\n        linewidth : float or array-like, optional\n            Width of the bar edge(s). If 0, don't draw edges.\n\n        tick_label : str or list of str, optional\n            The tick labels of the bars.\n            Default: None (Use default numeric labels.)\n\n        label : str or list of str, optional\n            A single label is attached to the resulting `.BarContainer` as a\n            label for the whole dataset.\n            If a list is provided, it must be the same length as *x* and\n            labels the individual bars. Repeated labels are not de-duplicated\n            and will cause repeated label entries, so this is best used when\n            bars also differ in style (e.g., by passing a list to *color*.)\n\n        xerr, yerr : float or array-like of shape(N,) or shape(2, N), optional\n            If not *None*, add horizontal / vertical errorbars to the bar tips.\n            The values are +/- sizes relative to the data:\n\n            - scalar: symmetric +/- values for all bars\n            - shape(N,): symmetric +/- values for each bar\n            - shape(2, N): Separate - and + values for each bar. First row\n              contains the lower errors, the second row contains the upper\n              errors.\n            - *None*: No errorbar. (Default)\n\n            See :doc:`/gallery/statistics/errorbar_features` for an example on\n            the usage of *xerr* and *yerr*.\n\n        ecolor : color or list of color, default: 'black'\n            The line color of the errorbars.\n\n        capsize : float, default: :rc:`errorbar.capsize`\n           The length of the error bar caps in points.\n\n        error_kw : dict, optional\n            Dictionary of keyword arguments to be passed to the\n            `~.Axes.errorbar` method. Values of *ecolor* or *capsize* defined\n            here take precedence over the independent keyword arguments.\n\n        log : bool, default: False\n            If *True*, set the y-axis to be log scale.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs : `.Rectangle` properties\n\n        %(Rectangle:kwdoc)s\n\n        See Also\n        --------\n        barh : Plot a horizontal bar plot.\n\n        Notes\n        -----\n        Stacked bars can be achieved by passing individual *bottom* values per\n        bar. See :doc:`/gallery/lines_bars_and_markers/bar_stacked`.\n        "
        kwargs = cbook.normalize_kwargs(kwargs, mpatches.Patch)
        color = kwargs.pop('color', None)
        if color is None:
            color = self._get_patches_for_fill.get_next_color()
        edgecolor = kwargs.pop('edgecolor', None)
        linewidth = kwargs.pop('linewidth', None)
        hatch = kwargs.pop('hatch', None)
        xerr = kwargs.pop('xerr', None)
        yerr = kwargs.pop('yerr', None)
        error_kw = kwargs.pop('error_kw', {})
        ezorder = error_kw.pop('zorder', None)
        if ezorder is None:
            ezorder = kwargs.get('zorder', None)
            if ezorder is not None:
                ezorder += 0.01
        error_kw.setdefault('zorder', ezorder)
        ecolor = kwargs.pop('ecolor', 'k')
        capsize = kwargs.pop('capsize', mpl.rcParams['errorbar.capsize'])
        error_kw.setdefault('ecolor', ecolor)
        error_kw.setdefault('capsize', capsize)
        orientation = kwargs.pop('orientation', 'vertical')
        _api.check_in_list(['vertical', 'horizontal'], orientation=orientation)
        log = kwargs.pop('log', False)
        label = kwargs.pop('label', '')
        tick_labels = kwargs.pop('tick_label', None)
        y = bottom
        if orientation == 'vertical':
            if y is None:
                y = 0
        elif x is None:
            x = 0
        if orientation == 'vertical':
            self._process_unit_info([('x', x), ('y', y), ('y', height)], kwargs, convert=False)
            if log:
                self.set_yscale('log', nonpositive='clip')
        else:
            self._process_unit_info([('x', x), ('x', width), ('y', y)], kwargs, convert=False)
            if log:
                self.set_xscale('log', nonpositive='clip')
        if self.xaxis is not None:
            x0 = x
            x = np.asarray(self.convert_xunits(x))
            width = self._convert_dx(width, x0, x, self.convert_xunits)
            if xerr is not None:
                xerr = self._convert_dx(xerr, x0, x, self.convert_xunits)
        if self.yaxis is not None:
            y0 = y
            y = np.asarray(self.convert_yunits(y))
            height = self._convert_dx(height, y0, y, self.convert_yunits)
            if yerr is not None:
                yerr = self._convert_dx(yerr, y0, y, self.convert_yunits)
        (x, height, width, y, linewidth, hatch) = np.broadcast_arrays(np.atleast_1d(x), height, width, y, linewidth, hatch)
        if orientation == 'vertical':
            tick_label_axis = self.xaxis
            tick_label_position = x
        else:
            tick_label_axis = self.yaxis
            tick_label_position = y
        if not isinstance(label, str) and np.iterable(label):
            bar_container_label = '_nolegend_'
            patch_labels = label
        else:
            bar_container_label = label
            patch_labels = ['_nolegend_'] * len(x)
        if len(patch_labels) != len(x):
            raise ValueError(f'number of labels ({len(patch_labels)}) does not match number of bars ({len(x)}).')
        linewidth = itertools.cycle(np.atleast_1d(linewidth))
        hatch = itertools.cycle(np.atleast_1d(hatch))
        color = itertools.chain(itertools.cycle(mcolors.to_rgba_array(color)), itertools.repeat('none'))
        if edgecolor is None:
            edgecolor = itertools.repeat(None)
        else:
            edgecolor = itertools.chain(itertools.cycle(mcolors.to_rgba_array(edgecolor)), itertools.repeat('none'))
        _api.check_in_list(['center', 'edge'], align=align)
        if align == 'center':
            if orientation == 'vertical':
                try:
                    left = x - width / 2
                except TypeError as e:
                    raise TypeError(f'the dtypes of parameters x ({x.dtype}) and width ({width.dtype}) are incompatible') from e
                bottom = y
            else:
                try:
                    bottom = y - height / 2
                except TypeError as e:
                    raise TypeError(f'the dtypes of parameters y ({y.dtype}) and height ({height.dtype}) are incompatible') from e
                left = x
        else:
            left = x
            bottom = y
        patches = []
        args = zip(left, bottom, width, height, color, edgecolor, linewidth, hatch, patch_labels)
        for (l, b, w, h, c, e, lw, htch, lbl) in args:
            r = mpatches.Rectangle(xy=(l, b), width=w, height=h, facecolor=c, edgecolor=e, linewidth=lw, label=lbl, hatch=htch)
            r._internal_update(kwargs)
            r.get_path()._interpolation_steps = 100
            if orientation == 'vertical':
                r.sticky_edges.y.append(b)
            else:
                r.sticky_edges.x.append(l)
            self.add_patch(r)
            patches.append(r)
        if xerr is not None or yerr is not None:
            if orientation == 'vertical':
                ex = [l + 0.5 * w for (l, w) in zip(left, width)]
                ey = [b + h for (b, h) in zip(bottom, height)]
            else:
                ex = [l + w for (l, w) in zip(left, width)]
                ey = [b + 0.5 * h for (b, h) in zip(bottom, height)]
            error_kw.setdefault('label', '_nolegend_')
            errorbar = self.errorbar(ex, ey, yerr=yerr, xerr=xerr, fmt='none', **error_kw)
        else:
            errorbar = None
        self._request_autoscale_view()
        if orientation == 'vertical':
            datavalues = height
        else:
            datavalues = width
        bar_container = BarContainer(patches, errorbar, datavalues=datavalues, orientation=orientation, label=bar_container_label)
        self.add_container(bar_container)
        if tick_labels is not None:
            tick_labels = np.broadcast_to(tick_labels, len(patches))
            tick_label_axis.set_ticks(tick_label_position)
            tick_label_axis.set_ticklabels(tick_labels)
        return bar_container

    @_docstring.dedent_interpd
    def barh(self, y, width, height=0.8, left=None, *, align='center', data=None, **kwargs):
        if False:
            print('Hello World!')
        "\n        Make a horizontal bar plot.\n\n        The bars are positioned at *y* with the given *align*\\ment. Their\n        dimensions are given by *width* and *height*. The horizontal baseline\n        is *left* (default 0).\n\n        Many parameters can take either a single value applying to all bars\n        or a sequence of values, one for each bar.\n\n        Parameters\n        ----------\n        y : float or array-like\n            The y coordinates of the bars. See also *align* for the\n            alignment of the bars to the coordinates.\n\n        width : float or array-like\n            The width(s) of the bars.\n\n            Note that if *left* has units (e.g. datetime), *width* should be in\n            units that are a difference from the value of *left* (e.g. timedelta).\n\n        height : float or array-like, default: 0.8\n            The heights of the bars.\n\n            Note that if *y* has units (e.g. datetime), then *height* should be in\n            units that are a difference (e.g. timedelta) around the *y* values.\n\n        left : float or array-like, default: 0\n            The x coordinates of the left side(s) of the bars.\n\n            Note that if *left* has units, then the x-axis will get a Locator and\n            Formatter appropriate for the units (e.g. dates, or categorical).\n\n        align : {'center', 'edge'}, default: 'center'\n            Alignment of the base to the *y* coordinates*:\n\n            - 'center': Center the bars on the *y* positions.\n            - 'edge': Align the bottom edges of the bars with the *y*\n              positions.\n\n            To align the bars on the top edge pass a negative *height* and\n            ``align='edge'``.\n\n        Returns\n        -------\n        `.BarContainer`\n            Container with all the bars and optionally errorbars.\n\n        Other Parameters\n        ----------------\n        color : color or list of color, optional\n            The colors of the bar faces.\n\n        edgecolor : color or list of color, optional\n            The colors of the bar edges.\n\n        linewidth : float or array-like, optional\n            Width of the bar edge(s). If 0, don't draw edges.\n\n        tick_label : str or list of str, optional\n            The tick labels of the bars.\n            Default: None (Use default numeric labels.)\n\n        label : str or list of str, optional\n            A single label is attached to the resulting `.BarContainer` as a\n            label for the whole dataset.\n            If a list is provided, it must be the same length as *y* and\n            labels the individual bars. Repeated labels are not de-duplicated\n            and will cause repeated label entries, so this is best used when\n            bars also differ in style (e.g., by passing a list to *color*.)\n\n        xerr, yerr : float or array-like of shape(N,) or shape(2, N), optional\n            If not *None*, add horizontal / vertical errorbars to the bar tips.\n            The values are +/- sizes relative to the data:\n\n            - scalar: symmetric +/- values for all bars\n            - shape(N,): symmetric +/- values for each bar\n            - shape(2, N): Separate - and + values for each bar. First row\n              contains the lower errors, the second row contains the upper\n              errors.\n            - *None*: No errorbar. (default)\n\n            See :doc:`/gallery/statistics/errorbar_features` for an example on\n            the usage of *xerr* and *yerr*.\n\n        ecolor : color or list of color, default: 'black'\n            The line color of the errorbars.\n\n        capsize : float, default: :rc:`errorbar.capsize`\n           The length of the error bar caps in points.\n\n        error_kw : dict, optional\n            Dictionary of keyword arguments to be passed to the\n            `~.Axes.errorbar` method. Values of *ecolor* or *capsize* defined\n            here take precedence over the independent keyword arguments.\n\n        log : bool, default: False\n            If ``True``, set the x-axis to be log scale.\n\n        data : indexable object, optional\n            If given, all parameters also accept a string ``s``, which is\n            interpreted as ``data[s]`` (unless this raises an exception).\n\n        **kwargs : `.Rectangle` properties\n\n        %(Rectangle:kwdoc)s\n\n        See Also\n        --------\n        bar : Plot a vertical bar plot.\n\n        Notes\n        -----\n        Stacked bars can be achieved by passing individual *left* values per\n        bar. See\n        :doc:`/gallery/lines_bars_and_markers/horizontal_barchart_distribution`.\n        "
        kwargs.setdefault('orientation', 'horizontal')
        patches = self.bar(x=left, height=height, width=width, bottom=y, align=align, data=data, **kwargs)
        return patches

    def bar_label(self, container, labels=None, *, fmt='%g', label_type='edge', padding=0, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Label a bar plot.\n\n        Adds labels to bars in the given `.BarContainer`.\n        You may need to adjust the axis limits to fit the labels.\n\n        Parameters\n        ----------\n        container : `.BarContainer`\n            Container with all the bars and optionally errorbars, likely\n            returned from `.bar` or `.barh`.\n\n        labels : array-like, optional\n            A list of label texts, that should be displayed. If not given, the\n            label texts will be the data values formatted with *fmt*.\n\n        fmt : str or callable, default: '%g'\n            An unnamed %-style or {}-style format string for the label or a\n            function to call with the value as the first argument.\n            When *fmt* is a string and can be interpreted in both formats,\n            %-style takes precedence over {}-style.\n\n            .. versionadded:: 3.7\n               Support for {}-style format string and callables.\n\n        label_type : {'edge', 'center'}, default: 'edge'\n            The label type. Possible values:\n\n            - 'edge': label placed at the end-point of the bar segment, and the\n              value displayed will be the position of that end-point.\n            - 'center': label placed in the center of the bar segment, and the\n              value displayed will be the length of that segment.\n              (useful for stacked bars, i.e.,\n              :doc:`/gallery/lines_bars_and_markers/bar_label_demo`)\n\n        padding : float, default: 0\n            Distance of label from the end of the bar, in points.\n\n        **kwargs\n            Any remaining keyword arguments are passed through to\n            `.Axes.annotate`. The alignment parameters (\n            *horizontalalignment* / *ha*, *verticalalignment* / *va*) are\n            not supported because the labels are automatically aligned to\n            the bars.\n\n        Returns\n        -------\n        list of `.Annotation`\n            A list of `.Annotation` instances for the labels.\n        "
        for key in ['horizontalalignment', 'ha', 'verticalalignment', 'va']:
            if key in kwargs:
                raise ValueError(f'Passing {key!r} to bar_label() is not supported.')
        (a, b) = self.yaxis.get_view_interval()
        y_inverted = a > b
        (c, d) = self.xaxis.get_view_interval()
        x_inverted = c > d

        def sign(x):
            if False:
                return 10
            return 1 if x >= 0 else -1
        _api.check_in_list(['edge', 'center'], label_type=label_type)
        bars = container.patches
        errorbar = container.errorbar
        datavalues = container.datavalues
        orientation = container.orientation
        if errorbar:
            lines = errorbar.lines
            barlinecols = lines[2]
            barlinecol = barlinecols[0]
            errs = barlinecol.get_segments()
        else:
            errs = []
        if labels is None:
            labels = []
        annotations = []
        for (bar, err, dat, lbl) in itertools.zip_longest(bars, errs, datavalues, labels):
            ((x0, y0), (x1, y1)) = bar.get_bbox().get_points()
            (xc, yc) = ((x0 + x1) / 2, (y0 + y1) / 2)
            if orientation == 'vertical':
                extrema = max(y0, y1) if dat >= 0 else min(y0, y1)
                length = abs(y0 - y1)
            else:
                extrema = max(x0, x1) if dat >= 0 else min(x0, x1)
                length = abs(x0 - x1)
            if err is None or np.size(err) == 0:
                endpt = extrema
            elif orientation == 'vertical':
                endpt = err[:, 1].max() if dat >= 0 else err[:, 1].min()
            else:
                endpt = err[:, 0].max() if dat >= 0 else err[:, 0].min()
            if label_type == 'center':
                value = sign(dat) * length
            else:
                value = extrema
            if label_type == 'center':
                xy = (0.5, 0.5)
                kwargs['xycoords'] = lambda r, b=bar: mtransforms.Bbox.intersection(b.get_window_extent(r), b.get_clip_box()) or mtransforms.Bbox.null()
            elif orientation == 'vertical':
                xy = (xc, endpt)
            else:
                xy = (endpt, yc)
            if orientation == 'vertical':
                y_direction = -1 if y_inverted else 1
                xytext = (0, y_direction * sign(dat) * padding)
            else:
                x_direction = -1 if x_inverted else 1
                xytext = (x_direction * sign(dat) * padding, 0)
            if label_type == 'center':
                (ha, va) = ('center', 'center')
            elif orientation == 'vertical':
                ha = 'center'
                if y_inverted:
                    va = 'top' if dat > 0 else 'bottom'
                else:
                    va = 'top' if dat < 0 else 'bottom'
            else:
                if x_inverted:
                    ha = 'right' if dat > 0 else 'left'
                else:
                    ha = 'right' if dat < 0 else 'left'
                va = 'center'
            if np.isnan(dat):
                lbl = ''
            if lbl is None:
                if isinstance(fmt, str):
                    lbl = cbook._auto_format_str(fmt, value)
                elif callable(fmt):
                    lbl = fmt(value)
                else:
                    raise TypeError('fmt must be a str or callable')
            annotation = self.annotate(lbl, xy, xytext, textcoords='offset points', ha=ha, va=va, **kwargs)
            annotations.append(annotation)
        return annotations

    @_preprocess_data()
    @_docstring.dedent_interpd
    def broken_barh(self, xranges, yrange, **kwargs):
        if False:
            print('Hello World!')
        "\n        Plot a horizontal sequence of rectangles.\n\n        A rectangle is drawn for each element of *xranges*. All rectangles\n        have the same vertical position and size defined by *yrange*.\n\n        Parameters\n        ----------\n        xranges : sequence of tuples (*xmin*, *xwidth*)\n            The x-positions and extents of the rectangles. For each tuple\n            (*xmin*, *xwidth*) a rectangle is drawn from *xmin* to *xmin* +\n            *xwidth*.\n        yrange : (*ymin*, *yheight*)\n            The y-position and extent for all the rectangles.\n\n        Returns\n        -------\n        `~.collections.PolyCollection`\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        **kwargs : `.PolyCollection` properties\n\n            Each *kwarg* can be either a single argument applying to all\n            rectangles, e.g.::\n\n                facecolors='black'\n\n            or a sequence of arguments over which is cycled, e.g.::\n\n                facecolors=('black', 'blue')\n\n            would create interleaving black and blue rectangles.\n\n            Supported keywords:\n\n            %(PolyCollection:kwdoc)s\n        "
        xdata = cbook._safe_first_finite(xranges) if len(xranges) else None
        ydata = cbook._safe_first_finite(yrange) if len(yrange) else None
        self._process_unit_info([('x', xdata), ('y', ydata)], kwargs, convert=False)
        vertices = []
        (y0, dy) = yrange
        (y0, y1) = self.convert_yunits((y0, y0 + dy))
        for xr in xranges:
            try:
                (x0, dx) = xr
            except Exception:
                raise ValueError('each range in xrange must be a sequence with two elements (i.e. xrange must be an (N, 2) array)') from None
            (x0, x1) = self.convert_xunits((x0, x0 + dx))
            vertices.append([(x0, y0), (x0, y1), (x1, y1), (x1, y0)])
        col = mcoll.PolyCollection(np.array(vertices), **kwargs)
        self.add_collection(col, autolim=True)
        self._request_autoscale_view()
        return col

    @_preprocess_data()
    def stem(self, *args, linefmt=None, markerfmt=None, basefmt=None, bottom=0, label=None, orientation='vertical'):
        if False:
            for i in range(10):
                print('nop')
        "\n        Create a stem plot.\n\n        A stem plot draws lines perpendicular to a baseline at each location\n        *locs* from the baseline to *heads*, and places a marker there. For\n        vertical stem plots (the default), the *locs* are *x* positions, and\n        the *heads* are *y* values. For horizontal stem plots, the *locs* are\n        *y* positions, and the *heads* are *x* values.\n\n        Call signature::\n\n          stem([locs,] heads, linefmt=None, markerfmt=None, basefmt=None)\n\n        The *locs*-positions are optional. *linefmt* may be provided as\n        positional, but all other formats must be provided as keyword\n        arguments.\n\n        Parameters\n        ----------\n        locs : array-like, default: (0, 1, ..., len(heads) - 1)\n            For vertical stem plots, the x-positions of the stems.\n            For horizontal stem plots, the y-positions of the stems.\n\n        heads : array-like\n            For vertical stem plots, the y-values of the stem heads.\n            For horizontal stem plots, the x-values of the stem heads.\n\n        linefmt : str, optional\n            A string defining the color and/or linestyle of the vertical lines:\n\n            =========  =============\n            Character  Line Style\n            =========  =============\n            ``'-'``    solid line\n            ``'--'``   dashed line\n            ``'-.'``   dash-dot line\n            ``':'``    dotted line\n            =========  =============\n\n            Default: 'C0-', i.e. solid line with the first color of the color\n            cycle.\n\n            Note: Markers specified through this parameter (e.g. 'x') will be\n            silently ignored. Instead, markers should be specified using\n            *markerfmt*.\n\n        markerfmt : str, optional\n            A string defining the color and/or shape of the markers at the stem\n            heads. If the marker is not given, use the marker 'o', i.e. filled\n            circles. If the color is not given, use the color from *linefmt*.\n\n        basefmt : str, default: 'C3-' ('C2-' in classic mode)\n            A format string defining the properties of the baseline.\n\n        orientation : {'vertical', 'horizontal'}, default: 'vertical'\n            The orientation of the stems.\n\n        bottom : float, default: 0\n            The y/x-position of the baseline (depending on *orientation*).\n\n        label : str, default: None\n            The label to use for the stems in legends.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        Returns\n        -------\n        `.StemContainer`\n            The container may be treated like a tuple\n            (*markerline*, *stemlines*, *baseline*)\n\n        Notes\n        -----\n        .. seealso::\n            The MATLAB function\n            `stem <https://www.mathworks.com/help/matlab/ref/stem.html>`_\n            which inspired this method.\n        "
        if not 1 <= len(args) <= 3:
            raise _api.nargs_error('stem', '1-3', len(args))
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)
        if len(args) == 1:
            (heads,) = args
            locs = np.arange(len(heads))
            args = ()
        elif isinstance(args[1], str):
            (heads, *args) = args
            locs = np.arange(len(heads))
        else:
            (locs, heads, *args) = args
        if orientation == 'vertical':
            (locs, heads) = self._process_unit_info([('x', locs), ('y', heads)])
        else:
            (heads, locs) = self._process_unit_info([('x', heads), ('y', locs)])
        if linefmt is None:
            linefmt = args[0] if len(args) > 0 else 'C0-'
        (linestyle, linemarker, linecolor) = _process_plot_format(linefmt)
        if markerfmt is None:
            markerfmt = 'o'
        if markerfmt == '':
            markerfmt = ' '
        (markerstyle, markermarker, markercolor) = _process_plot_format(markerfmt)
        if markermarker is None:
            markermarker = 'o'
        if markerstyle is None:
            markerstyle = 'None'
        if markercolor is None:
            markercolor = linecolor
        if basefmt is None:
            basefmt = 'C2-' if mpl.rcParams['_internal.classic_mode'] else 'C3-'
        (basestyle, basemarker, basecolor) = _process_plot_format(basefmt)
        if linestyle is None:
            linestyle = mpl.rcParams['lines.linestyle']
        xlines = self.vlines if orientation == 'vertical' else self.hlines
        stemlines = xlines(locs, bottom, heads, colors=linecolor, linestyles=linestyle, label='_nolegend_')
        if orientation == 'horizontal':
            marker_x = heads
            marker_y = locs
            baseline_x = [bottom, bottom]
            baseline_y = [np.min(locs), np.max(locs)]
        else:
            marker_x = locs
            marker_y = heads
            baseline_x = [np.min(locs), np.max(locs)]
            baseline_y = [bottom, bottom]
        (markerline,) = self.plot(marker_x, marker_y, color=markercolor, linestyle=markerstyle, marker=markermarker, label='_nolegend_')
        (baseline,) = self.plot(baseline_x, baseline_y, color=basecolor, linestyle=basestyle, marker=basemarker, label='_nolegend_')
        stem_container = StemContainer((markerline, stemlines, baseline), label=label)
        self.add_container(stem_container)
        return stem_container

    @_preprocess_data(replace_names=['x', 'explode', 'labels', 'colors'])
    def pie(self, x, explode=None, labels=None, colors=None, autopct=None, pctdistance=0.6, shadow=False, labeldistance=1.1, startangle=0, radius=1, counterclock=True, wedgeprops=None, textprops=None, center=(0, 0), frame=False, rotatelabels=False, *, normalize=True, hatch=None):
        if False:
            print('Hello World!')
        '\n        Plot a pie chart.\n\n        Make a pie chart of array *x*.  The fractional area of each wedge is\n        given by ``x/sum(x)``.\n\n        The wedges are plotted counterclockwise, by default starting from the\n        x-axis.\n\n        Parameters\n        ----------\n        x : 1D array-like\n            The wedge sizes.\n\n        explode : array-like, default: None\n            If not *None*, is a ``len(x)`` array which specifies the fraction\n            of the radius with which to offset each wedge.\n\n        labels : list, default: None\n            A sequence of strings providing the labels for each wedge\n\n        colors : color or array-like of color, default: None\n            A sequence of colors through which the pie chart will cycle.  If\n            *None*, will use the colors in the currently active cycle.\n\n        hatch : str or list, default: None\n            Hatching pattern applied to all pie wedges or sequence of patterns\n            through which the chart will cycle. For a list of valid patterns,\n            see :doc:`/gallery/shapes_and_collections/hatch_style_reference`.\n\n            .. versionadded:: 3.7\n\n        autopct : None or str or callable, default: None\n            If not *None*, *autopct* is a string or function used to label the\n            wedges with their numeric value. The label will be placed inside\n            the wedge. If *autopct* is a format string, the label will be\n            ``fmt % pct``. If *autopct* is a function, then it will be called.\n\n        pctdistance : float, default: 0.6\n            The relative distance along the radius at which the text\n            generated by *autopct* is drawn. To draw the text outside the pie,\n            set *pctdistance* > 1. This parameter is ignored if *autopct* is\n            ``None``.\n\n        labeldistance : float or None, default: 1.1\n            The relative distance along the radius at which the labels are\n            drawn. To draw the labels inside the pie, set  *labeldistance* < 1.\n            If set to ``None``, labels are not drawn but are still stored for\n            use in `.legend`.\n\n        shadow : bool or dict, default: False\n            If bool, whether to draw a shadow beneath the pie. If dict, draw a shadow\n            passing the properties in the dict to `.Shadow`.\n\n            .. versionadded:: 3.8\n                *shadow* can be a dict.\n\n        startangle : float, default: 0 degrees\n            The angle by which the start of the pie is rotated,\n            counterclockwise from the x-axis.\n\n        radius : float, default: 1\n            The radius of the pie.\n\n        counterclock : bool, default: True\n            Specify fractions direction, clockwise or counterclockwise.\n\n        wedgeprops : dict, default: None\n            Dict of arguments passed to each `.patches.Wedge` of the pie.\n            For example, ``wedgeprops = {\'linewidth\': 3}`` sets the width of\n            the wedge border lines equal to 3. By default, ``clip_on=False``.\n            When there is a conflict between these properties and other\n            keywords, properties passed to *wedgeprops* take precedence.\n\n        textprops : dict, default: None\n            Dict of arguments to pass to the text objects.\n\n        center : (float, float), default: (0, 0)\n            The coordinates of the center of the chart.\n\n        frame : bool, default: False\n            Plot Axes frame with the chart if true.\n\n        rotatelabels : bool, default: False\n            Rotate each label to the angle of the corresponding slice if true.\n\n        normalize : bool, default: True\n            When *True*, always make a full pie by normalizing x so that\n            ``sum(x) == 1``. *False* makes a partial pie if ``sum(x) <= 1``\n            and raises a `ValueError` for ``sum(x) > 1``.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        Returns\n        -------\n        patches : list\n            A sequence of `matplotlib.patches.Wedge` instances\n\n        texts : list\n            A list of the label `.Text` instances.\n\n        autotexts : list\n            A list of `.Text` instances for the numeric labels. This will only\n            be returned if the parameter *autopct* is not *None*.\n\n        Notes\n        -----\n        The pie chart will probably look best if the figure and Axes are\n        square, or the Axes aspect is equal.\n        This method sets the aspect ratio of the axis to "equal".\n        The Axes aspect ratio can be controlled with `.Axes.set_aspect`.\n        '
        self.set_aspect('equal')
        x = np.asarray(x, np.float32)
        if x.ndim > 1:
            raise ValueError('x must be 1D')
        if np.any(x < 0):
            raise ValueError("Wedge sizes 'x' must be non negative values")
        sx = x.sum()
        if normalize:
            x = x / sx
        elif sx > 1:
            raise ValueError('Cannot plot an unnormalized pie with sum(x) > 1')
        if labels is None:
            labels = [''] * len(x)
        if explode is None:
            explode = [0] * len(x)
        if len(x) != len(labels):
            raise ValueError("'label' must be of length 'x'")
        if len(x) != len(explode):
            raise ValueError("'explode' must be of length 'x'")
        if colors is None:
            get_next_color = self._get_patches_for_fill.get_next_color
        else:
            color_cycle = itertools.cycle(colors)

            def get_next_color():
                if False:
                    for i in range(10):
                        print('nop')
                return next(color_cycle)
        hatch_cycle = itertools.cycle(np.atleast_1d(hatch))
        _api.check_isinstance(Real, radius=radius, startangle=startangle)
        if radius <= 0:
            raise ValueError(f'radius must be a positive number, not {radius}')
        theta1 = startangle / 360
        if wedgeprops is None:
            wedgeprops = {}
        if textprops is None:
            textprops = {}
        texts = []
        slices = []
        autotexts = []
        for (frac, label, expl) in zip(x, labels, explode):
            (x, y) = center
            theta2 = theta1 + frac if counterclock else theta1 - frac
            thetam = 2 * np.pi * 0.5 * (theta1 + theta2)
            x += expl * math.cos(thetam)
            y += expl * math.sin(thetam)
            w = mpatches.Wedge((x, y), radius, 360.0 * min(theta1, theta2), 360.0 * max(theta1, theta2), facecolor=get_next_color(), hatch=next(hatch_cycle), clip_on=False, label=label)
            w.set(**wedgeprops)
            slices.append(w)
            self.add_patch(w)
            if shadow:
                shadow_dict = {'ox': -0.02, 'oy': -0.02, 'label': '_nolegend_'}
                if isinstance(shadow, dict):
                    shadow_dict.update(shadow)
                self.add_patch(mpatches.Shadow(w, **shadow_dict))
            if labeldistance is not None:
                xt = x + labeldistance * radius * math.cos(thetam)
                yt = y + labeldistance * radius * math.sin(thetam)
                label_alignment_h = 'left' if xt > 0 else 'right'
                label_alignment_v = 'center'
                label_rotation = 'horizontal'
                if rotatelabels:
                    label_alignment_v = 'bottom' if yt > 0 else 'top'
                    label_rotation = np.rad2deg(thetam) + (0 if xt > 0 else 180)
                t = self.text(xt, yt, label, clip_on=False, horizontalalignment=label_alignment_h, verticalalignment=label_alignment_v, rotation=label_rotation, size=mpl.rcParams['xtick.labelsize'])
                t.set(**textprops)
                texts.append(t)
            if autopct is not None:
                xt = x + pctdistance * radius * math.cos(thetam)
                yt = y + pctdistance * radius * math.sin(thetam)
                if isinstance(autopct, str):
                    s = autopct % (100.0 * frac)
                elif callable(autopct):
                    s = autopct(100.0 * frac)
                else:
                    raise TypeError('autopct must be callable or a format string')
                t = self.text(xt, yt, s, clip_on=False, horizontalalignment='center', verticalalignment='center')
                t.set(**textprops)
                autotexts.append(t)
            theta1 = theta2
        if frame:
            self._request_autoscale_view()
        else:
            self.set(frame_on=False, xticks=[], yticks=[], xlim=(-1.25 + center[0], 1.25 + center[0]), ylim=(-1.25 + center[1], 1.25 + center[1]))
        if autopct is None:
            return (slices, texts)
        else:
            return (slices, texts, autotexts)

    @staticmethod
    def _errorevery_to_mask(x, errorevery):
        if False:
            return 10
        "\n        Normalize `errorbar`'s *errorevery* to be a boolean mask for data *x*.\n\n        This function is split out to be usable both by 2D and 3D errorbars.\n        "
        if isinstance(errorevery, Integral):
            errorevery = (0, errorevery)
        if isinstance(errorevery, tuple):
            if len(errorevery) == 2 and isinstance(errorevery[0], Integral) and isinstance(errorevery[1], Integral):
                errorevery = slice(errorevery[0], None, errorevery[1])
            else:
                raise ValueError(f'errorevery={errorevery!r} is a not a tuple of two integers')
        elif isinstance(errorevery, slice):
            pass
        elif not isinstance(errorevery, str) and np.iterable(errorevery):
            try:
                x[errorevery]
            except (ValueError, IndexError) as err:
                raise ValueError(f"errorevery={errorevery!r} is iterable but not a valid NumPy fancy index to match 'xerr'/'yerr'") from err
        else:
            raise ValueError(f'errorevery={errorevery!r} is not a recognized value')
        everymask = np.zeros(len(x), bool)
        everymask[errorevery] = True
        return everymask

    @_preprocess_data(replace_names=['x', 'y', 'xerr', 'yerr'], label_namer='y')
    @_docstring.dedent_interpd
    def errorbar(self, x, y, yerr=None, xerr=None, fmt='', ecolor=None, elinewidth=None, capsize=None, barsabove=False, lolims=False, uplims=False, xlolims=False, xuplims=False, errorevery=1, capthick=None, **kwargs):
        if False:
            return 10
        "\n        Plot y versus x as lines and/or markers with attached errorbars.\n\n        *x*, *y* define the data locations, *xerr*, *yerr* define the errorbar\n        sizes. By default, this draws the data markers/lines as well as the\n        errorbars. Use fmt='none' to draw errorbars without any data markers.\n\n        .. versionadded:: 3.7\n           Caps and error lines are drawn in polar coordinates on polar plots.\n\n\n        Parameters\n        ----------\n        x, y : float or array-like\n            The data positions.\n\n        xerr, yerr : float or array-like, shape(N,) or shape(2, N), optional\n            The errorbar sizes:\n\n            - scalar: Symmetric +/- values for all data points.\n            - shape(N,): Symmetric +/-values for each data point.\n            - shape(2, N): Separate - and + values for each bar. First row\n              contains the lower errors, the second row contains the upper\n              errors.\n            - *None*: No errorbar.\n\n            All values must be >= 0.\n\n            See :doc:`/gallery/statistics/errorbar_features`\n            for an example on the usage of ``xerr`` and ``yerr``.\n\n        fmt : str, default: ''\n            The format for the data points / data lines. See `.plot` for\n            details.\n\n            Use 'none' (case-insensitive) to plot errorbars without any data\n            markers.\n\n        ecolor : color, default: None\n            The color of the errorbar lines.  If None, use the color of the\n            line connecting the markers.\n\n        elinewidth : float, default: None\n            The linewidth of the errorbar lines. If None, the linewidth of\n            the current style is used.\n\n        capsize : float, default: :rc:`errorbar.capsize`\n            The length of the error bar caps in points.\n\n        capthick : float, default: None\n            An alias to the keyword argument *markeredgewidth* (a.k.a. *mew*).\n            This setting is a more sensible name for the property that\n            controls the thickness of the error bar cap in points. For\n            backwards compatibility, if *mew* or *markeredgewidth* are given,\n            then they will over-ride *capthick*. This may change in future\n            releases.\n\n        barsabove : bool, default: False\n            If True, will plot the errorbars above the plot\n            symbols. Default is below.\n\n        lolims, uplims, xlolims, xuplims : bool or array-like, default: False\n            These arguments can be used to indicate that a value gives only\n            upper/lower limits.  In that case a caret symbol is used to\n            indicate this. *lims*-arguments may be scalars, or array-likes of\n            the same length as *xerr* and *yerr*.  To use limits with inverted\n            axes, `~.Axes.set_xlim` or `~.Axes.set_ylim` must be called before\n            :meth:`errorbar`.  Note the tricky parameter names: setting e.g.\n            *lolims* to True means that the y-value is a *lower* limit of the\n            True value, so, only an *upward*-pointing arrow will be drawn!\n\n        errorevery : int or (int, int), default: 1\n            draws error bars on a subset of the data. *errorevery* =N draws\n            error bars on the points (x[::N], y[::N]).\n            *errorevery* =(start, N) draws error bars on the points\n            (x[start::N], y[start::N]). e.g. errorevery=(6, 3)\n            adds error bars to the data at (x[6], x[9], x[12], x[15], ...).\n            Used to avoid overlapping error bars when two series share x-axis\n            values.\n\n        Returns\n        -------\n        `.ErrorbarContainer`\n            The container contains:\n\n            - plotline: `~matplotlib.lines.Line2D` instance of x, y plot markers\n              and/or line.\n            - caplines: A tuple of `~matplotlib.lines.Line2D` instances of the error\n              bar caps.\n            - barlinecols: A tuple of `.LineCollection` with the horizontal and\n              vertical error ranges.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            All other keyword arguments are passed on to the `~.Axes.plot` call\n            drawing the markers. For example, this code makes big red squares\n            with thick green edges::\n\n                x, y, yerr = rand(3, 10)\n                errorbar(x, y, yerr, marker='s', mfc='red',\n                         mec='green', ms=20, mew=4)\n\n            where *mfc*, *mec*, *ms* and *mew* are aliases for the longer\n            property names, *markerfacecolor*, *markeredgecolor*, *markersize*\n            and *markeredgewidth*.\n\n            Valid kwargs for the marker properties are:\n\n            - *dashes*\n            - *dash_capstyle*\n            - *dash_joinstyle*\n            - *drawstyle*\n            - *fillstyle*\n            - *linestyle*\n            - *marker*\n            - *markeredgecolor*\n            - *markeredgewidth*\n            - *markerfacecolor*\n            - *markerfacecoloralt*\n            - *markersize*\n            - *markevery*\n            - *solid_capstyle*\n            - *solid_joinstyle*\n\n            Refer to the corresponding `.Line2D` property for more details:\n\n            %(Line2D:kwdoc)s\n        "
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
        kwargs.setdefault('zorder', 2)
        if not isinstance(x, np.ndarray):
            x = np.asarray(x, dtype=object)
        if not isinstance(y, np.ndarray):
            y = np.asarray(y, dtype=object)

        def _upcast_err(err):
            if False:
                i = 10
                return i + 15
            '\n            Safely handle tuple of containers that carry units.\n\n            This function covers the case where the input to the xerr/yerr is a\n            length 2 tuple of equal length ndarray-subclasses that carry the\n            unit information in the container.\n\n            If we have a tuple of nested numpy array (subclasses), we defer\n            coercing the units to be consistent to the underlying unit\n            library (and implicitly the broadcasting).\n\n            Otherwise, fallback to casting to an object array.\n            '
            if np.iterable(err) and len(err) > 0 and isinstance(cbook._safe_first_finite(err), np.ndarray):
                atype = type(cbook._safe_first_finite(err))
                if atype is np.ndarray:
                    return np.asarray(err, dtype=object)
                return atype(err)
            return np.asarray(err, dtype=object)
        if xerr is not None and (not isinstance(xerr, np.ndarray)):
            xerr = _upcast_err(xerr)
        if yerr is not None and (not isinstance(yerr, np.ndarray)):
            yerr = _upcast_err(yerr)
        (x, y) = np.atleast_1d(x, y)
        if len(x) != len(y):
            raise ValueError("'x' and 'y' must have the same size")
        everymask = self._errorevery_to_mask(x, errorevery)
        label = kwargs.pop('label', None)
        kwargs['label'] = '_nolegend_'
        ((data_line, base_style),) = self._get_lines._plot_args(self, (x, y) if fmt == '' else (x, y, fmt), kwargs, return_kwargs=True)
        if barsabove:
            data_line.set_zorder(kwargs['zorder'] - 0.1)
        else:
            data_line.set_zorder(kwargs['zorder'] + 0.1)
        if fmt.lower() != 'none':
            self.add_line(data_line)
        else:
            data_line = None
            base_style.pop('color')
            if 'color' in kwargs:
                base_style['color'] = kwargs.pop('color')
        if 'color' not in base_style:
            base_style['color'] = 'C0'
        if ecolor is None:
            ecolor = base_style['color']
        for key in ['marker', 'markersize', 'markerfacecolor', 'markerfacecoloralt', 'markeredgewidth', 'markeredgecolor', 'markevery', 'linestyle', 'fillstyle', 'drawstyle', 'dash_capstyle', 'dash_joinstyle', 'solid_capstyle', 'solid_joinstyle', 'dashes']:
            base_style.pop(key, None)
        eb_lines_style = {**base_style, 'color': ecolor}
        if elinewidth is not None:
            eb_lines_style['linewidth'] = elinewidth
        elif 'linewidth' in kwargs:
            eb_lines_style['linewidth'] = kwargs['linewidth']
        for key in ('transform', 'alpha', 'zorder', 'rasterized'):
            if key in kwargs:
                eb_lines_style[key] = kwargs[key]
        eb_cap_style = {**base_style, 'linestyle': 'none'}
        if capsize is None:
            capsize = mpl.rcParams['errorbar.capsize']
        if capsize > 0:
            eb_cap_style['markersize'] = 2.0 * capsize
        if capthick is not None:
            eb_cap_style['markeredgewidth'] = capthick
        for key in ('markeredgewidth', 'transform', 'alpha', 'zorder', 'rasterized'):
            if key in kwargs:
                eb_cap_style[key] = kwargs[key]
        eb_cap_style['color'] = ecolor
        barcols = []
        caplines = {'x': [], 'y': []}

        def apply_mask(arrays, mask):
            if False:
                print('Hello World!')
            return [array[mask] for array in arrays]
        for (dep_axis, dep, err, lolims, uplims, indep, lines_func, marker, lomarker, himarker) in [('x', x, xerr, xlolims, xuplims, y, self.hlines, '|', mlines.CARETRIGHTBASE, mlines.CARETLEFTBASE), ('y', y, yerr, lolims, uplims, x, self.vlines, '_', mlines.CARETUPBASE, mlines.CARETDOWNBASE)]:
            if err is None:
                continue
            lolims = np.broadcast_to(lolims, len(dep)).astype(bool)
            uplims = np.broadcast_to(uplims, len(dep)).astype(bool)
            try:
                np.broadcast_to(err, (2, len(dep)))
            except ValueError:
                raise ValueError(f"'{dep_axis}err' (shape: {np.shape(err)}) must be a scalar or a 1D or (2, n) array-like whose shape matches '{dep_axis}' (shape: {np.shape(dep)})") from None
            res = np.zeros(err.shape, dtype=bool)
            if np.any(np.less(err, -err, out=res, where=err == err)):
                raise ValueError(f"'{dep_axis}err' must not contain negative values")
            (low, high) = dep + np.vstack([-(1 - lolims), 1 - uplims]) * err
            barcols.append(lines_func(*apply_mask([indep, low, high], everymask), **eb_lines_style))
            if self.name == 'polar' and dep_axis == 'x':
                for b in barcols:
                    for p in b.get_paths():
                        p._interpolation_steps = 2
            nolims = ~(lolims | uplims)
            if nolims.any() and capsize > 0:
                (indep_masked, lo_masked, hi_masked) = apply_mask([indep, low, high], nolims & everymask)
                for lh_masked in [lo_masked, hi_masked]:
                    line = mlines.Line2D(indep_masked, indep_masked, marker=marker, **eb_cap_style)
                    line.set(**{f'{dep_axis}data': lh_masked})
                    caplines[dep_axis].append(line)
            for (idx, (lims, hl)) in enumerate([(lolims, high), (uplims, low)]):
                if not lims.any():
                    continue
                hlmarker = himarker if self._axis_map[dep_axis].get_inverted() ^ idx else lomarker
                (x_masked, y_masked, hl_masked) = apply_mask([x, y, hl], lims & everymask)
                line = mlines.Line2D(x_masked, y_masked, marker=hlmarker, **eb_cap_style)
                line.set(**{f'{dep_axis}data': hl_masked})
                caplines[dep_axis].append(line)
                if capsize > 0:
                    caplines[dep_axis].append(mlines.Line2D(x_masked, y_masked, marker=marker, **eb_cap_style))
        if self.name == 'polar':
            for axis in caplines:
                for l in caplines[axis]:
                    for (theta, r) in zip(l.get_xdata(), l.get_ydata()):
                        rotation = mtransforms.Affine2D().rotate(theta)
                        if axis == 'y':
                            rotation.rotate(-np.pi / 2)
                        ms = mmarkers.MarkerStyle(marker=marker, transform=rotation)
                        self.add_line(mlines.Line2D([theta], [r], marker=ms, **eb_cap_style))
        else:
            for axis in caplines:
                for l in caplines[axis]:
                    self.add_line(l)
        self._request_autoscale_view()
        caplines = caplines['x'] + caplines['y']
        errorbar_container = ErrorbarContainer((data_line, tuple(caplines), tuple(barcols)), has_xerr=xerr is not None, has_yerr=yerr is not None, label=label)
        self.containers.append(errorbar_container)
        return errorbar_container

    @_preprocess_data()
    def boxplot(self, x, notch=None, sym=None, vert=None, whis=None, positions=None, widths=None, patch_artist=None, bootstrap=None, usermedians=None, conf_intervals=None, meanline=None, showmeans=None, showcaps=None, showbox=None, showfliers=None, boxprops=None, labels=None, flierprops=None, medianprops=None, meanprops=None, capprops=None, whiskerprops=None, manage_ticks=True, autorange=False, zorder=None, capwidths=None):
        if False:
            print('Hello World!')
        '\n        Draw a box and whisker plot.\n\n        The box extends from the first quartile (Q1) to the third\n        quartile (Q3) of the data, with a line at the median.\n        The whiskers extend from the box to the farthest data point\n        lying within 1.5x the inter-quartile range (IQR) from the box.\n        Flier points are those past the end of the whiskers.\n        See https://en.wikipedia.org/wiki/Box_plot for reference.\n\n        .. code-block:: none\n\n                  Q1-1.5IQR   Q1   median  Q3   Q3+1.5IQR\n                               |-----:-----|\n               o      |--------|     :     |--------|    o  o\n                               |-----:-----|\n             flier             <----------->            fliers\n                                    IQR\n\n\n        Parameters\n        ----------\n        x : Array or a sequence of vectors.\n            The input data.  If a 2D array, a boxplot is drawn for each column\n            in *x*.  If a sequence of 1D arrays, a boxplot is drawn for each\n            array in *x*.\n\n        notch : bool, default: False\n            Whether to draw a notched boxplot (`True`), or a rectangular\n            boxplot (`False`).  The notches represent the confidence interval\n            (CI) around the median.  The documentation for *bootstrap*\n            describes how the locations of the notches are computed by\n            default, but their locations may also be overridden by setting the\n            *conf_intervals* parameter.\n\n            .. note::\n\n                In cases where the values of the CI are less than the\n                lower quartile or greater than the upper quartile, the\n                notches will extend beyond the box, giving it a\n                distinctive "flipped" appearance. This is expected\n                behavior and consistent with other statistical\n                visualization packages.\n\n        sym : str, optional\n            The default symbol for flier points.  An empty string (\'\') hides\n            the fliers.  If `None`, then the fliers default to \'b+\'.  More\n            control is provided by the *flierprops* parameter.\n\n        vert : bool, default: True\n            If `True`, draws vertical boxes.\n            If `False`, draw horizontal boxes.\n\n        whis : float or (float, float), default: 1.5\n            The position of the whiskers.\n\n            If a float, the lower whisker is at the lowest datum above\n            ``Q1 - whis*(Q3-Q1)``, and the upper whisker at the highest datum\n            below ``Q3 + whis*(Q3-Q1)``, where Q1 and Q3 are the first and\n            third quartiles.  The default value of ``whis = 1.5`` corresponds\n            to Tukey\'s original definition of boxplots.\n\n            If a pair of floats, they indicate the percentiles at which to\n            draw the whiskers (e.g., (5, 95)).  In particular, setting this to\n            (0, 100) results in whiskers covering the whole range of the data.\n\n            In the edge case where ``Q1 == Q3``, *whis* is automatically set\n            to (0, 100) (cover the whole range of the data) if *autorange* is\n            True.\n\n            Beyond the whiskers, data are considered outliers and are plotted\n            as individual points.\n\n        bootstrap : int, optional\n            Specifies whether to bootstrap the confidence intervals\n            around the median for notched boxplots. If *bootstrap* is\n            None, no bootstrapping is performed, and notches are\n            calculated using a Gaussian-based asymptotic approximation\n            (see McGill, R., Tukey, J.W., and Larsen, W.A., 1978, and\n            Kendall and Stuart, 1967). Otherwise, bootstrap specifies\n            the number of times to bootstrap the median to determine its\n            95% confidence intervals. Values between 1000 and 10000 are\n            recommended.\n\n        usermedians : 1D array-like, optional\n            A 1D array-like of length ``len(x)``.  Each entry that is not\n            `None` forces the value of the median for the corresponding\n            dataset.  For entries that are `None`, the medians are computed\n            by Matplotlib as normal.\n\n        conf_intervals : array-like, optional\n            A 2D array-like of shape ``(len(x), 2)``.  Each entry that is not\n            None forces the location of the corresponding notch (which is\n            only drawn if *notch* is `True`).  For entries that are `None`,\n            the notches are computed by the method specified by the other\n            parameters (e.g., *bootstrap*).\n\n        positions : array-like, optional\n            The positions of the boxes. The ticks and limits are\n            automatically set to match the positions. Defaults to\n            ``range(1, N+1)`` where N is the number of boxes to be drawn.\n\n        widths : float or array-like\n            The widths of the boxes.  The default is 0.5, or ``0.15*(distance\n            between extreme positions)``, if that is smaller.\n\n        patch_artist : bool, default: False\n            If `False` produces boxes with the Line2D artist. Otherwise,\n            boxes are drawn with Patch artists.\n\n        labels : sequence, optional\n            Labels for each dataset (one per dataset).\n\n        manage_ticks : bool, default: True\n            If True, the tick locations and labels will be adjusted to match\n            the boxplot positions.\n\n        autorange : bool, default: False\n            When `True` and the data are distributed such that the 25th and\n            75th percentiles are equal, *whis* is set to (0, 100) such\n            that the whisker ends are at the minimum and maximum of the data.\n\n        meanline : bool, default: False\n            If `True` (and *showmeans* is `True`), will try to render the\n            mean as a line spanning the full width of the box according to\n            *meanprops* (see below).  Not recommended if *shownotches* is also\n            True.  Otherwise, means will be shown as points.\n\n        zorder : float, default: ``Line2D.zorder = 2``\n            The zorder of the boxplot.\n\n        Returns\n        -------\n        dict\n          A dictionary mapping each component of the boxplot to a list\n          of the `.Line2D` instances created. That dictionary has the\n          following keys (assuming vertical boxplots):\n\n          - ``boxes``: the main body of the boxplot showing the\n            quartiles and the median\'s confidence intervals if\n            enabled.\n\n          - ``medians``: horizontal lines at the median of each box.\n\n          - ``whiskers``: the vertical lines extending to the most\n            extreme, non-outlier data points.\n\n          - ``caps``: the horizontal lines at the ends of the\n            whiskers.\n\n          - ``fliers``: points representing data that extend beyond\n            the whiskers (fliers).\n\n          - ``means``: points or lines representing the means.\n\n        Other Parameters\n        ----------------\n        showcaps : bool, default: True\n            Show the caps on the ends of whiskers.\n        showbox : bool, default: True\n            Show the central box.\n        showfliers : bool, default: True\n            Show the outliers beyond the caps.\n        showmeans : bool, default: False\n            Show the arithmetic means.\n        capprops : dict, default: None\n            The style of the caps.\n        capwidths : float or array, default: None\n            The widths of the caps.\n        boxprops : dict, default: None\n            The style of the box.\n        whiskerprops : dict, default: None\n            The style of the whiskers.\n        flierprops : dict, default: None\n            The style of the fliers.\n        medianprops : dict, default: None\n            The style of the median.\n        meanprops : dict, default: None\n            The style of the mean.\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        See Also\n        --------\n        violinplot : Draw an estimate of the probability density function.\n        '
        if whis is None:
            whis = mpl.rcParams['boxplot.whiskers']
        if bootstrap is None:
            bootstrap = mpl.rcParams['boxplot.bootstrap']
        bxpstats = cbook.boxplot_stats(x, whis=whis, bootstrap=bootstrap, labels=labels, autorange=autorange)
        if notch is None:
            notch = mpl.rcParams['boxplot.notch']
        if vert is None:
            vert = mpl.rcParams['boxplot.vertical']
        if patch_artist is None:
            patch_artist = mpl.rcParams['boxplot.patchartist']
        if meanline is None:
            meanline = mpl.rcParams['boxplot.meanline']
        if showmeans is None:
            showmeans = mpl.rcParams['boxplot.showmeans']
        if showcaps is None:
            showcaps = mpl.rcParams['boxplot.showcaps']
        if showbox is None:
            showbox = mpl.rcParams['boxplot.showbox']
        if showfliers is None:
            showfliers = mpl.rcParams['boxplot.showfliers']
        if boxprops is None:
            boxprops = {}
        if whiskerprops is None:
            whiskerprops = {}
        if capprops is None:
            capprops = {}
        if medianprops is None:
            medianprops = {}
        if meanprops is None:
            meanprops = {}
        if flierprops is None:
            flierprops = {}
        if patch_artist:
            boxprops['linestyle'] = 'solid'
            if 'color' in boxprops:
                boxprops['edgecolor'] = boxprops.pop('color')
        if sym is not None:
            if sym == '':
                flierprops = dict(linestyle='none', marker='', color='none')
                showfliers = False
            else:
                (_, marker, color) = _process_plot_format(sym)
                if marker is not None:
                    flierprops['marker'] = marker
                if color is not None:
                    flierprops['color'] = color
                    flierprops['markerfacecolor'] = color
                    flierprops['markeredgecolor'] = color
        if usermedians is not None:
            if len(np.ravel(usermedians)) != len(bxpstats) or np.shape(usermedians)[0] != len(bxpstats):
                raise ValueError("'usermedians' and 'x' have different lengths")
            else:
                for (stats, med) in zip(bxpstats, usermedians):
                    if med is not None:
                        stats['med'] = med
        if conf_intervals is not None:
            if len(conf_intervals) != len(bxpstats):
                raise ValueError("'conf_intervals' and 'x' have different lengths")
            else:
                for (stats, ci) in zip(bxpstats, conf_intervals):
                    if ci is not None:
                        if len(ci) != 2:
                            raise ValueError('each confidence interval must have two values')
                        else:
                            if ci[0] is not None:
                                stats['cilo'] = ci[0]
                            if ci[1] is not None:
                                stats['cihi'] = ci[1]
        artists = self.bxp(bxpstats, positions=positions, widths=widths, vert=vert, patch_artist=patch_artist, shownotches=notch, showmeans=showmeans, showcaps=showcaps, showbox=showbox, boxprops=boxprops, flierprops=flierprops, medianprops=medianprops, meanprops=meanprops, meanline=meanline, showfliers=showfliers, capprops=capprops, whiskerprops=whiskerprops, manage_ticks=manage_ticks, zorder=zorder, capwidths=capwidths)
        return artists

    def bxp(self, bxpstats, positions=None, widths=None, vert=True, patch_artist=False, shownotches=False, showmeans=False, showcaps=True, showbox=True, showfliers=True, boxprops=None, whiskerprops=None, flierprops=None, medianprops=None, capprops=None, meanprops=None, meanline=False, manage_ticks=True, zorder=None, capwidths=None):
        if False:
            while True:
                i = 10
        "\n        Drawing function for box and whisker plots.\n\n        Make a box and whisker plot for each column of *x* or each\n        vector in sequence *x*.  The box extends from the lower to\n        upper quartile values of the data, with a line at the median.\n        The whiskers extend from the box to show the range of the\n        data.  Flier points are those past the end of the whiskers.\n\n        Parameters\n        ----------\n        bxpstats : list of dicts\n          A list of dictionaries containing stats for each boxplot.\n          Required keys are:\n\n          - ``med``: Median (scalar).\n          - ``q1``, ``q3``: First & third quartiles (scalars).\n          - ``whislo``, ``whishi``: Lower & upper whisker positions (scalars).\n\n          Optional keys are:\n\n          - ``mean``: Mean (scalar).  Needed if ``showmeans=True``.\n          - ``fliers``: Data beyond the whiskers (array-like).\n            Needed if ``showfliers=True``.\n          - ``cilo``, ``cihi``: Lower & upper confidence intervals\n            about the median. Needed if ``shownotches=True``.\n          - ``label``: Name of the dataset (str).  If available,\n            this will be used a tick label for the boxplot\n\n        positions : array-like, default: [1, 2, ..., n]\n          The positions of the boxes. The ticks and limits\n          are automatically set to match the positions.\n\n        widths : float or array-like, default: None\n          The widths of the boxes.  The default is\n          ``clip(0.15*(distance between extreme positions), 0.15, 0.5)``.\n\n        capwidths : float or array-like, default: None\n          Either a scalar or a vector and sets the width of each cap.\n          The default is ``0.5*(width of the box)``, see *widths*.\n\n        vert : bool, default: True\n          If `True` (default), makes the boxes vertical.\n          If `False`, makes horizontal boxes.\n\n        patch_artist : bool, default: False\n          If `False` produces boxes with the `.Line2D` artist.\n          If `True` produces boxes with the `~matplotlib.patches.Patch` artist.\n\n        shownotches, showmeans, showcaps, showbox, showfliers : bool\n          Whether to draw the CI notches, the mean value (both default to\n          False), the caps, the box, and the fliers (all three default to\n          True).\n\n        boxprops, whiskerprops, capprops, flierprops, medianprops, meanprops : dict, optional\n          Artist properties for the boxes, whiskers, caps, fliers, medians, and\n          means.\n\n        meanline : bool, default: False\n          If `True` (and *showmeans* is `True`), will try to render the mean\n          as a line spanning the full width of the box according to\n          *meanprops*. Not recommended if *shownotches* is also True.\n          Otherwise, means will be shown as points.\n\n        manage_ticks : bool, default: True\n          If True, the tick locations and labels will be adjusted to match the\n          boxplot positions.\n\n        zorder : float, default: ``Line2D.zorder = 2``\n          The zorder of the resulting boxplot.\n\n        Returns\n        -------\n        dict\n          A dictionary mapping each component of the boxplot to a list\n          of the `.Line2D` instances created. That dictionary has the\n          following keys (assuming vertical boxplots):\n\n          - ``boxes``: main bodies of the boxplot showing the quartiles, and\n            the median's confidence intervals if enabled.\n          - ``medians``: horizontal lines at the median of each box.\n          - ``whiskers``: vertical lines up to the last non-outlier data.\n          - ``caps``: horizontal lines at the ends of the whiskers.\n          - ``fliers``: points representing data beyond the whiskers (fliers).\n          - ``means``: points or lines representing the means.\n\n        Examples\n        --------\n        .. plot:: gallery/statistics/bxp.py\n        "
        medianprops = {'solid_capstyle': 'butt', 'dash_capstyle': 'butt', **(medianprops or {})}
        meanprops = {'solid_capstyle': 'butt', 'dash_capstyle': 'butt', **(meanprops or {})}
        whiskers = []
        caps = []
        boxes = []
        medians = []
        means = []
        fliers = []
        datalabels = []
        if zorder is None:
            zorder = mlines.Line2D.zorder
        zdelta = 0.1

        def merge_kw_rc(subkey, explicit, zdelta=0, usemarker=True):
            if False:
                while True:
                    i = 10
            d = {k.split('.')[-1]: v for (k, v) in mpl.rcParams.items() if k.startswith(f'boxplot.{subkey}props')}
            d['zorder'] = zorder + zdelta
            if not usemarker:
                d['marker'] = ''
            d.update(cbook.normalize_kwargs(explicit, mlines.Line2D))
            return d
        box_kw = {'linestyle': mpl.rcParams['boxplot.boxprops.linestyle'], 'linewidth': mpl.rcParams['boxplot.boxprops.linewidth'], 'edgecolor': mpl.rcParams['boxplot.boxprops.color'], 'facecolor': 'white' if mpl.rcParams['_internal.classic_mode'] else mpl.rcParams['patch.facecolor'], 'zorder': zorder, **cbook.normalize_kwargs(boxprops, mpatches.PathPatch)} if patch_artist else merge_kw_rc('box', boxprops, usemarker=False)
        whisker_kw = merge_kw_rc('whisker', whiskerprops, usemarker=False)
        cap_kw = merge_kw_rc('cap', capprops, usemarker=False)
        flier_kw = merge_kw_rc('flier', flierprops)
        median_kw = merge_kw_rc('median', medianprops, zdelta, usemarker=False)
        mean_kw = merge_kw_rc('mean', meanprops, zdelta)
        removed_prop = 'marker' if meanline else 'linestyle'
        if meanprops is None or removed_prop not in meanprops:
            mean_kw[removed_prop] = ''
        maybe_swap = slice(None) if vert else slice(None, None, -1)

        def do_plot(xs, ys, **kwargs):
            if False:
                return 10
            return self.plot(*[xs, ys][maybe_swap], **kwargs)[0]

        def do_patch(xs, ys, **kwargs):
            if False:
                while True:
                    i = 10
            path = mpath.Path._create_closed(np.column_stack([xs, ys][maybe_swap]))
            patch = mpatches.PathPatch(path, **kwargs)
            self.add_artist(patch)
            return patch
        N = len(bxpstats)
        datashape_message = 'List of boxplot statistics and `{0}` values must have same the length'
        if positions is None:
            positions = list(range(1, N + 1))
        elif len(positions) != N:
            raise ValueError(datashape_message.format('positions'))
        positions = np.array(positions)
        if len(positions) > 0 and (not all((isinstance(p, Real) for p in positions))):
            raise TypeError('positions should be an iterable of numbers')
        if widths is None:
            widths = [np.clip(0.15 * np.ptp(positions), 0.15, 0.5)] * N
        elif np.isscalar(widths):
            widths = [widths] * N
        elif len(widths) != N:
            raise ValueError(datashape_message.format('widths'))
        if capwidths is None:
            capwidths = 0.5 * np.array(widths)
        elif np.isscalar(capwidths):
            capwidths = [capwidths] * N
        elif len(capwidths) != N:
            raise ValueError(datashape_message.format('capwidths'))
        for (pos, width, stats, capwidth) in zip(positions, widths, bxpstats, capwidths):
            datalabels.append(stats.get('label', pos))
            whis_x = [pos, pos]
            whislo_y = [stats['q1'], stats['whislo']]
            whishi_y = [stats['q3'], stats['whishi']]
            cap_left = pos - capwidth * 0.5
            cap_right = pos + capwidth * 0.5
            cap_x = [cap_left, cap_right]
            cap_lo = np.full(2, stats['whislo'])
            cap_hi = np.full(2, stats['whishi'])
            box_left = pos - width * 0.5
            box_right = pos + width * 0.5
            med_y = [stats['med'], stats['med']]
            if shownotches:
                notch_left = pos - width * 0.25
                notch_right = pos + width * 0.25
                box_x = [box_left, box_right, box_right, notch_right, box_right, box_right, box_left, box_left, notch_left, box_left, box_left]
                box_y = [stats['q1'], stats['q1'], stats['cilo'], stats['med'], stats['cihi'], stats['q3'], stats['q3'], stats['cihi'], stats['med'], stats['cilo'], stats['q1']]
                med_x = [notch_left, notch_right]
            else:
                box_x = [box_left, box_right, box_right, box_left, box_left]
                box_y = [stats['q1'], stats['q1'], stats['q3'], stats['q3'], stats['q1']]
                med_x = [box_left, box_right]
            if showbox:
                do_box = do_patch if patch_artist else do_plot
                boxes.append(do_box(box_x, box_y, **box_kw))
            whiskers.append(do_plot(whis_x, whislo_y, **whisker_kw))
            whiskers.append(do_plot(whis_x, whishi_y, **whisker_kw))
            if showcaps:
                caps.append(do_plot(cap_x, cap_lo, **cap_kw))
                caps.append(do_plot(cap_x, cap_hi, **cap_kw))
            medians.append(do_plot(med_x, med_y, **median_kw))
            if showmeans:
                if meanline:
                    means.append(do_plot([box_left, box_right], [stats['mean'], stats['mean']], **mean_kw))
                else:
                    means.append(do_plot([pos], [stats['mean']], **mean_kw))
            if showfliers:
                flier_x = np.full(len(stats['fliers']), pos, dtype=np.float64)
                flier_y = stats['fliers']
                fliers.append(do_plot(flier_x, flier_y, **flier_kw))
        if manage_ticks:
            axis_name = 'x' if vert else 'y'
            interval = getattr(self.dataLim, f'interval{axis_name}')
            axis = self._axis_map[axis_name]
            positions = axis.convert_units(positions)
            interval[:] = (min(interval[0], min(positions) - 0.5), max(interval[1], max(positions) + 0.5))
            for (median, position) in zip(medians, positions):
                getattr(median.sticky_edges, axis_name).extend([position - 0.5, position + 0.5])
            locator = axis.get_major_locator()
            if not isinstance(axis.get_major_locator(), mticker.FixedLocator):
                locator = mticker.FixedLocator([])
                axis.set_major_locator(locator)
            locator.locs = np.array([*locator.locs, *positions])
            formatter = axis.get_major_formatter()
            if not isinstance(axis.get_major_formatter(), mticker.FixedFormatter):
                formatter = mticker.FixedFormatter([])
                axis.set_major_formatter(formatter)
            formatter.seq = [*formatter.seq, *datalabels]
            self._request_autoscale_view()
        return dict(whiskers=whiskers, caps=caps, boxes=boxes, medians=medians, fliers=fliers, means=means)

    @staticmethod
    def _parse_scatter_color_args(c, edgecolors, kwargs, xsize, get_next_color_func):
        if False:
            while True:
                i = 10
        "\n        Helper function to process color related arguments of `.Axes.scatter`.\n\n        Argument precedence for facecolors:\n\n        - c (if not None)\n        - kwargs['facecolor']\n        - kwargs['facecolors']\n        - kwargs['color'] (==kwcolor)\n        - 'b' if in classic mode else the result of ``get_next_color_func()``\n\n        Argument precedence for edgecolors:\n\n        - kwargs['edgecolor']\n        - edgecolors (is an explicit kw argument in scatter())\n        - kwargs['color'] (==kwcolor)\n        - 'face' if not in classic mode else None\n\n        Parameters\n        ----------\n        c : color or sequence or sequence of color or None\n            See argument description of `.Axes.scatter`.\n        edgecolors : color or sequence of color or {'face', 'none'} or None\n            See argument description of `.Axes.scatter`.\n        kwargs : dict\n            Additional kwargs. If these keys exist, we pop and process them:\n            'facecolors', 'facecolor', 'edgecolor', 'color'\n            Note: The dict is modified by this function.\n        xsize : int\n            The size of the x and y arrays passed to `.Axes.scatter`.\n        get_next_color_func : callable\n            A callable that returns a color. This color is used as facecolor\n            if no other color is provided.\n\n            Note, that this is a function rather than a fixed color value to\n            support conditional evaluation of the next color.  As of the\n            current implementation obtaining the next color from the\n            property cycle advances the cycle. This must only happen if we\n            actually use the color, which will only be decided within this\n            method.\n\n        Returns\n        -------\n        c\n            The input *c* if it was not *None*, else a color derived from the\n            other inputs or defaults.\n        colors : array(N, 4) or None\n            The facecolors as RGBA values, or *None* if a colormap is used.\n        edgecolors\n            The edgecolor.\n\n        "
        facecolors = kwargs.pop('facecolors', None)
        facecolors = kwargs.pop('facecolor', facecolors)
        edgecolors = kwargs.pop('edgecolor', edgecolors)
        kwcolor = kwargs.pop('color', None)
        if kwcolor is not None and c is not None:
            raise ValueError("Supply a 'c' argument or a 'color' kwarg but not both; they differ but their functionalities overlap.")
        if kwcolor is not None:
            try:
                mcolors.to_rgba_array(kwcolor)
            except ValueError as err:
                raise ValueError("'color' kwarg must be a color or sequence of color specs.  For a sequence of values to be color-mapped, use the 'c' argument instead.") from err
            if edgecolors is None:
                edgecolors = kwcolor
            if facecolors is None:
                facecolors = kwcolor
        if edgecolors is None and (not mpl.rcParams['_internal.classic_mode']):
            edgecolors = mpl.rcParams['scatter.edgecolors']
        c_was_none = c is None
        if c is None:
            c = facecolors if facecolors is not None else 'b' if mpl.rcParams['_internal.classic_mode'] else get_next_color_func()
        c_is_string_or_strings = isinstance(c, str) or (np.iterable(c) and len(c) > 0 and isinstance(cbook._safe_first_finite(c), str))

        def invalid_shape_exception(csize, xsize):
            if False:
                for i in range(10):
                    print('nop')
            return ValueError(f"'c' argument has {csize} elements, which is inconsistent with 'x' and 'y' with size {xsize}.")
        c_is_mapped = False
        valid_shape = True
        if not c_was_none and kwcolor is None and (not c_is_string_or_strings):
            try:
                c = np.asanyarray(c, dtype=float)
            except ValueError:
                pass
            else:
                if c.shape == (1, 4) or c.shape == (1, 3):
                    c_is_mapped = False
                    if c.size != xsize:
                        valid_shape = False
                elif c.size == xsize:
                    c = c.ravel()
                    c_is_mapped = True
                else:
                    if c.shape in ((3,), (4,)):
                        _api.warn_external('*c* argument looks like a single numeric RGB or RGBA sequence, which should be avoided as value-mapping will have precedence in case its length matches with *x* & *y*.  Please use the *color* keyword-argument or provide a 2D array with a single row if you intend to specify the same RGB or RGBA value for all points.')
                    valid_shape = False
        if not c_is_mapped:
            try:
                colors = mcolors.to_rgba_array(c)
            except (TypeError, ValueError) as err:
                if 'RGBA values should be within 0-1 range' in str(err):
                    raise
                else:
                    if not valid_shape:
                        raise invalid_shape_exception(c.size, xsize) from err
                    raise ValueError(f"'c' argument must be a color, a sequence of colors, or a sequence of numbers, not {c!r}") from err
            else:
                if len(colors) not in (0, 1, xsize):
                    raise invalid_shape_exception(len(colors), xsize)
        else:
            colors = None
        return (c, colors, edgecolors)

    @_preprocess_data(replace_names=['x', 'y', 's', 'linewidths', 'edgecolors', 'c', 'facecolor', 'facecolors', 'color'], label_namer='y')
    @_docstring.interpd
    def scatter(self, x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, *, edgecolors=None, plotnonfinite=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        A scatter plot of *y* vs. *x* with varying marker size and/or color.\n\n        Parameters\n        ----------\n        x, y : float or array-like, shape (n, )\n            The data positions.\n\n        s : float or array-like, shape (n, ), optional\n            The marker size in points**2 (typographic points are 1/72 in.).\n            Default is ``rcParams[\'lines.markersize\'] ** 2``.\n\n            The linewidth and edgecolor can visually interact with the marker\n            size, and can lead to artifacts if the marker size is smaller than\n            the linewidth.\n\n            If the linewidth is greater than 0 and the edgecolor is anything\n            but *\'none\'*, then the effective size of the marker will be\n            increased by half the linewidth because the stroke will be centered\n            on the edge of the shape.\n\n            To eliminate the marker edge either set *linewidth=0* or\n            *edgecolor=\'none\'*.\n\n        c : array-like or list of colors or color, optional\n            The marker colors. Possible values:\n\n            - A scalar or sequence of n numbers to be mapped to colors using\n              *cmap* and *norm*.\n            - A 2D array in which the rows are RGB or RGBA.\n            - A sequence of colors of length n.\n            - A single color format string.\n\n            Note that *c* should not be a single numeric RGB or RGBA sequence\n            because that is indistinguishable from an array of values to be\n            colormapped. If you want to specify the same RGB or RGBA value for\n            all points, use a 2D array with a single row.  Otherwise,\n            value-matching will have precedence in case of a size matching with\n            *x* and *y*.\n\n            If you wish to specify a single color for all points\n            prefer the *color* keyword argument.\n\n            Defaults to `None`. In that case the marker color is determined\n            by the value of *color*, *facecolor* or *facecolors*. In case\n            those are not specified or `None`, the marker color is determined\n            by the next color of the ``Axes``\' current "shape and fill" color\n            cycle. This cycle defaults to :rc:`axes.prop_cycle`.\n\n        marker : `~.markers.MarkerStyle`, default: :rc:`scatter.marker`\n            The marker style. *marker* can be either an instance of the class\n            or the text shorthand for a particular marker.\n            See :mod:`matplotlib.markers` for more information about marker\n            styles.\n\n        %(cmap_doc)s\n\n            This parameter is ignored if *c* is RGB(A).\n\n        %(norm_doc)s\n\n            This parameter is ignored if *c* is RGB(A).\n\n        %(vmin_vmax_doc)s\n\n            This parameter is ignored if *c* is RGB(A).\n\n        alpha : float, default: None\n            The alpha blending value, between 0 (transparent) and 1 (opaque).\n\n        linewidths : float or array-like, default: :rc:`lines.linewidth`\n            The linewidth of the marker edges. Note: The default *edgecolors*\n            is \'face\'. You may want to change this as well.\n\n        edgecolors : {\'face\', \'none\', *None*} or color or sequence of color, default: :rc:`scatter.edgecolors`\n            The edge color of the marker. Possible values:\n\n            - \'face\': The edge color will always be the same as the face color.\n            - \'none\': No patch boundary will be drawn.\n            - A color or sequence of colors.\n\n            For non-filled markers, *edgecolors* is ignored. Instead, the color\n            is determined like with \'face\', i.e. from *c*, *colors*, or\n            *facecolors*.\n\n        plotnonfinite : bool, default: False\n            Whether to plot points with nonfinite *c* (i.e. ``inf``, ``-inf``\n            or ``nan``). If ``True`` the points are drawn with the *bad*\n            colormap color (see `.Colormap.set_bad`).\n\n        Returns\n        -------\n        `~matplotlib.collections.PathCollection`\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n        **kwargs : `~matplotlib.collections.Collection` properties\n\n        See Also\n        --------\n        plot : To plot scatter plots when markers are identical in size and\n            color.\n\n        Notes\n        -----\n        * The `.plot` function will be faster for scatterplots where markers\n          don\'t vary in size or color.\n\n        * Any or all of *x*, *y*, *s*, and *c* may be masked arrays, in which\n          case all masks will be combined and only unmasked points will be\n          plotted.\n\n        * Fundamentally, scatter works with 1D arrays; *x*, *y*, *s*, and *c*\n          may be input as N-D arrays, but within scatter they will be\n          flattened. The exception is *c*, which will be flattened only if its\n          size matches the size of *x* and *y*.\n\n        '
        if edgecolors is not None:
            kwargs.update({'edgecolors': edgecolors})
        if linewidths is not None:
            kwargs.update({'linewidths': linewidths})
        kwargs = cbook.normalize_kwargs(kwargs, mcoll.Collection)
        linewidths = kwargs.pop('linewidth', None)
        edgecolors = kwargs.pop('edgecolor', None)
        (x, y) = self._process_unit_info([('x', x), ('y', y)], kwargs)
        x = np.ma.ravel(x)
        y = np.ma.ravel(y)
        if x.size != y.size:
            raise ValueError('x and y must be the same size')
        if s is None:
            s = 20 if mpl.rcParams['_internal.classic_mode'] else mpl.rcParams['lines.markersize'] ** 2.0
        s = np.ma.ravel(s)
        if len(s) not in (1, x.size) or (not np.issubdtype(s.dtype, np.floating) and (not np.issubdtype(s.dtype, np.integer))):
            raise ValueError('s must be a scalar, or float array-like with the same size as x and y')
        orig_edgecolor = edgecolors
        if edgecolors is None:
            orig_edgecolor = kwargs.get('edgecolor', None)
        (c, colors, edgecolors) = self._parse_scatter_color_args(c, edgecolors, kwargs, x.size, get_next_color_func=self._get_patches_for_fill.get_next_color)
        if plotnonfinite and colors is None:
            c = np.ma.masked_invalid(c)
            (x, y, s, edgecolors, linewidths) = cbook._combine_masks(x, y, s, edgecolors, linewidths)
        else:
            (x, y, s, c, colors, edgecolors, linewidths) = cbook._combine_masks(x, y, s, c, colors, edgecolors, linewidths)
        if x.size in (3, 4) and np.ma.is_masked(edgecolors) and (not np.ma.is_masked(orig_edgecolor)):
            edgecolors = edgecolors.data
        scales = s
        if marker is None:
            marker = mpl.rcParams['scatter.marker']
        if isinstance(marker, mmarkers.MarkerStyle):
            marker_obj = marker
        else:
            marker_obj = mmarkers.MarkerStyle(marker)
        path = marker_obj.get_path().transformed(marker_obj.get_transform())
        if not marker_obj.is_filled():
            if orig_edgecolor is not None:
                _api.warn_external(f'You passed a edgecolor/edgecolors ({orig_edgecolor!r}) for an unfilled marker ({marker!r}).  Matplotlib is ignoring the edgecolor in favor of the facecolor.  This behavior may change in the future.')
            if marker_obj.get_fillstyle() == 'none':
                edgecolors = colors
                colors = 'none'
            else:
                edgecolors = 'face'
            if linewidths is None:
                linewidths = mpl.rcParams['lines.linewidth']
            elif np.iterable(linewidths):
                linewidths = [lw if lw is not None else mpl.rcParams['lines.linewidth'] for lw in linewidths]
        offsets = np.ma.column_stack([x, y])
        collection = mcoll.PathCollection((path,), scales, facecolors=colors, edgecolors=edgecolors, linewidths=linewidths, offsets=offsets, offset_transform=kwargs.pop('transform', self.transData), alpha=alpha)
        collection.set_transform(mtransforms.IdentityTransform())
        if colors is None:
            collection.set_array(c)
            collection.set_cmap(cmap)
            collection.set_norm(norm)
            collection._scale_norm(norm, vmin, vmax)
        else:
            extra_kwargs = {'cmap': cmap, 'norm': norm, 'vmin': vmin, 'vmax': vmax}
            extra_keys = [k for (k, v) in extra_kwargs.items() if v is not None]
            if any(extra_keys):
                keys_str = ', '.join((f"'{k}'" for k in extra_keys))
                _api.warn_external(f"No data for colormapping provided via 'c'. Parameters {keys_str} will be ignored")
        collection._internal_update(kwargs)
        if mpl.rcParams['_internal.classic_mode']:
            if self._xmargin < 0.05 and x.size > 0:
                self.set_xmargin(0.05)
            if self._ymargin < 0.05 and x.size > 0:
                self.set_ymargin(0.05)
        self.add_collection(collection)
        self._request_autoscale_view()
        return collection

    @_preprocess_data(replace_names=['x', 'y', 'C'], label_namer='y')
    @_docstring.dedent_interpd
    def hexbin(self, x, y, C=None, gridsize=100, bins=None, xscale='linear', yscale='linear', extent=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, edgecolors='face', reduce_C_function=np.mean, mincnt=None, marginals=False, **kwargs):
        if False:
            return 10
        "\n        Make a 2D hexagonal binning plot of points *x*, *y*.\n\n        If *C* is *None*, the value of the hexagon is determined by the number\n        of points in the hexagon. Otherwise, *C* specifies values at the\n        coordinate (x[i], y[i]). For each hexagon, these values are reduced\n        using *reduce_C_function*.\n\n        Parameters\n        ----------\n        x, y : array-like\n            The data positions. *x* and *y* must be of the same length.\n\n        C : array-like, optional\n            If given, these values are accumulated in the bins. Otherwise,\n            every point has a value of 1. Must be of the same length as *x*\n            and *y*.\n\n        gridsize : int or (int, int), default: 100\n            If a single int, the number of hexagons in the *x*-direction.\n            The number of hexagons in the *y*-direction is chosen such that\n            the hexagons are approximately regular.\n\n            Alternatively, if a tuple (*nx*, *ny*), the number of hexagons\n            in the *x*-direction and the *y*-direction. In the\n            *y*-direction, counting is done along vertically aligned\n            hexagons, not along the zig-zag chains of hexagons; see the\n            following illustration.\n\n            .. plot::\n\n               import numpy\n               import matplotlib.pyplot as plt\n\n               np.random.seed(19680801)\n               n= 300\n               x = np.random.standard_normal(n)\n               y = np.random.standard_normal(n)\n\n               fig, ax = plt.subplots(figsize=(4, 4))\n               h = ax.hexbin(x, y, gridsize=(5, 3))\n               hx, hy = h.get_offsets().T\n               ax.plot(hx[24::3], hy[24::3], 'ro-')\n               ax.plot(hx[-3:], hy[-3:], 'ro-')\n               ax.set_title('gridsize=(5, 3)')\n               ax.axis('off')\n\n            To get approximately regular hexagons, choose\n            :math:`n_x = \\sqrt{3}\\,n_y`.\n\n        bins : 'log' or int or sequence, default: None\n            Discretization of the hexagon values.\n\n            - If *None*, no binning is applied; the color of each hexagon\n              directly corresponds to its count value.\n            - If 'log', use a logarithmic scale for the colormap.\n              Internally, :math:`log_{10}(i+1)` is used to determine the\n              hexagon color. This is equivalent to ``norm=LogNorm()``.\n            - If an integer, divide the counts in the specified number\n              of bins, and color the hexagons accordingly.\n            - If a sequence of values, the values of the lower bound of\n              the bins to be used.\n\n        xscale : {'linear', 'log'}, default: 'linear'\n            Use a linear or log10 scale on the horizontal axis.\n\n        yscale : {'linear', 'log'}, default: 'linear'\n            Use a linear or log10 scale on the vertical axis.\n\n        mincnt : int >= 0, default: *None*\n            If not *None*, only display cells with at least *mincnt*\n            number of points in the cell.\n\n        marginals : bool, default: *False*\n            If marginals is *True*, plot the marginal density as\n            colormapped rectangles along the bottom of the x-axis and\n            left of the y-axis.\n\n        extent : 4-tuple of float, default: *None*\n            The limits of the bins (xmin, xmax, ymin, ymax).\n            The default assigns the limits based on\n            *gridsize*, *x*, *y*, *xscale* and *yscale*.\n\n            If *xscale* or *yscale* is set to 'log', the limits are\n            expected to be the exponent for a power of 10. E.g. for\n            x-limits of 1 and 50 in 'linear' scale and y-limits\n            of 10 and 1000 in 'log' scale, enter (1, 50, 1, 3).\n\n        Returns\n        -------\n        `~matplotlib.collections.PolyCollection`\n            A `.PolyCollection` defining the hexagonal bins.\n\n            - `.PolyCollection.get_offsets` contains a Mx2 array containing\n              the x, y positions of the M hexagon centers.\n            - `.PolyCollection.get_array` contains the values of the M\n              hexagons.\n\n            If *marginals* is *True*, horizontal\n            bar and vertical bar (both PolyCollections) will be attached\n            to the return collection as attributes *hbar* and *vbar*.\n\n        Other Parameters\n        ----------------\n        %(cmap_doc)s\n\n        %(norm_doc)s\n\n        %(vmin_vmax_doc)s\n\n        alpha : float between 0 and 1, optional\n            The alpha blending value, between 0 (transparent) and 1 (opaque).\n\n        linewidths : float, default: *None*\n            If *None*, defaults to :rc:`patch.linewidth`.\n\n        edgecolors : {'face', 'none', *None*} or color, default: 'face'\n            The color of the hexagon edges. Possible values are:\n\n            - 'face': Draw the edges in the same color as the fill color.\n            - 'none': No edges are drawn. This can sometimes lead to unsightly\n              unpainted pixels between the hexagons.\n            - *None*: Draw outlines in the default color.\n            - An explicit color.\n\n        reduce_C_function : callable, default: `numpy.mean`\n            The function to aggregate *C* within the bins. It is ignored if\n            *C* is not given. This must have the signature::\n\n                def reduce_C_function(C: array) -> float\n\n            Commonly used functions are:\n\n            - `numpy.mean`: average of the points\n            - `numpy.sum`: integral of the point values\n            - `numpy.amax`: value taken from the largest point\n\n            By default will only reduce cells with at least 1 point because some\n            reduction functions (such as `numpy.amax`) will error/warn with empty\n            input. Changing *mincnt* will adjust the cutoff, and if set to 0 will\n            pass empty input to the reduction function.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs : `~matplotlib.collections.PolyCollection` properties\n            All other keyword arguments are passed on to `.PolyCollection`:\n\n            %(PolyCollection:kwdoc)s\n\n        See Also\n        --------\n        hist2d : 2D histogram rectangular bins\n        "
        self._process_unit_info([('x', x), ('y', y)], kwargs, convert=False)
        (x, y, C) = cbook.delete_masked_points(x, y, C)
        if np.iterable(gridsize):
            (nx, ny) = gridsize
        else:
            nx = gridsize
            ny = int(nx / math.sqrt(3))
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        tx = x
        ty = y
        if xscale == 'log':
            if np.any(x <= 0.0):
                raise ValueError('x contains non-positive values, so cannot be log-scaled')
            tx = np.log10(tx)
        if yscale == 'log':
            if np.any(y <= 0.0):
                raise ValueError('y contains non-positive values, so cannot be log-scaled')
            ty = np.log10(ty)
        if extent is not None:
            (xmin, xmax, ymin, ymax) = extent
        else:
            (xmin, xmax) = (tx.min(), tx.max()) if len(x) else (0, 1)
            (ymin, ymax) = (ty.min(), ty.max()) if len(y) else (0, 1)
            (xmin, xmax) = mtransforms.nonsingular(xmin, xmax, expander=0.1)
            (ymin, ymax) = mtransforms.nonsingular(ymin, ymax, expander=0.1)
        nx1 = nx + 1
        ny1 = ny + 1
        nx2 = nx
        ny2 = ny
        n = nx1 * ny1 + nx2 * ny2
        padding = 1e-09 * (xmax - xmin)
        xmin -= padding
        xmax += padding
        sx = (xmax - xmin) / nx
        sy = (ymax - ymin) / ny
        ix = (tx - xmin) / sx
        iy = (ty - ymin) / sy
        ix1 = np.round(ix).astype(int)
        iy1 = np.round(iy).astype(int)
        ix2 = np.floor(ix).astype(int)
        iy2 = np.floor(iy).astype(int)
        i1 = np.where((0 <= ix1) & (ix1 < nx1) & (0 <= iy1) & (iy1 < ny1), ix1 * ny1 + iy1 + 1, 0)
        i2 = np.where((0 <= ix2) & (ix2 < nx2) & (0 <= iy2) & (iy2 < ny2), ix2 * ny2 + iy2 + 1, 0)
        d1 = (ix - ix1) ** 2 + 3.0 * (iy - iy1) ** 2
        d2 = (ix - ix2 - 0.5) ** 2 + 3.0 * (iy - iy2 - 0.5) ** 2
        bdist = d1 < d2
        if C is None:
            counts1 = np.bincount(i1[bdist], minlength=1 + nx1 * ny1)[1:]
            counts2 = np.bincount(i2[~bdist], minlength=1 + nx2 * ny2)[1:]
            accum = np.concatenate([counts1, counts2]).astype(float)
            if mincnt is not None:
                accum[accum < mincnt] = np.nan
            C = np.ones(len(x))
        else:
            Cs_at_i1 = [[] for _ in range(1 + nx1 * ny1)]
            Cs_at_i2 = [[] for _ in range(1 + nx2 * ny2)]
            for i in range(len(x)):
                if bdist[i]:
                    Cs_at_i1[i1[i]].append(C[i])
                else:
                    Cs_at_i2[i2[i]].append(C[i])
            if mincnt is None:
                mincnt = 1
            accum = np.array([reduce_C_function(acc) if len(acc) >= mincnt else np.nan for Cs_at_i in [Cs_at_i1, Cs_at_i2] for acc in Cs_at_i[1:]], float)
        good_idxs = ~np.isnan(accum)
        offsets = np.zeros((n, 2), float)
        offsets[:nx1 * ny1, 0] = np.repeat(np.arange(nx1), ny1)
        offsets[:nx1 * ny1, 1] = np.tile(np.arange(ny1), nx1)
        offsets[nx1 * ny1:, 0] = np.repeat(np.arange(nx2) + 0.5, ny2)
        offsets[nx1 * ny1:, 1] = np.tile(np.arange(ny2), nx2) + 0.5
        offsets[:, 0] *= sx
        offsets[:, 1] *= sy
        offsets[:, 0] += xmin
        offsets[:, 1] += ymin
        offsets = offsets[good_idxs, :]
        accum = accum[good_idxs]
        polygon = [sx, sy / 3] * np.array([[0.5, -0.5], [0.5, 0.5], [0.0, 1.0], [-0.5, 0.5], [-0.5, -0.5], [0.0, -1.0]])
        if linewidths is None:
            linewidths = [mpl.rcParams['patch.linewidth']]
        if xscale == 'log' or yscale == 'log':
            polygons = np.expand_dims(polygon, 0) + np.expand_dims(offsets, 1)
            if xscale == 'log':
                polygons[:, :, 0] = 10.0 ** polygons[:, :, 0]
                xmin = 10.0 ** xmin
                xmax = 10.0 ** xmax
                self.set_xscale(xscale)
            if yscale == 'log':
                polygons[:, :, 1] = 10.0 ** polygons[:, :, 1]
                ymin = 10.0 ** ymin
                ymax = 10.0 ** ymax
                self.set_yscale(yscale)
            collection = mcoll.PolyCollection(polygons, edgecolors=edgecolors, linewidths=linewidths)
        else:
            collection = mcoll.PolyCollection([polygon], edgecolors=edgecolors, linewidths=linewidths, offsets=offsets, offset_transform=mtransforms.AffineDeltaTransform(self.transData))
        if bins == 'log':
            if norm is not None:
                _api.warn_external(f"Only one of 'bins' and 'norm' arguments can be supplied, ignoring bins={bins}")
            else:
                norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
                vmin = vmax = None
            bins = None
        if norm is not None:
            if norm.vmin is None and norm.vmax is None:
                norm.autoscale(accum)
        if bins is not None:
            if not np.iterable(bins):
                (minimum, maximum) = (min(accum), max(accum))
                bins -= 1
                bins = minimum + (maximum - minimum) * np.arange(bins) / bins
            bins = np.sort(bins)
            accum = bins.searchsorted(accum)
        collection.set_array(accum)
        collection.set_cmap(cmap)
        collection.set_norm(norm)
        collection.set_alpha(alpha)
        collection._internal_update(kwargs)
        collection._scale_norm(norm, vmin, vmax)
        corners = ((xmin, ymin), (xmax, ymax))
        self.update_datalim(corners)
        self._request_autoscale_view(tight=True)
        self.add_collection(collection, autolim=False)
        if not marginals:
            return collection
        bars = []
        for (zname, z, zmin, zmax, zscale, nbins) in [('x', x, xmin, xmax, xscale, nx), ('y', y, ymin, ymax, yscale, 2 * ny)]:
            if zscale == 'log':
                bin_edges = np.geomspace(zmin, zmax, nbins + 1)
            else:
                bin_edges = np.linspace(zmin, zmax, nbins + 1)
            verts = np.empty((nbins, 4, 2))
            verts[:, 0, 0] = verts[:, 1, 0] = bin_edges[:-1]
            verts[:, 2, 0] = verts[:, 3, 0] = bin_edges[1:]
            verts[:, 0, 1] = verts[:, 3, 1] = 0.0
            verts[:, 1, 1] = verts[:, 2, 1] = 0.05
            if zname == 'y':
                verts = verts[:, :, ::-1]
            bin_idxs = np.searchsorted(bin_edges, z) - 1
            values = np.empty(nbins)
            for i in range(nbins):
                ci = C[bin_idxs == i]
                values[i] = reduce_C_function(ci) if len(ci) > 0 else np.nan
            mask = ~np.isnan(values)
            verts = verts[mask]
            values = values[mask]
            trans = getattr(self, f'get_{zname}axis_transform')(which='grid')
            bar = mcoll.PolyCollection(verts, transform=trans, edgecolors='face')
            bar.set_array(values)
            bar.set_cmap(cmap)
            bar.set_norm(norm)
            bar.set_alpha(alpha)
            bar._internal_update(kwargs)
            bars.append(self.add_collection(bar, autolim=False))
        (collection.hbar, collection.vbar) = bars

        def on_changed(collection):
            if False:
                return 10
            collection.hbar.set_cmap(collection.get_cmap())
            collection.hbar.set_cmap(collection.get_cmap())
            collection.vbar.set_clim(collection.get_clim())
            collection.vbar.set_clim(collection.get_clim())
        collection.callbacks.connect('changed', on_changed)
        return collection

    @_docstring.dedent_interpd
    def arrow(self, x, y, dx, dy, **kwargs):
        if False:
            print('Hello World!')
        '\n        Add an arrow to the Axes.\n\n        This draws an arrow from ``(x, y)`` to ``(x+dx, y+dy)``.\n\n        Parameters\n        ----------\n        %(FancyArrow)s\n\n        Returns\n        -------\n        `.FancyArrow`\n            The created `.FancyArrow` object.\n\n        Notes\n        -----\n        The resulting arrow is affected by the Axes aspect ratio and limits.\n        This may produce an arrow whose head is not square with its stem. To\n        create an arrow whose head is square with its stem,\n        use :meth:`annotate` for example:\n\n        >>> ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),\n        ...             arrowprops=dict(arrowstyle="->"))\n\n        '
        x = self.convert_xunits(x)
        y = self.convert_yunits(y)
        dx = self.convert_xunits(dx)
        dy = self.convert_yunits(dy)
        a = mpatches.FancyArrow(x, y, dx, dy, **kwargs)
        self.add_patch(a)
        self._request_autoscale_view()
        return a

    @_docstring.copy(mquiver.QuiverKey.__init__)
    def quiverkey(self, Q, X, Y, U, label, **kwargs):
        if False:
            i = 10
            return i + 15
        qk = mquiver.QuiverKey(Q, X, Y, U, label, **kwargs)
        self.add_artist(qk)
        return qk

    def _quiver_units(self, args, kwargs):
        if False:
            i = 10
            return i + 15
        if len(args) > 3:
            (x, y) = args[0:2]
            (x, y) = self._process_unit_info([('x', x), ('y', y)], kwargs)
            return (x, y) + args[2:]
        return args

    @_preprocess_data()
    @_docstring.dedent_interpd
    def quiver(self, *args, **kwargs):
        if False:
            return 10
        '%(quiver_doc)s'
        args = self._quiver_units(args, kwargs)
        q = mquiver.Quiver(self, *args, **kwargs)
        self.add_collection(q, autolim=True)
        self._request_autoscale_view()
        return q

    @_preprocess_data()
    @_docstring.dedent_interpd
    def barbs(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        '%(barbs_doc)s'
        args = self._quiver_units(args, kwargs)
        b = mquiver.Barbs(self, *args, **kwargs)
        self.add_collection(b, autolim=True)
        self._request_autoscale_view()
        return b

    def fill(self, *args, data=None, **kwargs):
        if False:
            return 10
        '\n        Plot filled polygons.\n\n        Parameters\n        ----------\n        *args : sequence of x, y, [color]\n            Each polygon is defined by the lists of *x* and *y* positions of\n            its nodes, optionally followed by a *color* specifier. See\n            :mod:`matplotlib.colors` for supported color specifiers. The\n            standard color cycle is used for polygons without a color\n            specifier.\n\n            You can plot multiple polygons by providing multiple *x*, *y*,\n            *[color]* groups.\n\n            For example, each of the following is legal::\n\n                ax.fill(x, y)                    # a polygon with default color\n                ax.fill(x, y, "b")               # a blue polygon\n                ax.fill(x, y, x2, y2)            # two polygons\n                ax.fill(x, y, "b", x2, y2, "r")  # a blue and a red polygon\n\n        data : indexable object, optional\n            An object with labelled data. If given, provide the label names to\n            plot in *x* and *y*, e.g.::\n\n                ax.fill("time", "signal",\n                        data={"time": [0, 1, 2], "signal": [0, 1, 0]})\n\n        Returns\n        -------\n        list of `~matplotlib.patches.Polygon`\n\n        Other Parameters\n        ----------------\n        **kwargs : `~matplotlib.patches.Polygon` properties\n\n        Notes\n        -----\n        Use :meth:`fill_between` if you would like to fill the region between\n        two curves.\n        '
        kwargs = cbook.normalize_kwargs(kwargs, mlines.Line2D)
        patches = [*self._get_patches_for_fill(self, *args, data=data, **kwargs)]
        for poly in patches:
            self.add_patch(poly)
        self._request_autoscale_view()
        return patches

    def _fill_between_x_or_y(self, ind_dir, ind, dep1, dep2=0, *, where=None, interpolate=False, step=None, **kwargs):
        if False:
            return 10
        "\n        Fill the area between two {dir} curves.\n\n        The curves are defined by the points (*{ind}*, *{dep}1*) and (*{ind}*,\n        *{dep}2*).  This creates one or multiple polygons describing the filled\n        area.\n\n        You may exclude some {dir} sections from filling using *where*.\n\n        By default, the edges connect the given points directly.  Use *step*\n        if the filling should be a step function, i.e. constant in between\n        *{ind}*.\n\n        Parameters\n        ----------\n        {ind} : array (length N)\n            The {ind} coordinates of the nodes defining the curves.\n\n        {dep}1 : array (length N) or scalar\n            The {dep} coordinates of the nodes defining the first curve.\n\n        {dep}2 : array (length N) or scalar, default: 0\n            The {dep} coordinates of the nodes defining the second curve.\n\n        where : array of bool (length N), optional\n            Define *where* to exclude some {dir} regions from being filled.\n            The filled regions are defined by the coordinates ``{ind}[where]``.\n            More precisely, fill between ``{ind}[i]`` and ``{ind}[i+1]`` if\n            ``where[i] and where[i+1]``.  Note that this definition implies\n            that an isolated *True* value between two *False* values in *where*\n            will not result in filling.  Both sides of the *True* position\n            remain unfilled due to the adjacent *False* values.\n\n        interpolate : bool, default: False\n            This option is only relevant if *where* is used and the two curves\n            are crossing each other.\n\n            Semantically, *where* is often used for *{dep}1* > *{dep}2* or\n            similar.  By default, the nodes of the polygon defining the filled\n            region will only be placed at the positions in the *{ind}* array.\n            Such a polygon cannot describe the above semantics close to the\n            intersection.  The {ind}-sections containing the intersection are\n            simply clipped.\n\n            Setting *interpolate* to *True* will calculate the actual\n            intersection point and extend the filled region up to this point.\n\n        step : {{'pre', 'post', 'mid'}}, optional\n            Define *step* if the filling should be a step function,\n            i.e. constant in between *{ind}*.  The value determines where the\n            step will occur:\n\n            - 'pre': The y value is continued constantly to the left from\n              every *x* position, i.e. the interval ``(x[i-1], x[i]]`` has the\n              value ``y[i]``.\n            - 'post': The y value is continued constantly to the right from\n              every *x* position, i.e. the interval ``[x[i], x[i+1])`` has the\n              value ``y[i]``.\n            - 'mid': Steps occur half-way between the *x* positions.\n\n        Returns\n        -------\n        `.PolyCollection`\n            A `.PolyCollection` containing the plotted polygons.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            All other keyword arguments are passed on to `.PolyCollection`.\n            They control the `.Polygon` properties:\n\n            %(PolyCollection:kwdoc)s\n\n        See Also\n        --------\n        fill_between : Fill between two sets of y-values.\n        fill_betweenx : Fill between two sets of x-values.\n        "
        dep_dir = {'x': 'y', 'y': 'x'}[ind_dir]
        if not mpl.rcParams['_internal.classic_mode']:
            kwargs = cbook.normalize_kwargs(kwargs, mcoll.Collection)
            if not any((c in kwargs for c in ('color', 'facecolor'))):
                kwargs['facecolor'] = self._get_patches_for_fill.get_next_color()
        (ind, dep1, dep2) = map(ma.masked_invalid, self._process_unit_info([(ind_dir, ind), (dep_dir, dep1), (dep_dir, dep2)], kwargs))
        for (name, array) in [(ind_dir, ind), (f'{dep_dir}1', dep1), (f'{dep_dir}2', dep2)]:
            if array.ndim > 1:
                raise ValueError(f'{name!r} is not 1-dimensional')
        if where is None:
            where = True
        else:
            where = np.asarray(where, dtype=bool)
            if where.size != ind.size:
                raise ValueError(f'where size ({where.size}) does not match {ind_dir} size ({ind.size})')
        where = where & ~functools.reduce(np.logical_or, map(np.ma.getmaskarray, [ind, dep1, dep2]))
        (ind, dep1, dep2) = np.broadcast_arrays(np.atleast_1d(ind), dep1, dep2, subok=True)
        polys = []
        for (idx0, idx1) in cbook.contiguous_regions(where):
            indslice = ind[idx0:idx1]
            dep1slice = dep1[idx0:idx1]
            dep2slice = dep2[idx0:idx1]
            if step is not None:
                step_func = cbook.STEP_LOOKUP_MAP['steps-' + step]
                (indslice, dep1slice, dep2slice) = step_func(indslice, dep1slice, dep2slice)
            if not len(indslice):
                continue
            N = len(indslice)
            pts = np.zeros((2 * N + 2, 2))
            if interpolate:

                def get_interp_point(idx):
                    if False:
                        print('Hello World!')
                    im1 = max(idx - 1, 0)
                    ind_values = ind[im1:idx + 1]
                    diff_values = dep1[im1:idx + 1] - dep2[im1:idx + 1]
                    dep1_values = dep1[im1:idx + 1]
                    if len(diff_values) == 2:
                        if np.ma.is_masked(diff_values[1]):
                            return (ind[im1], dep1[im1])
                        elif np.ma.is_masked(diff_values[0]):
                            return (ind[idx], dep1[idx])
                    diff_order = diff_values.argsort()
                    diff_root_ind = np.interp(0, diff_values[diff_order], ind_values[diff_order])
                    ind_order = ind_values.argsort()
                    diff_root_dep = np.interp(diff_root_ind, ind_values[ind_order], dep1_values[ind_order])
                    return (diff_root_ind, diff_root_dep)
                start = get_interp_point(idx0)
                end = get_interp_point(idx1)
            else:
                start = (indslice[0], dep2slice[0])
                end = (indslice[-1], dep2slice[-1])
            pts[0] = start
            pts[N + 1] = end
            pts[1:N + 1, 0] = indslice
            pts[1:N + 1, 1] = dep1slice
            pts[N + 2:, 0] = indslice[::-1]
            pts[N + 2:, 1] = dep2slice[::-1]
            if ind_dir == 'y':
                pts = pts[:, ::-1]
            polys.append(pts)
        collection = mcoll.PolyCollection(polys, **kwargs)
        pts = np.vstack([np.hstack([ind[where, None], dep1[where, None]]), np.hstack([ind[where, None], dep2[where, None]])])
        if ind_dir == 'y':
            pts = pts[:, ::-1]
        up_x = up_y = True
        if 'transform' in kwargs:
            (up_x, up_y) = kwargs['transform'].contains_branch_seperately(self.transData)
        self.update_datalim(pts, updatex=up_x, updatey=up_y)
        self.add_collection(collection, autolim=False)
        self._request_autoscale_view()
        return collection

    def fill_between(self, x, y1, y2=0, where=None, interpolate=False, step=None, **kwargs):
        if False:
            while True:
                i = 10
        return self._fill_between_x_or_y('x', x, y1, y2, where=where, interpolate=interpolate, step=step, **kwargs)
    if _fill_between_x_or_y.__doc__:
        fill_between.__doc__ = _fill_between_x_or_y.__doc__.format(dir='horizontal', ind='x', dep='y')
    fill_between = _preprocess_data(_docstring.dedent_interpd(fill_between), replace_names=['x', 'y1', 'y2', 'where'])

    def fill_betweenx(self, y, x1, x2=0, where=None, step=None, interpolate=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self._fill_between_x_or_y('y', y, x1, x2, where=where, interpolate=interpolate, step=step, **kwargs)
    if _fill_between_x_or_y.__doc__:
        fill_betweenx.__doc__ = _fill_between_x_or_y.__doc__.format(dir='vertical', ind='y', dep='x')
    fill_betweenx = _preprocess_data(_docstring.dedent_interpd(fill_betweenx), replace_names=['y', 'x1', 'x2', 'where'])

    @_preprocess_data()
    @_docstring.interpd
    def imshow(self, X, cmap=None, norm=None, *, aspect=None, interpolation=None, alpha=None, vmin=None, vmax=None, origin=None, extent=None, interpolation_stage=None, filternorm=True, filterrad=4.0, resample=None, url=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Display data as an image, i.e., on a 2D regular raster.\n\n        The input may either be actual RGB(A) data, or 2D scalar data, which\n        will be rendered as a pseudocolor image. For displaying a grayscale\n        image, set up the colormapping using the parameters\n        ``cmap='gray', vmin=0, vmax=255``.\n\n        The number of pixels used to render an image is set by the Axes size\n        and the figure *dpi*. This can lead to aliasing artifacts when\n        the image is resampled, because the displayed image size will usually\n        not match the size of *X* (see\n        :doc:`/gallery/images_contours_and_fields/image_antialiasing`).\n        The resampling can be controlled via the *interpolation* parameter\n        and/or :rc:`image.interpolation`.\n\n        Parameters\n        ----------\n        X : array-like or PIL image\n            The image data. Supported array shapes are:\n\n            - (M, N): an image with scalar data. The values are mapped to\n              colors using normalization and a colormap. See parameters *norm*,\n              *cmap*, *vmin*, *vmax*.\n            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).\n            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),\n              i.e. including transparency.\n\n            The first two dimensions (M, N) define the rows and columns of\n            the image.\n\n            Out-of-range RGB(A) values are clipped.\n\n        %(cmap_doc)s\n\n            This parameter is ignored if *X* is RGB(A).\n\n        %(norm_doc)s\n\n            This parameter is ignored if *X* is RGB(A).\n\n        %(vmin_vmax_doc)s\n\n            This parameter is ignored if *X* is RGB(A).\n\n        aspect : {'equal', 'auto'} or float or None, default: None\n            The aspect ratio of the Axes.  This parameter is particularly\n            relevant for images since it determines whether data pixels are\n            square.\n\n            This parameter is a shortcut for explicitly calling\n            `.Axes.set_aspect`. See there for further details.\n\n            - 'equal': Ensures an aspect ratio of 1. Pixels will be square\n              (unless pixel sizes are explicitly made non-square in data\n              coordinates using *extent*).\n            - 'auto': The Axes is kept fixed and the aspect is adjusted so\n              that the data fit in the Axes. In general, this will result in\n              non-square pixels.\n\n            Normally, None (the default) means to use :rc:`image.aspect`.  However, if\n            the image uses a transform that does not contain the axes data transform,\n            then None means to not modify the axes aspect at all (in that case, directly\n            call `.Axes.set_aspect` if desired).\n\n        interpolation : str, default: :rc:`image.interpolation`\n            The interpolation method used.\n\n            Supported values are 'none', 'antialiased', 'nearest', 'bilinear',\n            'bicubic', 'spline16', 'spline36', 'hanning', 'hamming', 'hermite',\n            'kaiser', 'quadric', 'catrom', 'gaussian', 'bessel', 'mitchell',\n            'sinc', 'lanczos', 'blackman'.\n\n            The data *X* is resampled to the pixel size of the image on the\n            figure canvas, using the interpolation method to either up- or\n            downsample the data.\n\n            If *interpolation* is 'none', then for the ps, pdf, and svg\n            backends no down- or upsampling occurs, and the image data is\n            passed to the backend as a native image.  Note that different ps,\n            pdf, and svg viewers may display these raw pixels differently. On\n            other backends, 'none' is the same as 'nearest'.\n\n            If *interpolation* is the default 'antialiased', then 'nearest'\n            interpolation is used if the image is upsampled by more than a\n            factor of three (i.e. the number of display pixels is at least\n            three times the size of the data array).  If the upsampling rate is\n            smaller than 3, or the image is downsampled, then 'hanning'\n            interpolation is used to act as an anti-aliasing filter, unless the\n            image happens to be upsampled by exactly a factor of two or one.\n\n            See\n            :doc:`/gallery/images_contours_and_fields/interpolation_methods`\n            for an overview of the supported interpolation methods, and\n            :doc:`/gallery/images_contours_and_fields/image_antialiasing` for\n            a discussion of image antialiasing.\n\n            Some interpolation methods require an additional radius parameter,\n            which can be set by *filterrad*. Additionally, the antigrain image\n            resize filter is controlled by the parameter *filternorm*.\n\n        interpolation_stage : {'data', 'rgba'}, default: 'data'\n            If 'data', interpolation\n            is carried out on the data provided by the user.  If 'rgba', the\n            interpolation is carried out after the colormapping has been\n            applied (visual interpolation).\n\n        alpha : float or array-like, optional\n            The alpha blending value, between 0 (transparent) and 1 (opaque).\n            If *alpha* is an array, the alpha blending values are applied pixel\n            by pixel, and *alpha* must have the same shape as *X*.\n\n        origin : {'upper', 'lower'}, default: :rc:`image.origin`\n            Place the [0, 0] index of the array in the upper left or lower\n            left corner of the Axes. The convention (the default) 'upper' is\n            typically used for matrices and images.\n\n            Note that the vertical axis points upward for 'lower'\n            but downward for 'upper'.\n\n            See the :ref:`imshow_extent` tutorial for\n            examples and a more detailed description.\n\n        extent : floats (left, right, bottom, top), optional\n            The bounding box in data coordinates that the image will fill.\n            These values may be unitful and match the units of the Axes.\n            The image is stretched individually along x and y to fill the box.\n\n            The default extent is determined by the following conditions.\n            Pixels have unit size in data coordinates. Their centers are on\n            integer coordinates, and their center coordinates range from 0 to\n            columns-1 horizontally and from 0 to rows-1 vertically.\n\n            Note that the direction of the vertical axis and thus the default\n            values for top and bottom depend on *origin*:\n\n            - For ``origin == 'upper'`` the default is\n              ``(-0.5, numcols-0.5, numrows-0.5, -0.5)``.\n            - For ``origin == 'lower'`` the default is\n              ``(-0.5, numcols-0.5, -0.5, numrows-0.5)``.\n\n            See the :ref:`imshow_extent` tutorial for\n            examples and a more detailed description.\n\n        filternorm : bool, default: True\n            A parameter for the antigrain image resize filter (see the\n            antigrain documentation).  If *filternorm* is set, the filter\n            normalizes integer values and corrects the rounding errors. It\n            doesn't do anything with the source floating point values, it\n            corrects only integers according to the rule of 1.0 which means\n            that any sum of pixel weights must be equal to 1.0.  So, the\n            filter function must produce a graph of the proper shape.\n\n        filterrad : float > 0, default: 4.0\n            The filter radius for filters that have a radius parameter, i.e.\n            when interpolation is one of: 'sinc', 'lanczos' or 'blackman'.\n\n        resample : bool, default: :rc:`image.resample`\n            When *True*, use a full resampling method.  When *False*, only\n            resample when the output image is larger than the input image.\n\n        url : str, optional\n            Set the url of the created `.AxesImage`. See `.Artist.set_url`.\n\n        Returns\n        -------\n        `~matplotlib.image.AxesImage`\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs : `~matplotlib.artist.Artist` properties\n            These parameters are passed on to the constructor of the\n            `.AxesImage` artist.\n\n        See Also\n        --------\n        matshow : Plot a matrix or an array as an image.\n\n        Notes\n        -----\n        Unless *extent* is used, pixel centers will be located at integer\n        coordinates. In other words: the origin will coincide with the center\n        of pixel (0, 0).\n\n        There are two common representations for RGB images with an alpha\n        channel:\n\n        -   Straight (unassociated) alpha: R, G, and B channels represent the\n            color of the pixel, disregarding its opacity.\n        -   Premultiplied (associated) alpha: R, G, and B channels represent\n            the color of the pixel, adjusted for its opacity by multiplication.\n\n        `~matplotlib.pyplot.imshow` expects RGB images adopting the straight\n        (unassociated) alpha representation.\n        "
        im = mimage.AxesImage(self, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin, extent=extent, filternorm=filternorm, filterrad=filterrad, resample=resample, interpolation_stage=interpolation_stage, **kwargs)
        if aspect is None and (not (im.is_transform_set() and (not im.get_transform().contains_branch(self.transData)))):
            aspect = mpl.rcParams['image.aspect']
        if aspect is not None:
            self.set_aspect(aspect)
        im.set_data(X)
        im.set_alpha(alpha)
        if im.get_clip_path() is None:
            im.set_clip_path(self.patch)
        im._scale_norm(norm, vmin, vmax)
        im.set_url(url)
        im.set_extent(im.get_extent())
        self.add_image(im)
        return im

    def _pcolorargs(self, funcname, *args, shading='auto', **kwargs):
        if False:
            i = 10
            return i + 15
        _valid_shading = ['gouraud', 'nearest', 'flat', 'auto']
        try:
            _api.check_in_list(_valid_shading, shading=shading)
        except ValueError:
            _api.warn_external(f"shading value '{shading}' not in list of valid values {_valid_shading}. Setting shading='auto'.")
            shading = 'auto'
        if len(args) == 1:
            C = np.asanyarray(args[0])
            (nrows, ncols) = C.shape[:2]
            if shading in ['gouraud', 'nearest']:
                (X, Y) = np.meshgrid(np.arange(ncols), np.arange(nrows))
            else:
                (X, Y) = np.meshgrid(np.arange(ncols + 1), np.arange(nrows + 1))
                shading = 'flat'
            C = cbook.safe_masked_invalid(C, copy=True)
            return (X, Y, C, shading)
        if len(args) == 3:
            C = np.asanyarray(args[2])
            (X, Y) = args[:2]
            (X, Y) = self._process_unit_info([('x', X), ('y', Y)], kwargs)
            (X, Y) = [cbook.safe_masked_invalid(a, copy=True) for a in [X, Y]]
            if funcname == 'pcolormesh':
                if np.ma.is_masked(X) or np.ma.is_masked(Y):
                    raise ValueError('x and y arguments to pcolormesh cannot have non-finite values or be of type numpy.ma.MaskedArray with masked values')
            (nrows, ncols) = C.shape[:2]
        else:
            raise _api.nargs_error(funcname, takes='1 or 3', given=len(args))
        Nx = X.shape[-1]
        Ny = Y.shape[0]
        if X.ndim != 2 or X.shape[0] == 1:
            x = X.reshape(1, Nx)
            X = x.repeat(Ny, axis=0)
        if Y.ndim != 2 or Y.shape[1] == 1:
            y = Y.reshape(Ny, 1)
            Y = y.repeat(Nx, axis=1)
        if X.shape != Y.shape:
            raise TypeError(f'Incompatible X, Y inputs to {funcname}; see help({funcname})')
        if shading == 'auto':
            if ncols == Nx and nrows == Ny:
                shading = 'nearest'
            else:
                shading = 'flat'
        if shading == 'flat':
            if (Nx, Ny) != (ncols + 1, nrows + 1):
                raise TypeError(f"Dimensions of C {C.shape} should be one smaller than X({Nx}) and Y({Ny}) while using shading='flat' see help({funcname})")
        else:
            if (Nx, Ny) != (ncols, nrows):
                raise TypeError('Dimensions of C %s are incompatible with X (%d) and/or Y (%d); see help(%s)' % (C.shape, Nx, Ny, funcname))
            if shading == 'nearest':

                def _interp_grid(X):
                    if False:
                        return 10
                    if np.shape(X)[1] > 1:
                        dX = np.diff(X, axis=1) / 2.0
                        if not (np.all(dX >= 0) or np.all(dX <= 0)):
                            _api.warn_external(f'The input coordinates to {funcname} are interpreted as cell centers, but are not monotonically increasing or decreasing. This may lead to incorrectly calculated cell edges, in which case, please supply explicit cell edges to {funcname}.')
                        hstack = np.ma.hstack if np.ma.isMA(X) else np.hstack
                        X = hstack((X[:, [0]] - dX[:, [0]], X[:, :-1] + dX, X[:, [-1]] + dX[:, [-1]]))
                    else:
                        X = np.hstack((X, X))
                    return X
                if ncols == Nx:
                    X = _interp_grid(X)
                    Y = _interp_grid(Y)
                if nrows == Ny:
                    X = _interp_grid(X.T).T
                    Y = _interp_grid(Y.T).T
                shading = 'flat'
        C = cbook.safe_masked_invalid(C, copy=True)
        return (X, Y, C, shading)

    @_preprocess_data()
    @_docstring.dedent_interpd
    def pcolor(self, *args, shading=None, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Create a pseudocolor plot with a non-regular rectangular grid.\n\n        Call signature::\n\n            pcolor([X, Y,] C, **kwargs)\n\n        *X* and *Y* can be used to specify the corners of the quadrilaterals.\n\n        .. hint::\n\n            ``pcolor()`` can be very slow for large arrays. In most\n            cases you should use the similar but much faster\n            `~.Axes.pcolormesh` instead. See\n            :ref:`Differences between pcolor() and pcolormesh()\n            <differences-pcolor-pcolormesh>` for a discussion of the\n            differences.\n\n        Parameters\n        ----------\n        C : 2D array-like\n            The color-mapped values.  Color-mapping is controlled by *cmap*,\n            *norm*, *vmin*, and *vmax*.\n\n        X, Y : array-like, optional\n            The coordinates of the corners of quadrilaterals of a pcolormesh::\n\n                (X[i+1, j], Y[i+1, j])       (X[i+1, j+1], Y[i+1, j+1])\n                                      ●╶───╴●\n                                      │     │\n                                      ●╶───╴●\n                    (X[i, j], Y[i, j])       (X[i, j+1], Y[i, j+1])\n\n            Note that the column index corresponds to the x-coordinate, and\n            the row index corresponds to y. For details, see the\n            :ref:`Notes <axes-pcolormesh-grid-orientation>` section below.\n\n            If ``shading=\'flat\'`` the dimensions of *X* and *Y* should be one\n            greater than those of *C*, and the quadrilateral is colored due\n            to the value at ``C[i, j]``.  If *X*, *Y* and *C* have equal\n            dimensions, a warning will be raised and the last row and column\n            of *C* will be ignored.\n\n            If ``shading=\'nearest\'``, the dimensions of *X* and *Y* should be\n            the same as those of *C* (if not, a ValueError will be raised). The\n            color ``C[i, j]`` will be centered on ``(X[i, j], Y[i, j])``.\n\n            If *X* and/or *Y* are 1-D arrays or column vectors they will be\n            expanded as needed into the appropriate 2D arrays, making a\n            rectangular grid.\n\n        shading : {\'flat\', \'nearest\', \'auto\'}, default: :rc:`pcolor.shading`\n            The fill style for the quadrilateral. Possible values:\n\n            - \'flat\': A solid color is used for each quad. The color of the\n              quad (i, j), (i+1, j), (i, j+1), (i+1, j+1) is given by\n              ``C[i, j]``. The dimensions of *X* and *Y* should be\n              one greater than those of *C*; if they are the same as *C*,\n              then a deprecation warning is raised, and the last row\n              and column of *C* are dropped.\n            - \'nearest\': Each grid point will have a color centered on it,\n              extending halfway between the adjacent grid centers.  The\n              dimensions of *X* and *Y* must be the same as *C*.\n            - \'auto\': Choose \'flat\' if dimensions of *X* and *Y* are one\n              larger than *C*.  Choose \'nearest\' if dimensions are the same.\n\n            See :doc:`/gallery/images_contours_and_fields/pcolormesh_grids`\n            for more description.\n\n        %(cmap_doc)s\n\n        %(norm_doc)s\n\n        %(vmin_vmax_doc)s\n\n        edgecolors : {\'none\', None, \'face\', color, color sequence}, optional\n            The color of the edges. Defaults to \'none\'. Possible values:\n\n            - \'none\' or \'\': No edge.\n            - *None*: :rc:`patch.edgecolor` will be used. Note that currently\n              :rc:`patch.force_edgecolor` has to be True for this to work.\n            - \'face\': Use the adjacent face color.\n            - A color or sequence of colors will set the edge color.\n\n            The singular form *edgecolor* works as an alias.\n\n        alpha : float, default: None\n            The alpha blending value of the face color, between 0 (transparent)\n            and 1 (opaque). Note: The edgecolor is currently not affected by\n            this.\n\n        snap : bool, default: False\n            Whether to snap the mesh to pixel boundaries.\n\n        Returns\n        -------\n        `matplotlib.collections.PolyQuadMesh`\n\n        Other Parameters\n        ----------------\n        antialiaseds : bool, default: False\n            The default *antialiaseds* is False if the default\n            *edgecolors*\\ ="none" is used.  This eliminates artificial lines\n            at patch boundaries, and works regardless of the value of alpha.\n            If *edgecolors* is not "none", then the default *antialiaseds*\n            is taken from :rc:`patch.antialiased`.\n            Stroking the edges may be preferred if *alpha* is 1, but will\n            cause artifacts otherwise.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Additionally, the following arguments are allowed. They are passed\n            along to the `~matplotlib.collections.PolyQuadMesh` constructor:\n\n        %(PolyCollection:kwdoc)s\n\n        See Also\n        --------\n        pcolormesh : for an explanation of the differences between\n            pcolor and pcolormesh.\n        imshow : If *X* and *Y* are each equidistant, `~.Axes.imshow` can be a\n            faster alternative.\n\n        Notes\n        -----\n        **Masked arrays**\n\n        *X*, *Y* and *C* may be masked arrays. If either ``C[i, j]``, or one\n        of the vertices surrounding ``C[i, j]`` (*X* or *Y* at\n        ``[i, j], [i+1, j], [i, j+1], [i+1, j+1]``) is masked, nothing is\n        plotted.\n\n        .. _axes-pcolor-grid-orientation:\n\n        **Grid orientation**\n\n        The grid orientation follows the standard matrix convention: An array\n        *C* with shape (nrows, ncolumns) is plotted with the column number as\n        *X* and the row number as *Y*.\n        '
        if shading is None:
            shading = mpl.rcParams['pcolor.shading']
        shading = shading.lower()
        (X, Y, C, shading) = self._pcolorargs('pcolor', *args, shading=shading, kwargs=kwargs)
        linewidths = (0.25,)
        if 'linewidth' in kwargs:
            kwargs['linewidths'] = kwargs.pop('linewidth')
        kwargs.setdefault('linewidths', linewidths)
        if 'edgecolor' in kwargs:
            kwargs['edgecolors'] = kwargs.pop('edgecolor')
        ec = kwargs.setdefault('edgecolors', 'none')
        if 'antialiaseds' in kwargs:
            kwargs['antialiased'] = kwargs.pop('antialiaseds')
        if 'antialiased' not in kwargs and cbook._str_lower_equal(ec, 'none'):
            kwargs['antialiased'] = False
        kwargs.setdefault('snap', False)
        if np.ma.isMaskedArray(X) or np.ma.isMaskedArray(Y):
            stack = np.ma.stack
            X = np.ma.asarray(X)
            Y = np.ma.asarray(Y)
            x = X.compressed()
            y = Y.compressed()
        else:
            stack = np.stack
            x = X
            y = Y
        coords = stack([X, Y], axis=-1)
        collection = mcoll.PolyQuadMesh(coords, array=C, cmap=cmap, norm=norm, alpha=alpha, **kwargs)
        collection._scale_norm(norm, vmin, vmax)
        t = collection._transform
        if not isinstance(t, mtransforms.Transform) and hasattr(t, '_as_mpl_transform'):
            t = t._as_mpl_transform(self.axes)
        if t and any(t.contains_branch_seperately(self.transData)):
            trans_to_data = t - self.transData
            pts = np.vstack([x, y]).T.astype(float)
            transformed_pts = trans_to_data.transform(pts)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
        self.add_collection(collection, autolim=False)
        minx = np.min(x)
        maxx = np.max(x)
        miny = np.min(y)
        maxy = np.max(y)
        collection.sticky_edges.x[:] = [minx, maxx]
        collection.sticky_edges.y[:] = [miny, maxy]
        corners = ((minx, miny), (maxx, maxy))
        self.update_datalim(corners)
        self._request_autoscale_view()
        return collection

    @_preprocess_data()
    @_docstring.dedent_interpd
    def pcolormesh(self, *args, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, shading=None, antialiased=False, **kwargs):
        if False:
            print('Hello World!')
        "\n        Create a pseudocolor plot with a non-regular rectangular grid.\n\n        Call signature::\n\n            pcolormesh([X, Y,] C, **kwargs)\n\n        *X* and *Y* can be used to specify the corners of the quadrilaterals.\n\n        .. hint::\n\n           `~.Axes.pcolormesh` is similar to `~.Axes.pcolor`. It is much faster\n           and preferred in most cases. For a detailed discussion on the\n           differences see :ref:`Differences between pcolor() and pcolormesh()\n           <differences-pcolor-pcolormesh>`.\n\n        Parameters\n        ----------\n        C : array-like\n            The mesh data. Supported array shapes are:\n\n            - (M, N) or M*N: a mesh with scalar data. The values are mapped to\n              colors using normalization and a colormap. See parameters *norm*,\n              *cmap*, *vmin*, *vmax*.\n            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).\n            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),\n              i.e. including transparency.\n\n            The first two dimensions (M, N) define the rows and columns of\n            the mesh data.\n\n        X, Y : array-like, optional\n            The coordinates of the corners of quadrilaterals of a pcolormesh::\n\n                (X[i+1, j], Y[i+1, j])       (X[i+1, j+1], Y[i+1, j+1])\n                                      ●╶───╴●\n                                      │     │\n                                      ●╶───╴●\n                    (X[i, j], Y[i, j])       (X[i, j+1], Y[i, j+1])\n\n            Note that the column index corresponds to the x-coordinate, and\n            the row index corresponds to y. For details, see the\n            :ref:`Notes <axes-pcolormesh-grid-orientation>` section below.\n\n            If ``shading='flat'`` the dimensions of *X* and *Y* should be one\n            greater than those of *C*, and the quadrilateral is colored due\n            to the value at ``C[i, j]``.  If *X*, *Y* and *C* have equal\n            dimensions, a warning will be raised and the last row and column\n            of *C* will be ignored.\n\n            If ``shading='nearest'`` or ``'gouraud'``, the dimensions of *X*\n            and *Y* should be the same as those of *C* (if not, a ValueError\n            will be raised).  For ``'nearest'`` the color ``C[i, j]`` is\n            centered on ``(X[i, j], Y[i, j])``.  For ``'gouraud'``, a smooth\n            interpolation is caried out between the quadrilateral corners.\n\n            If *X* and/or *Y* are 1-D arrays or column vectors they will be\n            expanded as needed into the appropriate 2D arrays, making a\n            rectangular grid.\n\n        %(cmap_doc)s\n\n        %(norm_doc)s\n\n        %(vmin_vmax_doc)s\n\n        edgecolors : {'none', None, 'face', color, color sequence}, optional\n            The color of the edges. Defaults to 'none'. Possible values:\n\n            - 'none' or '': No edge.\n            - *None*: :rc:`patch.edgecolor` will be used. Note that currently\n              :rc:`patch.force_edgecolor` has to be True for this to work.\n            - 'face': Use the adjacent face color.\n            - A color or sequence of colors will set the edge color.\n\n            The singular form *edgecolor* works as an alias.\n\n        alpha : float, default: None\n            The alpha blending value, between 0 (transparent) and 1 (opaque).\n\n        shading : {'flat', 'nearest', 'gouraud', 'auto'}, optional\n            The fill style for the quadrilateral; defaults to\n            :rc:`pcolor.shading`. Possible values:\n\n            - 'flat': A solid color is used for each quad. The color of the\n              quad (i, j), (i+1, j), (i, j+1), (i+1, j+1) is given by\n              ``C[i, j]``. The dimensions of *X* and *Y* should be\n              one greater than those of *C*; if they are the same as *C*,\n              then a deprecation warning is raised, and the last row\n              and column of *C* are dropped.\n            - 'nearest': Each grid point will have a color centered on it,\n              extending halfway between the adjacent grid centers.  The\n              dimensions of *X* and *Y* must be the same as *C*.\n            - 'gouraud': Each quad will be Gouraud shaded: The color of the\n              corners (i', j') are given by ``C[i', j']``. The color values of\n              the area in between is interpolated from the corner values.\n              The dimensions of *X* and *Y* must be the same as *C*. When\n              Gouraud shading is used, *edgecolors* is ignored.\n            - 'auto': Choose 'flat' if dimensions of *X* and *Y* are one\n              larger than *C*.  Choose 'nearest' if dimensions are the same.\n\n            See :doc:`/gallery/images_contours_and_fields/pcolormesh_grids`\n            for more description.\n\n        snap : bool, default: False\n            Whether to snap the mesh to pixel boundaries.\n\n        rasterized : bool, optional\n            Rasterize the pcolormesh when drawing vector graphics.  This can\n            speed up rendering and produce smaller files for large data sets.\n            See also :doc:`/gallery/misc/rasterization_demo`.\n\n        Returns\n        -------\n        `matplotlib.collections.QuadMesh`\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Additionally, the following arguments are allowed. They are passed\n            along to the `~matplotlib.collections.QuadMesh` constructor:\n\n        %(QuadMesh:kwdoc)s\n\n        See Also\n        --------\n        pcolor : An alternative implementation with slightly different\n            features. For a detailed discussion on the differences see\n            :ref:`Differences between pcolor() and pcolormesh()\n            <differences-pcolor-pcolormesh>`.\n        imshow : If *X* and *Y* are each equidistant, `~.Axes.imshow` can be a\n            faster alternative.\n\n        Notes\n        -----\n        **Masked arrays**\n\n        *C* may be a masked array. If ``C[i, j]`` is masked, the corresponding\n        quadrilateral will be transparent. Masking of *X* and *Y* is not\n        supported. Use `~.Axes.pcolor` if you need this functionality.\n\n        .. _axes-pcolormesh-grid-orientation:\n\n        **Grid orientation**\n\n        The grid orientation follows the standard matrix convention: An array\n        *C* with shape (nrows, ncolumns) is plotted with the column number as\n        *X* and the row number as *Y*.\n\n        .. _differences-pcolor-pcolormesh:\n\n        **Differences between pcolor() and pcolormesh()**\n\n        Both methods are used to create a pseudocolor plot of a 2D array\n        using quadrilaterals.\n\n        The main difference lies in the created object and internal data\n        handling:\n        While `~.Axes.pcolor` returns a `.PolyQuadMesh`, `~.Axes.pcolormesh`\n        returns a `.QuadMesh`. The latter is more specialized for the given\n        purpose and thus is faster. It should almost always be preferred.\n\n        There is also a slight difference in the handling of masked arrays.\n        Both `~.Axes.pcolor` and `~.Axes.pcolormesh` support masked arrays\n        for *C*. However, only `~.Axes.pcolor` supports masked arrays for *X*\n        and *Y*. The reason lies in the internal handling of the masked values.\n        `~.Axes.pcolor` leaves out the respective polygons from the\n        PolyQuadMesh. `~.Axes.pcolormesh` sets the facecolor of the masked\n        elements to transparent. You can see the difference when using\n        edgecolors. While all edges are drawn irrespective of masking in a\n        QuadMesh, the edge between two adjacent masked quadrilaterals in\n        `~.Axes.pcolor` is not drawn as the corresponding polygons do not\n        exist in the PolyQuadMesh. Because PolyQuadMesh draws each individual\n        polygon, it also supports applying hatches and linestyles to the collection.\n\n        Another difference is the support of Gouraud shading in\n        `~.Axes.pcolormesh`, which is not available with `~.Axes.pcolor`.\n\n        "
        if shading is None:
            shading = mpl.rcParams['pcolor.shading']
        shading = shading.lower()
        kwargs.setdefault('edgecolors', 'none')
        (X, Y, C, shading) = self._pcolorargs('pcolormesh', *args, shading=shading, kwargs=kwargs)
        coords = np.stack([X, Y], axis=-1)
        kwargs.setdefault('snap', mpl.rcParams['pcolormesh.snap'])
        collection = mcoll.QuadMesh(coords, antialiased=antialiased, shading=shading, array=C, cmap=cmap, norm=norm, alpha=alpha, **kwargs)
        collection._scale_norm(norm, vmin, vmax)
        coords = coords.reshape(-1, 2)
        t = collection._transform
        if not isinstance(t, mtransforms.Transform) and hasattr(t, '_as_mpl_transform'):
            t = t._as_mpl_transform(self.axes)
        if t and any(t.contains_branch_seperately(self.transData)):
            trans_to_data = t - self.transData
            coords = trans_to_data.transform(coords)
        self.add_collection(collection, autolim=False)
        (minx, miny) = np.min(coords, axis=0)
        (maxx, maxy) = np.max(coords, axis=0)
        collection.sticky_edges.x[:] = [minx, maxx]
        collection.sticky_edges.y[:] = [miny, maxy]
        corners = ((minx, miny), (maxx, maxy))
        self.update_datalim(corners)
        self._request_autoscale_view()
        return collection

    @_preprocess_data()
    @_docstring.dedent_interpd
    def pcolorfast(self, *args, alpha=None, norm=None, cmap=None, vmin=None, vmax=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Create a pseudocolor plot with a non-regular rectangular grid.\n\n        Call signature::\n\n          ax.pcolorfast([X, Y], C, /, **kwargs)\n\n        This method is similar to `~.Axes.pcolor` and `~.Axes.pcolormesh`.\n        It's designed to provide the fastest pcolor-type plotting with the\n        Agg backend. To achieve this, it uses different algorithms internally\n        depending on the complexity of the input grid (regular rectangular,\n        non-regular rectangular or arbitrary quadrilateral).\n\n        .. warning::\n\n           This method is experimental. Compared to `~.Axes.pcolor` or\n           `~.Axes.pcolormesh` it has some limitations:\n\n           - It supports only flat shading (no outlines)\n           - It lacks support for log scaling of the axes.\n           - It does not have a pyplot wrapper.\n\n        Parameters\n        ----------\n        C : array-like\n            The image data. Supported array shapes are:\n\n            - (M, N): an image with scalar data.  Color-mapping is controlled\n              by *cmap*, *norm*, *vmin*, and *vmax*.\n            - (M, N, 3): an image with RGB values (0-1 float or 0-255 int).\n            - (M, N, 4): an image with RGBA values (0-1 float or 0-255 int),\n              i.e. including transparency.\n\n            The first two dimensions (M, N) define the rows and columns of\n            the image.\n\n            This parameter can only be passed positionally.\n\n        X, Y : tuple or array-like, default: ``(0, N)``, ``(0, M)``\n            *X* and *Y* are used to specify the coordinates of the\n            quadrilaterals. There are different ways to do this:\n\n            - Use tuples ``X=(xmin, xmax)`` and ``Y=(ymin, ymax)`` to define\n              a *uniform rectangular grid*.\n\n              The tuples define the outer edges of the grid. All individual\n              quadrilaterals will be of the same size. This is the fastest\n              version.\n\n            - Use 1D arrays *X*, *Y* to specify a *non-uniform rectangular\n              grid*.\n\n              In this case *X* and *Y* have to be monotonic 1D arrays of length\n              *N+1* and *M+1*, specifying the x and y boundaries of the cells.\n\n              The speed is intermediate. Note: The grid is checked, and if\n              found to be uniform the fast version is used.\n\n            - Use 2D arrays *X*, *Y* if you need an *arbitrary quadrilateral\n              grid* (i.e. if the quadrilaterals are not rectangular).\n\n              In this case *X* and *Y* are 2D arrays with shape (M + 1, N + 1),\n              specifying the x and y coordinates of the corners of the colored\n              quadrilaterals.\n\n              This is the most general, but the slowest to render.  It may\n              produce faster and more compact output using ps, pdf, and\n              svg backends, however.\n\n            These arguments can only be passed positionally.\n\n        %(cmap_doc)s\n\n            This parameter is ignored if *C* is RGB(A).\n\n        %(norm_doc)s\n\n            This parameter is ignored if *C* is RGB(A).\n\n        %(vmin_vmax_doc)s\n\n            This parameter is ignored if *C* is RGB(A).\n\n        alpha : float, default: None\n            The alpha blending value, between 0 (transparent) and 1 (opaque).\n\n        snap : bool, default: False\n            Whether to snap the mesh to pixel boundaries.\n\n        Returns\n        -------\n        `.AxesImage` or `.PcolorImage` or `.QuadMesh`\n            The return type depends on the type of grid:\n\n            - `.AxesImage` for a regular rectangular grid.\n            - `.PcolorImage` for a non-regular rectangular grid.\n            - `.QuadMesh` for a non-rectangular grid.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Supported additional parameters depend on the type of grid.\n            See return types of *image* for further description.\n        "
        C = args[-1]
        (nr, nc) = np.shape(C)[:2]
        if len(args) == 1:
            style = 'image'
            x = [0, nc]
            y = [0, nr]
        elif len(args) == 3:
            (x, y) = args[:2]
            x = np.asarray(x)
            y = np.asarray(y)
            if x.ndim == 1 and y.ndim == 1:
                if x.size == 2 and y.size == 2:
                    style = 'image'
                else:
                    dx = np.diff(x)
                    dy = np.diff(y)
                    if np.ptp(dx) < 0.01 * abs(dx.mean()) and np.ptp(dy) < 0.01 * abs(dy.mean()):
                        style = 'image'
                    else:
                        style = 'pcolorimage'
            elif x.ndim == 2 and y.ndim == 2:
                style = 'quadmesh'
            else:
                raise TypeError('arguments do not match valid signatures')
        else:
            raise _api.nargs_error('pcolorfast', '1 or 3', len(args))
        if style == 'quadmesh':
            coords = np.stack([x, y], axis=-1)
            if np.ndim(C) not in {2, 3}:
                raise ValueError('C must be 2D or 3D')
            collection = mcoll.QuadMesh(coords, array=C, alpha=alpha, cmap=cmap, norm=norm, antialiased=False, edgecolors='none')
            self.add_collection(collection, autolim=False)
            (xl, xr, yb, yt) = (x.min(), x.max(), y.min(), y.max())
            ret = collection
        else:
            extent = (xl, xr, yb, yt) = (x[0], x[-1], y[0], y[-1])
            if style == 'image':
                im = mimage.AxesImage(self, cmap=cmap, norm=norm, data=C, alpha=alpha, extent=extent, interpolation='nearest', origin='lower', **kwargs)
            elif style == 'pcolorimage':
                im = mimage.PcolorImage(self, x, y, C, cmap=cmap, norm=norm, alpha=alpha, extent=extent, **kwargs)
            self.add_image(im)
            ret = im
        if np.ndim(C) == 2:
            ret._scale_norm(norm, vmin, vmax)
        if ret.get_clip_path() is None:
            ret.set_clip_path(self.patch)
        ret.sticky_edges.x[:] = [xl, xr]
        ret.sticky_edges.y[:] = [yb, yt]
        self.update_datalim(np.array([[xl, yb], [xr, yt]]))
        self._request_autoscale_view(tight=True)
        return ret

    @_preprocess_data()
    @_docstring.dedent_interpd
    def contour(self, *args, **kwargs):
        if False:
            print('Hello World!')
        '\n        Plot contour lines.\n\n        Call signature::\n\n            contour([X, Y,] Z, [levels], **kwargs)\n        %(contour_doc)s\n        '
        kwargs['filled'] = False
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours

    @_preprocess_data()
    @_docstring.dedent_interpd
    def contourf(self, *args, **kwargs):
        if False:
            return 10
        '\n        Plot filled contours.\n\n        Call signature::\n\n            contourf([X, Y,] Z, [levels], **kwargs)\n        %(contour_doc)s\n        '
        kwargs['filled'] = True
        contours = mcontour.QuadContourSet(self, *args, **kwargs)
        self._request_autoscale_view()
        return contours

    def clabel(self, CS, levels=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Label a contour plot.\n\n        Adds labels to line contours in given `.ContourSet`.\n\n        Parameters\n        ----------\n        CS : `.ContourSet` instance\n            Line contours to label.\n\n        levels : array-like, optional\n            A list of level values, that should be labeled. The list must be\n            a subset of ``CS.levels``. If not given, all levels are labeled.\n\n        **kwargs\n            All other parameters are documented in `~.ContourLabeler.clabel`.\n        '
        return CS.clabel(levels, **kwargs)

    @_preprocess_data(replace_names=['x', 'weights'], label_namer='x')
    def hist(self, x, bins=None, range=None, density=False, weights=None, cumulative=False, bottom=None, histtype='bar', align='mid', orientation='vertical', rwidth=None, log=False, color=None, label=None, stacked=False, **kwargs):
        if False:
            return 10
        "\n        Compute and plot a histogram.\n\n        This method uses `numpy.histogram` to bin the data in *x* and count the\n        number of values in each bin, then draws the distribution either as a\n        `.BarContainer` or `.Polygon`. The *bins*, *range*, *density*, and\n        *weights* parameters are forwarded to `numpy.histogram`.\n\n        If the data has already been binned and counted, use `~.bar` or\n        `~.stairs` to plot the distribution::\n\n            counts, bins = np.histogram(x)\n            plt.stairs(counts, bins)\n\n        Alternatively, plot pre-computed bins and counts using ``hist()`` by\n        treating each bin as a single point with a weight equal to its count::\n\n            plt.hist(bins[:-1], bins, weights=counts)\n\n        The data input *x* can be a singular array, a list of datasets of\n        potentially different lengths ([*x0*, *x1*, ...]), or a 2D ndarray in\n        which each column is a dataset. Note that the ndarray form is\n        transposed relative to the list form. If the input is an array, then\n        the return value is a tuple (*n*, *bins*, *patches*); if the input is a\n        sequence of arrays, then the return value is a tuple\n        ([*n0*, *n1*, ...], *bins*, [*patches0*, *patches1*, ...]).\n\n        Masked arrays are not supported.\n\n        Parameters\n        ----------\n        x : (n,) array or sequence of (n,) arrays\n            Input values, this takes either a single array or a sequence of\n            arrays which are not required to be of the same length.\n\n        bins : int or sequence or str, default: :rc:`hist.bins`\n            If *bins* is an integer, it defines the number of equal-width bins\n            in the range.\n\n            If *bins* is a sequence, it defines the bin edges, including the\n            left edge of the first bin and the right edge of the last bin;\n            in this case, bins may be unequally spaced.  All but the last\n            (righthand-most) bin is half-open.  In other words, if *bins* is::\n\n                [1, 2, 3, 4]\n\n            then the first bin is ``[1, 2)`` (including 1, but excluding 2) and\n            the second ``[2, 3)``.  The last bin, however, is ``[3, 4]``, which\n            *includes* 4.\n\n            If *bins* is a string, it is one of the binning strategies\n            supported by `numpy.histogram_bin_edges`: 'auto', 'fd', 'doane',\n            'scott', 'stone', 'rice', 'sturges', or 'sqrt'.\n\n        range : tuple or None, default: None\n            The lower and upper range of the bins. Lower and upper outliers\n            are ignored. If not provided, *range* is ``(x.min(), x.max())``.\n            Range has no effect if *bins* is a sequence.\n\n            If *bins* is a sequence or *range* is specified, autoscaling\n            is based on the specified bin range instead of the\n            range of x.\n\n        density : bool, default: False\n            If ``True``, draw and return a probability density: each bin\n            will display the bin's raw count divided by the total number of\n            counts *and the bin width*\n            (``density = counts / (sum(counts) * np.diff(bins))``),\n            so that the area under the histogram integrates to 1\n            (``np.sum(density * np.diff(bins)) == 1``).\n\n            If *stacked* is also ``True``, the sum of the histograms is\n            normalized to 1.\n\n        weights : (n,) array-like or None, default: None\n            An array of weights, of the same shape as *x*.  Each value in\n            *x* only contributes its associated weight towards the bin count\n            (instead of 1).  If *density* is ``True``, the weights are\n            normalized, so that the integral of the density over the range\n            remains 1.\n\n        cumulative : bool or -1, default: False\n            If ``True``, then a histogram is computed where each bin gives the\n            counts in that bin plus all bins for smaller values. The last bin\n            gives the total number of datapoints.\n\n            If *density* is also ``True`` then the histogram is normalized such\n            that the last bin equals 1.\n\n            If *cumulative* is a number less than 0 (e.g., -1), the direction\n            of accumulation is reversed.  In this case, if *density* is also\n            ``True``, then the histogram is normalized such that the first bin\n            equals 1.\n\n        bottom : array-like, scalar, or None, default: None\n            Location of the bottom of each bin, i.e. bins are drawn from\n            ``bottom`` to ``bottom + hist(x, bins)`` If a scalar, the bottom\n            of each bin is shifted by the same amount. If an array, each bin\n            is shifted independently and the length of bottom must match the\n            number of bins. If None, defaults to 0.\n\n        histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, default: 'bar'\n            The type of histogram to draw.\n\n            - 'bar' is a traditional bar-type histogram.  If multiple data\n              are given the bars are arranged side by side.\n            - 'barstacked' is a bar-type histogram where multiple\n              data are stacked on top of each other.\n            - 'step' generates a lineplot that is by default unfilled.\n            - 'stepfilled' generates a lineplot that is by default filled.\n\n        align : {'left', 'mid', 'right'}, default: 'mid'\n            The horizontal alignment of the histogram bars.\n\n            - 'left': bars are centered on the left bin edges.\n            - 'mid': bars are centered between the bin edges.\n            - 'right': bars are centered on the right bin edges.\n\n        orientation : {'vertical', 'horizontal'}, default: 'vertical'\n            If 'horizontal', `~.Axes.barh` will be used for bar-type histograms\n            and the *bottom* kwarg will be the left edges.\n\n        rwidth : float or None, default: None\n            The relative width of the bars as a fraction of the bin width.  If\n            ``None``, automatically compute the width.\n\n            Ignored if *histtype* is 'step' or 'stepfilled'.\n\n        log : bool, default: False\n            If ``True``, the histogram axis will be set to a log scale.\n\n        color : color or array-like of colors or None, default: None\n            Color or sequence of colors, one per dataset.  Default (``None``)\n            uses the standard line color sequence.\n\n        label : str or None, default: None\n            String, or sequence of strings to match multiple datasets.  Bar\n            charts yield multiple patches per dataset, but only the first gets\n            the label, so that `~.Axes.legend` will work as expected.\n\n        stacked : bool, default: False\n            If ``True``, multiple data are stacked on top of each other If\n            ``False`` multiple data are arranged side by side if histtype is\n            'bar' or on top of each other if histtype is 'step'\n\n        Returns\n        -------\n        n : array or list of arrays\n            The values of the histogram bins. See *density* and *weights* for a\n            description of the possible semantics.  If input *x* is an array,\n            then this is an array of length *nbins*. If input is a sequence of\n            arrays ``[data1, data2, ...]``, then this is a list of arrays with\n            the values of the histograms for each of the arrays in the same\n            order.  The dtype of the array *n* (or of its element arrays) will\n            always be float even if no weighting or normalization is used.\n\n        bins : array\n            The edges of the bins. Length nbins + 1 (nbins left edges and right\n            edge of last bin).  Always a single array even when multiple data\n            sets are passed in.\n\n        patches : `.BarContainer` or list of a single `.Polygon` or list of such objects\n            Container of individual artists used to create the histogram\n            or list of such containers if there are multiple input datasets.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            `~matplotlib.patches.Patch` properties\n\n        See Also\n        --------\n        hist2d : 2D histogram with rectangular bins\n        hexbin : 2D histogram with hexagonal bins\n        stairs : Plot a pre-computed histogram\n        bar : Plot a pre-computed histogram\n\n        Notes\n        -----\n        For large numbers of bins (>1000), plotting can be significantly\n        accelerated by using `~.Axes.stairs` to plot a pre-computed histogram\n        (``plt.stairs(*np.histogram(data))``), or by setting *histtype* to\n        'step' or 'stepfilled' rather than 'bar' or 'barstacked'.\n        "
        bin_range = range
        from builtins import range
        if np.isscalar(x):
            x = [x]
        if bins is None:
            bins = mpl.rcParams['hist.bins']
        _api.check_in_list(['bar', 'barstacked', 'step', 'stepfilled'], histtype=histtype)
        _api.check_in_list(['left', 'mid', 'right'], align=align)
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)
        if histtype == 'barstacked' and (not stacked):
            stacked = True
        x = cbook._reshape_2D(x, 'x')
        nx = len(x)
        if orientation == 'vertical':
            convert_units = self.convert_xunits
            x = [*self._process_unit_info([('x', x[0])], kwargs), *map(convert_units, x[1:])]
        else:
            convert_units = self.convert_yunits
            x = [*self._process_unit_info([('y', x[0])], kwargs), *map(convert_units, x[1:])]
        if bin_range is not None:
            bin_range = convert_units(bin_range)
        if not cbook.is_scalar_or_string(bins):
            bins = convert_units(bins)
        if weights is not None:
            w = cbook._reshape_2D(weights, 'weights')
        else:
            w = [None] * nx
        if len(w) != nx:
            raise ValueError('weights should have the same shape as x')
        input_empty = True
        for (xi, wi) in zip(x, w):
            len_xi = len(xi)
            if wi is not None and len(wi) != len_xi:
                raise ValueError('weights should have the same shape as x')
            if len_xi:
                input_empty = False
        if color is None:
            colors = [self._get_lines.get_next_color() for i in range(nx)]
        else:
            colors = mcolors.to_rgba_array(color)
            if len(colors) != nx:
                raise ValueError(f"The 'color' keyword argument must have one color per dataset, but {nx} datasets and {len(colors)} colors were provided")
        hist_kwargs = dict()
        if bin_range is None:
            xmin = np.inf
            xmax = -np.inf
            for xi in x:
                if len(xi):
                    xmin = min(xmin, np.nanmin(xi))
                    xmax = max(xmax, np.nanmax(xi))
            if xmin <= xmax:
                bin_range = (xmin, xmax)
        if not input_empty and len(x) > 1:
            if weights is not None:
                _w = np.concatenate(w)
            else:
                _w = None
            bins = np.histogram_bin_edges(np.concatenate(x), bins, bin_range, _w)
        else:
            hist_kwargs['range'] = bin_range
        density = bool(density)
        if density and (not stacked):
            hist_kwargs['density'] = density
        tops = []
        for i in range(nx):
            (m, bins) = np.histogram(x[i], bins, weights=w[i], **hist_kwargs)
            tops.append(m)
        tops = np.array(tops, float)
        bins = np.array(bins, float)
        if stacked:
            tops = tops.cumsum(axis=0)
            if density:
                tops = tops / np.diff(bins) / tops[-1].sum()
        if cumulative:
            slc = slice(None)
            if isinstance(cumulative, Number) and cumulative < 0:
                slc = slice(None, None, -1)
            if density:
                tops = (tops * np.diff(bins))[:, slc].cumsum(axis=1)[:, slc]
            else:
                tops = tops[:, slc].cumsum(axis=1)[:, slc]
        patches = []
        if histtype.startswith('bar'):
            totwidth = np.diff(bins)
            if rwidth is not None:
                dr = np.clip(rwidth, 0, 1)
            elif len(tops) > 1 and (not stacked or mpl.rcParams['_internal.classic_mode']):
                dr = 0.8
            else:
                dr = 1.0
            if histtype == 'bar' and (not stacked):
                width = dr * totwidth / nx
                dw = width
                boffset = -0.5 * dr * totwidth * (1 - 1 / nx)
            elif histtype == 'barstacked' or stacked:
                width = dr * totwidth
                (boffset, dw) = (0.0, 0.0)
            if align == 'mid':
                boffset += 0.5 * totwidth
            elif align == 'right':
                boffset += totwidth
            if orientation == 'horizontal':
                _barfunc = self.barh
                bottom_kwarg = 'left'
            else:
                _barfunc = self.bar
                bottom_kwarg = 'bottom'
            for (top, color) in zip(tops, colors):
                if bottom is None:
                    bottom = np.zeros(len(top))
                if stacked:
                    height = top - bottom
                else:
                    height = top
                bars = _barfunc(bins[:-1] + boffset, height, width, align='center', log=log, color=color, **{bottom_kwarg: bottom})
                patches.append(bars)
                if stacked:
                    bottom = top
                boffset += dw
            for bars in patches[1:]:
                for patch in bars:
                    patch.sticky_edges.x[:] = patch.sticky_edges.y[:] = []
        elif histtype.startswith('step'):
            x = np.zeros(4 * len(bins) - 3)
            y = np.zeros(4 * len(bins) - 3)
            (x[0:2 * len(bins) - 1:2], x[1:2 * len(bins) - 1:2]) = (bins, bins[:-1])
            x[2 * len(bins) - 1:] = x[1:2 * len(bins) - 1][::-1]
            if bottom is None:
                bottom = 0
            y[1:2 * len(bins) - 1:2] = y[2:2 * len(bins):2] = bottom
            y[2 * len(bins) - 1:] = y[1:2 * len(bins) - 1][::-1]
            if log:
                if orientation == 'horizontal':
                    self.set_xscale('log', nonpositive='clip')
                else:
                    self.set_yscale('log', nonpositive='clip')
            if align == 'left':
                x -= 0.5 * (bins[1] - bins[0])
            elif align == 'right':
                x += 0.5 * (bins[1] - bins[0])
            fill = histtype == 'stepfilled'
            (xvals, yvals) = ([], [])
            for top in tops:
                if stacked:
                    y[2 * len(bins) - 1:] = y[1:2 * len(bins) - 1][::-1]
                y[1:2 * len(bins) - 1:2] = y[2:2 * len(bins):2] = top + bottom
                y[0] = y[-1]
                if orientation == 'horizontal':
                    xvals.append(y.copy())
                    yvals.append(x.copy())
                else:
                    xvals.append(x.copy())
                    yvals.append(y.copy())
            split = -1 if fill else 2 * len(bins)
            for (x, y, color) in reversed(list(zip(xvals, yvals, colors))):
                patches.append(self.fill(x[:split], y[:split], closed=True if fill else None, facecolor=color, edgecolor=None if fill else color, fill=fill if fill else None, zorder=None if fill else mlines.Line2D.zorder))
            for patch_list in patches:
                for patch in patch_list:
                    if orientation == 'vertical':
                        patch.sticky_edges.y.append(0)
                    elif orientation == 'horizontal':
                        patch.sticky_edges.x.append(0)
            patches.reverse()
        labels = [] if label is None else np.atleast_1d(np.asarray(label, str))
        for (patch, lbl) in itertools.zip_longest(patches, labels):
            if patch:
                p = patch[0]
                p._internal_update(kwargs)
                if lbl is not None:
                    p.set_label(lbl)
                for p in patch[1:]:
                    p._internal_update(kwargs)
                    p.set_label('_nolegend_')
        if nx == 1:
            return (tops[0], bins, patches[0])
        else:
            patch_type = 'BarContainer' if histtype.startswith('bar') else 'list[Polygon]'
            return (tops, bins, cbook.silent_list(patch_type, patches))

    @_preprocess_data()
    def stairs(self, values, edges=None, *, orientation='vertical', baseline=0, fill=False, **kwargs):
        if False:
            return 10
        "\n        A stepwise constant function as a line with bounding edges\n        or a filled plot.\n\n        Parameters\n        ----------\n        values : array-like\n            The step heights.\n\n        edges : array-like\n            The edge positions, with ``len(edges) == len(vals) + 1``,\n            between which the curve takes on vals values.\n\n        orientation : {'vertical', 'horizontal'}, default: 'vertical'\n            The direction of the steps. Vertical means that *values* are along\n            the y-axis, and edges are along the x-axis.\n\n        baseline : float, array-like or None, default: 0\n            The bottom value of the bounding edges or when\n            ``fill=True``, position of lower edge. If *fill* is\n            True or an array is passed to *baseline*, a closed\n            path is drawn.\n\n        fill : bool, default: False\n            Whether the area under the step curve should be filled.\n\n        Returns\n        -------\n        StepPatch : `~matplotlib.patches.StepPatch`\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            `~matplotlib.patches.StepPatch` properties\n\n        "
        if 'color' in kwargs:
            _color = kwargs.pop('color')
        else:
            _color = self._get_lines.get_next_color()
        if fill:
            kwargs.setdefault('linewidth', 0)
            kwargs.setdefault('facecolor', _color)
        else:
            kwargs.setdefault('edgecolor', _color)
        if edges is None:
            edges = np.arange(len(values) + 1)
        (edges, values, baseline) = self._process_unit_info([('x', edges), ('y', values), ('y', baseline)], kwargs)
        patch = mpatches.StepPatch(values, edges, baseline=baseline, orientation=orientation, fill=fill, **kwargs)
        self.add_patch(patch)
        if baseline is None:
            baseline = 0
        if orientation == 'vertical':
            patch.sticky_edges.y.append(np.min(baseline))
            self.update_datalim([(edges[0], np.min(baseline))])
        else:
            patch.sticky_edges.x.append(np.min(baseline))
            self.update_datalim([(np.min(baseline), edges[0])])
        self._request_autoscale_view()
        return patch

    @_preprocess_data(replace_names=['x', 'y', 'weights'])
    @_docstring.dedent_interpd
    def hist2d(self, x, y, bins=10, range=None, density=False, weights=None, cmin=None, cmax=None, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Make a 2D histogram plot.\n\n        Parameters\n        ----------\n        x, y : array-like, shape (n, )\n            Input values\n\n        bins : None or int or [int, int] or array-like or [array, array]\n\n            The bin specification:\n\n            - If int, the number of bins for the two dimensions\n              (``nx = ny = bins``).\n            - If ``[int, int]``, the number of bins in each dimension\n              (``nx, ny = bins``).\n            - If array-like, the bin edges for the two dimensions\n              (``x_edges = y_edges = bins``).\n            - If ``[array, array]``, the bin edges in each dimension\n              (``x_edges, y_edges = bins``).\n\n            The default value is 10.\n\n        range : array-like shape(2, 2), optional\n            The leftmost and rightmost edges of the bins along each dimension\n            (if not specified explicitly in the bins parameters): ``[[xmin,\n            xmax], [ymin, ymax]]``. All values outside of this range will be\n            considered outliers and not tallied in the histogram.\n\n        density : bool, default: False\n            Normalize histogram.  See the documentation for the *density*\n            parameter of `~.Axes.hist` for more details.\n\n        weights : array-like, shape (n, ), optional\n            An array of values w_i weighing each sample (x_i, y_i).\n\n        cmin, cmax : float, default: None\n            All bins that has count less than *cmin* or more than *cmax* will not be\n            displayed (set to NaN before passing to `~.Axes.pcolormesh`) and these count\n            values in the return value count histogram will also be set to nan upon\n            return.\n\n        Returns\n        -------\n        h : 2D array\n            The bi-dimensional histogram of samples x and y. Values in x are\n            histogrammed along the first dimension and values in y are\n            histogrammed along the second dimension.\n        xedges : 1D array\n            The bin edges along the x-axis.\n        yedges : 1D array\n            The bin edges along the y-axis.\n        image : `~.matplotlib.collections.QuadMesh`\n\n        Other Parameters\n        ----------------\n        %(cmap_doc)s\n\n        %(norm_doc)s\n\n        %(vmin_vmax_doc)s\n\n        alpha : ``0 <= scalar <= 1`` or ``None``, optional\n            The alpha blending value.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Additional parameters are passed along to the\n            `~.Axes.pcolormesh` method and `~matplotlib.collections.QuadMesh`\n            constructor.\n\n        See Also\n        --------\n        hist : 1D histogram plotting\n        hexbin : 2D histogram with hexagonal bins\n\n        Notes\n        -----\n        - Currently ``hist2d`` calculates its own axis limits, and any limits\n          previously set are ignored.\n        - Rendering the histogram with a logarithmic color scale is\n          accomplished by passing a `.colors.LogNorm` instance to the *norm*\n          keyword argument. Likewise, power-law normalization (similar\n          in effect to gamma correction) can be accomplished with\n          `.colors.PowerNorm`.\n        '
        (h, xedges, yedges) = np.histogram2d(x, y, bins=bins, range=range, density=density, weights=weights)
        if cmin is not None:
            h[h < cmin] = None
        if cmax is not None:
            h[h > cmax] = None
        pc = self.pcolormesh(xedges, yedges, h.T, **kwargs)
        self.set_xlim(xedges[0], xedges[-1])
        self.set_ylim(yedges[0], yedges[-1])
        return (h, xedges, yedges, pc)

    @_preprocess_data(replace_names=['x', 'weights'], label_namer='x')
    @_docstring.dedent_interpd
    def ecdf(self, x, weights=None, *, complementary=False, orientation='vertical', compress=False, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Compute and plot the empirical cumulative distribution function of *x*.\n\n        .. versionadded:: 3.8\n\n        Parameters\n        ----------\n        x : 1d array-like\n            The input data.  Infinite entries are kept (and move the relevant\n            end of the ecdf from 0/1), but NaNs and masked values are errors.\n\n        weights : 1d array-like or None, default: None\n            The weights of the entries; must have the same shape as *x*.\n            Weights corresponding to NaN data points are dropped, and then the\n            remaining weights are normalized to sum to 1.  If unset, all\n            entries have the same weight.\n\n        complementary : bool, default: False\n            Whether to plot a cumulative distribution function, which increases\n            from 0 to 1 (the default), or a complementary cumulative\n            distribution function, which decreases from 1 to 0.\n\n        orientation : {"vertical", "horizontal"}, default: "vertical"\n            Whether the entries are plotted along the x-axis ("vertical", the\n            default) or the y-axis ("horizontal").  This parameter takes the\n            same values as in `~.Axes.hist`.\n\n        compress : bool, default: False\n            Whether multiple entries with the same values are grouped together\n            (with a summed weight) before plotting.  This is mainly useful if\n            *x* contains many identical data points, to decrease the rendering\n            complexity of the plot. If *x* contains no duplicate points, this\n            has no effect and just uses some time and memory.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        Returns\n        -------\n        `.Line2D`\n\n        Notes\n        -----\n        The ecdf plot can be thought of as a cumulative histogram with one bin\n        per data entry; i.e. it reports on the entire dataset without any\n        arbitrary binning.\n\n        If *x* contains NaNs or masked entries, either remove them first from\n        the array (if they should not taken into account), or replace them by\n        -inf or +inf (if they should be sorted at the beginning or the end of\n        the array).\n        '
        _api.check_in_list(['horizontal', 'vertical'], orientation=orientation)
        if 'drawstyle' in kwargs or 'ds' in kwargs:
            raise TypeError("Cannot pass 'drawstyle' or 'ds' to ecdf()")
        if np.ma.getmask(x).any():
            raise ValueError('ecdf() does not support masked entries')
        x = np.asarray(x)
        if np.isnan(x).any():
            raise ValueError('ecdf() does not support NaNs')
        argsort = np.argsort(x)
        x = x[argsort]
        if weights is None:
            cum_weights = (1 + np.arange(len(x))) / len(x)
        else:
            weights = np.take(weights, argsort)
            cum_weights = np.cumsum(weights / np.sum(weights))
        if compress:
            compress_idxs = [0, *(x[:-1] != x[1:]).nonzero()[0] + 1]
            x = x[compress_idxs]
            cum_weights = cum_weights[compress_idxs]
        if orientation == 'vertical':
            if not complementary:
                (line,) = self.plot([x[0], *x], [0, *cum_weights], drawstyle='steps-post', **kwargs)
            else:
                (line,) = self.plot([*x, x[-1]], [1, *1 - cum_weights], drawstyle='steps-pre', **kwargs)
            line.sticky_edges.y[:] = [0, 1]
        else:
            if not complementary:
                (line,) = self.plot([0, *cum_weights], [x[0], *x], drawstyle='steps-pre', **kwargs)
            else:
                (line,) = self.plot([1, *1 - cum_weights], [*x, x[-1]], drawstyle='steps-post', **kwargs)
            line.sticky_edges.x[:] = [0, 1]
        return line

    @_preprocess_data(replace_names=['x'])
    @_docstring.dedent_interpd
    def psd(self, x, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None, return_line=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Plot the power spectral density.\n\n        The power spectral density :math:`P_{xx}` by Welch's average\n        periodogram method.  The vector *x* is divided into *NFFT* length\n        segments.  Each segment is detrended by function *detrend* and\n        windowed by function *window*.  *noverlap* gives the length of\n        the overlap between segments.  The :math:`|\\mathrm{fft}(i)|^2`\n        of each segment :math:`i` are averaged to compute :math:`P_{xx}`,\n        with a scaling to correct for power loss due to windowing.\n\n        If len(*x*) < *NFFT*, it will be zero padded to *NFFT*.\n\n        Parameters\n        ----------\n        x : 1-D array or sequence\n            Array or sequence containing the data\n\n        %(Spectral)s\n\n        %(PSD)s\n\n        noverlap : int, default: 0 (no overlap)\n            The number of points of overlap between segments.\n\n        Fc : int, default: 0\n            The center frequency of *x*, which offsets the x extents of the\n            plot to reflect the frequency range used when a signal is acquired\n            and then filtered and downsampled to baseband.\n\n        return_line : bool, default: False\n            Whether to include the line object plotted in the returned values.\n\n        Returns\n        -------\n        Pxx : 1-D array\n            The values for the power spectrum :math:`P_{xx}` before scaling\n            (real valued).\n\n        freqs : 1-D array\n            The frequencies corresponding to the elements in *Pxx*.\n\n        line : `~matplotlib.lines.Line2D`\n            The line created by this function.\n            Only returned if *return_line* is True.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        specgram\n            Differs in the default overlap; in not returning the mean of the\n            segment periodograms; in returning the times of the segments; and\n            in plotting a colormap instead of a line.\n        magnitude_spectrum\n            Plots the magnitude spectrum.\n        csd\n            Plots the spectral density between two signals.\n\n        Notes\n        -----\n        For plotting, the power is plotted as\n        :math:`10\\log_{10}(P_{xx})` for decibels, though *Pxx* itself\n        is returned.\n\n        References\n        ----------\n        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,\n        John Wiley & Sons (1986)\n        "
        if Fc is None:
            Fc = 0
        (pxx, freqs) = mlab.psd(x=x, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
        freqs += Fc
        if scale_by_freq in (None, True):
            psd_units = 'dB/Hz'
        else:
            psd_units = 'dB'
        line = self.plot(freqs, 10 * np.log10(pxx), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Power Spectral Density (%s)' % psd_units)
        self.grid(True)
        (vmin, vmax) = self.get_ybound()
        step = max(10 * int(np.log10(vmax - vmin)), 1)
        ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
        self.set_yticks(ticks)
        if return_line is None or not return_line:
            return (pxx, freqs)
        else:
            return (pxx, freqs, line)

    @_preprocess_data(replace_names=['x', 'y'], label_namer='y')
    @_docstring.dedent_interpd
    def csd(self, x, y, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, noverlap=None, pad_to=None, sides=None, scale_by_freq=None, return_line=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Plot the cross-spectral density.\n\n        The cross spectral density :math:`P_{xy}` by Welch's average\n        periodogram method.  The vectors *x* and *y* are divided into\n        *NFFT* length segments.  Each segment is detrended by function\n        *detrend* and windowed by function *window*.  *noverlap* gives\n        the length of the overlap between segments.  The product of\n        the direct FFTs of *x* and *y* are averaged over each segment\n        to compute :math:`P_{xy}`, with a scaling to correct for power\n        loss due to windowing.\n\n        If len(*x*) < *NFFT* or len(*y*) < *NFFT*, they will be zero\n        padded to *NFFT*.\n\n        Parameters\n        ----------\n        x, y : 1-D arrays or sequences\n            Arrays or sequences containing the data.\n\n        %(Spectral)s\n\n        %(PSD)s\n\n        noverlap : int, default: 0 (no overlap)\n            The number of points of overlap between segments.\n\n        Fc : int, default: 0\n            The center frequency of *x*, which offsets the x extents of the\n            plot to reflect the frequency range used when a signal is acquired\n            and then filtered and downsampled to baseband.\n\n        return_line : bool, default: False\n            Whether to include the line object plotted in the returned values.\n\n        Returns\n        -------\n        Pxy : 1-D array\n            The values for the cross spectrum :math:`P_{xy}` before scaling\n            (complex valued).\n\n        freqs : 1-D array\n            The frequencies corresponding to the elements in *Pxy*.\n\n        line : `~matplotlib.lines.Line2D`\n            The line created by this function.\n            Only returned if *return_line* is True.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        psd : is equivalent to setting ``y = x``.\n\n        Notes\n        -----\n        For plotting, the power is plotted as\n        :math:`10 \\log_{10}(P_{xy})` for decibels, though :math:`P_{xy}` itself\n        is returned.\n\n        References\n        ----------\n        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,\n        John Wiley & Sons (1986)\n        "
        if Fc is None:
            Fc = 0
        (pxy, freqs) = mlab.csd(x=x, y=y, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq)
        freqs += Fc
        line = self.plot(freqs, 10 * np.log10(np.abs(pxy)), **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Cross Spectrum Magnitude (dB)')
        self.grid(True)
        (vmin, vmax) = self.get_ybound()
        step = max(10 * int(np.log10(vmax - vmin)), 1)
        ticks = np.arange(math.floor(vmin), math.ceil(vmax) + 1, step)
        self.set_yticks(ticks)
        if return_line is None or not return_line:
            return (pxy, freqs)
        else:
            return (pxy, freqs, line)

    @_preprocess_data(replace_names=['x'])
    @_docstring.dedent_interpd
    def magnitude_spectrum(self, x, Fs=None, Fc=None, window=None, pad_to=None, sides=None, scale=None, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Plot the magnitude spectrum.\n\n        Compute the magnitude spectrum of *x*.  Data is padded to a\n        length of *pad_to* and the windowing function *window* is applied to\n        the signal.\n\n        Parameters\n        ----------\n        x : 1-D array or sequence\n            Array or sequence containing the data.\n\n        %(Spectral)s\n\n        %(Single_Spectrum)s\n\n        scale : {'default', 'linear', 'dB'}\n            The scaling of the values in the *spec*.  'linear' is no scaling.\n            'dB' returns the values in dB scale, i.e., the dB amplitude\n            (20 * log10). 'default' is 'linear'.\n\n        Fc : int, default: 0\n            The center frequency of *x*, which offsets the x extents of the\n            plot to reflect the frequency range used when a signal is acquired\n            and then filtered and downsampled to baseband.\n\n        Returns\n        -------\n        spectrum : 1-D array\n            The values for the magnitude spectrum before scaling (real valued).\n\n        freqs : 1-D array\n            The frequencies corresponding to the elements in *spectrum*.\n\n        line : `~matplotlib.lines.Line2D`\n            The line created by this function.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        psd\n            Plots the power spectral density.\n        angle_spectrum\n            Plots the angles of the corresponding frequencies.\n        phase_spectrum\n            Plots the phase (unwrapped angle) of the corresponding frequencies.\n        specgram\n            Can plot the magnitude spectrum of segments within the signal in a\n            colormap.\n        "
        if Fc is None:
            Fc = 0
        (spec, freqs) = mlab.magnitude_spectrum(x=x, Fs=Fs, window=window, pad_to=pad_to, sides=sides)
        freqs += Fc
        yunits = _api.check_getitem({None: 'energy', 'default': 'energy', 'linear': 'energy', 'dB': 'dB'}, scale=scale)
        if yunits == 'energy':
            Z = spec
        else:
            Z = 20.0 * np.log10(spec)
        (line,) = self.plot(freqs, Z, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Magnitude (%s)' % yunits)
        return (spec, freqs, line)

    @_preprocess_data(replace_names=['x'])
    @_docstring.dedent_interpd
    def angle_spectrum(self, x, Fs=None, Fc=None, window=None, pad_to=None, sides=None, **kwargs):
        if False:
            i = 10
            return i + 15
        '\n        Plot the angle spectrum.\n\n        Compute the angle spectrum (wrapped phase spectrum) of *x*.\n        Data is padded to a length of *pad_to* and the windowing function\n        *window* is applied to the signal.\n\n        Parameters\n        ----------\n        x : 1-D array or sequence\n            Array or sequence containing the data.\n\n        %(Spectral)s\n\n        %(Single_Spectrum)s\n\n        Fc : int, default: 0\n            The center frequency of *x*, which offsets the x extents of the\n            plot to reflect the frequency range used when a signal is acquired\n            and then filtered and downsampled to baseband.\n\n        Returns\n        -------\n        spectrum : 1-D array\n            The values for the angle spectrum in radians (real valued).\n\n        freqs : 1-D array\n            The frequencies corresponding to the elements in *spectrum*.\n\n        line : `~matplotlib.lines.Line2D`\n            The line created by this function.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        magnitude_spectrum\n            Plots the magnitudes of the corresponding frequencies.\n        phase_spectrum\n            Plots the unwrapped version of this function.\n        specgram\n            Can plot the angle spectrum of segments within the signal in a\n            colormap.\n        '
        if Fc is None:
            Fc = 0
        (spec, freqs) = mlab.angle_spectrum(x=x, Fs=Fs, window=window, pad_to=pad_to, sides=sides)
        freqs += Fc
        lines = self.plot(freqs, spec, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Angle (radians)')
        return (spec, freqs, lines[0])

    @_preprocess_data(replace_names=['x'])
    @_docstring.dedent_interpd
    def phase_spectrum(self, x, Fs=None, Fc=None, window=None, pad_to=None, sides=None, **kwargs):
        if False:
            print('Hello World!')
        '\n        Plot the phase spectrum.\n\n        Compute the phase spectrum (unwrapped angle spectrum) of *x*.\n        Data is padded to a length of *pad_to* and the windowing function\n        *window* is applied to the signal.\n\n        Parameters\n        ----------\n        x : 1-D array or sequence\n            Array or sequence containing the data\n\n        %(Spectral)s\n\n        %(Single_Spectrum)s\n\n        Fc : int, default: 0\n            The center frequency of *x*, which offsets the x extents of the\n            plot to reflect the frequency range used when a signal is acquired\n            and then filtered and downsampled to baseband.\n\n        Returns\n        -------\n        spectrum : 1-D array\n            The values for the phase spectrum in radians (real valued).\n\n        freqs : 1-D array\n            The frequencies corresponding to the elements in *spectrum*.\n\n        line : `~matplotlib.lines.Line2D`\n            The line created by this function.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        See Also\n        --------\n        magnitude_spectrum\n            Plots the magnitudes of the corresponding frequencies.\n        angle_spectrum\n            Plots the wrapped version of this function.\n        specgram\n            Can plot the phase spectrum of segments within the signal in a\n            colormap.\n        '
        if Fc is None:
            Fc = 0
        (spec, freqs) = mlab.phase_spectrum(x=x, Fs=Fs, window=window, pad_to=pad_to, sides=sides)
        freqs += Fc
        lines = self.plot(freqs, spec, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Phase (radians)')
        return (spec, freqs, lines[0])

    @_preprocess_data(replace_names=['x', 'y'])
    @_docstring.dedent_interpd
    def cohere(self, x, y, NFFT=256, Fs=2, Fc=0, detrend=mlab.detrend_none, window=mlab.window_hanning, noverlap=0, pad_to=None, sides='default', scale_by_freq=None, **kwargs):
        if False:
            return 10
        '\n        Plot the coherence between *x* and *y*.\n\n        Coherence is the normalized cross spectral density:\n\n        .. math::\n\n          C_{xy} = \\frac{|P_{xy}|^2}{P_{xx}P_{yy}}\n\n        Parameters\n        ----------\n        %(Spectral)s\n\n        %(PSD)s\n\n        noverlap : int, default: 0 (no overlap)\n            The number of points of overlap between blocks.\n\n        Fc : int, default: 0\n            The center frequency of *x*, which offsets the x extents of the\n            plot to reflect the frequency range used when a signal is acquired\n            and then filtered and downsampled to baseband.\n\n        Returns\n        -------\n        Cxy : 1-D array\n            The coherence vector.\n\n        freqs : 1-D array\n            The frequencies for the elements in *Cxy*.\n\n        Other Parameters\n        ----------------\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Keyword arguments control the `.Line2D` properties:\n\n            %(Line2D:kwdoc)s\n\n        References\n        ----------\n        Bendat & Piersol -- Random Data: Analysis and Measurement Procedures,\n        John Wiley & Sons (1986)\n        '
        (cxy, freqs) = mlab.cohere(x=x, y=y, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap, scale_by_freq=scale_by_freq, sides=sides, pad_to=pad_to)
        freqs += Fc
        self.plot(freqs, cxy, **kwargs)
        self.set_xlabel('Frequency')
        self.set_ylabel('Coherence')
        self.grid(True)
        return (cxy, freqs)

    @_preprocess_data(replace_names=['x'])
    @_docstring.dedent_interpd
    def specgram(self, x, NFFT=None, Fs=None, Fc=None, detrend=None, window=None, noverlap=None, cmap=None, xextent=None, pad_to=None, sides=None, scale_by_freq=None, mode=None, scale=None, vmin=None, vmax=None, **kwargs):
        if False:
            while True:
                i = 10
        "\n        Plot a spectrogram.\n\n        Compute and plot a spectrogram of data in *x*.  Data are split into\n        *NFFT* length segments and the spectrum of each section is\n        computed.  The windowing function *window* is applied to each\n        segment, and the amount of overlap of each segment is\n        specified with *noverlap*. The spectrogram is plotted as a colormap\n        (using imshow).\n\n        Parameters\n        ----------\n        x : 1-D array or sequence\n            Array or sequence containing the data.\n\n        %(Spectral)s\n\n        %(PSD)s\n\n        mode : {'default', 'psd', 'magnitude', 'angle', 'phase'}\n            What sort of spectrum to use.  Default is 'psd', which takes the\n            power spectral density.  'magnitude' returns the magnitude\n            spectrum.  'angle' returns the phase spectrum without unwrapping.\n            'phase' returns the phase spectrum with unwrapping.\n\n        noverlap : int, default: 128\n            The number of points of overlap between blocks.\n\n        scale : {'default', 'linear', 'dB'}\n            The scaling of the values in the *spec*.  'linear' is no scaling.\n            'dB' returns the values in dB scale.  When *mode* is 'psd',\n            this is dB power (10 * log10).  Otherwise, this is dB amplitude\n            (20 * log10). 'default' is 'dB' if *mode* is 'psd' or\n            'magnitude' and 'linear' otherwise.  This must be 'linear'\n            if *mode* is 'angle' or 'phase'.\n\n        Fc : int, default: 0\n            The center frequency of *x*, which offsets the x extents of the\n            plot to reflect the frequency range used when a signal is acquired\n            and then filtered and downsampled to baseband.\n\n        cmap : `.Colormap`, default: :rc:`image.cmap`\n\n        xextent : *None* or (xmin, xmax)\n            The image extent along the x-axis. The default sets *xmin* to the\n            left border of the first bin (*spectrum* column) and *xmax* to the\n            right border of the last bin. Note that for *noverlap>0* the width\n            of the bins is smaller than those of the segments.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        **kwargs\n            Additional keyword arguments are passed on to `~.axes.Axes.imshow`\n            which makes the specgram image. The origin keyword argument\n            is not supported.\n\n        Returns\n        -------\n        spectrum : 2D array\n            Columns are the periodograms of successive segments.\n\n        freqs : 1-D array\n            The frequencies corresponding to the rows in *spectrum*.\n\n        t : 1-D array\n            The times corresponding to midpoints of segments (i.e., the columns\n            in *spectrum*).\n\n        im : `.AxesImage`\n            The image created by imshow containing the spectrogram.\n\n        See Also\n        --------\n        psd\n            Differs in the default overlap; in returning the mean of the\n            segment periodograms; in not returning times; and in generating a\n            line plot instead of colormap.\n        magnitude_spectrum\n            A single spectrum, similar to having a single segment when *mode*\n            is 'magnitude'. Plots a line instead of a colormap.\n        angle_spectrum\n            A single spectrum, similar to having a single segment when *mode*\n            is 'angle'. Plots a line instead of a colormap.\n        phase_spectrum\n            A single spectrum, similar to having a single segment when *mode*\n            is 'phase'. Plots a line instead of a colormap.\n\n        Notes\n        -----\n        The parameters *detrend* and *scale_by_freq* do only apply when *mode*\n        is set to 'psd'.\n        "
        if NFFT is None:
            NFFT = 256
        if Fc is None:
            Fc = 0
        if noverlap is None:
            noverlap = 128
        if Fs is None:
            Fs = 2
        if mode == 'complex':
            raise ValueError('Cannot plot a complex specgram')
        if scale is None or scale == 'default':
            if mode in ['angle', 'phase']:
                scale = 'linear'
            else:
                scale = 'dB'
        elif mode in ['angle', 'phase'] and scale == 'dB':
            raise ValueError('Cannot use dB scale with angle or phase mode')
        (spec, freqs, t) = mlab.specgram(x=x, NFFT=NFFT, Fs=Fs, detrend=detrend, window=window, noverlap=noverlap, pad_to=pad_to, sides=sides, scale_by_freq=scale_by_freq, mode=mode)
        if scale == 'linear':
            Z = spec
        elif scale == 'dB':
            if mode is None or mode == 'default' or mode == 'psd':
                Z = 10.0 * np.log10(spec)
            else:
                Z = 20.0 * np.log10(spec)
        else:
            raise ValueError(f'Unknown scale {scale!r}')
        Z = np.flipud(Z)
        if xextent is None:
            pad_xextent = (NFFT - noverlap) / Fs / 2
            xextent = (np.min(t) - pad_xextent, np.max(t) + pad_xextent)
        (xmin, xmax) = xextent
        freqs += Fc
        extent = (xmin, xmax, freqs[0], freqs[-1])
        if 'origin' in kwargs:
            raise _api.kwarg_error('specgram', 'origin')
        im = self.imshow(Z, cmap, extent=extent, vmin=vmin, vmax=vmax, origin='upper', **kwargs)
        self.axis('auto')
        return (spec, freqs, t, im)

    @_docstring.dedent_interpd
    def spy(self, Z, precision=0, marker=None, markersize=None, aspect='equal', origin='upper', **kwargs):
        if False:
            for i in range(10):
                print('nop')
        "\n        Plot the sparsity pattern of a 2D array.\n\n        This visualizes the non-zero values of the array.\n\n        Two plotting styles are available: image and marker. Both\n        are available for full arrays, but only the marker style\n        works for `scipy.sparse.spmatrix` instances.\n\n        **Image style**\n\n        If *marker* and *markersize* are *None*, `~.Axes.imshow` is used. Any\n        extra remaining keyword arguments are passed to this method.\n\n        **Marker style**\n\n        If *Z* is a `scipy.sparse.spmatrix` or *marker* or *markersize* are\n        *None*, a `.Line2D` object will be returned with the value of marker\n        determining the marker type, and any remaining keyword arguments\n        passed to `~.Axes.plot`.\n\n        Parameters\n        ----------\n        Z : (M, N) array-like\n            The array to be plotted.\n\n        precision : float or 'present', default: 0\n            If *precision* is 0, any non-zero value will be plotted. Otherwise,\n            values of :math:`|Z| > precision` will be plotted.\n\n            For `scipy.sparse.spmatrix` instances, you can also\n            pass 'present'. In this case any value present in the array\n            will be plotted, even if it is identically zero.\n\n        aspect : {'equal', 'auto', None} or float, default: 'equal'\n            The aspect ratio of the Axes.  This parameter is particularly\n            relevant for images since it determines whether data pixels are\n            square.\n\n            This parameter is a shortcut for explicitly calling\n            `.Axes.set_aspect`. See there for further details.\n\n            - 'equal': Ensures an aspect ratio of 1. Pixels will be square.\n            - 'auto': The Axes is kept fixed and the aspect is adjusted so\n              that the data fit in the Axes. In general, this will result in\n              non-square pixels.\n            - *None*: Use :rc:`image.aspect`.\n\n        origin : {'upper', 'lower'}, default: :rc:`image.origin`\n            Place the [0, 0] index of the array in the upper left or lower left\n            corner of the Axes. The convention 'upper' is typically used for\n            matrices and images.\n\n        Returns\n        -------\n        `~matplotlib.image.AxesImage` or `.Line2D`\n            The return type depends on the plotting style (see above).\n\n        Other Parameters\n        ----------------\n        **kwargs\n            The supported additional parameters depend on the plotting style.\n\n            For the image style, you can pass the following additional\n            parameters of `~.Axes.imshow`:\n\n            - *cmap*\n            - *alpha*\n            - *url*\n            - any `.Artist` properties (passed on to the `.AxesImage`)\n\n            For the marker style, you can pass any `.Line2D` property except\n            for *linestyle*:\n\n            %(Line2D:kwdoc)s\n        "
        if marker is None and markersize is None and hasattr(Z, 'tocoo'):
            marker = 's'
        _api.check_in_list(['upper', 'lower'], origin=origin)
        if marker is None and markersize is None:
            Z = np.asarray(Z)
            mask = np.abs(Z) > precision
            if 'cmap' not in kwargs:
                kwargs['cmap'] = mcolors.ListedColormap(['w', 'k'], name='binary')
            if 'interpolation' in kwargs:
                raise _api.kwarg_error('spy', 'interpolation')
            if 'norm' not in kwargs:
                kwargs['norm'] = mcolors.NoNorm()
            ret = self.imshow(mask, interpolation='nearest', aspect=aspect, origin=origin, **kwargs)
        else:
            if hasattr(Z, 'tocoo'):
                c = Z.tocoo()
                if precision == 'present':
                    y = c.row
                    x = c.col
                else:
                    nonzero = np.abs(c.data) > precision
                    y = c.row[nonzero]
                    x = c.col[nonzero]
            else:
                Z = np.asarray(Z)
                nonzero = np.abs(Z) > precision
                (y, x) = np.nonzero(nonzero)
            if marker is None:
                marker = 's'
            if markersize is None:
                markersize = 10
            if 'linestyle' in kwargs:
                raise _api.kwarg_error('spy', 'linestyle')
            ret = mlines.Line2D(x, y, linestyle='None', marker=marker, markersize=markersize, **kwargs)
            self.add_line(ret)
            (nr, nc) = Z.shape
            self.set_xlim(-0.5, nc - 0.5)
            if origin == 'upper':
                self.set_ylim(nr - 0.5, -0.5)
            else:
                self.set_ylim(-0.5, nr - 0.5)
            self.set_aspect(aspect)
        self.title.set_y(1.05)
        if origin == 'upper':
            self.xaxis.tick_top()
        else:
            self.xaxis.tick_bottom()
        self.xaxis.set_ticks_position('both')
        self.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        self.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        return ret

    def matshow(self, Z, **kwargs):
        if False:
            i = 10
            return i + 15
        "\n        Plot the values of a 2D matrix or array as color-coded image.\n\n        The matrix will be shown the way it would be printed, with the first\n        row at the top.  Row and column numbering is zero-based.\n\n        Parameters\n        ----------\n        Z : (M, N) array-like\n            The matrix to be displayed.\n\n        Returns\n        -------\n        `~matplotlib.image.AxesImage`\n\n        Other Parameters\n        ----------------\n        **kwargs : `~matplotlib.axes.Axes.imshow` arguments\n\n        See Also\n        --------\n        imshow : More general function to plot data on a 2D regular raster.\n\n        Notes\n        -----\n        This is just a convenience function wrapping `.imshow` to set useful\n        defaults for displaying a matrix. In particular:\n\n        - Set ``origin='upper'``.\n        - Set ``interpolation='nearest'``.\n        - Set ``aspect='equal'``.\n        - Ticks are placed to the left and above.\n        - Ticks are formatted to show integer indices.\n\n        "
        Z = np.asanyarray(Z)
        kw = {'origin': 'upper', 'interpolation': 'nearest', 'aspect': 'equal', **kwargs}
        im = self.imshow(Z, **kw)
        self.title.set_y(1.05)
        self.xaxis.tick_top()
        self.xaxis.set_ticks_position('both')
        self.xaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        self.yaxis.set_major_locator(mticker.MaxNLocator(nbins=9, steps=[1, 2, 5, 10], integer=True))
        return im

    @_preprocess_data(replace_names=['dataset'])
    def violinplot(self, dataset, positions=None, vert=True, widths=0.5, showmeans=False, showextrema=True, showmedians=False, quantiles=None, points=100, bw_method=None):
        if False:
            print('Hello World!')
        "\n        Make a violin plot.\n\n        Make a violin plot for each column of *dataset* or each vector in\n        sequence *dataset*.  Each filled area extends to represent the\n        entire data range, with optional lines at the mean, the median,\n        the minimum, the maximum, and user-specified quantiles.\n\n        Parameters\n        ----------\n        dataset : Array or a sequence of vectors.\n          The input data.\n\n        positions : array-like, default: [1, 2, ..., n]\n          The positions of the violins. The ticks and limits are\n          automatically set to match the positions.\n\n        vert : bool, default: True.\n          If true, creates a vertical violin plot.\n          Otherwise, creates a horizontal violin plot.\n\n        widths : array-like, default: 0.5\n          Either a scalar or a vector that sets the maximal width of\n          each violin. The default is 0.5, which uses about half of the\n          available horizontal space.\n\n        showmeans : bool, default: False\n          If `True`, will toggle rendering of the means.\n\n        showextrema : bool, default: True\n          If `True`, will toggle rendering of the extrema.\n\n        showmedians : bool, default: False\n          If `True`, will toggle rendering of the medians.\n\n        quantiles : array-like, default: None\n          If not None, set a list of floats in interval [0, 1] for each violin,\n          which stands for the quantiles that will be rendered for that\n          violin.\n\n        points : int, default: 100\n          Defines the number of points to evaluate each of the\n          gaussian kernel density estimations at.\n\n        bw_method : str, scalar or callable, optional\n          The method used to calculate the estimator bandwidth.  This can be\n          'scott', 'silverman', a scalar constant or a callable.  If a\n          scalar, this will be used directly as `kde.factor`.  If a\n          callable, it should take a `matplotlib.mlab.GaussianKDE` instance as\n          its only parameter and return a scalar. If None (default), 'scott'\n          is used.\n\n        data : indexable object, optional\n            DATA_PARAMETER_PLACEHOLDER\n\n        Returns\n        -------\n        dict\n          A dictionary mapping each component of the violinplot to a\n          list of the corresponding collection instances created. The\n          dictionary has the following keys:\n\n          - ``bodies``: A list of the `~.collections.PolyCollection`\n            instances containing the filled area of each violin.\n\n          - ``cmeans``: A `~.collections.LineCollection` instance that marks\n            the mean values of each of the violin's distribution.\n\n          - ``cmins``: A `~.collections.LineCollection` instance that marks\n            the bottom of each violin's distribution.\n\n          - ``cmaxes``: A `~.collections.LineCollection` instance that marks\n            the top of each violin's distribution.\n\n          - ``cbars``: A `~.collections.LineCollection` instance that marks\n            the centers of each violin's distribution.\n\n          - ``cmedians``: A `~.collections.LineCollection` instance that\n            marks the median values of each of the violin's distribution.\n\n          - ``cquantiles``: A `~.collections.LineCollection` instance created\n            to identify the quantile values of each of the violin's\n            distribution.\n\n        "

        def _kde_method(X, coords):
            if False:
                return 10
            X = cbook._unpack_to_numpy(X)
            if np.all(X[0] == X):
                return (X[0] == coords).astype(float)
            kde = mlab.GaussianKDE(X, bw_method)
            return kde.evaluate(coords)
        vpstats = cbook.violin_stats(dataset, _kde_method, points=points, quantiles=quantiles)
        return self.violin(vpstats, positions=positions, vert=vert, widths=widths, showmeans=showmeans, showextrema=showextrema, showmedians=showmedians)

    def violin(self, vpstats, positions=None, vert=True, widths=0.5, showmeans=False, showextrema=True, showmedians=False):
        if False:
            while True:
                i = 10
        "\n        Drawing function for violin plots.\n\n        Draw a violin plot for each column of *vpstats*. Each filled area\n        extends to represent the entire data range, with optional lines at the\n        mean, the median, the minimum, the maximum, and the quantiles values.\n\n        Parameters\n        ----------\n        vpstats : list of dicts\n          A list of dictionaries containing stats for each violin plot.\n          Required keys are:\n\n          - ``coords``: A list of scalars containing the coordinates that\n            the violin's kernel density estimate were evaluated at.\n\n          - ``vals``: A list of scalars containing the values of the\n            kernel density estimate at each of the coordinates given\n            in *coords*.\n\n          - ``mean``: The mean value for this violin's dataset.\n\n          - ``median``: The median value for this violin's dataset.\n\n          - ``min``: The minimum value for this violin's dataset.\n\n          - ``max``: The maximum value for this violin's dataset.\n\n          Optional keys are:\n\n          - ``quantiles``: A list of scalars containing the quantile values\n            for this violin's dataset.\n\n        positions : array-like, default: [1, 2, ..., n]\n          The positions of the violins. The ticks and limits are\n          automatically set to match the positions.\n\n        vert : bool, default: True.\n          If true, plots the violins vertically.\n          Otherwise, plots the violins horizontally.\n\n        widths : array-like, default: 0.5\n          Either a scalar or a vector that sets the maximal width of\n          each violin. The default is 0.5, which uses about half of the\n          available horizontal space.\n\n        showmeans : bool, default: False\n          If true, will toggle rendering of the means.\n\n        showextrema : bool, default: True\n          If true, will toggle rendering of the extrema.\n\n        showmedians : bool, default: False\n          If true, will toggle rendering of the medians.\n\n        Returns\n        -------\n        dict\n          A dictionary mapping each component of the violinplot to a\n          list of the corresponding collection instances created. The\n          dictionary has the following keys:\n\n          - ``bodies``: A list of the `~.collections.PolyCollection`\n            instances containing the filled area of each violin.\n\n          - ``cmeans``: A `~.collections.LineCollection` instance that marks\n            the mean values of each of the violin's distribution.\n\n          - ``cmins``: A `~.collections.LineCollection` instance that marks\n            the bottom of each violin's distribution.\n\n          - ``cmaxes``: A `~.collections.LineCollection` instance that marks\n            the top of each violin's distribution.\n\n          - ``cbars``: A `~.collections.LineCollection` instance that marks\n            the centers of each violin's distribution.\n\n          - ``cmedians``: A `~.collections.LineCollection` instance that\n            marks the median values of each of the violin's distribution.\n\n          - ``cquantiles``: A `~.collections.LineCollection` instance created\n            to identify the quantiles values of each of the violin's\n            distribution.\n        "
        means = []
        mins = []
        maxes = []
        medians = []
        quantiles = []
        qlens = []
        artists = {}
        N = len(vpstats)
        datashape_message = 'List of violinplot statistics and `{0}` values must have the same length'
        if positions is None:
            positions = range(1, N + 1)
        elif len(positions) != N:
            raise ValueError(datashape_message.format('positions'))
        if np.isscalar(widths):
            widths = [widths] * N
        elif len(widths) != N:
            raise ValueError(datashape_message.format('widths'))
        line_ends = [[-0.25], [0.25]] * np.array(widths) + positions
        if mpl.rcParams['_internal.classic_mode']:
            fillcolor = 'y'
            linecolor = 'r'
        else:
            fillcolor = linecolor = self._get_lines.get_next_color()
        if vert:
            fill = self.fill_betweenx
            perp_lines = functools.partial(self.hlines, colors=linecolor)
            par_lines = functools.partial(self.vlines, colors=linecolor)
        else:
            fill = self.fill_between
            perp_lines = functools.partial(self.vlines, colors=linecolor)
            par_lines = functools.partial(self.hlines, colors=linecolor)
        bodies = []
        for (stats, pos, width) in zip(vpstats, positions, widths):
            vals = np.array(stats['vals'])
            vals = 0.5 * width * vals / vals.max()
            bodies += [fill(stats['coords'], -vals + pos, vals + pos, facecolor=fillcolor, alpha=0.3)]
            means.append(stats['mean'])
            mins.append(stats['min'])
            maxes.append(stats['max'])
            medians.append(stats['median'])
            q = stats.get('quantiles')
            if q is None:
                q = []
            quantiles.extend(q)
            qlens.append(len(q))
        artists['bodies'] = bodies
        if showmeans:
            artists['cmeans'] = perp_lines(means, *line_ends)
        if showextrema:
            artists['cmaxes'] = perp_lines(maxes, *line_ends)
            artists['cmins'] = perp_lines(mins, *line_ends)
            artists['cbars'] = par_lines(positions, mins, maxes)
        if showmedians:
            artists['cmedians'] = perp_lines(medians, *line_ends)
        if quantiles:
            artists['cquantiles'] = perp_lines(quantiles, *np.repeat(line_ends, qlens, axis=1))
        return artists
    table = mtable.table
    stackplot = _preprocess_data()(mstack.stackplot)
    streamplot = _preprocess_data(replace_names=['x', 'y', 'u', 'v', 'start_points'])(mstream.streamplot)
    tricontour = mtri.tricontour
    tricontourf = mtri.tricontourf
    tripcolor = mtri.tripcolor
    triplot = mtri.triplot

    def _get_aspect_ratio(self):
        if False:
            while True:
                i = 10
        '\n        Convenience method to calculate the aspect ratio of the axes in\n        the display coordinate system.\n        '
        figure_size = self.get_figure().get_size_inches()
        (ll, ur) = self.get_position() * figure_size
        (width, height) = ur - ll
        return height / (width * self.get_data_ratio())