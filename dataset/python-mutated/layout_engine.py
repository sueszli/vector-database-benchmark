"""
Classes to layout elements in a `.Figure`.

Figures have a ``layout_engine`` property that holds a subclass of
`~.LayoutEngine` defined here (or *None* for no layout).  At draw time
``figure.get_layout_engine().execute()`` is called, the goal of which is
usually to rearrange Axes on the figure to produce a pleasing layout. This is
like a ``draw`` callback but with two differences.  First, when printing we
disable the layout engine for the final draw. Second, it is useful to know the
layout engine while the figure is being created.  In particular, colorbars are
made differently with different layout engines (for historical reasons).

Matplotlib supplies two layout engines, `.TightLayoutEngine` and
`.ConstrainedLayoutEngine`.  Third parties can create their own layout engine
by subclassing `.LayoutEngine`.
"""
from contextlib import nullcontext
import matplotlib as mpl
from matplotlib._constrained_layout import do_constrained_layout
from matplotlib._tight_layout import get_subplotspec_list, get_tight_layout_figure

class LayoutEngine:
    """
    Base class for Matplotlib layout engines.

    A layout engine can be passed to a figure at instantiation or at any time
    with `~.figure.Figure.set_layout_engine`.  Once attached to a figure, the
    layout engine ``execute`` function is called at draw time by
    `~.figure.Figure.draw`, providing a special draw-time hook.

    .. note::

       However, note that layout engines affect the creation of colorbars, so
       `~.figure.Figure.set_layout_engine` should be called before any
       colorbars are created.

    Currently, there are two properties of `LayoutEngine` classes that are
    consulted while manipulating the figure:

    - ``engine.colorbar_gridspec`` tells `.Figure.colorbar` whether to make the
       axes using the gridspec method (see `.colorbar.make_axes_gridspec`) or
       not (see `.colorbar.make_axes`);
    - ``engine.adjust_compatible`` stops `.Figure.subplots_adjust` from being
        run if it is not compatible with the layout engine.

    To implement a custom `LayoutEngine`:

    1. override ``_adjust_compatible`` and ``_colorbar_gridspec``
    2. override `LayoutEngine.set` to update *self._params*
    3. override `LayoutEngine.execute` with your implementation

    """
    _adjust_compatible = None
    _colorbar_gridspec = None

    def __init__(self, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self._params = {}

    def set(self, **kwargs):
        if False:
            return 10
        '\n        Set the parameters for the layout engine.\n        '
        raise NotImplementedError

    @property
    def colorbar_gridspec(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a boolean if the layout engine creates colorbars using a\n        gridspec.\n        '
        if self._colorbar_gridspec is None:
            raise NotImplementedError
        return self._colorbar_gridspec

    @property
    def adjust_compatible(self):
        if False:
            while True:
                i = 10
        '\n        Return a boolean if the layout engine is compatible with\n        `~.Figure.subplots_adjust`.\n        '
        if self._adjust_compatible is None:
            raise NotImplementedError
        return self._adjust_compatible

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return copy of the parameters for the layout engine.\n        '
        return dict(self._params)

    def execute(self, fig):
        if False:
            while True:
                i = 10
        '\n        Execute the layout on the figure given by *fig*.\n        '
        raise NotImplementedError

class PlaceHolderLayoutEngine(LayoutEngine):
    """
    This layout engine does not adjust the figure layout at all.

    The purpose of this `.LayoutEngine` is to act as a placeholder when the user removes
    a layout engine to ensure an incompatible `.LayoutEngine` cannot be set later.

    Parameters
    ----------
    adjust_compatible, colorbar_gridspec : bool
        Allow the PlaceHolderLayoutEngine to mirror the behavior of whatever
        layout engine it is replacing.

    """

    def __init__(self, adjust_compatible, colorbar_gridspec, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._adjust_compatible = adjust_compatible
        self._colorbar_gridspec = colorbar_gridspec
        super().__init__(**kwargs)

    def execute(self, fig):
        if False:
            for i in range(10):
                print('nop')
        '\n        Do nothing.\n        '
        return

class TightLayoutEngine(LayoutEngine):
    """
    Implements the ``tight_layout`` geometry management.  See
    :ref:`tight_layout_guide` for details.
    """
    _adjust_compatible = True
    _colorbar_gridspec = True

    def __init__(self, *, pad=1.08, h_pad=None, w_pad=None, rect=(0, 0, 1, 1), **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize tight_layout engine.\n\n        Parameters\n        ----------\n        pad : float, default: 1.08\n            Padding between the figure edge and the edges of subplots, as a\n            fraction of the font size.\n        h_pad, w_pad : float\n            Padding (height/width) between edges of adjacent subplots.\n            Defaults to *pad*.\n        rect : tuple (left, bottom, right, top), default: (0, 0, 1, 1).\n            rectangle in normalized figure coordinates that the subplots\n            (including labels) will fit into.\n        '
        super().__init__(**kwargs)
        for td in ['pad', 'h_pad', 'w_pad', 'rect']:
            self._params[td] = None
        self.set(pad=pad, h_pad=h_pad, w_pad=w_pad, rect=rect)

    def execute(self, fig):
        if False:
            return 10
        '\n        Execute tight_layout.\n\n        This decides the subplot parameters given the padding that\n        will allow the axes labels to not be covered by other labels\n        and axes.\n\n        Parameters\n        ----------\n        fig : `.Figure` to perform layout on.\n\n        See Also\n        --------\n        .figure.Figure.tight_layout\n        .pyplot.tight_layout\n        '
        info = self._params
        renderer = fig._get_renderer()
        with getattr(renderer, '_draw_disabled', nullcontext)():
            kwargs = get_tight_layout_figure(fig, fig.axes, get_subplotspec_list(fig.axes), renderer, pad=info['pad'], h_pad=info['h_pad'], w_pad=info['w_pad'], rect=info['rect'])
        if kwargs:
            fig.subplots_adjust(**kwargs)

    def set(self, *, pad=None, w_pad=None, h_pad=None, rect=None):
        if False:
            i = 10
            return i + 15
        '\n        Set the pads for tight_layout.\n\n        Parameters\n        ----------\n        pad : float\n            Padding between the figure edge and the edges of subplots, as a\n            fraction of the font size.\n        w_pad, h_pad : float\n            Padding (width/height) between edges of adjacent subplots.\n            Defaults to *pad*.\n        rect : tuple (left, bottom, right, top)\n            rectangle in normalized figure coordinates that the subplots\n            (including labels) will fit into.\n        '
        for td in self.set.__kwdefaults__:
            if locals()[td] is not None:
                self._params[td] = locals()[td]

class ConstrainedLayoutEngine(LayoutEngine):
    """
    Implements the ``constrained_layout`` geometry management.  See
    :ref:`constrainedlayout_guide` for details.
    """
    _adjust_compatible = False
    _colorbar_gridspec = False

    def __init__(self, *, h_pad=None, w_pad=None, hspace=None, wspace=None, rect=(0, 0, 1, 1), compress=False, **kwargs):
        if False:
            print('Hello World!')
        '\n        Initialize ``constrained_layout`` settings.\n\n        Parameters\n        ----------\n        h_pad, w_pad : float\n            Padding around the axes elements in inches.\n            Default to :rc:`figure.constrained_layout.h_pad` and\n            :rc:`figure.constrained_layout.w_pad`.\n        hspace, wspace : float\n            Fraction of the figure to dedicate to space between the\n            axes.  These are evenly spread between the gaps between the axes.\n            A value of 0.2 for a three-column layout would have a space\n            of 0.1 of the figure width between each column.\n            If h/wspace < h/w_pad, then the pads are used instead.\n            Default to :rc:`figure.constrained_layout.hspace` and\n            :rc:`figure.constrained_layout.wspace`.\n        rect : tuple of 4 floats\n            Rectangle in figure coordinates to perform constrained layout in\n            (left, bottom, width, height), each from 0-1.\n        compress : bool\n            Whether to shift Axes so that white space in between them is\n            removed. This is useful for simple grids of fixed-aspect Axes (e.g.\n            a grid of images).  See :ref:`compressed_layout`.\n        '
        super().__init__(**kwargs)
        self.set(w_pad=mpl.rcParams['figure.constrained_layout.w_pad'], h_pad=mpl.rcParams['figure.constrained_layout.h_pad'], wspace=mpl.rcParams['figure.constrained_layout.wspace'], hspace=mpl.rcParams['figure.constrained_layout.hspace'], rect=(0, 0, 1, 1))
        self.set(w_pad=w_pad, h_pad=h_pad, wspace=wspace, hspace=hspace, rect=rect)
        self._compress = compress

    def execute(self, fig):
        if False:
            return 10
        '\n        Perform constrained_layout and move and resize axes accordingly.\n\n        Parameters\n        ----------\n        fig : `.Figure` to perform layout on.\n        '
        (width, height) = fig.get_size_inches()
        w_pad = self._params['w_pad'] / width
        h_pad = self._params['h_pad'] / height
        return do_constrained_layout(fig, w_pad=w_pad, h_pad=h_pad, wspace=self._params['wspace'], hspace=self._params['hspace'], rect=self._params['rect'], compress=self._compress)

    def set(self, *, h_pad=None, w_pad=None, hspace=None, wspace=None, rect=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the pads for constrained_layout.\n\n        Parameters\n        ----------\n        h_pad, w_pad : float\n            Padding around the axes elements in inches.\n            Default to :rc:`figure.constrained_layout.h_pad` and\n            :rc:`figure.constrained_layout.w_pad`.\n        hspace, wspace : float\n            Fraction of the figure to dedicate to space between the\n            axes.  These are evenly spread between the gaps between the axes.\n            A value of 0.2 for a three-column layout would have a space\n            of 0.1 of the figure width between each column.\n            If h/wspace < h/w_pad, then the pads are used instead.\n            Default to :rc:`figure.constrained_layout.hspace` and\n            :rc:`figure.constrained_layout.wspace`.\n        rect : tuple of 4 floats\n            Rectangle in figure coordinates to perform constrained layout in\n            (left, bottom, width, height), each from 0-1.\n        '
        for td in self.set.__kwdefaults__:
            if locals()[td] is not None:
                self._params[td] = locals()[td]