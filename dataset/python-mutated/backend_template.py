"""
A fully functional, do-nothing backend intended as a template for backend
writers.  It is fully functional in that you can select it as a backend e.g.
with ::

    import matplotlib
    matplotlib.use("template")

and your program will (should!) run without error, though no output is
produced.  This provides a starting point for backend writers; you can
selectively implement drawing methods (`~.RendererTemplate.draw_path`,
`~.RendererTemplate.draw_image`, etc.) and slowly see your figure come to life
instead having to have a full-blown implementation before getting any results.

Copy this file to a directory outside the Matplotlib source tree, somewhere
where Python can import it (by adding the directory to your ``sys.path`` or by
packaging it as a normal Python package); if the backend is importable as
``import my.backend`` you can then select it using ::

    import matplotlib
    matplotlib.use("module://my.backend")

If your backend implements support for saving figures (i.e. has a `print_xyz`
method), you can register it as the default handler for a given file type::

    from matplotlib.backend_bases import register_backend
    register_backend('xyz', 'my_backend', 'XYZ File Format')
    ...
    plt.savefig("figure.xyz")
"""
from matplotlib import _api
from matplotlib._pylab_helpers import Gcf
from matplotlib.backend_bases import FigureCanvasBase, FigureManagerBase, GraphicsContextBase, RendererBase
from matplotlib.figure import Figure

class RendererTemplate(RendererBase):
    """
    The renderer handles drawing/rendering operations.

    This is a minimal do-nothing class that can be used to get started when
    writing a new backend.  Refer to `.backend_bases.RendererBase` for
    documentation of the methods.
    """

    def __init__(self, dpi):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dpi = dpi

    def draw_path(self, gc, path, transform, rgbFace=None):
        if False:
            print('Hello World!')
        pass

    def draw_image(self, gc, x, y, im):
        if False:
            return 10
        pass

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if False:
            i = 10
            return i + 15
        pass

    def flipy(self):
        if False:
            print('Hello World!')
        return True

    def get_canvas_width_height(self):
        if False:
            i = 10
            return i + 15
        return (100, 100)

    def get_text_width_height_descent(self, s, prop, ismath):
        if False:
            while True:
                i = 10
        return (1, 1, 1)

    def new_gc(self):
        if False:
            i = 10
            return i + 15
        return GraphicsContextTemplate()

    def points_to_pixels(self, points):
        if False:
            for i in range(10):
                print('nop')
        return points

class GraphicsContextTemplate(GraphicsContextBase):
    """
    The graphics context provides the color, line styles, etc.  See the cairo
    and postscript backends for examples of mapping the graphics context
    attributes (cap styles, join styles, line widths, colors) to a particular
    backend.  In cairo this is done by wrapping a cairo.Context object and
    forwarding the appropriate calls to it using a dictionary mapping styles
    to gdk constants.  In Postscript, all the work is done by the renderer,
    mapping line styles to postscript calls.

    If it's more appropriate to do the mapping at the renderer level (as in
    the postscript backend), you don't need to override any of the GC methods.
    If it's more appropriate to wrap an instance (as in the cairo backend) and
    do the mapping here, you'll need to override several of the setter
    methods.

    The base GraphicsContext stores colors as an RGB tuple on the unit
    interval, e.g., (0.5, 0.0, 1.0). You may need to map this to colors
    appropriate for your backend.
    """

class FigureManagerTemplate(FigureManagerBase):
    """
    Helper class for pyplot mode, wraps everything up into a neat bundle.

    For non-interactive backends, the base class is sufficient.  For
    interactive backends, see the documentation of the `.FigureManagerBase`
    class for the list of methods that can/should be overridden.
    """

class FigureCanvasTemplate(FigureCanvasBase):
    """
    The canvas the figure renders into.  Calls the draw and print fig
    methods, creates the renderers, etc.

    Note: GUI templates will want to connect events for button presses,
    mouse movements and key presses to functions that call the base
    class methods button_press_event, button_release_event,
    motion_notify_event, key_press_event, and key_release_event.  See the
    implementations of the interactive backends for examples.

    Attributes
    ----------
    figure : `~matplotlib.figure.Figure`
        A high-level Figure instance
    """
    manager_class = FigureManagerTemplate

    def draw(self):
        if False:
            while True:
                i = 10
        '\n        Draw the figure using the renderer.\n\n        It is important that this method actually walk the artist tree\n        even if not output is produced because this will trigger\n        deferred work (like computing limits auto-limits and tick\n        values) that users may want access to before saving to disk.\n        '
        renderer = RendererTemplate(self.figure.dpi)
        self.figure.draw(renderer)
    filetypes = {**FigureCanvasBase.filetypes, 'foo': 'My magic Foo format'}

    def print_foo(self, filename, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        '\n        Write out format foo.\n\n        This method is normally called via `.Figure.savefig` and\n        `.FigureCanvasBase.print_figure`, which take care of setting the figure\n        facecolor, edgecolor, and dpi to the desired output values, and will\n        restore them to the original values.  Therefore, `print_foo` does not\n        need to handle these settings.\n        '
        self.draw()

    def get_default_filetype(self):
        if False:
            print('Hello World!')
        return 'foo'
FigureCanvas = FigureCanvasTemplate
FigureManager = FigureManagerTemplate