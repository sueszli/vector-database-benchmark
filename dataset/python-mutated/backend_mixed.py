import numpy as np
from matplotlib import cbook
from .backend_agg import RendererAgg
from matplotlib._tight_bbox import process_figure_for_rasterizing

class MixedModeRenderer:
    """
    A helper class to implement a renderer that switches between
    vector and raster drawing.  An example may be a PDF writer, where
    most things are drawn with PDF vector commands, but some very
    complex objects, such as quad meshes, are rasterised and then
    output as images.
    """

    def __init__(self, figure, width, height, dpi, vector_renderer, raster_renderer_class=None, bbox_inches_restore=None):
        if False:
            return 10
        '\n        Parameters\n        ----------\n        figure : `~matplotlib.figure.Figure`\n            The figure instance.\n        width : scalar\n            The width of the canvas in logical units\n        height : scalar\n            The height of the canvas in logical units\n        dpi : float\n            The dpi of the canvas\n        vector_renderer : `~matplotlib.backend_bases.RendererBase`\n            An instance of a subclass of\n            `~matplotlib.backend_bases.RendererBase` that will be used for the\n            vector drawing.\n        raster_renderer_class : `~matplotlib.backend_bases.RendererBase`\n            The renderer class to use for the raster drawing.  If not provided,\n            this will use the Agg backend (which is currently the only viable\n            option anyway.)\n\n        '
        if raster_renderer_class is None:
            raster_renderer_class = RendererAgg
        self._raster_renderer_class = raster_renderer_class
        self._width = width
        self._height = height
        self.dpi = dpi
        self._vector_renderer = vector_renderer
        self._raster_renderer = None
        self.figure = figure
        self._figdpi = figure.dpi
        self._bbox_inches_restore = bbox_inches_restore
        self._renderer = vector_renderer

    def __getattr__(self, attr):
        if False:
            while True:
                i = 10
        return getattr(self._renderer, attr)

    def start_rasterizing(self):
        if False:
            i = 10
            return i + 15
        '\n        Enter "raster" mode.  All subsequent drawing commands (until\n        `stop_rasterizing` is called) will be drawn with the raster backend.\n        '
        self.figure.dpi = self.dpi
        if self._bbox_inches_restore:
            r = process_figure_for_rasterizing(self.figure, self._bbox_inches_restore)
            self._bbox_inches_restore = r
        self._raster_renderer = self._raster_renderer_class(self._width * self.dpi, self._height * self.dpi, self.dpi)
        self._renderer = self._raster_renderer

    def stop_rasterizing(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Exit "raster" mode.  All of the drawing that was done since\n        the last `start_rasterizing` call will be copied to the\n        vector backend by calling draw_image.\n        '
        self._renderer = self._vector_renderer
        height = self._height * self.dpi
        img = np.asarray(self._raster_renderer.buffer_rgba())
        (slice_y, slice_x) = cbook._get_nonzero_slices(img[..., 3])
        cropped_img = img[slice_y, slice_x]
        if cropped_img.size:
            gc = self._renderer.new_gc()
            self._renderer.draw_image(gc, slice_x.start * self._figdpi / self.dpi, (height - slice_y.stop) * self._figdpi / self.dpi, cropped_img[::-1])
        self._raster_renderer = None
        self.figure.dpi = self._figdpi
        if self._bbox_inches_restore:
            r = process_figure_for_rasterizing(self.figure, self._bbox_inches_restore, self._figdpi)
            self._bbox_inches_restore = r