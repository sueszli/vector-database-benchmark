"""
An `Anti-Grain Geometry`_ (AGG) backend.

Features that are implemented:

* capstyles and join styles
* dashes
* linewidth
* lines, rectangles, ellipses
* clipping to a rectangle
* output to RGBA and Pillow-supported image formats
* alpha blending
* DPI scaling properly - everything scales properly (dashes, linewidths, etc)
* draw polygon
* freetype2 w/ ft2font

Still TODO:

* integrate screen dpi w/ ppi and text

.. _Anti-Grain Geometry: http://agg.sourceforge.net/antigrain.com
"""
from contextlib import nullcontext
from math import radians, cos, sin
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
from matplotlib.backend_bases import _Backend, FigureCanvasBase, FigureManagerBase, RendererBase
from matplotlib.font_manager import fontManager as _fontManager, get_font
from matplotlib.ft2font import LOAD_FORCE_AUTOHINT, LOAD_NO_HINTING, LOAD_DEFAULT, LOAD_NO_AUTOHINT
from matplotlib.mathtext import MathTextParser
from matplotlib.path import Path
from matplotlib.transforms import Bbox, BboxBase
from matplotlib.backends._backend_agg import RendererAgg as _RendererAgg

def get_hinting_flag():
    if False:
        print('Hello World!')
    mapping = {'default': LOAD_DEFAULT, 'no_autohint': LOAD_NO_AUTOHINT, 'force_autohint': LOAD_FORCE_AUTOHINT, 'no_hinting': LOAD_NO_HINTING, True: LOAD_FORCE_AUTOHINT, False: LOAD_NO_HINTING, 'either': LOAD_DEFAULT, 'native': LOAD_NO_AUTOHINT, 'auto': LOAD_FORCE_AUTOHINT, 'none': LOAD_NO_HINTING}
    return mapping[mpl.rcParams['text.hinting']]

class RendererAgg(RendererBase):
    """
    The renderer handles all the drawing primitives using a graphics
    context instance that controls the colors/styles
    """

    def __init__(self, width, height, dpi):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.dpi = dpi
        self.width = width
        self.height = height
        self._renderer = _RendererAgg(int(width), int(height), dpi)
        self._filter_renderers = []
        self._update_methods()
        self.mathtext_parser = MathTextParser('agg')
        self.bbox = Bbox.from_bounds(0, 0, self.width, self.height)

    def __getstate__(self):
        if False:
            while True:
                i = 10
        return {'width': self.width, 'height': self.height, 'dpi': self.dpi}

    def __setstate__(self, state):
        if False:
            print('Hello World!')
        self.__init__(state['width'], state['height'], state['dpi'])

    def _update_methods(self):
        if False:
            print('Hello World!')
        self.draw_gouraud_triangles = self._renderer.draw_gouraud_triangles
        self.draw_image = self._renderer.draw_image
        self.draw_markers = self._renderer.draw_markers
        self.draw_path_collection = self._renderer.draw_path_collection
        self.draw_quad_mesh = self._renderer.draw_quad_mesh
        self.copy_from_bbox = self._renderer.copy_from_bbox

    def draw_path(self, gc, path, transform, rgbFace=None):
        if False:
            print('Hello World!')
        nmax = mpl.rcParams['agg.path.chunksize']
        npts = path.vertices.shape[0]
        if npts > nmax > 100 and path.should_simplify and (rgbFace is None) and (gc.get_hatch() is None):
            nch = np.ceil(npts / nmax)
            chsize = int(np.ceil(npts / nch))
            i0 = np.arange(0, npts, chsize)
            i1 = np.zeros_like(i0)
            i1[:-1] = i0[1:] - 1
            i1[-1] = npts
            for (ii0, ii1) in zip(i0, i1):
                v = path.vertices[ii0:ii1, :]
                c = path.codes
                if c is not None:
                    c = c[ii0:ii1]
                    c[0] = Path.MOVETO
                p = Path(v, c)
                p.simplify_threshold = path.simplify_threshold
                try:
                    self._renderer.draw_path(gc, p, transform, rgbFace)
                except OverflowError:
                    msg = f"Exceeded cell block limit in Agg.\n\nPlease reduce the value of rcParams['agg.path.chunksize'] (currently {nmax}) or increase the path simplification threshold(rcParams['path.simplify_threshold'] = {mpl.rcParams['path.simplify_threshold']:.2f} by default and path.simplify_threshold = {path.simplify_threshold:.2f} on the input)."
                    raise OverflowError(msg) from None
        else:
            try:
                self._renderer.draw_path(gc, path, transform, rgbFace)
            except OverflowError:
                cant_chunk = ''
                if rgbFace is not None:
                    cant_chunk += '- cannot split filled path\n'
                if gc.get_hatch() is not None:
                    cant_chunk += '- cannot split hatched path\n'
                if not path.should_simplify:
                    cant_chunk += '- path.should_simplify is False\n'
                if len(cant_chunk):
                    msg = f'Exceeded cell block limit in Agg, however for the following reasons:\n\n{cant_chunk}\nwe cannot automatically split up this path to draw.\n\nPlease manually simplify your path.'
                else:
                    inc_threshold = f"or increase the path simplification threshold(rcParams['path.simplify_threshold'] = {mpl.rcParams['path.simplify_threshold']} by default and path.simplify_threshold = {path.simplify_threshold} on the input)."
                    if nmax > 100:
                        msg = f"Exceeded cell block limit in Agg.  Please reduce the value of rcParams['agg.path.chunksize'] (currently {nmax}) {inc_threshold}"
                    else:
                        msg = f"Exceeded cell block limit in Agg.  Please set the value of rcParams['agg.path.chunksize'], (currently {nmax}) to be greater than 100 " + inc_threshold
                raise OverflowError(msg) from None

    def draw_mathtext(self, gc, x, y, s, prop, angle):
        if False:
            while True:
                i = 10
        'Draw mathtext using :mod:`matplotlib.mathtext`.'
        (ox, oy, width, height, descent, font_image) = self.mathtext_parser.parse(s, self.dpi, prop, antialiased=gc.get_antialiased())
        xd = descent * sin(radians(angle))
        yd = descent * cos(radians(angle))
        x = round(x + ox + xd)
        y = round(y - oy + yd)
        self._renderer.draw_text_image(font_image, x, y + 1, angle, gc)

    def draw_text(self, gc, x, y, s, prop, angle, ismath=False, mtext=None):
        if False:
            for i in range(10):
                print('nop')
        if ismath:
            return self.draw_mathtext(gc, x, y, s, prop, angle)
        font = self._prepare_font(prop)
        font.set_text(s, 0, flags=get_hinting_flag())
        font.draw_glyphs_to_bitmap(antialiased=gc.get_antialiased())
        d = font.get_descent() / 64.0
        (xo, yo) = font.get_bitmap_offset()
        xo /= 64.0
        yo /= 64.0
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xo + xd)
        y = round(y + yo + yd)
        self._renderer.draw_text_image(font, x, y + 1, angle, gc)

    def get_text_width_height_descent(self, s, prop, ismath):
        if False:
            print('Hello World!')
        _api.check_in_list(['TeX', True, False], ismath=ismath)
        if ismath == 'TeX':
            return super().get_text_width_height_descent(s, prop, ismath)
        if ismath:
            (ox, oy, width, height, descent, font_image) = self.mathtext_parser.parse(s, self.dpi, prop)
            return (width, height, descent)
        font = self._prepare_font(prop)
        font.set_text(s, 0.0, flags=get_hinting_flag())
        (w, h) = font.get_width_height()
        d = font.get_descent()
        w /= 64.0
        h /= 64.0
        d /= 64.0
        return (w, h, d)

    def draw_tex(self, gc, x, y, s, prop, angle, *, mtext=None):
        if False:
            for i in range(10):
                print('nop')
        size = prop.get_size_in_points()
        texmanager = self.get_texmanager()
        Z = texmanager.get_grey(s, size, self.dpi)
        Z = np.array(Z * 255.0, np.uint8)
        (w, h, d) = self.get_text_width_height_descent(s, prop, ismath='TeX')
        xd = d * sin(radians(angle))
        yd = d * cos(radians(angle))
        x = round(x + xd)
        y = round(y + yd)
        self._renderer.draw_text_image(Z, x, y, angle, gc)

    def get_canvas_width_height(self):
        if False:
            i = 10
            return i + 15
        return (self.width, self.height)

    def _prepare_font(self, font_prop):
        if False:
            return 10
        '\n        Get the `.FT2Font` for *font_prop*, clear its buffer, and set its size.\n        '
        font = get_font(_fontManager._find_fonts_by_props(font_prop))
        font.clear()
        size = font_prop.get_size_in_points()
        font.set_size(size, self.dpi)
        return font

    def points_to_pixels(self, points):
        if False:
            i = 10
            return i + 15
        return points * self.dpi / 72

    def buffer_rgba(self):
        if False:
            print('Hello World!')
        return memoryview(self._renderer)

    def tostring_argb(self):
        if False:
            for i in range(10):
                print('nop')
        return np.asarray(self._renderer).take([3, 0, 1, 2], axis=2).tobytes()

    @_api.deprecated('3.8', alternative='buffer_rgba')
    def tostring_rgb(self):
        if False:
            i = 10
            return i + 15
        return np.asarray(self._renderer).take([0, 1, 2], axis=2).tobytes()

    def clear(self):
        if False:
            i = 10
            return i + 15
        self._renderer.clear()

    def option_image_nocomposite(self):
        if False:
            return 10
        return True

    def option_scale_image(self):
        if False:
            for i in range(10):
                print('nop')
        return False

    def restore_region(self, region, bbox=None, xy=None):
        if False:
            while True:
                i = 10
        '\n        Restore the saved region. If bbox (instance of BboxBase, or\n        its extents) is given, only the region specified by the bbox\n        will be restored. *xy* (a pair of floats) optionally\n        specifies the new position (the LLC of the original region,\n        not the LLC of the bbox) where the region will be restored.\n\n        >>> region = renderer.copy_from_bbox()\n        >>> x1, y1, x2, y2 = region.get_extents()\n        >>> renderer.restore_region(region, bbox=(x1+dx, y1, x2, y2),\n        ...                         xy=(x1-dx, y1))\n\n        '
        if bbox is not None or xy is not None:
            if bbox is None:
                (x1, y1, x2, y2) = region.get_extents()
            elif isinstance(bbox, BboxBase):
                (x1, y1, x2, y2) = bbox.extents
            else:
                (x1, y1, x2, y2) = bbox
            if xy is None:
                (ox, oy) = (x1, y1)
            else:
                (ox, oy) = xy
            self._renderer.restore_region(region, int(x1), int(y1), int(x2), int(y2), int(ox), int(oy))
        else:
            self._renderer.restore_region(region)

    def start_filter(self):
        if False:
            print('Hello World!')
        '\n        Start filtering. It simply creates a new canvas (the old one is saved).\n        '
        self._filter_renderers.append(self._renderer)
        self._renderer = _RendererAgg(int(self.width), int(self.height), self.dpi)
        self._update_methods()

    def stop_filter(self, post_processing):
        if False:
            while True:
                i = 10
        '\n        Save the current canvas as an image and apply post processing.\n\n        The *post_processing* function::\n\n           def post_processing(image, dpi):\n             # ny, nx, depth = image.shape\n             # image (numpy array) has RGBA channels and has a depth of 4.\n             ...\n             # create a new_image (numpy array of 4 channels, size can be\n             # different). The resulting image may have offsets from\n             # lower-left corner of the original image\n             return new_image, offset_x, offset_y\n\n        The saved renderer is restored and the returned image from\n        post_processing is plotted (using draw_image) on it.\n        '
        orig_img = np.asarray(self.buffer_rgba())
        (slice_y, slice_x) = cbook._get_nonzero_slices(orig_img[..., 3])
        cropped_img = orig_img[slice_y, slice_x]
        self._renderer = self._filter_renderers.pop()
        self._update_methods()
        if cropped_img.size:
            (img, ox, oy) = post_processing(cropped_img / 255, self.dpi)
            gc = self.new_gc()
            if img.dtype.kind == 'f':
                img = np.asarray(img * 255.0, np.uint8)
            self._renderer.draw_image(gc, slice_x.start + ox, int(self.height) - slice_y.stop + oy, img[::-1])

class FigureCanvasAgg(FigureCanvasBase):
    _lastKey = None

    def copy_from_bbox(self, bbox):
        if False:
            i = 10
            return i + 15
        renderer = self.get_renderer()
        return renderer.copy_from_bbox(bbox)

    def restore_region(self, region, bbox=None, xy=None):
        if False:
            for i in range(10):
                print('nop')
        renderer = self.get_renderer()
        return renderer.restore_region(region, bbox, xy)

    def draw(self):
        if False:
            return 10
        self.renderer = self.get_renderer()
        self.renderer.clear()
        with self.toolbar._wait_cursor_for_draw_cm() if self.toolbar else nullcontext():
            self.figure.draw(self.renderer)
            super().draw()

    def get_renderer(self):
        if False:
            print('Hello World!')
        (w, h) = self.figure.bbox.size
        key = (w, h, self.figure.dpi)
        reuse_renderer = self._lastKey == key
        if not reuse_renderer:
            self.renderer = RendererAgg(w, h, self.figure.dpi)
            self._lastKey = key
        return self.renderer

    @_api.deprecated('3.8', alternative='buffer_rgba')
    def tostring_rgb(self):
        if False:
            while True:
                i = 10
        '\n        Get the image as RGB `bytes`.\n\n        `draw` must be called at least once before this function will work and\n        to update the renderer for any subsequent changes to the Figure.\n        '
        return self.renderer.tostring_rgb()

    def tostring_argb(self):
        if False:
            return 10
        '\n        Get the image as ARGB `bytes`.\n\n        `draw` must be called at least once before this function will work and\n        to update the renderer for any subsequent changes to the Figure.\n        '
        return self.renderer.tostring_argb()

    def buffer_rgba(self):
        if False:
            i = 10
            return i + 15
        "\n        Get the image as a `memoryview` to the renderer's buffer.\n\n        `draw` must be called at least once before this function will work and\n        to update the renderer for any subsequent changes to the Figure.\n        "
        return self.renderer.buffer_rgba()

    def print_raw(self, filename_or_obj, *, metadata=None):
        if False:
            while True:
                i = 10
        if metadata is not None:
            raise ValueError('metadata not supported for raw/rgba')
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        with cbook.open_file_cm(filename_or_obj, 'wb') as fh:
            fh.write(renderer.buffer_rgba())
    print_rgba = print_raw

    def _print_pil(self, filename_or_obj, fmt, pil_kwargs, metadata=None):
        if False:
            return 10
        '\n        Draw the canvas, then save it using `.image.imsave` (to which\n        *pil_kwargs* and *metadata* are forwarded).\n        '
        FigureCanvasAgg.draw(self)
        mpl.image.imsave(filename_or_obj, self.buffer_rgba(), format=fmt, origin='upper', dpi=self.figure.dpi, metadata=metadata, pil_kwargs=pil_kwargs)

    def print_png(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        if False:
            print('Hello World!')
        "\n        Write the figure to a PNG file.\n\n        Parameters\n        ----------\n        filename_or_obj : str or path-like or file-like\n            The file to write to.\n\n        metadata : dict, optional\n            Metadata in the PNG file as key-value pairs of bytes or latin-1\n            encodable strings.\n            According to the PNG specification, keys must be shorter than 79\n            chars.\n\n            The `PNG specification`_ defines some common keywords that may be\n            used as appropriate:\n\n            - Title: Short (one line) title or caption for image.\n            - Author: Name of image's creator.\n            - Description: Description of image (possibly long).\n            - Copyright: Copyright notice.\n            - Creation Time: Time of original image creation\n              (usually RFC 1123 format).\n            - Software: Software used to create the image.\n            - Disclaimer: Legal disclaimer.\n            - Warning: Warning of nature of content.\n            - Source: Device used to create the image.\n            - Comment: Miscellaneous comment;\n              conversion from other image format.\n\n            Other keywords may be invented for other purposes.\n\n            If 'Software' is not given, an autogenerated value for Matplotlib\n            will be used.  This can be removed by setting it to *None*.\n\n            For more details see the `PNG specification`_.\n\n            .. _PNG specification:                 https://www.w3.org/TR/2003/REC-PNG-20031110/#11keywords\n\n        pil_kwargs : dict, optional\n            Keyword arguments passed to `PIL.Image.Image.save`.\n\n            If the 'pnginfo' key is present, it completely overrides\n            *metadata*, including the default 'Software' key.\n        "
        self._print_pil(filename_or_obj, 'png', pil_kwargs, metadata)

    def print_to_buffer(self):
        if False:
            print('Hello World!')
        FigureCanvasAgg.draw(self)
        renderer = self.get_renderer()
        return (bytes(renderer.buffer_rgba()), (int(renderer.width), int(renderer.height)))

    def print_jpg(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        if False:
            return 10
        with mpl.rc_context({'savefig.facecolor': 'white'}):
            self._print_pil(filename_or_obj, 'jpeg', pil_kwargs, metadata)
    print_jpeg = print_jpg

    def print_tif(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        if False:
            i = 10
            return i + 15
        self._print_pil(filename_or_obj, 'tiff', pil_kwargs, metadata)
    print_tiff = print_tif

    def print_webp(self, filename_or_obj, *, metadata=None, pil_kwargs=None):
        if False:
            while True:
                i = 10
        self._print_pil(filename_or_obj, 'webp', pil_kwargs, metadata)
    (print_jpg.__doc__, print_tif.__doc__, print_webp.__doc__) = map('\n        Write the figure to a {} file.\n\n        Parameters\n        ----------\n        filename_or_obj : str or path-like or file-like\n            The file to write to.\n        pil_kwargs : dict, optional\n            Additional keyword arguments that are passed to\n            `PIL.Image.Image.save` when saving the figure.\n        '.format, ['JPEG', 'TIFF', 'WebP'])

@_Backend.export
class _BackendAgg(_Backend):
    backend_version = 'v2.2'
    FigureCanvas = FigureCanvasAgg
    FigureManager = FigureManagerBase