from . import Image

class HDC:
    """
    Wraps an HDC integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods.
    """

    def __init__(self, dc):
        if False:
            return 10
        self.dc = dc

    def __int__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.dc

class HWND:
    """
    Wraps an HWND integer. The resulting object can be passed to the
    :py:meth:`~PIL.ImageWin.Dib.draw` and :py:meth:`~PIL.ImageWin.Dib.expose`
    methods, instead of a DC.
    """

    def __init__(self, wnd):
        if False:
            return 10
        self.wnd = wnd

    def __int__(self):
        if False:
            print('Hello World!')
        return self.wnd

class Dib:
    """
    A Windows bitmap with the given mode and size.  The mode can be one of "1",
    "L", "P", or "RGB".

    If the display requires a palette, this constructor creates a suitable
    palette and associates it with the image. For an "L" image, 128 graylevels
    are allocated. For an "RGB" image, a 6x6x6 colour cube is used, together
    with 20 graylevels.

    To make sure that palettes work properly under Windows, you must call the
    ``palette`` method upon certain events from Windows.

    :param image: Either a PIL image, or a mode string. If a mode string is
                  used, a size must also be given.  The mode can be one of "1",
                  "L", "P", or "RGB".
    :param size: If the first argument is a mode string, this
                 defines the size of the image.
    """

    def __init__(self, image, size=None):
        if False:
            return 10
        if hasattr(image, 'mode') and hasattr(image, 'size'):
            mode = image.mode
            size = image.size
        else:
            mode = image
            image = None
        if mode not in ['1', 'L', 'P', 'RGB']:
            mode = Image.getmodebase(mode)
        self.image = Image.core.display(mode, size)
        self.mode = mode
        self.size = size
        if image:
            self.paste(image)

    def expose(self, handle):
        if False:
            print('Hello World!')
        '\n        Copy the bitmap contents to a device context.\n\n        :param handle: Device context (HDC), cast to a Python integer, or an\n                       HDC or HWND instance.  In PythonWin, you can use\n                       ``CDC.GetHandleAttrib()`` to get a suitable handle.\n        '
        if isinstance(handle, HWND):
            dc = self.image.getdc(handle)
            try:
                result = self.image.expose(dc)
            finally:
                self.image.releasedc(handle, dc)
        else:
            result = self.image.expose(handle)
        return result

    def draw(self, handle, dst, src=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Same as expose, but allows you to specify where to draw the image, and\n        what part of it to draw.\n\n        The destination and source areas are given as 4-tuple rectangles. If\n        the source is omitted, the entire image is copied. If the source and\n        the destination have different sizes, the image is resized as\n        necessary.\n        '
        if not src:
            src = (0, 0) + self.size
        if isinstance(handle, HWND):
            dc = self.image.getdc(handle)
            try:
                result = self.image.draw(dc, dst, src)
            finally:
                self.image.releasedc(handle, dc)
        else:
            result = self.image.draw(handle, dst, src)
        return result

    def query_palette(self, handle):
        if False:
            for i in range(10):
                print('nop')
        '\n        Installs the palette associated with the image in the given device\n        context.\n\n        This method should be called upon **QUERYNEWPALETTE** and\n        **PALETTECHANGED** events from Windows. If this method returns a\n        non-zero value, one or more display palette entries were changed, and\n        the image should be redrawn.\n\n        :param handle: Device context (HDC), cast to a Python integer, or an\n                       HDC or HWND instance.\n        :return: A true value if one or more entries were changed (this\n                 indicates that the image should be redrawn).\n        '
        if isinstance(handle, HWND):
            handle = self.image.getdc(handle)
            try:
                result = self.image.query_palette(handle)
            finally:
                self.image.releasedc(handle, handle)
        else:
            result = self.image.query_palette(handle)
        return result

    def paste(self, im, box=None):
        if False:
            return 10
        '\n        Paste a PIL image into the bitmap image.\n\n        :param im: A PIL image.  The size must match the target region.\n                   If the mode does not match, the image is converted to the\n                   mode of the bitmap image.\n        :param box: A 4-tuple defining the left, upper, right, and\n                    lower pixel coordinate.  See :ref:`coordinate-system`. If\n                    None is given instead of a tuple, all of the image is\n                    assumed.\n        '
        im.load()
        if self.mode != im.mode:
            im = im.convert(self.mode)
        if box:
            self.image.paste(im.im, box)
        else:
            self.image.paste(im.im)

    def frombytes(self, buffer):
        if False:
            print('Hello World!')
        '\n        Load display memory contents from byte data.\n\n        :param buffer: A buffer containing display data (usually\n                       data returned from :py:func:`~PIL.ImageWin.Dib.tobytes`)\n        '
        return self.image.frombytes(buffer)

    def tobytes(self):
        if False:
            return 10
        '\n        Copy display memory contents to bytes object.\n\n        :return: A bytes object containing display data.\n        '
        return self.image.tobytes()

class Window:
    """Create a Window with the given title size."""

    def __init__(self, title='PIL', width=None, height=None):
        if False:
            for i in range(10):
                print('nop')
        self.hwnd = Image.core.createwindow(title, self.__dispatcher, width or 0, height or 0)

    def __dispatcher(self, action, *args):
        if False:
            for i in range(10):
                print('nop')
        return getattr(self, 'ui_handle_' + action)(*args)

    def ui_handle_clear(self, dc, x0, y0, x1, y1):
        if False:
            while True:
                i = 10
        pass

    def ui_handle_damage(self, x0, y0, x1, y1):
        if False:
            while True:
                i = 10
        pass

    def ui_handle_destroy(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def ui_handle_repair(self, dc, x0, y0, x1, y1):
        if False:
            i = 10
            return i + 15
        pass

    def ui_handle_resize(self, width, height):
        if False:
            while True:
                i = 10
        pass

    def mainloop(self):
        if False:
            for i in range(10):
                print('nop')
        Image.core.eventloop()

class ImageWindow(Window):
    """Create an image window which displays the given image."""

    def __init__(self, image, title='PIL'):
        if False:
            while True:
                i = 10
        if not isinstance(image, Dib):
            image = Dib(image)
        self.image = image
        (width, height) = image.size
        super().__init__(title, width=width, height=height)

    def ui_handle_repair(self, dc, x0, y0, x1, y1):
        if False:
            i = 10
            return i + 15
        self.image.draw(dc, (x0, y0, x1, y1))