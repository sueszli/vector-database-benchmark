import tkinter
from io import BytesIO
from . import Image
_pilbitmap_ok = None

def _pilbitmap_check():
    if False:
        for i in range(10):
            print('nop')
    global _pilbitmap_ok
    if _pilbitmap_ok is None:
        try:
            im = Image.new('1', (1, 1))
            tkinter.BitmapImage(data=f'PIL:{im.im.id}')
            _pilbitmap_ok = 1
        except tkinter.TclError:
            _pilbitmap_ok = 0
    return _pilbitmap_ok

def _get_image_from_kw(kw):
    if False:
        while True:
            i = 10
    source = None
    if 'file' in kw:
        source = kw.pop('file')
    elif 'data' in kw:
        source = BytesIO(kw.pop('data'))
    if source:
        return Image.open(source)

def _pyimagingtkcall(command, photo, id):
    if False:
        print('Hello World!')
    tk = photo.tk
    try:
        tk.call(command, photo, id)
    except tkinter.TclError:
        from . import _imagingtk
        _imagingtk.tkinit(tk.interpaddr())
        tk.call(command, photo, id)

class PhotoImage:
    """
    A Tkinter-compatible photo image.  This can be used
    everywhere Tkinter expects an image object.  If the image is an RGBA
    image, pixels having alpha 0 are treated as transparent.

    The constructor takes either a PIL image, or a mode and a size.
    Alternatively, you can use the ``file`` or ``data`` options to initialize
    the photo image object.

    :param image: Either a PIL image, or a mode string.  If a mode string is
                  used, a size must also be given.
    :param size: If the first argument is a mode string, this defines the size
                 of the image.
    :keyword file: A filename to load the image from (using
                   ``Image.open(file)``).
    :keyword data: An 8-bit string containing image data (as loaded from an
                   image file).
    """

    def __init__(self, image=None, size=None, **kw):
        if False:
            while True:
                i = 10
        if image is None:
            image = _get_image_from_kw(kw)
        if hasattr(image, 'mode') and hasattr(image, 'size'):
            mode = image.mode
            if mode == 'P':
                image.apply_transparency()
                image.load()
                try:
                    mode = image.palette.mode
                except AttributeError:
                    mode = 'RGB'
            size = image.size
            (kw['width'], kw['height']) = size
        else:
            mode = image
            image = None
        if mode not in ['1', 'L', 'RGB', 'RGBA']:
            mode = Image.getmodebase(mode)
        self.__mode = mode
        self.__size = size
        self.__photo = tkinter.PhotoImage(**kw)
        self.tk = self.__photo.tk
        if image:
            self.paste(image)

    def __del__(self):
        if False:
            i = 10
            return i + 15
        name = self.__photo.name
        self.__photo.name = None
        try:
            self.__photo.tk.call('image', 'delete', name)
        except Exception:
            pass

    def __str__(self):
        if False:
            return 10
        '\n        Get the Tkinter photo image identifier.  This method is automatically\n        called by Tkinter whenever a PhotoImage object is passed to a Tkinter\n        method.\n\n        :return: A Tkinter photo image identifier (a string).\n        '
        return str(self.__photo)

    def width(self):
        if False:
            return 10
        '\n        Get the width of the image.\n\n        :return: The width, in pixels.\n        '
        return self.__size[0]

    def height(self):
        if False:
            while True:
                i = 10
        '\n        Get the height of the image.\n\n        :return: The height, in pixels.\n        '
        return self.__size[1]

    def paste(self, im):
        if False:
            i = 10
            return i + 15
        '\n        Paste a PIL image into the photo image.  Note that this can\n        be very slow if the photo image is displayed.\n\n        :param im: A PIL image. The size must match the target region.  If the\n                   mode does not match, the image is converted to the mode of\n                   the bitmap image.\n        '
        im.load()
        image = im.im
        if image.isblock() and im.mode == self.__mode:
            block = image
        else:
            block = image.new_block(self.__mode, im.size)
            image.convert2(block, image)
        _pyimagingtkcall('PyImagingPhoto', self.__photo, block.id)

class BitmapImage:
    """
    A Tkinter-compatible bitmap image.  This can be used everywhere Tkinter
    expects an image object.

    The given image must have mode "1".  Pixels having value 0 are treated as
    transparent.  Options, if any, are passed on to Tkinter.  The most commonly
    used option is ``foreground``, which is used to specify the color for the
    non-transparent parts.  See the Tkinter documentation for information on
    how to specify colours.

    :param image: A PIL image.
    """

    def __init__(self, image=None, **kw):
        if False:
            for i in range(10):
                print('nop')
        if image is None:
            image = _get_image_from_kw(kw)
        self.__mode = image.mode
        self.__size = image.size
        if _pilbitmap_check():
            image.load()
            kw['data'] = f'PIL:{image.im.id}'
            self.__im = image
        else:
            kw['data'] = image.tobitmap()
        self.__photo = tkinter.BitmapImage(**kw)

    def __del__(self):
        if False:
            while True:
                i = 10
        name = self.__photo.name
        self.__photo.name = None
        try:
            self.__photo.tk.call('image', 'delete', name)
        except Exception:
            pass

    def width(self):
        if False:
            print('Hello World!')
        '\n        Get the width of the image.\n\n        :return: The width, in pixels.\n        '
        return self.__size[0]

    def height(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the height of the image.\n\n        :return: The height, in pixels.\n        '
        return self.__size[1]

    def __str__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Get the Tkinter bitmap image identifier.  This method is automatically\n        called by Tkinter whenever a BitmapImage object is passed to a Tkinter\n        method.\n\n        :return: A Tkinter bitmap image identifier (a string).\n        '
        return str(self.__photo)

def getimage(photo):
    if False:
        return 10
    'Copies the contents of a PhotoImage to a PIL image memory.'
    im = Image.new('RGBA', (photo.width(), photo.height()))
    block = im.im
    _pyimagingtkcall('PyImagingPhotoGet', photo, block.id)
    return im

def _show(image, title):
    if False:
        while True:
            i = 10
    'Helper for the Image.show method.'

    class UI(tkinter.Label):

        def __init__(self, master, im):
            if False:
                return 10
            if im.mode == '1':
                self.image = BitmapImage(im, foreground='white', master=master)
            else:
                self.image = PhotoImage(im, master=master)
            super().__init__(master, image=self.image, bg='black', bd=0)
    if not tkinter._default_root:
        msg = 'tkinter not initialized'
        raise OSError(msg)
    top = tkinter.Toplevel()
    if title:
        top.title(title)
    UI(top, image).pack()