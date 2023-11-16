from .globject import GLObject
from .texture import Texture2D
from .wrappers import _check_valid, read_pixels
from .context import get_current_canvas

class RenderBuffer(GLObject):
    """Base class for render buffer object

    A render buffer can be in color, depth or stencil format. If this
    format is not specified, it is set when attached to the FrameBuffer.

    Parameters
    ----------
    shape : tuple
        The shape of the render buffer.
    format : {None, 'color', 'depth', 'stencil'}
        The format of the render buffer. See resize.
    resizeable : bool
        Indicates whether texture can be resized
    """
    _GLIR_TYPE = 'RenderBuffer'

    def __init__(self, shape=None, format=None, resizeable=True):
        if False:
            for i in range(10):
                print('nop')
        GLObject.__init__(self)
        self._format = None
        self._resizeable = True
        self.resize(shape, format)
        self._resizeable = bool(resizeable)

    @property
    def shape(self):
        if False:
            while True:
                i = 10
        'Render Buffer shape'
        return self._shape

    @property
    def format(self):
        if False:
            i = 10
            return i + 15
        'Render Buffer format'
        return self._format

    def resize(self, shape, format=None):
        if False:
            i = 10
            return i + 15
        "Set the render-buffer size and format\n\n        Parameters\n        ----------\n        shape : tuple of integers\n            New shape in yx order. A render buffer is always 2D. For\n            symmetry with the texture class, a 3-element tuple can also\n            be given, in which case the last dimension is ignored.\n        format : {None, 'color', 'depth', 'stencil'}\n            The buffer format. If None, the current format is maintained.\n            If that is also None, the format will be set upon attaching\n            it to a framebuffer. One can also specify the explicit enum:\n            GL_RGB565, GL_RGBA4, GL_RGB5_A1, GL_DEPTH_COMPONENT16, or\n            GL_STENCIL_INDEX8\n        "
        if not self._resizeable:
            raise RuntimeError('RenderBuffer is not resizeable')
        if not (isinstance(shape, tuple) and len(shape) in (2, 3)):
            raise ValueError('RenderBuffer shape must be a 2/3 element tuple')
        if format is None:
            format = self._format
        elif isinstance(format, int):
            pass
        elif isinstance(format, str):
            if format not in ('color', 'depth', 'stencil'):
                raise ValueError('RenderBuffer format must be "color", "depth" or "stencil", not %r' % format)
        else:
            raise ValueError('Invalid RenderBuffer format: %r' % format)
        self._shape = tuple(shape[:2])
        self._format = format
        if self._format is not None:
            self._glir.command('SIZE', self._id, self._shape, self._format)

class FrameBuffer(GLObject):
    """Frame buffer object

    Parameters
    ----------
    color : RenderBuffer (optional)
        The color buffer to attach to this frame buffer
    depth : RenderBuffer (optional)
        The depth buffer to attach to this frame buffer
    stencil : RenderBuffer (optional)
        The stencil buffer to attach to this frame buffer
    """
    _GLIR_TYPE = 'FrameBuffer'

    def __init__(self, color=None, depth=None, stencil=None):
        if False:
            for i in range(10):
                print('nop')
        GLObject.__init__(self)
        self._color_buffer = None
        self._depth_buffer = None
        self._stencil_buffer = None
        if color is not None:
            self.color_buffer = color
        if depth is not None:
            self.depth_buffer = depth
        if stencil is not None:
            self.stencil_buffer = stencil

    def activate(self):
        if False:
            while True:
                i = 10
        'Activate/use this frame buffer.'
        self._glir.command('FRAMEBUFFER', self._id, True)
        canvas = get_current_canvas()
        if canvas is not None:
            canvas.context.glir.associate(self.glir)

    def deactivate(self):
        if False:
            i = 10
            return i + 15
        'Stop using this frame buffer, the previous framebuffer will be\n        made active.\n        '
        self._glir.command('FRAMEBUFFER', self._id, False)

    def __enter__(self):
        if False:
            return 10
        self.activate()
        return self

    def __exit__(self, t, val, trace):
        if False:
            while True:
                i = 10
        self.deactivate()

    def _set_buffer(self, buffer, format):
        if False:
            while True:
                i = 10
        formats = ('color', 'depth', 'stencil')
        assert format in formats
        if isinstance(buffer, RenderBuffer):
            if buffer.format is None:
                buffer.resize(buffer.shape, format)
            elif buffer.format in formats and buffer.format != format:
                raise ValueError('Cannot attach a %s buffer as %s buffer.' % (buffer.format, format))
        if buffer is None:
            setattr(self, '_%s_buffer' % format, None)
            self._glir.command('ATTACH', self._id, format, 0)
        elif isinstance(buffer, (Texture2D, RenderBuffer)):
            self.glir.associate(buffer.glir)
            setattr(self, '_%s_buffer' % format, buffer)
            self._glir.command('ATTACH', self._id, format, buffer.id)
        else:
            raise TypeError('Buffer must be a RenderBuffer, Texture2D or None. (got %s)' % type(buffer))

    @property
    def color_buffer(self):
        if False:
            for i in range(10):
                print('nop')
        'Color buffer attachment'
        return self._color_buffer

    @color_buffer.setter
    def color_buffer(self, buffer):
        if False:
            print('Hello World!')
        self._set_buffer(buffer, 'color')

    @property
    def depth_buffer(self):
        if False:
            return 10
        'Depth buffer attachment'
        return self._depth_buffer

    @depth_buffer.setter
    def depth_buffer(self, buffer):
        if False:
            i = 10
            return i + 15
        self._set_buffer(buffer, 'depth')

    @property
    def stencil_buffer(self):
        if False:
            return 10
        'Stencil buffer attachment'
        return self._stencil_buffer

    @stencil_buffer.setter
    def stencil_buffer(self, buffer):
        if False:
            for i in range(10):
                print('nop')
        self._set_buffer(buffer, 'stencil')

    @property
    def shape(self):
        if False:
            i = 10
            return i + 15
        'The shape of the Texture/RenderBuffer attached to this FrameBuffer'
        if self.color_buffer is not None:
            return self.color_buffer.shape[:2]
        if self.depth_buffer is not None:
            return self.depth_buffer.shape[:2]
        if self.stencil_buffer is not None:
            return self.stencil_buffer.shape[:2]
        raise RuntimeError('FrameBuffer without buffers has undefined shape')

    def resize(self, shape):
        if False:
            i = 10
            return i + 15
        'Resize all attached buffers with the given shape\n\n        Parameters\n        ----------\n        shape : tuple of two integers\n            New buffer shape (h, w), to be applied to all currently\n            attached buffers. For buffers that are a texture, the number\n            of color channels is preserved.\n        '
        if not (isinstance(shape, tuple) and len(shape) == 2):
            raise ValueError('RenderBuffer shape must be a 2-element tuple')
        for buf in (self.color_buffer, self.depth_buffer, self.stencil_buffer):
            if buf is None:
                continue
            shape_ = shape
            if isinstance(buf, Texture2D):
                shape_ = shape + (buf._inv_formats[buf.format],)
            buf.resize(shape_, buf.format)

    def read(self, mode='color', alpha=True, crop=None):
        if False:
            while True:
                i = 10
        "Return array of pixel values in an attached buffer\n\n        Parameters\n        ----------\n        mode : str\n            The buffer type to read. May be 'color', 'depth', or 'stencil'.\n        alpha : bool\n            If True, returns RGBA array. Otherwise, returns RGB.\n        crop : array-like\n            If not None, specifies pixels to read from buffer.\n            Format is (x, y, w, h).\n\n        Returns\n        -------\n        buffer : array\n            3D array of pixels in np.uint8 format.\n            The array shape is (h, w, 3) or (h, w, 4), with the top-left\n            corner of the framebuffer at index [0, 0] in the returned array if\n            crop was not specified. If crop was given, the result will match\n            the offset and dimensions of the crop.\n\n        "
        _check_valid('mode', mode, ['color', 'depth', 'stencil'])
        buffer = getattr(self, mode + '_buffer')
        if buffer is None:
            raise ValueError("Can't read pixels for buffer {}, buffer does not exist.".format(mode))
        if crop is None:
            (h, w) = buffer.shape[:2]
            crop = (0, 0, w, h)
        return read_pixels(crop, alpha=alpha, mode=mode)