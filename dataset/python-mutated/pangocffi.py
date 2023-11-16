from libqtile.pango_ffi import pango_ffi as ffi
gobject = ffi.dlopen('libgobject-2.0.so.0')
pango = ffi.dlopen('libpango-1.0.so.0')
pangocairo = ffi.dlopen('libpangocairo-1.0.so.0')

def patch_cairo_context(cairo_t):
    if False:
        while True:
            i = 10

    def create_layout():
        if False:
            print('Hello World!')
        return PangoLayout(ffi.cast('struct _cairo *', cairo_t._pointer))
    cairo_t.create_layout = create_layout

    def show_layout(layout):
        if False:
            print('Hello World!')
        pangocairo.pango_cairo_show_layout(ffi.cast('struct _cairo *', cairo_t._pointer), layout._pointer)
    cairo_t.show_layout = show_layout
    return cairo_t
ALIGN_CENTER = pango.PANGO_ALIGN_CENTER
ELLIPSIZE_END = pango.PANGO_ELLIPSIZE_END
units_from_double = pango.pango_units_from_double
ALIGNMENTS = {'left': pango.PANGO_ALIGN_LEFT, 'center': pango.PANGO_ALIGN_CENTER, 'right': pango.PANGO_ALIGN_RIGHT}

class PangoLayout:

    def __init__(self, cairo_t):
        if False:
            for i in range(10):
                print('nop')
        self._cairo_t = cairo_t
        self._pointer = pangocairo.pango_cairo_create_layout(cairo_t)

        def free(p):
            if False:
                print('Hello World!')
            p = ffi.cast('gpointer', p)
            gobject.g_object_unref(p)
        self._pointer = ffi.gc(self._pointer, free)

    def finalize(self):
        if False:
            for i in range(10):
                print('nop')
        self._desc = None
        self._pointer = None
        self._cairo_t = None

    def finalized(self):
        if False:
            return 10
        return self._pointer is None

    def set_font_description(self, desc):
        if False:
            for i in range(10):
                print('nop')
        self._desc = desc
        pango.pango_layout_set_font_description(self._pointer, desc._pointer)

    def get_font_description(self):
        if False:
            while True:
                i = 10
        descr = pango.pango_layout_get_font_description(self._pointer)
        return FontDescription(descr)

    def set_alignment(self, alignment):
        if False:
            print('Hello World!')
        pango.pango_layout_set_alignment(self._pointer, alignment)

    def set_attributes(self, attrs):
        if False:
            return 10
        pango.pango_layout_set_attributes(self._pointer, attrs)

    def set_text(self, text):
        if False:
            return 10
        text = text.encode('utf-8')
        pango.pango_layout_set_text(self._pointer, text, -1)

    def get_text(self):
        if False:
            print('Hello World!')
        ret = pango.pango_layout_get_text(self._pointer)
        return ffi.string(ret).decode()

    def set_ellipsize(self, ellipzize):
        if False:
            while True:
                i = 10
        pango.pango_layout_set_ellipsize(self._pointer, ellipzize)

    def get_ellipsize(self):
        if False:
            for i in range(10):
                print('nop')
        return pango.pango_layout_get_ellipsize(self._pointer)

    def get_pixel_size(self):
        if False:
            while True:
                i = 10
        width = ffi.new('int[1]')
        height = ffi.new('int[1]')
        pango.pango_layout_get_pixel_size(self._pointer, width, height)
        return (width[0], height[0])

    def set_width(self, width):
        if False:
            while True:
                i = 10
        pango.pango_layout_set_width(self._pointer, width)

class FontDescription:

    def __init__(self, pointer=None):
        if False:
            for i in range(10):
                print('nop')
        if pointer is None:
            self._pointer = pango.pango_font_description_new()
            self._pointer = ffi.gc(self._pointer, pango.pango_font_description_free)
        else:
            self._pointer = pointer

    @classmethod
    def from_string(cls, string):
        if False:
            print('Hello World!')
        pointer = pango.pango_font_description_from_string(string.encode())
        pointer = ffi.gc(pointer, pango.pango_font_description_free)
        return cls(pointer)

    def set_family(self, family):
        if False:
            print('Hello World!')
        pango.pango_font_description_set_family(self._pointer, family.encode())

    def get_family(self):
        if False:
            return 10
        ret = pango.pango_font_description_get_family(self._pointer)
        return ffi.string(ret).decode()

    def set_absolute_size(self, size):
        if False:
            for i in range(10):
                print('nop')
        pango.pango_font_description_set_absolute_size(self._pointer, size)

    def set_size(self, size):
        if False:
            print('Hello World!')
        pango.pango_font_description_set_size(self._pointer, size)

    def get_size(self):
        if False:
            while True:
                i = 10
        return pango.pango_font_description_get_size(self._pointer)

def parse_markup(value, accel_marker=0):
    if False:
        i = 10
        return i + 15
    attr_list = ffi.new('PangoAttrList**')
    text = ffi.new('char**')
    error = ffi.new('GError**')
    value = value.encode()
    ret = pango.pango_parse_markup(value, -1, accel_marker, attr_list, text, ffi.NULL, error)
    if ret == 0:
        raise Exception('parse_markup() failed for %s' % value)
    return (attr_list[0], ffi.string(text[0]), chr(accel_marker))

def markup_escape_text(text):
    if False:
        for i in range(10):
            print('nop')
    ret = gobject.g_markup_escape_text(text.encode('utf-8'), -1)
    return ffi.string(ret).decode()