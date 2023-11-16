import os
if os.environ.get('SDL_VIDEODRIVER') == 'dummy':
    __tags__ = ('ignore', 'subprocess_ignore')
import unittest
import ctypes
import weakref
import gc
import pathlib
import platform
IS_PYPY = 'PyPy' == platform.python_implementation()
try:
    from pygame.tests.test_utils import arrinter
except NameError:
    pass
import pygame
try:
    import pygame.freetype as ft
except ImportError:
    ft = None
FONTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fixtures', 'fonts')

def nullfont():
    if False:
        i = 10
        return i + 15
    'return an uninitialized font instance'
    return ft.Font.__new__(ft.Font)
max_point_size_FX6 = 2147483647
max_point_size = max_point_size_FX6 >> 6
max_point_size_f = max_point_size_FX6 * 0.015625

def surf_same_image(a, b):
    if False:
        while True:
            i = 10
    "Return True if a's pixel buffer is identical to b's"
    a_sz = a.get_height() * a.get_pitch()
    b_sz = b.get_height() * b.get_pitch()
    if a_sz != b_sz:
        return False
    a_bytes = ctypes.string_at(a._pixels_address, a_sz)
    b_bytes = ctypes.string_at(b._pixels_address, b_sz)
    return a_bytes == b_bytes

class FreeTypeFontTest(unittest.TestCase):
    _fixed_path = os.path.join(FONTDIR, 'test_fixed.otf')
    _sans_path = os.path.join(FONTDIR, 'test_sans.ttf')
    _mono_path = os.path.join(FONTDIR, 'PyGameMono.otf')
    _bmp_8_75dpi_path = os.path.join(FONTDIR, 'PyGameMono-8.bdf')
    _bmp_18_75dpi_path = os.path.join(FONTDIR, 'PyGameMono-18-75dpi.bdf')
    _bmp_18_100dpi_path = os.path.join(FONTDIR, 'PyGameMono-18-100dpi.bdf')
    _TEST_FONTS = {}

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        ft.init()
        cls._TEST_FONTS['fixed'] = ft.Font(cls._fixed_path)
        cls._TEST_FONTS['sans'] = ft.Font(cls._sans_path)
        cls._TEST_FONTS['mono'] = ft.Font(cls._mono_path)
        cls._TEST_FONTS['bmp-8-75dpi'] = ft.Font(cls._bmp_8_75dpi_path)
        cls._TEST_FONTS['bmp-18-75dpi'] = ft.Font(cls._bmp_18_75dpi_path)
        cls._TEST_FONTS['bmp-18-100dpi'] = ft.Font(cls._bmp_18_100dpi_path)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        ft.quit()

    def test_freetype_defaultfont(self):
        if False:
            while True:
                i = 10
        font = ft.Font(None)
        self.assertEqual(font.name, 'FreeSans')

    def test_freetype_Font_init(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertRaises(FileNotFoundError, ft.Font, os.path.join(FONTDIR, 'nonexistent.ttf'))
        f = self._TEST_FONTS['sans']
        self.assertIsInstance(f, ft.Font)
        f = self._TEST_FONTS['fixed']
        self.assertIsInstance(f, ft.Font)
        f = ft.Font(size=22, file=None)
        self.assertEqual(f.size, 22)
        f = ft.Font(font_index=0, file=None)
        self.assertNotEqual(ft.get_default_resolution(), 100)
        f = ft.Font(resolution=100, file=None)
        self.assertEqual(f.resolution, 100)
        f = ft.Font(ucs4=True, file=None)
        self.assertTrue(f.ucs4)
        self.assertRaises(OverflowError, ft.Font, file=None, size=max_point_size + 1)
        self.assertRaises(OverflowError, ft.Font, file=None, size=-1)
        f = ft.Font(None, size=24)
        self.assertTrue(f.height > 0)
        self.assertRaises(FileNotFoundError, f.__init__, os.path.join(FONTDIR, 'nonexistent.ttf'))
        f = ft.Font(self._sans_path, size=24, ucs4=True)
        self.assertEqual(f.name, 'Liberation Sans')
        self.assertTrue(f.scalable)
        self.assertFalse(f.fixed_width)
        self.assertTrue(f.antialiased)
        self.assertFalse(f.oblique)
        self.assertTrue(f.ucs4)
        f.antialiased = False
        f.oblique = True
        f.__init__(self._mono_path)
        self.assertEqual(f.name, 'PyGameMono')
        self.assertTrue(f.scalable)
        self.assertTrue(f.fixed_width)
        self.assertFalse(f.antialiased)
        self.assertTrue(f.oblique)
        self.assertTrue(f.ucs4)
        f = ft.Font(self._bmp_8_75dpi_path)
        sizes = f.get_sizes()
        self.assertEqual(len(sizes), 1)
        (size_pt, width_px, height_px, x_ppem, y_ppem) = sizes[0]
        self.assertEqual(f.size, (x_ppem, y_ppem))
        f.__init__(self._bmp_8_75dpi_path, size=12)
        self.assertEqual(f.size, 12.0)

    @unittest.skipIf(IS_PYPY, "PyPy doesn't use refcounting")
    def test_freetype_Font_dealloc(self):
        if False:
            i = 10
            return i + 15
        import sys
        handle = open(self._sans_path, 'rb')

        def load_font():
            if False:
                i = 10
                return i + 15
            tempFont = ft.Font(handle)
        try:
            load_font()
            self.assertEqual(sys.getrefcount(handle), 2)
        finally:
            handle.close()

    def test_freetype_Font_kerning(self):
        if False:
            print('Hello World!')
        'Ensures get/set works with the kerning property.'
        ft_font = self._TEST_FONTS['sans']
        self.assertFalse(ft_font.kerning)
        ft_font.kerning = True
        self.assertTrue(ft_font.kerning)
        ft_font.kerning = False
        self.assertFalse(ft_font.kerning)

    def test_freetype_Font_kerning__enabled(self):
        if False:
            while True:
                i = 10
        'Ensures exceptions are not raised when calling freetype methods\n        while kerning is enabled.\n\n        Note: This does not test what changes occur to a rendered font by\n              having kerning enabled.\n\n        Related to issue #367.\n        '
        surface = pygame.Surface((10, 10), 0, 32)
        TEST_TEXT = 'Freetype Font'
        ft_font = self._TEST_FONTS['bmp-8-75dpi']
        ft_font.kerning = True
        metrics = ft_font.get_metrics(TEST_TEXT)
        self.assertIsInstance(metrics, list)
        rect = ft_font.get_rect(TEST_TEXT)
        self.assertIsInstance(rect, pygame.Rect)
        (font_surf, rect) = ft_font.render(TEST_TEXT)
        self.assertIsInstance(font_surf, pygame.Surface)
        self.assertIsInstance(rect, pygame.Rect)
        rect = ft_font.render_to(surface, (0, 0), TEST_TEXT)
        self.assertIsInstance(rect, pygame.Rect)
        (buf, size) = ft_font.render_raw(TEST_TEXT)
        self.assertIsInstance(buf, bytes)
        self.assertIsInstance(size, tuple)
        rect = ft_font.render_raw_to(surface.get_view('2'), TEST_TEXT)
        self.assertIsInstance(rect, pygame.Rect)

    def test_freetype_Font_scalable(self):
        if False:
            while True:
                i = 10
        f = self._TEST_FONTS['sans']
        self.assertTrue(f.scalable)
        self.assertRaises(RuntimeError, lambda : nullfont().scalable)

    def test_freetype_Font_fixed_width(self):
        if False:
            return 10
        f = self._TEST_FONTS['sans']
        self.assertFalse(f.fixed_width)
        f = self._TEST_FONTS['mono']
        self.assertTrue(f.fixed_width)
        self.assertRaises(RuntimeError, lambda : nullfont().fixed_width)

    def test_freetype_Font_fixed_sizes(self):
        if False:
            i = 10
            return i + 15
        f = self._TEST_FONTS['sans']
        self.assertEqual(f.fixed_sizes, 0)
        f = self._TEST_FONTS['bmp-8-75dpi']
        self.assertEqual(f.fixed_sizes, 1)
        f = self._TEST_FONTS['mono']
        self.assertEqual(f.fixed_sizes, 2)

    def test_freetype_Font_get_sizes(self):
        if False:
            for i in range(10):
                print('nop')
        f = self._TEST_FONTS['sans']
        szlist = f.get_sizes()
        self.assertIsInstance(szlist, list)
        self.assertEqual(len(szlist), 0)
        f = self._TEST_FONTS['bmp-8-75dpi']
        szlist = f.get_sizes()
        self.assertIsInstance(szlist, list)
        self.assertEqual(len(szlist), 1)
        size8 = szlist[0]
        self.assertIsInstance(size8[0], int)
        self.assertEqual(size8[0], 8)
        self.assertIsInstance(size8[1], int)
        self.assertIsInstance(size8[2], int)
        self.assertIsInstance(size8[3], float)
        self.assertEqual(int(size8[3] * 64.0 + 0.5), 8 * 64)
        self.assertIsInstance(size8[4], float)
        self.assertEqual(int(size8[4] * 64.0 + 0.5), 8 * 64)
        f = self._TEST_FONTS['mono']
        szlist = f.get_sizes()
        self.assertIsInstance(szlist, list)
        self.assertEqual(len(szlist), 2)
        size8 = szlist[0]
        self.assertEqual(size8[3], 8)
        self.assertEqual(int(size8[3] * 64.0 + 0.5), 8 * 64)
        self.assertEqual(int(size8[4] * 64.0 + 0.5), 8 * 64)
        size19 = szlist[1]
        self.assertEqual(size19[3], 19)
        self.assertEqual(int(size19[3] * 64.0 + 0.5), 19 * 64)
        self.assertEqual(int(size19[4] * 64.0 + 0.5), 19 * 64)

    def test_freetype_Font_use_bitmap_strikes(self):
        if False:
            return 10
        f = self._TEST_FONTS['mono']
        try:
            self.assertTrue(f.use_bitmap_strikes)
            (s_strike, sz) = f.render_raw('A', size=19)
            try:
                f.vertical = True
                (s_strike_vert, sz) = f.render_raw('A', size=19)
            finally:
                f.vertical = False
            try:
                f.wide = True
                (s_strike_wide, sz) = f.render_raw('A', size=19)
            finally:
                f.wide = False
            try:
                f.underline = True
                (s_strike_underline, sz) = f.render_raw('A', size=19)
            finally:
                f.underline = False
            (s_strike_rot45, sz) = f.render_raw('A', size=19, rotation=45)
            try:
                f.strong = True
                (s_strike_strong, sz) = f.render_raw('A', size=19)
            finally:
                f.strong = False
            try:
                f.oblique = True
                (s_strike_oblique, sz) = f.render_raw('A', size=19)
            finally:
                f.oblique = False
            f.use_bitmap_strikes = False
            self.assertFalse(f.use_bitmap_strikes)
            (s_outline, sz) = f.render_raw('A', size=19)
            self.assertNotEqual(s_outline, s_strike)
            try:
                f.vertical = True
                (s_outline, sz) = f.render_raw('A', size=19)
                self.assertNotEqual(s_outline, s_strike_vert)
            finally:
                f.vertical = False
            try:
                f.wide = True
                (s_outline, sz) = f.render_raw('A', size=19)
                self.assertNotEqual(s_outline, s_strike_wide)
            finally:
                f.wide = False
            try:
                f.underline = True
                (s_outline, sz) = f.render_raw('A', size=19)
                self.assertNotEqual(s_outline, s_strike_underline)
            finally:
                f.underline = False
            (s_outline, sz) = f.render_raw('A', size=19, rotation=45)
            self.assertEqual(s_outline, s_strike_rot45)
            try:
                f.strong = True
                (s_outline, sz) = f.render_raw('A', size=19)
                self.assertEqual(s_outline, s_strike_strong)
            finally:
                f.strong = False
            try:
                f.oblique = True
                (s_outline, sz) = f.render_raw('A', size=19)
                self.assertEqual(s_outline, s_strike_oblique)
            finally:
                f.oblique = False
        finally:
            f.use_bitmap_strikes = True

    def test_freetype_Font_bitmap_files(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure bitmap file restrictions are caught'
        f = self._TEST_FONTS['bmp-8-75dpi']
        f_null = nullfont()
        s = pygame.Surface((10, 10), 0, 32)
        a = s.get_view('3')
        exception = AttributeError
        self.assertRaises(exception, setattr, f, 'strong', True)
        self.assertRaises(exception, setattr, f, 'oblique', True)
        self.assertRaises(exception, setattr, f, 'style', ft.STYLE_STRONG)
        self.assertRaises(exception, setattr, f, 'style', ft.STYLE_OBLIQUE)
        exception = RuntimeError
        self.assertRaises(exception, setattr, f_null, 'strong', True)
        self.assertRaises(exception, setattr, f_null, 'oblique', True)
        self.assertRaises(exception, setattr, f_null, 'style', ft.STYLE_STRONG)
        self.assertRaises(exception, setattr, f_null, 'style', ft.STYLE_OBLIQUE)
        exception = ValueError
        self.assertRaises(exception, f.render, 'A', (0, 0, 0), size=8, rotation=1)
        self.assertRaises(exception, f.render, 'A', (0, 0, 0), size=8, style=ft.STYLE_OBLIQUE)
        self.assertRaises(exception, f.render, 'A', (0, 0, 0), size=8, style=ft.STYLE_STRONG)
        self.assertRaises(exception, f.render_raw, 'A', size=8, rotation=1)
        self.assertRaises(exception, f.render_raw, 'A', size=8, style=ft.STYLE_OBLIQUE)
        self.assertRaises(exception, f.render_raw, 'A', size=8, style=ft.STYLE_STRONG)
        self.assertRaises(exception, f.render_to, s, (0, 0), 'A', (0, 0, 0), size=8, rotation=1)
        self.assertRaises(exception, f.render_to, s, (0, 0), 'A', (0, 0, 0), size=8, style=ft.STYLE_OBLIQUE)
        self.assertRaises(exception, f.render_to, s, (0, 0), 'A', (0, 0, 0), size=8, style=ft.STYLE_STRONG)
        self.assertRaises(exception, f.render_raw_to, a, 'A', size=8, rotation=1)
        self.assertRaises(exception, f.render_raw_to, a, 'A', size=8, style=ft.STYLE_OBLIQUE)
        self.assertRaises(exception, f.render_raw_to, a, 'A', size=8, style=ft.STYLE_STRONG)
        self.assertRaises(exception, f.get_rect, 'A', size=8, rotation=1)
        self.assertRaises(exception, f.get_rect, 'A', size=8, style=ft.STYLE_OBLIQUE)
        self.assertRaises(exception, f.get_rect, 'A', size=8, style=ft.STYLE_STRONG)
        exception = pygame.error
        self.assertRaises(exception, f.get_rect, 'A', size=42)
        self.assertRaises(exception, f.get_metrics, 'A', size=42)
        self.assertRaises(exception, f.get_sized_ascender, 42)
        self.assertRaises(exception, f.get_sized_descender, 42)
        self.assertRaises(exception, f.get_sized_height, 42)
        self.assertRaises(exception, f.get_sized_glyph_height, 42)

    def test_freetype_Font_get_metrics(self):
        if False:
            return 10
        font = self._TEST_FONTS['sans']
        metrics = font.get_metrics('ABCD', size=24)
        self.assertEqual(len(metrics), len('ABCD'))
        self.assertIsInstance(metrics, list)
        for metrics_tuple in metrics:
            self.assertIsInstance(metrics_tuple, tuple, metrics_tuple)
            self.assertEqual(len(metrics_tuple), 6)
            for m in metrics_tuple[:4]:
                self.assertIsInstance(m, int)
            for m in metrics_tuple[4:]:
                self.assertIsInstance(m, float)
        metrics = font.get_metrics('', size=24)
        self.assertEqual(metrics, [])
        self.assertRaises(TypeError, font.get_metrics, 24, 24)
        self.assertRaises(RuntimeError, nullfont().get_metrics, 'a', size=24)

    def test_freetype_Font_get_rect(self):
        if False:
            return 10
        font = self._TEST_FONTS['sans']

        def test_rect(r):
            if False:
                i = 10
                return i + 15
            self.assertIsInstance(r, pygame.Rect)
        rect_default = font.get_rect('ABCDabcd', size=24)
        test_rect(rect_default)
        self.assertTrue(rect_default.size > (0, 0))
        self.assertTrue(rect_default.width > rect_default.height)
        rect_bigger = font.get_rect('ABCDabcd', size=32)
        test_rect(rect_bigger)
        self.assertTrue(rect_bigger.size > rect_default.size)
        rect_strong = font.get_rect('ABCDabcd', size=24, style=ft.STYLE_STRONG)
        test_rect(rect_strong)
        self.assertTrue(rect_strong.size > rect_default.size)
        font.vertical = True
        rect_vert = font.get_rect('ABCDabcd', size=24)
        test_rect(rect_vert)
        self.assertTrue(rect_vert.width < rect_vert.height)
        font.vertical = False
        rect_oblique = font.get_rect('ABCDabcd', size=24, style=ft.STYLE_OBLIQUE)
        test_rect(rect_oblique)
        self.assertTrue(rect_oblique.width > rect_default.width)
        self.assertTrue(rect_oblique.height == rect_default.height)
        rect_under = font.get_rect('ABCDabcd', size=24, style=ft.STYLE_UNDERLINE)
        test_rect(rect_under)
        self.assertTrue(rect_under.width == rect_default.width)
        self.assertTrue(rect_under.height > rect_default.height)
        ufont = self._TEST_FONTS['mono']
        rect_utf32 = ufont.get_rect('𓁹', size=24)
        rect_utf16 = ufont.get_rect('\ud80c\udc79', size=24)
        self.assertEqual(rect_utf16, rect_utf32)
        ufont.ucs4 = True
        try:
            rect_utf16 = ufont.get_rect('\ud80c\udc79', size=24)
        finally:
            ufont.ucs4 = False
        self.assertNotEqual(rect_utf16, rect_utf32)
        self.assertRaises(RuntimeError, nullfont().get_rect, 'a', size=24)
        rect12 = font.get_rect('A', size=12.0)
        rect24 = font.get_rect('A', size=24.0)
        rect_x = font.get_rect('A', size=(24.0, 12.0))
        self.assertEqual(rect_x.width, rect24.width)
        self.assertEqual(rect_x.height, rect12.height)
        rect_y = font.get_rect('A', size=(12.0, 24.0))
        self.assertEqual(rect_y.width, rect12.width)
        self.assertEqual(rect_y.height, rect24.height)

    def test_freetype_Font_height(self):
        if False:
            print('Hello World!')
        f = self._TEST_FONTS['sans']
        self.assertEqual(f.height, 2355)
        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.height, 1100)
        self.assertRaises(RuntimeError, lambda : nullfont().height)

    def test_freetype_Font_name(self):
        if False:
            while True:
                i = 10
        f = self._TEST_FONTS['sans']
        self.assertEqual(f.name, 'Liberation Sans')
        f = self._TEST_FONTS['fixed']
        self.assertEqual(f.name, 'Inconsolata')
        nf = nullfont()
        self.assertEqual(nf.name, repr(nf))

    def test_freetype_Font_size(self):
        if False:
            return 10
        f = ft.Font(None, size=12)
        self.assertEqual(f.size, 12)
        f.size = 22
        self.assertEqual(f.size, 22)
        f.size = 0
        self.assertEqual(f.size, 0)
        f.size = max_point_size
        self.assertEqual(f.size, max_point_size)
        f.size = 6.5
        self.assertEqual(f.size, 6.5)
        f.size = max_point_size_f
        self.assertEqual(f.size, max_point_size_f)
        self.assertRaises(OverflowError, setattr, f, 'size', -1)
        self.assertRaises(OverflowError, setattr, f, 'size', max_point_size + 1)
        f.size = (24.0, 0)
        size = f.size
        self.assertIsInstance(size, float)
        self.assertEqual(size, 24.0)
        f.size = (16, 16)
        size = f.size
        self.assertIsInstance(size, tuple)
        self.assertEqual(len(size), 2)
        (x, y) = size
        self.assertIsInstance(x, float)
        self.assertEqual(x, 16.0)
        self.assertIsInstance(y, float)
        self.assertEqual(y, 16.0)
        f.size = (20.5, 22.25)
        (x, y) = f.size
        self.assertEqual(x, 20.5)
        self.assertEqual(y, 22.25)
        f.size = (0, 0)
        size = f.size
        self.assertIsInstance(size, float)
        self.assertEqual(size, 0.0)
        self.assertRaises(ValueError, setattr, f, 'size', (0, 24.0))
        self.assertRaises(TypeError, setattr, f, 'size', (24.0,))
        self.assertRaises(TypeError, setattr, f, 'size', (24.0, 0, 0))
        self.assertRaises(TypeError, setattr, f, 'size', (24j, 24.0))
        self.assertRaises(TypeError, setattr, f, 'size', (24.0, 24j))
        self.assertRaises(OverflowError, setattr, f, 'size', (-1, 16))
        self.assertRaises(OverflowError, setattr, f, 'size', (max_point_size + 1, 16))
        self.assertRaises(OverflowError, setattr, f, 'size', (16, -1))
        self.assertRaises(OverflowError, setattr, f, 'size', (16, max_point_size + 1))
        f75 = self._TEST_FONTS['bmp-18-75dpi']
        sizes = f75.get_sizes()
        self.assertEqual(len(sizes), 1)
        (size_pt, width_px, height_px, x_ppem, y_ppem) = sizes[0]
        self.assertEqual(size_pt, 18)
        self.assertEqual(x_ppem, 19.0)
        self.assertEqual(y_ppem, 19.0)
        rect = f75.get_rect('A', size=18)
        rect = f75.get_rect('A', size=19)
        rect = f75.get_rect('A', size=(19.0, 19.0))
        self.assertRaises(pygame.error, f75.get_rect, 'A', size=17)
        f100 = self._TEST_FONTS['bmp-18-100dpi']
        sizes = f100.get_sizes()
        self.assertEqual(len(sizes), 1)
        (size_pt, width_px, height_px, x_ppem, y_ppem) = sizes[0]
        self.assertEqual(size_pt, 18)
        self.assertEqual(x_ppem, 25.0)
        self.assertEqual(y_ppem, 25.0)
        rect = f100.get_rect('A', size=18)
        rect = f100.get_rect('A', size=25)
        rect = f100.get_rect('A', size=(25.0, 25.0))
        self.assertRaises(pygame.error, f100.get_rect, 'A', size=17)

    def test_freetype_Font_rotation(self):
        if False:
            while True:
                i = 10
        test_angles = [(30, 30), (360, 0), (390, 30), (720, 0), (764, 44), (-30, 330), (-360, 0), (-390, 330), (-720, 0), (-764, 316)]
        f = ft.Font(None)
        self.assertEqual(f.rotation, 0)
        for (r, r_reduced) in test_angles:
            f.rotation = r
            self.assertEqual(f.rotation, r_reduced, 'for angle %d: %d != %d' % (r, f.rotation, r_reduced))
        self.assertRaises(TypeError, setattr, f, 'rotation', '12')

    def test_freetype_Font_render_to(self):
        if False:
            return 10
        font = self._TEST_FONTS['sans']
        surf = pygame.Surface((800, 600))
        color = pygame.Color(0, 0, 0)
        rrect = font.render_to(surf, (32, 32), 'FoobarBaz', color, None, size=24)
        self.assertIsInstance(rrect, pygame.Rect)
        self.assertEqual(rrect.topleft, (32, 32))
        self.assertNotEqual(rrect.bottomright, (32, 32))
        rcopy = rrect.copy()
        rcopy.topleft = (32, 32)
        self.assertTrue(surf.get_rect().contains(rcopy))
        rect = pygame.Rect(20, 20, 2, 2)
        rrect = font.render_to(surf, rect, 'FoobarBax', color, None, size=24)
        self.assertEqual(rect.topleft, rrect.topleft)
        self.assertNotEqual(rrect.size, rect.size)
        rrect = font.render_to(surf, (20.1, 18.9), 'FoobarBax', color, None, size=24)
        rrect = font.render_to(surf, rect, '', color, None, size=24)
        self.assertFalse(rrect)
        self.assertEqual(rrect.height, font.get_sized_height(24))
        self.assertRaises(TypeError, font.render_to, 'not a surface', 'text', color)
        self.assertRaises(TypeError, font.render_to, pygame.Surface, 'text', color)
        for dest in [None, 0, 'a', 'ab', (), (1,), ('a', 2), (1, 'a'), (1 + 2j, 2), (1, 1 + 2j), (1, int), (int, 1)]:
            self.assertRaises(TypeError, font.render_to, surf, dest, 'foobar', color, size=24)
        self.assertRaises(ValueError, font.render_to, surf, (0, 0), 'foobar', color)
        self.assertRaises(TypeError, font.render_to, surf, (0, 0), 'foobar', color, 2.3, size=24)
        self.assertRaises(ValueError, font.render_to, surf, (0, 0), 'foobar', color, None, style=42, size=24)
        self.assertRaises(TypeError, font.render_to, surf, (0, 0), 'foobar', color, None, style=None, size=24)
        self.assertRaises(ValueError, font.render_to, surf, (0, 0), 'foobar', color, None, style=97, size=24)

    def test_freetype_Font_render(self):
        if False:
            while True:
                i = 10
        font = self._TEST_FONTS['sans']
        surf = pygame.Surface((800, 600))
        color = pygame.Color(0, 0, 0)
        rend = font.render('FoobarBaz', pygame.Color(0, 0, 0), None, size=24)
        self.assertIsInstance(rend, tuple)
        self.assertEqual(len(rend), 2)
        self.assertIsInstance(rend[0], pygame.Surface)
        self.assertIsInstance(rend[1], pygame.Rect)
        self.assertEqual(rend[0].get_rect().size, rend[1].size)
        (s, r) = font.render('', pygame.Color(0, 0, 0), None, size=24)
        self.assertEqual(r.width, 0)
        self.assertEqual(r.height, font.get_sized_height(24))
        self.assertEqual(s.get_size(), r.size)
        self.assertEqual(s.get_bitsize(), 32)
        self.assertRaises(ValueError, font.render, 'foobar', color)
        self.assertRaises(TypeError, font.render, 'foobar', color, 2.3, size=24)
        self.assertRaises(ValueError, font.render, 'foobar', color, None, style=42, size=24)
        self.assertRaises(TypeError, font.render, 'foobar', color, None, style=None, size=24)
        self.assertRaises(ValueError, font.render, 'foobar', color, None, style=97, size=24)
        font2 = self._TEST_FONTS['mono']
        ucs4 = font2.ucs4
        try:
            font2.ucs4 = False
            rend1 = font2.render('\ud80c\udc79', color, size=24)
            rend2 = font2.render('𓁹', color, size=24)
            self.assertEqual(rend1[1], rend2[1])
            font2.ucs4 = True
            rend1 = font2.render('\ud80c\udc79', color, size=24)
            self.assertNotEqual(rend1[1], rend2[1])
        finally:
            font2.ucs4 = ucs4
        self.assertRaises(UnicodeEncodeError, font.render, '\ud80c', color, size=24)
        self.assertRaises(UnicodeEncodeError, font.render, '\udca7', color, size=24)
        self.assertRaises(UnicodeEncodeError, font.render, '\ud7ff\udca7', color, size=24)
        self.assertRaises(UnicodeEncodeError, font.render, '\udc00\udca7', color, size=24)
        self.assertRaises(UnicodeEncodeError, font.render, '\ud80c\udbff', color, size=24)
        self.assertRaises(UnicodeEncodeError, font.render, '\ud80c\ue000', color, size=24)
        self.assertRaises(RuntimeError, nullfont().render, 'a', (0, 0, 0), size=24)
        path = os.path.join(FONTDIR, 'A_PyGameMono-8.png')
        A = pygame.image.load(path)
        path = os.path.join(FONTDIR, 'u13079_PyGameMono-8.png')
        u13079 = pygame.image.load(path)
        font = self._TEST_FONTS['mono']
        font.ucs4 = False
        (A_rendered, r) = font.render('A', bgcolor=pygame.Color('white'), size=8)
        (u13079_rendered, r) = font.render('𓁹', bgcolor=pygame.Color('white'), size=8)
        bitmap = pygame.Surface(A.get_size(), pygame.SRCALPHA, 32)
        bitmap.blit(A, (0, 0))
        rendering = pygame.Surface(A_rendered.get_size(), pygame.SRCALPHA, 32)
        rendering.blit(A_rendered, (0, 0))
        self.assertTrue(surf_same_image(rendering, bitmap))
        bitmap = pygame.Surface(u13079.get_size(), pygame.SRCALPHA, 32)
        bitmap.blit(u13079, (0, 0))
        rendering = pygame.Surface(u13079_rendered.get_size(), pygame.SRCALPHA, 32)
        rendering.blit(u13079_rendered, (0, 0))
        self.assertTrue(surf_same_image(rendering, bitmap))

    def test_freetype_Font_render_mono(self):
        if False:
            print('Hello World!')
        font = self._TEST_FONTS['sans']
        color = pygame.Color('black')
        colorkey = pygame.Color('white')
        text = '.'
        save_antialiased = font.antialiased
        font.antialiased = False
        try:
            (surf, r) = font.render(text, color, size=24)
            self.assertEqual(surf.get_bitsize(), 8)
            flags = surf.get_flags()
            self.assertTrue(flags & pygame.SRCCOLORKEY)
            self.assertFalse(flags & (pygame.SRCALPHA | pygame.HWSURFACE))
            self.assertEqual(surf.get_colorkey(), colorkey)
            self.assertIsNone(surf.get_alpha())
            translucent_color = pygame.Color(*color)
            translucent_color.a = 55
            (surf, r) = font.render(text, translucent_color, size=24)
            self.assertEqual(surf.get_bitsize(), 8)
            flags = surf.get_flags()
            self.assertTrue(flags & (pygame.SRCCOLORKEY | pygame.SRCALPHA))
            self.assertFalse(flags & pygame.HWSURFACE)
            self.assertEqual(surf.get_colorkey(), colorkey)
            self.assertEqual(surf.get_alpha(), translucent_color.a)
            (surf, r) = font.render(text, color, colorkey, size=24)
            self.assertEqual(surf.get_bitsize(), 32)
        finally:
            font.antialiased = save_antialiased

    def test_freetype_Font_render_to_mono(self):
        if False:
            i = 10
            return i + 15
        font = self._TEST_FONTS['sans']
        text = ' .'
        rect = font.get_rect(text, size=24)
        size = rect.size
        fg = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        bg = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surrogate = pygame.Surface((1, 1), pygame.SRCALPHA, 32)
        surfaces = [pygame.Surface(size, 0, 8), pygame.Surface(size, 0, 16), pygame.Surface(size, pygame.SRCALPHA, 16), pygame.Surface(size, 0, 24), pygame.Surface(size, 0, 32), pygame.Surface(size, pygame.SRCALPHA, 32)]
        fg_colors = [surfaces[0].get_palette_at(2), surfaces[1].unmap_rgb(surfaces[1].map_rgb((128, 64, 200))), surfaces[2].unmap_rgb(surfaces[2].map_rgb((99, 0, 100, 64))), (128, 97, 213), (128, 97, 213), (128, 97, 213, 60)]
        fg_colors = [pygame.Color(*c) for c in fg_colors]
        self.assertEqual(len(surfaces), len(fg_colors))
        bg_colors = [surfaces[0].get_palette_at(4), surfaces[1].unmap_rgb(surfaces[1].map_rgb((220, 20, 99))), surfaces[2].unmap_rgb(surfaces[2].map_rgb((55, 200, 0, 86))), (255, 120, 13), (255, 120, 13), (255, 120, 13, 180)]
        bg_colors = [pygame.Color(*c) for c in bg_colors]
        self.assertEqual(len(surfaces), len(bg_colors))
        save_antialiased = font.antialiased
        font.antialiased = False
        try:
            fill_color = pygame.Color('black')
            for (i, surf) in enumerate(surfaces):
                surf.fill(fill_color)
                fg_color = fg_colors[i]
                fg.set_at((0, 0), fg_color)
                surf.blit(fg, (0, 0))
                r_fg_color = surf.get_at((0, 0))
                surf.set_at((0, 0), fill_color)
                rrect = font.render_to(surf, (0, 0), text, fg_color, size=24)
                bottomleft = (0, rrect.height - 1)
                self.assertEqual(surf.get_at(bottomleft), fill_color, 'Position: {}. Depth: {}. fg_color: {}.'.format(bottomleft, surf.get_bitsize(), fg_color))
                bottomright = (rrect.width - 1, rrect.height - 1)
                self.assertEqual(surf.get_at(bottomright), r_fg_color, 'Position: {}. Depth: {}. fg_color: {}.'.format(bottomright, surf.get_bitsize(), fg_color))
            for (i, surf) in enumerate(surfaces):
                surf.fill(fill_color)
                fg_color = fg_colors[i]
                bg_color = bg_colors[i]
                bg.set_at((0, 0), bg_color)
                fg.set_at((0, 0), fg_color)
                if surf.get_bitsize() == 24:
                    surrogate.set_at((0, 0), fill_color)
                    surrogate.blit(bg, (0, 0))
                    r_bg_color = surrogate.get_at((0, 0))
                    surrogate.blit(fg, (0, 0))
                    r_fg_color = surrogate.get_at((0, 0))
                else:
                    surf.blit(bg, (0, 0))
                    r_bg_color = surf.get_at((0, 0))
                    surf.blit(fg, (0, 0))
                    r_fg_color = surf.get_at((0, 0))
                    surf.set_at((0, 0), fill_color)
                rrect = font.render_to(surf, (0, 0), text, fg_color, bg_color, size=24)
                bottomleft = (0, rrect.height - 1)
                self.assertEqual(surf.get_at(bottomleft), r_bg_color)
                bottomright = (rrect.width - 1, rrect.height - 1)
                self.assertEqual(surf.get_at(bottomright), r_fg_color)
        finally:
            font.antialiased = save_antialiased

    def test_freetype_Font_render_raw(self):
        if False:
            return 10
        font = self._TEST_FONTS['sans']
        text = 'abc'
        size = font.get_rect(text, size=24).size
        rend = font.render_raw(text, size=24)
        self.assertIsInstance(rend, tuple)
        self.assertEqual(len(rend), 2)
        (r, s) = rend
        self.assertIsInstance(r, bytes)
        self.assertIsInstance(s, tuple)
        self.assertTrue(len(s), 2)
        (w, h) = s
        self.assertIsInstance(w, int)
        self.assertIsInstance(h, int)
        self.assertEqual(s, size)
        self.assertEqual(len(r), w * h)
        (r, (w, h)) = font.render_raw('', size=24)
        self.assertEqual(w, 0)
        self.assertEqual(h, font.height)
        self.assertEqual(len(r), 0)
        rend = font.render_raw('render_raw', size=24)
        text = ''.join([chr(i) for i in range(31, 64)])
        rend = font.render_raw(text, size=10)

    def test_freetype_Font_render_raw_to(self):
        if False:
            i = 10
            return i + 15
        font = self._TEST_FONTS['sans']
        text = 'abc'
        srect = font.get_rect(text, size=24)
        surf = pygame.Surface(srect.size, 0, 8)
        rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
        self.assertEqual(rrect, srect)
        for bpp in [24, 32]:
            surf = pygame.Surface(srect.size, 0, bpp)
            rrect = font.render_raw_to(surf.get_view('r'), text, size=24)
            self.assertEqual(rrect, srect)
        srect = font.get_rect(text, size=24, style=ft.STYLE_UNDERLINE)
        surf = pygame.Surface(srect.size, 0, 8)
        rrect = font.render_raw_to(surf.get_view('2'), text, size=24, style=ft.STYLE_UNDERLINE)
        self.assertEqual(rrect, srect)
        for bpp in [24, 32]:
            surf = pygame.Surface(srect.size, 0, bpp)
            rrect = font.render_raw_to(surf.get_view('r'), text, size=24, style=ft.STYLE_UNDERLINE)
            self.assertEqual(rrect, srect)
        font.antialiased = False
        try:
            srect = font.get_rect(text, size=24)
            surf = pygame.Surface(srect.size, 0, 8)
            rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
            self.assertEqual(rrect, srect)
            for bpp in [24, 32]:
                surf = pygame.Surface(srect.size, 0, bpp)
                rrect = font.render_raw_to(surf.get_view('r'), text, size=24)
                self.assertEqual(rrect, srect)
        finally:
            font.antialiased = True
        srect = font.get_rect(text, size=24)
        for bpp in [16, 24, 32]:
            surf = pygame.Surface(srect.size, 0, bpp)
            rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
            self.assertEqual(rrect, srect)
        srect = font.get_rect(text, size=24, style=ft.STYLE_UNDERLINE)
        for bpp in [16, 24, 32]:
            surf = pygame.Surface(srect.size, 0, bpp)
            rrect = font.render_raw_to(surf.get_view('2'), text, size=24, style=ft.STYLE_UNDERLINE)
            self.assertEqual(rrect, srect)
        font.antialiased = False
        try:
            srect = font.get_rect(text, size=24)
            for bpp in [16, 24, 32]:
                surf = pygame.Surface(srect.size, 0, bpp)
                rrect = font.render_raw_to(surf.get_view('2'), text, size=24)
                self.assertEqual(rrect, srect)
        finally:
            font.antialiased = True
        srect = font.get_rect(text, size=24)
        surf_buf = pygame.Surface(srect.size, 0, 32).get_view('2')
        for dest in [0, 'a', 'ab', (), (1,), ('a', 2), (1, 'a'), (1 + 2j, 2), (1, 1 + 2j), (1, int), (int, 1)]:
            self.assertRaises(TypeError, font.render_raw_to, surf_buf, text, dest, size=24)

    def test_freetype_Font_text_is_None_with_arr(self):
        if False:
            print('Hello World!')
        f = ft.Font(self._sans_path, 36)
        f.style = ft.STYLE_NORMAL
        f.rotation = 0
        text = 'ABCD'
        get_rect = f.get_rect(text)
        f.vertical = True
        get_rect_vert = f.get_rect(text)
        self.assertTrue(get_rect_vert.width < get_rect.width)
        self.assertTrue(get_rect_vert.height > get_rect.height)
        f.vertical = False
        render_to_surf = pygame.Surface(get_rect.size, pygame.SRCALPHA, 32)
        if IS_PYPY:
            return
        arr = arrinter.Array(get_rect.size, 'u', 1)
        render = f.render(text, (0, 0, 0))
        render_to = f.render_to(render_to_surf, (0, 0), text, (0, 0, 0))
        render_raw = f.render_raw(text)
        render_raw_to = f.render_raw_to(arr, text)
        surf = pygame.Surface(get_rect.size, pygame.SRCALPHA, 32)
        self.assertEqual(f.get_rect(None), get_rect)
        (s, r) = f.render(None, (0, 0, 0))
        self.assertEqual(r, render[1])
        self.assertTrue(surf_same_image(s, render[0]))
        r = f.render_to(surf, (0, 0), None, (0, 0, 0))
        self.assertEqual(r, render_to)
        self.assertTrue(surf_same_image(surf, render_to_surf))
        (px, sz) = f.render_raw(None)
        self.assertEqual(sz, render_raw[1])
        self.assertEqual(px, render_raw[0])
        sz = f.render_raw_to(arr, None)
        self.assertEqual(sz, render_raw_to)

    def test_freetype_Font_text_is_None(self):
        if False:
            while True:
                i = 10
        f = ft.Font(self._sans_path, 36)
        f.style = ft.STYLE_NORMAL
        f.rotation = 0
        text = 'ABCD'
        get_rect = f.get_rect(text)
        f.vertical = True
        get_rect_vert = f.get_rect(text)
        f.vertical = True
        r = f.get_rect(None)
        self.assertEqual(r, get_rect_vert)
        f.vertical = False
        r = f.get_rect(None, style=ft.STYLE_WIDE)
        self.assertEqual(r.height, get_rect.height)
        self.assertTrue(r.width > get_rect.width)
        r = f.get_rect(None)
        self.assertEqual(r, get_rect)
        r = f.get_rect(None, rotation=90)
        self.assertEqual(r.width, get_rect.height)
        self.assertEqual(r.height, get_rect.width)
        self.assertRaises(TypeError, f.get_metrics, None)

    def test_freetype_Font_fgcolor(self):
        if False:
            print('Hello World!')
        f = ft.Font(self._bmp_8_75dpi_path)
        notdef = '\x00'
        f.origin = False
        f.pad = False
        black = pygame.Color('black')
        green = pygame.Color('green')
        alpha128 = pygame.Color(10, 20, 30, 128)
        c = f.fgcolor
        self.assertIsInstance(c, pygame.Color)
        self.assertEqual(c, black)
        (s, r) = f.render(notdef)
        self.assertEqual(s.get_at((0, 0)), black)
        f.fgcolor = green
        self.assertEqual(f.fgcolor, green)
        (s, r) = f.render(notdef)
        self.assertEqual(s.get_at((0, 0)), green)
        f.fgcolor = alpha128
        (s, r) = f.render(notdef)
        self.assertEqual(s.get_at((0, 0)), alpha128)
        surf = pygame.Surface(f.get_rect(notdef).size, pygame.SRCALPHA, 32)
        f.render_to(surf, (0, 0), None)
        self.assertEqual(surf.get_at((0, 0)), alpha128)
        self.assertRaises(AttributeError, setattr, f, 'fgcolor', None)

    def test_freetype_Font_bgcolor(self):
        if False:
            while True:
                i = 10
        f = ft.Font(None, 32)
        zero = '0'
        f.origin = False
        f.pad = False
        transparent_black = pygame.Color(0, 0, 0, 0)
        green = pygame.Color('green')
        alpha128 = pygame.Color(10, 20, 30, 128)
        c = f.bgcolor
        self.assertIsInstance(c, pygame.Color)
        self.assertEqual(c, transparent_black)
        (s, r) = f.render(zero, pygame.Color(255, 255, 255))
        self.assertEqual(s.get_at((0, 0)), transparent_black)
        f.bgcolor = green
        self.assertEqual(f.bgcolor, green)
        (s, r) = f.render(zero)
        self.assertEqual(s.get_at((0, 0)), green)
        f.bgcolor = alpha128
        (s, r) = f.render(zero)
        self.assertEqual(s.get_at((0, 0)), alpha128)
        surf = pygame.Surface(f.get_rect(zero).size, pygame.SRCALPHA, 32)
        f.render_to(surf, (0, 0), None)
        self.assertEqual(surf.get_at((0, 0)), alpha128)
        self.assertRaises(AttributeError, setattr, f, 'bgcolor', None)

    @unittest.skipIf(not pygame.HAVE_NEWBUF, 'newbuf not implemented')
    @unittest.skipIf(IS_PYPY, 'pypy no likey')
    def test_newbuf(self):
        if False:
            i = 10
            return i + 15
        from pygame.tests.test_utils import buftools
        Exporter = buftools.Exporter
        font = self._TEST_FONTS['sans']
        srect = font.get_rect('Hi', size=12)
        for format in ['b', 'B', 'h', 'H', 'i', 'I', 'l', 'L', 'q', 'Q', 'x', '1x', '2x', '3x', '4x', '5x', '6x', '7x', '8x', '9x', '<h', '>h', '=h', '@h', '!h', '1h', '=1h']:
            newbuf = Exporter(srect.size, format=format)
            rrect = font.render_raw_to(newbuf, 'Hi', size=12)
            self.assertEqual(rrect, srect)
        for format in ['f', 'd', '2h', '?', 'hh']:
            newbuf = Exporter(srect.size, format=format, itemsize=4)
            self.assertRaises(ValueError, font.render_raw_to, newbuf, 'Hi', size=12)

    def test_freetype_Font_style(self):
        if False:
            while True:
                i = 10
        font = self._TEST_FONTS['sans']
        self.assertEqual(ft.STYLE_NORMAL, font.style)
        with self.assertRaises(TypeError):
            font.style = 'None'
        with self.assertRaises(TypeError):
            font.style = None
        with self.assertRaises(ValueError):
            font.style = 112
        self.assertEqual(ft.STYLE_NORMAL, font.style)
        font.style = ft.STYLE_UNDERLINE
        self.assertEqual(ft.STYLE_UNDERLINE, font.style)
        st = ft.STYLE_STRONG | ft.STYLE_UNDERLINE | ft.STYLE_OBLIQUE
        font.style = st
        self.assertEqual(st, font.style)
        self.assertNotEqual(st, ft.STYLE_DEFAULT)
        font.style = ft.STYLE_DEFAULT
        self.assertEqual(st, font.style)
        font.style = ft.STYLE_NORMAL
        self.assertEqual(ft.STYLE_NORMAL, font.style)

    def test_freetype_Font_resolution(self):
        if False:
            i = 10
            return i + 15
        text = '|'
        resolution = ft.get_default_resolution()
        new_font = ft.Font(self._sans_path, resolution=2 * resolution)
        self.assertEqual(new_font.resolution, 2 * resolution)
        size_normal = self._TEST_FONTS['sans'].get_rect(text, size=24).size
        size_scaled = new_font.get_rect(text, size=24).size
        size_by_2 = size_normal[0] * 2
        self.assertTrue(size_by_2 + 2 >= size_scaled[0] >= size_by_2 - 2, '%i not equal %i' % (size_scaled[1], size_by_2))
        size_by_2 = size_normal[1] * 2
        self.assertTrue(size_by_2 + 2 >= size_scaled[1] >= size_by_2 - 2, '%i not equal %i' % (size_scaled[1], size_by_2))
        new_resolution = resolution + 10
        ft.set_default_resolution(new_resolution)
        try:
            new_font = ft.Font(self._sans_path, resolution=0)
            self.assertEqual(new_font.resolution, new_resolution)
        finally:
            ft.set_default_resolution()

    def test_freetype_Font_path(self):
        if False:
            return 10
        self.assertEqual(self._TEST_FONTS['sans'].path, self._sans_path)
        self.assertRaises(AttributeError, getattr, nullfont(), 'path')

    def test_freetype_Font_cache(self):
        if False:
            for i in range(10):
                print('nop')
        glyphs = 'abcde'
        glen = len(glyphs)
        other_glyphs = '123'
        oglen = len(other_glyphs)
        uempty = ''
        many_glyphs = uempty.join([chr(i) for i in range(32, 127)])
        mglen = len(many_glyphs)
        count = 0
        access = 0
        hit = 0
        miss = 0
        f = ft.Font(None, size=24, font_index=0, resolution=72, ucs4=False)
        f.style = ft.STYLE_NORMAL
        f.antialiased = True
        self.assertEqual(f._debug_cache_stats, (0, 0, 0, 0, 0))
        count = access = miss = glen
        f.render_raw(glyphs)
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        access += glen
        hit += glen
        f.vertical = True
        f.render_raw(glyphs)
        f.vertical = False
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        count += oglen
        access += oglen
        miss += oglen
        f.render_raw(other_glyphs)
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        count += glen
        access += glen
        miss += glen
        f.render_raw(glyphs, size=12)
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        access += oglen
        hit += oglen
        f.underline = True
        f.render_raw(other_glyphs)
        f.underline = False
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        count += glen
        access += glen
        miss += glen
        f.oblique = True
        f.render_raw(glyphs)
        f.oblique = False
        self.assertEqual(f._debug_cache_stats, (count, 0, access, hit, miss))
        count += glen
        access += glen
        miss += glen
        f.strong = True
        f.render_raw(glyphs)
        f.strong = False
        (ccount, cdelete_count, caccess, chit, cmiss) = f._debug_cache_stats
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))
        count += glen
        access += glen
        miss += glen
        f.render_raw(glyphs, rotation=10)
        (ccount, cdelete_count, caccess, chit, cmiss) = f._debug_cache_stats
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))
        count += oglen
        access += oglen
        miss += oglen
        f.antialiased = False
        f.render_raw(other_glyphs)
        f.antialiased = True
        (ccount, cdelete_count, caccess, chit, cmiss) = f._debug_cache_stats
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))
        count += 2 * mglen
        access += 2 * mglen
        miss += 2 * mglen
        f.get_metrics(many_glyphs, size=8)
        f.get_metrics(many_glyphs, size=10)
        (ccount, cdelete_count, caccess, chit, cmiss) = f._debug_cache_stats
        self.assertTrue(ccount < count)
        self.assertEqual((ccount + cdelete_count, caccess, chit, cmiss), (count, access, hit, miss))
    try:
        ft.Font._debug_cache_stats
    except AttributeError:
        del test_freetype_Font_cache

    def test_undefined_character_code(self):
        if False:
            while True:
                i = 10
        font = self._TEST_FONTS['sans']
        (img, size1) = font.render(chr(1), (0, 0, 0), size=24)
        (img, size0) = font.render('', (0, 0, 0), size=24)
        self.assertTrue(size1.width > size0.width)
        metrics = font.get_metrics(chr(1) + chr(48), size=24)
        self.assertEqual(len(metrics), 2)
        self.assertIsNone(metrics[0])
        self.assertIsInstance(metrics[1], tuple)

    def test_issue_242(self):
        if False:
            while True:
                i = 10
        'Issue #242: get_rect() uses 0 as default style'
        font = self._TEST_FONTS['sans']
        prev_style = font.wide
        font.wide = True
        try:
            rect = font.get_rect('M', size=64)
            (surf, rrect) = font.render(None, size=64)
            self.assertEqual(rect, rrect)
        finally:
            font.wide = prev_style
        prev_style = font.strong
        font.strong = True
        try:
            rect = font.get_rect('Mm_', size=64)
            (surf, rrect) = font.render(None, size=64)
            self.assertEqual(rect, rrect)
        finally:
            font.strong = prev_style
        prev_style = font.oblique
        font.oblique = True
        try:
            rect = font.get_rect('|', size=64)
            (surf, rrect) = font.render(None, size=64)
            self.assertEqual(rect, rrect)
        finally:
            font.oblique = prev_style
        prev_style = font.underline
        font.underline = True
        try:
            rect = font.get_rect(' ', size=64)
            (surf, rrect) = font.render(None, size=64)
            self.assertEqual(rect, rrect)
        finally:
            font.underline = prev_style

    def test_issue_237(self):
        if False:
            for i in range(10):
                print('nop')
        'Issue #237: Memory overrun when rendered with underlining'
        name = 'Times New Roman'
        font = ft.SysFont(name, 19)
        if font.name != name:
            return
        font.underline = True
        (s, r) = font.render('Amazon', size=19)
        for adj in [-2, -1.9, -1, 0, 1.9, 2]:
            font.underline_adjustment = adj
            (s, r) = font.render('Amazon', size=19)

    def test_issue_243(self):
        if False:
            i = 10
            return i + 15
        'Issue Y: trailing space ignored in boundary calculation'
        font = self._TEST_FONTS['fixed']
        r1 = font.get_rect(' ', size=64)
        self.assertTrue(r1.width > 1)
        r2 = font.get_rect('  ', size=64)
        self.assertEqual(r2.width, 2 * r1.width)

    def test_garbage_collection(self):
        if False:
            while True:
                i = 10
        'Check reference counting on returned new references'

        def ref_items(seq):
            if False:
                while True:
                    i = 10
            return [weakref.ref(o) for o in seq]
        font = self._TEST_FONTS['bmp-8-75dpi']
        font.size = font.get_sizes()[0][0]
        text = 'A'
        rect = font.get_rect(text)
        surf = pygame.Surface(rect.size, pygame.SRCALPHA, 32)
        refs = []
        refs.extend(ref_items(font.render(text, (0, 0, 0))))
        refs.append(weakref.ref(font.render_to(surf, (0, 0), text, (0, 0, 0))))
        refs.append(weakref.ref(font.get_rect(text)))
        n = len(refs)
        self.assertTrue(n > 0)
        for i in range(2):
            gc.collect()
        for i in range(n):
            self.assertIsNone(refs[i](), 'ref %d not collected' % i)
        try:
            from sys import getrefcount
        except ImportError:
            pass
        else:
            array = arrinter.Array(rect.size, 'u', 1)
            o = font.render_raw(text)
            self.assertEqual(getrefcount(o), 2)
            self.assertEqual(getrefcount(o[0]), 2)
            self.assertEqual(getrefcount(o[1]), 2)
            self.assertEqual(getrefcount(font.render_raw_to(array, text)), 1)
            o = font.get_metrics('AB')
            self.assertEqual(getrefcount(o), 2)
            for i in range(len(o)):
                self.assertEqual(getrefcount(o[i]), 2, 'refcount fail for item %d' % i)
            o = font.get_sizes()
            self.assertEqual(getrefcount(o), 2)
            for i in range(len(o)):
                self.assertEqual(getrefcount(o[i]), 2, 'refcount fail for item %d' % i)

    def test_display_surface_quit(self):
        if False:
            i = 10
            return i + 15
        'Font.render_to() on a closed display surface'
        null_surface = pygame.Surface.__new__(pygame.Surface)
        f = self._TEST_FONTS['sans']
        self.assertRaises(pygame.error, f.render_to, null_surface, (0, 0), 'Crash!', size=12)

    def test_issue_565(self):
        if False:
            return 10
        'get_metrics supporting rotation/styles/size'
        tests = [{'method': 'size', 'value': 36, 'msg': 'metrics same for size'}, {'method': 'rotation', 'value': 90, 'msg': 'metrics same for rotation'}, {'method': 'oblique', 'value': True, 'msg': 'metrics same for oblique'}]
        text = '|'

        def run_test(method, value, msg):
            if False:
                for i in range(10):
                    print('nop')
            font = ft.Font(self._sans_path, size=24)
            before = font.get_metrics(text)
            font.__setattr__(method, value)
            after = font.get_metrics(text)
            self.assertNotEqual(before, after, msg)
        for test in tests:
            run_test(test['method'], test['value'], test['msg'])

    def test_freetype_SysFont_name(self):
        if False:
            print('Hello World!')
        'that SysFont accepts names of various types'
        fonts = pygame.font.get_fonts()
        size = 12
        font_name = ft.SysFont(fonts[0], size).name
        self.assertFalse(font_name is None)
        names = ','.join(fonts)
        font_name_2 = ft.SysFont(names, size).name
        self.assertEqual(font_name_2, font_name)
        font_name_2 = ft.SysFont(fonts, size).name
        self.assertEqual(font_name_2, font_name)
        names = (name for name in fonts)
        font_name_2 = ft.SysFont(names, size).name
        self.assertEqual(font_name_2, font_name)
        fonts_b = [f.encode() for f in fonts]
        font_name_2 = ft.SysFont(fonts_b[0], size).name
        self.assertEqual(font_name_2, font_name)
        names = b','.join(fonts_b)
        font_name_2 = ft.SysFont(names, size).name
        self.assertEqual(font_name_2, font_name)
        font_name_2 = ft.SysFont(fonts_b, size).name
        self.assertEqual(font_name_2, font_name)
        names = [fonts[0], fonts_b[1], fonts[2], fonts_b[3]]
        font_name_2 = ft.SysFont(names, size).name
        self.assertEqual(font_name_2, font_name)

    def test_pathlib(self):
        if False:
            print('Hello World!')
        f = ft.Font(pathlib.Path(self._fixed_path), 20)

class FreeTypeTest(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        ft.init()

    def tearDown(self):
        if False:
            print('Hello World!')
        ft.quit()

    def test_resolution(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            ft.set_default_resolution()
            resolution = ft.get_default_resolution()
            self.assertEqual(resolution, 72)
            new_resolution = resolution + 10
            ft.set_default_resolution(new_resolution)
            self.assertEqual(ft.get_default_resolution(), new_resolution)
            ft.init(resolution=resolution + 20)
            self.assertEqual(ft.get_default_resolution(), new_resolution)
        finally:
            ft.set_default_resolution()

    def test_autoinit_and_autoquit(self):
        if False:
            print('Hello World!')
        pygame.init()
        self.assertTrue(ft.get_init())
        pygame.quit()
        self.assertFalse(ft.get_init())
        pygame.init()
        self.assertTrue(ft.get_init())
        pygame.quit()
        self.assertFalse(ft.get_init())

    def test_init(self):
        if False:
            i = 10
            return i + 15
        ft.quit()
        ft.init()
        self.assertTrue(ft.get_init())

    def test_init__multiple(self):
        if False:
            i = 10
            return i + 15
        ft.init()
        ft.init()
        self.assertTrue(ft.get_init())

    def test_quit(self):
        if False:
            return 10
        ft.quit()
        self.assertFalse(ft.get_init())

    def test_quit__multiple(self):
        if False:
            for i in range(10):
                print('nop')
        ft.quit()
        ft.quit()
        self.assertFalse(ft.get_init())

    def test_get_init(self):
        if False:
            while True:
                i = 10
        self.assertTrue(ft.get_init())

    def test_cache_size(self):
        if False:
            print('Hello World!')
        DEFAULT_CACHE_SIZE = 64
        self.assertEqual(ft.get_cache_size(), DEFAULT_CACHE_SIZE)
        ft.quit()
        self.assertEqual(ft.get_cache_size(), 0)
        new_cache_size = DEFAULT_CACHE_SIZE * 2
        ft.init(cache_size=new_cache_size)
        self.assertEqual(ft.get_cache_size(), new_cache_size)

    def test_get_error(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures get_error() is initially empty (None).'
        error_msg = ft.get_error()
        self.assertIsNone(error_msg)

    def test_get_version(self):
        if False:
            while True:
                i = 10
        ft.quit()
        self.assertIsNotNone(ft.get_version(linked=False))
        self.assertIsNotNone(ft.get_version(linked=True))
if __name__ == '__main__':
    unittest.main()