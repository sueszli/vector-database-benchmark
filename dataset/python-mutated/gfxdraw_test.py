import unittest
import pygame
import pygame.gfxdraw
from pygame.locals import *
from pygame.tests.test_utils import SurfaceSubclass

def intensity(c, i):
    if False:
        for i in range(10):
            print('nop')
    'Return color c changed by intensity i\n\n    For 0 <= i <= 127 the color is a shade, with 0 being black, 127 being the\n    unaltered color.\n\n    For 128 <= i <= 255 the color is a tint, with 255 being white, 128 the\n    unaltered color.\n\n    '
    (r, g, b) = c[0:3]
    if 0 <= i <= 127:
        return (r * i // 127, g * i // 127, b * i // 127)
    return (r + (255 - r) * (255 - i) // 127, g + (255 - g) * (255 - i) // 127, b + (255 - b) * (255 - i) // 127)

class GfxdrawDefaultTest(unittest.TestCase):
    is_started = False
    foreground_color = (128, 64, 8)
    background_color = (255, 255, 255)

    def make_palette(base_color):
        if False:
            i = 10
            return i + 15
        'Return color palette that is various intensities of base_color'
        return [intensity(base_color, i) for i in range(0, 256)]
    default_palette = make_palette(foreground_color)
    default_size = (100, 100)

    def check_at(self, surf, posn, color):
        if False:
            return 10
        sc = surf.get_at(posn)
        fail_msg = '%s != %s at %s, bitsize: %i, flags: %i, masks: %s' % (sc, color, posn, surf.get_bitsize(), surf.get_flags(), surf.get_masks())
        self.assertEqual(sc, color, fail_msg)

    def check_not_at(self, surf, posn, color):
        if False:
            return 10
        sc = surf.get_at(posn)
        fail_msg = '%s != %s at %s, bitsize: %i, flags: %i, masks: %s' % (sc, color, posn, surf.get_bitsize(), surf.get_flags(), surf.get_masks())
        self.assertNotEqual(sc, color, fail_msg)

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        pygame.init()
        pygame.display.set_mode((1, 1))

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        pygame.quit()

    def setUp(self):
        if False:
            i = 10
            return i + 15
        if not pygame.get_init():
            pygame.init()
        Surface = pygame.Surface
        size = self.default_size
        palette = self.default_palette
        if not self.is_started:
            self.surfaces = [Surface(size, 0, 8), Surface(size, SRCALPHA, 16), Surface(size, SRCALPHA, 32)]
            self.surfaces[0].set_palette(palette)
            nonpalette_fmts = ((12, (3840, 240, 15, 0)), (15, (31744, 992, 31, 0)), (15, (31, 992, 31744, 0)), (16, (3840, 240, 15, 61440)), (16, (61440, 3840, 240, 15)), (16, (15, 240, 3840, 61440)), (16, (240, 3840, 61440, 15)), (16, (31744, 992, 31, 32768)), (16, (63488, 1984, 62, 1)), (16, (31, 992, 31744, 32768)), (16, (62, 1984, 63488, 1)), (16, (63488, 2016, 31, 0)), (16, (31, 2016, 63488, 0)), (24, (255, 65280, 16711680, 0)), (24, (16711680, 65280, 255, 0)), (32, (16711680, 65280, 255, 0)), (32, (4278190080, 16711680, 65280, 0)), (32, (255, 65280, 16711680, 0)), (32, (65280, 16711680, 4278190080, 0)), (32, (16711680, 65280, 255, 4278190080)), (32, (4278190080, 16711680, 65280, 255)), (32, (255, 65280, 16711680, 4278190080)), (32, (65280, 16711680, 4278190080, 255)))
            for (bitsize, masks) in nonpalette_fmts:
                self.surfaces.append(Surface(size, 0, bitsize, masks))
        for surf in self.surfaces:
            surf.fill(self.background_color)

    def test_gfxdraw__subclassed_surface(self):
        if False:
            while True:
                i = 10
        'Ensure pygame.gfxdraw works on subclassed surfaces.'
        surface = SurfaceSubclass((11, 13), SRCALPHA, 32)
        surface.fill(pygame.Color('blue'))
        expected_color = pygame.Color('red')
        (x, y) = (1, 2)
        pygame.gfxdraw.pixel(surface, x, y, expected_color)
        self.assertEqual(surface.get_at((x, y)), expected_color)

    def test_pixel(self):
        if False:
            print('Hello World!')
        'pixel(surface, x, y, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.pixel(surf, 2, 2, fg)
            for x in range(1, 4):
                for y in range(1, 4):
                    if x == 2 and y == 2:
                        self.check_at(surf, (x, y), fg_adjusted)
                    else:
                        self.check_at(surf, (x, y), bg_adjusted)

    def test_hline(self):
        if False:
            return 10
        'hline(surface, x1, x2, y, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        startx = 10
        stopx = 80
        y = 50
        fg_test_points = [(startx, y), (stopx, y), ((stopx - startx) // 2, y)]
        bg_test_points = [(startx - 1, y), (stopx + 1, y), (startx, y - 1), (startx, y + 1), (stopx, y - 1), (stopx, y + 1)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.hline(surf, startx, stopx, y, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_vline(self):
        if False:
            i = 10
            return i + 15
        'vline(surface, x, y1, y2, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 50
        starty = 10
        stopy = 80
        fg_test_points = [(x, starty), (x, stopy), (x, (stopy - starty) // 2)]
        bg_test_points = [(x, starty - 1), (x, stopy + 1), (x - 1, starty), (x + 1, starty), (x - 1, stopy), (x + 1, stopy)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.vline(surf, x, starty, stopy, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_rectangle(self):
        if False:
            print('Hello World!')
        'rectangle(surface, rect, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        rect = pygame.Rect(10, 15, 55, 62)
        rect_tuple = tuple(rect)
        fg_test_points = [rect.topleft, (rect.right - 1, rect.top), (rect.left, rect.bottom - 1), (rect.right - 1, rect.bottom - 1)]
        bg_test_points = [(rect.left - 1, rect.top - 1), (rect.left + 1, rect.top + 1), (rect.right, rect.top - 1), (rect.right - 2, rect.top + 1), (rect.left - 1, rect.bottom), (rect.left + 1, rect.bottom - 2), (rect.right, rect.bottom), (rect.right - 2, rect.bottom - 2)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.rectangle(surf, rect, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)
            surf.fill(bg)
            pygame.gfxdraw.rectangle(surf, rect_tuple, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_box(self):
        if False:
            for i in range(10):
                print('nop')
        'box(surface, rect, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        rect = pygame.Rect(10, 15, 55, 62)
        rect_tuple = tuple(rect)
        fg_test_points = [rect.topleft, (rect.left + 1, rect.top + 1), (rect.right - 1, rect.top), (rect.right - 2, rect.top + 1), (rect.left, rect.bottom - 1), (rect.left + 1, rect.bottom - 2), (rect.right - 1, rect.bottom - 1), (rect.right - 2, rect.bottom - 2)]
        bg_test_points = [(rect.left - 1, rect.top - 1), (rect.right, rect.top - 1), (rect.left - 1, rect.bottom), (rect.right, rect.bottom)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.box(surf, rect, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)
            surf.fill(bg)
            pygame.gfxdraw.box(surf, rect_tuple, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_line(self):
        if False:
            print('Hello World!')
        'line(surface, x1, y1, x2, y2, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x1 = 10
        y1 = 15
        x2 = 92
        y2 = 77
        fg_test_points = [(x1, y1), (x2, y2)]
        bg_test_points = [(x1 - 1, y1), (x1, y1 - 1), (x1 - 1, y1 - 1), (x2 + 1, y2), (x2, y2 + 1), (x2 + 1, y2 + 1)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.line(surf, x1, y1, x2, y2, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_circle(self):
        if False:
            for i in range(10):
                print('nop')
        'circle(surface, x, y, r, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        r = 30
        fg_test_points = [(x, y - r), (x, y + r), (x - r, y), (x + r, y)]
        bg_test_points = [(x, y), (x, y - r + 1), (x, y - r - 1), (x, y + r + 1), (x, y + r - 1), (x - r - 1, y), (x - r + 1, y), (x + r + 1, y), (x + r - 1, y)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.circle(surf, x, y, r, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_arc(self):
        if False:
            print('Hello World!')
        'arc(surface, x, y, r, start, end, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        r = 30
        start = 0
        end = 90
        fg_test_points = [(x, y + r), (x + r, y + 1)]
        bg_test_points = [(x, y), (x, y - r), (x - r, y), (x, y + r + 1), (x, y + r - 1), (x - 1, y + r), (x + r + 1, y), (x + r - 1, y), (x + r, y)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.arc(surf, x, y, r, start, end, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_aacircle(self):
        if False:
            for i in range(10):
                print('nop')
        'aacircle(surface, x, y, r, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        r = 30
        fg_test_points = [(x, y - r), (x, y + r), (x - r, y), (x + r, y)]
        bg_test_points = [(x, y), (x, y - r + 1), (x, y - r - 1), (x, y + r + 1), (x, y + r - 1), (x - r - 1, y), (x - r + 1, y), (x + r + 1, y), (x + r - 1, y)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.aacircle(surf, x, y, r, fg)
            for posn in fg_test_points:
                self.check_not_at(surf, posn, bg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_filled_circle(self):
        if False:
            while True:
                i = 10
        'filled_circle(surface, x, y, r, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        r = 30
        fg_test_points = [(x, y - r), (x, y - r + 1), (x, y + r), (x, y + r - 1), (x - r, y), (x - r + 1, y), (x + r, y), (x + r - 1, y), (x, y)]
        bg_test_points = [(x, y - r - 1), (x, y + r + 1), (x - r - 1, y), (x + r + 1, y)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.filled_circle(surf, x, y, r, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_ellipse(self):
        if False:
            i = 10
            return i + 15
        'ellipse(surface, x, y, rx, ry, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        rx = 30
        ry = 35
        fg_test_points = [(x, y - ry), (x, y + ry), (x - rx, y), (x + rx, y)]
        bg_test_points = [(x, y), (x, y - ry + 1), (x, y - ry - 1), (x, y + ry + 1), (x, y + ry - 1), (x - rx - 1, y), (x - rx + 1, y), (x + rx + 1, y), (x + rx - 1, y)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.ellipse(surf, x, y, rx, ry, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_aaellipse(self):
        if False:
            i = 10
            return i + 15
        'aaellipse(surface, x, y, rx, ry, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        rx = 30
        ry = 35
        fg_test_points = [(x, y - ry), (x, y + ry), (x - rx, y), (x + rx, y)]
        bg_test_points = [(x, y), (x, y - ry + 1), (x, y - ry - 1), (x, y + ry + 1), (x, y + ry - 1), (x - rx - 1, y), (x - rx + 1, y), (x + rx + 1, y), (x + rx - 1, y)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.aaellipse(surf, x, y, rx, ry, fg)
            for posn in fg_test_points:
                self.check_not_at(surf, posn, bg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_filled_ellipse(self):
        if False:
            while True:
                i = 10
        'filled_ellipse(surface, x, y, rx, ry, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        rx = 30
        ry = 35
        fg_test_points = [(x, y - ry), (x, y - ry + 1), (x, y + ry), (x, y + ry - 1), (x - rx, y), (x - rx + 1, y), (x + rx, y), (x + rx - 1, y), (x, y)]
        bg_test_points = [(x, y - ry - 1), (x, y + ry + 1), (x - rx - 1, y), (x + rx + 1, y)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.filled_ellipse(surf, x, y, rx, ry, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_pie(self):
        if False:
            while True:
                i = 10
        'pie(surface, x, y, r, start, end, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x = 45
        y = 40
        r = 30
        start = 0
        end = 90
        fg_test_points = [(x, y), (x + 1, y), (x, y + 1), (x + r, y)]
        bg_test_points = [(x - 1, y), (x, y - 1), (x - 1, y - 1), (x + 1, y + 1), (x + r + 1, y), (x + r, y - 1), (x, y + r + 1)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.pie(surf, x, y, r, start, end, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_trigon(self):
        if False:
            print('Hello World!')
        'trigon(surface, x1, y1, x2, y2, x3, y3, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x1 = 10
        y1 = 15
        x2 = 92
        y2 = 77
        x3 = 20
        y3 = 60
        fg_test_points = [(x1, y1), (x2, y2), (x3, y3)]
        bg_test_points = [(x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (x3 - 1, y3 + 1), (x1 + 10, y1 + 30)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.trigon(surf, x1, y1, x2, y2, x3, y3, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_aatrigon(self):
        if False:
            return 10
        'aatrigon(surface, x1, y1, x2, y2, x3, y3, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x1 = 10
        y1 = 15
        x2 = 92
        y2 = 77
        x3 = 20
        y3 = 60
        fg_test_points = [(x1, y1), (x2, y2), (x3, y3)]
        bg_test_points = [(x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (x3 - 1, y3 + 1), (x1 + 10, y1 + 30)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.aatrigon(surf, x1, y1, x2, y2, x3, y3, fg)
            for posn in fg_test_points:
                self.check_not_at(surf, posn, bg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_aatrigon__with_horizontal_edge(self):
        if False:
            while True:
                i = 10
        'Ensure aatrigon draws horizontal edges correctly.\n\n        This test creates 2 surfaces and draws an aatrigon on each. The pixels\n        on each surface are compared to ensure they are the same. The only\n        difference between the 2 aatrigons is the order the points are drawn.\n        The order of the points should have no impact on the final drawing.\n\n        Related to issue #622.\n        '
        bg_color = pygame.Color('white')
        line_color = pygame.Color('black')
        (width, height) = (11, 10)
        expected_surface = pygame.Surface((width, height), 0, 32)
        expected_surface.fill(bg_color)
        surface = pygame.Surface((width, height), 0, 32)
        surface.fill(bg_color)
        (x1, y1) = (width - 1, 0)
        (x2, y2) = ((width - 1) // 2, height - 1)
        (x3, y3) = (0, 0)
        pygame.gfxdraw.aatrigon(expected_surface, x1, y1, x2, y2, x3, y3, line_color)
        pygame.gfxdraw.aatrigon(surface, x3, y3, x2, y2, x1, y1, line_color)
        expected_surface.lock()
        surface.lock()
        for x in range(width):
            for y in range(height):
                self.assertEqual(expected_surface.get_at((x, y)), surface.get_at((x, y)), f'pos=({x}, {y})')
        surface.unlock()
        expected_surface.unlock()

    def test_filled_trigon(self):
        if False:
            print('Hello World!')
        'filled_trigon(surface, x1, y1, x2, y2, x3, y3, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        x1 = 10
        y1 = 15
        x2 = 92
        y2 = 77
        x3 = 20
        y3 = 60
        fg_test_points = [(x1, y1), (x2, y2), (x3, y3), (x1 + 10, y1 + 30)]
        bg_test_points = [(x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (x3 - 1, y3 + 1)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.filled_trigon(surf, x1, y1, x2, y2, x3, y3, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_polygon(self):
        if False:
            while True:
                i = 10
        'polygon(surface, points, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        points = [(10, 80), (10, 15), (92, 25), (92, 80)]
        fg_test_points = points + [(points[0][0], points[0][1] - 1), (points[0][0] + 1, points[0][1]), (points[3][0] - 1, points[3][1]), (points[3][0], points[3][1] - 1), (points[2][0], points[2][1] + 1)]
        bg_test_points = [(points[0][0] - 1, points[0][1]), (points[0][0], points[0][1] + 1), (points[0][0] - 1, points[0][1] + 1), (points[0][0] + 1, points[0][1] - 1), (points[3][0] + 1, points[3][1]), (points[3][0], points[3][1] + 1), (points[3][0] + 1, points[3][1] + 1), (points[3][0] - 1, points[3][1] - 1), (points[2][0] + 1, points[2][1]), (points[2][0] - 1, points[2][1] + 1), (points[1][0] - 1, points[1][1]), (points[1][0], points[1][1] - 1), (points[1][0] - 1, points[1][1] - 1)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.polygon(surf, points, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_aapolygon(self):
        if False:
            i = 10
            return i + 15
        'aapolygon(surface, points, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        points = [(10, 80), (10, 15), (92, 25), (92, 80)]
        fg_test_points = points
        bg_test_points = [(points[0][0] - 1, points[0][1]), (points[0][0], points[0][1] + 1), (points[0][0] - 1, points[0][1] + 1), (points[0][0] + 1, points[0][1] - 1), (points[3][0] + 1, points[3][1]), (points[3][0], points[3][1] + 1), (points[3][0] + 1, points[3][1] + 1), (points[3][0] - 1, points[3][1] - 1), (points[2][0] + 1, points[2][1]), (points[2][0] - 1, points[2][1] + 1), (points[1][0] - 1, points[1][1]), (points[1][0], points[1][1] - 1), (points[1][0] - 1, points[1][1] - 1)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.aapolygon(surf, points, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_not_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_aapolygon__with_horizontal_edge(self):
        if False:
            while True:
                i = 10
        'Ensure aapolygon draws horizontal edges correctly.\n\n        This test creates 2 surfaces and draws a polygon on each. The pixels\n        on each surface are compared to ensure they are the same. The only\n        difference between the 2 polygons is that one is drawn using\n        aapolygon() and the other using multiple line() calls. They should\n        produce the same final drawing.\n\n        Related to issue #622.\n        '
        bg_color = pygame.Color('white')
        line_color = pygame.Color('black')
        (width, height) = (11, 10)
        expected_surface = pygame.Surface((width, height), 0, 32)
        expected_surface.fill(bg_color)
        surface = pygame.Surface((width, height), 0, 32)
        surface.fill(bg_color)
        points = ((0, 0), (0, height - 1), (width - 1, height - 1), (width - 1, 0))
        for ((x1, y1), (x2, y2)) in zip(points, points[1:] + points[:1]):
            pygame.gfxdraw.line(expected_surface, x1, y1, x2, y2, line_color)
        pygame.gfxdraw.aapolygon(surface, points, line_color)
        expected_surface.lock()
        surface.lock()
        for x in range(width):
            for y in range(height):
                self.assertEqual(expected_surface.get_at((x, y)), surface.get_at((x, y)), f'pos=({x}, {y})')
        surface.unlock()
        expected_surface.unlock()

    def test_filled_polygon(self):
        if False:
            print('Hello World!')
        'filled_polygon(surface, points, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        points = [(10, 80), (10, 15), (92, 25), (92, 80)]
        fg_test_points = points + [(points[0][0], points[0][1] - 1), (points[0][0] + 1, points[0][1]), (points[0][0] + 1, points[0][1] - 1), (points[3][0] - 1, points[3][1]), (points[3][0], points[3][1] - 1), (points[3][0] - 1, points[3][1] - 1), (points[2][0], points[2][1] + 1), (points[2][0] - 1, points[2][1] + 1)]
        bg_test_points = [(points[0][0] - 1, points[0][1]), (points[0][0], points[0][1] + 1), (points[0][0] - 1, points[0][1] + 1), (points[3][0] + 1, points[3][1]), (points[3][0], points[3][1] + 1), (points[3][0] + 1, points[3][1] + 1), (points[2][0] + 1, points[2][1]), (points[1][0] - 1, points[1][1]), (points[1][0], points[1][1] - 1), (points[1][0] - 1, points[1][1] - 1)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.filled_polygon(surf, points, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)

    def test_textured_polygon(self):
        if False:
            i = 10
            return i + 15
        'textured_polygon(surface, points, texture, tx, ty): return None'
        (w, h) = self.default_size
        fg = self.foreground_color
        bg = self.background_color
        tx = 0
        ty = 0
        texture = pygame.Surface((w + tx, h + ty), 0, 24)
        texture.fill(fg, (0, 0, w, h))
        points = [(10, 80), (10, 15), (92, 25), (92, 80)]
        fg_test_points = [(points[1][0] + 30, points[1][1] + 40)]
        bg_test_points = [(points[0][0] - 1, points[0][1]), (points[0][0], points[0][1] + 1), (points[0][0] - 1, points[0][1] + 1), (points[3][0] + 1, points[3][1]), (points[3][0], points[3][1] + 1), (points[3][0] + 1, points[3][1] + 1), (points[2][0] + 1, points[2][1]), (points[1][0] - 1, points[1][1]), (points[1][0], points[1][1] - 1), (points[1][0] - 1, points[1][1] - 1)]
        for surf in self.surfaces[1:]:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.textured_polygon(surf, points, texture, -tx, -ty)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)
        texture = pygame.Surface(self.default_size, SRCALPHA, 32)
        self.assertRaises(ValueError, pygame.gfxdraw.textured_polygon, self.surfaces[0], points, texture, 0, 0)

    def test_bezier(self):
        if False:
            return 10
        'bezier(surface, points, steps, color): return None'
        fg = self.foreground_color
        bg = self.background_color
        points = [(10, 50), (25, 15), (60, 80), (92, 30)]
        fg_test_points = [points[0], points[3]]
        bg_test_points = [(points[0][0] - 1, points[0][1]), (points[3][0] + 1, points[3][1]), (points[1][0], points[1][1] + 3), (points[2][0], points[2][1] - 3)]
        for surf in self.surfaces:
            fg_adjusted = surf.unmap_rgb(surf.map_rgb(fg))
            bg_adjusted = surf.unmap_rgb(surf.map_rgb(bg))
            pygame.gfxdraw.bezier(surf, points, 30, fg)
            for posn in fg_test_points:
                self.check_at(surf, posn, fg_adjusted)
            for posn in bg_test_points:
                self.check_at(surf, posn, bg_adjusted)
if __name__ == '__main__':
    unittest.main()