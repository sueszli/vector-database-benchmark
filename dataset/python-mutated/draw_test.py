import math
import unittest
import sys
import warnings
import pygame
from pygame import draw
from pygame import draw_py
from pygame.locals import SRCALPHA
from pygame.tests import test_utils
from pygame.math import Vector2
RED = BG_RED = pygame.Color('red')
GREEN = FG_GREEN = pygame.Color('green')
RECT_POSITION_ATTRIBUTES = ('topleft', 'midtop', 'topright', 'midright', 'bottomright', 'midbottom', 'bottomleft', 'midleft', 'center')

def get_border_values(surface, width, height):
    if False:
        i = 10
        return i + 15
    "Returns a list containing lists with the values of the surface's\n    borders.\n    "
    border_top = [surface.get_at((x, 0)) for x in range(width)]
    border_left = [surface.get_at((0, y)) for y in range(height)]
    border_right = [surface.get_at((width - 1, y)) for y in range(height)]
    border_bottom = [surface.get_at((x, height - 1)) for x in range(width)]
    return [border_top, border_left, border_right, border_bottom]

def corners(surface):
    if False:
        i = 10
        return i + 15
    'Returns a tuple with the corner positions of the given surface.\n\n    Clockwise from the top left corner.\n    '
    (width, height) = surface.get_size()
    return ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))

def rect_corners_mids_and_center(rect):
    if False:
        for i in range(10):
            print('nop')
    'Returns a tuple with each corner, mid, and the center for a given rect.\n\n    Clockwise from the top left corner and ending with the center point.\n    '
    return (rect.topleft, rect.midtop, rect.topright, rect.midright, rect.bottomright, rect.midbottom, rect.bottomleft, rect.midleft, rect.center)

def border_pos_and_color(surface):
    if False:
        for i in range(10):
            print('nop')
    'Yields each border position and its color for a given surface.\n\n    Clockwise from the top left corner.\n    '
    (width, height) = surface.get_size()
    (right, bottom) = (width - 1, height - 1)
    for x in range(width):
        pos = (x, 0)
        yield (pos, surface.get_at(pos))
    for y in range(1, height):
        pos = (right, y)
        yield (pos, surface.get_at(pos))
    for x in range(right - 1, -1, -1):
        pos = (x, bottom)
        yield (pos, surface.get_at(pos))
    for y in range(bottom - 1, 0, -1):
        pos = (0, y)
        yield (pos, surface.get_at(pos))

def get_color_points(surface, color, bounds_rect=None, match_color=True):
    if False:
        while True:
            i = 10
    'Get all the points of a given color on the surface within the given\n    bounds.\n\n    If bounds_rect is None the full surface is checked.\n    If match_color is True, all points matching the color are returned,\n        otherwise all points not matching the color are returned.\n    '
    get_at = surface.get_at
    if bounds_rect is None:
        x_range = range(surface.get_width())
        y_range = range(surface.get_height())
    else:
        x_range = range(bounds_rect.left, bounds_rect.right)
        y_range = range(bounds_rect.top, bounds_rect.bottom)
    surface.lock()
    if match_color:
        pts = [(x, y) for x in x_range for y in y_range if get_at((x, y)) == color]
    else:
        pts = [(x, y) for x in x_range for y in y_range if get_at((x, y)) != color]
    surface.unlock()
    return pts

def create_bounding_rect(surface, surf_color, default_pos):
    if False:
        i = 10
        return i + 15
    "Create a rect to bound all the pixels that don't match surf_color.\n\n    The default_pos parameter is used to position the bounding rect for the\n    case where all pixels match the surf_color.\n    "
    (width, height) = surface.get_clip().size
    (xmin, ymin) = (width, height)
    (xmax, ymax) = (-1, -1)
    get_at = surface.get_at
    surface.lock()
    for y in range(height):
        for x in range(width):
            if get_at((x, y)) != surf_color:
                xmin = min(x, xmin)
                xmax = max(x, xmax)
                ymin = min(y, ymin)
                ymax = max(y, ymax)
    surface.unlock()
    if -1 == xmax:
        return pygame.Rect(default_pos, (0, 0))
    return pygame.Rect((xmin, ymin), (xmax - xmin + 1, ymax - ymin + 1))

class InvalidBool:
    """To help test invalid bool values."""
    __bool__ = None

class DrawTestCase(unittest.TestCase):
    """Base class to test draw module functions."""
    draw_rect = staticmethod(draw.rect)
    draw_polygon = staticmethod(draw.polygon)
    draw_circle = staticmethod(draw.circle)
    draw_ellipse = staticmethod(draw.ellipse)
    draw_arc = staticmethod(draw.arc)
    draw_line = staticmethod(draw.line)
    draw_lines = staticmethod(draw.lines)
    draw_aaline = staticmethod(draw.aaline)
    draw_aalines = staticmethod(draw.aalines)

class PythonDrawTestCase(unittest.TestCase):
    """Base class to test draw_py module functions."""
    draw_polygon = staticmethod(draw_py.draw_polygon)
    draw_line = staticmethod(draw_py.draw_line)
    draw_lines = staticmethod(draw_py.draw_lines)
    draw_aaline = staticmethod(draw_py.draw_aaline)
    draw_aalines = staticmethod(draw_py.draw_aalines)

class DrawEllipseMixin:
    """Mixin tests for drawing ellipses.

    This class contains all the general ellipse drawing tests.
    """

    def test_ellipse__args(self):
        if False:
            while True:
                i = 10
        'Ensures draw ellipse accepts the correct args.'
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), pygame.Rect((0, 0), (3, 2)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__args_without_width(self):
        if False:
            while True:
                i = 10
        'Ensures draw ellipse accepts the args without a width.'
        bounds_rect = self.draw_ellipse(pygame.Surface((2, 2)), (1, 1, 1, 99), pygame.Rect((1, 1), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__args_with_negative_width(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw ellipse accepts the args with negative width.'
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), pygame.Rect((2, 3), (3, 2)), -1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(2, 3, 0, 0))

    def test_ellipse__args_with_width_gt_radius(self):
        if False:
            while True:
                i = 10
        'Ensures draw ellipse accepts the args with\n        width > rect.w // 2 and width > rect.h // 2.\n        '
        rect = pygame.Rect((0, 0), (4, 4))
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), rect, rect.w // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        bounds_rect = self.draw_ellipse(pygame.Surface((3, 3)), (0, 10, 0, 50), rect, rect.h // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__kwargs(self):
        if False:
            while True:
                i = 10
        'Ensures draw ellipse accepts the correct kwargs\n        with and without a width arg.\n        '
        kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'rect': pygame.Rect((0, 0), (3, 2)), 'width': 1}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'rect': (0, 0, 1, 1)}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__kwargs_order_independent(self):
        if False:
            i = 10
            return i + 15
        "Ensures draw ellipse's kwargs are not order dependent."
        bounds_rect = self.draw_ellipse(color=(1, 2, 3), surface=pygame.Surface((3, 2)), width=0, rect=pygame.Rect((1, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__args_missing(self):
        if False:
            while True:
                i = 10
        'Ensures draw ellipse detects any missing required args.'
        surface = pygame.Surface((1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, pygame.Color('red'))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse()

    def test_ellipse__kwargs_missing(self):
        if False:
            return 10
        'Ensures draw ellipse detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'rect': pygame.Rect((1, 0), (2, 2)), 'width': 2}
        for name in ('rect', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**invalid_kwargs)

    def test_ellipse__arg_invalid_types(self):
        if False:
            print('Hello World!')
        'Ensures draw ellipse detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        rect = pygame.Rect((1, 1), (1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, color, rect, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, color, (1, 2, 3, 4, 5), 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(surface, 2.3, rect, 0)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_ellipse(rect, color, rect, 2)

    def test_ellipse__kwarg_invalid_types(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw ellipse detects invalid kwarg types.'
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        rect = pygame.Rect((0, 1), (1, 1))
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'width': 1}, {'surface': surface, 'color': 2.3, 'rect': rect, 'width': 1}, {'surface': surface, 'color': color, 'rect': (0, 0, 0), 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1.1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**kwargs)

    def test_ellipse__kwarg_invalid_name(self):
        if False:
            while True:
                i = 10
        'Ensures draw ellipse detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        rect = pygame.Rect((0, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**kwargs)

    def test_ellipse__args_and_kwargs(self):
        if False:
            while True:
                i = 10
        'Ensures draw ellipse accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        rect = pygame.Rect((1, 0), (2, 1))
        width = 0
        kwargs = {'surface': surface, 'color': color, 'rect': rect, 'width': width}
        for name in ('surface', 'color', 'rect', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_ellipse(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_ellipse(surface, color, **kwargs)
            elif 'rect' == name:
                bounds_rect = self.draw_ellipse(surface, color, rect, **kwargs)
            else:
                bounds_rect = self.draw_ellipse(surface, color, rect, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__valid_width_values(self):
        if False:
            return 10
        'Ensures draw ellipse accepts different width values.'
        pos = (1, 1)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'rect': pygame.Rect(pos, (3, 2)), 'width': None}
        for width in (-1000, -10, -1, 0, 1, 10, 1000):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__valid_rect_formats(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw ellipse accepts different rect formats.'
        pos = (1, 1)
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'rect': None, 'width': 0}
        rects = (pygame.Rect(pos, (1, 3)), (pos, (2, 1)), (pos[0], pos[1], 1, 1))
        for rect in rects:
            surface.fill(surface_color)
            kwargs['rect'] = rect
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__valid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw ellipse accepts different color formats.'
        pos = (1, 1)
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 2)), 'width': 0}
        reds = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in reds:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_ellipse(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_ellipse__invalid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw ellipse handles invalid color formats correctly.'
        pos = (1, 1)
        surface = pygame.Surface((4, 3))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (2, 2)), 'width': 1}
        for expected_color in (2.3, surface):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_ellipse(**kwargs)

    def test_ellipse(self):
        if False:
            return 10
        'Tests ellipses of differing sizes on surfaces of differing sizes.\n\n        Checks if the number of sides touching the border of the surface is\n        correct.\n        '
        left_top = [(0, 0), (1, 0), (0, 1), (1, 1)]
        sizes = [(4, 4), (5, 4), (4, 5), (5, 5)]
        color = (1, 13, 24, 255)

        def same_size(width, height, border_width):
            if False:
                i = 10
                return i + 15
            'Test for ellipses with the same size as the surface.'
            surface = pygame.Surface((width, height))
            self.draw_ellipse(surface, color, (0, 0, width, height), border_width)
            borders = get_border_values(surface, width, height)
            for border in borders:
                self.assertTrue(color in border)

        def not_same_size(width, height, border_width, left, top):
            if False:
                i = 10
                return i + 15
            "Test for ellipses that aren't the same size as the surface."
            surface = pygame.Surface((width, height))
            self.draw_ellipse(surface, color, (left, top, width - 1, height - 1), border_width)
            borders = get_border_values(surface, width, height)
            sides_touching = [color in border for border in borders].count(True)
            self.assertEqual(sides_touching, 2)
        for (width, height) in sizes:
            for border_width in (0, 1):
                same_size(width, height, border_width)
                for (left, top) in left_top:
                    not_same_size(width, height, border_width, left, top)

    def test_ellipse__big_ellipse(self):
        if False:
            for i in range(10):
                print('nop')
        'Test for big ellipse that could overflow in algorithm'
        width = 1025
        height = 1025
        border = 1
        x_value_test = int(0.4 * height)
        y_value_test = int(0.4 * height)
        surface = pygame.Surface((width, height))
        self.draw_ellipse(surface, (255, 0, 0), (0, 0, width, height), border)
        colored_pixels = 0
        for y in range(height):
            if surface.get_at((x_value_test, y)) == (255, 0, 0):
                colored_pixels += 1
        for x in range(width):
            if surface.get_at((x, y_value_test)) == (255, 0, 0):
                colored_pixels += 1
        self.assertEqual(colored_pixels, border * 4)

    def test_ellipse__thick_line(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures a thick lined ellipse is drawn correctly.'
        ellipse_color = pygame.Color('yellow')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((40, 40))
        rect = pygame.Rect((0, 0), (31, 23))
        rect.center = surface.get_rect().center
        for thickness in range(1, min(*rect.size) // 2 - 2):
            surface.fill(surface_color)
            self.draw_ellipse(surface, ellipse_color, rect, thickness)
            surface.lock()
            x = rect.centerx
            y_start = rect.top
            y_end = rect.top + thickness - 1
            for y in range(y_start, y_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x, y_start - 1)), surface_color, thickness)
            self.assertEqual(surface.get_at((x, y_end + 1)), surface_color, thickness)
            x = rect.centerx
            y_start = rect.bottom - thickness
            y_end = rect.bottom - 1
            for y in range(y_start, y_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x, y_start - 1)), surface_color, thickness)
            self.assertEqual(surface.get_at((x, y_end + 1)), surface_color, thickness)
            x_start = rect.left
            x_end = rect.left + thickness - 1
            y = rect.centery
            for x in range(x_start, x_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x_start - 1, y)), surface_color, thickness)
            self.assertEqual(surface.get_at((x_end + 1, y)), surface_color, thickness)
            x_start = rect.right - thickness
            x_end = rect.right - 1
            y = rect.centery
            for x in range(x_start, x_end + 1):
                self.assertEqual(surface.get_at((x, y)), ellipse_color, thickness)
            self.assertEqual(surface.get_at((x_start - 1, y)), surface_color, thickness)
            self.assertEqual(surface.get_at((x_end + 1, y)), surface_color, thickness)
            surface.unlock()

    def test_ellipse__no_holes(self):
        if False:
            print('Hello World!')
        width = 80
        height = 70
        surface = pygame.Surface((width + 1, height))
        rect = pygame.Rect(0, 0, width, height)
        for thickness in range(1, 37, 5):
            surface.fill('BLACK')
            self.draw_ellipse(surface, 'RED', rect, thickness)
            for y in range(height):
                number_of_changes = 0
                drawn_pixel = False
                for x in range(width + 1):
                    if not drawn_pixel and surface.get_at((x, y)) == pygame.Color('RED') or (drawn_pixel and surface.get_at((x, y)) == pygame.Color('BLACK')):
                        drawn_pixel = not drawn_pixel
                        number_of_changes += 1
                if y < thickness or y > height - thickness - 1:
                    self.assertEqual(number_of_changes, 2)
                else:
                    self.assertEqual(number_of_changes, 4)

    def test_ellipse__max_width(self):
        if False:
            while True:
                i = 10
        'Ensures an ellipse with max width (and greater) is drawn correctly.'
        ellipse_color = pygame.Color('yellow')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((40, 40))
        rect = pygame.Rect((0, 0), (31, 21))
        rect.center = surface.get_rect().center
        max_thickness = (min(*rect.size) + 1) // 2
        for thickness in range(max_thickness, max_thickness + 3):
            surface.fill(surface_color)
            self.draw_ellipse(surface, ellipse_color, rect, thickness)
            surface.lock()
            for y in range(rect.top, rect.bottom):
                self.assertEqual(surface.get_at((rect.centerx, y)), ellipse_color)
            for x in range(rect.left, rect.right):
                self.assertEqual(surface.get_at((x, rect.centery)), ellipse_color)
            self.assertEqual(surface.get_at((rect.centerx, rect.top - 1)), surface_color)
            self.assertEqual(surface.get_at((rect.centerx, rect.bottom + 1)), surface_color)
            self.assertEqual(surface.get_at((rect.left - 1, rect.centery)), surface_color)
            self.assertEqual(surface.get_at((rect.right + 1, rect.centery)), surface_color)
            surface.unlock()

    def _check_1_pixel_sized_ellipse(self, surface, collide_rect, surface_color, ellipse_color):
        if False:
            i = 10
            return i + 15
        (surf_w, surf_h) = surface.get_size()
        surface.lock()
        for pos in ((x, y) for y in range(surf_h) for x in range(surf_w)):
            if collide_rect.collidepoint(pos):
                expected_color = ellipse_color
            else:
                expected_color = surface_color
            self.assertEqual(surface.get_at(pos), expected_color, f'collide_rect={collide_rect}, pos={pos}')
        surface.unlock()

    def test_ellipse__1_pixel_width(self):
        if False:
            i = 10
            return i + 15
        'Ensures an ellipse with a width of 1 is drawn correctly.\n\n        An ellipse with a width of 1 pixel is a vertical line.\n        '
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        (surf_w, surf_h) = (10, 20)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (1, 0))
        collide_rect = rect.copy()
        off_left = -1
        off_right = surf_w
        off_bottom = surf_h
        center_x = surf_w // 2
        center_y = surf_h // 2
        for ellipse_h in range(6, 10):
            collide_rect.h = ellipse_h
            rect.h = ellipse_h
            off_top = -(ellipse_h + 1)
            half_off_top = -(ellipse_h // 2)
            half_off_bottom = surf_h - ellipse_h // 2
            positions = ((off_left, off_top), (off_left, half_off_top), (off_left, center_y), (off_left, half_off_bottom), (off_left, off_bottom), (center_x, off_top), (center_x, half_off_top), (center_x, center_y), (center_x, half_off_bottom), (center_x, off_bottom), (off_right, off_top), (off_right, half_off_top), (off_right, center_y), (off_right, half_off_bottom), (off_right, off_bottom))
            for rect_pos in positions:
                surface.fill(surface_color)
                rect.topleft = rect_pos
                collide_rect.topleft = rect_pos
                self.draw_ellipse(surface, ellipse_color, rect)
                self._check_1_pixel_sized_ellipse(surface, collide_rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_width_spanning_surface(self):
        if False:
            i = 10
            return i + 15
        'Ensures an ellipse with a width of 1 is drawn correctly\n        when spanning the height of the surface.\n\n        An ellipse with a width of 1 pixel is a vertical line.\n        '
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        (surf_w, surf_h) = (10, 20)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (1, surf_h + 2))
        positions = ((-1, -1), (0, -1), (surf_w // 2, -1), (surf_w - 1, -1), (surf_w, -1))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_height(self):
        if False:
            while True:
                i = 10
        'Ensures an ellipse with a height of 1 is drawn correctly.\n\n        An ellipse with a height of 1 pixel is a horizontal line.\n        '
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        (surf_w, surf_h) = (20, 10)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (0, 1))
        collide_rect = rect.copy()
        off_right = surf_w
        off_top = -1
        off_bottom = surf_h
        center_x = surf_w // 2
        center_y = surf_h // 2
        for ellipse_w in range(6, 10):
            collide_rect.w = ellipse_w
            rect.w = ellipse_w
            off_left = -(ellipse_w + 1)
            half_off_left = -(ellipse_w // 2)
            half_off_right = surf_w - ellipse_w // 2
            positions = ((off_left, off_top), (half_off_left, off_top), (center_x, off_top), (half_off_right, off_top), (off_right, off_top), (off_left, center_y), (half_off_left, center_y), (center_x, center_y), (half_off_right, center_y), (off_right, center_y), (off_left, off_bottom), (half_off_left, off_bottom), (center_x, off_bottom), (half_off_right, off_bottom), (off_right, off_bottom))
            for rect_pos in positions:
                surface.fill(surface_color)
                rect.topleft = rect_pos
                collide_rect.topleft = rect_pos
                self.draw_ellipse(surface, ellipse_color, rect)
                self._check_1_pixel_sized_ellipse(surface, collide_rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_height_spanning_surface(self):
        if False:
            print('Hello World!')
        'Ensures an ellipse with a height of 1 is drawn correctly\n        when spanning the width of the surface.\n\n        An ellipse with a height of 1 pixel is a horizontal line.\n        '
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        (surf_w, surf_h) = (20, 10)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (surf_w + 2, 1))
        positions = ((-1, -1), (-1, 0), (-1, surf_h // 2), (-1, surf_h - 1), (-1, surf_h))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)

    def test_ellipse__1_pixel_width_and_height(self):
        if False:
            i = 10
            return i + 15
        'Ensures an ellipse with a width and height of 1 is drawn correctly.\n\n        An ellipse with a width and height of 1 pixel is a single pixel.\n        '
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        (surf_w, surf_h) = (10, 10)
        surface = pygame.Surface((surf_w, surf_h))
        rect = pygame.Rect((0, 0), (1, 1))
        off_left = -1
        off_right = surf_w
        off_top = -1
        off_bottom = surf_h
        left_edge = 0
        right_edge = surf_w - 1
        top_edge = 0
        bottom_edge = surf_h - 1
        center_x = surf_w // 2
        center_y = surf_h // 2
        positions = ((off_left, off_top), (off_left, top_edge), (off_left, center_y), (off_left, bottom_edge), (off_left, off_bottom), (left_edge, off_top), (left_edge, top_edge), (left_edge, center_y), (left_edge, bottom_edge), (left_edge, off_bottom), (center_x, off_top), (center_x, top_edge), (center_x, center_y), (center_x, bottom_edge), (center_x, off_bottom), (right_edge, off_top), (right_edge, top_edge), (right_edge, center_y), (right_edge, bottom_edge), (right_edge, off_bottom), (off_right, off_top), (off_right, top_edge), (off_right, center_y), (off_right, bottom_edge), (off_right, off_bottom))
        for rect_pos in positions:
            surface.fill(surface_color)
            rect.topleft = rect_pos
            self.draw_ellipse(surface, ellipse_color, rect)
            self._check_1_pixel_sized_ellipse(surface, rect, surface_color, ellipse_color)

    def test_ellipse__bounding_rect(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw ellipse returns the correct bounding rect.\n\n        Tests ellipses on and off the surface and a range of width/thickness\n        values.\n        '
        ellipse_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        min_width = min_height = 5
        max_width = max_height = 7
        sizes = ((min_width, min_height), (max_width, max_height))
        surface = pygame.Surface((20, 20), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for attr in RECT_POSITION_ATTRIBUTES:
                for (width, height) in sizes:
                    ellipse_rect = pygame.Rect((0, 0), (width, height))
                    setattr(ellipse_rect, attr, pos)
                    for thickness in (0, 1, 2, 3, min(width, height)):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_ellipse(surface, ellipse_color, ellipse_rect, thickness)
                        expected_rect = create_bounding_rect(surface, surf_color, ellipse_rect.topleft)
                        self.assertEqual(bounding_rect, expected_rect)

    def test_ellipse__surface_clip(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensures draw ellipse respects a surface's clip area.\n\n        Tests drawing the ellipse filled and unfilled.\n        "
        surfw = surfh = 30
        ellipse_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_ellipse(surface, ellipse_color, pos_rect, width)
                expected_pts = get_color_points(surface, ellipse_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_ellipse(surface, ellipse_color, pos_rect, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = ellipse_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()

class DrawEllipseTest(DrawEllipseMixin, DrawTestCase):
    """Test draw module function ellipse.

    This class inherits the general tests from DrawEllipseMixin. It is also
    the class to add any draw.ellipse specific tests to.
    """

class BaseLineMixin:
    """Mixin base for drawing various lines.

    This class contains general helper methods and setup for testing the
    different types of lines.
    """
    COLORS = ((0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 255, 255))

    @staticmethod
    def _create_surfaces():
        if False:
            while True:
                i = 10
        surfaces = []
        for size in ((49, 49), (50, 50)):
            for depth in (8, 16, 24, 32):
                for flags in (0, SRCALPHA):
                    surface = pygame.display.set_mode(size, flags, depth)
                    surfaces.append(surface)
                    surfaces.append(surface.convert_alpha())
        return surfaces

    @staticmethod
    def _rect_lines(rect):
        if False:
            return 10
        for pt in rect_corners_mids_and_center(rect):
            if pt in [rect.midleft, rect.center]:
                continue
            yield (rect.midleft, pt)
            yield (pt, rect.midleft)

class LineMixin(BaseLineMixin):
    """Mixin test for drawing a single line.

    This class contains all the general single line drawing tests.
    """

    def test_line__args(self):
        if False:
            return 10
        'Ensures draw line accepts the correct args.'
        bounds_rect = self.draw_line(pygame.Surface((3, 3)), (0, 10, 0, 50), (0, 0), (1, 1), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__args_without_width(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw line accepts the args without a width.'
        bounds_rect = self.draw_line(pygame.Surface((2, 2)), (0, 0, 0, 50), (0, 0), (2, 2))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__kwargs(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw line accepts the correct kwargs\n        with and without a width arg.\n        '
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        start_pos = (1, 1)
        end_pos = (2, 2)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_line(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__kwargs_order_independent(self):
        if False:
            return 10
        "Ensures draw line's kwargs are not order dependent."
        bounds_rect = self.draw_line(start_pos=(1, 2), end_pos=(2, 1), width=2, color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__args_missing(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw line detects any missing required args.'
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, (0, 0))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line()

    def test_line__kwargs_missing(self):
        if False:
            return 10
        'Ensures draw line detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'start_pos': (2, 1), 'end_pos': (2, 2), 'width': 1}
        for name in ('end_pos', 'start_pos', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**invalid_kwargs)

    def test_line__arg_invalid_types(self):
        if False:
            while True:
                i = 10
        'Ensures draw line detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        start_pos = (0, 1)
        end_pos = (1, 2)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, start_pos, end_pos, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, start_pos, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, color, (1,), end_pos)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line(surface, 2.3, start_pos, end_pos)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_line((1, 2, 3, 4), color, start_pos, end_pos)

    def test_line__kwarg_invalid_types(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw line detects invalid kwarg types.'
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        start_pos = (1, 0)
        end_pos = (2, 0)
        width = 1
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': 2.3, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': color, 'start_pos': (0, 0, 0), 'end_pos': end_pos, 'width': width}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': (0,), 'width': width}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1.2}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__kwarg_invalid_name(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw line detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        start_pos = (1, 1)
        end_pos = (2, 0)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__args_and_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw line accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        start_pos = (0, 1)
        end_pos = (1, 2)
        width = 0
        kwargs = {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'width': width}
        for name in ('surface', 'color', 'start_pos', 'end_pos', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_line(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_line(surface, color, **kwargs)
            elif 'start_pos' == name:
                bounds_rect = self.draw_line(surface, color, start_pos, **kwargs)
            elif 'end_pos' == name:
                bounds_rect = self.draw_line(surface, color, start_pos, end_pos, **kwargs)
            else:
                bounds_rect = self.draw_line(surface, color, start_pos, end_pos, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__valid_width_values(self):
        if False:
            while True:
                i = 10
        'Ensures draw line accepts different width values.'
        line_color = pygame.Color('yellow')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        pos = (2, 1)
        kwargs = {'surface': surface, 'color': line_color, 'start_pos': pos, 'end_pos': (2, 2), 'width': None}
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = line_color if width > 0 else surface_color
            bounds_rect = self.draw_line(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__valid_start_pos_formats(self):
        if False:
            while True:
                i = 10
        'Ensures draw line accepts different start_pos formats.'
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': None, 'end_pos': (2, 2), 'width': 2}
        (x, y) = (2, 1)
        for start_pos in ((x, y), (x + 0.1, y), (x, y + 0.1), (x + 0.1, y + 0.1)):
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['start_pos'] = seq_type(start_pos)
                bounds_rect = self.draw_line(**kwargs)
                self.assertEqual(surface.get_at((x, y)), expected_color)
                self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__valid_end_pos_formats(self):
        if False:
            while True:
                i = 10
        'Ensures draw line accepts different end_pos formats.'
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': (2, 1), 'end_pos': None, 'width': 2}
        (x, y) = (2, 2)
        for end_pos in ((x, y), (x + 0.2, y), (x, y + 0.2), (x + 0.2, y + 0.2)):
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['end_pos'] = seq_type(end_pos)
                bounds_rect = self.draw_line(**kwargs)
                self.assertEqual(surface.get_at((x, y)), expected_color)
                self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__invalid_start_pos_formats(self):
        if False:
            print('Hello World!')
        'Ensures draw line handles invalid start_pos formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': None, 'end_pos': (2, 2), 'width': 1}
        start_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for start_pos in start_pos_fmts:
            kwargs['start_pos'] = start_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__invalid_end_pos_formats(self):
        if False:
            return 10
        'Ensures draw line handles invalid end_pos formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': (2, 2), 'end_pos': None, 'width': 1}
        end_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for end_pos in end_pos_fmts:
            kwargs['end_pos'] = end_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__valid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw line accepts different color formats.'
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'start_pos': pos, 'end_pos': (2, 1), 'width': 3}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_line(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_line__invalid_color_formats(self):
        if False:
            return 10
        'Ensures draw line handles invalid color formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'start_pos': (1, 1), 'end_pos': (2, 1), 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_line(**kwargs)

    def test_line__color(self):
        if False:
            while True:
                i = 10
        'Tests if the line drawn is the correct color.'
        pos = (0, 0)
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_line(surface, expected_color, pos, (1, 0))
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__color_with_thickness(self):
        if False:
            i = 10
            return i + 15
        'Ensures a thick line is drawn using the correct color.'
        from_x = 5
        to_x = 10
        y = 5
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_line(surface, expected_color, (from_x, y), (to_x, y), 5)
                for pos in ((x, y + i) for i in (-2, 0, 2) for x in (from_x, to_x)):
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__gaps(self):
        if False:
            while True:
                i = 10
        'Tests if the line drawn contains any gaps.'
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            width = surface.get_width()
            self.draw_line(surface, expected_color, (0, 0), (width - 1, 0))
            for x in range(width):
                pos = (x, 0)
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__gaps_with_thickness(self):
        if False:
            i = 10
            return i + 15
        'Ensures a thick line is drawn without any gaps.'
        expected_color = (255, 255, 255)
        thickness = 5
        for surface in self._create_surfaces():
            width = surface.get_width() - 1
            h = width // 5
            w = h * 5
            self.draw_line(surface, expected_color, (0, 5), (w, 5 + h), thickness)
            for x in range(w + 1):
                for y in range(3, 8):
                    pos = (x, y + (x + 2) // 5)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_line__bounding_rect(self):
        if False:
            return 10
        'Ensures draw line returns the correct bounding rect.\n\n        Tests lines with endpoints on and off the surface and a range of\n        width/thickness values.\n        '
        if isinstance(self, PythonDrawTestCase):
            self.skipTest('bounding rects not supported in draw_py.draw_line')
        line_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        width = height = 30
        helper_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                helper_rect.center = pos
                for thickness in range(-1, 5):
                    for (start, end) in self._rect_lines(helper_rect):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_line(surface, line_color, start, end, thickness)
                        if 0 < thickness:
                            expected_rect = create_bounding_rect(surface, surf_color, start)
                        else:
                            expected_rect = pygame.Rect(start, (0, 0))
                        self.assertEqual(bounding_rect, expected_rect, 'start={}, end={}, size={}, thickness={}'.format(start, end, size, thickness))

    def test_line__surface_clip(self):
        if False:
            i = 10
            return i + 15
        "Ensures draw line respects a surface's clip area."
        surfw = surfh = 30
        line_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for thickness in (1, 3):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_line(surface, line_color, pos_rect.midtop, pos_rect.midbottom, thickness)
                expected_pts = get_color_points(surface, line_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_line(surface, line_color, pos_rect.midtop, pos_rect.midbottom, thickness)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = line_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()

class DrawLineTest(LineMixin, DrawTestCase):
    """Test draw module function line.

    This class inherits the general tests from LineMixin. It is also the class
    to add any draw.line specific tests to.
    """

    def test_line_endianness(self):
        if False:
            return 10
        'test color component order'
        for depth in (24, 32):
            surface = pygame.Surface((5, 3), 0, depth)
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_line(surface, pygame.Color(255, 0, 0), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).r, 0, 'there should be red here')
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_line(surface, pygame.Color(0, 0, 255), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).b, 0, 'there should be blue here')

    def test_line(self):
        if False:
            for i in range(10):
                print('nop')
        self.surf_size = (320, 200)
        self.surf = pygame.Surface(self.surf_size, pygame.SRCALPHA)
        self.color = (1, 13, 24, 205)
        drawn = draw.line(self.surf, self.color, (1, 0), (200, 0))
        self.assertEqual(drawn.right, 201, 'end point arg should be (or at least was) inclusive')
        for pt in test_utils.rect_area_pts(drawn):
            self.assertEqual(self.surf.get_at(pt), self.color)
        for pt in test_utils.rect_outer_bounds(drawn):
            self.assertNotEqual(self.surf.get_at(pt), self.color)
        line_width = 2
        offset = 5
        a = (offset, offset)
        b = (self.surf_size[0] - offset, a[1])
        c = (a[0], self.surf_size[1] - offset)
        d = (b[0], c[1])
        e = (a[0] + offset, c[1])
        f = (b[0], c[0] + 5)
        lines = [(a, d), (b, c), (c, b), (d, a), (a, b), (b, a), (a, c), (c, a), (a, e), (e, a), (a, f), (f, a), (a, a)]
        for (p1, p2) in lines:
            msg = f'{p1} - {p2}'
            if p1[0] <= p2[0]:
                plow = p1
                phigh = p2
            else:
                plow = p2
                phigh = p1
            self.surf.fill((0, 0, 0))
            rec = draw.line(self.surf, (255, 255, 255), p1, p2, line_width)
            xinc = yinc = 0
            if abs(p1[0] - p2[0]) > abs(p1[1] - p2[1]):
                yinc = 1
            else:
                xinc = 1
            for i in range(line_width):
                p = (p1[0] + xinc * i, p1[1] + yinc * i)
                self.assertEqual(self.surf.get_at(p), (255, 255, 255), msg)
                p = (p2[0] + xinc * i, p2[1] + yinc * i)
                self.assertEqual(self.surf.get_at(p), (255, 255, 255), msg)
            p = (plow[0] - 1, plow[1])
            self.assertEqual(self.surf.get_at(p), (0, 0, 0), msg)
            p = (plow[0] + xinc * line_width, plow[1] + yinc * line_width)
            self.assertEqual(self.surf.get_at(p), (0, 0, 0), msg)
            p = (phigh[0] + xinc * line_width, phigh[1] + yinc * line_width)
            self.assertEqual(self.surf.get_at(p), (0, 0, 0), msg)
            if p1[0] < p2[0]:
                rx = p1[0]
            else:
                rx = p2[0]
            if p1[1] < p2[1]:
                ry = p1[1]
            else:
                ry = p2[1]
            w = abs(p2[0] - p1[0]) + 1 + xinc * (line_width - 1)
            h = abs(p2[1] - p1[1]) + 1 + yinc * (line_width - 1)
            msg += f', {rec}'
            self.assertEqual(rec, (rx, ry, w, h), msg)

    def test_line_for_gaps(self):
        if False:
            i = 10
            return i + 15
        width = 200
        height = 200
        surf = pygame.Surface((width, height), pygame.SRCALPHA)

        def white_surrounded_pixels(x, y):
            if False:
                while True:
                    i = 10
            offsets = [(1, 0), (0, 1), (-1, 0), (0, -1)]
            WHITE = (255, 255, 255, 255)
            return len([1 for (dx, dy) in offsets if surf.get_at((x + dx, y + dy)) == WHITE])

        def check_white_line(start, end):
            if False:
                print('Hello World!')
            surf.fill((0, 0, 0))
            pygame.draw.line(surf, (255, 255, 255), start, end, 30)
            BLACK = (0, 0, 0, 255)
            for x in range(1, width - 1):
                for y in range(1, height - 1):
                    if surf.get_at((x, y)) == BLACK:
                        self.assertTrue(white_surrounded_pixels(x, y) < 3)
        check_white_line((50, 50), (140, 0))
        check_white_line((50, 50), (0, 120))
        check_white_line((50, 50), (199, 198))

class LinesMixin(BaseLineMixin):
    """Mixin test for drawing lines.

    This class contains all the general lines drawing tests.
    """

    def test_lines__args(self):
        if False:
            print('Hello World!')
        'Ensures draw lines accepts the correct args.'
        bounds_rect = self.draw_lines(pygame.Surface((3, 3)), (0, 10, 0, 50), False, ((0, 0), (1, 1)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__args_without_width(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw lines accepts the args without a width.'
        bounds_rect = self.draw_lines(pygame.Surface((2, 2)), (0, 0, 0, 50), False, ((0, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw lines accepts the correct kwargs\n        with and without a width arg.\n        '
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        points = ((0, 0), (1, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': False, 'points': points, 'width': 1}, {'surface': surface, 'color': color, 'closed': False, 'points': points}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_lines(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__kwargs_order_independent(self):
        if False:
            i = 10
            return i + 15
        "Ensures draw lines's kwargs are not order dependent."
        bounds_rect = self.draw_lines(closed=1, points=((0, 0), (1, 1), (2, 2)), width=2, color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__args_missing(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw lines detects any missing required args.'
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, 0)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines()

    def test_lines__kwargs_missing(self):
        if False:
            while True:
                i = 10
        'Ensures draw lines detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'closed': 1, 'points': ((2, 2), (1, 1)), 'width': 1}
        for name in ('points', 'closed', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**invalid_kwargs)

    def test_lines__arg_invalid_types(self):
        if False:
            return 10
        'Ensures draw lines detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        closed = 0
        points = ((1, 2), (2, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, closed, points, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, closed, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, color, InvalidBool(), points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines(surface, 2.3, closed, points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_lines((1, 2, 3, 4), color, closed, points)

    def test_lines__kwarg_invalid_types(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw lines detects invalid kwarg types.'
        valid_kwargs = {'surface': pygame.Surface((3, 3)), 'color': pygame.Color('green'), 'closed': False, 'points': ((1, 2), (2, 1)), 'width': 1}
        invalid_kwargs = {'surface': pygame.Surface, 'color': 2.3, 'closed': InvalidBool(), 'points': (0, 0, 0), 'width': 1.2}
        for kwarg in ('surface', 'color', 'closed', 'points', 'width'):
            kwargs = dict(valid_kwargs)
            kwargs[kwarg] = invalid_kwargs[kwarg]
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__kwarg_invalid_name(self):
        if False:
            print('Hello World!')
        'Ensures draw lines detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        closed = 1
        points = ((1, 2), (2, 1))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': closed, 'points': points, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__args_and_kwargs(self):
        if False:
            print('Hello World!')
        'Ensures draw lines accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        closed = 0
        points = ((1, 2), (2, 1))
        width = 1
        kwargs = {'surface': surface, 'color': color, 'closed': closed, 'points': points, 'width': width}
        for name in ('surface', 'color', 'closed', 'points', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_lines(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_lines(surface, color, **kwargs)
            elif 'closed' == name:
                bounds_rect = self.draw_lines(surface, color, closed, **kwargs)
            elif 'points' == name:
                bounds_rect = self.draw_lines(surface, color, closed, points, **kwargs)
            else:
                bounds_rect = self.draw_lines(surface, color, closed, points, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__valid_width_values(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw lines accepts different width values.'
        line_color = pygame.Color('yellow')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': line_color, 'closed': False, 'points': (pos, (2, 1)), 'width': None}
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = line_color if width > 0 else surface_color
            bounds_rect = self.draw_lines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__valid_points_format(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw lines accepts different points formats.'
        expected_color = (10, 20, 30, 255)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'closed': False, 'points': None, 'width': 1}
        point_types = ((tuple, tuple, tuple, tuple), (list, list, list, list), (Vector2, Vector2, Vector2, Vector2), (list, Vector2, tuple, Vector2))
        point_values = (((1, 1), (2, 1), (2, 2), (1, 2)), ((1, 1), (2.2, 1), (2.1, 2.2), (1, 2.1)))
        seq_types = (tuple, list)
        for point_type in point_types:
            for values in point_values:
                check_pos = values[0]
                points = [point_type[i](pt) for (i, pt) in enumerate(values)]
                for seq_type in seq_types:
                    surface.fill(surface_color)
                    kwargs['points'] = seq_type(points)
                    bounds_rect = self.draw_lines(**kwargs)
                    self.assertEqual(surface.get_at(check_pos), expected_color)
                    self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__invalid_points_formats(self):
        if False:
            return 10
        'Ensures draw lines handles invalid points formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None, 'width': 1}
        points_fmts = (((1, 1), (2,)), ((1, 1), (2, 2, 2)), ((1, 1), (2, '2')), ((1, 1), {2, 3}), ((1, 1), dict(((2, 2), (3, 3)))), {(1, 1), (1, 2)}, dict(((1, 1), (4, 4))))
        for points in points_fmts:
            kwargs['points'] = points
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__invalid_points_values(self):
        if False:
            print('Hello World!')
        'Ensures draw lines handles invalid points values correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None, 'width': 1}
        for points in ([], ((1, 1),)):
            for seq_type in (tuple, list):
                kwargs['points'] = seq_type(points)
                with self.assertRaises(ValueError):
                    bounds_rect = self.draw_lines(**kwargs)

    def test_lines__valid_closed_values(self):
        if False:
            print('Hello World!')
        'Ensures draw lines accepts different closed values.'
        line_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        pos = (1, 2)
        kwargs = {'surface': surface, 'color': line_color, 'closed': None, 'points': ((1, 1), (3, 1), (3, 3), (1, 3)), 'width': 1}
        true_values = (-7, 1, 10, '2', 3.1, (4,), [5], True)
        false_values = (None, '', 0, (), [], False)
        for closed in true_values + false_values:
            surface.fill(surface_color)
            kwargs['closed'] = closed
            expected_color = line_color if closed else surface_color
            bounds_rect = self.draw_lines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__valid_color_formats(self):
        if False:
            print('Hello World!')
        'Ensures draw lines accepts different color formats.'
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'closed': False, 'points': (pos, (2, 1)), 'width': 3}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_lines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_lines__invalid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw lines handles invalid color formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'closed': False, 'points': ((1, 1), (1, 2)), 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_lines(**kwargs)

    def test_lines__color(self):
        if False:
            print('Hello World!')
        'Tests if the lines drawn are the correct color.\n\n        Draws lines around the border of the given surface and checks if all\n        borders of the surface only contain the given color.\n        '
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_lines(surface, expected_color, True, corners(surface))
                for (pos, color) in border_pos_and_color(surface):
                    self.assertEqual(color, expected_color, f'pos={pos}')

    def test_lines__color_with_thickness(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures thick lines are drawn using the correct color.'
        x_left = y_top = 5
        for surface in self._create_surfaces():
            x_right = surface.get_width() - 5
            y_bottom = surface.get_height() - 5
            endpoints = ((x_left, y_top), (x_right, y_top), (x_right, y_bottom), (x_left, y_bottom))
            for expected_color in self.COLORS:
                self.draw_lines(surface, expected_color, True, endpoints, 3)
                for t in (-1, 0, 1):
                    for x in range(x_left, x_right + 1):
                        for y in (y_top, y_bottom):
                            pos = (x, y + t)
                            self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
                    for y in range(y_top, y_bottom + 1):
                        for x in (x_left, x_right):
                            pos = (x + t, y)
                            self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_lines__gaps(self):
        if False:
            i = 10
            return i + 15
        'Tests if the lines drawn contain any gaps.\n\n        Draws lines around the border of the given surface and checks if\n        all borders of the surface contain any gaps.\n        '
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            self.draw_lines(surface, expected_color, True, corners(surface))
            for (pos, color) in border_pos_and_color(surface):
                self.assertEqual(color, expected_color, f'pos={pos}')

    def test_lines__gaps_with_thickness(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures thick lines are drawn without any gaps.'
        expected_color = (255, 255, 255)
        x_left = y_top = 5
        for surface in self._create_surfaces():
            h = (surface.get_width() - 11) // 5
            w = h * 5
            x_right = x_left + w
            y_bottom = y_top + h
            endpoints = ((x_left, y_top), (x_right, y_top), (x_right, y_bottom))
            self.draw_lines(surface, expected_color, True, endpoints, 3)
            for x in range(x_left, x_right + 1):
                for t in (-1, 0, 1):
                    pos = (x, y_top + t)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
                    pos = (x, y_top + t + (x - 3) // 5)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')
            for y in range(y_top, y_bottom + 1):
                for t in (-1, 0, 1):
                    pos = (x_right + t, y)
                    self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_lines__bounding_rect(self):
        if False:
            print('Hello World!')
        'Ensures draw lines returns the correct bounding rect.\n\n        Tests lines with endpoints on and off the surface and a range of\n        width/thickness values.\n        '
        line_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        width = height = 30
        pos_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                pos_rect.center = pos
                pts = (pos_rect.midleft, pos_rect.midtop, pos_rect.midright)
                pos = pts[0]
                for thickness in range(-1, 5):
                    for closed in (True, False):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_lines(surface, line_color, closed, pts, thickness)
                        if 0 < thickness:
                            expected_rect = create_bounding_rect(surface, surf_color, pos)
                        else:
                            expected_rect = pygame.Rect(pos, (0, 0))
                        self.assertEqual(bounding_rect, expected_rect)

    def test_lines__surface_clip(self):
        if False:
            while True:
                i = 10
        "Ensures draw lines respects a surface's clip area."
        surfw = surfh = 30
        line_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for center in rect_corners_mids_and_center(clip_rect):
            pos_rect.center = center
            pts = (pos_rect.midtop, pos_rect.center, pos_rect.midbottom)
            for closed in (True, False):
                for thickness in (1, 3):
                    surface.set_clip(None)
                    surface.fill(surface_color)
                    self.draw_lines(surface, line_color, closed, pts, thickness)
                    expected_pts = get_color_points(surface, line_color, clip_rect)
                    surface.fill(surface_color)
                    surface.set_clip(clip_rect)
                    self.draw_lines(surface, line_color, closed, pts, thickness)
                    surface.lock()
                    for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                        if pt in expected_pts:
                            expected_color = line_color
                        else:
                            expected_color = surface_color
                        self.assertEqual(surface.get_at(pt), expected_color, pt)
                    surface.unlock()

class DrawLinesTest(LinesMixin, DrawTestCase):
    """Test draw module function lines.

    This class inherits the general tests from LinesMixin. It is also the class
    to add any draw.lines specific tests to.
    """

class AALineMixin(BaseLineMixin):
    """Mixin test for drawing a single aaline.

    This class contains all the general single aaline drawing tests.
    """

    def test_aaline__args(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw aaline accepts the correct args.'
        bounds_rect = self.draw_aaline(pygame.Surface((3, 3)), (0, 10, 0, 50), (0, 0), (1, 1), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__args_without_blend(self):
        if False:
            while True:
                i = 10
        'Ensures draw aaline accepts the args without a blend.'
        bounds_rect = self.draw_aaline(pygame.Surface((2, 2)), (0, 0, 0, 50), (0, 0), (2, 2))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__blend_warning(self):
        if False:
            print('Hello World!')
        'From pygame 2, blend=False should raise DeprecationWarning.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.draw_aaline(pygame.Surface((2, 2)), (0, 0, 0, 50), (0, 0), (2, 2), False)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_aaline__kwargs(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw aaline accepts the correct kwargs'
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        start_pos = (1, 1)
        end_pos = (2, 2)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_aaline(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__kwargs_order_independent(self):
        if False:
            print('Hello World!')
        "Ensures draw aaline's kwargs are not order dependent."
        bounds_rect = self.draw_aaline(start_pos=(1, 2), end_pos=(2, 1), color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__args_missing(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw aaline detects any missing required args.'
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color, (0, 0))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline()

    def test_aaline__kwargs_missing(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw aaline detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'start_pos': (2, 1), 'end_pos': (2, 2)}
        for name in ('end_pos', 'start_pos', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**invalid_kwargs)

    def test_aaline__arg_invalid_types(self):
        if False:
            return 10
        'Ensures draw aaline detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        start_pos = (0, 1)
        end_pos = (1, 2)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color, start_pos, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline(surface, color, (1,), end_pos)
        with self.assertRaises(ValueError):
            bounds_rect = self.draw_aaline(surface, 'invalid-color', start_pos, end_pos)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aaline((1, 2, 3, 4), color, start_pos, end_pos)

    def test_aaline__kwarg_invalid_types(self):
        if False:
            print('Hello World!')
        'Ensures draw aaline detects invalid kwarg types.'
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        start_pos = (1, 0)
        end_pos = (2, 0)
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}, {'surface': surface, 'color': 2.3, 'start_pos': start_pos, 'end_pos': end_pos}, {'surface': surface, 'color': color, 'start_pos': (0, 0, 0), 'end_pos': end_pos}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': (0,)}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__kwarg_invalid_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw aaline detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        start_pos = (1, 1)
        end_pos = (2, 0)
        kwargs_list = [{'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}, {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__args_and_kwargs(self):
        if False:
            print('Hello World!')
        'Ensures draw aaline accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        start_pos = (0, 1)
        end_pos = (1, 2)
        kwargs = {'surface': surface, 'color': color, 'start_pos': start_pos, 'end_pos': end_pos}
        for name in ('surface', 'color', 'start_pos', 'end_pos'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_aaline(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_aaline(surface, color, **kwargs)
            elif 'start_pos' == name:
                bounds_rect = self.draw_aaline(surface, color, start_pos, **kwargs)
            elif 'end_pos' == name:
                bounds_rect = self.draw_aaline(surface, color, start_pos, end_pos, **kwargs)
            else:
                bounds_rect = self.draw_aaline(surface, color, start_pos, end_pos, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__valid_start_pos_formats(self):
        if False:
            return 10
        'Ensures draw aaline accepts different start_pos formats.'
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': None, 'end_pos': (2, 2)}
        (x, y) = (2, 1)
        positions = ((x, y), (x + 0.01, y), (x, y + 0.01), (x + 0.01, y + 0.01))
        for start_pos in positions:
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['start_pos'] = seq_type(start_pos)
                bounds_rect = self.draw_aaline(**kwargs)
                color = surface.get_at((x, y))
                for (i, sub_color) in enumerate(expected_color):
                    self.assertGreaterEqual(color[i] + 6, sub_color, start_pos)
                self.assertIsInstance(bounds_rect, pygame.Rect, start_pos)

    def test_aaline__valid_end_pos_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw aaline accepts different end_pos formats.'
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'start_pos': (2, 1), 'end_pos': None}
        (x, y) = (2, 2)
        positions = ((x, y), (x + 0.02, y), (x, y + 0.02), (x + 0.02, y + 0.02))
        for end_pos in positions:
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['end_pos'] = seq_type(end_pos)
                bounds_rect = self.draw_aaline(**kwargs)
                color = surface.get_at((x, y))
                for (i, sub_color) in enumerate(expected_color):
                    self.assertGreaterEqual(color[i] + 15, sub_color, end_pos)
                self.assertIsInstance(bounds_rect, pygame.Rect, end_pos)

    def test_aaline__invalid_start_pos_formats(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw aaline handles invalid start_pos formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': None, 'end_pos': (2, 2)}
        start_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for start_pos in start_pos_fmts:
            kwargs['start_pos'] = start_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__invalid_end_pos_formats(self):
        if False:
            print('Hello World!')
        'Ensures draw aaline handles invalid end_pos formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'start_pos': (2, 2), 'end_pos': None}
        end_pos_fmts = ((2,), (2, 1, 0), (2, '1'), {2, 1}, dict(((2, 1),)))
        for end_pos in end_pos_fmts:
            kwargs['end_pos'] = end_pos
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__valid_color_formats(self):
        if False:
            while True:
                i = 10
        'Ensures draw aaline accepts different color formats.'
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'start_pos': pos, 'end_pos': (2, 1)}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_aaline(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aaline__invalid_color_formats(self):
        if False:
            return 10
        'Ensures draw aaline handles invalid color formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'start_pos': (1, 1), 'end_pos': (2, 1)}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aaline(**kwargs)

    def test_aaline__color(self):
        if False:
            print('Hello World!')
        'Tests if the aaline drawn is the correct color.'
        pos = (0, 0)
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_aaline(surface, expected_color, pos, (1, 0))
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_aaline__gaps(self):
        if False:
            return 10
        'Tests if the aaline drawn contains any gaps.\n\n        See: #512\n        '
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            width = surface.get_width()
            self.draw_aaline(surface, expected_color, (0, 0), (width - 1, 0))
            for x in range(width):
                pos = (x, 0)
                self.assertEqual(surface.get_at(pos), expected_color, f'pos={pos}')

    def test_aaline__bounding_rect(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw aaline returns the correct bounding rect.\n\n        Tests lines with endpoints on and off the surface.\n        '
        line_color = pygame.Color('red')
        surf_color = pygame.Color('blue')
        width = height = 30
        helper_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                helper_rect.center = pos
                for (start, end) in self._rect_lines(helper_rect):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_aaline(surface, line_color, start, end)
                    expected_rect = create_bounding_rect(surface, surf_color, start)
                    self.assertEqual(bounding_rect, expected_rect)

    def test_aaline__surface_clip(self):
        if False:
            print('Hello World!')
        "Ensures draw aaline respects a surface's clip area."
        surfw = surfh = 30
        aaline_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for center in rect_corners_mids_and_center(clip_rect):
            pos_rect.center = center
            surface.set_clip(None)
            surface.fill(surface_color)
            self.draw_aaline(surface, aaline_color, pos_rect.midtop, pos_rect.midbottom)
            expected_pts = get_color_points(surface, surface_color, clip_rect, False)
            surface.fill(surface_color)
            surface.set_clip(clip_rect)
            self.draw_aaline(surface, aaline_color, pos_rect.midtop, pos_rect.midbottom)
            surface.lock()
            for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                if pt in expected_pts:
                    self.assertNotEqual(surface.get_at(pt), surface_color, pt)
                else:
                    self.assertEqual(surface.get_at(pt), surface_color, pt)
            surface.unlock()

class DrawAALineTest(AALineMixin, DrawTestCase):
    """Test draw module function aaline.

    This class inherits the general tests from AALineMixin. It is also the
    class to add any draw.aaline specific tests to.
    """

    def test_aaline_endianness(self):
        if False:
            for i in range(10):
                print('nop')
        'test color component order'
        for depth in (24, 32):
            surface = pygame.Surface((5, 3), 0, depth)
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_aaline(surface, pygame.Color(255, 0, 0), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).r, 0, 'there should be red here')
            surface.fill(pygame.Color(0, 0, 0))
            self.draw_aaline(surface, pygame.Color(0, 0, 255), (0, 1), (2, 1), 1)
            self.assertGreater(surface.get_at((1, 1)).b, 0, 'there should be blue here')

    def _check_antialiasing(self, from_point, to_point, should, check_points, set_endpoints=True):
        if False:
            print('Hello World!')
        'Draw a line between two points and check colors of check_points.'
        if set_endpoints:
            should[from_point] = should[to_point] = FG_GREEN

        def check_one_direction(from_point, to_point, should):
            if False:
                i = 10
                return i + 15
            self.draw_aaline(self.surface, FG_GREEN, from_point, to_point, True)
            for pt in check_points:
                color = should.get(pt, BG_RED)
                with self.subTest(from_pt=from_point, pt=pt, to=to_point):
                    self.assertEqual(self.surface.get_at(pt), color)
            draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_one_direction(from_point, to_point, should)
        if from_point != to_point:
            check_one_direction(to_point, from_point, should)

    def test_short_non_antialiased_lines(self):
        if False:
            i = 10
            return i + 15
        'test very short not anti aliased lines in all directions.'
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(3, 8) for j in range(3, 8)]

        def check_both_directions(from_pt, to_pt, other_points):
            if False:
                i = 10
                return i + 15
            should = {pt: FG_GREEN for pt in other_points}
            self._check_antialiasing(from_pt, to_pt, should, check_points)
        check_both_directions((5, 5), (5, 5), [])
        check_both_directions((4, 7), (5, 7), [])
        check_both_directions((5, 4), (7, 4), [(6, 4)])
        check_both_directions((5, 5), (5, 6), [])
        check_both_directions((6, 4), (6, 6), [(6, 5)])
        check_both_directions((5, 5), (6, 6), [])
        check_both_directions((5, 5), (7, 7), [(6, 6)])
        check_both_directions((5, 6), (6, 5), [])
        check_both_directions((6, 4), (4, 6), [(5, 5)])

    def test_short_line_anti_aliasing(self):
        if False:
            while True:
                i = 10
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(3, 8) for j in range(3, 8)]

        def check_both_directions(from_pt, to_pt, should):
            if False:
                print('Hello World!')
            self._check_antialiasing(from_pt, to_pt, should, check_points)
        brown = (127, 127, 0)
        reddish = (191, 63, 0)
        greenish = (63, 191, 0)
        check_both_directions((4, 4), (6, 5), {(5, 4): brown, (5, 5): brown})
        check_both_directions((4, 5), (6, 4), {(5, 4): brown, (5, 5): brown})
        check_both_directions((4, 4), (5, 6), {(4, 5): brown, (5, 5): brown})
        check_both_directions((5, 4), (4, 6), {(4, 5): brown, (5, 5): brown})
        check_points = [(i, j) for i in range(2, 9) for j in range(2, 9)]
        should = {(4, 3): greenish, (5, 3): brown, (6, 3): reddish, (4, 4): reddish, (5, 4): brown, (6, 4): greenish}
        check_both_directions((3, 3), (7, 4), should)
        should = {(4, 3): reddish, (5, 3): brown, (6, 3): greenish, (4, 4): greenish, (5, 4): brown, (6, 4): reddish}
        check_both_directions((3, 4), (7, 3), should)
        should = {(4, 4): greenish, (4, 5): brown, (4, 6): reddish, (5, 4): reddish, (5, 5): brown, (5, 6): greenish}
        check_both_directions((4, 3), (5, 7), should)
        should = {(4, 4): reddish, (4, 5): brown, (4, 6): greenish, (5, 4): greenish, (5, 5): brown, (5, 6): reddish}
        check_both_directions((5, 3), (4, 7), should)

    def test_anti_aliasing_float_coordinates(self):
        if False:
            i = 10
            return i + 15
        'Float coordinates should be blended smoothly.'
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(5) for j in range(5)]
        brown = (127, 127, 0)
        reddish = (191, 63, 0)
        greenish = (63, 191, 0)
        expected = {(2, 2): FG_GREEN}
        self._check_antialiasing((1.5, 2), (1.5, 2), expected, check_points, set_endpoints=False)
        expected = {(2, 3): FG_GREEN}
        self._check_antialiasing((2.49, 2.7), (2.49, 2.7), expected, check_points, set_endpoints=False)
        expected = {(1, 2): brown, (2, 2): FG_GREEN}
        self._check_antialiasing((1.5, 2), (2, 2), expected, check_points, set_endpoints=False)
        expected = {(1, 2): brown, (2, 2): FG_GREEN, (3, 2): brown}
        self._check_antialiasing((1.5, 2), (2.5, 2), expected, check_points, set_endpoints=False)
        expected = {(2, 2): brown, (1, 2): FG_GREEN}
        self._check_antialiasing((1, 2), (1.5, 2), expected, check_points, set_endpoints=False)
        expected = {(1, 2): brown, (2, 2): greenish}
        self._check_antialiasing((1.5, 2), (1.75, 2), expected, check_points, set_endpoints=False)
        expected = {(x, y): brown for x in range(2, 5) for y in (1, 2)}
        self._check_antialiasing((2, 1.5), (4, 1.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): brown, (2, 2): FG_GREEN, (2, 3): brown}
        self._check_antialiasing((2, 1.5), (2, 2.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): brown, (2, 2): greenish}
        self._check_antialiasing((2, 1.5), (2, 1.75), expected, check_points, set_endpoints=False)
        expected = {(x, y): brown for x in (1, 2) for y in range(2, 5)}
        self._check_antialiasing((1.5, 2), (1.5, 4), expected, check_points, set_endpoints=False)
        expected = {(1, 1): brown, (2, 2): FG_GREEN, (3, 3): brown}
        self._check_antialiasing((1.5, 1.5), (2.5, 2.5), expected, check_points, set_endpoints=False)
        expected = {(3, 1): brown, (2, 2): FG_GREEN, (1, 3): brown}
        self._check_antialiasing((2.5, 1.5), (1.5, 2.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): brown, (2, 2): brown, (3, 2): brown, (3, 3): brown}
        self._check_antialiasing((2, 1.5), (3, 2.5), expected, check_points, set_endpoints=False)
        expected = {(2, 1): greenish, (2, 2): reddish, (3, 2): greenish, (3, 3): reddish, (4, 3): greenish, (4, 4): reddish}
        self._check_antialiasing((2, 1.25), (4, 3.25), expected, check_points, set_endpoints=False)

    def test_anti_aliasing_at_and_outside_the_border(self):
        if False:
            return 10
        "Ensures antialiasing works correct at a surface's borders."
        self.surface = pygame.Surface((10, 10))
        draw.rect(self.surface, BG_RED, (0, 0, 10, 10), 0)
        check_points = [(i, j) for i in range(10) for j in range(10)]
        reddish = (191, 63, 0)
        brown = (127, 127, 0)
        greenish = (63, 191, 0)
        (from_point, to_point) = ((3, 3), (7, 4))
        should = {(4, 3): greenish, (5, 3): brown, (6, 3): reddish, (4, 4): reddish, (5, 4): brown, (6, 4): greenish}
        for (dx, dy) in ((-4, 0), (4, 0), (0, -5), (0, -4), (0, -3), (0, 5), (0, 6), (0, 7), (-4, -4), (-4, -3), (-3, -4)):
            first = (from_point[0] + dx, from_point[1] + dy)
            second = (to_point[0] + dx, to_point[1] + dy)
            expected = {(x + dx, y + dy): color for ((x, y), color) in should.items()}
            self._check_antialiasing(first, second, expected, check_points)

class AALinesMixin(BaseLineMixin):
    """Mixin test for drawing aalines.

    This class contains all the general aalines drawing tests.
    """

    def test_aalines__args(self):
        if False:
            return 10
        'Ensures draw aalines accepts the correct args.'
        bounds_rect = self.draw_aalines(pygame.Surface((3, 3)), (0, 10, 0, 50), False, ((0, 0), (1, 1)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__args_without_blend(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw aalines accepts the args without a blend.'
        bounds_rect = self.draw_aalines(pygame.Surface((2, 2)), (0, 0, 0, 50), False, ((0, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__blend_warning(self):
        if False:
            while True:
                i = 10
        'From pygame 2, blend=False should raise DeprecationWarning.'
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            self.draw_aalines(pygame.Surface((2, 2)), (0, 0, 0, 50), False, ((0, 0), (1, 1)), False)
            self.assertEqual(len(w), 1)
            self.assertTrue(issubclass(w[-1].category, DeprecationWarning))

    def test_aalines__kwargs(self):
        if False:
            print('Hello World!')
        'Ensures draw aalines accepts the correct kwargs.'
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        points = ((0, 0), (1, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': False, 'points': points}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_aalines(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__kwargs_order_independent(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensures draw aalines's kwargs are not order dependent."
        bounds_rect = self.draw_aalines(closed=1, points=((0, 0), (1, 1), (2, 2)), color=(10, 20, 30), surface=pygame.Surface((3, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__args_missing(self):
        if False:
            while True:
                i = 10
        'Ensures draw aalines detects any missing required args.'
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, 0)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines()

    def test_aalines__kwargs_missing(self):
        if False:
            print('Hello World!')
        'Ensures draw aalines detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((3, 2)), 'color': pygame.Color('red'), 'closed': 1, 'points': ((2, 2), (1, 1))}
        for name in ('points', 'closed', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**invalid_kwargs)

    def test_aalines__arg_invalid_types(self):
        if False:
            print('Hello World!')
        'Ensures draw aalines detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        closed = 0
        points = ((1, 2), (2, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, closed, points, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, closed, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, color, InvalidBool(), points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines(surface, 2.3, closed, points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_aalines((1, 2, 3, 4), color, closed, points)

    def test_aalines__kwarg_invalid_types(self):
        if False:
            print('Hello World!')
        'Ensures draw aalines detects invalid kwarg types.'
        valid_kwargs = {'surface': pygame.Surface((3, 3)), 'color': pygame.Color('green'), 'closed': False, 'points': ((1, 2), (2, 1))}
        invalid_kwargs = {'surface': pygame.Surface, 'color': 2.3, 'closed': InvalidBool(), 'points': (0, 0, 0)}
        for kwarg in ('surface', 'color', 'closed', 'points'):
            kwargs = dict(valid_kwargs)
            kwargs[kwarg] = invalid_kwargs[kwarg]
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__kwarg_invalid_name(self):
        if False:
            return 10
        'Ensures draw aalines detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        closed = 1
        points = ((1, 2), (2, 1))
        kwargs_list = [{'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}, {'surface': surface, 'color': color, 'closed': closed, 'points': points, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__args_and_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw aalines accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 2))
        color = (255, 255, 0, 0)
        closed = 0
        points = ((1, 2), (2, 1))
        kwargs = {'surface': surface, 'color': color, 'closed': closed, 'points': points}
        for name in ('surface', 'color', 'closed', 'points'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_aalines(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_aalines(surface, color, **kwargs)
            elif 'closed' == name:
                bounds_rect = self.draw_aalines(surface, color, closed, **kwargs)
            elif 'points' == name:
                bounds_rect = self.draw_aalines(surface, color, closed, points, **kwargs)
            else:
                bounds_rect = self.draw_aalines(surface, color, closed, points, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__valid_points_format(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw aalines accepts different points formats.'
        expected_color = (10, 20, 30, 255)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'closed': False, 'points': None}
        point_types = ((tuple, tuple, tuple, tuple), (list, list, list, list), (Vector2, Vector2, Vector2, Vector2), (list, Vector2, tuple, Vector2))
        point_values = (((1, 1), (2, 1), (2, 2), (1, 2)), ((1, 1), (2.2, 1), (2.1, 2.2), (1, 2.1)))
        seq_types = (tuple, list)
        for point_type in point_types:
            for values in point_values:
                check_pos = values[0]
                points = [point_type[i](pt) for (i, pt) in enumerate(values)]
                for seq_type in seq_types:
                    surface.fill(surface_color)
                    kwargs['points'] = seq_type(points)
                    bounds_rect = self.draw_aalines(**kwargs)
                    self.assertEqual(surface.get_at(check_pos), expected_color)
                    self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__invalid_points_formats(self):
        if False:
            return 10
        'Ensures draw aalines handles invalid points formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None}
        points_fmts = (((1, 1), (2,)), ((1, 1), (2, 2, 2)), ((1, 1), (2, '2')), ((1, 1), {2, 3}), ((1, 1), dict(((2, 2), (3, 3)))), {(1, 1), (1, 2)}, dict(((1, 1), (4, 4))))
        for points in points_fmts:
            kwargs['points'] = points
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__invalid_points_values(self):
        if False:
            print('Hello World!')
        'Ensures draw aalines handles invalid points values correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'closed': False, 'points': None}
        for points in ([], ((1, 1),)):
            for seq_type in (tuple, list):
                kwargs['points'] = seq_type(points)
                with self.assertRaises(ValueError):
                    bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__valid_closed_values(self):
        if False:
            while True:
                i = 10
        'Ensures draw aalines accepts different closed values.'
        line_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((5, 5))
        pos = (1, 3)
        kwargs = {'surface': surface, 'color': line_color, 'closed': None, 'points': ((1, 1), (4, 1), (4, 4), (1, 4))}
        true_values = (-7, 1, 10, '2', 3.1, (4,), [5], True)
        false_values = (None, '', 0, (), [], False)
        for closed in true_values + false_values:
            surface.fill(surface_color)
            kwargs['closed'] = closed
            expected_color = line_color if closed else surface_color
            bounds_rect = self.draw_aalines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__valid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw aalines accepts different color formats.'
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        pos = (1, 1)
        kwargs = {'surface': surface, 'color': None, 'closed': False, 'points': (pos, (2, 1))}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_aalines(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_aalines__invalid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw aalines handles invalid color formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'closed': False, 'points': ((1, 1), (1, 2))}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_aalines(**kwargs)

    def test_aalines__color(self):
        if False:
            i = 10
            return i + 15
        'Tests if the aalines drawn are the correct color.\n\n        Draws aalines around the border of the given surface and checks if all\n        borders of the surface only contain the given color.\n        '
        for surface in self._create_surfaces():
            for expected_color in self.COLORS:
                self.draw_aalines(surface, expected_color, True, corners(surface))
                for (pos, color) in border_pos_and_color(surface):
                    self.assertEqual(color, expected_color, f'pos={pos}')

    def test_aalines__gaps(self):
        if False:
            return 10
        'Tests if the aalines drawn contain any gaps.\n\n        Draws aalines around the border of the given surface and checks if\n        all borders of the surface contain any gaps.\n\n        See: #512\n        '
        expected_color = (255, 255, 255)
        for surface in self._create_surfaces():
            self.draw_aalines(surface, expected_color, True, corners(surface))
            for (pos, color) in border_pos_and_color(surface):
                self.assertEqual(color, expected_color, f'pos={pos}')

    def test_aalines__bounding_rect(self):
        if False:
            while True:
                i = 10
        'Ensures draw aalines returns the correct bounding rect.\n\n        Tests lines with endpoints on and off the surface and blending\n        enabled and disabled.\n        '
        line_color = pygame.Color('red')
        surf_color = pygame.Color('blue')
        width = height = 30
        pos_rect = pygame.Rect((0, 0), (width, height))
        for size in ((width + 5, height + 5), (width - 5, height - 5)):
            surface = pygame.Surface(size, 0, 32)
            surf_rect = surface.get_rect()
            for pos in rect_corners_mids_and_center(surf_rect):
                pos_rect.center = pos
                pts = (pos_rect.midleft, pos_rect.midtop, pos_rect.midright)
                pos = pts[0]
                for closed in (True, False):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_aalines(surface, line_color, closed, pts)
                    expected_rect = create_bounding_rect(surface, surf_color, pos)
                    self.assertEqual(bounding_rect, expected_rect)

    def test_aalines__surface_clip(self):
        if False:
            i = 10
            return i + 15
        "Ensures draw aalines respects a surface's clip area."
        surfw = surfh = 30
        aaline_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for center in rect_corners_mids_and_center(clip_rect):
            pos_rect.center = center
            pts = (pos_rect.midtop, pos_rect.center, pos_rect.midbottom)
            for closed in (True, False):
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_aalines(surface, aaline_color, closed, pts)
                expected_pts = get_color_points(surface, surface_color, clip_rect, False)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_aalines(surface, aaline_color, closed, pts)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        self.assertNotEqual(surface.get_at(pt), surface_color, pt)
                    else:
                        self.assertEqual(surface.get_at(pt), surface_color, pt)
                surface.unlock()

class DrawAALinesTest(AALinesMixin, DrawTestCase):
    """Test draw module function aalines.

    This class inherits the general tests from AALinesMixin. It is also the
    class to add any draw.aalines specific tests to.
    """
SQUARE = ([0, 0], [3, 0], [3, 3], [0, 3])
DIAMOND = [(1, 3), (3, 5), (5, 3), (3, 1)]
CROSS = ([2, 0], [4, 0], [4, 2], [6, 2], [6, 4], [4, 4], [4, 6], [2, 6], [2, 4], [0, 4], [0, 2], [2, 2])

class DrawPolygonMixin:
    """Mixin tests for drawing polygons.

    This class contains all the general polygon drawing tests.
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.surface = pygame.Surface((20, 20))

    def test_polygon__args(self):
        if False:
            print('Hello World!')
        'Ensures draw polygon accepts the correct args.'
        bounds_rect = self.draw_polygon(pygame.Surface((3, 3)), (0, 10, 0, 50), ((0, 0), (1, 1), (2, 2)), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__args_without_width(self):
        if False:
            print('Hello World!')
        'Ensures draw polygon accepts the args without a width.'
        bounds_rect = self.draw_polygon(pygame.Surface((2, 2)), (0, 0, 0, 50), ((0, 0), (1, 1), (2, 2)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__kwargs(self):
        if False:
            return 10
        'Ensures draw polygon accepts the correct kwargs\n        with and without a width arg.\n        '
        surface = pygame.Surface((4, 4))
        color = pygame.Color('yellow')
        points = ((0, 0), (1, 1), (2, 2))
        kwargs_list = [{'surface': surface, 'color': color, 'points': points, 'width': 1}, {'surface': surface, 'color': color, 'points': points}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_polygon(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__kwargs_order_independent(self):
        if False:
            return 10
        "Ensures draw polygon's kwargs are not order dependent."
        bounds_rect = self.draw_polygon(color=(10, 20, 30), surface=pygame.Surface((3, 2)), width=0, points=((0, 1), (1, 2), (2, 3)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__args_missing(self):
        if False:
            print('Hello World!')
        'Ensures draw polygon detects any missing required args.'
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon()

    def test_polygon__kwargs_missing(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw polygon detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'points': ((2, 1), (2, 2), (2, 3)), 'width': 1}
        for name in ('points', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**invalid_kwargs)

    def test_polygon__arg_invalid_types(self):
        if False:
            return 10
        'Ensures draw polygon detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        points = ((0, 1), (1, 2), (1, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, color, points, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, color, (1, 2, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon(surface, 2.3, points)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_polygon((1, 2, 3, 4), color, points)

    def test_polygon__kwarg_invalid_types(self):
        if False:
            return 10
        'Ensures draw polygon detects invalid kwarg types.'
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        points = ((0, 0), (1, 0), (2, 0))
        width = 1
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'points': points, 'width': width}, {'surface': surface, 'color': 2.3, 'points': points, 'width': width}, {'surface': surface, 'color': color, 'points': ((1,), (1,), (1,)), 'width': width}, {'surface': surface, 'color': color, 'points': points, 'width': 1.2}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__kwarg_invalid_name(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw polygon detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        points = ((1, 1), (1, 2), (1, 3))
        kwargs_list = [{'surface': surface, 'color': color, 'points': points, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'points': points, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__args_and_kwargs(self):
        if False:
            while True:
                i = 10
        'Ensures draw polygon accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        points = ((0, 1), (1, 2), (2, 3))
        width = 0
        kwargs = {'surface': surface, 'color': color, 'points': points, 'width': width}
        for name in ('surface', 'color', 'points', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_polygon(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_polygon(surface, color, **kwargs)
            elif 'points' == name:
                bounds_rect = self.draw_polygon(surface, color, points, **kwargs)
            else:
                bounds_rect = self.draw_polygon(surface, color, points, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__valid_width_values(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw polygon accepts different width values.'
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': None}
        pos = kwargs['points'][0]
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_polygon(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__valid_points_format(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw polygon accepts different points formats.'
        expected_color = (10, 20, 30, 255)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'points': None, 'width': 0}
        point_types = ((tuple, tuple, tuple, tuple), (list, list, list, list), (Vector2, Vector2, Vector2, Vector2), (list, Vector2, tuple, Vector2))
        point_values = (((1, 1), (2, 1), (2, 2), (1, 2)), ((1, 1), (2.2, 1), (2.1, 2.2), (1, 2.1)))
        seq_types = (tuple, list)
        for point_type in point_types:
            for values in point_values:
                check_pos = values[0]
                points = [point_type[i](pt) for (i, pt) in enumerate(values)]
                for seq_type in seq_types:
                    surface.fill(surface_color)
                    kwargs['points'] = seq_type(points)
                    bounds_rect = self.draw_polygon(**kwargs)
                    self.assertEqual(surface.get_at(check_pos), expected_color)
                    self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__invalid_points_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw polygon handles invalid points formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'points': None, 'width': 0}
        points_fmts = (((1, 1), (2, 1), (2,)), ((1, 1), (2, 1), (2, 2, 2)), ((1, 1), (2, 1), (2, '2')), ((1, 1), (2, 1), {2, 3}), ((1, 1), (2, 1), dict(((2, 2), (3, 3)))), {(1, 1), (2, 1), (2, 2), (1, 2)}, dict(((1, 1), (2, 2), (3, 3), (4, 4))))
        for points in points_fmts:
            kwargs['points'] = points
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__invalid_points_values(self):
        if False:
            print('Hello World!')
        'Ensures draw polygon handles invalid points values correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'points': None, 'width': 0}
        points_fmts = (tuple(), ((1, 1),), ((1, 1), (2, 1)))
        for points in points_fmts:
            for seq_type in (tuple, list):
                kwargs['points'] = seq_type(points)
                with self.assertRaises(ValueError):
                    bounds_rect = self.draw_polygon(**kwargs)

    def test_polygon__valid_color_formats(self):
        if False:
            print('Hello World!')
        'Ensures draw polygon accepts different color formats.'
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': 0}
        pos = kwargs['points'][0]
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_polygon(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_polygon__invalid_color_formats(self):
        if False:
            print('Hello World!')
        'Ensures draw polygon handles invalid color formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'points': ((1, 1), (2, 1), (2, 2), (1, 2)), 'width': 0}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_polygon(**kwargs)

    def test_draw_square(self):
        if False:
            print('Hello World!')
        self.draw_polygon(self.surface, RED, SQUARE, 0)
        for x in range(4):
            for y in range(4):
                self.assertEqual(self.surface.get_at((x, y)), RED)

    def test_draw_diamond(self):
        if False:
            i = 10
            return i + 15
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, DIAMOND, 0)
        for (x, y) in DIAMOND:
            self.assertEqual(self.surface.get_at((x, y)), GREEN, msg=str((x, y)))
        for x in range(2, 5):
            for y in range(2, 5):
                self.assertEqual(self.surface.get_at((x, y)), GREEN)

    def test_1_pixel_high_or_wide_shapes(self):
        if False:
            print('Hello World!')
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(x, 2) for (x, _y) in CROSS], 0)
        cross_size = 6
        for x in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((x, 1)), RED)
            self.assertEqual(self.surface.get_at((x, 2)), GREEN)
            self.assertEqual(self.surface.get_at((x, 3)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(x, 5) for (x, _y) in CROSS], 1)
        for x in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((x, 4)), RED)
            self.assertEqual(self.surface.get_at((x, 5)), GREEN)
            self.assertEqual(self.surface.get_at((x, 6)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(3, y) for (_x, y) in CROSS], 0)
        for y in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((2, y)), RED)
            self.assertEqual(self.surface.get_at((3, y)), GREEN)
            self.assertEqual(self.surface.get_at((4, y)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, [(4, y) for (_x, y) in CROSS], 1)
        for y in range(cross_size + 1):
            self.assertEqual(self.surface.get_at((3, y)), RED)
            self.assertEqual(self.surface.get_at((4, y)), GREEN)
            self.assertEqual(self.surface.get_at((5, y)), RED)

    def test_draw_symetric_cross(self):
        if False:
            for i in range(10):
                print('nop')
        'non-regression on issue #234 : x and y where handled inconsistently.\n\n        Also, the result is/was different whether we fill or not the polygon.\n        '
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, CROSS, 1)
        inside = [(x, 3) for x in range(1, 6)] + [(3, y) for y in range(1, 6)]
        for x in range(10):
            for y in range(10):
                if (x, y) in inside:
                    self.assertEqual(self.surface.get_at((x, y)), RED)
                elif x in range(2, 5) and y < 7 or (y in range(2, 5) and x < 7):
                    self.assertEqual(self.surface.get_at((x, y)), GREEN)
                else:
                    self.assertEqual(self.surface.get_at((x, y)), RED)
        pygame.draw.rect(self.surface, RED, (0, 0, 10, 10), 0)
        self.draw_polygon(self.surface, GREEN, CROSS, 0)
        inside = [(x, 3) for x in range(1, 6)] + [(3, y) for y in range(1, 6)]
        for x in range(10):
            for y in range(10):
                if x in range(2, 5) and y < 7 or (y in range(2, 5) and x < 7):
                    self.assertEqual(self.surface.get_at((x, y)), GREEN, msg=str((x, y)))
                else:
                    self.assertEqual(self.surface.get_at((x, y)), RED)

    def test_illumine_shape(self):
        if False:
            for i in range(10):
                print('nop')
        'non-regression on issue #313'
        rect = pygame.Rect((0, 0, 20, 20))
        path_data = [(0, 0), (rect.width - 1, 0), (rect.width - 5, 5 - 1), (5 - 1, 5 - 1), (5 - 1, rect.height - 5), (0, rect.height - 1)]
        pygame.draw.rect(self.surface, RED, (0, 0, 20, 20), 0)
        self.draw_polygon(self.surface, GREEN, path_data[:4], 0)
        for x in range(20):
            self.assertEqual(self.surface.get_at((x, 0)), GREEN)
        for x in range(4, rect.width - 5 + 1):
            self.assertEqual(self.surface.get_at((x, 4)), GREEN)
        pygame.draw.rect(self.surface, RED, (0, 0, 20, 20), 0)
        self.draw_polygon(self.surface, GREEN, path_data, 0)
        for x in range(4, rect.width - 5 + 1):
            self.assertEqual(self.surface.get_at((x, 4)), GREEN)

    def test_invalid_points(self):
        if False:
            return 10
        self.assertRaises(TypeError, lambda : self.draw_polygon(self.surface, RED, ((0, 0), (0, 20), (20, 20), 20), 0))

    def test_polygon__bounding_rect(self):
        if False:
            while True:
                i = 10
        'Ensures draw polygon returns the correct bounding rect.\n\n        Tests polygons on and off the surface and a range of width/thickness\n        values.\n        '
        polygon_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        min_width = min_height = 5
        max_width = max_height = 7
        sizes = ((min_width, min_height), (max_width, max_height))
        surface = pygame.Surface((20, 20), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for attr in RECT_POSITION_ATTRIBUTES:
                for (width, height) in sizes:
                    pos_rect = pygame.Rect((0, 0), (width, height))
                    setattr(pos_rect, attr, pos)
                    vertices = (pos_rect.midleft, pos_rect.midtop, pos_rect.bottomright)
                    for thickness in range(4):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_polygon(surface, polygon_color, vertices, thickness)
                        expected_rect = create_bounding_rect(surface, surf_color, vertices[0])
                        self.assertEqual(bounding_rect, expected_rect, f'thickness={thickness}')

    def test_polygon__surface_clip(self):
        if False:
            return 10
        "Ensures draw polygon respects a surface's clip area.\n\n        Tests drawing the polygon filled and unfilled.\n        "
        surfw = surfh = 30
        polygon_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (8, 10))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                vertices = (pos_rect.topleft, pos_rect.topright, pos_rect.bottomright, pos_rect.bottomleft)
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_polygon(surface, polygon_color, vertices, width)
                expected_pts = get_color_points(surface, polygon_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_polygon(surface, polygon_color, vertices, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = polygon_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()

class DrawPolygonTest(DrawPolygonMixin, DrawTestCase):
    """Test draw module function polygon.

    This class inherits the general tests from DrawPolygonMixin. It is also
    the class to add any draw.polygon specific tests to.
    """

class DrawRectMixin:
    """Mixin tests for drawing rects.

    This class contains all the general rect drawing tests.
    """

    def test_rect__args(self):
        if False:
            return 10
        'Ensures draw rect accepts the correct args.'
        bounds_rect = self.draw_rect(pygame.Surface((2, 2)), (20, 10, 20, 150), pygame.Rect((0, 0), (1, 1)), 2, 1, 2, 3, 4, 5)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__args_without_width(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw rect accepts the args without a width and borders.'
        bounds_rect = self.draw_rect(pygame.Surface((3, 5)), (0, 0, 0, 255), pygame.Rect((0, 0), (1, 1)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__kwargs(self):
        if False:
            while True:
                i = 10
        'Ensures draw rect accepts the correct kwargs\n        with and without a width and border_radius arg.\n        '
        kwargs_list = [{'surface': pygame.Surface((5, 5)), 'color': pygame.Color('red'), 'rect': pygame.Rect((0, 0), (1, 2)), 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': pygame.Surface((1, 2)), 'color': (0, 100, 200), 'rect': (0, 0, 1, 1)}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_rect(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__kwargs_order_independent(self):
        if False:
            print('Hello World!')
        "Ensures draw rect's kwargs are not order dependent."
        bounds_rect = self.draw_rect(color=(0, 1, 2), border_radius=10, surface=pygame.Surface((2, 3)), border_top_left_radius=5, width=-2, border_top_right_radius=20, border_bottom_right_radius=0, rect=pygame.Rect((0, 0), (0, 0)), border_bottom_left_radius=15)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__args_missing(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw rect detects any missing required args.'
        surface = pygame.Surface((1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, pygame.Color('white'))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect()

    def test_rect__kwargs_missing(self):
        if False:
            print('Hello World!')
        'Ensures draw rect detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((1, 3)), 'color': pygame.Color('red'), 'rect': pygame.Rect((0, 0), (2, 2)), 'width': 5, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}
        for name in ('rect', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**invalid_kwargs)

    def test_rect__arg_invalid_types(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw rect detects invalid arg types.'
        surface = pygame.Surface((3, 3))
        color = pygame.Color('white')
        rect = pygame.Rect((1, 1), (1, 1))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_bottom_right_radius='rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_bottom_left_radius='rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_top_right_radius='rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, border_top_left_radius='draw')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, 2, 'rad')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, rect, '2', 4)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, color, (1, 2, 3), 2, 6)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(surface, 2.3, rect, 3, 8)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_rect(rect, color, rect, 4, 10)

    def test_rect__kwarg_invalid_types(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw rect detects invalid kwarg types.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('red')
        rect = pygame.Rect((0, 0), (1, 1))
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': 2.3, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': (1, 1, 2), 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1.1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10.5, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5.5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 'a', 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 'c', 'border_bottom_right_radius': 0}, {'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 'd'}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__kwarg_invalid_name(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw rect detects invalid kwarg names.'
        surface = pygame.Surface((2, 1))
        color = pygame.Color('green')
        rect = pygame.Rect((0, 0), (3, 3))
        kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'width': 1, 'border_radius': 10, 'border_top_left_radius': 5, 'border_top_right_radius': 20, 'border_bottom_left_radius': 15, 'border_bottom_right_radius': 0, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__args_and_kwargs(self):
        if False:
            while True:
                i = 10
        'Ensures draw rect accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 1))
        color = (255, 255, 255, 0)
        rect = pygame.Rect((1, 0), (2, 5))
        width = 0
        kwargs = {'surface': surface, 'color': color, 'rect': rect, 'width': width}
        for name in ('surface', 'color', 'rect', 'width'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_rect(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_rect(surface, color, **kwargs)
            elif 'rect' == name:
                bounds_rect = self.draw_rect(surface, color, rect, **kwargs)
            else:
                bounds_rect = self.draw_rect(surface, color, rect, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__valid_width_values(self):
        if False:
            print('Hello World!')
        'Ensures draw rect accepts different width values.'
        pos = (1, 1)
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        color = (1, 2, 3, 255)
        kwargs = {'surface': surface, 'color': color, 'rect': pygame.Rect(pos, (2, 2)), 'width': None}
        for width in (-1000, -10, -1, 0, 1, 10, 1000):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_rect(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__valid_rect_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw rect accepts different rect formats.'
        pos = (1, 1)
        expected_color = pygame.Color('yellow')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'rect': None, 'width': 0}
        rects = (pygame.Rect(pos, (1, 1)), (pos, (2, 2)), (pos[0], pos[1], 3, 3), [pos, (2.1, 2.2)])
        for rect in rects:
            surface.fill(surface_color)
            kwargs['rect'] = rect
            bounds_rect = self.draw_rect(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__invalid_rect_formats(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw rect handles invalid rect formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 4)), 'color': pygame.Color('red'), 'rect': None, 'width': 0}
        invalid_fmts = ([], [1], [1, 2], [1, 2, 3], [1, 2, 3, 4, 5], {1, 2, 3, 4}, [1, 2, 3, '4'])
        for rect in invalid_fmts:
            kwargs['rect'] = rect
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__valid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw rect accepts different color formats.'
        pos = (1, 1)
        red_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 1)), 'width': 3}
        reds = ((255, 0, 0), (255, 0, 0, 255), surface.map_rgb(red_color), red_color)
        for color in reds:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = red_color
            bounds_rect = self.draw_rect(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_rect__invalid_color_formats(self):
        if False:
            while True:
                i = 10
        'Ensures draw rect handles invalid color formats correctly.'
        pos = (1, 1)
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (1, 1)), 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_rect(**kwargs)

    def test_rect__fill(self):
        if False:
            print('Hello World!')
        (self.surf_w, self.surf_h) = self.surf_size = (320, 200)
        self.surf = pygame.Surface(self.surf_size, pygame.SRCALPHA)
        self.color = (1, 13, 24, 205)
        rect = pygame.Rect(10, 10, 25, 20)
        drawn = self.draw_rect(self.surf, self.color, rect, 0)
        self.assertEqual(drawn, rect)
        for pt in test_utils.rect_area_pts(rect):
            color_at_pt = self.surf.get_at(pt)
            self.assertEqual(color_at_pt, self.color)
        for pt in test_utils.rect_outer_bounds(rect):
            color_at_pt = self.surf.get_at(pt)
            self.assertNotEqual(color_at_pt, self.color)
        bgcolor = pygame.Color('black')
        self.surf.fill(bgcolor)
        hrect = pygame.Rect(1, 1, self.surf_w - 2, 1)
        vrect = pygame.Rect(1, 3, 1, self.surf_h - 4)
        drawn = self.draw_rect(self.surf, self.color, hrect, 0)
        self.assertEqual(drawn, hrect)
        (x, y) = hrect.topleft
        (w, h) = hrect.size
        self.assertEqual(self.surf.get_at((x - 1, y)), bgcolor)
        self.assertEqual(self.surf.get_at((x + w, y)), bgcolor)
        for i in range(x, x + w):
            self.assertEqual(self.surf.get_at((i, y)), self.color)
        drawn = self.draw_rect(self.surf, self.color, vrect, 0)
        self.assertEqual(drawn, vrect)
        (x, y) = vrect.topleft
        (w, h) = vrect.size
        self.assertEqual(self.surf.get_at((x, y - 1)), bgcolor)
        self.assertEqual(self.surf.get_at((x, y + h)), bgcolor)
        for i in range(y, y + h):
            self.assertEqual(self.surf.get_at((x, i)), self.color)

    def test_rect__one_pixel_lines(self):
        if False:
            print('Hello World!')
        self.surf = pygame.Surface((320, 200), pygame.SRCALPHA)
        self.color = (1, 13, 24, 205)
        rect = pygame.Rect(10, 10, 56, 20)
        drawn = self.draw_rect(self.surf, self.color, rect, 1)
        self.assertEqual(drawn, rect)
        for pt in test_utils.rect_perimeter_pts(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assertEqual(color_at_pt, self.color)
        for pt in test_utils.rect_outer_bounds(drawn):
            color_at_pt = self.surf.get_at(pt)
            self.assertNotEqual(color_at_pt, self.color)

    def test_rect__draw_line_width(self):
        if False:
            while True:
                i = 10
        surface = pygame.Surface((100, 100))
        surface.fill('black')
        color = pygame.Color(255, 255, 255)
        rect_width = 80
        rect_height = 50
        line_width = 10
        pygame.draw.rect(surface, color, pygame.Rect(0, 0, rect_width, rect_height), line_width)
        for i in range(line_width):
            self.assertEqual(surface.get_at((i, i)), color)
            self.assertEqual(surface.get_at((rect_width - i - 1, i)), color)
            self.assertEqual(surface.get_at((i, rect_height - i - 1)), color)
            self.assertEqual(surface.get_at((rect_width - i - 1, rect_height - i - 1)), color)
        self.assertEqual(surface.get_at((line_width, line_width)), (0, 0, 0))
        self.assertEqual(surface.get_at((rect_width - line_width - 1, line_width)), (0, 0, 0))
        self.assertEqual(surface.get_at((line_width, rect_height - line_width - 1)), (0, 0, 0))
        self.assertEqual(surface.get_at((rect_width - line_width - 1, rect_height - line_width - 1)), (0, 0, 0))

    def test_rect__bounding_rect(self):
        if False:
            return 10
        'Ensures draw rect returns the correct bounding rect.\n\n        Tests rects on and off the surface and a range of width/thickness\n        values.\n        '
        rect_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        min_width = min_height = 5
        max_width = max_height = 7
        sizes = ((min_width, min_height), (max_width, max_height))
        surface = pygame.Surface((20, 20), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for attr in RECT_POSITION_ATTRIBUTES:
                for (width, height) in sizes:
                    rect = pygame.Rect((0, 0), (width, height))
                    setattr(rect, attr, pos)
                    for thickness in range(4):
                        surface.fill(surf_color)
                        bounding_rect = self.draw_rect(surface, rect_color, rect, thickness)
                        expected_rect = create_bounding_rect(surface, surf_color, rect.topleft)
                        self.assertEqual(bounding_rect, expected_rect, f'thickness={thickness}')

    def test_rect__surface_clip(self):
        if False:
            return 10
        "Ensures draw rect respects a surface's clip area.\n\n        Tests drawing the rect filled and unfilled.\n        "
        surfw = surfh = 30
        rect_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (8, 10))
        clip_rect.center = surface.get_rect().center
        test_rect = clip_rect.copy()
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                test_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_rect(surface, rect_color, test_rect, width)
                expected_pts = get_color_points(surface, rect_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_rect(surface, rect_color, test_rect, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = rect_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()

class DrawRectTest(DrawRectMixin, DrawTestCase):
    """Test draw module function rect.

    This class inherits the general tests from DrawRectMixin. It is also the
    class to add any draw.rect specific tests to.
    """

class DrawCircleMixin:
    """Mixin tests for drawing circles.

    This class contains all the general circle drawing tests.
    """

    def test_circle__args(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw circle accepts the correct args.'
        bounds_rect = self.draw_circle(pygame.Surface((3, 3)), (0, 10, 0, 50), (0, 0), 3, 1, 1, 0, 1, 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__args_without_width(self):
        if False:
            return 10
        'Ensures draw circle accepts the args without a width and\n        quadrants.'
        bounds_rect = self.draw_circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1, 1), 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__args_with_negative_width(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw circle accepts the args with negative width.'
        bounds_rect = self.draw_circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1, 1), 1, -1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(1, 1, 0, 0))

    def test_circle__args_with_width_gt_radius(self):
        if False:
            while True:
                i = 10
        'Ensures draw circle accepts the args with width > radius.'
        bounds_rect = self.draw_circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1, 1), 2, 3, 0, 0, 0, 0)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(0, 0, 2, 2))

    def test_circle__kwargs(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw circle accepts the correct kwargs\n        with and without a width and quadrant arguments.\n        '
        kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'center': (2, 2), 'radius': 2, 'width': 1, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': False, 'draw_bottom_right': True}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'center': (1, 1), 'radius': 1}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_circle(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__kwargs_order_independent(self):
        if False:
            print('Hello World!')
        "Ensures draw circle's kwargs are not order dependent."
        bounds_rect = self.draw_circle(draw_top_right=False, color=(10, 20, 30), surface=pygame.Surface((3, 2)), width=0, draw_bottom_left=False, center=(1, 0), draw_bottom_right=False, radius=2, draw_top_left=True)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__args_missing(self):
        if False:
            return 10
        'Ensures draw circle detects any missing required args.'
        surface = pygame.Surface((1, 1))
        color = pygame.Color('blue')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, (0, 0))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle()

    def test_circle__kwargs_missing(self):
        if False:
            print('Hello World!')
        'Ensures draw circle detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'center': (1, 0), 'radius': 2, 'width': 1, 'draw_top_right': False, 'draw_top_left': False, 'draw_bottom_left': False, 'draw_bottom_right': True}
        for name in ('radius', 'center', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**invalid_kwargs)

    def test_circle__arg_invalid_types(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw circle detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        center = (1, 1)
        radius = 1
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 'a', 1, 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 'b', 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 1, 'c', 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, 1, 1, 1, 1, 'd')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, radius, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, center, '2')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, color, (1, 2, 3), radius)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle(surface, 2.3, center, radius)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_circle((1, 2, 3, 4), color, center, radius)

    def test_circle__kwarg_invalid_types(self):
        if False:
            print('Hello World!')
        'Ensures draw circle detects invalid kwarg types.'
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        center = (0, 1)
        radius = 1
        width = 1
        quadrant = 1
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': 2.3, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': (1, 1, 1), 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': '1', 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': 1.2, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': 'True', 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': 'True', 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': 3.14, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': 'quadrant'}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**kwargs)

    def test_circle__kwarg_invalid_name(self):
        if False:
            return 10
        'Ensures draw circle detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        center = (0, 0)
        radius = 2
        kwargs_list = [{'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': 1, 'quadrant': 1, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}, {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**kwargs)

    def test_circle__args_and_kwargs(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw circle accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        center = (1, 0)
        radius = 2
        width = 0
        draw_top_right = True
        draw_top_left = False
        draw_bottom_left = False
        draw_bottom_right = True
        kwargs = {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': width, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for name in ('surface', 'color', 'center', 'radius', 'width', 'draw_top_right', 'draw_top_left', 'draw_bottom_left', 'draw_bottom_right'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_circle(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_circle(surface, color, **kwargs)
            elif 'center' == name:
                bounds_rect = self.draw_circle(surface, color, center, **kwargs)
            elif 'radius' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, **kwargs)
            elif 'width' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, **kwargs)
            elif 'draw_top_right' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, **kwargs)
            elif 'draw_top_left' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, draw_top_left, **kwargs)
            elif 'draw_bottom_left' == name:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, draw_top_left, draw_bottom_left, **kwargs)
            else:
                bounds_rect = self.draw_circle(surface, color, center, radius, width, draw_top_right, draw_top_left, draw_bottom_left, draw_bottom_right, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_width_values(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw circle accepts different width values.'
        center = (2, 2)
        radius = 1
        pos = (center[0] - radius, center[1])
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'center': center, 'radius': radius, 'width': None, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for width in (-100, -10, -1, 0, 1, 10, 100):
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = color if width >= 0 else surface_color
            bounds_rect = self.draw_circle(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_radius_values(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw circle accepts different radius values.'
        pos = center = (2, 2)
        surface_color = pygame.Color('white')
        surface = pygame.Surface((3, 4))
        color = (10, 20, 30, 255)
        kwargs = {'surface': surface, 'color': color, 'center': center, 'radius': None, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for radius in (-10, -1, 0, 1, 10):
            surface.fill(surface_color)
            kwargs['radius'] = radius
            expected_color = color if radius > 0 else surface_color
            bounds_rect = self.draw_circle(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_center_formats(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw circle accepts different center formats.'
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((4, 4))
        kwargs = {'surface': surface, 'color': expected_color, 'center': None, 'radius': 1, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        (x, y) = (2, 2)
        for center in ((x, y), (x + 0.1, y), (x, y + 0.1), (x + 0.1, y + 0.1)):
            for seq_type in (tuple, list, Vector2):
                surface.fill(surface_color)
                kwargs['center'] = seq_type(center)
                bounds_rect = self.draw_circle(**kwargs)
                self.assertEqual(surface.get_at((x, y)), expected_color)
                self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__valid_color_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw circle accepts different color formats.'
        center = (2, 2)
        radius = 1
        pos = (center[0] - radius, center[1])
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((3, 4))
        kwargs = {'surface': surface, 'color': None, 'center': center, 'radius': radius, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_circle(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_circle__invalid_color_formats(self):
        if False:
            print('Hello World!')
        'Ensures draw circle handles invalid color formats correctly.'
        kwargs = {'surface': pygame.Surface((4, 3)), 'color': None, 'center': (1, 2), 'radius': 1, 'width': 0, 'draw_top_right': True, 'draw_top_left': True, 'draw_bottom_left': True, 'draw_bottom_right': True}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_circle(**kwargs)

    def test_circle__floats(self):
        if False:
            i = 10
            return i + 15
        'Ensure that floats are accepted.'
        draw.circle(surface=pygame.Surface((4, 4)), color=(255, 255, 127), center=(1.5, 1.5), radius=1.3, width=0, draw_top_right=True, draw_top_left=True, draw_bottom_left=True, draw_bottom_right=True)
        draw.circle(surface=pygame.Surface((4, 4)), color=(255, 255, 127), center=Vector2(1.5, 1.5), radius=1.3, width=0, draw_top_right=True, draw_top_left=True, draw_bottom_left=True, draw_bottom_right=True)
        draw.circle(pygame.Surface((2, 2)), (0, 0, 0, 50), (1.3, 1.3), 1.2)

    def test_circle__bounding_rect(self):
        if False:
            return 10
        'Ensures draw circle returns the correct bounding rect.\n\n        Tests circles on and off the surface and a range of width/thickness\n        values.\n        '
        circle_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        max_radius = 3
        surface = pygame.Surface((30, 30), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(max_radius * 2 - 1, max_radius * 2 - 1)
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for radius in range(max_radius + 1):
                for thickness in range(radius + 1):
                    surface.fill(surf_color)
                    bounding_rect = self.draw_circle(surface, circle_color, pos, radius, thickness)
                    expected_rect = create_bounding_rect(surface, surf_color, pos)
                    with self.subTest(surface=surface, circle_color=circle_color, pos=pos, radius=radius, thickness=thickness):
                        self.assertEqual(bounding_rect, expected_rect)

    def test_circle_negative_radius(self):
        if False:
            while True:
                i = 10
        'Ensures negative radius circles return zero sized bounding rect.'
        surf = pygame.Surface((200, 200))
        color = (0, 0, 0, 50)
        center = (surf.get_height() // 2, surf.get_height() // 2)
        bounding_rect = self.draw_circle(surf, color, center, radius=-1, width=1)
        self.assertEqual(bounding_rect.size, (0, 0))

    def test_circle_zero_radius(self):
        if False:
            while True:
                i = 10
        'Ensures zero radius circles does not draw a center pixel.\n\n        NOTE: This is backwards incompatible behaviour with 1.9.x.\n        '
        surf = pygame.Surface((200, 200))
        circle_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        center = (100, 100)
        radius = 0
        width = 1
        bounding_rect = self.draw_circle(surf, circle_color, center, radius, width)
        expected_rect = create_bounding_rect(surf, surf_color, center)
        self.assertEqual(bounding_rect, expected_rect)
        self.assertEqual(bounding_rect, pygame.Rect(100, 100, 0, 0))

    def test_circle__surface_clip(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensures draw circle respects a surface's clip area.\n\n        Tests drawing the circle filled and unfilled.\n        "
        surfw = surfh = 25
        circle_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (10, 10))
        clip_rect.center = surface.get_rect().center
        radius = clip_rect.w // 2 + 1
        for width in (0, 1):
            for center in rect_corners_mids_and_center(clip_rect):
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_circle(surface, circle_color, center, radius, width)
                expected_pts = get_color_points(surface, circle_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_circle(surface, circle_color, center, radius, width)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = circle_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()

    def test_circle_shape(self):
        if False:
            return 10
        'Ensures there are no holes in the circle, and no overdrawing.\n\n        Tests drawing a thick circle.\n        Measures the distance of the drawn pixels from the circle center.\n        '
        surfw = surfh = 100
        circle_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        (cx, cy) = center = (50, 50)
        radius = 45
        width = 25
        dest_rect = self.draw_circle(surface, circle_color, center, radius, width)
        for pt in test_utils.rect_area_pts(dest_rect):
            (x, y) = pt
            sqr_distance = (x - cx) ** 2 + (y - cy) ** 2
            if (radius - width + 1) ** 2 < sqr_distance < (radius - 1) ** 2:
                self.assertEqual(surface.get_at(pt), circle_color)
            if sqr_distance < (radius - width - 1) ** 2 or sqr_distance > (radius + 1) ** 2:
                self.assertEqual(surface.get_at(pt), surface_color)

    def test_circle__diameter(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw circle is twice size of radius high and wide.'
        surf = pygame.Surface((200, 200))
        color = (0, 0, 0, 50)
        center = (surf.get_height() // 2, surf.get_height() // 2)
        width = 1
        radius = 6
        for radius in range(1, 65):
            bounding_rect = self.draw_circle(surf, color, center, radius, width)
            self.assertEqual(bounding_rect.width, radius * 2)
            self.assertEqual(bounding_rect.height, radius * 2)

    def test_x_bounds(self):
        if False:
            print('Hello World!')
        'ensures a circle is drawn properly when there is a negative x, or a big x.'
        surf = pygame.Surface((200, 200))
        bgcolor = (0, 0, 0, 255)
        surf.fill(bgcolor)
        color = (255, 0, 0, 255)
        width = 1
        radius = 10
        where = (0, 30)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(0, where[1] - radius, where[0] + radius, radius * 2))
        self.assertEqual(surf.get_at((where[0] if where[0] > 0 else 0, where[1])), color)
        self.assertEqual(surf.get_at((where[0] + radius + 1, where[1])), bgcolor)
        self.assertEqual(surf.get_at((where[0] + radius - 1, where[1])), color)
        surf.fill(bgcolor)
        where = (-1e+30, 80)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(where[0], where[1], 0, 0))
        self.assertEqual(surf.get_at((0 + radius, where[1])), bgcolor)
        surf.fill(bgcolor)
        where = (surf.get_width() + radius * 2, 80)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(where[0], where[1], 0, 0))
        self.assertEqual(surf.get_at((0, where[1])), bgcolor)
        self.assertEqual(surf.get_at((0 + radius // 2, where[1])), bgcolor)
        self.assertEqual(surf.get_at((surf.get_width() - 1, where[1])), bgcolor)
        self.assertEqual(surf.get_at((surf.get_width() - radius, where[1])), bgcolor)
        surf.fill(bgcolor)
        where = (-1, 80)
        bounding_rect1 = self.draw_circle(surf, color, where, radius=radius)
        self.assertEqual(bounding_rect1, pygame.Rect(0, where[1] - radius, where[0] + radius, radius * 2))
        self.assertEqual(surf.get_at((where[0] if where[0] > 0 else 0, where[1])), color)
        self.assertEqual(surf.get_at((where[0] + radius, where[1])), bgcolor)
        self.assertEqual(surf.get_at((where[0] + radius - 1, where[1])), color)

class DrawCircleTest(DrawCircleMixin, DrawTestCase):
    """Test draw module function circle.

    This class inherits the general tests from DrawCircleMixin. It is also
    the class to add any draw.circle specific tests to.
    """

class DrawArcMixin:
    """Mixin tests for drawing arcs.

    This class contains all the general arc drawing tests.
    """

    def test_arc__args(self):
        if False:
            return 10
        'Ensures draw arc accepts the correct args.'
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (0, 10, 0, 50), (1, 1, 2, 2), 0, 1, 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__args_without_width(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw arc accepts the args without a width.'
        bounds_rect = self.draw_arc(pygame.Surface((2, 2)), (1, 1, 1, 99), pygame.Rect((0, 0), (2, 2)), 1.1, 2.1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__args_with_negative_width(self):
        if False:
            return 10
        'Ensures draw arc accepts the args with negative width.'
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (10, 10, 50, 50), (1, 1, 2, 2), 0, 1, -1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        self.assertEqual(bounds_rect, pygame.Rect(1, 1, 0, 0))

    def test_arc__args_with_width_gt_radius(self):
        if False:
            print('Hello World!')
        'Ensures draw arc accepts the args with\n        width > rect.w // 2 and width > rect.h // 2.\n        '
        rect = pygame.Rect((0, 0), (4, 4))
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (10, 10, 50, 50), rect, 0, 45, rect.w // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)
        bounds_rect = self.draw_arc(pygame.Surface((3, 3)), (10, 10, 50, 50), rect, 0, 45, rect.h // 2 + 1)
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__kwargs(self):
        if False:
            return 10
        'Ensures draw arc accepts the correct kwargs\n        with and without a width arg.\n        '
        kwargs_list = [{'surface': pygame.Surface((4, 4)), 'color': pygame.Color('yellow'), 'rect': pygame.Rect((0, 0), (3, 2)), 'start_angle': 0.5, 'stop_angle': 3, 'width': 1}, {'surface': pygame.Surface((2, 1)), 'color': (0, 10, 20), 'rect': (0, 0, 2, 2), 'start_angle': 1, 'stop_angle': 3.1}]
        for kwargs in kwargs_list:
            bounds_rect = self.draw_arc(**kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__kwargs_order_independent(self):
        if False:
            print('Hello World!')
        "Ensures draw arc's kwargs are not order dependent."
        bounds_rect = self.draw_arc(stop_angle=1, start_angle=2.2, color=(1, 2, 3), surface=pygame.Surface((3, 2)), width=1, rect=pygame.Rect((1, 0), (2, 3)))
        self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__args_missing(self):
        if False:
            print('Hello World!')
        'Ensures draw arc detects any missing required args.'
        surface = pygame.Surface((1, 1))
        color = pygame.Color('red')
        rect = pygame.Rect((0, 0), (2, 2))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, 0.1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc()

    def test_arc__kwargs_missing(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw arc detects any missing required kwargs.'
        kwargs = {'surface': pygame.Surface((1, 2)), 'color': pygame.Color('red'), 'rect': pygame.Rect((1, 0), (2, 2)), 'start_angle': 0.1, 'stop_angle': 2, 'width': 1}
        for name in ('stop_angle', 'start_angle', 'rect', 'color', 'surface'):
            invalid_kwargs = dict(kwargs)
            invalid_kwargs.pop(name)
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**invalid_kwargs)

    def test_arc__arg_invalid_types(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw arc detects invalid arg types.'
        surface = pygame.Surface((2, 2))
        color = pygame.Color('blue')
        rect = pygame.Rect((1, 1), (3, 3))
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, 0, 1, '1')
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, 0, '1', 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, rect, '1', 0, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, color, (1, 2, 3, 4, 5), 0, 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(surface, 2.3, rect, 0, 1, 1)
        with self.assertRaises(TypeError):
            bounds_rect = self.draw_arc(rect, color, rect, 0, 1, 1)

    def test_arc__kwarg_invalid_types(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw arc detects invalid kwarg types.'
        surface = pygame.Surface((3, 3))
        color = pygame.Color('green')
        rect = pygame.Rect((0, 1), (4, 2))
        start = 3
        stop = 4
        kwargs_list = [{'surface': pygame.Surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': 2.3, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': color, 'rect': (0, 0, 0), 'start_angle': start, 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': '1', 'stop_angle': stop, 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': '1', 'width': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1.1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**kwargs)

    def test_arc__kwarg_invalid_name(self):
        if False:
            return 10
        'Ensures draw arc detects invalid kwarg names.'
        surface = pygame.Surface((2, 3))
        color = pygame.Color('cyan')
        rect = pygame.Rect((0, 1), (2, 2))
        start = 0.9
        stop = 2.3
        kwargs_list = [{'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': 1, 'invalid': 1}, {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'invalid': 1}]
        for kwargs in kwargs_list:
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**kwargs)

    def test_arc__args_and_kwargs(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw arc accepts a combination of args/kwargs'
        surface = pygame.Surface((3, 1))
        color = (255, 255, 0, 0)
        rect = pygame.Rect((1, 0), (2, 3))
        start = 0.6
        stop = 2
        width = 1
        kwargs = {'surface': surface, 'color': color, 'rect': rect, 'start_angle': start, 'stop_angle': stop, 'width': width}
        for name in ('surface', 'color', 'rect', 'start_angle', 'stop_angle'):
            kwargs.pop(name)
            if 'surface' == name:
                bounds_rect = self.draw_arc(surface, **kwargs)
            elif 'color' == name:
                bounds_rect = self.draw_arc(surface, color, **kwargs)
            elif 'rect' == name:
                bounds_rect = self.draw_arc(surface, color, rect, **kwargs)
            elif 'start_angle' == name:
                bounds_rect = self.draw_arc(surface, color, rect, start, **kwargs)
            elif 'stop_angle' == name:
                bounds_rect = self.draw_arc(surface, color, rect, start, stop, **kwargs)
            else:
                bounds_rect = self.draw_arc(surface, color, rect, start, stop, width, **kwargs)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__valid_width_values(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw arc accepts different width values.'
        arc_color = pygame.Color('yellow')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': arc_color, 'rect': rect, 'start_angle': 0, 'stop_angle': 7, 'width': None}
        for width in (-50, -10, -3, -2, -1, 0, 1, 2, 3, 10, 50):
            msg = f'width={width}'
            surface.fill(surface_color)
            kwargs['width'] = width
            expected_color = arc_color if width > 0 else surface_color
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color, msg)
            self.assertIsInstance(bounds_rect, pygame.Rect, msg)

    def test_arc__valid_stop_angle_values(self):
        if False:
            return 10
        'Ensures draw arc accepts different stop_angle values.'
        expected_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx, rect.centery + 1)
        kwargs = {'surface': surface, 'color': expected_color, 'rect': rect, 'start_angle': -17, 'stop_angle': None, 'width': 1}
        for stop_angle in (-10, -5.5, -1, 0, 1, 5.5, 10):
            msg = f'stop_angle={stop_angle}'
            surface.fill(surface_color)
            kwargs['stop_angle'] = stop_angle
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color, msg)
            self.assertIsInstance(bounds_rect, pygame.Rect, msg)

    def test_arc__valid_start_angle_values(self):
        if False:
            return 10
        'Ensures draw arc accepts different start_angle values.'
        expected_color = pygame.Color('blue')
        surface_color = pygame.Color('white')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': expected_color, 'rect': rect, 'start_angle': None, 'stop_angle': 17, 'width': 1}
        for start_angle in (-10.0, -5.5, -1, 0, 1, 5.5, 10.0):
            msg = f'start_angle={start_angle}'
            surface.fill(surface_color)
            kwargs['start_angle'] = start_angle
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color, msg)
            self.assertIsInstance(bounds_rect, pygame.Rect, msg)

    def test_arc__valid_rect_formats(self):
        if False:
            i = 10
            return i + 15
        'Ensures draw arc accepts different rect formats.'
        expected_color = pygame.Color('red')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': expected_color, 'rect': None, 'start_angle': 0, 'stop_angle': 7, 'width': 1}
        rects = (rect, (rect.topleft, rect.size), (rect.x, rect.y, rect.w, rect.h))
        for rect in rects:
            surface.fill(surface_color)
            kwargs['rect'] = rect
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__valid_color_formats(self):
        if False:
            print('Hello World!')
        'Ensures draw arc accepts different color formats.'
        green_color = pygame.Color('green')
        surface_color = pygame.Color('black')
        surface = pygame.Surface((6, 6))
        rect = pygame.Rect((0, 0), (4, 4))
        rect.center = surface.get_rect().center
        pos = (rect.centerx + 1, rect.centery + 1)
        kwargs = {'surface': surface, 'color': None, 'rect': rect, 'start_angle': 0, 'stop_angle': 7, 'width': 1}
        greens = ((0, 255, 0), (0, 255, 0, 255), surface.map_rgb(green_color), green_color)
        for color in greens:
            surface.fill(surface_color)
            kwargs['color'] = color
            if isinstance(color, int):
                expected_color = surface.unmap_rgb(color)
            else:
                expected_color = green_color
            bounds_rect = self.draw_arc(**kwargs)
            self.assertEqual(surface.get_at(pos), expected_color)
            self.assertIsInstance(bounds_rect, pygame.Rect)

    def test_arc__invalid_color_formats(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures draw arc handles invalid color formats correctly.'
        pos = (1, 1)
        surface = pygame.Surface((4, 3))
        kwargs = {'surface': surface, 'color': None, 'rect': pygame.Rect(pos, (2, 2)), 'start_angle': 5, 'stop_angle': 6.1, 'width': 1}
        for expected_color in (2.3, self):
            kwargs['color'] = expected_color
            with self.assertRaises(TypeError):
                bounds_rect = self.draw_arc(**kwargs)

    def test_arc(self):
        if False:
            while True:
                i = 10
        'Ensure draw arc works correctly.'
        black = pygame.Color('black')
        red = pygame.Color('red')
        surface = pygame.Surface((100, 150))
        surface.fill(black)
        rect = (0, 0, 80, 40)
        start_angle = 0.0
        stop_angle = 3.14
        width = 3
        pygame.draw.arc(surface, red, rect, start_angle, stop_angle, width)
        pygame.image.save(surface, 'arc.png')
        x = 20
        for y in range(2, 5):
            self.assertEqual(surface.get_at((x, y)), red)
        self.assertEqual(surface.get_at((0, 0)), black)

    def test_arc__bounding_rect(self):
        if False:
            return 10
        'Ensures draw arc returns the correct bounding rect.\n\n        Tests arcs on and off the surface and a range of width/thickness\n        values.\n        '
        arc_color = pygame.Color('red')
        surf_color = pygame.Color('black')
        min_width = min_height = 5
        max_width = max_height = 7
        sizes = ((min_width, min_height), (max_width, max_height))
        surface = pygame.Surface((20, 20), 0, 32)
        surf_rect = surface.get_rect()
        big_rect = surf_rect.inflate(min_width * 2 + 1, min_height * 2 + 1)
        start_angle = 0
        stop_angles = (0, 2, 3, 5, math.ceil(2 * math.pi))
        for pos in rect_corners_mids_and_center(surf_rect) + rect_corners_mids_and_center(big_rect):
            for attr in RECT_POSITION_ATTRIBUTES:
                for (width, height) in sizes:
                    arc_rect = pygame.Rect((0, 0), (width, height))
                    setattr(arc_rect, attr, pos)
                    for thickness in (0, 1, 2, 3, min(width, height)):
                        for stop_angle in stop_angles:
                            surface.fill(surf_color)
                            bounding_rect = self.draw_arc(surface, arc_color, arc_rect, start_angle, stop_angle, thickness)
                            expected_rect = create_bounding_rect(surface, surf_color, arc_rect.topleft)
                            self.assertEqual(bounding_rect, expected_rect, f'thickness={thickness}')

    def test_arc__surface_clip(self):
        if False:
            for i in range(10):
                print('nop')
        "Ensures draw arc respects a surface's clip area."
        surfw = surfh = 30
        start = 0.1
        end = 0
        arc_color = pygame.Color('red')
        surface_color = pygame.Color('green')
        surface = pygame.Surface((surfw, surfh))
        surface.fill(surface_color)
        clip_rect = pygame.Rect((0, 0), (11, 11))
        clip_rect.center = surface.get_rect().center
        pos_rect = clip_rect.copy()
        for thickness in (1, 3):
            for center in rect_corners_mids_and_center(clip_rect):
                pos_rect.center = center
                surface.set_clip(None)
                surface.fill(surface_color)
                self.draw_arc(surface, arc_color, pos_rect, start, end, thickness)
                expected_pts = get_color_points(surface, arc_color, clip_rect)
                surface.fill(surface_color)
                surface.set_clip(clip_rect)
                self.draw_arc(surface, arc_color, pos_rect, start, end, thickness)
                surface.lock()
                for pt in ((x, y) for x in range(surfw) for y in range(surfh)):
                    if pt in expected_pts:
                        expected_color = arc_color
                    else:
                        expected_color = surface_color
                    self.assertEqual(surface.get_at(pt), expected_color, pt)
                surface.unlock()

class DrawArcTest(DrawArcMixin, DrawTestCase):
    """Test draw module function arc.

    This class inherits the general tests from DrawArcMixin. It is also the
    class to add any draw.arc specific tests to.
    """

class DrawModuleTest(unittest.TestCase):
    """General draw module tests."""

    def test_path_data_validation(self):
        if False:
            print('Hello World!')
        'Test validation of multi-point drawing methods.\n\n        See bug #521\n        '
        surf = pygame.Surface((5, 5))
        rect = pygame.Rect(0, 0, 5, 5)
        bad_values = ('text', b'bytes', 1 + 1j, object(), lambda x: x)
        bad_points = list(bad_values) + [(1,), (1, 2, 3)]
        bad_points.extend(((1, v) for v in bad_values))
        good_path = [(1, 1), (1, 3), (3, 3), (3, 1)]
        check_pts = [(x, y) for x in range(5) for y in range(5)]
        for (method, is_polgon) in ((draw.lines, 0), (draw.aalines, 0), (draw.polygon, 1)):
            for val in bad_values:
                draw.rect(surf, RED, rect, 0)
                with self.assertRaises(TypeError):
                    if is_polgon:
                        method(surf, GREEN, [val] + good_path, 0)
                    else:
                        method(surf, GREEN, True, [val] + good_path)
                self.assertTrue(all((surf.get_at(pt) == RED for pt in check_pts)))
                draw.rect(surf, RED, rect, 0)
                with self.assertRaises(TypeError):
                    path = good_path[:2] + [val] + good_path[2:]
                    if is_polgon:
                        method(surf, GREEN, path, 0)
                    else:
                        method(surf, GREEN, True, path)
                self.assertTrue(all((surf.get_at(pt) == RED for pt in check_pts)))

    def test_color_validation(self):
        if False:
            i = 10
            return i + 15
        surf = pygame.Surface((10, 10))
        colors = (123456, (1, 10, 100), RED, '#ab12df', 'red')
        points = ((0, 0), (1, 1), (1, 0))
        for col in colors:
            draw.line(surf, col, (0, 0), (1, 1))
            draw.aaline(surf, col, (0, 0), (1, 1))
            draw.aalines(surf, col, True, points)
            draw.lines(surf, col, True, points)
            draw.arc(surf, col, pygame.Rect(0, 0, 3, 3), 15, 150)
            draw.ellipse(surf, col, pygame.Rect(0, 0, 3, 6), 1)
            draw.circle(surf, col, (7, 3), 2)
            draw.polygon(surf, col, points, 0)
        for col in (1.256, object(), None):
            with self.assertRaises(TypeError):
                draw.line(surf, col, (0, 0), (1, 1))
            with self.assertRaises(TypeError):
                draw.aaline(surf, col, (0, 0), (1, 1))
            with self.assertRaises(TypeError):
                draw.aalines(surf, col, True, points)
            with self.assertRaises(TypeError):
                draw.lines(surf, col, True, points)
            with self.assertRaises(TypeError):
                draw.arc(surf, col, pygame.Rect(0, 0, 3, 3), 15, 150)
            with self.assertRaises(TypeError):
                draw.ellipse(surf, col, pygame.Rect(0, 0, 3, 6), 1)
            with self.assertRaises(TypeError):
                draw.circle(surf, col, (7, 3), 2)
            with self.assertRaises(TypeError):
                draw.polygon(surf, col, points, 0)
if __name__ == '__main__':
    unittest.main()