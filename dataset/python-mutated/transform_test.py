import unittest
import os
import platform
from pygame.tests import test_utils
from pygame.tests.test_utils import example_path
import pygame
import pygame.transform
from pygame.locals import *

def show_image(s, images=[]):
    if False:
        while True:
            i = 10
    size = s.get_rect()[2:]
    screen = pygame.display.set_mode(size)
    screen.blit(s, (0, 0))
    pygame.display.flip()
    pygame.event.pump()
    going = True
    idx = 0
    while going:
        events = pygame.event.get()
        for e in events:
            if e.type == QUIT:
                going = False
            if e.type == KEYDOWN:
                if e.key in [K_s, K_a]:
                    if e.key == K_s:
                        idx += 1
                    if e.key == K_a:
                        idx -= 1
                    s = images[idx]
                    screen.blit(s, (0, 0))
                    pygame.display.flip()
                    pygame.event.pump()
                elif e.key in [K_ESCAPE]:
                    going = False
    pygame.display.quit()
    pygame.display.init()

def threshold(return_surf, surf, color, threshold=(0, 0, 0), diff_color=(0, 0, 0), change_return=True):
    if False:
        print('Hello World!')
    'given the color it makes return_surf only have areas with the given colour.'
    (width, height) = (surf.get_width(), surf.get_height())
    if change_return:
        return_surf.fill(diff_color)
    try:
        (r, g, b) = color
    except ValueError:
        (r, g, b, a) = color
    try:
        (tr, tg, tb) = color
    except ValueError:
        (tr, tg, tb, ta) = color
    similar = 0
    for y in range(height):
        for x in range(width):
            c1 = surf.get_at((x, y))
            if (abs(c1[0] - r) < tr) & (abs(c1[1] - g) < tg) & (abs(c1[2] - b) < tb):
                if change_return:
                    return_surf.set_at((x, y), c1)
                similar += 1
    return similar

class TransformModuleTest(unittest.TestCase):

    def test_scale__alpha(self):
        if False:
            i = 10
            return i + 15
        'see if set_alpha information is kept.'
        s = pygame.Surface((32, 32))
        s.set_alpha(55)
        self.assertEqual(s.get_alpha(), 55)
        s = pygame.Surface((32, 32))
        s.set_alpha(55)
        s2 = pygame.transform.scale(s, (64, 64))
        s3 = s.copy()
        self.assertEqual(s.get_alpha(), s3.get_alpha())
        self.assertEqual(s.get_alpha(), s2.get_alpha())

    def test_scale__destination(self):
        if False:
            print('Hello World!')
        'see if the destination surface can be passed in to use.'
        s = pygame.Surface((32, 32))
        s2 = pygame.transform.scale(s, (64, 64))
        s3 = s2.copy()
        s3 = pygame.transform.scale(surface=s, size=(64, 64), dest_surface=s3)
        pygame.transform.scale(s, (64, 64), s2)
        self.assertRaises(ValueError, pygame.transform.scale, s, (33, 64), s3)
        s = pygame.Surface((32, 32))
        s2 = pygame.transform.smoothscale(s, (64, 64))
        s3 = s2.copy()
        s3 = pygame.transform.smoothscale(surface=s, size=(64, 64), dest_surface=s3)
        self.assertRaises(ValueError, pygame.transform.smoothscale, s, (33, 64), s3)

    def test_scale__vector2(self):
        if False:
            for i in range(10):
                print('nop')
        s = pygame.Surface((32, 32))
        s2 = pygame.transform.scale(s, pygame.Vector2(64, 64))
        s3 = pygame.transform.smoothscale(s, pygame.Vector2(64, 64))
        self.assertEqual((64, 64), s2.get_size())
        self.assertEqual((64, 64), s3.get_size())

    def test_scale__zero_surface_transform(self):
        if False:
            while True:
                i = 10
        tmp_surface = pygame.transform.scale(pygame.Surface((128, 128)), (0, 0))
        self.assertEqual(tmp_surface.get_size(), (0, 0))
        tmp_surface = pygame.transform.scale(tmp_surface, (128, 128))
        self.assertEqual(tmp_surface.get_size(), (128, 128))

    def test_scale_by(self):
        if False:
            print('Hello World!')
        s = pygame.Surface((32, 32))
        s2 = pygame.transform.scale_by(s, 2)
        self.assertEqual((64, 64), s2.get_size())
        s2 = pygame.transform.scale_by(s, factor=(2.0, 1.5))
        self.assertEqual((64, 48), s2.get_size())
        dest = pygame.Surface((64, 48))
        pygame.transform.scale_by(s, (2.0, 1.5), dest_surface=dest)

    def test_smoothscale_by(self):
        if False:
            while True:
                i = 10
        s = pygame.Surface((32, 32))
        s2 = pygame.transform.smoothscale_by(s, 2)
        self.assertEqual((64, 64), s2.get_size())
        s2 = pygame.transform.smoothscale_by(s, factor=(2.0, 1.5))
        self.assertEqual((64, 48), s2.get_size())
        dest = pygame.Surface((64, 48))
        pygame.transform.smoothscale_by(s, (2.0, 1.5), dest_surface=dest)

    def test_grayscale(self):
        if False:
            return 10
        s = pygame.Surface((32, 32))
        s.fill((255, 0, 0))
        s2 = pygame.transform.grayscale(s)
        self.assertEqual(pygame.transform.average_color(s2)[0], 76)
        self.assertEqual(pygame.transform.average_color(s2)[1], 76)
        self.assertEqual(pygame.transform.average_color(s2)[2], 76)
        dest = pygame.Surface((32, 32), depth=32)
        pygame.transform.grayscale(s, dest)
        self.assertEqual(pygame.transform.average_color(dest)[0], 76)
        self.assertEqual(pygame.transform.average_color(dest)[1], 76)
        self.assertEqual(pygame.transform.average_color(dest)[2], 76)
        dest = pygame.Surface((32, 32), depth=32)
        s.fill((34, 12, 65))
        pygame.transform.grayscale(s, dest)
        self.assertEqual(pygame.transform.average_color(dest)[0], 24)
        self.assertEqual(pygame.transform.average_color(dest)[1], 24)
        self.assertEqual(pygame.transform.average_color(dest)[2], 24)
        dest = pygame.Surface((32, 32), depth=32)
        s.fill((123, 123, 123))
        pygame.transform.grayscale(s, dest)
        self.assertIn(pygame.transform.average_color(dest)[0], [123, 122])
        self.assertIn(pygame.transform.average_color(dest)[1], [123, 122])
        self.assertIn(pygame.transform.average_color(dest)[2], [123, 122])
        s = pygame.Surface((32, 32), depth=24)
        s.fill((255, 0, 0))
        dest = pygame.Surface((32, 32), depth=24)
        pygame.transform.grayscale(s, dest)
        self.assertEqual(pygame.transform.average_color(dest)[0], 76)
        self.assertEqual(pygame.transform.average_color(dest)[1], 76)
        self.assertEqual(pygame.transform.average_color(dest)[2], 76)
        s = pygame.Surface((32, 32), depth=16)
        s.fill((255, 0, 0))
        dest = pygame.Surface((32, 32), depth=16)
        pygame.transform.grayscale(s, dest)
        self.assertEqual(pygame.transform.average_color(dest)[0], 72)
        self.assertEqual(pygame.transform.average_color(dest)[1], 76)
        self.assertEqual(pygame.transform.average_color(dest)[2], 72)

    def test_threshold__honors_third_surface(self):
        if False:
            print('Hello World!')
        (w, h) = size = (32, 32)
        threshold = (20, 20, 20, 20)
        original_color = (25, 25, 25, 25)
        threshold_color = (10, 10, 10, 10)
        original_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        dest_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        third_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        original_surface.fill(original_color)
        third_surface.fill(threshold_color)
        pixels_within_threshold = pygame.transform.threshold(dest_surface=None, surface=original_surface, search_color=threshold_color, threshold=threshold, set_color=None, set_behavior=0)
        self.assertEqual(w * h, pixels_within_threshold)
        pixels_within_threshold = pygame.transform.threshold(dest_surface=None, surface=original_surface, search_color=None, threshold=threshold, set_color=None, set_behavior=0, search_surf=third_surface)
        self.assertEqual(w * h, pixels_within_threshold)

    def test_threshold_dest_surf_not_change(self):
        if False:
            while True:
                i = 10
        'the pixels within the threshold.\n\n        All pixels not within threshold are changed to set_color.\n        So there should be none changed in this test.\n        '
        (w, h) = size = (32, 32)
        threshold = (20, 20, 20, 20)
        original_color = (25, 25, 25, 25)
        original_dest_color = (65, 65, 65, 55)
        threshold_color = (10, 10, 10, 10)
        set_color = (255, 10, 10, 10)
        surf = pygame.Surface(size, pygame.SRCALPHA, 32)
        dest_surf = pygame.Surface(size, pygame.SRCALPHA, 32)
        search_surf = pygame.Surface(size, pygame.SRCALPHA, 32)
        surf.fill(original_color)
        search_surf.fill(threshold_color)
        dest_surf.fill(original_dest_color)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR = 1
        pixels_within_threshold = pygame.transform.threshold(dest_surface=dest_surf, surface=surf, search_color=None, threshold=threshold, set_color=set_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR, search_surf=search_surf)
        self.assertEqual(w * h, pixels_within_threshold)
        dest_rect = dest_surf.get_rect()
        dest_size = dest_rect.size
        self.assertEqual(size, dest_size)
        for pt in test_utils.rect_area_pts(dest_rect):
            self.assertNotEqual(dest_surf.get_at(pt), set_color)
            self.assertEqual(dest_surf.get_at(pt), original_dest_color)

    def test_threshold_dest_surf_all_changed(self):
        if False:
            while True:
                i = 10
        'Lowering the threshold, expecting changed surface'
        (w, h) = size = (32, 32)
        threshold = (20, 20, 20, 20)
        original_color = (25, 25, 25, 25)
        original_dest_color = (65, 65, 65, 55)
        threshold_color = (10, 10, 10, 10)
        set_color = (255, 10, 10, 10)
        surf = pygame.Surface(size, pygame.SRCALPHA, 32)
        dest_surf = pygame.Surface(size, pygame.SRCALPHA, 32)
        search_surf = pygame.Surface(size, pygame.SRCALPHA, 32)
        surf.fill(original_color)
        search_surf.fill(threshold_color)
        dest_surf.fill(original_dest_color)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR = 1
        pixels_within_threshold = pygame.transform.threshold(dest_surf, surf, search_color=None, set_color=set_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR, search_surf=search_surf)
        self.assertEqual(0, pixels_within_threshold)
        dest_rect = dest_surf.get_rect()
        dest_size = dest_rect.size
        self.assertEqual(size, dest_size)
        for pt in test_utils.rect_area_pts(dest_rect):
            self.assertEqual(dest_surf.get_at(pt), set_color)

    def test_threshold_count(self):
        if False:
            i = 10
            return i + 15
        'counts the colors, and not changes them.'
        surf_size = (32, 32)
        surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
        search_surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
        search_color = (55, 55, 55, 255)
        original_color = (10, 10, 10, 255)
        surf.fill(original_color)
        surf.set_at((0, 0), search_color)
        surf.set_at((12, 5), search_color)
        self.assertRaises(TypeError, pygame.transform.threshold, None, surf, search_color)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
        self.assertRaises(TypeError, pygame.transform.threshold, None, surf, search_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF)
        THRESHOLD_BEHAVIOR_COUNT = 0
        num_threshold_pixels = pygame.transform.threshold(dest_surface=None, surface=surf, search_color=search_color, set_behavior=THRESHOLD_BEHAVIOR_COUNT)
        self.assertEqual(num_threshold_pixels, 2)

    def test_threshold_search_surf(self):
        if False:
            print('Hello World!')
        surf_size = (32, 32)
        surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
        search_surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
        dest_surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
        original_color = (10, 10, 10, 255)
        search_color = (55, 55, 55, 255)
        surf.fill(original_color)
        dest_surf.fill(original_color)
        surf.set_at((0, 0), search_color)
        surf.set_at((12, 5), search_color)
        search_surf.fill(search_color)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
        self.assertRaises(TypeError, pygame.transform.threshold, dest_surf, surf, search_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf)
        different_sized_surf = pygame.Surface((22, 33), pygame.SRCALPHA, 32)
        self.assertRaises(TypeError, pygame.transform.threshold, different_sized_surf, surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf)
        self.assertRaises(TypeError, pygame.transform.threshold, dest_surf, surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=different_sized_surf)
        num_threshold_pixels = pygame.transform.threshold(dest_surface=dest_surf, surface=surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf)
        num_pixels_within = 2
        self.assertEqual(num_threshold_pixels, num_pixels_within)
        dest_surf.fill(original_color)
        num_threshold_pixels = pygame.transform.threshold(dest_surf, surf, search_color=None, set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, search_surf=search_surf, inverse_set=True)
        self.assertEqual(num_threshold_pixels, 2)

    def test_threshold_inverse_set(self):
        if False:
            print('Hello World!')
        'changes the pixels within the threshold, and not outside.'
        surf_size = (32, 32)
        _dest_surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
        _surf = pygame.Surface(surf_size, pygame.SRCALPHA, 32)
        dest_surf = _dest_surf
        surf = _surf
        search_color = (55, 55, 55, 255)
        threshold = (0, 0, 0, 0)
        set_color = (245, 245, 245, 255)
        inverse_set = 1
        original_color = (10, 10, 10, 255)
        surf.fill(original_color)
        surf.set_at((0, 0), search_color)
        surf.set_at((12, 5), search_color)
        dest_surf.fill(original_color)
        dest_surf.set_at((0, 0), search_color)
        dest_surf.set_at((12, 5), search_color)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR = 1
        num_threshold_pixels = pygame.transform.threshold(dest_surf, surf, search_color=search_color, threshold=threshold, set_color=set_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR, inverse_set=1)
        self.assertEqual(num_threshold_pixels, 2)
        self.assertEqual(dest_surf.get_at((0, 0)), set_color)
        self.assertEqual(dest_surf.get_at((12, 5)), set_color)
        self.assertEqual(dest_surf.get_at((2, 2)), original_color)

    def test_threshold_non_src_alpha(self):
        if False:
            i = 10
            return i + 15
        result = pygame.Surface((10, 10))
        s1 = pygame.Surface((10, 10))
        s2 = pygame.Surface((10, 10))
        s3 = pygame.Surface((10, 10))
        s4 = pygame.Surface((10, 10))
        x = s1.fill((0, 0, 0))
        s1.set_at((0, 0), (32, 20, 0))
        x = s2.fill((0, 20, 0))
        x = s3.fill((0, 0, 0))
        x = s4.fill((0, 0, 0))
        s2.set_at((0, 0), (33, 21, 0))
        s2.set_at((3, 0), (63, 61, 0))
        s3.set_at((0, 0), (112, 31, 0))
        s4.set_at((0, 0), (11, 31, 0))
        s4.set_at((1, 1), (12, 31, 0))
        self.assertEqual(s1.get_at((0, 0)), (32, 20, 0, 255))
        self.assertEqual(s2.get_at((0, 0)), (33, 21, 0, 255))
        self.assertEqual((0, 0), (s1.get_flags(), s2.get_flags()))
        similar_color = (255, 255, 255, 255)
        diff_color = (222, 0, 0, 255)
        threshold_color = (20, 20, 20, 255)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR = 1
        num_threshold_pixels = pygame.transform.threshold(dest_surface=result, surface=s1, search_color=similar_color, threshold=threshold_color, set_color=diff_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR)
        self.assertEqual(num_threshold_pixels, 0)
        num_threshold_pixels = pygame.transform.threshold(dest_surface=result, surface=s1, search_color=(40, 40, 0), threshold=threshold_color, set_color=diff_color, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_COLOR)
        self.assertEqual(num_threshold_pixels, 1)
        self.assertEqual(result.get_at((0, 0)), diff_color)

    def test_threshold__uneven_colors(self):
        if False:
            while True:
                i = 10
        (w, h) = size = (16, 16)
        original_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        dest_surface = pygame.Surface(size, pygame.SRCALPHA, 32)
        original_surface.fill(0)
        threshold_color_template = [5, 5, 5, 5]
        threshold_template = [6, 6, 6, 6]
        for pos in range(len('rgb')):
            threshold_color = threshold_color_template[:]
            threshold = threshold_template[:]
            threshold_color[pos] = 45
            threshold[pos] = 50
            pixels_within_threshold = pygame.transform.threshold(None, original_surface, threshold_color, threshold, set_color=None, set_behavior=0)
            self.assertEqual(w * h, pixels_within_threshold)

    def test_threshold_set_behavior2(self):
        if False:
            for i in range(10):
                print('nop')
        'raises an error when set_behavior=2 and set_color is not None.'
        from pygame.transform import threshold
        s1 = pygame.Surface((32, 32), SRCALPHA, 32)
        s2 = pygame.Surface((32, 32), SRCALPHA, 32)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
        self.assertRaises(TypeError, threshold, dest_surface=s2, surface=s1, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=(255, 0, 0), set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF)

    def test_threshold_set_behavior0(self):
        if False:
            for i in range(10):
                print('nop')
        'raises an error when set_behavior=1\n        and set_color is not None,\n        and dest_surf is not None.\n        '
        from pygame.transform import threshold
        s1 = pygame.Surface((32, 32), SRCALPHA, 32)
        s2 = pygame.Surface((32, 32), SRCALPHA, 32)
        THRESHOLD_BEHAVIOR_COUNT = 0
        self.assertRaises(TypeError, threshold, dest_surface=None, surface=s2, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=(0, 0, 0), set_behavior=THRESHOLD_BEHAVIOR_COUNT)
        self.assertRaises(TypeError, threshold, dest_surface=s1, surface=s2, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_COUNT)
        threshold(dest_surface=None, surface=s2, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_COUNT)

    def test_threshold_from_surface(self):
        if False:
            return 10
        "Set similar pixels in 'dest_surf' to color in the 'surf'."
        from pygame.transform import threshold
        surf = pygame.Surface((32, 32), SRCALPHA, 32)
        dest_surf = pygame.Surface((32, 32), SRCALPHA, 32)
        surf_color = (40, 40, 40, 255)
        dest_color = (255, 255, 255)
        surf.fill(surf_color)
        dest_surf.fill(dest_color)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
        num_threshold_pixels = threshold(dest_surface=dest_surf, surface=surf, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF, inverse_set=1)
        self.assertEqual(num_threshold_pixels, dest_surf.get_height() * dest_surf.get_width())
        self.assertEqual(dest_surf.get_at((0, 0)), surf_color)

    def test_threshold__surface(self):
        if False:
            return 10
        ' '
        from pygame.transform import threshold
        s1 = pygame.Surface((32, 32), SRCALPHA, 32)
        s2 = pygame.Surface((32, 32), SRCALPHA, 32)
        s3 = pygame.Surface((1, 1), SRCALPHA, 32)
        THRESHOLD_BEHAVIOR_FROM_SEARCH_SURF = 2
        s1.fill((40, 40, 40))
        s1.set_at((0, 0), (170, 170, 170))
        THRESHOLD_BEHAVIOR_COUNT = 0
        num_threshold_pixels = threshold(dest_surface=None, surface=s1, search_color=(30, 30, 30), threshold=(11, 11, 11), set_color=None, set_behavior=THRESHOLD_BEHAVIOR_COUNT)
        self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width() - 1)
        s1.fill((254, 254, 254))
        s2.fill((255, 255, 255))
        s3.fill((255, 255, 255))
        s1.set_at((0, 0), (170, 170, 170))
        num_threshold_pixels = threshold(None, s1, (254, 254, 254), (1, 1, 1), None, THRESHOLD_BEHAVIOR_COUNT)
        self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width() - 1)
        num_threshold_pixels = threshold(None, s1, None, (1, 1, 1), None, THRESHOLD_BEHAVIOR_COUNT, s2)
        self.assertEqual(num_threshold_pixels, s1.get_height() * s1.get_width() - 1)
        num_threshold_pixels = threshold(None, s1, (253, 253, 253), (0, 0, 0), None, THRESHOLD_BEHAVIOR_COUNT)
        self.assertEqual(num_threshold_pixels, 0)
        num_threshold_pixels = threshold(None, s1, None, (0, 0, 0), None, THRESHOLD_BEHAVIOR_COUNT, s2)
        self.assertEqual(num_threshold_pixels, 0)

    def test_threshold__subclassed_surface(self):
        if False:
            print('Hello World!')
        'Ensure threshold accepts subclassed surfaces.'
        expected_size = (13, 11)
        expected_flags = 0
        expected_depth = 32
        expected_color = (90, 80, 70, 255)
        expected_count = 0
        surface = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
        dest_surface = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
        search_surface = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
        surface.fill((10, 10, 10))
        dest_surface.fill((255, 255, 255))
        search_surface.fill((20, 20, 20))
        count = pygame.transform.threshold(dest_surface=dest_surface, surface=surface, threshold=(1, 1, 1), set_color=expected_color, search_color=None, search_surf=search_surface)
        self.assertIsInstance(dest_surface, pygame.Surface)
        self.assertIsInstance(dest_surface, test_utils.SurfaceSubclass)
        self.assertEqual(count, expected_count)
        self.assertEqual(dest_surface.get_at((0, 0)), expected_color)
        self.assertEqual(dest_surface.get_bitsize(), expected_depth)
        self.assertEqual(dest_surface.get_size(), expected_size)
        self.assertEqual(dest_surface.get_flags(), expected_flags)

    def test_laplacian(self):
        if False:
            print('Hello World!')
        ' '
        SIZE = 32
        s1 = pygame.Surface((SIZE, SIZE))
        s2 = pygame.Surface((SIZE, SIZE))
        s1.fill((10, 10, 70))
        pygame.draw.line(s1, (255, 0, 0), (3, 10), (20, 20))
        pygame.draw.line(s1, (255, 0, 0), (0, 31), (31, 31))
        pygame.transform.laplacian(s1, s2)
        self.assertEqual(s2.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s2.get_at((3, 10)), (255, 0, 0, 255))
        self.assertEqual(s2.get_at((0, 31)), (255, 0, 0, 255))
        self.assertEqual(s2.get_at((31, 31)), (255, 0, 0, 255))
        s2 = pygame.transform.laplacian(s1)
        self.assertEqual(s2.get_at((0, 0)), (0, 0, 0, 255))
        self.assertEqual(s2.get_at((3, 10)), (255, 0, 0, 255))
        self.assertEqual(s2.get_at((0, 31)), (255, 0, 0, 255))
        self.assertEqual(s2.get_at((31, 31)), (255, 0, 0, 255))

    def test_laplacian__24_big_endian(self):
        if False:
            while True:
                i = 10
        ' '
        pygame.display.init()
        try:
            surf_1 = pygame.image.load(example_path(os.path.join('data', 'laplacian.png')))
            SIZE = 32
            surf_2 = pygame.Surface((SIZE, SIZE), 0, 24)
            pygame.transform.laplacian(surface=surf_1, dest_surface=surf_2)
            self.assertEqual(surf_2.get_at((0, 0)), (0, 0, 0, 255))
            self.assertEqual(surf_2.get_at((3, 10)), (255, 0, 0, 255))
            self.assertEqual(surf_2.get_at((0, 31)), (255, 0, 0, 255))
            self.assertEqual(surf_2.get_at((31, 31)), (255, 0, 0, 255))
            surf_2 = pygame.transform.laplacian(surf_1)
            self.assertEqual(surf_2.get_at((0, 0)), (0, 0, 0, 255))
            self.assertEqual(surf_2.get_at((3, 10)), (255, 0, 0, 255))
            self.assertEqual(surf_2.get_at((0, 31)), (255, 0, 0, 255))
            self.assertEqual(surf_2.get_at((31, 31)), (255, 0, 0, 255))
        finally:
            pygame.display.quit()

    def test_average_surfaces(self):
        if False:
            return 10
        ' '
        SIZE = 32
        s1 = pygame.Surface((SIZE, SIZE))
        s2 = pygame.Surface((SIZE, SIZE))
        s3 = pygame.Surface((SIZE, SIZE))
        s1.fill((10, 10, 70))
        s2.fill((10, 20, 70))
        s3.fill((10, 130, 10))
        surfaces = [s1, s2, s3]
        surfaces = [s1, s2]
        sr = pygame.transform.average_surfaces(surfaces)
        self.assertEqual(sr.get_at((0, 0)), (10, 15, 70, 255))
        self.assertRaises(TypeError, pygame.transform.average_surfaces, 1)
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [1])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [s1, 1])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [1, s1])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, [s1, s2, 1])
        self.assertRaises(TypeError, pygame.transform.average_surfaces, (s for s in [s1, s2, s3]))

    def test_average_surfaces__24(self):
        if False:
            for i in range(10):
                print('nop')
        SIZE = 32
        depth = 24
        s1 = pygame.Surface((SIZE, SIZE), 0, depth)
        s2 = pygame.Surface((SIZE, SIZE), 0, depth)
        s3 = pygame.Surface((SIZE, SIZE), 0, depth)
        s1.fill((10, 10, 70, 255))
        s2.fill((10, 20, 70, 255))
        s3.fill((10, 130, 10, 255))
        surfaces = [s1, s2, s3]
        sr = pygame.transform.average_surfaces(surfaces)
        self.assertEqual(sr.get_masks(), s1.get_masks())
        self.assertEqual(sr.get_flags(), s1.get_flags())
        self.assertEqual(sr.get_losses(), s1.get_losses())
        if 0:
            print(sr, s1)
            print(sr.get_masks(), s1.get_masks())
            print(sr.get_flags(), s1.get_flags())
            print(sr.get_losses(), s1.get_losses())
            print(sr.get_shifts(), s1.get_shifts())
        self.assertEqual(sr.get_at((0, 0)), (10, 53, 50, 255))

    def test_average_surfaces__24_big_endian(self):
        if False:
            for i in range(10):
                print('nop')
        pygame.display.init()
        try:
            surf_1 = pygame.image.load(example_path(os.path.join('data', 'BGR.png')))
            surf_2 = surf_1.copy()
            surfaces = [surf_1, surf_2]
            self.assertEqual(surf_1.get_at((0, 0)), (255, 0, 0, 255))
            self.assertEqual(surf_2.get_at((0, 0)), (255, 0, 0, 255))
            surf_av = pygame.transform.average_surfaces(surfaces)
            self.assertEqual(surf_av.get_masks(), surf_1.get_masks())
            self.assertEqual(surf_av.get_flags(), surf_1.get_flags())
            self.assertEqual(surf_av.get_losses(), surf_1.get_losses())
            self.assertEqual(surf_av.get_at((0, 0)), (255, 0, 0, 255))
        finally:
            pygame.display.quit()

    def test_average_surfaces__subclassed_surfaces(self):
        if False:
            return 10
        'Ensure average_surfaces accepts subclassed surfaces.'
        expected_size = (23, 17)
        expected_flags = 0
        expected_depth = 32
        expected_color = (50, 50, 50, 255)
        surfaces = []
        for color in ((40, 60, 40), (60, 40, 60)):
            s = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
            s.fill(color)
            surfaces.append(s)
        surface = pygame.transform.average_surfaces(surfaces)
        self.assertIsInstance(surface, pygame.Surface)
        self.assertNotIsInstance(surface, test_utils.SurfaceSubclass)
        self.assertEqual(surface.get_at((0, 0)), expected_color)
        self.assertEqual(surface.get_bitsize(), expected_depth)
        self.assertEqual(surface.get_size(), expected_size)
        self.assertEqual(surface.get_flags(), expected_flags)

    def test_average_surfaces__subclassed_destination_surface(self):
        if False:
            return 10
        'Ensure average_surfaces accepts a destination subclassed surface.'
        expected_size = (13, 27)
        expected_flags = 0
        expected_depth = 32
        expected_color = (15, 15, 15, 255)
        surfaces = []
        for color in ((10, 10, 20), (20, 20, 10), (30, 30, 30)):
            s = test_utils.SurfaceSubclass(expected_size, expected_flags, expected_depth)
            s.fill(color)
            surfaces.append(s)
        expected_dest_surface = surfaces.pop()
        dest_surface = pygame.transform.average_surfaces(surfaces=surfaces, dest_surface=expected_dest_surface)
        self.assertIsInstance(dest_surface, pygame.Surface)
        self.assertIsInstance(dest_surface, test_utils.SurfaceSubclass)
        self.assertIs(dest_surface, expected_dest_surface)
        self.assertEqual(dest_surface.get_at((0, 0)), expected_color)
        self.assertEqual(dest_surface.get_bitsize(), expected_depth)
        self.assertEqual(dest_surface.get_size(), expected_size)
        self.assertEqual(dest_surface.get_flags(), expected_flags)

    def test_average_color(self):
        if False:
            return 10
        ' '
        for i in (24, 32):
            with self.subTest(f'Testing {i}-bit surface'):
                s = pygame.Surface((32, 32), 0, i)
                s.fill((0, 100, 200))
                s.fill((10, 50, 100), (0, 0, 16, 32))
                self.assertEqual(pygame.transform.average_color(s), (5, 75, 150, 0))
                avg_color = pygame.transform.average_color(surface=s, rect=(16, 0, 16, 32))
                self.assertEqual(avg_color, (0, 100, 200, 0))

    def test_average_color_considering_alpha_all_pixels_opaque(self):
        if False:
            print('Hello World!')
        ' '
        s = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        s.fill((0, 100, 200, 255))
        s.fill((10, 50, 100, 255), (0, 0, 16, 32))
        self.assertEqual(pygame.transform.average_color(s, consider_alpha=True), (5, 75, 150, 255))
        avg_color = pygame.transform.average_color(surface=s, rect=(16, 0, 16, 32), consider_alpha=True)
        self.assertEqual(avg_color, (0, 100, 200, 255))

    def test_average_color_considering_alpha(self):
        if False:
            for i in range(10):
                print('nop')
        ' '
        s = pygame.Surface((32, 32), pygame.SRCALPHA, 32)
        s.fill((0, 100, 200, 255))
        s.fill((10, 50, 100, 128), (0, 0, 16, 32))
        self.assertEqual(pygame.transform.average_color(s, consider_alpha=True), (3, 83, 166, 191))
        avg_color = pygame.transform.average_color(surface=s, rect=(0, 0, 16, 32), consider_alpha=True)
        self.assertEqual(avg_color, (10, 50, 100, 128))

    def test_rotate(self):
        if False:
            return 10
        blue = (0, 0, 255, 255)
        red = (255, 0, 0, 255)
        black = (0, 0, 0)
        canvas = pygame.Surface((3, 3))
        rotation = 0
        canvas.set_at((2, 0), blue)
        canvas.set_at((0, 2), red)
        self.assertEqual(canvas.get_at((0, 0)), black)
        self.assertEqual(canvas.get_at((2, 0)), blue)
        self.assertEqual(canvas.get_at((0, 2)), red)
        for i in range(0, 4):
            if i % 2 == 0:
                self.assertEqual(canvas.get_at((0, 0)), black)
            elif i == 1:
                self.assertEqual(canvas.get_at((0, 0)), blue)
            elif i == 3:
                self.assertEqual(canvas.get_at((0, 0)), red)
            rotation += 90
            canvas = pygame.transform.rotate(surface=canvas, angle=90)
        self.assertEqual(canvas.get_at((0, 0)), black)

    def test_rotate_of_0_sized_surface(self):
        if False:
            print('Hello World!')
        canvas1 = pygame.Surface((0, 1))
        canvas2 = pygame.Surface((1, 0))
        pygame.transform.rotate(canvas1, 42)
        pygame.transform.rotate(canvas2, 42)

    def test_rotate__lossless_at_90_degrees(self):
        if False:
            print('Hello World!')
        (w, h) = (32, 32)
        s = pygame.Surface((w, h), pygame.SRCALPHA)
        gradient = list(test_utils.gradient(w, h))
        for (pt, color) in gradient:
            s.set_at(pt, color)
        for rotation in (90, -90):
            s = pygame.transform.rotate(s, rotation)
        for (pt, color) in gradient:
            self.assertTrue(s.get_at(pt) == color)

    def test_scale2x(self):
        if False:
            print('Hello World!')
        (w, h) = (32, 32)
        s = pygame.Surface((w, h), pygame.SRCALPHA, 32)
        s1 = pygame.transform.scale2x(s)
        s2 = pygame.transform.scale2x(surface=s)
        self.assertEqual(s1.get_rect().size, (64, 64))
        self.assertEqual(s2.get_rect().size, (64, 64))

    def test_scale2xraw(self):
        if False:
            print('Hello World!')
        (w, h) = (32, 32)
        s = pygame.Surface((w, h), pygame.SRCALPHA, 32)
        s.fill((0, 0, 0))
        pygame.draw.circle(s, (255, 0, 0), (w // 2, h // 2), w // 3)
        s2 = pygame.transform.scale(s, (w * 2, h * 2))
        s2_2 = pygame.transform.scale(s2, (w * 4, h * 4))
        s4 = pygame.transform.scale(s, (w * 4, h * 4))
        self.assertEqual(s2_2.get_rect().size, (128, 128))
        for pt in test_utils.rect_area_pts(s2_2.get_rect()):
            self.assertEqual(s2_2.get_at(pt), s4.get_at(pt))

    def test_get_smoothscale_backend(self):
        if False:
            for i in range(10):
                print('nop')
        filter_type = pygame.transform.get_smoothscale_backend()
        self.assertTrue(filter_type in ['GENERIC', 'MMX', 'SSE'])

    def test_set_smoothscale_backend(self):
        if False:
            print('Hello World!')
        original_type = pygame.transform.get_smoothscale_backend()
        pygame.transform.set_smoothscale_backend('GENERIC')
        filter_type = pygame.transform.get_smoothscale_backend()
        self.assertEqual(filter_type, 'GENERIC')
        pygame.transform.set_smoothscale_backend(backend=original_type)

        def change():
            if False:
                i = 10
                return i + 15
            pygame.transform.set_smoothscale_backend('mmx')
        self.assertRaises(ValueError, change)

        def change():
            if False:
                for i in range(10):
                    print('nop')
            pygame.transform.set_smoothscale_backend(t='GENERIC')
        self.assertRaises(TypeError, change)

        def change():
            if False:
                while True:
                    i = 10
            pygame.transform.set_smoothscale_backend(1)
        self.assertRaises(TypeError, change)
        if original_type != 'SSE':

            def change():
                if False:
                    return 10
                pygame.transform.set_smoothscale_backend('SSE')
            self.assertRaises(ValueError, change)
        filter_type = pygame.transform.get_smoothscale_backend()
        self.assertEqual(filter_type, original_type)

    def test_chop(self):
        if False:
            for i in range(10):
                print('nop')
        original_surface = pygame.Surface((20, 20))
        pygame.draw.rect(original_surface, (255, 0, 0), (0, 0, 10, 10))
        pygame.draw.rect(original_surface, (0, 255, 0), (0, 10, 10, 10))
        pygame.draw.rect(original_surface, (0, 0, 255), (10, 0, 10, 10))
        pygame.draw.rect(original_surface, (255, 255, 0), (10, 10, 10, 10))
        rect = pygame.Rect(0, 0, 5, 15)
        test_surface = pygame.transform.chop(original_surface, rect)
        self.assertEqual(test_surface.get_size(), (15, 5))
        for x in range(15):
            for y in range(5):
                if x < 5:
                    self.assertEqual(test_surface.get_at((x, y)), (0, 255, 0))
                else:
                    self.assertEqual(test_surface.get_at((x, y)), (255, 255, 0))
        self.assertEqual(original_surface.get_size(), (20, 20))
        for x in range(20):
            for y in range(20):
                if x < 10 and y < 10:
                    self.assertEqual(original_surface.get_at((x, y)), (255, 0, 0))
                if x < 10 < y:
                    self.assertEqual(original_surface.get_at((x, y)), (0, 255, 0))
                if x > 10 > y:
                    self.assertEqual(original_surface.get_at((x, y)), (0, 0, 255))
                if x > 10 and y > 10:
                    self.assertEqual(original_surface.get_at((x, y)), (255, 255, 0))
        rect = pygame.Rect(0, 0, 10, 10)
        rect.center = original_surface.get_rect().center
        test_surface = pygame.transform.chop(surface=original_surface, rect=rect)
        self.assertEqual(test_surface.get_size(), (10, 10))
        for x in range(10):
            for y in range(10):
                if x < 5 and y < 5:
                    self.assertEqual(test_surface.get_at((x, y)), (255, 0, 0))
                if x < 5 < y:
                    self.assertEqual(test_surface.get_at((x, y)), (0, 255, 0))
                if x > 5 > y:
                    self.assertEqual(test_surface.get_at((x, y)), (0, 0, 255))
                if x > 5 and y > 5:
                    self.assertEqual(test_surface.get_at((x, y)), (255, 255, 0))
        rect = pygame.Rect(10, 10, 0, 0)
        test_surface = pygame.transform.chop(original_surface, rect)
        self.assertEqual(test_surface.get_size(), (20, 20))
        rect = pygame.Rect(0, 0, 20, 20)
        test_surface = pygame.transform.chop(original_surface, rect)
        self.assertEqual(test_surface.get_size(), (0, 0))
        rect = pygame.Rect(5, 15, 20, 20)
        test_surface = pygame.transform.chop(original_surface, rect)
        self.assertEqual(test_surface.get_size(), (5, 15))
        rect = pygame.Rect(400, 400, 10, 10)
        test_surface = pygame.transform.chop(original_surface, rect)
        self.assertEqual(test_surface.get_size(), (20, 20))

    def test_rotozoom(self):
        if False:
            print('Hello World!')
        s = pygame.Surface((10, 0))
        pygame.transform.scale(s, (10, 2))
        s1 = pygame.transform.rotozoom(s, 30, 1)
        s2 = pygame.transform.rotozoom(surface=s, angle=30, scale=1)
        self.assertEqual(s1.get_rect(), pygame.Rect(0, 0, 0, 0))
        self.assertEqual(s2.get_rect(), pygame.Rect(0, 0, 0, 0))

    def test_smoothscale(self):
        if False:
            print('Hello World!')
        'Tests the stated boundaries, sizing, and color blending of smoothscale function'

        def smoothscale_low_bpp():
            if False:
                while True:
                    i = 10
            starting_surface = pygame.Surface((20, 20), depth=12)
            smoothscaled_surface = pygame.transform.smoothscale(starting_surface, (10, 10))
        self.assertRaises(ValueError, smoothscale_low_bpp)

        def smoothscale_high_bpp():
            if False:
                for i in range(10):
                    print('nop')
            starting_surface = pygame.Surface((20, 20), depth=48)
            smoothscaled_surface = pygame.transform.smoothscale(starting_surface, (10, 10))
        self.assertRaises(ValueError, smoothscale_high_bpp)

        def smoothscale_invalid_scale():
            if False:
                while True:
                    i = 10
            starting_surface = pygame.Surface((20, 20), depth=32)
            smoothscaled_surface = pygame.transform.smoothscale(starting_surface, (-1, -1))
        self.assertRaises(ValueError, smoothscale_invalid_scale)
        two_pixel_surface = pygame.Surface((2, 1), depth=32)
        two_pixel_surface.fill(pygame.Color(0, 0, 0), pygame.Rect(0, 0, 1, 1))
        two_pixel_surface.fill(pygame.Color(255, 255, 255), pygame.Rect(1, 0, 1, 1))
        for k in [2 ** x for x in range(5, 8)]:
            bigger_surface = pygame.transform.smoothscale(two_pixel_surface, (k, 1))
            self.assertEqual(bigger_surface.get_at((k // 2, 0)), pygame.Color(127, 127, 127))
            self.assertEqual(bigger_surface.get_size(), (k, 1))
        two_five_six_surf = pygame.Surface((256, 1), depth=32)
        two_five_six_surf.fill(pygame.Color(0, 0, 0), pygame.Rect(0, 0, 128, 1))
        two_five_six_surf.fill(pygame.Color(255, 255, 255), pygame.Rect(128, 0, 128, 1))
        for k in range(3, 11, 2):
            smaller_surface = pygame.transform.smoothscale(two_five_six_surf, (k, 1))
            self.assertEqual(smaller_surface.get_at((k // 2, 0)), pygame.Color(127, 127, 127))
            self.assertEqual(smaller_surface.get_size(), (k, 1))

class TransformDisplayModuleTest(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        pygame.display.init()
        pygame.display.set_mode((320, 200))

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pygame.display.quit()

    def test_flip(self):
        if False:
            print('Hello World!')
        'honors the set_color key on the returned surface from flip.'
        image_loaded = pygame.image.load(example_path('data/chimp.png'))
        image = pygame.Surface(image_loaded.get_size(), 0, 32)
        image.blit(image_loaded, (0, 0))
        image_converted = image_loaded.convert()
        self.assertFalse(image.get_flags() & pygame.SRCALPHA)
        self.assertFalse(image_converted.get_flags() & pygame.SRCALPHA)
        surf = pygame.Surface(image.get_size(), 0, 32)
        surf2 = pygame.Surface(image.get_size(), 0, 32)
        surf.fill((255, 255, 255))
        surf2.fill((255, 255, 255))
        colorkey = image.get_at((0, 0))
        image.set_colorkey(colorkey, RLEACCEL)
        timage = pygame.transform.flip(image, 1, 0)
        colorkey = image_converted.get_at((0, 0))
        image_converted.set_colorkey(colorkey, RLEACCEL)
        timage_converted = pygame.transform.flip(surface=image_converted, flip_x=1, flip_y=0)
        surf.blit(timage, (0, 0))
        surf2.blit(image, (0, 0))
        self.assertEqual(surf.get_at((0, 0)), surf2.get_at((0, 0)))
        self.assertEqual(surf2.get_at((0, 0)), (255, 255, 255, 255))
        surf.fill((255, 255, 255))
        surf2.fill((255, 255, 255))
        surf.blit(timage_converted, (0, 0))
        surf2.blit(image_converted, (0, 0))
        self.assertEqual(surf.get_at((0, 0)), surf2.get_at((0, 0)))

    def test_flip_alpha(self):
        if False:
            print('Hello World!')
        'returns a surface with the same properties as the input.'
        image_loaded = pygame.image.load(example_path('data/chimp.png'))
        image_alpha = pygame.Surface(image_loaded.get_size(), pygame.SRCALPHA, 32)
        image_alpha.blit(image_loaded, (0, 0))
        surf = pygame.Surface(image_loaded.get_size(), 0, 32)
        surf2 = pygame.Surface(image_loaded.get_size(), 0, 32)
        colorkey = image_alpha.get_at((0, 0))
        image_alpha.set_colorkey(colorkey, RLEACCEL)
        timage_alpha = pygame.transform.flip(image_alpha, 1, 0)
        self.assertTrue(image_alpha.get_flags() & pygame.SRCALPHA)
        self.assertTrue(timage_alpha.get_flags() & pygame.SRCALPHA)
        surf.fill((255, 255, 255))
        surf2.fill((255, 255, 255))
        surf.blit(timage_alpha, (0, 0))
        surf2.blit(image_alpha, (0, 0))
        self.assertEqual(surf.get_at((0, 0)), surf2.get_at((0, 0)))
        self.assertEqual(surf2.get_at((0, 0)), (255, 0, 0, 255))
if __name__ == '__main__':
    unittest.main()