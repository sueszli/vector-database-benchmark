import unittest
import os
import platform
import warnings
import pygame
DARWIN = 'Darwin' in platform.platform()

class MouseTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            while True:
                i = 10
        pygame.display.init()

    @classmethod
    def tearDownClass(cls):
        if False:
            for i in range(10):
                print('nop')
        pygame.display.quit()

class MouseModuleInteractiveTest(MouseTests):
    __tags__ = ['interactive']

    def test_set_pos(self):
        if False:
            while True:
                i = 10
        'Ensures set_pos works correctly.\n        Requires tester to move the mouse to be on the window.\n        '
        pygame.display.set_mode((500, 500))
        pygame.event.get()
        if not pygame.mouse.get_focused():
            return
        clock = pygame.time.Clock()
        expected_pos = ((10, 0), (0, 0), (499, 0), (499, 499), (341, 143), (94, 49))
        for (x, y) in expected_pos:
            pygame.mouse.set_pos(x, y)
            pygame.event.get()
            found_pos = pygame.mouse.get_pos()
            clock.tick()
            time_passed = 0.0
            ready_to_test = False
            while not ready_to_test and time_passed <= 1000.0:
                time_passed += clock.tick()
                for event in pygame.event.get():
                    if event.type == pygame.MOUSEMOTION:
                        ready_to_test = True
            self.assertEqual(found_pos, (x, y))

class MouseModuleTest(MouseTests):

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER', '') == 'dummy', 'Cursors not supported on headless test machines')
    def test_get_cursor(self):
        if False:
            while True:
                i = 10
        'Ensures get_cursor works correctly.'
        with self.assertRaises(pygame.error):
            pygame.display.quit()
            pygame.mouse.get_cursor()
        pygame.display.init()
        size = (8, 8)
        hotspot = (0, 0)
        xormask = (0, 96, 120, 126, 112, 96, 0, 0)
        andmask = (224, 240, 254, 255, 254, 240, 96, 0)
        expected_length = 4
        expected_cursor = pygame.cursors.Cursor(size, hotspot, xormask, andmask)
        pygame.mouse.set_cursor(expected_cursor)
        try:
            cursor = pygame.mouse.get_cursor()
            self.assertIsInstance(cursor, pygame.cursors.Cursor)
            self.assertEqual(len(cursor), expected_length)
            for info in cursor:
                self.assertIsInstance(info, tuple)
            pygame.mouse.set_cursor(size, hotspot, xormask, andmask)
            self.assertEqual(pygame.mouse.get_cursor(), expected_cursor)
        except pygame.error:
            with self.assertRaises(pygame.error):
                pygame.mouse.get_cursor()

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER', '') == 'dummy', 'mouse.set_system_cursor only available in SDL2')
    def test_set_system_cursor(self):
        if False:
            return 10
        'Ensures set_system_cursor works correctly.'
        with warnings.catch_warnings(record=True) as w:
            'From Pygame 2.0.1, set_system_cursor() should raise a deprecation warning'
            warnings.simplefilter('always')
            with self.assertRaises(pygame.error):
                pygame.display.quit()
                pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_HAND)
            pygame.display.init()
            with self.assertRaises(TypeError):
                pygame.mouse.set_system_cursor('b')
            with self.assertRaises(TypeError):
                pygame.mouse.set_system_cursor(None)
            with self.assertRaises(TypeError):
                pygame.mouse.set_system_cursor((8, 8), (0, 0))
            with self.assertRaises(pygame.error):
                pygame.mouse.set_system_cursor(2000)
            self.assertEqual(pygame.mouse.set_system_cursor(pygame.SYSTEM_CURSOR_ARROW), None)
            self.assertEqual(len(w), 6)
            self.assertTrue(all([issubclass(warn.category, DeprecationWarning) for warn in w]))

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER', '') == 'dummy', 'Cursors not supported on headless test machines')
    def test_set_cursor(self):
        if False:
            i = 10
            return i + 15
        'Ensures set_cursor works correctly.'
        size = (8, 8)
        hotspot = (0, 0)
        xormask = (0, 126, 64, 64, 32, 16, 0, 0)
        andmask = (254, 255, 254, 112, 56, 28, 12, 0)
        bitmap_cursor = pygame.cursors.Cursor(size, hotspot, xormask, andmask)
        constant = pygame.SYSTEM_CURSOR_ARROW
        system_cursor = pygame.cursors.Cursor(constant)
        surface = pygame.Surface((10, 10))
        color_cursor = pygame.cursors.Cursor(hotspot, surface)
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            pygame.mouse.set_cursor(bitmap_cursor)
        with self.assertRaises(pygame.error):
            pygame.mouse.set_cursor(system_cursor)
        with self.assertRaises(pygame.error):
            pygame.mouse.set_cursor(color_cursor)
        pygame.display.init()
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(('w', 'h'), hotspot, xormask, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, ('0', '0'), xormask, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, ('x', 'y', 'z'), xormask, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, 12345678, andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, xormask, 12345678)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, '00000000', andmask)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(size, hotspot, xormask, (2, [0], 4, 0, 0, 8, 0, 1))
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((3, 8), hotspot, xormask, andmask)
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((16, 2), hotspot, (128, 64, 32), andmask)
        with self.assertRaises(ValueError):
            pygame.mouse.set_cursor((16, 2), hotspot, xormask, (192, 96, 48, 0, 1))
        self.assertEqual(pygame.mouse.set_cursor((16, 1), hotspot, (8, 0), (0, 192)), None)
        pygame.mouse.set_cursor(size, hotspot, xormask, andmask)
        self.assertEqual(pygame.mouse.get_cursor(), bitmap_cursor)
        pygame.mouse.set_cursor(size, hotspot, list(xormask), list(andmask))
        self.assertEqual(pygame.mouse.get_cursor(), bitmap_cursor)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(-50021232)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor('yellow')
        self.assertEqual(pygame.mouse.set_cursor(constant), None)
        pygame.mouse.set_cursor(constant)
        self.assertEqual(pygame.mouse.get_cursor(), system_cursor)
        pygame.mouse.set_cursor(system_cursor)
        self.assertEqual(pygame.mouse.get_cursor(), system_cursor)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(('x', 'y'), surface)
        with self.assertRaises(TypeError):
            pygame.mouse.set_cursor(hotspot, 'not_a_surface')
        self.assertEqual(pygame.mouse.set_cursor(hotspot, surface), None)
        pygame.mouse.set_cursor(hotspot, surface)
        self.assertEqual(pygame.mouse.get_cursor(), color_cursor)
        pygame.mouse.set_cursor(color_cursor)
        self.assertEqual(pygame.mouse.get_cursor(), color_cursor)
        pygame.mouse.set_cursor((0, 0), pygame.Surface((20, 20)))
        cursor = pygame.mouse.get_cursor()
        self.assertEqual(cursor.type, 'color')
        self.assertEqual(cursor.data[0], (0, 0))
        self.assertEqual(cursor.data[1].get_size(), (20, 20))

    def test_get_focused(self):
        if False:
            i = 10
            return i + 15
        'Ensures get_focused returns the correct type.'
        focused = pygame.mouse.get_focused()
        self.assertIsInstance(focused, int)

    def test_get_pressed(self):
        if False:
            return 10
        'Ensures get_pressed returns the correct types.'
        expected_length = 3
        buttons_pressed = pygame.mouse.get_pressed()
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)
        expected_length = 5
        buttons_pressed = pygame.mouse.get_pressed(num_buttons=5)
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)
        expected_length = 3
        buttons_pressed = pygame.mouse.get_pressed(3)
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)
        expected_length = 5
        buttons_pressed = pygame.mouse.get_pressed(5)
        self.assertIsInstance(buttons_pressed, tuple)
        self.assertEqual(len(buttons_pressed), expected_length)
        for value in buttons_pressed:
            self.assertIsInstance(value, bool)
        with self.assertRaises(ValueError):
            pygame.mouse.get_pressed(4)

    def test_get_pos(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures get_pos returns the correct types.'
        expected_length = 2
        pos = pygame.mouse.get_pos()
        self.assertIsInstance(pos, tuple)
        self.assertEqual(len(pos), expected_length)
        for value in pos:
            self.assertIsInstance(value, int)

    def test_set_pos__invalid_pos(self):
        if False:
            i = 10
            return i + 15
        'Ensures set_pos handles invalid positions correctly.'
        for invalid_pos in ((1,), [1, 2, 3], 1, '1', (1, '1'), []):
            with self.assertRaises(TypeError):
                pygame.mouse.set_pos(invalid_pos)

    def test_get_rel(self):
        if False:
            i = 10
            return i + 15
        'Ensures get_rel returns the correct types.'
        expected_length = 2
        rel = pygame.mouse.get_rel()
        self.assertIsInstance(rel, tuple)
        self.assertEqual(len(rel), expected_length)
        for value in rel:
            self.assertIsInstance(value, int)

    def test_get_visible(self):
        if False:
            i = 10
            return i + 15
        'Ensures get_visible works correctly.'
        for expected_value in (False, True):
            pygame.mouse.set_visible(expected_value)
            visible = pygame.mouse.get_visible()
            self.assertEqual(visible, expected_value)

    def test_set_visible(self):
        if False:
            print('Hello World!')
        'Ensures set_visible returns the correct values.'
        pygame.mouse.set_visible(True)
        for expected_visible in (False, True):
            prev_visible = pygame.mouse.set_visible(expected_visible)
            self.assertEqual(prev_visible, not expected_visible)

    def test_set_visible__invalid_value(self):
        if False:
            print('Hello World!')
        'Ensures set_visible handles invalid positions correctly.'
        for invalid_value in ((1,), [1, 2, 3], 1.1, '1', (1, '1'), []):
            with self.assertRaises(TypeError):
                prev_visible = pygame.mouse.set_visible(invalid_value)
if __name__ == '__main__':
    unittest.main()