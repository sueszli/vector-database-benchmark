import unittest
import os
import sys
import time
import pygame, pygame.transform
from pygame.tests.test_utils import question
from pygame import display

class DisplayModuleTest(unittest.TestCase):
    default_caption = 'pygame window'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        display.init()

    def tearDown(self):
        if False:
            return 10
        display.quit()

    def test_Info(self):
        if False:
            return 10
        inf = pygame.display.Info()
        self.assertNotEqual(inf.current_h, -1)
        self.assertNotEqual(inf.current_w, -1)
        screen = pygame.display.set_mode((128, 128))
        inf = pygame.display.Info()
        self.assertEqual(inf.current_h, 128)
        self.assertEqual(inf.current_w, 128)

    def test_flip(self):
        if False:
            while True:
                i = 10
        screen = pygame.display.set_mode((100, 100))
        self.assertIsNone(pygame.display.flip())
        pygame.Surface.fill(screen, (66, 66, 53))
        self.assertIsNone(pygame.display.flip())
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            pygame.display.flip()
        del screen
        with self.assertRaises(pygame.error):
            pygame.display.flip()

    def test_get_active(self):
        if False:
            i = 10
            return i + 15
        'Test the get_active function'
        pygame.display.quit()
        self.assertEqual(pygame.display.get_active(), False)
        pygame.display.init()
        pygame.display.set_mode((640, 480))
        self.assertEqual(pygame.display.get_active(), True)
        pygame.display.quit()
        pygame.display.init()
        self.assertEqual(pygame.display.get_active(), False)

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'requires the SDL_VIDEODRIVER to be a non dummy value')
    def test_get_active_iconify(self):
        if False:
            return 10
        'Test the get_active function after an iconify'
        pygame.display.set_mode((640, 480))
        pygame.event.clear()
        pygame.display.iconify()
        for _ in range(100):
            time.sleep(0.01)
            pygame.event.pump()
        self.assertEqual(pygame.display.get_active(), False)

    def test_get_caption(self):
        if False:
            return 10
        screen = display.set_mode((100, 100))
        self.assertEqual(display.get_caption()[0], self.default_caption)

    def test_set_caption(self):
        if False:
            while True:
                i = 10
        TEST_CAPTION = 'test'
        screen = display.set_mode((100, 100))
        self.assertIsNone(display.set_caption(TEST_CAPTION))
        self.assertEqual(display.get_caption()[0], TEST_CAPTION)
        self.assertEqual(display.get_caption()[1], TEST_CAPTION)

    def test_set_caption_kwargs(self):
        if False:
            print('Hello World!')
        TEST_CAPTION = 'test'
        screen = display.set_mode((100, 100))
        self.assertIsNone(display.set_caption(title=TEST_CAPTION))
        self.assertEqual(display.get_caption()[0], TEST_CAPTION)
        self.assertEqual(display.get_caption()[1], TEST_CAPTION)

    def test_caption_unicode(self):
        if False:
            print('Hello World!')
        TEST_CAPTION = '台'
        display.set_caption(TEST_CAPTION)
        self.assertEqual(display.get_caption()[0], TEST_CAPTION)

    def test_get_driver(self):
        if False:
            return 10
        drivers = ['aalib', 'android', 'arm', 'cocoa', 'dga', 'directx', 'directfb', 'dummy', 'emscripten', 'fbcon', 'ggi', 'haiku', 'khronos', 'kmsdrm', 'nacl', 'offscreen', 'pandora', 'psp', 'qnx', 'raspberry', 'svgalib', 'uikit', 'vgl', 'vivante', 'wayland', 'windows', 'windib', 'winrt', 'x11']
        driver = display.get_driver()
        self.assertIn(driver, drivers)
        display.quit()
        with self.assertRaises(pygame.error):
            driver = display.get_driver()

    def test_get_init(self):
        if False:
            print('Hello World!')
        "Ensures the module's initialization state can be retrieved."
        self.assertTrue(display.get_init())

    @unittest.skipIf(True, 'SDL2 issues')
    def test_get_surface(self):
        if False:
            i = 10
            return i + 15
        'Ensures get_surface gets the current display surface.'
        lengths = (1, 5, 100)
        for expected_size in ((w, h) for w in lengths for h in lengths):
            for expected_depth in (8, 16, 24, 32):
                expected_surface = display.set_mode(expected_size, 0, expected_depth)
                surface = pygame.display.get_surface()
                self.assertEqual(surface, expected_surface)
                self.assertIsInstance(surface, pygame.Surface)
                self.assertEqual(surface.get_size(), expected_size)
                self.assertEqual(surface.get_bitsize(), expected_depth)

    def test_get_surface__mode_not_set(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures get_surface handles the display mode not being set.'
        surface = pygame.display.get_surface()
        self.assertIsNone(surface)

    def test_get_wm_info(self):
        if False:
            for i in range(10):
                print('nop')
        wm_info = display.get_wm_info()
        self.assertIsInstance(wm_info, dict)
        wm_info_potential_keys = {'colorbuffer', 'connection', 'data', 'dfb', 'display', 'framebuffer', 'fswindow', 'hdc', 'hglrc', 'hinstance', 'lock_func', 'resolveFramebuffer', 'shell_surface', 'surface', 'taskHandle', 'unlock_func', 'wimpVersion', 'window', 'wmwindow'}
        wm_info_remaining_keys = set(wm_info.keys()).difference(wm_info_potential_keys)
        self.assertFalse(wm_info_remaining_keys)

    @unittest.skipIf('skipping for all because some failures on rasppi and maybe other platforms' or os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'OpenGL requires a non-"dummy" SDL_VIDEODRIVER')
    def test_gl_get_attribute(self):
        if False:
            return 10
        screen = display.set_mode((0, 0), pygame.OPENGL)
        original_values = []
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_ALPHA_SIZE))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_DEPTH_SIZE))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_RED_SIZE))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_GREEN_SIZE))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_BLUE_SIZE))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_ALPHA_SIZE))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLEBUFFERS))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLESAMPLES))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_STEREO))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCELERATED_VISUAL))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MAJOR_VERSION))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MINOR_VERSION))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_FLAGS))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_PROFILE_MASK))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT))
        original_values.append(pygame.display.gl_get_attribute(pygame.GL_FRAMEBUFFER_SRGB_CAPABLE))
        pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, 8)
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
        pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, 8)
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_RED_SIZE, 16)
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_GREEN_SIZE, 16)
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_BLUE_SIZE, 16)
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_ALPHA_SIZE, 16)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 1)
        pygame.display.gl_set_attribute(pygame.GL_STEREO, 0)
        pygame.display.gl_set_attribute(pygame.GL_ACCELERATED_VISUAL, 0)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 1)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 1)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_FLAGS, 0)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, 0)
        pygame.display.gl_set_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT, 0)
        pygame.display.gl_set_attribute(pygame.GL_FRAMEBUFFER_SRGB_CAPABLE, 0)
        set_values = [8, 24, 8, 16, 16, 16, 16, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
        get_values = []
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_DEPTH_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_RED_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_GREEN_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_BLUE_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLEBUFFERS))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLESAMPLES))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_STEREO))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCELERATED_VISUAL))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MAJOR_VERSION))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_MINOR_VERSION))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_FLAGS))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_CONTEXT_PROFILE_MASK))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_SHARE_WITH_CURRENT_CONTEXT))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_FRAMEBUFFER_SRGB_CAPABLE))
        for i in range(len(original_values)):
            self.assertTrue(get_values[i] == original_values[i] or get_values[i] == set_values[i])
        with self.assertRaises(TypeError):
            pygame.display.gl_get_attribute('DUMMY')

    @unittest.skipIf('skipping for all because some failures on rasppi and maybe other platforms' or os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'OpenGL requires a non-"dummy" SDL_VIDEODRIVER')
    def test_gl_get_attribute_kwargs(self):
        if False:
            while True:
                i = 10
        screen = display.set_mode((0, 0), pygame.OPENGL)
        original_values = []
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ALPHA_SIZE))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_DEPTH_SIZE))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STENCIL_SIZE))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_RED_SIZE))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_GREEN_SIZE))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_BLUE_SIZE))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_ALPHA_SIZE))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLEBUFFERS))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLESAMPLES))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STEREO))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCELERATED_VISUAL))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_MAJOR_VERSION))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_MINOR_VERSION))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_FLAGS))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_PROFILE_MASK))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_SHARE_WITH_CURRENT_CONTEXT))
        original_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_FRAMEBUFFER_SRGB_CAPABLE))
        pygame.display.gl_set_attribute(flag=pygame.GL_ALPHA_SIZE, value=8)
        pygame.display.gl_set_attribute(flag=pygame.GL_DEPTH_SIZE, value=24)
        pygame.display.gl_set_attribute(flag=pygame.GL_STENCIL_SIZE, value=8)
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_RED_SIZE, value=16)
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_GREEN_SIZE, value=16)
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_BLUE_SIZE, value=16)
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_ALPHA_SIZE, value=16)
        pygame.display.gl_set_attribute(flag=pygame.GL_MULTISAMPLEBUFFERS, value=1)
        pygame.display.gl_set_attribute(flag=pygame.GL_MULTISAMPLESAMPLES, value=1)
        pygame.display.gl_set_attribute(flag=pygame.GL_STEREO, value=0)
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCELERATED_VISUAL, value=0)
        pygame.display.gl_set_attribute(flag=pygame.GL_CONTEXT_MAJOR_VERSION, value=1)
        pygame.display.gl_set_attribute(flag=pygame.GL_CONTEXT_MINOR_VERSION, value=1)
        pygame.display.gl_set_attribute(flag=pygame.GL_CONTEXT_FLAGS, value=0)
        pygame.display.gl_set_attribute(flag=pygame.GL_CONTEXT_PROFILE_MASK, value=0)
        pygame.display.gl_set_attribute(flag=pygame.GL_SHARE_WITH_CURRENT_CONTEXT, value=0)
        pygame.display.gl_set_attribute(flag=pygame.GL_FRAMEBUFFER_SRGB_CAPABLE, value=0)
        set_values = [8, 24, 8, 16, 16, 16, 16, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0]
        get_values = []
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_DEPTH_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STENCIL_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_RED_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_GREEN_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_BLUE_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLEBUFFERS))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLESAMPLES))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STEREO))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCELERATED_VISUAL))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_MAJOR_VERSION))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_MINOR_VERSION))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_FLAGS))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_CONTEXT_PROFILE_MASK))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_SHARE_WITH_CURRENT_CONTEXT))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_FRAMEBUFFER_SRGB_CAPABLE))
        for i in range(len(original_values)):
            self.assertTrue(get_values[i] == original_values[i] or get_values[i] == set_values[i])
        with self.assertRaises(TypeError):
            pygame.display.gl_get_attribute('DUMMY')

    @unittest.skipIf('skipping for all because some failures on rasppi and maybe other platforms' or os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'OpenGL requires a non-"dummy" SDL_VIDEODRIVER')
    def test_gl_set_attribute(self):
        if False:
            while True:
                i = 10
        screen = display.set_mode((0, 0), pygame.OPENGL)
        set_values = [8, 24, 8, 16, 16, 16, 16, 1, 1, 0]
        pygame.display.gl_set_attribute(pygame.GL_ALPHA_SIZE, set_values[0])
        pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, set_values[1])
        pygame.display.gl_set_attribute(pygame.GL_STENCIL_SIZE, set_values[2])
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_RED_SIZE, set_values[3])
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_GREEN_SIZE, set_values[4])
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_BLUE_SIZE, set_values[5])
        pygame.display.gl_set_attribute(pygame.GL_ACCUM_ALPHA_SIZE, set_values[6])
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, set_values[7])
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, set_values[8])
        pygame.display.gl_set_attribute(pygame.GL_STEREO, set_values[9])
        get_values = []
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_DEPTH_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_STENCIL_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_RED_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_GREEN_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_BLUE_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_ACCUM_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLEBUFFERS))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_MULTISAMPLESAMPLES))
        get_values.append(pygame.display.gl_get_attribute(pygame.GL_STEREO))
        for i in range(len(set_values)):
            self.assertTrue(get_values[i] == set_values[i])
        with self.assertRaises(TypeError):
            pygame.display.gl_get_attribute('DUMMY')

    @unittest.skipIf('skipping for all because some failures on rasppi and maybe other platforms' or os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'OpenGL requires a non-"dummy" SDL_VIDEODRIVER')
    def test_gl_set_attribute_kwargs(self):
        if False:
            return 10
        screen = display.set_mode((0, 0), pygame.OPENGL)
        set_values = [8, 24, 8, 16, 16, 16, 16, 1, 1, 0]
        pygame.display.gl_set_attribute(flag=pygame.GL_ALPHA_SIZE, value=set_values[0])
        pygame.display.gl_set_attribute(flag=pygame.GL_DEPTH_SIZE, value=set_values[1])
        pygame.display.gl_set_attribute(flag=pygame.GL_STENCIL_SIZE, value=set_values[2])
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_RED_SIZE, value=set_values[3])
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_GREEN_SIZE, value=set_values[4])
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_BLUE_SIZE, value=set_values[5])
        pygame.display.gl_set_attribute(flag=pygame.GL_ACCUM_ALPHA_SIZE, value=set_values[6])
        pygame.display.gl_set_attribute(flag=pygame.GL_MULTISAMPLEBUFFERS, value=set_values[7])
        pygame.display.gl_set_attribute(flag=pygame.GL_MULTISAMPLESAMPLES, value=set_values[8])
        pygame.display.gl_set_attribute(flag=pygame.GL_STEREO, value=set_values[9])
        get_values = []
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_DEPTH_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STENCIL_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_RED_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_GREEN_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_BLUE_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_ACCUM_ALPHA_SIZE))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLEBUFFERS))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_MULTISAMPLESAMPLES))
        get_values.append(pygame.display.gl_get_attribute(flag=pygame.GL_STEREO))
        for i in range(len(set_values)):
            self.assertTrue(get_values[i] == set_values[i])
        with self.assertRaises(TypeError):
            pygame.display.gl_get_attribute('DUMMY')

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') in ['dummy', 'android'], 'iconify is only supported on some video drivers/platforms')
    def test_iconify(self):
        if False:
            while True:
                i = 10
        pygame.display.set_mode((640, 480))
        self.assertEqual(pygame.display.get_active(), True)
        success = pygame.display.iconify()
        if success:
            active_event = window_minimized_event = False
            for _ in range(50):
                time.sleep(0.01)
                for event in pygame.event.get():
                    if event.type == pygame.ACTIVEEVENT:
                        if not event.gain and event.state == pygame.APPACTIVE:
                            active_event = True
                    if event.type == pygame.WINDOWMINIMIZED:
                        window_minimized_event = True
            self.assertTrue(window_minimized_event)
            self.assertTrue(active_event)
            self.assertFalse(pygame.display.get_active())
        else:
            self.fail('Iconify not supported on this platform, please skip')

    def test_init(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensures the module is initialized after init called.'
        display.quit()
        display.init()
        self.assertTrue(display.get_init())

    def test_init__multiple(self):
        if False:
            i = 10
            return i + 15
        'Ensures the module is initialized after multiple init calls.'
        display.init()
        display.init()
        self.assertTrue(display.get_init())

    def test_list_modes(self):
        if False:
            return 10
        modes = pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN, display=0)
        if modes != -1:
            self.assertEqual(len(modes[0]), 2)
            self.assertEqual(type(modes[0][0]), int)
        modes = pygame.display.list_modes()
        if modes != -1:
            self.assertEqual(len(modes[0]), 2)
            self.assertEqual(type(modes[0][0]), int)
            self.assertEqual(len(modes), len(set(modes)))
        modes = pygame.display.list_modes(depth=0, flags=0, display=0)
        if modes != -1:
            self.assertEqual(len(modes[0]), 2)
            self.assertEqual(type(modes[0][0]), int)

    def test_mode_ok(self):
        if False:
            i = 10
            return i + 15
        pygame.display.mode_ok((128, 128))
        modes = pygame.display.list_modes()
        if modes != -1:
            size = modes[0]
            self.assertNotEqual(pygame.display.mode_ok(size), 0)
        pygame.display.mode_ok((128, 128), 0, 32)
        pygame.display.mode_ok((128, 128), flags=0, depth=32, display=0)

    def test_mode_ok_fullscreen(self):
        if False:
            return 10
        modes = pygame.display.list_modes()
        if modes != -1:
            size = modes[0]
            self.assertNotEqual(pygame.display.mode_ok(size, flags=pygame.FULLSCREEN), 0)

    def test_mode_ok_scaled(self):
        if False:
            while True:
                i = 10
        modes = pygame.display.list_modes()
        if modes != -1:
            size = modes[0]
            self.assertNotEqual(pygame.display.mode_ok(size, flags=pygame.SCALED), 0)

    def test_get_num_displays(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertGreater(pygame.display.get_num_displays(), 0)

    def test_quit(self):
        if False:
            while True:
                i = 10
        'Ensures the module is not initialized after quit called.'
        display.quit()
        self.assertFalse(display.get_init())

    def test_quit__multiple(self):
        if False:
            i = 10
            return i + 15
        'Ensures the module is not initialized after multiple quit calls.'
        display.quit()
        display.quit()
        self.assertFalse(display.get_init())

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'Needs a not dummy videodriver')
    def test_set_gamma(self):
        if False:
            print('Hello World!')
        pygame.display.set_mode((1, 1))
        gammas = [0.25, 0.5, 0.88, 1.0]
        for gamma in gammas:
            with self.subTest(gamma=gamma):
                with self.assertWarns(DeprecationWarning):
                    self.assertEqual(pygame.display.set_gamma(gamma), True)
                self.assertEqual(pygame.display.set_gamma(gamma), True)

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'Needs a not dummy videodriver')
    def test_set_gamma__tuple(self):
        if False:
            i = 10
            return i + 15
        pygame.display.set_mode((1, 1))
        gammas = [(0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (0.25, 0.33, 0.44)]
        for (r, g, b) in gammas:
            with self.subTest(r=r, g=g, b=b):
                self.assertEqual(pygame.display.set_gamma(r, g, b), True)

    @unittest.skipIf(not hasattr(pygame.display, 'set_gamma_ramp'), 'Not all systems and hardware support gamma ramps')
    def test_set_gamma_ramp(self):
        if False:
            i = 10
            return i + 15
        pygame.display.set_mode((5, 5))
        r = list(range(256))
        g = [number + 256 for number in r]
        b = [number + 256 for number in g]
        with self.assertWarns(DeprecationWarning):
            isSupported = pygame.display.set_gamma_ramp(r, g, b)
        if isSupported:
            self.assertTrue(pygame.display.set_gamma_ramp(r, g, b))
        else:
            self.assertFalse(pygame.display.set_gamma_ramp(r, g, b))

    def test_set_mode_kwargs(self):
        if False:
            while True:
                i = 10
        pygame.display.set_mode(size=(1, 1), flags=0, depth=0, display=0)

    def test_set_mode_scaled(self):
        if False:
            i = 10
            return i + 15
        surf = pygame.display.set_mode(size=(1, 1), flags=pygame.SCALED, depth=0, display=0)
        winsize = pygame.display.get_window_size()
        self.assertEqual(winsize[0] % surf.get_size()[0], 0, 'window width should be a multiple of the surface width')
        self.assertEqual(winsize[1] % surf.get_size()[1], 0, 'window height should be a multiple of the surface height')
        self.assertEqual(winsize[0] / surf.get_size()[0], winsize[1] / surf.get_size()[1])

    def test_set_mode_vector2(self):
        if False:
            print('Hello World!')
        pygame.display.set_mode(pygame.Vector2(1, 1))

    def test_set_mode_unscaled(self):
        if False:
            print('Hello World!')
        'Ensures a window created with SCALED can become smaller.'
        screen = pygame.display.set_mode((300, 300), pygame.SCALED)
        self.assertEqual(screen.get_size(), (300, 300))
        screen = pygame.display.set_mode((200, 200))
        self.assertEqual(screen.get_size(), (200, 200))

    def test_screensaver_support(self):
        if False:
            i = 10
            return i + 15
        pygame.display.set_allow_screensaver(True)
        self.assertTrue(pygame.display.get_allow_screensaver())
        pygame.display.set_allow_screensaver(False)
        self.assertFalse(pygame.display.get_allow_screensaver())
        pygame.display.set_allow_screensaver()
        self.assertTrue(pygame.display.get_allow_screensaver())

    @unittest.skipIf(True, 'set_palette() not supported in SDL2')
    def test_set_palette(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises(pygame.error):
            palette = [1, 2, 3]
            pygame.display.set_palette(palette)
        pygame.display.set_mode((1024, 768), 0, 8)
        palette = []
        self.assertIsNone(pygame.display.set_palette(palette))
        with self.assertRaises(ValueError):
            palette = 12
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [[1, 2], [1, 2]]
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [[0, 0, 0, 0, 0]] + [[x, x, x, x, x] for x in range(1, 255)]
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = 'qwerty'
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [[123, 123, 123] * 10000]
            pygame.display.set_palette(palette)
        with self.assertRaises(TypeError):
            palette = [1, 2, 3]
            pygame.display.set_palette(palette)
    skip_list = ['dummy', 'android']

    @unittest.skipIf(True, 'set_palette() not supported in SDL2')
    def test_set_palette_kwargs(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(pygame.error):
            palette = [1, 2, 3]
            pygame.display.set_palette(palette=palette)
        pygame.display.set_mode((1024, 768), 0, 8)
        palette = []
        self.assertIsNone(pygame.display.set_palette(palette=palette))
        with self.assertRaises(ValueError):
            palette = 12
            pygame.display.set_palette(palette=palette)
        with self.assertRaises(TypeError):
            palette = [[1, 2], [1, 2]]
            pygame.display.set_palette(palette=palette)
        with self.assertRaises(TypeError):
            palette = [[0, 0, 0, 0, 0]] + [[x, x, x, x, x] for x in range(1, 255)]
            pygame.display.set_palette(palette=palette)
        with self.assertRaises(TypeError):
            palette = 'qwerty'
            pygame.display.set_palette(palette=palette)
        with self.assertRaises(TypeError):
            palette = [[123, 123, 123] * 10000]
            pygame.display.set_palette(palette=palette)
        with self.assertRaises(TypeError):
            palette = [1, 2, 3]
            pygame.display.set_palette(palette=palette)
    skip_list = ['dummy', 'android']

    @unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') in skip_list, 'requires the SDL_VIDEODRIVER to be non dummy')
    def test_toggle_fullscreen(self):
        if False:
            while True:
                i = 10
        'Test for toggle fullscreen'
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            pygame.display.toggle_fullscreen()
        pygame.display.init()
        width_height = (640, 480)
        test_surf = pygame.display.set_mode(width_height)
        try:
            pygame.display.toggle_fullscreen()
        except pygame.error:
            self.fail()
        else:
            if pygame.display.toggle_fullscreen() == 1:
                boolean = (test_surf.get_width(), test_surf.get_height()) in pygame.display.list_modes(depth=0, flags=pygame.FULLSCREEN, display=0)
                self.assertEqual(boolean, True)
            else:
                self.assertEqual((test_surf.get_width(), test_surf.get_height()), width_height)

class DisplayUpdateTest(unittest.TestCase):

    def question(self, qstr):
        if False:
            return 10
        'this is used in the interactive subclass.'

    def setUp(self):
        if False:
            print('Hello World!')
        display.init()
        self.screen = pygame.display.set_mode((500, 500))
        self.screen.fill('black')
        pygame.display.flip()
        pygame.event.pump()

    def tearDown(self):
        if False:
            return 10
        display.quit()

    def test_update_negative(self):
        if False:
            print('Hello World!')
        'takes rects with negative values.'
        self.screen.fill('green')
        r1 = pygame.Rect(0, 0, 100, 100)
        pygame.display.update(r1)
        r2 = pygame.Rect(-10, 0, 100, 100)
        pygame.display.update(r2)
        r3 = pygame.Rect(-10, 0, -100, -100)
        pygame.display.update(r3)
        self.question('Is the screen green in (0, 0, 100, 100)?')

    def test_update_sequence(self):
        if False:
            for i in range(10):
                print('nop')
        'only updates the part of the display given by the rects.'
        self.screen.fill('green')
        rects = [pygame.Rect(0, 0, 100, 100), pygame.Rect(100, 0, 100, 100), pygame.Rect(200, 0, 100, 100), pygame.Rect(300, 300, 100, 100)]
        pygame.display.update(rects)
        pygame.event.pump()
        self.question(f'Is the screen green in {rects}?')

    def test_update_none_skipped(self):
        if False:
            for i in range(10):
                print('nop')
        'None is skipped inside sequences.'
        self.screen.fill('green')
        rects = (None, pygame.Rect(100, 0, 100, 100), None, pygame.Rect(200, 0, 100, 100), pygame.Rect(300, 300, 100, 100))
        pygame.display.update(rects)
        pygame.event.pump()
        self.question(f'Is the screen green in {rects}?')

    def test_update_none(self):
        if False:
            return 10
        'does NOT update the display.'
        self.screen.fill('green')
        pygame.display.update(None)
        pygame.event.pump()
        self.question(f'Is the screen black and NOT green?')

    def test_update_no_args(self):
        if False:
            print('Hello World!')
        'does NOT update the display.'
        self.screen.fill('green')
        pygame.display.update()
        pygame.event.pump()
        self.question(f'Is the WHOLE screen green?')

    def test_update_args(self):
        if False:
            return 10
        'updates the display using the args as a rect.'
        self.screen.fill('green')
        pygame.display.update(100, 100, 100, 100)
        pygame.event.pump()
        self.question('Is the screen green in (100, 100, 100, 100)?')

    def test_update_incorrect_args(self):
        if False:
            return 10
        'raises a ValueError when inputs are wrong.'
        with self.assertRaises(ValueError):
            pygame.display.update(100, 'asdf', 100, 100)
        with self.assertRaises(ValueError):
            pygame.display.update([100, 'asdf', 100, 100])

    def test_update_no_init(self):
        if False:
            i = 10
            return i + 15
        'raises a pygame.error.'
        pygame.display.quit()
        with self.assertRaises(pygame.error):
            pygame.display.update()

class DisplayUpdateInteractiveTest(DisplayUpdateTest):
    """Because we want these tests to run as interactive and not interactive."""
    __tags__ = ['interactive']

    def question(self, qstr):
        if False:
            print('Hello World!')
        'since this is the interactive sublcass we ask a question.'
        question(qstr)

class DisplayInteractiveTest(unittest.TestCase):
    __tags__ = ['interactive']

    def test_set_icon_interactive(self):
        if False:
            print('Hello World!')
        os.environ['SDL_VIDEO_WINDOW_POS'] = '100,250'
        pygame.display.quit()
        pygame.display.init()
        test_icon = pygame.Surface((32, 32))
        test_icon.fill((255, 0, 0))
        pygame.display.set_icon(test_icon)
        screen = pygame.display.set_mode((400, 100))
        pygame.display.set_caption('Is the window icon a red square?')
        response = question('Is the display icon red square?')
        self.assertTrue(response)
        pygame.display.quit()

    def test_set_gamma_ramp(self):
        if False:
            print('Hello World!')
        os.environ['SDL_VIDEO_WINDOW_POS'] = '100,250'
        pygame.display.quit()
        pygame.display.init()
        screen = pygame.display.set_mode((400, 100))
        screen.fill((100, 100, 100))
        blue_ramp = [x * 256 for x in range(0, 256)]
        blue_ramp[100] = 150 * 256
        normal_ramp = [x * 256 for x in range(0, 256)]
        gamma_success = False
        if pygame.display.set_gamma_ramp(normal_ramp, normal_ramp, blue_ramp):
            pygame.display.update()
            gamma_success = True
        if gamma_success:
            response = question('Is the window background tinted blue?')
            self.assertTrue(response)
            pygame.display.set_gamma_ramp(normal_ramp, normal_ramp, normal_ramp)
        pygame.display.quit()

class FullscreenToggleTests(unittest.TestCase):
    __tags__ = ['interactive']
    screen = None
    font = None
    isfullscreen = False
    WIDTH = 800
    HEIGHT = 600

    def setUp(self):
        if False:
            return 10
        pygame.init()
        if sys.platform == 'win32':
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), flags=pygame.SCALED)
        else:
            self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Fullscreen Tests')
        self.screen.fill((255, 255, 255))
        pygame.display.flip()
        self.font = pygame.font.Font(None, 32)

    def tearDown(self):
        if False:
            return 10
        if self.isfullscreen:
            pygame.display.toggle_fullscreen()
        pygame.quit()

    def visual_test(self, fullscreen=False):
        if False:
            i = 10
            return i + 15
        text = ''
        if fullscreen:
            if not self.isfullscreen:
                pygame.display.toggle_fullscreen()
                self.isfullscreen = True
            text = 'Is this in fullscreen? [y/n]'
        else:
            if self.isfullscreen:
                pygame.display.toggle_fullscreen()
                self.isfullscreen = False
            text = 'Is this not in fullscreen [y/n]'
        s = self.font.render(text, False, (0, 0, 0))
        self.screen.blit(s, (self.WIDTH / 2 - self.font.size(text)[0] / 2, 100))
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        return False
                    if event.key == pygame.K_y:
                        return True
                    if event.key == pygame.K_n:
                        return False
                if event.type == pygame.QUIT:
                    return False

    def test_fullscreen_true(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(self.visual_test(fullscreen=True))

    def test_fullscreen_false(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.visual_test(fullscreen=False))

@unittest.skipIf(os.environ.get('SDL_VIDEODRIVER') == 'dummy', 'OpenGL requires a non-"dummy" SDL_VIDEODRIVER')
class DisplayOpenGLTest(unittest.TestCase):

    def test_screen_size_opengl(self):
        if False:
            return 10
        'returns a surface with the same size requested.\n        |tags:display,slow,opengl|\n        '
        pygame.display.init()
        screen = pygame.display.set_mode((640, 480), pygame.OPENGL)
        self.assertEqual((640, 480), screen.get_size())

class X11CrashTest(unittest.TestCase):

    def test_x11_set_mode_crash_gh1654(self):
        if False:
            i = 10
            return i + 15
        pygame.display.init()
        pygame.display.quit()
        screen = pygame.display.set_mode((640, 480), 0)
        self.assertEqual((640, 480), screen.get_size())
if __name__ == '__main__':
    unittest.main()