import os
import os.path
import sys
import unittest
from pygame.tests.test_utils import example_path
import pygame, pygame.image, pygame.pkgdata
imageext = sys.modules['pygame.imageext']

class ImageextModuleTest(unittest.TestCase):

    def test_save_non_string_file(self):
        if False:
            for i in range(10):
                print('nop')
        im = pygame.Surface((10, 10), 0, 32)
        self.assertRaises(TypeError, imageext.save_extended, im, [])

    def test_load_non_string_file(self):
        if False:
            return 10
        self.assertRaises(TypeError, imageext.load_extended, [])

    @unittest.skip('SDL silently removes invalid characters')
    def test_save_bad_filename(self):
        if False:
            for i in range(10):
                print('nop')
        im = pygame.Surface((10, 10), 0, 32)
        u = 'a\x00b\x00c.png'
        self.assertRaises(pygame.error, imageext.save_extended, im, u)

    @unittest.skip('SDL silently removes invalid characters')
    def test_load_bad_filename(self):
        if False:
            print('Hello World!')
        u = 'a\x00b\x00c.png'
        self.assertRaises(pygame.error, imageext.load_extended, u)

    def test_save_unknown_extension(self):
        if False:
            return 10
        im = pygame.Surface((10, 10), 0, 32)
        s = 'foo.bar'
        self.assertRaises(pygame.error, imageext.save_extended, im, s)

    def test_load_unknown_extension(self):
        if False:
            for i in range(10):
                print('nop')
        s = 'foo.bar'
        self.assertRaises(FileNotFoundError, imageext.load_extended, s)

    def test_load_unknown_file(self):
        if False:
            print('Hello World!')
        s = 'nonexistent.png'
        self.assertRaises(FileNotFoundError, imageext.load_extended, s)

    def test_load_unicode_path_0(self):
        if False:
            for i in range(10):
                print('nop')
        u = example_path('data/alien1.png')
        im = imageext.load_extended(u)

    def test_load_unicode_path_1(self):
        if False:
            for i in range(10):
                print('nop')
        'non-ASCII unicode'
        import shutil
        orig = example_path('data/alien1.png')
        temp = os.path.join(example_path('data'), '你好.png')
        shutil.copy(orig, temp)
        try:
            im = imageext.load_extended(temp)
        finally:
            os.remove(temp)

    def _unicode_save(self, temp_file):
        if False:
            i = 10
            return i + 15
        im = pygame.Surface((10, 10), 0, 32)
        try:
            with open(temp_file, 'w') as f:
                pass
            os.remove(temp_file)
        except OSError:
            raise unittest.SkipTest('the path cannot be opened')
        self.assertFalse(os.path.exists(temp_file))
        try:
            imageext.save_extended(im, temp_file)
            self.assertGreater(os.path.getsize(temp_file), 10)
        finally:
            try:
                os.remove(temp_file)
            except OSError:
                pass

    def test_save_unicode_path_0(self):
        if False:
            for i in range(10):
                print('nop')
        'unicode object with ASCII chars'
        self._unicode_save('temp_file.png')

    def test_save_unicode_path_1(self):
        if False:
            while True:
                i = 10
        self._unicode_save('你好.png')
if __name__ == '__main__':
    unittest.main()