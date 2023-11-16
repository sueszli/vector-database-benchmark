import unittest
from pygame.tests.test_utils import fixture_path
import pygame

class CursorsModuleTest(unittest.TestCase):

    def test_compile(self):
        if False:
            print('Hello World!')
        test_cursor1 = ('X.X.XXXX', 'XXXXXX..', '  XXXX  ')
        test_cursor2 = ('X.X.XXXX', 'XXXXXX..', 'XXXXXX ', 'XXXXXX..', 'XXXXXX..', 'XXXXXX', 'XXXXXX..', 'XXXXXX..')
        test_cursor3 = ('.XX.', '  ', '..  ', 'X.. X')
        with self.assertRaises(ValueError):
            pygame.cursors.compile(test_cursor1)
        with self.assertRaises(ValueError):
            pygame.cursors.compile(test_cursor2)
        with self.assertRaises(ValueError):
            pygame.cursors.compile(test_cursor3)
        actual_byte_data = ((192, 0, 0, 224, 0, 0, 240, 0, 0, 216, 0, 0, 204, 0, 0, 198, 0, 0, 195, 0, 0, 193, 128, 0, 192, 192, 0, 192, 96, 0, 192, 48, 0, 192, 56, 0, 192, 248, 0, 220, 192, 0, 246, 96, 0, 198, 96, 0, 6, 96, 0, 3, 48, 0, 3, 48, 0, 1, 224, 0, 1, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0), (192, 0, 0, 224, 0, 0, 240, 0, 0, 248, 0, 0, 252, 0, 0, 254, 0, 0, 255, 0, 0, 255, 128, 0, 255, 192, 0, 255, 224, 0, 255, 240, 0, 255, 248, 0, 255, 248, 0, 255, 192, 0, 247, 224, 0, 199, 224, 0, 7, 224, 0, 3, 240, 0, 3, 240, 0, 1, 224, 0, 1, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
        cursor = pygame.cursors.compile(pygame.cursors.thickarrow_strings)
        self.assertEqual(cursor, actual_byte_data)
        pygame.display.init()
        try:
            pygame.mouse.set_cursor((24, 24), (0, 0), *cursor)
        except pygame.error as e:
            if 'not currently supported' in str(e):
                unittest.skip('skipping test as set_cursor() is not supported')
        finally:
            pygame.display.quit()

    def test_load_xbm(self):
        if False:
            for i in range(10):
                print('nop')
        cursorfile = fixture_path('xbm_cursors/white_sizing.xbm')
        maskfile = fixture_path('xbm_cursors/white_sizing_mask.xbm')
        cursor = pygame.cursors.load_xbm(cursorfile, maskfile)
        with open(cursorfile) as cursor_f, open(maskfile) as mask_f:
            cursor = pygame.cursors.load_xbm(cursor_f, mask_f)
        import pathlib
        cursor = pygame.cursors.load_xbm(pathlib.Path(cursorfile), pathlib.Path(maskfile))
        pygame.display.init()
        try:
            pygame.mouse.set_cursor(*cursor)
        except pygame.error as e:
            if 'not currently supported' in str(e):
                unittest.skip('skipping test as set_cursor() is not supported')
        finally:
            pygame.display.quit()

    def test_Cursor(self):
        if False:
            for i in range(10):
                print('nop')
        'Ensure that the cursor object parses information properly'
        c1 = pygame.cursors.Cursor(pygame.SYSTEM_CURSOR_CROSSHAIR)
        self.assertEqual(c1.data, (pygame.SYSTEM_CURSOR_CROSSHAIR,))
        self.assertEqual(c1.type, 'system')
        c2 = pygame.cursors.Cursor(c1)
        self.assertEqual(c1, c2)
        with self.assertRaises(TypeError):
            pygame.cursors.Cursor(-34002)
        with self.assertRaises(TypeError):
            pygame.cursors.Cursor('a', 'b', 'c', 'd')
        with self.assertRaises(TypeError):
            pygame.cursors.Cursor((2,))
        c3 = pygame.cursors.Cursor((0, 0), pygame.Surface((20, 20)))
        self.assertEqual(c3.data[0], (0, 0))
        self.assertEqual(c3.data[1].get_size(), (20, 20))
        self.assertEqual(c3.type, 'color')
        (xormask, andmask) = pygame.cursors.compile(pygame.cursors.thickarrow_strings)
        c4 = pygame.cursors.Cursor((24, 24), (0, 0), xormask, andmask)
        self.assertEqual(c4.data, ((24, 24), (0, 0), xormask, andmask))
        self.assertEqual(c4.type, 'bitmap')
if __name__ == '__main__':
    unittest.main()